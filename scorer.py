import json
import re

from config import SCORER_PROMPT
from gemini import GEMINI_RETRY_STATUS_CODES, generate_url, post_with_retry
from gemini import GeminiBudgetExhausted as GeminiBudgetExhausted
from gemini import GeminiQuotaExhausted as GeminiQuotaExhausted
from gemini import _is_quota_exhausted as _is_quota_exhausted

_FENCE_RE = re.compile(r'```(?:json)?', re.IGNORECASE)
_REQUIRED_RESULT_FIELDS = ('relevance_reason',)


def gemini(prompt, api_key, cfg, retries=3, on_attempt=None):
	# Auth via header rather than ?key= query param keeps the secret out of any URL that may surface in HTTPError messages and downstream logs.
	headers = {'Content-Type': 'application/json', 'x-goog-api-key': api_key}
	body = {'contents': [{'parts': [{'text': prompt}]}]}
	# 5s pre-attempt pacing keeps a batch run under the per-minute limit; 15s steps between retries.
	backoff = tuple(15 * (i + 1) for i in range(retries - 1))
	r = post_with_retry(
		generate_url(cfg),
		backoff_delays=backoff,
		pre_attempt_sleep=5,
		on_attempt=on_attempt,
		headers=headers,
		json=body,
		timeout=120,
	)
	if r.status_code in GEMINI_RETRY_STATUS_CODES:
		raise Exception(f'Gemini failed after retries (last status {r.status_code})')
	r.raise_for_status()
	try:
		return r.json()['candidates'][0]['content']['parts'][0]['text'].strip()
	except (KeyError, IndexError, TypeError) as e:
		raise ValueError(f'Gemini response missing expected fields: {e}') from e


def parse_gemini_scores(response_text):
	cleaned = _FENCE_RE.sub('', response_text).strip()
	parsed = json.loads(cleaned)
	if not isinstance(parsed, list):
		raise ValueError(f'Expected JSON array, got {type(parsed).__name__}')
	return parsed


def apply_scores(papers, scores, threshold):
	# Returns (enriched, responded_paper_ids). responded covers any paper with a valid numeric score; missing IDs were never scored and should be retried.
	if not isinstance(scores, list):
		print(f'Dropped batch (scores payload not a list, got {type(scores).__name__})')
		return [], set()

	results_by_index = {}
	for r in scores:
		if not isinstance(r, dict):
			print(f'Skipped (non-dict result: {r!r})')
			continue
		idx = r.get('index')
		if not isinstance(idx, int) or isinstance(idx, bool):
			print(f'Skipped (missing or invalid index: {idx!r})')
			continue
		results_by_index[idx] = r

	enriched = []
	responded = set()
	for i, paper in enumerate(papers):
		# covers an explicit null title from the API, which .get's default cannot.
		title = (paper.get('title') or '')[:60]
		data = results_by_index.get(i)
		if data is None:
			print(f'Missing result for [{i}]: {title}')
			continue
		score = data.get('relevance_score')
		if not isinstance(score, int) or isinstance(score, bool):
			print(f'Dropped (invalid score {score!r}): {title}')
			continue
		paper_id = paper.get('paperId')
		if paper_id:
			responded.add(paper_id)
		if score < threshold:
			print(f' Dropped (score {score}/10): {title}')
			continue
		missing = [f for f in _REQUIRED_RESULT_FIELDS if not isinstance(data.get(f), str)]
		if missing:
			print(f'Dropped (missing fields {missing}): {title}')
			continue
		paper['ai_score'] = score
		paper['ai_reason'] = data['relevance_reason']
		print(f'Kept (score {score}/10): {title}')
		enriched.append(paper)
	return enriched, responded


def score_and_summarise(papers, gemini_fn, cfg):
	# Returns (enriched, responded, attempted) sets of paper ids; attempted only covers batches Gemini actually saw.
	if not papers:
		return [], set(), set()

	enriched, responded, attempted, halted = _score_batches(papers, gemini_fn, cfg)

	# Re-batch unanswered papers once at the end; id-less papers are skipped because responded can never contain them.
	missed = [] if halted else [p for p in papers if p.get('paperId') and p['paperId'] not in responded]
	if missed:
		print(f'Rescoring {len(missed)} unresponded paper(s) in a second pass...')
		more_enriched, more_responded, more_attempted, _ = _score_batches(missed, gemini_fn, cfg)
		enriched.extend(more_enriched)
		responded.update(more_responded)
		attempted.update(more_attempted)

	return enriched, responded, attempted


def _score_batches(papers, gemini_fn, cfg):
	# halted means quota or budget stopped the run; callers must not send further Gemini work.
	enriched = []
	responded = set()
	attempted = set()
	batch_size = cfg.batch_size
	for chunk_start in range(0, len(papers), batch_size):
		chunk = papers[chunk_start : chunk_start + batch_size]
		print(f'Scoring batch {chunk_start // batch_size + 1} ({len(chunk)} papers)...')
		try:
			chunk_enriched, chunk_responded = _score_chunk(chunk, gemini_fn, cfg)
		except GeminiQuotaExhausted as e:
			print(f'Quota exhausted, halting remaining batches: {e}')
			return enriched, responded, attempted, True
		except GeminiBudgetExhausted as e:
			print(f'Budget exhausted, halting remaining batches: {e}')
			return enriched, responded, attempted, True
		enriched.extend(chunk_enriched)
		responded.update(chunk_responded)
		# Halted batches are excluded so their papers stay eligible for future sweeps.
		attempted.update(p['paperId'] for p in chunk if p.get('paperId'))

	return enriched, responded, attempted, False


def _score_chunk(papers, gemini_fn, cfg):
	if not papers:
		return [], set()

	paper_entries = []
	for i, p in enumerate(papers):
		title = p.get('title', '')
		abstract = p.get('abstract') or 'No abstract available.'
		paper_entries.append(f'[{i}] Title: {title}\nAbstract: {abstract}')

	papers_block = '\n\n'.join(paper_entries)
	prompt = SCORER_PROMPT.format(research_context=cfg.research_context, papers_block=papers_block)

	try:
		response = gemini_fn(prompt)
		results = parse_gemini_scores(response)
	except (GeminiQuotaExhausted, GeminiBudgetExhausted):
		raise
	except Exception as e:
		print(f'Batch Gemini error: {e}')
		return [], set()

	return apply_scores(papers, results, cfg.relevance_threshold)
