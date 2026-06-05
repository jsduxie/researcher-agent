import json
import re
import time

import requests

from config import SCORER_PROMPT

_FENCE_RE = re.compile(r'```(?:json)?', re.IGNORECASE)
_REQUIRED_RESULT_FIELDS = ('relevance_reason',)

# Mirrors summariser.GEMINI_RETRY_STATUS_CODES; scorer cannot import it back without a cycle (summariser imports from scorer).
GEMINI_RETRY_STATUS_CODES = (429, 500, 502, 503, 504)


# API-side halt: Gemini returned RESOURCE_EXHAUSTED, meaning the daily RPD or RPM quota is gone and no further calls will succeed until the window resets.
class GeminiQuotaExhausted(Exception):
	pass


# Caller-side halt: our local GEMINI_CALL_COUNT has hit the configured GEMINI_CALL_BUDGET cap. Fires per-attempt so a 429 retry storm can't blow past it.
class GeminiBudgetExhausted(Exception):
	pass


def _is_quota_exhausted(response):
	# Google's RESOURCE_EXHAUSTED status is the canonical RPD/RPM exhaustion signal.
	try:
		if (response.json().get('error') or {}).get('status') == 'RESOURCE_EXHAUSTED':
			return True
	except (ValueError, AttributeError):
		pass
	return 'RESOURCE_EXHAUSTED' in (response.text or '')


def gemini_url(cfg):
	return f'{cfg.gemini_base_url}/models/{cfg.gemini_model}:generateContent'


def _retry_after_seconds(response):
	# Mirrors summariser._retry_after_seconds: integer form only; HTTP-date form falls back to the backoff schedule since the API never sends it.
	header = response.headers.get('Retry-After')
	if header is None:
		return None
	try:
		return int(header)
	except (TypeError, ValueError):
		return None


def gemini(prompt, api_key, cfg, retries=3, on_attempt=None):
	# Auth via header rather than ?key= query param keeps the secret out of any URL that may surface in HTTPError messages and downstream logs.
	headers = {'Content-Type': 'application/json', 'x-goog-api-key': api_key}
	body = {'contents': [{'parts': [{'text': prompt}]}]}
	url = gemini_url(cfg)

	last_status = None
	for attempt in range(retries):
		if on_attempt:
			on_attempt()
		time.sleep(5)
		r = requests.post(url, headers=headers, json=body, timeout=120)
		if r.status_code == 429 and _is_quota_exhausted(r):
			raise GeminiQuotaExhausted('Gemini daily quota exhausted (RESOURCE_EXHAUSTED)')
		if r.status_code in GEMINI_RETRY_STATUS_CODES:
			last_status = r.status_code
			wait = _retry_after_seconds(r) or 15 * (attempt + 1)
			print(f'Gemini {r.status_code}, retrying in {wait}s')
			time.sleep(wait)
			continue
		r.raise_for_status()
		try:
			return r.json()['candidates'][0]['content']['parts'][0]['text'].strip()
		except (KeyError, IndexError, TypeError) as e:
			raise ValueError(f'Gemini response missing expected fields: {e}') from e

	raise Exception(f'Gemini failed after retries (last status {last_status})')


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
		title = paper.get('title', '')[:60]
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
	if not papers:
		return [], set()

	enriched = []
	responded = set()
	batch_size = cfg.batch_size
	for chunk_start in range(0, len(papers), batch_size):
		chunk = papers[chunk_start : chunk_start + batch_size]
		print(f'Scoring batch {chunk_start // batch_size + 1} ({len(chunk)} papers)...')
		try:
			chunk_enriched, chunk_responded = _score_chunk(chunk, gemini_fn, cfg)
		except GeminiQuotaExhausted as e:
			print(f'Quota exhausted, halting remaining batches: {e}')
			break
		except GeminiBudgetExhausted as e:
			print(f'Budget exhausted, halting remaining batches: {e}')
			break
		enriched.extend(chunk_enriched)
		responded.update(chunk_responded)

	return enriched, responded


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
