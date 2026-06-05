import json
import re
import time

import requests

from config import SCORER_PROMPT

_FENCE_RE = re.compile(r'```(?:json)?', re.IGNORECASE)
_REQUIRED_RESULT_FIELDS = ('relevance_reason',)

# Mirrors summariser.GEMINI_RETRY_STATUS_CODES; scorer cannot import it back without a cycle (summariser imports from scorer).
GEMINI_RETRY_STATUS_CODES = (429, 500, 502, 503, 504)


# API-side halt: a PerDay quota violation means no call can succeed until the daily window resets, so the whole run stops calling Gemini.
class GeminiQuotaExhausted(Exception):
	pass


# Caller-side halt: our local GEMINI_CALL_COUNT has hit the configured GEMINI_CALL_BUDGET cap. Fires per-attempt so a 429 retry storm can't blow past it.
class GeminiBudgetExhausted(Exception):
	pass


def _error_details(response):
	try:
		details = (response.json().get('error') or {}).get('details')
	except (ValueError, AttributeError):
		return []
	return details if isinstance(details, list) else []


def _is_quota_exhausted(response):
	# Gemini reports daily, per-minute and burst limits all as 429 RESOURCE_EXHAUSTED; only a PerDay quotaId is terminal, the rest recover within the run.
	for detail in _error_details(response):
		if not isinstance(detail, dict):
			continue
		for violation in detail.get('violations') or []:
			if isinstance(violation, dict) and 'PerDay' in (violation.get('quotaId') or ''):
				return True
	# Substring fallback covers bodies that arrive truncated or as plain text rather than parseable JSON.
	return 'PerDay' in (response.text or '')


def _retry_delay_seconds(response):
	# Server-provided RetryInfo.retryDelay (e.g. '39s') is better-informed than our fixed schedule.
	for detail in _error_details(response):
		if not isinstance(detail, dict):
			continue
		delay = detail.get('retryDelay')
		if isinstance(delay, str) and delay.endswith('s'):
			try:
				return float(delay[:-1])
			except ValueError:
				continue
	return None


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
			raise GeminiQuotaExhausted('Gemini daily quota exhausted (PerDay quota violated)')
		if r.status_code in GEMINI_RETRY_STATUS_CODES:
			last_status = r.status_code
			wait = _retry_after_seconds(r) or _retry_delay_seconds(r) or 15 * (attempt + 1)
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

	enriched, responded, halted = _score_batches(papers, gemini_fn, cfg)

	# Re-batch unanswered papers once at the end; id-less papers are skipped because responded can never contain them.
	missed = [] if halted else [p for p in papers if p.get('paperId') and p['paperId'] not in responded]
	if missed:
		print(f'Rescoring {len(missed)} unresponded paper(s) in a second pass...')
		more_enriched, more_responded, _ = _score_batches(missed, gemini_fn, cfg)
		enriched.extend(more_enriched)
		responded.update(more_responded)

	return enriched, responded


def _score_batches(papers, gemini_fn, cfg):
	# halted means quota or budget stopped the run; callers must not send further Gemini work.
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
			return enriched, responded, True
		except GeminiBudgetExhausted as e:
			print(f'Budget exhausted, halting remaining batches: {e}')
			return enriched, responded, True
		enriched.extend(chunk_enriched)
		responded.update(chunk_responded)

	return enriched, responded, False


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
