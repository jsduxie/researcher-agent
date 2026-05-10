import json
import re
import time

import requests

from config import BATCH_SIZE, RELEVANCE_THRESHOLD, RESEARCH_CONTEXT, SCORER_PROMPT

GEMINI_URL = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent'

_FENCE_RE = re.compile(r'```(?:json)?', re.IGNORECASE)
_REQUIRED_RESULT_FIELDS = ('relevance_reason', 'summary', 'key_contribution')


def gemini(prompt, api_key, retries=3):
	headers = {'Content-Type': 'application/json'}
	body = {'contents': [{'parts': [{'text': prompt}]}]}

	for attempt in range(retries):
		time.sleep(5)
		r = requests.post(f'{GEMINI_URL}?key={api_key}', headers=headers, json=body, timeout=120)
		if r.status_code == 429:
			wait = 15 * (attempt + 1)
			print(f'Gemini rate limited, waiting {wait}s')
			time.sleep(wait)
			continue
		r.raise_for_status()
		try:
			return r.json()['candidates'][0]['content']['parts'][0]['text'].strip()
		except (KeyError, IndexError, TypeError) as e:
			raise ValueError(f'Gemini response missing expected fields: {e}') from e

	raise Exception('Gemini failed after retries')


def parse_gemini_scores(response_text):
	cleaned = _FENCE_RE.sub('', response_text).strip()
	parsed = json.loads(cleaned)
	if not isinstance(parsed, list):
		raise ValueError(f'Expected JSON array, got {type(parsed).__name__}')
	return parsed


def apply_scores(papers, scores, threshold):
	# Returns (enriched, responded_paper_ids). responded covers any paper that got
	# a valid numeric Gemini score, regardless of threshold or whether the text
	# fields were complete. Papers missing from this set were never scored
	# successfully and should be retried on a later run.
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
		paper['ai_summary'] = data['summary']
		paper['ai_contribution'] = data['key_contribution']
		print(f'Kept (score {score}/10): {title}')
		enriched.append(paper)
	return enriched, responded


def score_and_summarise(papers, gemini_fn):
	if not papers:
		return [], set()

	enriched = []
	responded = set()
	for chunk_start in range(0, len(papers), BATCH_SIZE):
		chunk = papers[chunk_start : chunk_start + BATCH_SIZE]
		print(f'Scoring batch {chunk_start // BATCH_SIZE + 1} ({len(chunk)} papers)...')
		chunk_enriched, chunk_responded = _score_chunk(chunk, gemini_fn)
		enriched.extend(chunk_enriched)
		responded.update(chunk_responded)

	return enriched, responded


def _score_chunk(papers, gemini_fn):
	if not papers:
		return [], set()

	paper_entries = []
	for i, p in enumerate(papers):
		title = p.get('title', '')
		abstract = p.get('abstract') or 'No abstract available.'
		paper_entries.append(f'[{i}] Title: {title}\nAbstract: {abstract}')

	papers_block = '\n\n'.join(paper_entries)
	prompt = SCORER_PROMPT.format(research_context=RESEARCH_CONTEXT, papers_block=papers_block)

	try:
		response = gemini_fn(prompt)
		results = parse_gemini_scores(response)
	except Exception as e:
		print(f'Batch Gemini error: {e}')
		return [], set()

	return apply_scores(papers, results, RELEVANCE_THRESHOLD)
