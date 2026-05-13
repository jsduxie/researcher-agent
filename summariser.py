import json
import re
import time

import requests

import db
from config import (
	GEMINI_BASE_URL,
	GEMINI_MODEL,
	GEMINI_UPLOAD_BASE_URL,
	PDF_MAX_SIZE_MB,
	RESEARCH_CONTEXT,
	SUMMARISER_PROMPT,
)
from scorer import GeminiBudgetExhausted, GeminiQuotaExhausted, _is_quota_exhausted

MODEL_VERSION = GEMINI_MODEL
MISSING_FIELD_PLACEHOLDER = 'Not available from this source.'

GEMINI_GENERATE_URL = f'{GEMINI_BASE_URL}/models/{GEMINI_MODEL}:generateContent'
GEMINI_FILES_UPLOAD_URL = f'{GEMINI_UPLOAD_BASE_URL}/files'

GEMINI_RETRY_STATUS_CODES = (429, 500, 502, 503, 504)
GEMINI_BACKOFF_DELAYS = (5, 15, 45)

_FENCE_RE = re.compile(r'```(?:json)?', re.IGNORECASE)
_FIELDS = ('methodology', 'findings', 'relevance', 'limitations')
# Prompt uses `relevance_to_research` for clarity; DB column is `relevance`. Map at this boundary so downstream code aligns with the schema.
_PROMPT_KEY_BY_COLUMN = {'relevance': 'relevance_to_research'}


def summarise_paper(paper, gemini_fn, database_url=None, api_key=None, on_gemini_call=None):
	# Two short DB sessions (cache check, then persist); Gemini work in between runs with no connection open so Neon can auto-suspend.
	paper_id = paper.get('paperId')
	title = (paper.get('title') or '')[:60]

	if database_url and paper_id:
		with db.session(database_url) as conn:
			cached = db.get_summary(conn, paper_id)
		if cached is not None:
			print(f'Cache hit, skipping summarisation: {title}')
			return cached

	fields = None
	pdf_url = (paper.get('openAccessPdf') or {}).get('url')
	if pdf_url and api_key:
		fields = _summarise_via_pdf(title, pdf_url, api_key, on_gemini_call)

	if fields is None:
		fields = _summarise_via_abstract(paper, gemini_fn)

	if fields is None:
		return None

	if database_url and paper_id:
		with db.session(database_url) as conn:
			db.upsert_summary(conn, paper_id, fields, MODEL_VERSION)

	return fields


def _summarise_via_pdf(title, pdf_url, api_key, on_gemini_call):
	# on_gemini_call fires per attempt of each Gemini API call (Files API upload-init and generateContent), not per success. Transport errors count too.
	try:
		max_bytes = PDF_MAX_SIZE_MB * 1024 * 1024
		pdf_bytes = download_pdf(pdf_url, max_bytes)
		display_name = f'{(title or "paper")[:50]}.pdf'
		file_uri = upload_pdf_to_gemini(pdf_bytes, display_name, api_key, on_attempt=on_gemini_call)
		prompt = SUMMARISER_PROMPT.format(research_context=RESEARCH_CONTEXT, source_material='See the attached PDF.')
		response = generate_with_file(prompt, file_uri, api_key, on_attempt=on_gemini_call)
		return parse_summary_response(response)
	except (GeminiQuotaExhausted, GeminiBudgetExhausted):
		raise
	except Exception as e:
		print(f'PDF summariser failed for "{title}": {e}; falling back to abstract')
		return None


def _summarise_via_abstract(paper, gemini_fn):
	# Abstract-path counter is fired inside gemini_fn (main wires scorer.gemini's on_attempt directly); nothing to fire here.
	title = (paper.get('title') or '')[:60]
	abstract = paper.get('abstract')
	if not abstract:
		print(f'No abstract available, cannot summarise: {title}')
		return None
	source_material = f'Abstract:\n{abstract}'
	prompt = SUMMARISER_PROMPT.format(research_context=RESEARCH_CONTEXT, source_material=source_material)
	try:
		response = gemini_fn(prompt)
		return parse_summary_response(response)
	except (GeminiQuotaExhausted, GeminiBudgetExhausted):
		raise
	except Exception as e:
		print(f'Summariser Gemini error for "{title}": {e}')
		return None


def _retry_after_seconds(response):
	# HTTP-date form (RFC 7231) is ignored; we fall back to exponential backoff rather than parse dates the API never sends.
	header = response.headers.get('Retry-After')
	if header is None:
		return None
	try:
		return int(header)
	except (TypeError, ValueError):
		return None


def _post_with_retry(url, on_attempt=None, **kwargs):
	# Retries 429 and 5xx with backoff. After exhaustion returns the final failed response so the caller's raise_for_status surfaces consistently.
	last = None
	for attempt in range(len(GEMINI_BACKOFF_DELAYS) + 1):
		if on_attempt:
			on_attempt()
		last = requests.post(url, **kwargs)
		if last.status_code == 429 and _is_quota_exhausted(last):
			raise GeminiQuotaExhausted('Gemini daily quota exhausted (RESOURCE_EXHAUSTED)')
		if last.status_code not in GEMINI_RETRY_STATUS_CODES:
			return last
		if attempt == len(GEMINI_BACKOFF_DELAYS):
			return last
		delay = _retry_after_seconds(last) or GEMINI_BACKOFF_DELAYS[attempt]
		time.sleep(delay)
	return last


def download_pdf(url, max_size_bytes):
	with requests.get(url, stream=True, timeout=60) as r:
		r.raise_for_status()
		declared = r.headers.get('Content-Length')
		if declared and declared.isdigit() and int(declared) > max_size_bytes:
			raise ValueError(f'PDF declared size {declared} exceeds cap {max_size_bytes}')
		buffer = bytearray()
		for chunk in r.iter_content(chunk_size=64 * 1024):
			buffer.extend(chunk)
			if len(buffer) > max_size_bytes:
				raise ValueError(f'PDF stream exceeded cap {max_size_bytes}')
		return bytes(buffer)


def upload_pdf_to_gemini(pdf_bytes, display_name, api_key, on_attempt=None):
	# Resumable upload: first request gets an upload URL, second uploads the bytes. Auth on x-goog-api-key (not URL) so the key can't leak via HTTPError.
	start_headers = {
		'X-Goog-Upload-Protocol': 'resumable',
		'X-Goog-Upload-Command': 'start',
		'X-Goog-Upload-Header-Content-Length': str(len(pdf_bytes)),
		'X-Goog-Upload-Header-Content-Type': 'application/pdf',
		'Content-Type': 'application/json',
		'x-goog-api-key': api_key,
	}
	# Files API upload-init is a Gemini API call; the signed-URL upload below is not.
	start = _post_with_retry(
		GEMINI_FILES_UPLOAD_URL,
		on_attempt=on_attempt,
		headers=start_headers,
		json={'file': {'display_name': display_name}},
		timeout=60,
	)
	start.raise_for_status()
	upload_url = start.headers.get('X-Goog-Upload-URL')
	if not upload_url:
		raise ValueError('Upload start did not return X-Goog-Upload-URL header')

	upload_headers = {
		'Content-Length': str(len(pdf_bytes)),
		'X-Goog-Upload-Offset': '0',
		'X-Goog-Upload-Command': 'upload, finalize',
	}
	upload = _post_with_retry(upload_url, headers=upload_headers, data=pdf_bytes, timeout=120)
	upload.raise_for_status()
	try:
		return upload.json()['file']['uri']
	except (KeyError, TypeError) as e:
		raise ValueError(f'Upload response missing file.uri: {e}') from e


def generate_with_file(prompt, file_uri, api_key, on_attempt=None):
	body = {
		'contents': [
			{'parts': [{'file_data': {'mime_type': 'application/pdf', 'file_uri': file_uri}}, {'text': prompt}]}
		]
	}
	r = _post_with_retry(
		GEMINI_GENERATE_URL, on_attempt=on_attempt, headers={'x-goog-api-key': api_key}, json=body, timeout=120
	)
	r.raise_for_status()
	try:
		return r.json()['candidates'][0]['content']['parts'][0]['text'].strip()
	except (KeyError, IndexError, TypeError) as e:
		raise ValueError(f'Gemini response missing expected fields: {e}') from e


def parse_summary_response(response_text):
	cleaned = _FENCE_RE.sub('', response_text).strip()
	parsed = json.loads(cleaned)
	if not isinstance(parsed, dict):
		raise ValueError(f'Expected JSON object, got {type(parsed).__name__}')

	fields = {}
	for column in _FIELDS:
		prompt_key = _PROMPT_KEY_BY_COLUMN.get(column, column)
		value = parsed.get(prompt_key)
		if not isinstance(value, str) or not value.strip():
			fields[column] = MISSING_FIELD_PLACEHOLDER
		else:
			fields[column] = value.strip()
	return fields
