import time

import requests

GEMINI_RETRY_STATUS_CODES = (429, 500, 502, 503, 504)
GEMINI_BACKOFF_DELAYS = (5, 15, 45)


# API-side halt: a PerDay quota violation means no call can succeed until the daily window resets, so the whole run stops calling Gemini.
class GeminiQuotaExhausted(Exception):
	pass


# Caller-side halt: the local call count has hit the configured budget cap. Fires per-attempt so a 429 retry storm can't blow past it.
class GeminiBudgetExhausted(Exception):
	pass


def generate_url(cfg):
	return f'{cfg.gemini_base_url}/models/{cfg.gemini_model}:generateContent'


def files_upload_url(cfg):
	return f'{cfg.gemini_upload_base_url}/files'


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


def _retry_after_seconds(response):
	# Integer form only; HTTP-date form falls back to the backoff schedule since the API never sends it.
	header = response.headers.get('Retry-After')
	if header is None:
		return None
	try:
		return int(header)
	except (TypeError, ValueError):
		return None


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


def post_with_retry(url, backoff_delays=GEMINI_BACKOFF_DELAYS, pre_attempt_sleep=0, on_attempt=None, **kwargs):
	# Returns the final response (failed or not) after exhausting backoff_delays, so each caller's raise_for_status surfaces its own error shape.
	last = None
	for attempt in range(len(backoff_delays) + 1):
		if on_attempt:
			on_attempt()
		if pre_attempt_sleep:
			time.sleep(pre_attempt_sleep)
		last = requests.post(url, **kwargs)
		if last.status_code == 429 and _is_quota_exhausted(last):
			raise GeminiQuotaExhausted('Gemini daily quota exhausted (PerDay quota violated)')
		if last.status_code not in GEMINI_RETRY_STATUS_CODES:
			return last
		if attempt == len(backoff_delays):
			return last
		delay = _retry_after_seconds(last) or _retry_delay_seconds(last) or backoff_delays[attempt]
		print(f'Gemini {last.status_code}, retrying in {delay}s')
		time.sleep(delay)


def upload_pdf_to_gemini(pdf_bytes, display_name, api_key, cfg, on_attempt=None):
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
	start = post_with_retry(
		files_upload_url(cfg),
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
	upload = post_with_retry(upload_url, headers=upload_headers, data=pdf_bytes, timeout=120)
	upload.raise_for_status()
	try:
		return upload.json()['file']['uri']
	except (KeyError, TypeError) as e:
		raise ValueError(f'Upload response missing file.uri: {e}') from e


def generate_with_file(prompt, file_uri, api_key, cfg, on_attempt=None):
	body = {
		'contents': [
			{'parts': [{'file_data': {'mime_type': 'application/pdf', 'file_uri': file_uri}}, {'text': prompt}]}
		]
	}
	r = post_with_retry(
		generate_url(cfg), on_attempt=on_attempt, headers={'x-goog-api-key': api_key}, json=body, timeout=120
	)
	r.raise_for_status()
	try:
		return r.json()['candidates'][0]['content']['parts'][0]['text'].strip()
	except (KeyError, IndexError, TypeError) as e:
		raise ValueError(f'Gemini response missing expected fields: {e}') from e
