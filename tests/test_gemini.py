import json

import pytest
import requests
import responses

import config
import gemini
from gemini import (
	GeminiBudgetExhausted,
	GeminiQuotaExhausted,
	generate_with_file,
	post_with_retry,
	upload_pdf_to_gemini,
)

_TEST_CFG = config.Config(**config.load_seed())
GENERATE_URL = gemini.generate_url(_TEST_CFG)
FILES_UPLOAD_URL = gemini.files_upload_url(_TEST_CFG)
UPLOAD_TARGET = 'https://upload.example/u/1'
URL = 'https://gemini.example/post'

_OK_BODY = {'candidates': [{'content': {'parts': [{'text': 'ok'}]}}]}


@pytest.fixture(autouse=True)
def no_sleep(mocker):
	return mocker.patch('gemini.time.sleep')


def _quota_429_body(quota_id, retry_delay=None):
	details = [{'@type': 'type.googleapis.com/google.rpc.QuotaFailure', 'violations': [{'quotaId': quota_id}]}]
	if retry_delay is not None:
		details.append({'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': retry_delay})
	return {'error': {'code': 429, 'status': 'RESOURCE_EXHAUSTED', 'details': details}}


_PER_DAY_429 = _quota_429_body('GenerateRequestsPerDayPerProjectPerModel-FreeTier')
_PER_MINUTE_429 = _quota_429_body('GenerateRequestsPerMinutePerProjectPerModel-FreeTier')


# -- URL builders --


def test_generate_url_built_from_cfg():
	assert (
		gemini.generate_url(_TEST_CFG) == f'{_TEST_CFG.gemini_base_url}/models/{_TEST_CFG.gemini_model}:generateContent'
	)


def test_files_upload_url_built_from_cfg():
	assert gemini.files_upload_url(_TEST_CFG) == f'{_TEST_CFG.gemini_upload_base_url}/files'


# -- post_with_retry: success and retriable statuses --


@responses.activate
def test_post_with_retry_returns_response_on_immediate_success():
	responses.post(URL, json=_OK_BODY)
	r = post_with_retry(URL, backoff_delays=(15,))
	assert r.json() == _OK_BODY
	assert len(responses.calls) == 1


@pytest.mark.parametrize('status', [429, 500, 502, 503, 504])
@responses.activate
def test_post_with_retry_retries_each_retriable_status_then_succeeds(status):
	responses.post(URL, json={'error': 'transient'}, status=status)
	responses.post(URL, json=_OK_BODY)
	assert post_with_retry(URL, backoff_delays=(15,)).json() == _OK_BODY


@responses.activate
def test_post_with_retry_returns_last_failed_response_after_exhaustion():
	# The caller owns raise_for_status, so exhaustion must hand back the final response rather than raise.
	responses.post(URL, json={'error': 'down'}, status=503)
	r = post_with_retry(URL, backoff_delays=(15, 30))
	assert r.status_code == 503
	assert len(responses.calls) == 3


@responses.activate
def test_post_with_retry_raises_http_error_path_for_non_retriable_status():
	responses.post(URL, json={'error': 'forbidden'}, status=403)
	r = post_with_retry(URL, backoff_delays=(15,))
	assert r.status_code == 403
	assert len(responses.calls) == 1
	with pytest.raises(requests.HTTPError):
		r.raise_for_status()


# -- post_with_retry: sleep patterns --


def test_post_with_retry_sleeps_backoff_schedule(no_sleep):
	with responses.RequestsMock() as rmock:
		rmock.post(URL, json={}, status=503)
		post_with_retry(URL, backoff_delays=(15, 30))
	assert [c.args[0] for c in no_sleep.call_args_list] == [15, 30]


def test_post_with_retry_pre_attempt_sleep_paces_every_attempt(no_sleep):
	with responses.RequestsMock() as rmock:
		rmock.post(URL, json={}, status=503)
		post_with_retry(URL, backoff_delays=(15, 30), pre_attempt_sleep=5)
	assert [c.args[0] for c in no_sleep.call_args_list] == [5, 15, 5, 30, 5]


def test_post_with_retry_no_sleep_on_immediate_success(no_sleep):
	with responses.RequestsMock() as rmock:
		rmock.post(URL, json=_OK_BODY)
		post_with_retry(URL, backoff_delays=(15,))
	no_sleep.assert_not_called()


def test_post_with_retry_honours_retry_after_header(no_sleep):
	with responses.RequestsMock() as rmock:
		rmock.post(URL, json={}, status=503, headers={'Retry-After': '7'})
		rmock.post(URL, json=_OK_BODY)
		post_with_retry(URL, backoff_delays=(15,))
	assert [c.args[0] for c in no_sleep.call_args_list] == [7]


def test_post_with_retry_honours_retry_info_delay(no_sleep):
	with responses.RequestsMock() as rmock:
		rmock.post(URL, json=_quota_429_body('PerMinute', retry_delay='7s'), status=429)
		rmock.post(URL, json=_OK_BODY)
		post_with_retry(URL, backoff_delays=(15,))
	assert [c.args[0] for c in no_sleep.call_args_list] == [7.0]


def test_post_with_retry_retry_after_header_beats_retry_info(no_sleep):
	# Retry-After is the transport-level directive; the body's RetryInfo is advisory and only consulted when the header is absent.
	with responses.RequestsMock() as rmock:
		rmock.post(URL, json=_quota_429_body('PerMinute', retry_delay='9s'), status=429, headers={'Retry-After': '3'})
		rmock.post(URL, json=_OK_BODY)
		post_with_retry(URL, backoff_delays=(15,))
	assert [c.args[0] for c in no_sleep.call_args_list] == [3]


def test_post_with_retry_falls_back_to_schedule_when_hints_malformed(no_sleep):
	# Non-dict detail entries, unparseable retryDelay and HTTP-date Retry-After must not crash; the schedule applies.
	malformed = {'error': {'status': 'RESOURCE_EXHAUSTED', 'details': ['bogus', {'retryDelay': 'bads'}]}}
	with responses.RequestsMock() as rmock:
		rmock.post(URL, json=malformed, status=429, headers={'Retry-After': 'Wed, 21 Oct 2026 07:28:00 GMT'})
		rmock.post(URL, json=_OK_BODY)
		post_with_retry(URL, backoff_delays=(15,))
	assert [c.args[0] for c in no_sleep.call_args_list] == [15]


# -- post_with_retry: quota classification --


@responses.activate
def test_post_with_retry_raises_quota_exhausted_on_per_day_429():
	responses.post(URL, json=_PER_DAY_429, status=429)
	with pytest.raises(GeminiQuotaExhausted):
		post_with_retry(URL, backoff_delays=(15,))


def test_post_with_retry_quota_raise_skips_backoff(no_sleep):
	with responses.RequestsMock() as rmock:
		rmock.post(URL, json=_PER_DAY_429, status=429)
		with pytest.raises(GeminiQuotaExhausted):
			post_with_retry(URL, backoff_delays=(15,))
	no_sleep.assert_not_called()


@responses.activate
def test_post_with_retry_detects_per_day_via_substring_when_body_not_json():
	responses.post(URL, body='quota violated: GenerateRequestsPerDayPerProjectPerModel', status=429)
	with pytest.raises(GeminiQuotaExhausted):
		post_with_retry(URL, backoff_delays=(15,))


@responses.activate
def test_post_with_retry_retries_per_minute_429():
	# Per-minute limits recover within the run; they must enter the retry path, not halt as daily exhaustion.
	responses.post(URL, json=_PER_MINUTE_429, status=429)
	responses.post(URL, json=_OK_BODY)
	assert post_with_retry(URL, backoff_delays=(15,)).json() == _OK_BODY


@responses.activate
def test_post_with_retry_retries_resource_exhausted_without_quota_details():
	# A bare RESOURCE_EXHAUSTED with no quotaId is ambiguous; retry rather than halt the run on a guess.
	responses.post(URL, json={'error': {'status': 'RESOURCE_EXHAUSTED'}}, status=429)
	responses.post(URL, json=_OK_BODY)
	assert post_with_retry(URL, backoff_delays=(15,)).json() == _OK_BODY


# -- post_with_retry: on_attempt --


@responses.activate
def test_post_with_retry_fires_on_attempt_once_per_attempt():
	responses.post(URL, json={}, status=503)
	counter = []
	post_with_retry(URL, backoff_delays=(15, 30), on_attempt=lambda: counter.append(1))
	assert len(counter) == 3


@responses.activate
def test_post_with_retry_fires_on_attempt_before_quota_raise():
	responses.post(URL, json=_PER_DAY_429, status=429)
	counter = []
	with pytest.raises(GeminiQuotaExhausted):
		post_with_retry(URL, backoff_delays=(15,), on_attempt=lambda: counter.append(1))
	assert len(counter) == 1


def test_post_with_retry_on_attempt_raise_propagates_without_posting(no_sleep):
	# main enforces its budget cap by raising from on_attempt; the iteration must stop before sleeping or posting.
	def raising_on_attempt():
		raise GeminiBudgetExhausted('budget 0 reached after 0 calls')

	with responses.RequestsMock() as rmock:
		with pytest.raises(GeminiBudgetExhausted):
			post_with_retry(URL, backoff_delays=(15,), pre_attempt_sleep=5, on_attempt=raising_on_attempt)
		assert len(rmock.calls) == 0
	no_sleep.assert_not_called()


# -- upload_pdf_to_gemini --


@responses.activate
def test_upload_pdf_returns_file_uri_on_happy_path():
	responses.post(FILES_UPLOAD_URL, json={}, headers={'X-Goog-Upload-URL': UPLOAD_TARGET})
	responses.post(UPLOAD_TARGET, json={'file': {'uri': 'files/abc'}})
	assert upload_pdf_to_gemini(b'pdf', 'paper.pdf', 'key', _TEST_CFG) == 'files/abc'


@responses.activate
def test_upload_pdf_raises_when_upload_url_header_missing():
	responses.post(FILES_UPLOAD_URL, json={})
	with pytest.raises(ValueError, match='X-Goog-Upload-URL'):
		upload_pdf_to_gemini(b'pdf', 'paper.pdf', 'key', _TEST_CFG)


@responses.activate
def test_upload_pdf_raises_when_file_uri_missing():
	responses.post(FILES_UPLOAD_URL, json={}, headers={'X-Goog-Upload-URL': UPLOAD_TARGET})
	responses.post(UPLOAD_TARGET, json={})
	with pytest.raises(ValueError, match='file.uri'):
		upload_pdf_to_gemini(b'pdf', 'paper.pdf', 'key', _TEST_CFG)


@responses.activate
def test_upload_pdf_sends_api_key_in_header_not_url():
	# Auth via header keeps the key out of any URL that may surface in HTTPError messages and downstream logs.
	responses.post(FILES_UPLOAD_URL, json={}, headers={'X-Goog-Upload-URL': UPLOAD_TARGET})
	responses.post(UPLOAD_TARGET, json={'file': {'uri': 'files/abc'}})
	upload_pdf_to_gemini(b'pdf', 'paper.pdf', 'secret-key', _TEST_CFG)
	assert responses.calls[0].request.headers['x-goog-api-key'] == 'secret-key'
	assert 'secret-key' not in responses.calls[0].request.url


@responses.activate
def test_upload_pdf_propagates_http_error_from_start_request():
	responses.post(FILES_UPLOAD_URL, json={'error': 'forbidden'}, status=403)
	with pytest.raises(requests.HTTPError):
		upload_pdf_to_gemini(b'pdf', 'paper.pdf', 'key', _TEST_CFG)


# -- generate_with_file --


@responses.activate
def test_generate_with_file_returns_stripped_text():
	responses.post(GENERATE_URL, json={'candidates': [{'content': {'parts': [{'text': '  out  \n'}]}}]})
	assert generate_with_file('p', 'files/abc', 'key', _TEST_CFG) == 'out'


@responses.activate
def test_generate_with_file_raises_when_fields_missing():
	responses.post(GENERATE_URL, json={'candidates': []})
	with pytest.raises(ValueError, match='missing expected fields'):
		generate_with_file('p', 'files/abc', 'key', _TEST_CFG)


@responses.activate
def test_generate_with_file_sends_api_key_in_header_not_url():
	responses.post(GENERATE_URL, json=_OK_BODY)
	generate_with_file('p', 'files/abc', 'secret-key', _TEST_CFG)
	assert responses.calls[0].request.headers['x-goog-api-key'] == 'secret-key'
	assert 'secret-key' not in responses.calls[0].request.url


@responses.activate
def test_generate_with_file_sends_file_data_and_prompt_parts():
	responses.post(GENERATE_URL, json=_OK_BODY)
	generate_with_file('the prompt', 'files/abc', 'key', _TEST_CFG)
	sent = json.loads(responses.calls[0].request.body)
	parts = sent['contents'][0]['parts']
	assert parts[0]['file_data'] == {'mime_type': 'application/pdf', 'file_uri': 'files/abc'}
	assert parts[1]['text'] == 'the prompt'
