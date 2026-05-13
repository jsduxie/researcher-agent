import json

import pytest
import requests
import responses

import summariser
from summariser import (
	MISSING_FIELD_PLACEHOLDER,
	MODEL_VERSION,
	download_pdf,
	generate_with_file,
	parse_summary_response,
	summarise_paper,
	upload_pdf_to_gemini,
)


@pytest.fixture(autouse=True)
def _no_sleep(mocker):
	return mocker.patch('summariser.time.sleep')


@pytest.fixture(autouse=True)
def mock_session(mocker):
	# summariser opens short DB sessions internally via db.session; patch the helper to yield a stable mock conn so tests passing database_url= get a usable session.
	conn = mocker.MagicMock()
	session_cm = mocker.MagicMock()
	session_cm.__enter__.return_value = conn
	session_cm.__exit__.return_value = False
	mocker.patch('summariser.db.session', return_value=session_cm)
	return {'conn': conn, 'session_cm': session_cm}


def _valid_response(**overrides):
	body = {'methodology': 'method', 'findings': 'find', 'relevance_to_research': 'rel', 'limitations': 'lim'}
	body.update(overrides)
	return json.dumps(body)


# -- parse_summary_response --


def test_parse_returns_four_fields_on_happy_path():
	result = parse_summary_response(_valid_response())
	assert result == {'methodology': 'method', 'findings': 'find', 'relevance': 'rel', 'limitations': 'lim'}


def test_parse_strips_markdown_fences():
	wrapped = f'```json\n{_valid_response()}\n```'
	assert parse_summary_response(wrapped)['methodology'] == 'method'


def test_parse_strips_bare_code_fences():
	wrapped = f'```\n{_valid_response()}\n```'
	assert parse_summary_response(wrapped)['methodology'] == 'method'


def test_parse_strips_surrounding_whitespace():
	assert parse_summary_response(f'   \n{_valid_response()}\n   ')['methodology'] == 'method'


def test_parse_maps_relevance_to_research_to_relevance_column():
	# The DB column is `relevance` but the prompt key is `relevance_to_research`. The mapping has to happen here or the data will silently drop.
	result = parse_summary_response(_valid_response(relevance_to_research='mapped'))
	assert result['relevance'] == 'mapped'
	assert 'relevance_to_research' not in result


def test_parse_uses_placeholder_when_field_missing():
	body = json.dumps({'methodology': 'm', 'findings': 'f', 'limitations': 'l'})  # no relevance
	result = parse_summary_response(body)
	assert result['relevance'] == MISSING_FIELD_PLACEHOLDER


def test_parse_uses_placeholder_when_field_is_non_string():
	body = json.dumps({'methodology': None, 'findings': 'f', 'relevance_to_research': 'r', 'limitations': 'l'})
	result = parse_summary_response(body)
	assert result['methodology'] == MISSING_FIELD_PLACEHOLDER


def test_parse_uses_placeholder_when_field_is_empty_string():
	result = parse_summary_response(_valid_response(findings=''))
	assert result['findings'] == MISSING_FIELD_PLACEHOLDER


def test_parse_strips_whitespace_from_field_values():
	result = parse_summary_response(_valid_response(methodology='  spaced  '))
	assert result['methodology'] == 'spaced'


def test_parse_ignores_extra_keys():
	result = parse_summary_response(_valid_response(extra='ignored'))
	assert 'extra' not in result


def test_parse_raises_on_malformed_json():
	with pytest.raises(json.JSONDecodeError):
		parse_summary_response('not valid')


def test_parse_raises_when_response_is_a_list():
	with pytest.raises(ValueError, match='Expected JSON object'):
		parse_summary_response('[]')


def test_parse_raises_when_response_is_a_scalar():
	with pytest.raises(ValueError, match='Expected JSON object'):
		parse_summary_response('42')


# -- summarise_paper: cache hit --


def test_summarise_paper_short_circuits_on_cache_hit(mocker):
	cached = {'methodology': 'm', 'findings': 'f', 'relevance': 'r', 'limitations': 'l', 'model_version': 'v'}
	mocker.patch('summariser.db.get_summary', return_value=cached)
	upsert = mocker.patch('summariser.db.upsert_summary')
	gemini_fn = mocker.Mock()

	result = summarise_paper({'paperId': 'p1', 'abstract': 'a'}, gemini_fn, database_url='postgresql://x')

	assert result == cached
	gemini_fn.assert_not_called()
	upsert.assert_not_called()


def test_summarise_paper_skips_cache_check_when_no_conn(mocker):
	get_summary = mocker.patch('summariser.db.get_summary')
	mocker.patch('summariser.db.upsert_summary')
	gemini_fn = mocker.Mock(return_value=_valid_response())

	summarise_paper({'paperId': 'p1', 'abstract': 'a'}, gemini_fn, database_url=None)

	get_summary.assert_not_called()


def test_summarise_paper_skips_cache_check_when_no_paper_id(mocker):
	# A paper without a paperId can't be keyed in the cache. We still summarise (some upstream paths may want a transient summary) but persistence is skipped.
	get_summary = mocker.patch('summariser.db.get_summary')
	upsert = mocker.patch('summariser.db.upsert_summary')
	gemini_fn = mocker.Mock(return_value=_valid_response())

	result = summarise_paper({'abstract': 'a'}, gemini_fn, database_url='postgresql://x')

	get_summary.assert_not_called()
	upsert.assert_not_called()
	assert result is not None
	assert result['methodology'] == 'method'


# -- summarise_paper: happy path --


def test_summarise_paper_calls_gemini_and_returns_four_fields(mocker):
	mocker.patch('summariser.db.get_summary', return_value=None)
	mocker.patch('summariser.db.upsert_summary')
	gemini_fn = mocker.Mock(return_value=_valid_response())

	result = summarise_paper({'paperId': 'p1', 'abstract': 'a'}, gemini_fn, database_url='postgresql://x')

	assert result == {'methodology': 'method', 'findings': 'find', 'relevance': 'rel', 'limitations': 'lim'}
	gemini_fn.assert_called_once()


def test_summarise_paper_persists_fresh_summary(mocker, mock_session):
	mocker.patch('summariser.db.get_summary', return_value=None)
	upsert = mocker.patch('summariser.db.upsert_summary')
	gemini_fn = mocker.Mock(return_value=_valid_response())

	summarise_paper({'paperId': 'p1', 'abstract': 'a'}, gemini_fn, database_url='postgresql://x')

	upsert.assert_called_once()
	call = upsert.call_args
	assert call.args[0] is mock_session['conn']
	assert call.args[1] == 'p1'
	assert call.args[2] == {'methodology': 'method', 'findings': 'find', 'relevance': 'rel', 'limitations': 'lim'}
	assert call.args[3] == MODEL_VERSION


def test_summarise_paper_does_not_persist_when_conn_is_none(mocker):
	upsert = mocker.patch('summariser.db.upsert_summary')
	gemini_fn = mocker.Mock(return_value=_valid_response())

	summarise_paper({'paperId': 'p1', 'abstract': 'a'}, gemini_fn, database_url=None)

	upsert.assert_not_called()


def test_summarise_paper_does_not_persist_when_no_paper_id(mocker):
	mocker.patch('summariser.db.get_summary')
	upsert = mocker.patch('summariser.db.upsert_summary')
	gemini_fn = mocker.Mock(return_value=_valid_response())

	summarise_paper({'abstract': 'a'}, gemini_fn, database_url='postgresql://x')

	upsert.assert_not_called()


def test_summarise_paper_prompt_includes_abstract_and_context(mocker):
	mocker.patch('summariser.db.get_summary', return_value=None)
	mocker.patch('summariser.db.upsert_summary')
	captured = {}

	def fake_gemini(prompt):
		captured['prompt'] = prompt
		return _valid_response()

	summarise_paper(
		{'paperId': 'p1', 'abstract': 'unique-abstract-text-xyz'}, fake_gemini, database_url='postgresql://x'
	)

	assert 'unique-abstract-text-xyz' in captured['prompt']
	assert 'RESEARCH CONTEXT' in captured['prompt']


# -- summarise_paper: failure modes --


def test_summarise_paper_returns_none_when_no_abstract(mocker, capsys):
	get_summary = mocker.patch('summariser.db.get_summary', return_value=None)
	upsert = mocker.patch('summariser.db.upsert_summary')
	gemini_fn = mocker.Mock()

	result = summarise_paper({'paperId': 'p1'}, gemini_fn, database_url='postgresql://x')

	assert result is None
	gemini_fn.assert_not_called()
	upsert.assert_not_called()
	# Cache miss still happens before we discover no abstract; that's fine.
	get_summary.assert_called_once()
	assert 'No abstract available' in capsys.readouterr().out


def test_summarise_paper_returns_none_when_abstract_is_empty(mocker):
	mocker.patch('summariser.db.get_summary', return_value=None)
	gemini_fn = mocker.Mock()
	assert summarise_paper({'paperId': 'p1', 'abstract': ''}, gemini_fn, database_url='postgresql://x') is None
	gemini_fn.assert_not_called()


def test_summarise_paper_returns_none_on_gemini_exception(mocker, capsys):
	mocker.patch('summariser.db.get_summary', return_value=None)
	upsert = mocker.patch('summariser.db.upsert_summary')
	gemini_fn = mocker.Mock(side_effect=Exception('rate limit'))

	result = summarise_paper({'paperId': 'p1', 'abstract': 'a'}, gemini_fn, database_url='postgresql://x')

	assert result is None
	upsert.assert_not_called()
	assert 'Summariser Gemini error' in capsys.readouterr().out


def test_summarise_paper_returns_none_on_malformed_response(mocker):
	mocker.patch('summariser.db.get_summary', return_value=None)
	upsert = mocker.patch('summariser.db.upsert_summary')
	gemini_fn = mocker.Mock(return_value='not json at all')

	result = summarise_paper({'paperId': 'p1', 'abstract': 'a'}, gemini_fn, database_url='postgresql://x')

	assert result is None
	upsert.assert_not_called()


def test_summarise_paper_persists_partial_response_with_placeholders(mocker):
	# A response missing one field should still be persisted with the placeholder; this is the abstract-only fallback behaviour the spec calls for.
	mocker.patch('summariser.db.get_summary', return_value=None)
	upsert = mocker.patch('summariser.db.upsert_summary')
	body = json.dumps({'methodology': 'm', 'findings': 'f', 'relevance_to_research': 'r'})  # no limitations
	gemini_fn = mocker.Mock(return_value=body)

	result = summarise_paper({'paperId': 'p1', 'abstract': 'a'}, gemini_fn, database_url='postgresql://x')

	assert result is not None
	assert result['limitations'] == MISSING_FIELD_PLACEHOLDER
	upsert.assert_called_once()
	assert upsert.call_args.args[2]['limitations'] == MISSING_FIELD_PLACEHOLDER


def test_summarise_paper_handles_paper_with_missing_title(mocker):
	# The log lines slice title to 60 chars; a missing title must not crash.
	mocker.patch('summariser.db.get_summary', return_value=None)
	mocker.patch('summariser.db.upsert_summary')
	gemini_fn = mocker.Mock(return_value=_valid_response())

	result = summarise_paper({'paperId': 'p1', 'abstract': 'a'}, gemini_fn, database_url='postgresql://x')

	assert result is not None


# -- module surface --


def test_module_version_is_set():
	# Persisted alongside summaries so a later prompt or model change can be reasoned about against historical data.
	assert summariser.MODEL_VERSION


# -- on_gemini_call callback (per-attempt for PDF path; abstract path counts inside gemini_fn) --


def test_on_gemini_call_does_not_fire_for_abstract_path_via_summarise_paper(mocker):
	# Abstract path no longer relays on_gemini_call; the counter fires inside gemini_fn (scorer.gemini's on_attempt, wired by main). Mocked gemini_fn = 0 fires.
	mocker.patch('summariser.db.get_summary', return_value=None)
	mocker.patch('summariser.db.upsert_summary')
	gemini_fn = mocker.Mock(return_value=_valid_response())
	counter = mocker.Mock()

	summarise_paper(
		{'paperId': 'p1', 'abstract': 'a'}, gemini_fn, database_url='postgresql://x', on_gemini_call=counter
	)

	counter.assert_not_called()


def test_on_gemini_call_does_not_fire_on_cache_hit(mocker):
	mocker.patch(
		'summariser.db.get_summary',
		return_value={'methodology': 'm', 'findings': 'f', 'relevance': 'r', 'limitations': 'l'},
	)
	gemini_fn = mocker.Mock()
	counter = mocker.Mock()

	summarise_paper(
		{'paperId': 'p1', 'abstract': 'a'}, gemini_fn, database_url='postgresql://x', on_gemini_call=counter
	)

	counter.assert_not_called()


def test_on_gemini_call_does_not_fire_when_abstract_gemini_raises(mocker):
	# gemini_fn is mocked and raises; the abstract path doesn't relay on_gemini_call, so it stays at 0 fires. Real per-attempt firing is in test_scorer.py.
	mocker.patch('summariser.db.get_summary', return_value=None)
	gemini_fn = mocker.Mock(side_effect=Exception('boom'))
	counter = mocker.Mock()

	summarise_paper(
		{'paperId': 'p1', 'abstract': 'a'}, gemini_fn, database_url='postgresql://x', on_gemini_call=counter
	)

	counter.assert_not_called()


@responses.activate
def test_on_gemini_call_fires_per_gemini_attempt_in_pdf_path(mocker):
	# PDF success path fires per attempt of upload-init + generate (signed-URL data upload is not a Gemini API call). One attempt each on happy path = 2.
	responses.get(PDF_URL, body=b'%PDF-1.4 fake')
	responses.post(summariser.GEMINI_FILES_UPLOAD_URL, json={}, headers={'X-Goog-Upload-URL': UPLOAD_TARGET})
	responses.post(UPLOAD_TARGET, json={'file': {'uri': 'files/abc'}})
	responses.post(
		summariser.GEMINI_GENERATE_URL, json={'candidates': [{'content': {'parts': [{'text': _valid_response()}]}}]}
	)
	mocker.patch('summariser.db.get_summary', return_value=None)
	mocker.patch('summariser.db.upsert_summary')
	counter = mocker.Mock()

	summarise_paper(
		{'paperId': 'p1', 'abstract': 'a', 'openAccessPdf': {'url': PDF_URL}},
		mocker.Mock(),
		database_url='postgresql://x',
		api_key='fake-key',
		on_gemini_call=counter,
	)

	assert counter.call_count == 2


@responses.activate
def test_on_gemini_call_does_not_fire_when_pdf_fails_then_abstract_succeeds(mocker):
	# PDF download fails before any _post_with_retry (0 fires from PDF path); abstract fallback uses mocked gemini_fn which doesn't fire on_attempt internally.
	responses.get(PDF_URL, json={'error': 'oops'}, status=500)
	mocker.patch('summariser.db.get_summary', return_value=None)
	mocker.patch('summariser.db.upsert_summary')
	gemini_fn = mocker.Mock(return_value=_valid_response())
	counter = mocker.Mock()

	summarise_paper(
		{'paperId': 'p1', 'abstract': 'a', 'openAccessPdf': {'url': PDF_URL}},
		gemini_fn,
		database_url='postgresql://x',
		api_key='fake-key',
		on_gemini_call=counter,
	)

	counter.assert_not_called()


@responses.activate
def test_post_with_retry_fires_on_attempt_per_iteration_via_generate_with_file():
	# Two 500s then success: on_attempt fires once per iteration of _post_with_retry.
	responses.post(summariser.GEMINI_GENERATE_URL, json={}, status=500)
	responses.post(summariser.GEMINI_GENERATE_URL, json={}, status=500)
	responses.post(summariser.GEMINI_GENERATE_URL, json={'candidates': [{'content': {'parts': [{'text': 'ok'}]}}]})
	counter = []
	generate_with_file('p', 'files/abc', 'k', on_attempt=lambda: counter.append(1))
	assert len(counter) == 3


# -- db session lifecycle inside summarise_paper --


def test_no_db_session_is_open_during_summariser_gemini_work(mocker):
	# Replace the default session mock with a tracker that counts active opens; snapshot active count at the moment gemini_fn is invoked.
	from contextlib import contextmanager

	active = [0]
	active_during_gemini = []

	@contextmanager
	def tracking_session(url):
		active[0] += 1
		try:
			yield mocker.MagicMock()
		finally:
			active[0] -= 1

	mocker.patch('summariser.db.session', side_effect=tracking_session)
	mocker.patch('summariser.db.get_summary', return_value=None)
	mocker.patch('summariser.db.upsert_summary')

	def gemini_snap(prompt):
		active_during_gemini.append(active[0])
		return _valid_response()

	summarise_paper({'paperId': 'p1', 'abstract': 'a'}, gemini_snap, database_url='postgresql://x')

	# Cache-check session closed before gemini_fn; persist session opens only after.
	assert active_during_gemini == [0]


# -- download_pdf --


PDF_URL = 'https://example.com/paper.pdf'


@responses.activate
def test_download_pdf_returns_bytes_on_happy_path():
	responses.get(PDF_URL, body=b'%PDF-1.4 content')
	assert download_pdf(PDF_URL, max_size_bytes=1024) == b'%PDF-1.4 content'


def test_download_pdf_raises_when_declared_content_length_exceeds_cap(mocker):
	# Isolates the declared-size check from the stream check: a streamed `responses` body doesn't surface Content-Length on r.headers, so direct-mock the response.
	mock_response = mocker.MagicMock()
	mock_response.__enter__.return_value = mock_response
	mock_response.headers = {'Content-Length': '200'}
	mocker.patch('summariser.requests.get', return_value=mock_response)
	with pytest.raises(ValueError, match='declared size .* exceeds cap'):
		download_pdf(PDF_URL, max_size_bytes=100)


def test_download_pdf_raises_when_stream_exceeds_cap(mocker):
	# Simulates a server that lies about (or omits) Content-Length. The stream check is the safety net; it must fire before the buffer eats unbounded memory.
	mock_response = mocker.MagicMock()
	mock_response.__enter__.return_value = mock_response
	mock_response.headers = {}
	mock_response.iter_content.return_value = iter([b'x' * 60, b'x' * 60])
	mocker.patch('summariser.requests.get', return_value=mock_response)
	with pytest.raises(ValueError, match='stream exceeded cap'):
		download_pdf(PDF_URL, max_size_bytes=100)


def test_download_pdf_ignores_non_numeric_declared_content_length(mocker):
	# A non-digit Content-Length must not crash; we fall through to the stream check.
	mock_response = mocker.MagicMock()
	mock_response.__enter__.return_value = mock_response
	mock_response.headers = {'Content-Length': 'unknown'}
	mock_response.iter_content.return_value = iter([b'small'])
	mocker.patch('summariser.requests.get', return_value=mock_response)
	assert download_pdf(PDF_URL, max_size_bytes=100) == b'small'


@responses.activate
def test_download_pdf_propagates_http_error():
	responses.get(PDF_URL, json={'error': 'oops'}, status=500)
	with pytest.raises(requests.HTTPError):
		download_pdf(PDF_URL, max_size_bytes=1024)


# -- upload_pdf_to_gemini --


UPLOAD_TARGET = 'https://upload.example.com/sessions/abc'


@responses.activate
def test_upload_pdf_to_gemini_returns_file_uri_on_happy_path():
	responses.post(summariser.GEMINI_FILES_UPLOAD_URL, json={}, headers={'X-Goog-Upload-URL': UPLOAD_TARGET})
	responses.post(UPLOAD_TARGET, json={'file': {'uri': 'https://files/abc', 'name': 'files/abc'}})
	assert upload_pdf_to_gemini(b'pdf', 'paper.pdf', 'fake-key') == 'https://files/abc'


@responses.activate
def test_upload_pdf_to_gemini_sends_api_key_in_header_not_url():
	# Auth via header keeps the key out of any URL that may surface in HTTPError messages and downstream logs (PR #17 Copilot review).
	responses.post(summariser.GEMINI_FILES_UPLOAD_URL, json={}, headers={'X-Goog-Upload-URL': UPLOAD_TARGET})
	responses.post(UPLOAD_TARGET, json={'file': {'uri': 'u'}})
	upload_pdf_to_gemini(b'pdf', 'my-key-value', 'my-key-value-secret')
	assert responses.calls[0].request.headers['x-goog-api-key'] == 'my-key-value-secret'
	assert 'my-key-value-secret' not in responses.calls[0].request.url


@responses.activate
def test_upload_pdf_to_gemini_sends_display_name_in_metadata():
	responses.post(summariser.GEMINI_FILES_UPLOAD_URL, json={}, headers={'X-Goog-Upload-URL': UPLOAD_TARGET})
	responses.post(UPLOAD_TARGET, json={'file': {'uri': 'u'}})
	upload_pdf_to_gemini(b'pdf', 'a-paper.pdf', 'k')
	start_body = json.loads(responses.calls[0].request.body)
	assert start_body == {'file': {'display_name': 'a-paper.pdf'}}


@responses.activate
def test_upload_pdf_to_gemini_raises_after_retries_exhausted_on_start_5xx():
	for _ in range(4):
		responses.post(summariser.GEMINI_FILES_UPLOAD_URL, json={}, status=500)
	with pytest.raises(requests.HTTPError):
		upload_pdf_to_gemini(b'pdf', 'paper.pdf', 'k')
	assert len(responses.calls) == 4


@responses.activate
def test_upload_pdf_to_gemini_does_not_retry_start_on_400():
	responses.post(summariser.GEMINI_FILES_UPLOAD_URL, json={}, status=400)
	with pytest.raises(requests.HTTPError):
		upload_pdf_to_gemini(b'pdf', 'paper.pdf', 'k')
	assert len(responses.calls) == 1


@responses.activate
def test_upload_pdf_to_gemini_raises_when_no_upload_url_header():
	responses.post(summariser.GEMINI_FILES_UPLOAD_URL, json={})
	with pytest.raises(ValueError, match='X-Goog-Upload-URL'):
		upload_pdf_to_gemini(b'pdf', 'paper.pdf', 'k')


@responses.activate
def test_upload_pdf_to_gemini_raises_after_retries_exhausted_on_upload_5xx():
	responses.post(summariser.GEMINI_FILES_UPLOAD_URL, json={}, headers={'X-Goog-Upload-URL': UPLOAD_TARGET})
	for _ in range(4):
		responses.post(UPLOAD_TARGET, json={}, status=500)
	with pytest.raises(requests.HTTPError):
		upload_pdf_to_gemini(b'pdf', 'paper.pdf', 'k')


@responses.activate
def test_upload_pdf_to_gemini_retries_start_on_429_then_succeeds():
	responses.post(summariser.GEMINI_FILES_UPLOAD_URL, json={}, status=429)
	responses.post(summariser.GEMINI_FILES_UPLOAD_URL, json={}, headers={'X-Goog-Upload-URL': UPLOAD_TARGET})
	responses.post(UPLOAD_TARGET, json={'file': {'uri': 'files/x'}})
	assert upload_pdf_to_gemini(b'pdf', 'paper.pdf', 'k') == 'files/x'


@responses.activate
def test_upload_pdf_to_gemini_raises_when_response_missing_file_uri():
	responses.post(summariser.GEMINI_FILES_UPLOAD_URL, json={}, headers={'X-Goog-Upload-URL': UPLOAD_TARGET})
	responses.post(UPLOAD_TARGET, json={'file': {}})
	with pytest.raises(ValueError, match='missing file.uri'):
		upload_pdf_to_gemini(b'pdf', 'paper.pdf', 'k')


@responses.activate
def test_upload_pdf_to_gemini_raises_when_file_key_missing():
	responses.post(summariser.GEMINI_FILES_UPLOAD_URL, json={}, headers={'X-Goog-Upload-URL': UPLOAD_TARGET})
	responses.post(UPLOAD_TARGET, json={})
	with pytest.raises(ValueError, match='missing file.uri'):
		upload_pdf_to_gemini(b'pdf', 'paper.pdf', 'k')


# -- generate_with_file --


@responses.activate
def test_generate_with_file_returns_text_on_happy_path():
	responses.post(summariser.GEMINI_GENERATE_URL, json={'candidates': [{'content': {'parts': [{'text': 'ok'}]}}]})
	assert generate_with_file('prompt', 'files/abc', 'k') == 'ok'


@responses.activate
def test_generate_with_file_strips_response_whitespace():
	responses.post(summariser.GEMINI_GENERATE_URL, json={'candidates': [{'content': {'parts': [{'text': '  ok  '}]}}]})
	assert generate_with_file('prompt', 'files/abc', 'k') == 'ok'


@responses.activate
def test_generate_with_file_sends_file_data_and_prompt_parts():
	responses.post(summariser.GEMINI_GENERATE_URL, json={'candidates': [{'content': {'parts': [{'text': 'ok'}]}}]})
	generate_with_file('the-prompt', 'files/abc', 'k')
	body = json.loads(responses.calls[0].request.body)
	parts = body['contents'][0]['parts']
	assert parts[0] == {'file_data': {'mime_type': 'application/pdf', 'file_uri': 'files/abc'}}
	assert parts[1] == {'text': 'the-prompt'}


@responses.activate
def test_generate_with_file_sends_api_key_in_header_not_url():
	# Auth via header keeps the key out of any URL that may surface in HTTPError messages and downstream logs (PR #17 Copilot review).
	responses.post(summariser.GEMINI_GENERATE_URL, json={'candidates': [{'content': {'parts': [{'text': 'ok'}]}}]})
	generate_with_file('prompt', 'files/abc', 'my-secret-key')
	assert responses.calls[0].request.headers['x-goog-api-key'] == 'my-secret-key'
	assert 'my-secret-key' not in responses.calls[0].request.url


@responses.activate
def test_generate_with_file_raises_after_retries_exhausted_on_5xx():
	for _ in range(4):
		responses.post(summariser.GEMINI_GENERATE_URL, json={}, status=500)
	with pytest.raises(requests.HTTPError):
		generate_with_file('p', 'files/abc', 'k')
	assert len(responses.calls) == 4


@responses.activate
def test_generate_with_file_does_not_retry_on_400():
	responses.post(summariser.GEMINI_GENERATE_URL, json={}, status=400)
	with pytest.raises(requests.HTTPError):
		generate_with_file('p', 'files/abc', 'k')
	assert len(responses.calls) == 1


@responses.activate
def test_generate_with_file_retries_on_429_then_succeeds():
	responses.post(summariser.GEMINI_GENERATE_URL, json={}, status=429)
	responses.post(summariser.GEMINI_GENERATE_URL, json={'candidates': [{'content': {'parts': [{'text': 'ok'}]}}]})
	assert generate_with_file('p', 'files/abc', 'k') == 'ok'
	assert len(responses.calls) == 2


@responses.activate
def test_generate_with_file_retries_on_500_then_succeeds():
	responses.post(summariser.GEMINI_GENERATE_URL, json={}, status=500)
	responses.post(summariser.GEMINI_GENERATE_URL, json={'candidates': [{'content': {'parts': [{'text': 'ok'}]}}]})
	assert generate_with_file('p', 'files/abc', 'k') == 'ok'
	assert len(responses.calls) == 2


@responses.activate
def test_generate_with_file_honours_retry_after_seconds(_no_sleep):
	responses.post(summariser.GEMINI_GENERATE_URL, status=429, headers={'Retry-After': '11'})
	responses.post(summariser.GEMINI_GENERATE_URL, json={'candidates': [{'content': {'parts': [{'text': 'ok'}]}}]})
	generate_with_file('p', 'files/abc', 'k')
	# Retry-After is the only sleep this function triggers; the 11s override is honoured in place of the 5s default backoff for the first retry.
	delays = [c.args[0] for c in _no_sleep.call_args_list]
	assert delays == [11]


@responses.activate
def test_generate_with_file_raises_when_candidates_missing():
	responses.post(summariser.GEMINI_GENERATE_URL, json={})
	with pytest.raises(ValueError, match='missing expected fields'):
		generate_with_file('p', 'files/abc', 'k')


@responses.activate
def test_generate_with_file_raises_when_candidates_empty():
	responses.post(summariser.GEMINI_GENERATE_URL, json={'candidates': []})
	with pytest.raises(ValueError, match='missing expected fields'):
		generate_with_file('p', 'files/abc', 'k')


@responses.activate
def test_generate_with_file_raises_when_text_field_missing():
	responses.post(summariser.GEMINI_GENERATE_URL, json={'candidates': [{'content': {'parts': [{}]}}]})
	with pytest.raises(ValueError, match='missing expected fields'):
		generate_with_file('p', 'files/abc', 'k')


# -- summarise_paper: PDF path integration --


def _mock_pdf_pipeline(*, gemini_text=None, upload_url=UPLOAD_TARGET, file_uri='files/abc'):
	# Stand up the three HTTP boundaries the PDF path uses with happy-path defaults.
	if gemini_text is None:
		gemini_text = _valid_response()
	responses.get(PDF_URL, body=b'%PDF-1.4 fake')
	responses.post(summariser.GEMINI_FILES_UPLOAD_URL, json={}, headers={'X-Goog-Upload-URL': upload_url})
	responses.post(upload_url, json={'file': {'uri': file_uri}})
	responses.post(
		summariser.GEMINI_GENERATE_URL, json={'candidates': [{'content': {'parts': [{'text': gemini_text}]}}]}
	)


@responses.activate
def test_summarise_paper_pdf_path_returns_four_fields(mocker):
	_mock_pdf_pipeline()
	mocker.patch('summariser.db.get_summary', return_value=None)
	mocker.patch('summariser.db.upsert_summary')
	gemini_fn = mocker.Mock()

	result = summarise_paper(
		{'paperId': 'p1', 'title': 'T', 'abstract': 'a', 'openAccessPdf': {'url': PDF_URL}},
		gemini_fn,
		database_url='postgresql://x',
		api_key='fake-key',
	)

	assert result == {'methodology': 'method', 'findings': 'find', 'relevance': 'rel', 'limitations': 'lim'}
	# PDF path was used, so the abstract-mode callable must not have been invoked.
	gemini_fn.assert_not_called()


@responses.activate
def test_summarise_paper_pdf_path_persists_summary(mocker):
	_mock_pdf_pipeline()
	mocker.patch('summariser.db.get_summary', return_value=None)
	upsert = mocker.patch('summariser.db.upsert_summary')

	summarise_paper(
		{'paperId': 'p1', 'title': 'T', 'openAccessPdf': {'url': PDF_URL}},
		mocker.Mock(),
		database_url='postgresql://x',
		api_key='fake-key',
	)

	upsert.assert_called_once()
	call = upsert.call_args
	assert call.args[1] == 'p1'
	assert call.args[2]['methodology'] == 'method'
	assert call.args[3] == MODEL_VERSION


@responses.activate
def test_summarise_paper_falls_back_to_abstract_when_pdf_download_fails(mocker, capsys):
	responses.get(PDF_URL, json={'error': 'oops'}, status=500)
	mocker.patch('summariser.db.get_summary', return_value=None)
	mocker.patch('summariser.db.upsert_summary')
	gemini_fn = mocker.Mock(return_value=_valid_response())

	result = summarise_paper(
		{'paperId': 'p1', 'abstract': 'a', 'openAccessPdf': {'url': PDF_URL}},
		gemini_fn,
		database_url='postgresql://x',
		api_key='fake-key',
	)

	assert result is not None
	assert result['methodology'] == 'method'
	gemini_fn.assert_called_once()
	assert 'falling back to abstract' in capsys.readouterr().out


@responses.activate
def test_summarise_paper_falls_back_when_upload_start_fails(mocker):
	responses.get(PDF_URL, body=b'pdf')
	# 400 fails immediately without engaging the retry path so the test stays focused on the fallback behaviour rather than retry mechanics.
	responses.post(summariser.GEMINI_FILES_UPLOAD_URL, json={}, status=400)
	mocker.patch('summariser.db.get_summary', return_value=None)
	mocker.patch('summariser.db.upsert_summary')
	gemini_fn = mocker.Mock(return_value=_valid_response())

	result = summarise_paper(
		{'paperId': 'p1', 'abstract': 'a', 'openAccessPdf': {'url': PDF_URL}},
		gemini_fn,
		database_url='postgresql://x',
		api_key='fake-key',
	)

	assert result is not None
	gemini_fn.assert_called_once()


@responses.activate
def test_summarise_paper_falls_back_when_generate_fails(mocker):
	responses.get(PDF_URL, body=b'pdf')
	responses.post(summariser.GEMINI_FILES_UPLOAD_URL, json={}, headers={'X-Goog-Upload-URL': UPLOAD_TARGET})
	responses.post(UPLOAD_TARGET, json={'file': {'uri': 'files/abc'}})
	responses.post(summariser.GEMINI_GENERATE_URL, json={}, status=400)
	mocker.patch('summariser.db.get_summary', return_value=None)
	mocker.patch('summariser.db.upsert_summary')
	gemini_fn = mocker.Mock(return_value=_valid_response())

	result = summarise_paper(
		{'paperId': 'p1', 'abstract': 'a', 'openAccessPdf': {'url': PDF_URL}},
		gemini_fn,
		database_url='postgresql://x',
		api_key='fake-key',
	)

	assert result is not None
	gemini_fn.assert_called_once()


@responses.activate
def test_summarise_paper_falls_back_when_pdf_response_is_malformed(mocker):
	_mock_pdf_pipeline(gemini_text='not json at all')
	mocker.patch('summariser.db.get_summary', return_value=None)
	mocker.patch('summariser.db.upsert_summary')
	gemini_fn = mocker.Mock(return_value=_valid_response())

	result = summarise_paper(
		{'paperId': 'p1', 'abstract': 'a', 'openAccessPdf': {'url': PDF_URL}},
		gemini_fn,
		database_url='postgresql://x',
		api_key='fake-key',
	)

	assert result is not None
	gemini_fn.assert_called_once()


@responses.activate
def test_summarise_paper_returns_none_when_pdf_fails_and_no_abstract(mocker):
	responses.get(PDF_URL, json={'error': 'oops'}, status=500)
	mocker.patch('summariser.db.get_summary', return_value=None)
	upsert = mocker.patch('summariser.db.upsert_summary')
	gemini_fn = mocker.Mock()

	result = summarise_paper(
		{'paperId': 'p1', 'openAccessPdf': {'url': PDF_URL}},
		gemini_fn,
		database_url='postgresql://x',
		api_key='fake-key',
	)

	assert result is None
	upsert.assert_not_called()


def test_summarise_paper_skips_pdf_path_when_no_api_key(mocker):
	# Without an api_key the PDF path can't authenticate, so we go straight to the abstract path rather than burn the download trying.
	mocker.patch('summariser.db.get_summary', return_value=None)
	mocker.patch('summariser.db.upsert_summary')
	get_request = mocker.patch('summariser.requests.get')
	gemini_fn = mocker.Mock(return_value=_valid_response())

	result = summarise_paper(
		{'paperId': 'p1', 'abstract': 'a', 'openAccessPdf': {'url': PDF_URL}}, gemini_fn, database_url='postgresql://x'
	)

	assert result is not None
	get_request.assert_not_called()
	gemini_fn.assert_called_once()


def test_summarise_paper_cache_hit_short_circuits_even_with_pdf_url(mocker):
	cached = {'methodology': 'cached', 'findings': 'f', 'relevance': 'r', 'limitations': 'l'}
	mocker.patch('summariser.db.get_summary', return_value=cached)
	get_request = mocker.patch('summariser.requests.get')
	gemini_fn = mocker.Mock()

	result = summarise_paper(
		{'paperId': 'p1', 'abstract': 'a', 'openAccessPdf': {'url': PDF_URL}},
		gemini_fn,
		database_url='postgresql://x',
		api_key='fake-key',
	)

	assert result == cached
	get_request.assert_not_called()
	gemini_fn.assert_not_called()


# -- quota exhaustion (RESOURCE_EXHAUSTED) --


@responses.activate
def test_generate_with_file_raises_quota_exhausted_on_resource_exhausted_429():
	import scorer

	responses.post(summariser.GEMINI_GENERATE_URL, json={'error': {'status': 'RESOURCE_EXHAUSTED'}}, status=429)
	with pytest.raises(scorer.GeminiQuotaExhausted):
		generate_with_file('p', 'files/abc', 'k')


@responses.activate
def test_generate_with_file_skips_backoff_on_quota_exhausted(_no_sleep):
	import scorer

	responses.post(summariser.GEMINI_GENERATE_URL, json={'error': {'status': 'RESOURCE_EXHAUSTED'}}, status=429)
	with pytest.raises(scorer.GeminiQuotaExhausted):
		generate_with_file('p', 'files/abc', 'k')
	# No backoff sleeps; quota detection bypasses the retry loop entirely.
	_no_sleep.assert_not_called()


@responses.activate
def test_upload_pdf_raises_quota_exhausted_on_resource_exhausted_429():
	import scorer

	responses.post(summariser.GEMINI_FILES_UPLOAD_URL, json={'error': {'status': 'RESOURCE_EXHAUSTED'}}, status=429)
	with pytest.raises(scorer.GeminiQuotaExhausted):
		upload_pdf_to_gemini(b'pdf', 'paper.pdf', 'k')


def test_summarise_paper_propagates_quota_exhausted_from_abstract_path(mocker):
	import scorer

	mocker.patch('summariser.db.get_summary', return_value=None)
	gemini_fn = mocker.Mock(side_effect=scorer.GeminiQuotaExhausted('quota'))
	with pytest.raises(scorer.GeminiQuotaExhausted, match='quota'):
		summarise_paper({'paperId': 'p1', 'abstract': 'a'}, gemini_fn, database_url='postgresql://x')


@responses.activate
def test_summarise_paper_propagates_quota_exhausted_from_pdf_path(mocker):
	import scorer

	responses.get(PDF_URL, body=b'pdf')
	responses.post(summariser.GEMINI_FILES_UPLOAD_URL, json={}, headers={'X-Goog-Upload-URL': UPLOAD_TARGET})
	responses.post(UPLOAD_TARGET, json={'file': {'uri': 'files/abc'}})
	responses.post(summariser.GEMINI_GENERATE_URL, json={'error': {'status': 'RESOURCE_EXHAUSTED'}}, status=429)
	mocker.patch('summariser.db.get_summary', return_value=None)
	gemini_fn = mocker.Mock(return_value=_valid_response())
	# PDF path raises quota exhausted; abstract fallback must not run.
	with pytest.raises(scorer.GeminiQuotaExhausted):
		summarise_paper(
			{'paperId': 'p1', 'abstract': 'a', 'openAccessPdf': {'url': PDF_URL}},
			gemini_fn,
			database_url='postgresql://x',
			api_key='fake-key',
		)
	gemini_fn.assert_not_called()


# -- budget exhaustion (caller-owned, raised through on_gemini_call or gemini_fn) --


def test_summarise_paper_propagates_budget_exhausted_from_abstract_path(mocker):
	import scorer

	mocker.patch('summariser.db.get_summary', return_value=None)
	gemini_fn = mocker.Mock(side_effect=scorer.GeminiBudgetExhausted('budget'))
	with pytest.raises(scorer.GeminiBudgetExhausted, match='budget'):
		summarise_paper({'paperId': 'p1', 'abstract': 'a'}, gemini_fn, database_url='postgresql://x')


def test_summarise_paper_propagates_budget_exhausted_from_pdf_path(mocker):
	import scorer

	# The PDF path's _post_with_retry fires on_attempt before each post; main wires that to a counter that raises when the budget is reached.
	def raising_on_call():
		raise scorer.GeminiBudgetExhausted('budget')

	mocker.patch('summariser.db.get_summary', return_value=None)
	mocker.patch('summariser.download_pdf', return_value=b'pdf')
	gemini_fn = mocker.Mock()

	with pytest.raises(scorer.GeminiBudgetExhausted, match='budget'):
		summarise_paper(
			{'paperId': 'p1', 'abstract': 'a', 'openAccessPdf': {'url': PDF_URL}},
			gemini_fn,
			database_url='postgresql://x',
			api_key='fake-key',
			on_gemini_call=raising_on_call,
		)
	gemini_fn.assert_not_called()
