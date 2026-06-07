import json

import pytest
import requests
import responses

import config
from summariser import (
	MISSING_FIELD_PLACEHOLDER,
	SummariserContext,
	build_fewshot_block,
	download_pdf,
	parse_summary_response,
	summarise_paper,
)

# Tests share a single cfg derived from the seed file. URLs and the persisted model_version come from it so production builds match what `responses` mocks expect.
_TEST_CFG = config.Config(**config.load_seed())
MODEL_VERSION = _TEST_CFG.gemini_model
GEMINI_GENERATE_URL = f'{_TEST_CFG.gemini_base_url}/models/{_TEST_CFG.gemini_model}:generateContent'
GEMINI_FILES_UPLOAD_URL = f'{_TEST_CFG.gemini_upload_base_url}/files'
UPLOAD_TARGET = 'https://upload.example.com/sessions/abc'

_summarise_paper = summarise_paper


def _ctx(**kwargs):
	return SummariserContext(cfg=_TEST_CFG, **kwargs)


def _summarise(paper, gemini_fn, **kwargs):
	return _summarise_paper(paper, gemini_fn, _ctx(**kwargs))


@pytest.fixture(autouse=True)
def _no_sleep(mocker):
	# Sleeps now live in the transport module.
	return mocker.patch('gemini.time.sleep')


@pytest.fixture(autouse=True)
def mock_session(mocker):
	# summariser opens short DB sessions internally via db.session; patch the helper to yield a stable mock conn so tests passing database_url= get a usable session.
	conn = mocker.MagicMock()
	session_cm = mocker.MagicMock()
	session_cm.__enter__.return_value = conn
	session_cm.__exit__.return_value = False
	mocker.patch('summariser.db.session', return_value=session_cm)
	# Default the few-shot gate off so existing summarise_paper tests keep a byte-identical prompt; the few-shot tests override these.
	mocker.patch('summariser.db.count_summary_feedback', return_value=0)
	mocker.patch('summariser.db.get_paper_embedding', return_value=None)
	mocker.patch('summariser.db.similar_feedback_papers', return_value=[])
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

	result = _summarise({'paperId': 'p1', 'abstract': 'a'}, gemini_fn, database_url='postgresql://x')

	assert result == cached
	gemini_fn.assert_not_called()
	upsert.assert_not_called()


def test_summarise_paper_skips_cache_check_when_no_conn(mocker):
	get_summary = mocker.patch('summariser.db.get_summary')
	mocker.patch('summariser.db.upsert_summary')
	gemini_fn = mocker.Mock(return_value=_valid_response())

	_summarise({'paperId': 'p1', 'abstract': 'a'}, gemini_fn, database_url=None)

	get_summary.assert_not_called()


def test_summarise_paper_skips_cache_check_when_no_paper_id(mocker):
	# A paper without a paperId can't be keyed in the cache. We still summarise (some upstream paths may want a transient summary) but persistence is skipped.
	get_summary = mocker.patch('summariser.db.get_summary')
	upsert = mocker.patch('summariser.db.upsert_summary')
	gemini_fn = mocker.Mock(return_value=_valid_response())

	result = _summarise({'abstract': 'a'}, gemini_fn, database_url='postgresql://x')

	get_summary.assert_not_called()
	upsert.assert_not_called()
	assert result is not None
	assert result['methodology'] == 'method'


# -- summarise_paper: happy path --


def test_summarise_paper_calls_gemini_and_returns_four_fields(mocker):
	mocker.patch('summariser.db.get_summary', return_value=None)
	mocker.patch('summariser.db.upsert_summary')
	gemini_fn = mocker.Mock(return_value=_valid_response())

	result = _summarise({'paperId': 'p1', 'abstract': 'a'}, gemini_fn, database_url='postgresql://x')

	assert result == {'methodology': 'method', 'findings': 'find', 'relevance': 'rel', 'limitations': 'lim'}
	gemini_fn.assert_called_once()


def test_summarise_paper_persists_fresh_summary(mocker, mock_session):
	mocker.patch('summariser.db.get_summary', return_value=None)
	upsert = mocker.patch('summariser.db.upsert_summary')
	gemini_fn = mocker.Mock(return_value=_valid_response())

	_summarise({'paperId': 'p1', 'abstract': 'a'}, gemini_fn, database_url='postgresql://x')

	upsert.assert_called_once()
	call = upsert.call_args
	assert call.args[0] is mock_session['conn']
	assert call.args[1] == 'p1'
	assert call.args[2] == {'methodology': 'method', 'findings': 'find', 'relevance': 'rel', 'limitations': 'lim'}
	assert call.args[3] == MODEL_VERSION


def test_summarise_paper_does_not_persist_when_conn_is_none(mocker):
	upsert = mocker.patch('summariser.db.upsert_summary')
	gemini_fn = mocker.Mock(return_value=_valid_response())

	_summarise({'paperId': 'p1', 'abstract': 'a'}, gemini_fn, database_url=None)

	upsert.assert_not_called()


def test_summarise_paper_does_not_persist_when_no_paper_id(mocker):
	mocker.patch('summariser.db.get_summary')
	upsert = mocker.patch('summariser.db.upsert_summary')
	gemini_fn = mocker.Mock(return_value=_valid_response())

	_summarise({'abstract': 'a'}, gemini_fn, database_url='postgresql://x')

	upsert.assert_not_called()


def test_summarise_paper_prompt_includes_abstract_and_context(mocker):
	mocker.patch('summariser.db.get_summary', return_value=None)
	mocker.patch('summariser.db.upsert_summary')
	captured = {}

	def fake_gemini(prompt):
		captured['prompt'] = prompt
		return _valid_response()

	_summarise({'paperId': 'p1', 'abstract': 'unique-abstract-text-xyz'}, fake_gemini, database_url='postgresql://x')

	assert 'unique-abstract-text-xyz' in captured['prompt']
	assert 'RESEARCH CONTEXT' in captured['prompt']


# -- summarise_paper: failure modes --


def test_summarise_paper_returns_none_when_no_abstract(mocker, capsys):
	get_summary = mocker.patch('summariser.db.get_summary', return_value=None)
	upsert = mocker.patch('summariser.db.upsert_summary')
	gemini_fn = mocker.Mock()

	result = _summarise({'paperId': 'p1'}, gemini_fn, database_url='postgresql://x')

	assert result is None
	gemini_fn.assert_not_called()
	upsert.assert_not_called()
	# Cache miss still happens before we discover no abstract; that's fine.
	get_summary.assert_called_once()
	assert 'No abstract available' in capsys.readouterr().out


def test_summarise_paper_returns_none_when_abstract_is_empty(mocker):
	mocker.patch('summariser.db.get_summary', return_value=None)
	gemini_fn = mocker.Mock()
	assert _summarise({'paperId': 'p1', 'abstract': ''}, gemini_fn, database_url='postgresql://x') is None
	gemini_fn.assert_not_called()


def test_summarise_paper_returns_none_on_gemini_exception(mocker, capsys):
	mocker.patch('summariser.db.get_summary', return_value=None)
	upsert = mocker.patch('summariser.db.upsert_summary')
	gemini_fn = mocker.Mock(side_effect=Exception('rate limit'))

	result = _summarise({'paperId': 'p1', 'abstract': 'a'}, gemini_fn, database_url='postgresql://x')

	assert result is None
	upsert.assert_not_called()
	assert 'Summariser Gemini error' in capsys.readouterr().out


def test_summarise_paper_returns_none_on_malformed_response(mocker):
	mocker.patch('summariser.db.get_summary', return_value=None)
	upsert = mocker.patch('summariser.db.upsert_summary')
	gemini_fn = mocker.Mock(return_value='not json at all')

	result = _summarise({'paperId': 'p1', 'abstract': 'a'}, gemini_fn, database_url='postgresql://x')

	assert result is None
	upsert.assert_not_called()


def test_summarise_paper_persists_partial_response_with_placeholders(mocker):
	# A response missing one field should still be persisted with the placeholder; this is the abstract-only fallback behaviour the spec calls for.
	mocker.patch('summariser.db.get_summary', return_value=None)
	upsert = mocker.patch('summariser.db.upsert_summary')
	body = json.dumps({'methodology': 'm', 'findings': 'f', 'relevance_to_research': 'r'})  # no limitations
	gemini_fn = mocker.Mock(return_value=body)

	result = _summarise({'paperId': 'p1', 'abstract': 'a'}, gemini_fn, database_url='postgresql://x')

	assert result is not None
	assert result['limitations'] == MISSING_FIELD_PLACEHOLDER
	upsert.assert_called_once()
	assert upsert.call_args.args[2]['limitations'] == MISSING_FIELD_PLACEHOLDER


def test_summarise_paper_handles_paper_with_missing_title(mocker):
	# The log lines slice title to 60 chars; a missing title must not crash.
	mocker.patch('summariser.db.get_summary', return_value=None)
	mocker.patch('summariser.db.upsert_summary')
	gemini_fn = mocker.Mock(return_value=_valid_response())

	result = _summarise({'paperId': 'p1', 'abstract': 'a'}, gemini_fn, database_url='postgresql://x')

	assert result is not None


# -- module surface --


def test_module_version_is_set():
	# Persisted alongside summaries so a later prompt or model change can be reasoned about against historical data.
	assert MODEL_VERSION


# -- on_gemini_call callback (per-attempt for PDF path; abstract path counts inside gemini_fn) --


def test_on_gemini_call_does_not_fire_for_abstract_path_via__summarise(mocker):
	# Abstract path no longer relays on_gemini_call; the counter fires inside gemini_fn (scorer.gemini's on_attempt, wired by main). Mocked gemini_fn = 0 fires.
	mocker.patch('summariser.db.get_summary', return_value=None)
	mocker.patch('summariser.db.upsert_summary')
	gemini_fn = mocker.Mock(return_value=_valid_response())
	counter = mocker.Mock()

	_summarise({'paperId': 'p1', 'abstract': 'a'}, gemini_fn, database_url='postgresql://x', on_gemini_call=counter)

	counter.assert_not_called()


def test_on_gemini_call_does_not_fire_on_cache_hit(mocker):
	mocker.patch(
		'summariser.db.get_summary',
		return_value={'methodology': 'm', 'findings': 'f', 'relevance': 'r', 'limitations': 'l'},
	)
	gemini_fn = mocker.Mock()
	counter = mocker.Mock()

	_summarise({'paperId': 'p1', 'abstract': 'a'}, gemini_fn, database_url='postgresql://x', on_gemini_call=counter)

	counter.assert_not_called()


def test_on_gemini_call_does_not_fire_when_abstract_gemini_raises(mocker):
	# gemini_fn is mocked and raises; the abstract path doesn't relay on_gemini_call, so it stays at 0 fires. Real per-attempt firing is in test_scorer.py.
	mocker.patch('summariser.db.get_summary', return_value=None)
	gemini_fn = mocker.Mock(side_effect=Exception('boom'))
	counter = mocker.Mock()

	_summarise({'paperId': 'p1', 'abstract': 'a'}, gemini_fn, database_url='postgresql://x', on_gemini_call=counter)

	counter.assert_not_called()


@responses.activate
def test_on_gemini_call_fires_per_gemini_attempt_in_pdf_path(mocker):
	# PDF success path fires per attempt of upload-init + generate (signed-URL data upload is not a Gemini API call). One attempt each on happy path = 2.
	responses.get(PDF_URL, body=b'%PDF-1.4 fake')
	responses.post(GEMINI_FILES_UPLOAD_URL, json={}, headers={'X-Goog-Upload-URL': UPLOAD_TARGET})
	responses.post(UPLOAD_TARGET, json={'file': {'uri': 'files/abc'}})
	responses.post(GEMINI_GENERATE_URL, json={'candidates': [{'content': {'parts': [{'text': _valid_response()}]}}]})
	mocker.patch('summariser.db.get_summary', return_value=None)
	mocker.patch('summariser.db.upsert_summary')
	counter = mocker.Mock()

	_summarise(
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

	_summarise(
		{'paperId': 'p1', 'abstract': 'a', 'openAccessPdf': {'url': PDF_URL}},
		gemini_fn,
		database_url='postgresql://x',
		api_key='fake-key',
		on_gemini_call=counter,
	)

	counter.assert_not_called()


@responses.activate
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

	_summarise({'paperId': 'p1', 'abstract': 'a'}, gemini_snap, database_url='postgresql://x')

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


# -- summarise_paper: PDF path integration --


def _mock_pdf_pipeline(*, gemini_text=None, upload_url=UPLOAD_TARGET, file_uri='files/abc'):
	# Stand up the three HTTP boundaries the PDF path uses with happy-path defaults.
	if gemini_text is None:
		gemini_text = _valid_response()
	responses.get(PDF_URL, body=b'%PDF-1.4 fake')
	responses.post(GEMINI_FILES_UPLOAD_URL, json={}, headers={'X-Goog-Upload-URL': upload_url})
	responses.post(upload_url, json={'file': {'uri': file_uri}})
	responses.post(GEMINI_GENERATE_URL, json={'candidates': [{'content': {'parts': [{'text': gemini_text}]}}]})


@responses.activate
def test_summarise_paper_pdf_path_returns_four_fields(mocker):
	_mock_pdf_pipeline()
	mocker.patch('summariser.db.get_summary', return_value=None)
	mocker.patch('summariser.db.upsert_summary')
	gemini_fn = mocker.Mock()

	result = _summarise(
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

	_summarise(
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

	result = _summarise(
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
	responses.post(GEMINI_FILES_UPLOAD_URL, json={}, status=400)
	mocker.patch('summariser.db.get_summary', return_value=None)
	mocker.patch('summariser.db.upsert_summary')
	gemini_fn = mocker.Mock(return_value=_valid_response())

	result = _summarise(
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
	responses.post(GEMINI_FILES_UPLOAD_URL, json={}, headers={'X-Goog-Upload-URL': UPLOAD_TARGET})
	responses.post(UPLOAD_TARGET, json={'file': {'uri': 'files/abc'}})
	responses.post(GEMINI_GENERATE_URL, json={}, status=400)
	mocker.patch('summariser.db.get_summary', return_value=None)
	mocker.patch('summariser.db.upsert_summary')
	gemini_fn = mocker.Mock(return_value=_valid_response())

	result = _summarise(
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

	result = _summarise(
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

	result = _summarise(
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

	result = _summarise(
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

	result = _summarise(
		{'paperId': 'p1', 'abstract': 'a', 'openAccessPdf': {'url': PDF_URL}},
		gemini_fn,
		database_url='postgresql://x',
		api_key='fake-key',
	)

	assert result == cached
	get_request.assert_not_called()
	gemini_fn.assert_not_called()


# -- quota exhaustion (daily PerDay quota) --

_PER_DAY_429 = {
	'error': {
		'code': 429,
		'status': 'RESOURCE_EXHAUSTED',
		'details': [
			{
				'@type': 'type.googleapis.com/google.rpc.QuotaFailure',
				'violations': [{'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier'}],
			}
		],
	}
}


@responses.activate
@responses.activate
@responses.activate
def test_summarise_paper_propagates_quota_exhausted_from_abstract_path(mocker):
	import gemini

	mocker.patch('summariser.db.get_summary', return_value=None)
	gemini_fn = mocker.Mock(side_effect=gemini.GeminiQuotaExhausted('quota'))
	with pytest.raises(gemini.GeminiQuotaExhausted, match='quota'):
		_summarise({'paperId': 'p1', 'abstract': 'a'}, gemini_fn, database_url='postgresql://x')


@responses.activate
def test_summarise_paper_propagates_quota_exhausted_from_pdf_path(mocker):
	import gemini

	responses.get(PDF_URL, body=b'pdf')
	responses.post(GEMINI_FILES_UPLOAD_URL, json={}, headers={'X-Goog-Upload-URL': UPLOAD_TARGET})
	responses.post(UPLOAD_TARGET, json={'file': {'uri': 'files/abc'}})
	responses.post(GEMINI_GENERATE_URL, json=_PER_DAY_429, status=429)
	mocker.patch('summariser.db.get_summary', return_value=None)
	gemini_fn = mocker.Mock(return_value=_valid_response())
	# PDF path raises quota exhausted; abstract fallback must not run.
	with pytest.raises(gemini.GeminiQuotaExhausted):
		_summarise(
			{'paperId': 'p1', 'abstract': 'a', 'openAccessPdf': {'url': PDF_URL}},
			gemini_fn,
			database_url='postgresql://x',
			api_key='fake-key',
		)
	gemini_fn.assert_not_called()


# -- budget exhaustion (caller-owned, raised through on_gemini_call or gemini_fn) --


def test_summarise_paper_propagates_budget_exhausted_from_abstract_path(mocker):
	import gemini

	mocker.patch('summariser.db.get_summary', return_value=None)
	gemini_fn = mocker.Mock(side_effect=gemini.GeminiBudgetExhausted('budget'))
	with pytest.raises(gemini.GeminiBudgetExhausted, match='budget'):
		_summarise({'paperId': 'p1', 'abstract': 'a'}, gemini_fn, database_url='postgresql://x')


def test_summarise_paper_propagates_budget_exhausted_from_pdf_path(mocker):
	import gemini

	# The PDF path's _post_with_retry fires on_attempt before each post; main wires that to a counter that raises when the budget is reached.
	def raising_on_call():
		raise gemini.GeminiBudgetExhausted('budget')

	mocker.patch('summariser.db.get_summary', return_value=None)
	mocker.patch('summariser.download_pdf', return_value=b'pdf')
	gemini_fn = mocker.Mock()

	with pytest.raises(gemini.GeminiBudgetExhausted, match='budget'):
		_summarise(
			{'paperId': 'p1', 'abstract': 'a', 'openAccessPdf': {'url': PDF_URL}},
			gemini_fn,
			database_url='postgresql://x',
			api_key='fake-key',
			on_gemini_call=raising_on_call,
		)
	gemini_fn.assert_not_called()


# -- few-shot field-feedback injection (layer 1 of the feedback loop) --


def _feedback_paper(title='Neighbour', **fields):
	# Shapes a db.similar_feedback_papers row: per-field {'rating', 'correction'} as get_latest_field_feedback returns.
	return {'paper_id': 'n1', 'title': title, 'feedback': fields}


def test_build_fewshot_block_renders_good_fields_rated_four_or_more():
	papers = [
		_feedback_paper(methodology={'rating': 5, 'correction': None}, findings={'rating': 4, 'correction': None})
	]
	block = build_fewshot_block(papers)
	assert 'methodology' in block
	assert 'findings' in block
	assert 'happy with' in block


def test_build_fewshot_block_renders_bad_fields_with_correction_text():
	papers = [_feedback_paper(limitations={'rating': 2, 'correction': 'should mention sample size'})]
	block = build_fewshot_block(papers)
	assert 'limitations' in block
	assert 'should mention sample size' in block
	assert 'corrected' in block


def test_build_fewshot_block_excludes_middling_ratings():
	# Rating 3 is neither a good-shape example (>=4) nor a correction to avoid (<=2).
	papers = [_feedback_paper(methodology={'rating': 3, 'correction': 'ignored'})]
	assert build_fewshot_block(papers) == ''


def test_build_fewshot_block_skips_low_rated_field_without_correction():
	# Nothing to inject as a pattern to avoid when the low rating carries no correction text.
	papers = [_feedback_paper(findings={'rating': 1, 'correction': None})]
	assert build_fewshot_block(papers) == ''


def test_build_fewshot_block_ignores_fields_with_no_rating():
	papers = [_feedback_paper(methodology={'rating': None, 'correction': 'orphan correction'})]
	assert build_fewshot_block(papers) == ''


def test_build_fewshot_block_empty_when_no_neighbours():
	assert build_fewshot_block([]) == ''


def _mock_fewshot_db(mocker, feedback_count, embedding, neighbours):
	mocker.patch('summariser.db.get_summary', return_value=None)
	mocker.patch('summariser.db.upsert_summary')
	mocker.patch('summariser.db.count_summary_feedback', return_value=feedback_count)
	mocker.patch('summariser.db.get_paper_embedding', return_value=embedding)
	return mocker.patch('summariser.db.similar_feedback_papers', return_value=neighbours)


def _captured_prompt(gemini_fn):
	return gemini_fn.call_args.args[0]


def test_fewshot_gate_off_below_five_feedback_rows_keeps_prompt_unchanged(mocker):
	similar = _mock_fewshot_db(mocker, feedback_count=4, embedding=[0.1] * 384, neighbours=[])
	gemini_fn = mocker.Mock(return_value=_valid_response())
	_summarise({'paperId': 'p1', 'abstract': 'a'}, gemini_fn, database_url='postgresql://x')
	expected = config.SUMMARISER_PROMPT.format(
		research_context=_TEST_CFG.research_context, source_material='Abstract:\na'
	)
	assert _captured_prompt(gemini_fn) == expected
	similar.assert_not_called()


def test_fewshot_gate_on_at_five_feedback_rows_injects_block(mocker):
	neighbours = [_feedback_paper(methodology={'rating': 5, 'correction': None})]
	_mock_fewshot_db(mocker, feedback_count=5, embedding=[0.1] * 384, neighbours=neighbours)
	gemini_fn = mocker.Mock(return_value=_valid_response())
	_summarise({'paperId': 'p1', 'abstract': 'a'}, gemini_fn, database_url='postgresql://x')
	prompt = _captured_prompt(gemini_fn)
	assert 'FEEDBACK FROM SIMILAR PAPERS' in prompt
	assert 'methodology' in prompt


def test_fewshot_skipped_when_paper_has_no_embedding(mocker):
	similar = _mock_fewshot_db(mocker, feedback_count=10, embedding=None, neighbours=[])
	gemini_fn = mocker.Mock(return_value=_valid_response())
	_summarise({'paperId': 'p1', 'abstract': 'a'}, gemini_fn, database_url='postgresql://x')
	expected = config.SUMMARISER_PROMPT.format(
		research_context=_TEST_CFG.research_context, source_material='Abstract:\na'
	)
	assert _captured_prompt(gemini_fn) == expected
	similar.assert_not_called()


def test_fewshot_excludes_the_paper_being_summarised_from_its_own_examples(mocker):
	similar = _mock_fewshot_db(mocker, feedback_count=5, embedding=[0.1] * 384, neighbours=[])
	gemini_fn = mocker.Mock(return_value=_valid_response())
	_summarise({'paperId': 'p1', 'abstract': 'a'}, gemini_fn, database_url='postgresql://x')
	assert similar.call_args.kwargs['exclude_id'] == 'p1'


def test_fewshot_not_built_without_database_url(mocker):
	mocker.patch('summariser.db.get_summary', return_value=None)
	count = mocker.patch('summariser.db.count_summary_feedback')
	gemini_fn = mocker.Mock(return_value=_valid_response())
	_summarise({'paperId': 'p1', 'abstract': 'a'}, gemini_fn)
	count.assert_not_called()
