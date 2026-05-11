import json

import pytest
import requests
import responses

import scorer
from scorer import _score_chunk, apply_scores, gemini, parse_gemini_scores, score_and_summarise


@pytest.fixture(autouse=True)
def no_sleep(mocker):
	return mocker.patch('scorer.time.sleep')


def _valid_score(index, score):
	return {'index': index, 'relevance_score': score, 'relevance_reason': 'r'}


# -- parse_gemini_scores --


def test_parse_returns_list_for_valid_json():
	assert parse_gemini_scores('[{"index": 0, "relevance_score": 8}]') == [{'index': 0, 'relevance_score': 8}]


def test_parse_strips_markdown_fences():
	assert parse_gemini_scores('```json\n[{"index": 0}]\n```') == [{'index': 0}]


def test_parse_strips_bare_code_fences():
	assert parse_gemini_scores('```\n[{"index": 0}]\n```') == [{'index': 0}]


def test_parse_strips_surrounding_whitespace():
	assert parse_gemini_scores('   \n[{"index": 0}]\n   ') == [{'index': 0}]


def test_parse_returns_empty_list_for_empty_array():
	assert parse_gemini_scores('[]') == []


def test_parse_raises_on_malformed_json():
	with pytest.raises(json.JSONDecodeError):
		parse_gemini_scores('not valid')


# -- apply_scores --


def test_apply_scores_keeps_paper_above_threshold():
	papers = [{'title': 'p'}]
	enriched, _ = apply_scores(papers, [_valid_score(0, 8)], threshold=6)
	assert len(enriched) == 1
	assert enriched[0]['ai_score'] == 8
	assert enriched[0]['ai_reason'] == 'r'


def test_apply_scores_drops_paper_below_threshold():
	papers = [{'title': 'p'}]
	enriched, _ = apply_scores(papers, [_valid_score(0, 5)], threshold=6)
	assert enriched == []


def test_apply_scores_keeps_paper_at_threshold_boundary():
	papers = [{'title': 'p'}]
	enriched, _ = apply_scores(papers, [_valid_score(0, 6)], threshold=6)
	assert len(enriched) == 1


def test_apply_scores_skips_paper_with_no_matching_index():
	papers = [{'title': 'p1'}, {'title': 'p2'}]
	enriched, _ = apply_scores(papers, [_valid_score(0, 8)], threshold=6)
	assert [p['title'] for p in enriched] == ['p1']


def test_apply_scores_empty_papers_and_empty_scores_returns_empty():
	assert apply_scores([], [], threshold=6) == ([], set())


def test_apply_scores_empty_scores_with_papers_drops_all():
	papers = [{'title': 'p1'}, {'title': 'p2'}]
	enriched, _ = apply_scores(papers, [], threshold=6)
	assert enriched == []


def test_apply_scores_missing_score_field_drops_paper(capsys):
	papers = [{'title': 'p'}]
	scores = [{'index': 0, 'relevance_reason': 'r'}]
	enriched, _ = apply_scores(papers, scores, threshold=1)
	assert enriched == []
	assert 'invalid score' in capsys.readouterr().out


def test_apply_scores_string_score_drops_paper(capsys):
	papers = [{'title': 'p'}]
	scores = [{'index': 0, 'relevance_score': '8', 'relevance_reason': 'r'}]
	enriched, _ = apply_scores(papers, scores, threshold=6)
	assert enriched == []
	assert 'invalid score' in capsys.readouterr().out


def test_apply_scores_none_score_drops_paper(capsys):
	papers = [{'title': 'p'}]
	scores = [{'index': 0, 'relevance_score': None, 'relevance_reason': 'r'}]
	enriched, _ = apply_scores(papers, scores, threshold=6)
	assert enriched == []
	assert 'invalid score' in capsys.readouterr().out


def test_apply_scores_bool_score_drops_paper(capsys):
	# True passes isinstance(_, int) in Python; refuse it explicitly so a stray
	# boolean from a malformed response can't be promoted to a numeric score.
	papers = [{'title': 'p'}]
	scores = [{'index': 0, 'relevance_score': True, 'relevance_reason': 'r'}]
	enriched, _ = apply_scores(papers, scores, threshold=0)
	assert enriched == []
	assert 'invalid score' in capsys.readouterr().out


def test_apply_scores_float_score_drops_paper(capsys):
	# Downstream rendering requires int; refuse floats rather than coerce silently.
	papers = [{'title': 'p'}]
	scores = [{'index': 0, 'relevance_score': 8.5, 'relevance_reason': 'r'}]
	enriched, _ = apply_scores(papers, scores, threshold=6)
	assert enriched == []
	assert 'invalid score' in capsys.readouterr().out


# -- _score_chunk --


def test_score_chunk_returns_empty_for_empty_input(mocker):
	gemini_fn = mocker.Mock()
	assert _score_chunk([], gemini_fn) == ([], set())
	gemini_fn.assert_not_called()


def test_score_chunk_returns_empty_on_gemini_exception(capsys, mocker):
	gemini_fn = mocker.Mock(side_effect=Exception('boom'))
	papers = [{'paperId': 'p1', 'title': 'p', 'abstract': 'a'}]
	assert _score_chunk(papers, gemini_fn) == ([], set())
	assert 'Batch Gemini error' in capsys.readouterr().out


def test_score_chunk_returns_empty_on_malformed_gemini_response(capsys, mocker):
	gemini_fn = mocker.Mock(return_value='not json')
	papers = [{'paperId': 'p1', 'title': 'p', 'abstract': 'a'}]
	assert _score_chunk(papers, gemini_fn) == ([], set())
	assert 'Batch Gemini error' in capsys.readouterr().out


def test_score_chunk_replaces_missing_abstract_in_prompt():
	captured = {}

	def fake_gemini(prompt):
		captured['prompt'] = prompt
		return json.dumps([_valid_score(0, 8)])

	_score_chunk([{'title': 'p'}], fake_gemini)
	assert 'No abstract available.' in captured['prompt']


def test_score_chunk_returns_enriched_papers_on_happy_path(mocker):
	gemini_fn = mocker.Mock(return_value=json.dumps([_valid_score(0, 8)]))
	papers = [{'paperId': 'p1', 'title': 'p', 'abstract': 'a'}]
	enriched, responded = _score_chunk(papers, gemini_fn)
	assert len(enriched) == 1
	assert enriched[0]['ai_score'] == 8
	assert responded == {'p1'}


# -- score_and_summarise --


def test_score_and_summarise_empty_input(mocker):
	gemini_fn = mocker.Mock()
	assert score_and_summarise([], gemini_fn) == ([], set())
	gemini_fn.assert_not_called()


def test_score_and_summarise_single_small_batch(mocker):
	mock_chunk = mocker.patch('scorer._score_chunk', return_value=([{'title': 'x'}], {'x'}))
	gemini_fn = mocker.Mock()
	enriched, responded = score_and_summarise([{'title': 'p'}], gemini_fn)
	assert enriched == [{'title': 'x'}]
	assert responded == {'x'}
	mock_chunk.assert_called_once()


def test_score_and_summarise_multiple_batches(mocker):
	mock_chunk = mocker.patch('scorer._score_chunk', side_effect=lambda c, fn: (c, set()))
	gemini_fn = mocker.Mock()
	papers = [{'i': i} for i in range(scorer.BATCH_SIZE * 2 + 3)]
	enriched, _ = score_and_summarise(papers, gemini_fn)
	assert len(enriched) == len(papers)
	assert mock_chunk.call_count == 3
	call_sizes = [len(call.args[0]) for call in mock_chunk.call_args_list]
	assert call_sizes == [scorer.BATCH_SIZE, scorer.BATCH_SIZE, 3]


def test_score_and_summarise_exact_batch_size(mocker):
	mock_chunk = mocker.patch('scorer._score_chunk', side_effect=lambda c, fn: (c, set()))
	gemini_fn = mocker.Mock()
	papers = [{'i': i} for i in range(scorer.BATCH_SIZE)]
	score_and_summarise(papers, gemini_fn)
	assert mock_chunk.call_count == 1


def test_score_and_summarise_unions_responded_across_batches(mocker):
	# Each batch reports a different responded set. score_and_summarise should union them.
	mocker.patch('scorer._score_chunk', side_effect=[([], {'p1'}), ([], {'p2', 'p3'})])
	gemini_fn = mocker.Mock()
	papers = [{'i': i} for i in range(scorer.BATCH_SIZE + 1)]
	enriched, responded = score_and_summarise(papers, gemini_fn)
	assert enriched == []
	assert responded == {'p1', 'p2', 'p3'}


# -- gemini (HTTP) --


@responses.activate
def test_gemini_returns_response_text_on_happy_path():
	responses.post(scorer.GEMINI_URL, json={'candidates': [{'content': {'parts': [{'text': 'hello'}]}}]})
	assert gemini('prompt', 'fake-key') == 'hello'


@responses.activate
def test_gemini_strips_trailing_whitespace_from_response():
	responses.post(scorer.GEMINI_URL, json={'candidates': [{'content': {'parts': [{'text': '  hello  \n'}]}}]})
	assert gemini('prompt', 'fake-key') == 'hello'


@responses.activate
def test_gemini_retries_on_429_then_succeeds(capsys):
	responses.post(scorer.GEMINI_URL, json={}, status=429)
	responses.post(scorer.GEMINI_URL, json={'candidates': [{'content': {'parts': [{'text': 'ok'}]}}]})
	assert gemini('prompt', 'fake-key') == 'ok'
	assert 'rate limited' in capsys.readouterr().out


@responses.activate
def test_gemini_raises_after_all_retries_are_429():
	for _ in range(3):
		responses.post(scorer.GEMINI_URL, json={}, status=429)
	with pytest.raises(Exception, match='Gemini failed after retries'):
		gemini('prompt', 'fake-key')


@responses.activate
def test_gemini_raises_on_500():
	responses.post(scorer.GEMINI_URL, json={'error': 'oops'}, status=500)
	with pytest.raises(requests.HTTPError):
		gemini('prompt', 'fake-key')


@responses.activate
def test_gemini_propagates_network_error():
	responses.post(scorer.GEMINI_URL, body=requests.ConnectionError('boom'))
	with pytest.raises(requests.ConnectionError):
		gemini('prompt', 'fake-key')


def test_gemini_backoff_pattern_across_three_429s(no_sleep):
	with responses.RequestsMock() as rmock:
		for _ in range(3):
			rmock.post(scorer.GEMINI_URL, json={}, status=429)
		with pytest.raises(Exception, match='Gemini failed after retries'):
			gemini('prompt', 'fake-key')
	# Each attempt: pre-attempt sleep(5), then on 429 sleep(15 * (attempt + 1)).
	sleep_calls = [call.args[0] for call in no_sleep.call_args_list]
	assert sleep_calls == [5, 15, 5, 30, 5, 45]


@responses.activate
def test_gemini_sends_api_key_in_header_not_url():
	# Auth via header keeps the key out of any URL that may surface in HTTPError
	# messages and downstream logs (PR #17 Copilot review).
	responses.post(scorer.GEMINI_URL, json={'candidates': [{'content': {'parts': [{'text': 'ok'}]}}]})
	gemini('prompt', 'my-secret-key')
	assert responses.calls[0].request.headers['x-goog-api-key'] == 'my-secret-key'
	assert 'my-secret-key' not in responses.calls[0].request.url


# -- hardening: parse_gemini_scores --


def test_parse_strips_uppercase_json_fence():
	assert parse_gemini_scores('```JSON\n[{"index": 0}]\n```') == [{'index': 0}]


def test_parse_raises_when_result_is_a_dict():
	with pytest.raises(ValueError, match='Expected JSON array'):
		parse_gemini_scores('{"index": 0}')


def test_parse_raises_when_result_is_a_scalar():
	with pytest.raises(ValueError, match='Expected JSON array'):
		parse_gemini_scores('42')


# -- hardening: apply_scores --


@pytest.mark.parametrize('value', [None, {}, 42, 'list'])
def test_apply_scores_returns_empty_when_scores_not_a_list(value, capsys):
	assert apply_scores([{'title': 'p'}], value, threshold=6) == ([], set())
	assert 'not a list' in capsys.readouterr().out


def test_apply_scores_skips_non_dict_result(capsys):
	papers = [{'title': 'p1'}, {'title': 'p2'}]
	scores = ['not a dict', _valid_score(1, 8)]
	enriched, _ = apply_scores(papers, scores, threshold=6)
	assert [p['title'] for p in enriched] == ['p2']
	assert 'non-dict result' in capsys.readouterr().out


def test_apply_scores_skips_result_missing_index(capsys):
	papers = [{'title': 'p1'}, {'title': 'p2'}]
	scores = [{'relevance_score': 8, 'relevance_reason': 'r'}, _valid_score(1, 8)]
	enriched, _ = apply_scores(papers, scores, threshold=6)
	assert [p['title'] for p in enriched] == ['p2']
	assert 'missing or invalid index' in capsys.readouterr().out


@pytest.mark.parametrize('idx', ['0', True, 1.5, None])
def test_apply_scores_skips_result_with_non_int_index(idx, capsys):
	papers = [{'title': 'p1'}, {'title': 'p2'}]
	scores = [{'index': idx, 'relevance_score': 8, 'relevance_reason': 'r'}, _valid_score(1, 8)]
	enriched, _ = apply_scores(papers, scores, threshold=6)
	assert [p['title'] for p in enriched] == ['p2']


def test_apply_scores_drops_paper_when_relevance_reason_missing(capsys):
	papers = [{'title': 'p'}]
	scores = [{'index': 0, 'relevance_score': 8}]
	enriched, _ = apply_scores(papers, scores, threshold=6)
	assert enriched == []
	assert 'missing fields' in capsys.readouterr().out


def test_apply_scores_drops_paper_when_relevance_reason_is_non_string(capsys):
	papers = [{'title': 'p'}]
	scores = [{'index': 0, 'relevance_score': 8, 'relevance_reason': None}]
	enriched, _ = apply_scores(papers, scores, threshold=6)
	assert enriched == []
	assert 'missing fields' in capsys.readouterr().out


# -- responded set semantics --


def test_apply_scores_responded_includes_above_threshold_papers():
	papers = [{'paperId': 'p1', 'title': 'p1'}]
	_, responded = apply_scores(papers, [_valid_score(0, 8)], threshold=6)
	assert responded == {'p1'}


def test_apply_scores_responded_includes_below_threshold_papers():
	# Below-threshold papers were scored successfully; we just didn't email them. Don't retry.
	papers = [{'paperId': 'p1', 'title': 'p1'}]
	_, responded = apply_scores(papers, [_valid_score(0, 3)], threshold=6)
	assert responded == {'p1'}


def test_apply_scores_responded_includes_papers_with_missing_relevance_reason():
	# Numeric score present but relevance_reason missing; still counts as responded.
	# Gemini gave us what it could; no point retrying.
	papers = [{'paperId': 'p1', 'title': 'p1'}]
	scores = [{'index': 0, 'relevance_score': 8}]
	_, responded = apply_scores(papers, scores, threshold=6)
	assert responded == {'p1'}


def test_apply_scores_responded_excludes_papers_with_invalid_score():
	# Invalid score means Gemini gave us nothing useful for this paper. Retry next run.
	papers = [{'paperId': 'p1', 'title': 'p1'}]
	scores = [{'index': 0, 'relevance_score': 'eight', 'relevance_reason': 'r'}]
	_, responded = apply_scores(papers, scores, threshold=6)
	assert responded == set()


def test_apply_scores_responded_excludes_papers_with_no_matching_index():
	# Gemini omitted this paper from its response. Retry next run.
	papers = [{'paperId': 'p1', 'title': 'p1'}, {'paperId': 'p2', 'title': 'p2'}]
	_, responded = apply_scores(papers, [_valid_score(0, 8)], threshold=6)
	assert responded == {'p1'}


def test_apply_scores_responded_empty_when_scores_not_a_list():
	assert apply_scores([{'paperId': 'p1'}], None, threshold=6) == ([], set())


# -- hardening: gemini response shape --


@responses.activate
def test_gemini_raises_when_candidates_key_missing():
	responses.post(scorer.GEMINI_URL, json={})
	with pytest.raises(ValueError, match='missing expected fields'):
		gemini('prompt', 'fake-key')


@responses.activate
def test_gemini_raises_when_candidates_is_empty():
	responses.post(scorer.GEMINI_URL, json={'candidates': []})
	with pytest.raises(ValueError, match='missing expected fields'):
		gemini('prompt', 'fake-key')


@responses.activate
def test_gemini_raises_when_text_field_missing():
	responses.post(scorer.GEMINI_URL, json={'candidates': [{'content': {'parts': [{}]}}]})
	with pytest.raises(ValueError, match='missing expected fields'):
		gemini('prompt', 'fake-key')
