import json

import pytest
import requests
import responses

import config
import scorer
from gemini import generate_url
from scorer import (
	GeminiBudgetExhausted,
	GeminiQuotaExhausted,
	_score_chunk,
	apply_scores,
	parse_gemini_scores,
	score_and_summarise,
)

# Tests share the seed-derived cfg; GEMINI_URL and BATCH_SIZE come from it so `responses` mocks match what production builds.
_TEST_CFG = config.Config(**config.load_seed())
GEMINI_URL = generate_url(_TEST_CFG)
BATCH_SIZE = _TEST_CFG.batch_size


def gemini(prompt, api_key, retries=3, on_attempt=None):
	return scorer.gemini(prompt, api_key, _TEST_CFG, retries, on_attempt=on_attempt)


@pytest.fixture(autouse=True)
def no_sleep(mocker):
	# Sleeps now live in the transport module.
	return mocker.patch('gemini.time.sleep')


def _valid_score(index, score):
	return {'index': index, 'relevance_score': score, 'relevance_reason': 'r'}


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
	# True passes isinstance(_, int) in Python; refuse it explicitly so a stray boolean from a malformed response can't be promoted to a numeric score.
	papers = [{'title': 'p'}]
	scores = [{'index': 0, 'relevance_score': True, 'relevance_reason': 'r'}]
	enriched, _ = apply_scores(papers, scores, threshold=0)
	assert enriched == []
	assert 'invalid score' in capsys.readouterr().out


def test_apply_scores_handles_explicit_none_title():
	# Semantic Scholar can return title: null; slicing None must not take down the run.
	papers = [{'paperId': 'p1', 'title': None, 'abstract': 'a'}]
	enriched, responded = apply_scores(papers, [_valid_score(0, 8)], threshold=6)
	assert len(enriched) == 1
	assert responded == {'p1'}


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
	assert _score_chunk([], gemini_fn, _TEST_CFG) == ([], set())
	gemini_fn.assert_not_called()


def test_score_chunk_returns_empty_on_gemini_exception(capsys, mocker):
	gemini_fn = mocker.Mock(side_effect=Exception('boom'))
	papers = [{'paperId': 'p1', 'title': 'p', 'abstract': 'a'}]
	assert _score_chunk(papers, gemini_fn, _TEST_CFG) == ([], set())
	assert 'Batch Gemini error' in capsys.readouterr().out


def test_score_chunk_returns_empty_on_malformed_gemini_response(capsys, mocker):
	gemini_fn = mocker.Mock(return_value='not json')
	papers = [{'paperId': 'p1', 'title': 'p', 'abstract': 'a'}]
	assert _score_chunk(papers, gemini_fn, _TEST_CFG) == ([], set())
	assert 'Batch Gemini error' in capsys.readouterr().out


def test_score_chunk_replaces_missing_abstract_in_prompt():
	captured = {}

	def fake_gemini(prompt):
		captured['prompt'] = prompt
		return json.dumps([_valid_score(0, 8)])

	_score_chunk([{'title': 'p'}], fake_gemini, _TEST_CFG)
	assert 'No abstract available.' in captured['prompt']


def test_score_chunk_returns_enriched_papers_on_happy_path(mocker):
	gemini_fn = mocker.Mock(return_value=json.dumps([_valid_score(0, 8)]))
	papers = [{'paperId': 'p1', 'title': 'p', 'abstract': 'a'}]
	enriched, responded = _score_chunk(papers, gemini_fn, _TEST_CFG)
	assert len(enriched) == 1
	assert enriched[0]['ai_score'] == 8
	assert responded == {'p1'}


# -- score_and_summarise --


def test_score_and_summarise_empty_input(mocker):
	gemini_fn = mocker.Mock()
	assert score_and_summarise([], gemini_fn, _TEST_CFG) == ([], set(), set())
	gemini_fn.assert_not_called()


def test_score_and_summarise_single_small_batch(mocker):
	mock_chunk = mocker.patch('scorer._score_chunk', return_value=([{'title': 'x'}], {'x'}))
	gemini_fn = mocker.Mock()
	enriched, responded, _ = score_and_summarise([{'title': 'p'}], gemini_fn, _TEST_CFG)
	assert enriched == [{'title': 'x'}]
	assert responded == {'x'}
	mock_chunk.assert_called_once()


def test_score_and_summarise_multiple_batches(mocker):
	mock_chunk = mocker.patch('scorer._score_chunk', side_effect=lambda c, fn, cfg: (c, set()))
	gemini_fn = mocker.Mock()
	papers = [{'i': i} for i in range(BATCH_SIZE * 2 + 3)]
	enriched, _, _ = score_and_summarise(papers, gemini_fn, _TEST_CFG)
	assert len(enriched) == len(papers)
	assert mock_chunk.call_count == 3
	call_sizes = [len(call.args[0]) for call in mock_chunk.call_args_list]
	assert call_sizes == [BATCH_SIZE, BATCH_SIZE, 3]


def test_score_and_summarise_exact_batch_size(mocker):
	mock_chunk = mocker.patch('scorer._score_chunk', side_effect=lambda c, fn, cfg: (c, set()))
	gemini_fn = mocker.Mock()
	papers = [{'i': i} for i in range(BATCH_SIZE)]
	score_and_summarise(papers, gemini_fn, _TEST_CFG)
	assert mock_chunk.call_count == 1


def test_score_and_summarise_unions_responded_across_batches(mocker):
	# Each batch reports a different responded set. score_and_summarise should union them.
	mocker.patch('scorer._score_chunk', side_effect=[([], {'p1'}), ([], {'p2', 'p3'})])
	gemini_fn = mocker.Mock()
	papers = [{'i': i} for i in range(BATCH_SIZE + 1)]
	enriched, responded, _ = score_and_summarise(papers, gemini_fn, _TEST_CFG)
	assert enriched == []
	assert responded == {'p1', 'p2', 'p3'}


# -- second pass for unresponded papers --


def test_score_and_summarise_rescores_failed_batch_in_second_pass(mocker, capsys):
	# Batch fails outright on pass one (e.g. malformed payload); the second pass re-batches every paper and its results count.
	papers = [{'paperId': 'p1'}, {'paperId': 'p2'}]
	scored = [dict(p, ai_score=8) for p in papers]
	mock_chunk = mocker.patch('scorer._score_chunk', side_effect=[([], set()), (scored, {'p1', 'p2'})])
	enriched, responded, _ = score_and_summarise(papers, mocker.Mock(), _TEST_CFG)
	assert mock_chunk.call_count == 2
	assert mock_chunk.call_args_list[1].args[0] == papers
	assert enriched == scored
	assert responded == {'p1', 'p2'}
	assert 'second pass' in capsys.readouterr().out


def test_score_and_summarise_second_pass_contains_only_unresponded_papers(mocker):
	# Gemini answered p1 but omitted p2; only p2 goes back.
	papers = [{'paperId': 'p1'}, {'paperId': 'p2'}]
	mock_chunk = mocker.patch('scorer._score_chunk', side_effect=[([], {'p1'}), ([], {'p2'})])
	score_and_summarise(papers, mocker.Mock(), _TEST_CFG)
	assert mock_chunk.call_args_list[1].args[0] == [{'paperId': 'p2'}]


def test_score_and_summarise_skips_second_pass_when_all_responded(mocker):
	papers = [{'paperId': 'p1'}]
	mock_chunk = mocker.patch('scorer._score_chunk', return_value=([], {'p1'}))
	score_and_summarise(papers, mocker.Mock(), _TEST_CFG)
	mock_chunk.assert_called_once()


def test_score_and_summarise_excludes_papers_without_paper_id_from_second_pass(mocker):
	# A paper with no paperId can never appear in responded; retrying it could only duplicate an enrichment.
	papers = [{'title': 'no id'}]
	mock_chunk = mocker.patch('scorer._score_chunk', return_value=([], set()))
	score_and_summarise(papers, mocker.Mock(), _TEST_CFG)
	mock_chunk.assert_called_once()


def test_score_and_summarise_second_pass_halt_preserves_first_pass_results(mocker, capsys):
	# Quota dies during the recovery pass; everything already scored must survive.
	papers = [{'paperId': 'p1'}, {'paperId': 'p2'}]
	first = [{'paperId': 'p1', 'ai_score': 9}]
	mock_chunk = mocker.patch('scorer._score_chunk', side_effect=[(first, {'p1'}), GeminiQuotaExhausted('quota')])
	enriched, responded, _ = score_and_summarise(papers, mocker.Mock(), _TEST_CFG)
	assert mock_chunk.call_count == 2
	assert enriched == first
	assert responded == {'p1'}
	assert 'Quota exhausted' in capsys.readouterr().out


def test_score_and_summarise_runs_exactly_one_recovery_pass(mocker):
	# Papers that stay unresponded after the second pass wait for the next run; no third pass.
	papers = [{'paperId': 'p1'}]
	mock_chunk = mocker.patch('scorer._score_chunk', return_value=([], set()))
	enriched, responded, _ = score_and_summarise(papers, mocker.Mock(), _TEST_CFG)
	assert mock_chunk.call_count == 2
	assert (enriched, responded) == ([], set())


# -- attempted set semantics --


def test_score_and_summarise_attempted_includes_unresponded_papers_in_sent_batches(mocker):
	# Sent but unanswered still consumed a real Gemini call; it must count towards score_attempts.
	papers = [{'paperId': 'p1'}]
	mocker.patch('scorer._score_chunk', return_value=([], set()))
	_, _, attempted = score_and_summarise(papers, mocker.Mock(), _TEST_CFG)
	assert attempted == {'p1'}


def test_score_and_summarise_attempted_excludes_batches_after_halt(mocker):
	# Papers in batches the halt prevented from being sent must stay eligible for future sweeps.
	papers = [{'paperId': f'p{i}'} for i in range(BATCH_SIZE + 2)]
	mocker.patch(
		'scorer._score_chunk', side_effect=[([], {f'p{i}' for i in range(BATCH_SIZE)}), GeminiQuotaExhausted('quota')]
	)
	_, _, attempted = score_and_summarise(papers, mocker.Mock(), _TEST_CFG)
	assert attempted == {f'p{i}' for i in range(BATCH_SIZE)}


def test_score_and_summarise_attempted_empty_when_first_batch_halts(mocker):
	papers = [{'paperId': 'p1'}]
	mocker.patch('scorer._score_chunk', side_effect=GeminiBudgetExhausted('budget'))
	_, _, attempted = score_and_summarise(papers, mocker.Mock(), _TEST_CFG)
	assert attempted == set()


# -- gemini (HTTP) --


@responses.activate
def test_gemini_returns_response_text_on_happy_path():
	responses.post(GEMINI_URL, json={'candidates': [{'content': {'parts': [{'text': 'hello'}]}}]})
	assert gemini('prompt', 'fake-key') == 'hello'


@responses.activate
def test_gemini_strips_trailing_whitespace_from_response():
	responses.post(GEMINI_URL, json={'candidates': [{'content': {'parts': [{'text': '  hello  \n'}]}}]})
	assert gemini('prompt', 'fake-key') == 'hello'


@responses.activate
def test_gemini_raises_after_all_retries_are_429():
	for _ in range(3):
		responses.post(GEMINI_URL, json={}, status=429)
	with pytest.raises(Exception, match='Gemini failed after retries'):
		gemini('prompt', 'fake-key')


@responses.activate
def test_gemini_names_last_status_after_exhausting_429_retries():
	responses.post(GEMINI_URL, json={}, status=429)
	with pytest.raises(Exception, match='last status 429'):
		gemini('prompt', 'fake-key')


@responses.activate
def test_gemini_raises_http_error_on_non_retriable_status():
	# 4xx other than 429 is a permanent request failure; it must surface immediately, not burn retries.
	responses.post(GEMINI_URL, json={'error': 'forbidden'}, status=403)
	with pytest.raises(requests.HTTPError):
		gemini('prompt', 'fake-key')
	assert len(responses.calls) == 1


@responses.activate
def test_gemini_propagates_network_error():
	responses.post(GEMINI_URL, body=requests.ConnectionError('boom'))
	with pytest.raises(requests.ConnectionError):
		gemini('prompt', 'fake-key')


def test_gemini_backoff_pattern_across_three_429s(no_sleep):
	with responses.RequestsMock() as rmock:
		for _ in range(3):
			rmock.post(GEMINI_URL, json={}, status=429)
		with pytest.raises(Exception, match='Gemini failed after retries'):
			gemini('prompt', 'fake-key')
	# Each attempt: pre-attempt sleep(5), then 15s-step backoff between attempts; no sleep after the final failure.
	sleep_calls = [call.args[0] for call in no_sleep.call_args_list]
	assert sleep_calls == [5, 15, 5, 30, 5]


@responses.activate
def test_gemini_sends_api_key_in_header_not_url():
	# Auth via header keeps the key out of any URL that may surface in HTTPError messages and downstream logs.
	responses.post(GEMINI_URL, json={'candidates': [{'content': {'parts': [{'text': 'ok'}]}}]})
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
	# Numeric score present but relevance_reason missing; still counts as responded since Gemini gave us what it could, no point retrying.
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
	responses.post(GEMINI_URL, json={})
	with pytest.raises(ValueError, match='missing expected fields'):
		gemini('prompt', 'fake-key')


@responses.activate
def test_gemini_raises_when_candidates_is_empty():
	responses.post(GEMINI_URL, json={'candidates': []})
	with pytest.raises(ValueError, match='missing expected fields'):
		gemini('prompt', 'fake-key')


@responses.activate
def test_gemini_raises_when_text_field_missing():
	responses.post(GEMINI_URL, json={'candidates': [{'content': {'parts': [{}]}}]})
	with pytest.raises(ValueError, match='missing expected fields'):
		gemini('prompt', 'fake-key')


# -- quota exhaustion (daily PerDay quota) --


@responses.activate
def test_gemini_raises_quota_exhausted_on_per_day_429():
	responses.post(GEMINI_URL, json=_PER_DAY_429, status=429)
	with pytest.raises(GeminiQuotaExhausted):
		gemini('prompt', 'fake-key')


@responses.activate
@responses.activate
@responses.activate
@responses.activate
def test_score_chunk_propagates_quota_exhausted_without_swallowing(mocker):
	# _score_chunk wraps gemini_fn in try/except; it must re-raise quota exhaustion rather than treat it as a regular per-batch error and return empty.
	gemini_fn = mocker.Mock(side_effect=GeminiQuotaExhausted('quota'))
	papers = [{'paperId': 'p1', 'title': 'p', 'abstract': 'a'}]
	with pytest.raises(GeminiQuotaExhausted, match='quota'):
		_score_chunk(papers, gemini_fn, _TEST_CFG)


def test_score_and_summarise_short_circuits_on_quota_exhausted(mocker, capsys):
	# Batch 2 of 3 raises quota exhausted; batch 3 must not be attempted, batch 1 results retained.
	mock_chunk = mocker.patch(
		'scorer._score_chunk',
		side_effect=[
			([{'paperId': 'p1', 'ai_score': 8}], {'p1'}),
			GeminiQuotaExhausted('quota'),
			([{'paperId': 'p3', 'ai_score': 7}], {'p3'}),
		],
	)
	gemini_fn = mocker.Mock()
	papers = [{'paperId': f'p{i}'} for i in range(BATCH_SIZE * 3)]
	enriched, responded, _ = score_and_summarise(papers, gemini_fn, _TEST_CFG)
	assert mock_chunk.call_count == 2
	assert enriched == [{'paperId': 'p1', 'ai_score': 8}]
	assert responded == {'p1'}
	assert 'Quota exhausted' in capsys.readouterr().out


# -- on_attempt callback (per-attempt counting) --


@responses.activate
@responses.activate
def test_gemini_fires_on_attempt_once_on_immediate_success():
	responses.post(GEMINI_URL, json={'candidates': [{'content': {'parts': [{'text': 'ok'}]}}]})
	counter = []
	gemini('prompt', 'fake-key', on_attempt=lambda: counter.append(1))
	assert len(counter) == 1


@responses.activate
# -- budget exhaustion (caller-owned, raised through on_attempt) --


def test_score_chunk_propagates_budget_exhausted_without_swallowing(mocker):
	# _score_chunk wraps gemini_fn in a generic Exception catch; it must re-raise budget exhaustion (a halt signal) rather than treat it as a per-batch error.
	gemini_fn = mocker.Mock(side_effect=GeminiBudgetExhausted('budget'))
	papers = [{'paperId': 'p1', 'title': 'p', 'abstract': 'a'}]
	with pytest.raises(GeminiBudgetExhausted, match='budget'):
		_score_chunk(papers, gemini_fn, _TEST_CFG)


def test_score_and_summarise_short_circuits_on_budget_exhausted(mocker, capsys):
	# Batch 2 of 3 raises budget exhausted; batch 3 must not be attempted, batch 1 results retained.
	mock_chunk = mocker.patch(
		'scorer._score_chunk',
		side_effect=[
			([{'paperId': 'p1', 'ai_score': 8}], {'p1'}),
			GeminiBudgetExhausted('budget'),
			([{'paperId': 'p3', 'ai_score': 7}], {'p3'}),
		],
	)
	gemini_fn = mocker.Mock()
	papers = [{'paperId': f'p{i}'} for i in range(BATCH_SIZE * 3)]
	enriched, responded, _ = score_and_summarise(papers, gemini_fn, _TEST_CFG)
	assert mock_chunk.call_count == 2
	assert enriched == [{'paperId': 'p1', 'ai_score': 8}]
	assert responded == {'p1'}
	assert 'Budget exhausted' in capsys.readouterr().out
