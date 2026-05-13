from datetime import datetime
from urllib.parse import parse_qs, urlsplit

import pytest
import requests
import responses

import fetcher
from fetcher import FetchError, _cutoff_year, dedup_papers, fetch_papers


@pytest.fixture(autouse=True)
def _no_sleep(mocker):
	return mocker.patch('fetcher.time.sleep')


# -- dedup_papers --


def test_dedup_empty_list_returns_empty_list():
	assert dedup_papers([]) == []


def test_dedup_single_paper_passes_through():
	papers = [{'paperId': 'a', 'title': 't'}]
	assert dedup_papers(papers) == papers


def test_dedup_collapses_duplicate_paper_ids_keeps_first():
	papers = [{'paperId': 'a', 'title': 'first'}, {'paperId': 'a', 'title': 'second'}]
	assert dedup_papers(papers) == [{'paperId': 'a', 'title': 'first'}]


def test_dedup_keeps_distinct_paper_ids_in_order():
	papers = [{'paperId': 'b'}, {'paperId': 'a'}, {'paperId': 'c'}]
	assert dedup_papers(papers) == papers


def test_dedup_falls_back_to_title_when_paper_id_missing():
	papers = [{'title': 'shared'}, {'title': 'shared'}, {'title': 'unique'}]
	unique = dedup_papers(papers)
	assert [p['title'] for p in unique] == ['shared', 'unique']


def test_dedup_keeps_papers_with_distinct_titles_when_no_ids():
	papers = [{'title': 'one'}, {'title': 'two'}]
	assert dedup_papers(papers) == papers


def test_dedup_collapses_papers_with_neither_id_nor_title():
	papers = [{}, {}, {'title': 'real'}]
	unique = dedup_papers(papers)
	assert len(unique) == 2


def test_dedup_does_not_collide_paper_id_with_matching_title():
	# Regression for namespaced dedup keys: paperId 'abc' must not collide with a paper whose only identifier is title='abc'.
	papers = [{'paperId': 'abc'}, {'title': 'abc'}]
	assert dedup_papers(papers) == papers


# -- _cutoff_year --


@pytest.mark.parametrize(
	'today,days_back,expected',
	[
		(datetime(2026, 5, 10), 365, 2025),
		(datetime(2026, 5, 10), 30, 2026),
		(datetime(2026, 1, 5), 365, 2025),
		(datetime(2026, 1, 5), 10, 2025),
		(datetime(2026, 12, 31), 0, 2026),
	],
)
def test_cutoff_year(today, days_back, expected):
	assert _cutoff_year(today, days_back) == expected


# -- fetch_papers --


@responses.activate
def test_fetch_papers_returns_data_list_on_happy_path():
	responses.get(fetcher.SEMANTIC_SCHOLAR_URL, json={'data': [{'paperId': 'a'}, {'paperId': 'b'}]})
	assert fetch_papers('attention', 'test-key') == [{'paperId': 'a'}, {'paperId': 'b'}]


@responses.activate
def test_fetch_papers_returns_empty_list_when_data_is_empty():
	responses.get(fetcher.SEMANTIC_SCHOLAR_URL, json={'data': []})
	assert fetch_papers('query', 'test-key') == []


@responses.activate
def test_fetch_papers_returns_empty_list_when_data_key_missing():
	responses.get(fetcher.SEMANTIC_SCHOLAR_URL, json={})
	assert fetch_papers('query', 'test-key') == []


@responses.activate
def test_fetch_papers_raises_fetch_error_on_invalid_json_body():
	# 2xx with malformed body indicates an upstream contract change, not a transient; retrying would just resend the same broken response.
	responses.get(fetcher.SEMANTIC_SCHOLAR_URL, body='<html>not json</html>', status=200)
	with pytest.raises(FetchError, match='not valid JSON'):
		fetch_papers('q', 'test-key')
	assert len(responses.calls) == 1


@responses.activate
def test_fetch_papers_raises_when_500_persists():
	for _ in range(4):
		responses.get(fetcher.SEMANTIC_SCHOLAR_URL, json={'error': 'oops'}, status=500)
	with pytest.raises(FetchError):
		fetch_papers('query', 'test-key')
	assert len(responses.calls) == 4


@responses.activate
def test_fetch_papers_raises_when_connection_error_persists():
	for _ in range(4):
		responses.get(fetcher.SEMANTIC_SCHOLAR_URL, body=requests.ConnectionError('boom'))
	with pytest.raises(FetchError):
		fetch_papers('query', 'test-key')
	assert len(responses.calls) == 4


@responses.activate
def test_fetch_papers_raises_when_timeout_persists():
	for _ in range(4):
		responses.get(fetcher.SEMANTIC_SCHOLAR_URL, body=requests.Timeout('slow'))
	with pytest.raises(FetchError):
		fetch_papers('query', 'test-key')
	assert len(responses.calls) == 4


# -- retry on 429 / 5xx --


@responses.activate
def test_fetch_papers_retries_on_429_then_succeeds():
	responses.get(fetcher.SEMANTIC_SCHOLAR_URL, status=429)
	responses.get(fetcher.SEMANTIC_SCHOLAR_URL, json={'data': [{'paperId': 'a'}]})
	assert fetch_papers('q', 'test-key') == [{'paperId': 'a'}]
	assert len(responses.calls) == 2


@responses.activate
def test_fetch_papers_retries_on_500_then_succeeds():
	responses.get(fetcher.SEMANTIC_SCHOLAR_URL, status=500)
	responses.get(fetcher.SEMANTIC_SCHOLAR_URL, json={'data': [{'paperId': 'a'}]})
	assert fetch_papers('q', 'test-key') == [{'paperId': 'a'}]
	assert len(responses.calls) == 2


@responses.activate
def test_fetch_papers_honours_retry_after_seconds(_no_sleep):
	responses.get(fetcher.SEMANTIC_SCHOLAR_URL, status=429, headers={'Retry-After': '7'})
	responses.get(fetcher.SEMANTIC_SCHOLAR_URL, json={'data': []})
	fetch_papers('q', 'test-key')
	# 2s pre-call sleep, then 7s honoured from Retry-After instead of the 5s default.
	delays = [c.args[0] for c in _no_sleep.call_args_list]
	assert delays == [2, 7]


@responses.activate
def test_fetch_papers_ignores_non_integer_retry_after(_no_sleep):
	# HTTP-date form is unsupported and falls back to the default backoff schedule.
	responses.get(fetcher.SEMANTIC_SCHOLAR_URL, status=429, headers={'Retry-After': 'Wed, 21 Oct 2026 07:28:00 GMT'})
	responses.get(fetcher.SEMANTIC_SCHOLAR_URL, json={'data': []})
	fetch_papers('q', 'test-key')
	delays = [c.args[0] for c in _no_sleep.call_args_list]
	assert delays == [2, 5]


@responses.activate
def test_fetch_papers_raises_when_retries_exhausted():
	for _ in range(4):
		responses.get(fetcher.SEMANTIC_SCHOLAR_URL, status=429)
	with pytest.raises(FetchError):
		fetch_papers('q', 'test-key')
	assert len(responses.calls) == 4


@responses.activate
def test_fetch_papers_does_not_retry_on_404():
	responses.get(fetcher.SEMANTIC_SCHOLAR_URL, status=404)
	with pytest.raises(FetchError):
		fetch_papers('q', 'test-key')
	assert len(responses.calls) == 1


@responses.activate
def test_fetch_papers_backoff_uses_expected_delays(_no_sleep):
	for _ in range(4):
		responses.get(fetcher.SEMANTIC_SCHOLAR_URL, status=429)
	with pytest.raises(FetchError):
		fetch_papers('q', 'test-key')
	delays = [c.args[0] for c in _no_sleep.call_args_list]
	assert delays == [2, 5, 15, 45]


# -- api key header --


@responses.activate
def test_fetch_papers_sends_x_api_key_header():
	responses.get(fetcher.SEMANTIC_SCHOLAR_URL, json={'data': []})
	fetch_papers('q', 'secret-key')
	assert responses.calls[0].request.headers['x-api-key'] == 'secret-key'


@responses.activate
def test_fetch_papers_sends_x_api_key_header_on_every_retry():
	responses.get(fetcher.SEMANTIC_SCHOLAR_URL, status=429)
	responses.get(fetcher.SEMANTIC_SCHOLAR_URL, status=429)
	responses.get(fetcher.SEMANTIC_SCHOLAR_URL, json={'data': []})
	fetch_papers('q', 'secret-key')
	for call in responses.calls:
		assert call.request.headers['x-api-key'] == 'secret-key'


@responses.activate
def test_fetch_papers_forwards_query_param():
	responses.get(fetcher.SEMANTIC_SCHOLAR_URL, json={'data': []})
	fetch_papers('attention is all you need', 'test-key')
	qs = parse_qs(urlsplit(responses.calls[0].request.url).query)
	assert qs['query'] == ['attention is all you need']


@responses.activate
def test_fetch_papers_sends_limit_from_config():
	responses.get(fetcher.SEMANTIC_SCHOLAR_URL, json={'data': []})
	fetch_papers('q', 'test-key')
	qs = parse_qs(urlsplit(responses.calls[0].request.url).query)
	assert qs['limit'] == [str(fetcher.MAX_PER_QUERY)]


@responses.activate
def test_fetch_papers_publication_date_param_is_year_shaped():
	responses.get(fetcher.SEMANTIC_SCHOLAR_URL, json={'data': []})
	fetch_papers('q', 'test-key')
	qs = parse_qs(urlsplit(responses.calls[0].request.url).query)
	value = qs['publicationDateOrYear'][0]
	assert value.endswith('-')
	year_part = value[:-1]
	assert year_part.isdigit() and len(year_part) == 4


@responses.activate
def test_fetch_papers_sends_expected_fields():
	responses.get(fetcher.SEMANTIC_SCHOLAR_URL, json={'data': []})
	fetch_papers('q', 'test-key')
	qs = parse_qs(urlsplit(responses.calls[0].request.url).query)
	expected = {
		'title',
		'abstract',
		'authors',
		'year',
		'citationCount',
		'externalIds',
		'openAccessPdf',
		'url',
		'publicationDate',
	}
	assert set(qs['fields'][0].split(',')) == expected
