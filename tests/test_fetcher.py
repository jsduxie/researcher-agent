from datetime import datetime
from urllib.parse import parse_qs, urlsplit

import pytest
import requests
import responses

import fetcher
from fetcher import _cutoff_year, dedup_papers, fetch_papers


@pytest.fixture(autouse=True)
def _no_sleep(mocker):
	mocker.patch('fetcher.time.sleep')


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
	# Regression for namespaced dedup keys: paperId 'abc' must not collide
	# with a paper whose only identifier is title='abc'.
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
	assert fetch_papers('attention') == [{'paperId': 'a'}, {'paperId': 'b'}]


@responses.activate
def test_fetch_papers_returns_empty_list_when_data_is_empty():
	responses.get(fetcher.SEMANTIC_SCHOLAR_URL, json={'data': []})
	assert fetch_papers('query') == []


@responses.activate
def test_fetch_papers_returns_empty_list_when_data_key_missing():
	responses.get(fetcher.SEMANTIC_SCHOLAR_URL, json={})
	assert fetch_papers('query') == []


@responses.activate
def test_fetch_papers_returns_empty_list_on_http_500(capsys):
	responses.get(fetcher.SEMANTIC_SCHOLAR_URL, json={'error': 'oops'}, status=500)
	assert fetch_papers('query') == []
	assert 'Error fetching' in capsys.readouterr().out


@responses.activate
def test_fetch_papers_returns_empty_list_on_connection_error(capsys):
	responses.get(fetcher.SEMANTIC_SCHOLAR_URL, body=requests.ConnectionError('boom'))
	assert fetch_papers('query') == []
	assert 'Error fetching' in capsys.readouterr().out


@responses.activate
def test_fetch_papers_returns_empty_list_on_timeout(capsys):
	responses.get(fetcher.SEMANTIC_SCHOLAR_URL, body=requests.Timeout('slow'))
	assert fetch_papers('query') == []
	assert 'Error fetching' in capsys.readouterr().out


@responses.activate
def test_fetch_papers_forwards_query_param():
	responses.get(fetcher.SEMANTIC_SCHOLAR_URL, json={'data': []})
	fetch_papers('attention is all you need')
	qs = parse_qs(urlsplit(responses.calls[0].request.url).query)
	assert qs['query'] == ['attention is all you need']


@responses.activate
def test_fetch_papers_sends_limit_from_config():
	responses.get(fetcher.SEMANTIC_SCHOLAR_URL, json={'data': []})
	fetch_papers('q')
	qs = parse_qs(urlsplit(responses.calls[0].request.url).query)
	assert qs['limit'] == [str(fetcher.MAX_PER_QUERY)]


@responses.activate
def test_fetch_papers_publication_date_param_is_year_shaped():
	responses.get(fetcher.SEMANTIC_SCHOLAR_URL, json={'data': []})
	fetch_papers('q')
	qs = parse_qs(urlsplit(responses.calls[0].request.url).query)
	value = qs['publicationDateOrYear'][0]
	assert value.endswith('-')
	year_part = value[:-1]
	assert year_part.isdigit() and len(year_part) == 4


@responses.activate
def test_fetch_papers_sends_expected_fields():
	responses.get(fetcher.SEMANTIC_SCHOLAR_URL, json={'data': []})
	fetch_papers('q')
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
