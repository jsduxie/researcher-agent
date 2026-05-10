import json
from datetime import datetime
from pathlib import Path

import pytest

from render import _build_links, _format_authors, _paper_render_data, build_email, score_colour

FIXTURES = Path(__file__).parent / 'fixtures'


# -- score_colour --


@pytest.mark.parametrize(
	'score,expected', [(10, '#059669'), (9, '#059669'), (8, '#2563eb'), (7, '#2563eb'), (6, '#64748b'), (0, '#64748b')]
)
def test_score_colour_int_bands(score, expected):
	assert score_colour(score) == expected


def test_score_colour_below_lowest_band_falls_back():
	assert score_colour(-1) == '#94a3b8'


@pytest.mark.parametrize('value', ['N/A', None, 1.5, [], {}])
def test_score_colour_non_int_falls_back(value):
	assert score_colour(value) == '#94a3b8'


def test_score_colour_bool_treated_as_int_per_current_behaviour():
	# bool subclasses int in Python; current behaviour maps True -> 1 -> grey.
	# Documented to surface this if it ever needs to change.
	assert score_colour(True) == '#64748b'


# -- _format_authors --


def test_format_authors_empty_returns_empty_string():
	assert _format_authors([]) == ''


def test_format_authors_single():
	assert _format_authors([{'name': 'Smith J.'}]) == 'Smith J.'


def test_format_authors_three_no_truncation():
	authors = [{'name': 'A'}, {'name': 'B'}, {'name': 'C'}]
	assert _format_authors(authors) == 'A, B, C'


def test_format_authors_four_truncates_with_et_al():
	authors = [{'name': 'A'}, {'name': 'B'}, {'name': 'C'}, {'name': 'D'}]
	assert _format_authors(authors) == 'A, B, C et al.'


def test_format_authors_five_truncates_with_et_al():
	authors = [{'name': n} for n in 'ABCDE']
	assert _format_authors(authors) == 'A, B, C et al.'


# -- _build_links --


def test_build_links_url_only():
	assert _build_links({'url': 'https://ss/abc'}) == [{'url': 'https://ss/abc', 'label': 'Semantic Scholar'}]


def test_build_links_doi_only():
	links = _build_links({'externalIds': {'DOI': '10.1/abc'}})
	assert links == [{'url': 'https://doi.org/10.1/abc', 'label': 'DOI'}]


def test_build_links_pdf_only():
	links = _build_links({'openAccessPdf': {'url': 'https://x/p.pdf'}})
	assert links == [{'url': 'https://x/p.pdf', 'label': 'PDF'}]


def test_build_links_all_three_in_order():
	paper = {'url': 'https://ss/abc', 'externalIds': {'DOI': '10.1/abc'}, 'openAccessPdf': {'url': 'https://x/p.pdf'}}
	labels = [link['label'] for link in _build_links(paper)]
	assert labels == ['Semantic Scholar', 'DOI', 'PDF']


def test_build_links_none_returns_empty_list():
	assert _build_links({}) == []


def test_build_links_external_ids_none_does_not_crash():
	assert _build_links({'externalIds': None}) == []


def test_build_links_open_access_pdf_none_does_not_crash():
	assert _build_links({'openAccessPdf': None}) == []


def test_build_links_doi_value_none_skipped():
	assert _build_links({'externalIds': {'DOI': None}}) == []


def test_build_links_pdf_url_empty_skipped():
	assert _build_links({'openAccessPdf': {'url': ''}}) == []


# -- _paper_render_data --


def _full_paper():
	return {
		'title': 'A',
		'authors': [{'name': 'X'}],
		'publicationDate': '2024-01-01',
		'year': 2024,
		'citationCount': 7,
		'ai_score': 8,
		'ai_summary': 'sum',
		'ai_contribution': 'con',
		'ai_reason': 'rea',
		'url': 'https://ss/a',
	}


def test_paper_render_data_full_paper_populates_all_fields():
	data = _paper_render_data(_full_paper())
	assert data['title'] == 'A'
	assert data['authors_display'] == 'X'
	assert data['pub_date'] == '2024-01-01'
	assert data['citation_count'] == 7
	assert data['score_colour_value'] == '#2563eb'
	assert data['score_display'] == 8
	assert data['ai_summary'] == 'sum'
	assert data['ai_contribution'] == 'con'
	assert data['ai_reason'] == 'rea'
	assert data['links'] == [{'url': 'https://ss/a', 'label': 'Semantic Scholar'}]


def test_paper_render_data_missing_title_falls_back():
	assert _paper_render_data({})['title'] == 'No title'


def test_paper_render_data_missing_authors_yields_empty_string():
	assert _paper_render_data({})['authors_display'] == ''


def test_paper_render_data_publication_date_falls_back_to_year():
	assert _paper_render_data({'year': 2024})['pub_date'] == '2024'


def test_paper_render_data_missing_date_and_year_returns_NA():
	assert _paper_render_data({})['pub_date'] == 'N/A'


def test_paper_render_data_missing_citation_count_defaults_to_zero():
	assert _paper_render_data({})['citation_count'] == 0


def test_paper_render_data_non_int_score_renders_question_mark():
	data = _paper_render_data({'ai_score': 'N/A'})
	assert data['score_display'] == '?'
	assert data['score_colour_value'] == '#94a3b8'


def test_paper_render_data_missing_ai_fields_default_to_empty_strings():
	data = _paper_render_data({})
	assert data['ai_summary'] == ''
	assert data['ai_contribution'] == ''
	assert data['ai_reason'] == ''


# -- build_email --


def test_build_email_renders_snapshot_byte_for_byte(mocker):
	mock_dt = mocker.MagicMock()
	mock_dt.now.return_value = datetime(2026, 1, 15)
	mocker.patch('render.datetime', mock_dt)

	with open(FIXTURES / 'enriched_papers.json') as f:
		papers = json.load(f)

	expected = (FIXTURES / 'email_snapshot.html').read_text()
	assert build_email(papers) == expected


def test_build_email_renders_empty_state_with_no_papers(mocker):
	mock_dt = mocker.MagicMock()
	mock_dt.now.return_value = datetime(2026, 1, 15)
	mocker.patch('render.datetime', mock_dt)

	html = build_email([])
	assert 'No relevant papers found this period.' in html
	assert '>0<' in html
