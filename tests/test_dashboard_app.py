from datetime import datetime
from pathlib import Path

import pytest
import yaml
from streamlit.testing.v1 import AppTest

_APP_PATH = 'streamlit_app.py'
_COPY = yaml.safe_load((Path(__file__).resolve().parents[1] / 'copy.yaml').read_text())


@pytest.fixture(autouse=True)
def _stub_env(monkeypatch):
	monkeypatch.setenv('DATABASE_URL', 'postgresql://fake')


@pytest.fixture(autouse=True)
def _clear_streamlit_caches():
	# st.cache_resource persists across AppTest runs in the same process and would leak a mocked connection from a prior test; reset before and after.
	import streamlit as st

	st.cache_resource.clear()
	yield
	st.cache_resource.clear()


@pytest.fixture
def stub_db(mocker):
	# Patch db.connect so the cached resource is a benign mock; nothing else in db.py is touched at import time, so call sites can be patched per test.
	mocker.patch('db.connect', return_value=mocker.MagicMock())
	return {
		'search': mocker.patch('db.search_papers', return_value=[]),
		'count': mocker.patch('db.count_papers', return_value=0),
	}


def _make_paper(**overrides):
	base = {
		'paper_id': 'p1',
		'title': 'Attention Is All You Need',
		'abstract': 'An abstract',
		'year': 2017,
		'citation_count': 87432,
		'url': 'https://example/p1',
		'doi': '10.1/p1',
		'pdf_url': 'https://example/p1.pdf',
		'fetched_at': datetime(2026, 5, 1),
		'authors': ['Vaswani', 'Shazeer', 'Parmar'],
		'methodology': 'multi-head self-attention replaces recurrence',
		'findings': 'beats RNN/CNN on WMT-14',
		'relevance': 'foundational transformer paper',
		'limitations': 'no clinical evaluation',
		'latest_rating': 4,
	}
	base.update(overrides)
	return base


# -- top-level rendering --


def test_app_renders_brand_title(stub_db):
	at = AppTest.from_file(_APP_PATH).run()
	assert any(_COPY['brand'] in m.value for m in at.markdown)


def test_app_uses_dark_palette_in_css(stub_db):
	# The injected CSS carries the theme tokens; verifies the palette is the signed-off one rather than Streamlit's default light theme bleeding in.
	at = AppTest.from_file(_APP_PATH).run()
	css = '\n'.join(m.value for m in at.markdown)
	assert '#131110' in css  # background (biome lowercases hex)
	assert '#d99565' in css  # accent
	assert 'Space Grotesk' in css
	assert 'JetBrains Mono' in css


def test_app_calls_search_with_default_pagination_on_first_load(stub_db):
	AppTest.from_file(_APP_PATH).run()
	call = stub_db['search'].call_args
	assert call.kwargs['limit'] == 20
	assert call.kwargs['offset'] == 0
	assert call.kwargs['q'] is None


def test_app_shows_empty_state_when_no_papers(stub_db):
	at = AppTest.from_file(_APP_PATH).run()
	assert any(_COPY['empty_state'] in m.value for m in at.markdown)


# -- paper row contents --


def test_app_renders_paper_title_as_button(stub_db):
	stub_db['search'].return_value = [_make_paper()]
	stub_db['count'].return_value = 1
	at = AppTest.from_file(_APP_PATH).run()
	titles = [b.label for b in at.button]
	assert 'Attention Is All You Need' in titles


def test_app_renders_row_index_padded_to_two_digits(stub_db):
	stub_db['search'].return_value = [_make_paper()]
	stub_db['count'].return_value = 1
	at = AppTest.from_file(_APP_PATH).run()
	indices = [m.value for m in at.markdown if 'class="row-index"' in m.value]
	assert any('01' in m for m in indices)


def test_app_formats_authors_with_etal_when_more_than_three(stub_db):
	stub_db['search'].return_value = [_make_paper(authors=['A', 'B', 'C', 'D', 'E'])]
	stub_db['count'].return_value = 1
	at = AppTest.from_file(_APP_PATH).run()
	row_meta = [m.value for m in at.markdown if 'class="row-meta"' in m.value]
	assert any(_COPY['authors']['etal_suffix'].strip() in m for m in row_meta)


def test_app_formats_authors_inline_when_three_or_fewer(stub_db):
	stub_db['search'].return_value = [_make_paper(authors=['Vaswani', 'Shazeer', 'Parmar'])]
	stub_db['count'].return_value = 1
	at = AppTest.from_file(_APP_PATH).run()
	row_meta = [m.value for m in at.markdown if 'class="row-meta"' in m.value]
	assert any('Vaswani, Shazeer, Parmar' in m for m in row_meta)
	assert not any(_COPY['authors']['etal_suffix'].strip() in m for m in row_meta)


def test_app_renders_unrated_glyph_when_no_rating(stub_db):
	stub_db['search'].return_value = [_make_paper(latest_rating=None)]
	stub_db['count'].return_value = 1
	at = AppTest.from_file(_APP_PATH).run()
	row_rating = [m.value for m in at.markdown if 'class="row-rating"' in m.value]
	assert any(_COPY['icons']['rating_unrated'] in m for m in row_rating)


def test_app_renders_star_rating_when_rated(stub_db):
	stub_db['search'].return_value = [_make_paper(latest_rating=4)]
	stub_db['count'].return_value = 1
	at = AppTest.from_file(_APP_PATH).run()
	row_rating = [m.value for m in at.markdown if 'class="row-rating"' in m.value]
	expected = f'4 {_COPY["icons"]["rating_star"]}'
	assert any(expected in m for m in row_rating)


def test_app_renders_date_unknown_glyph_when_no_fetched_at(stub_db):
	stub_db['search'].return_value = [_make_paper(fetched_at=None)]
	stub_db['count'].return_value = 1
	at = AppTest.from_file(_APP_PATH).run()
	row_meta = [m.value for m in at.markdown if 'class="row-meta"' in m.value]
	assert any(_COPY['icons']['date_unknown'] in m for m in row_meta)


# -- modal on click --


def test_app_does_not_render_summary_sections_before_click(stub_db):
	stub_db['search'].return_value = [_make_paper()]
	stub_db['count'].return_value = 1
	at = AppTest.from_file(_APP_PATH).run()
	section_labels = [m.value for m in at.markdown if 'class="section-label"' in m.value]
	assert section_labels == []


def test_app_renders_summary_sections_in_modal_after_click(stub_db):
	stub_db['search'].return_value = [_make_paper()]
	stub_db['count'].return_value = 1
	at = AppTest.from_file(_APP_PATH).run()
	(title_button,) = [b for b in at.button if b.label == 'Attention Is All You Need']
	title_button.click().run()
	section_labels = [m.value for m in at.markdown if 'class="section-label"' in m.value]
	assert any('METHODOLOGY' in m for m in section_labels)
	assert any('FINDINGS' in m for m in section_labels)
	assert any('RELEVANCE' in m for m in section_labels)
	assert any('LIMITATIONS' in m for m in section_labels)


def test_modal_renders_paper_title_and_authors(stub_db):
	stub_db['search'].return_value = [_make_paper()]
	stub_db['count'].return_value = 1
	at = AppTest.from_file(_APP_PATH).run()
	(title_button,) = [b for b in at.button if b.label == 'Attention Is All You Need']
	title_button.click().run()
	modal_titles = [m.value for m in at.markdown if 'class="modal-title"' in m.value]
	modal_authors = [m.value for m in at.markdown if 'class="modal-authors"' in m.value]
	assert any('Attention Is All You Need' in m for m in modal_titles)
	assert any('Vaswani' in m for m in modal_authors)


def test_modal_renders_meta_with_year_citations_and_rating(stub_db):
	stub_db['search'].return_value = [_make_paper(year=2017, citation_count=87432, latest_rating=4)]
	stub_db['count'].return_value = 1
	at = AppTest.from_file(_APP_PATH).run()
	(title_button,) = [b for b in at.button if b.label == 'Attention Is All You Need']
	title_button.click().run()
	modal_meta = [m.value for m in at.markdown if 'class="modal-meta"' in m.value]
	assert any('2017' in m for m in modal_meta)
	assert any('87,432' in m for m in modal_meta)  # thousands-formatted
	assert any(f'4 {_COPY["icons"]["rating_star"]}' in m for m in modal_meta)


def test_modal_meta_omits_rating_when_unrated(stub_db):
	stub_db['search'].return_value = [_make_paper(latest_rating=None)]
	stub_db['count'].return_value = 1
	at = AppTest.from_file(_APP_PATH).run()
	(title_button,) = [b for b in at.button if b.label == 'Attention Is All You Need']
	title_button.click().run()
	modal_meta = [m.value for m in at.markdown if 'class="modal-meta"' in m.value]
	assert not any(_COPY['icons']['rating_star'] in m for m in modal_meta)


def test_modal_renders_paper_and_pdf_links_as_anchor_tags(stub_db):
	stub_db['search'].return_value = [_make_paper()]
	stub_db['count'].return_value = 1
	at = AppTest.from_file(_APP_PATH).run()
	(title_button,) = [b for b in at.button if b.label == 'Attention Is All You Need']
	title_button.click().run()
	link_blocks = [m.value for m in at.markdown if 'class="row-links"' in m.value]
	assert any('<a href="https://example/p1"' in m for m in link_blocks)
	assert any('<a href="https://example/p1.pdf"' in m for m in link_blocks)
	assert any('open paper' in m for m in link_blocks)
	assert any('open pdf' in m for m in link_blocks)


def test_modal_omits_pdf_link_when_paper_lacks_pdf_url(stub_db):
	stub_db['search'].return_value = [_make_paper(pdf_url=None)]
	stub_db['count'].return_value = 1
	at = AppTest.from_file(_APP_PATH).run()
	(title_button,) = [b for b in at.button if b.label == 'Attention Is All You Need']
	title_button.click().run()
	link_blocks = [m.value for m in at.markdown if 'class="row-links"' in m.value]
	assert any('open paper' in m for m in link_blocks)
	assert not any('open pdf' in m for m in link_blocks)


def test_modal_skips_missing_summary_fields(stub_db):
	# A paper scored but never summarised (budget hit, abstract missing) still opens without crashing; only populated sections render.
	stub_db['search'].return_value = [_make_paper(methodology=None, findings=None, relevance=None, limitations=None)]
	stub_db['count'].return_value = 1
	at = AppTest.from_file(_APP_PATH).run()
	(title_button,) = [b for b in at.button if b.label == 'Attention Is All You Need']
	title_button.click().run()
	section_labels = [m.value for m in at.markdown if 'class="section-label"' in m.value]
	assert section_labels == []


# -- column headers --


def test_app_renders_column_headers_when_papers_present(stub_db):
	stub_db['search'].return_value = [_make_paper()]
	stub_db['count'].return_value = 1
	at = AppTest.from_file(_APP_PATH).run()
	headers = [m.value for m in at.markdown if 'class="col-header"' in m.value]
	assert any(_COPY['headers']['title'] in m for m in headers)
	assert any(_COPY['headers']['authors'] in m for m in headers)
	assert any(_COPY['headers']['date'] in m for m in headers)
	assert any(_COPY['headers']['rating'] in m for m in headers)


def test_app_hides_column_headers_when_no_papers(stub_db):
	at = AppTest.from_file(_APP_PATH).run()
	headers = [m.value for m in at.markdown if 'class="col-header"' in m.value]
	assert headers == []


# -- filters --


def test_app_passes_search_query_to_db(stub_db):
	at = AppTest.from_file(_APP_PATH).run()
	at.text_input(key='search_q').set_value('borderline').run()
	call = stub_db['search'].call_args
	assert call.kwargs['q'] == 'borderline'


def test_app_passes_none_for_empty_search_query(stub_db):
	AppTest.from_file(_APP_PATH).run()
	assert stub_db['search'].call_args.kwargs['q'] is None


def test_app_passes_date_range_namedtuple_to_db(stub_db):
	from datetime import date

	import db

	at = AppTest.from_file(_APP_PATH).run()
	at.date_input(key='date_since').set_value(date(2026, 1, 1)).run()
	at.date_input(key='date_until').set_value(date(2026, 5, 1)).run()
	call = stub_db['search'].call_args
	assert isinstance(call.kwargs['date_range'], db.DateRange)
	assert call.kwargs['date_range'].since == date(2026, 1, 1)
	assert call.kwargs['date_range'].until == date(2026, 5, 1)


def test_app_resets_to_page_one_when_filter_changes(stub_db):
	stub_db['search'].return_value = [_make_paper(paper_id=f'p{i}', title=f'Paper {i}') for i in range(20)]
	stub_db['count'].return_value = 50
	at = AppTest.from_file(_APP_PATH).run()
	at.button(key='page_next').click().run()
	assert at.session_state['page'] == 2
	at.text_input(key='search_q').set_value('different').run()
	assert at.session_state['page'] == 1


# -- pagination --


def test_app_hides_pagination_when_results_fit_one_page(stub_db):
	stub_db['search'].return_value = [_make_paper()]
	stub_db['count'].return_value = 1
	at = AppTest.from_file(_APP_PATH).run()
	page_keys = [b.key for b in at.button if b.key in ('page_prev', 'page_next')]
	assert page_keys == []


def test_app_shows_pagination_when_results_exceed_page_size(stub_db):
	stub_db['search'].return_value = [_make_paper(paper_id=f'p{i}', title=f'P{i}') for i in range(20)]
	stub_db['count'].return_value = 25
	at = AppTest.from_file(_APP_PATH).run()
	page_keys = {b.key for b in at.button if b.key in ('page_prev', 'page_next')}
	assert page_keys == {'page_prev', 'page_next'}


def test_app_next_button_advances_offset(stub_db):
	stub_db['search'].return_value = [_make_paper(paper_id=f'p{i}') for i in range(20)]
	stub_db['count'].return_value = 50
	at = AppTest.from_file(_APP_PATH).run()
	at.button(key='page_next').click().run()
	assert stub_db['search'].call_args.kwargs['offset'] == 20


def test_app_prev_button_decrements_offset(stub_db):
	stub_db['search'].return_value = [_make_paper(paper_id=f'p{i}') for i in range(20)]
	stub_db['count'].return_value = 50
	at = AppTest.from_file(_APP_PATH).run()
	at.button(key='page_next').click().run()
	at.button(key='page_prev').click().run()
	assert stub_db['search'].call_args.kwargs['offset'] == 0


def test_app_page_status_reflects_current_page_and_total(stub_db):
	stub_db['search'].return_value = [_make_paper(paper_id=f'p{i}') for i in range(20)]
	stub_db['count'].return_value = 50
	at = AppTest.from_file(_APP_PATH).run()
	page_status = [m.value for m in at.markdown if 'class="page-status"' in m.value]
	assert any('1 / 3' in m for m in page_status)
