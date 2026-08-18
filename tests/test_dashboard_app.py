from datetime import datetime
from pathlib import Path

import pytest
import yaml
from streamlit.testing.v1 import AppTest

_APP_PATH = Path(__file__).resolve().parents[1] / 'streamlit_app.py'
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


@pytest.fixture(autouse=True)
def _stub_page_link(mocker):
	# AppTest doesn't populate the multipage page registry, so st.page_link raises KeyError('url_pathname'). The sidebar nav rows are not the subject of these tests, so no-op them.
	mocker.patch('streamlit.page_link')


@pytest.fixture
def stub_db(mocker):
	# Patch db.connect so the cached resource is a benign mock; nothing else in db.py is touched at import time, so call sites can be patched per test.
	mocker.patch('db.connect', return_value=mocker.MagicMock())
	return {
		'search': mocker.patch('db.search_papers', return_value=[]),
		'count': mocker.patch('db.count_papers', return_value=0),
		'latest_rating': mocker.patch('db.get_latest_rating', return_value=None),
		'latest_field': mocker.patch('db.get_latest_field_feedback', return_value={}),
		'insert_rating': mocker.patch('db.insert_rating'),
		'insert_feedback': mocker.patch('db.insert_summary_feedback'),
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


def test_modal_renders_abstract_section_with_abstract_text(stub_db):
	stub_db['search'].return_value = [_make_paper(abstract='Original abstract text from Semantic Scholar.')]
	stub_db['count'].return_value = 1
	at = AppTest.from_file(_APP_PATH).run()
	(title_button,) = [b for b in at.button if b.label == 'Attention Is All You Need']
	title_button.click().run()
	section_blocks = [m.value for m in at.markdown if 'class="section-label"' in m.value]
	assert any('ABSTRACT' in m for m in section_blocks)
	body_blocks = [m.value for m in at.markdown if 'class="section-body"' in m.value]
	assert any('Original abstract text from Semantic Scholar.' in m for m in body_blocks)


def test_modal_renders_abstract_section_before_methodology(stub_db):
	# Abstract is the original source material; it must appear above the first Gemini summary section so the operator reads source then summary.
	stub_db['search'].return_value = [_make_paper(abstract='An abstract')]
	stub_db['count'].return_value = 1
	at = AppTest.from_file(_APP_PATH).run()
	(title_button,) = [b for b in at.button if b.label == 'Attention Is All You Need']
	title_button.click().run()
	section_labels = [m.value for m in at.markdown if 'class="section-label"' in m.value]
	abstract_idx = next(i for i, m in enumerate(section_labels) if 'ABSTRACT' in m)
	methodology_idx = next(i for i, m in enumerate(section_labels) if 'METHODOLOGY' in m)
	assert abstract_idx < methodology_idx


def test_modal_omits_abstract_section_when_paper_has_no_abstract(stub_db):
	stub_db['search'].return_value = [_make_paper(abstract=None)]
	stub_db['count'].return_value = 1
	at = AppTest.from_file(_APP_PATH).run()
	(title_button,) = [b for b in at.button if b.label == 'Attention Is All You Need']
	title_button.click().run()
	section_labels = [m.value for m in at.markdown if 'class="section-label"' in m.value]
	assert not any('ABSTRACT' in m for m in section_labels)


def test_modal_omits_abstract_section_when_abstract_is_empty_string(stub_db):
	stub_db['search'].return_value = [_make_paper(abstract='')]
	stub_db['count'].return_value = 1
	at = AppTest.from_file(_APP_PATH).run()
	(title_button,) = [b for b in at.button if b.label == 'Attention Is All You Need']
	title_button.click().run()
	section_labels = [m.value for m in at.markdown if 'class="section-label"' in m.value]
	assert not any('ABSTRACT' in m for m in section_labels)


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


def test_modal_renders_paper_and_pdf_links_in_actions_row(stub_db):
	# Both action links sit on their own right-aligned row just above the first section divider, reachable without scrolling past the summary.
	stub_db['search'].return_value = [_make_paper()]
	stub_db['count'].return_value = 1
	at = AppTest.from_file(_APP_PATH).run()
	(title_button,) = [b for b in at.button if b.label == 'Attention Is All You Need']
	title_button.click().run()
	action_blocks = [m.value for m in at.markdown if 'class="modal-actions"' in m.value]
	assert any('class="modal-action-link" href="https://example/p1"' in m for m in action_blocks)
	assert any('class="modal-action-link" href="https://example/p1.pdf"' in m for m in action_blocks)
	assert any('open paper' in m for m in action_blocks)
	assert any('open pdf' in m for m in action_blocks)


def test_modal_does_not_render_bottom_row_links_block(stub_db):
	# Links live in the modal-actions row now; the bottom row-links container must not appear at all.
	stub_db['search'].return_value = [_make_paper()]
	stub_db['count'].return_value = 1
	at = AppTest.from_file(_APP_PATH).run()
	(title_button,) = [b for b in at.button if b.label == 'Attention Is All You Need']
	title_button.click().run()
	assert not any('class="row-links"' in m.value for m in at.markdown)


def test_modal_actions_row_omits_pdf_link_when_paper_lacks_pdf_url(stub_db):
	stub_db['search'].return_value = [_make_paper(pdf_url=None)]
	stub_db['count'].return_value = 1
	at = AppTest.from_file(_APP_PATH).run()
	(title_button,) = [b for b in at.button if b.label == 'Attention Is All You Need']
	title_button.click().run()
	action_blocks = [m.value for m in at.markdown if 'class="modal-actions"' in m.value]
	assert action_blocks
	assert not any('open pdf' in m for m in action_blocks)
	# Open-paper link is still present because that URL is set.
	assert any('open paper' in m for m in action_blocks)


def test_modal_actions_row_omits_open_paper_link_when_paper_lacks_url(stub_db):
	stub_db['search'].return_value = [_make_paper(url=None)]
	stub_db['count'].return_value = 1
	at = AppTest.from_file(_APP_PATH).run()
	(title_button,) = [b for b in at.button if b.label == 'Attention Is All You Need']
	title_button.click().run()
	action_blocks = [m.value for m in at.markdown if 'class="modal-actions"' in m.value]
	assert action_blocks
	assert not any('open paper' in m for m in action_blocks)
	# PDF link is still present because that URL is set.
	assert any('open pdf' in m for m in action_blocks)


def test_modal_actions_row_does_not_render_when_paper_has_neither_url(stub_db):
	stub_db['search'].return_value = [_make_paper(url=None, pdf_url=None)]
	stub_db['count'].return_value = 1
	at = AppTest.from_file(_APP_PATH).run()
	(title_button,) = [b for b in at.button if b.label == 'Attention Is All You Need']
	title_button.click().run()
	# When both URLs are missing the actions wrapper itself shouldn't render at all.
	assert not any('class="modal-actions"' in m.value for m in at.markdown)


def test_modal_skips_missing_summary_fields(stub_db):
	# A paper scored but never summarised (budget hit, no Gemini output) still opens without crashing; no Gemini summary sections render even when the abstract does.
	stub_db['search'].return_value = [
		_make_paper(methodology=None, findings=None, relevance=None, limitations=None, abstract=None)
	]
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


# -- review form (rating + per-field feedback) --


def _open_modal(stub_db, paper=None):
	# Render the app, open the modal for the (only) paper, return the AppTest handle.
	stub_db['search'].return_value = [paper or _make_paper()]
	stub_db['count'].return_value = 1
	at = AppTest.from_file(_APP_PATH).run()
	(title_button,) = [b for b in at.button if b.label == 'Attention Is All You Need']
	title_button.click().run()
	return at


def _submit_button(at):
	return next(b for b in at.button if b.label == _COPY['review']['submit'])


def test_form_renders_five_sliders_four_textareas_and_submit_when_all_fields_present(stub_db):
	at = _open_modal(stub_db)
	slider_keys = {s.key for s in at.slider}
	expected_field_keys = {
		f'rating_p1_{field}'
		for field, _ in [('methodology', None), ('findings', None), ('relevance', None), ('limitations', None)]
	}
	assert expected_field_keys.issubset(slider_keys)
	assert 'overall_p1' in slider_keys
	# Five sliders total: four field + one overall. Four textareas, one per field.
	assert len([s for s in at.slider if s.key in expected_field_keys or s.key == 'overall_p1']) == 5
	assert len(at.text_area) == 4
	assert _submit_button(at) is not None


def test_form_does_not_render_before_modal_opens(stub_db):
	stub_db['search'].return_value = [_make_paper()]
	stub_db['count'].return_value = 1
	at = AppTest.from_file(_APP_PATH).run()
	# Modal-only widgets must not bleed onto the list view.
	assert len(at.slider) == 0
	assert len(at.text_area) == 0


def test_form_omits_section_slider_and_textarea_when_field_is_missing(stub_db):
	# Only methodology populated; the form should render 1 field slider + 1 textarea + 1 overall slider, with submit still present.
	at = _open_modal(stub_db, paper=_make_paper(findings=None, relevance=None, limitations=None))
	slider_keys = {s.key for s in at.slider}
	assert 'rating_p1_methodology' in slider_keys
	assert 'rating_p1_findings' not in slider_keys
	assert 'overall_p1' in slider_keys
	assert len(at.text_area) == 1


def test_form_defaults_to_three_when_no_prior_rating(stub_db):
	at = _open_modal(stub_db)
	overall = next(s for s in at.slider if s.key == 'overall_p1')
	field = next(s for s in at.slider if s.key == 'rating_p1_methodology')
	assert overall.value == 3
	assert field.value == 3


def test_form_pre_fills_overall_from_latest_rating(stub_db):
	stub_db['latest_rating'].return_value = 4
	at = _open_modal(stub_db)
	overall = next(s for s in at.slider if s.key == 'overall_p1')
	assert overall.value == 4


def test_form_pre_fills_field_rating_and_correction_from_latest_feedback(stub_db):
	stub_db['latest_field'].return_value = {'methodology': {'rating': 5, 'correction': 'reworded'}}
	at = _open_modal(stub_db)
	slider = next(s for s in at.slider if s.key == 'rating_p1_methodology')
	textarea = next(t for t in at.text_area if t.key == 'correction_p1_methodology')
	assert slider.value == 5
	assert textarea.value == 'reworded'


def test_form_submit_writes_one_rating_row_with_overall_value(stub_db):
	at = _open_modal(stub_db)
	at.slider(key='overall_p1').set_value(5)
	_submit_button(at).click().run()
	stub_db['insert_rating'].assert_called_once()
	call = stub_db['insert_rating'].call_args
	assert call.args[1] == 'p1'
	assert call.args[2] == 5


def test_form_submit_writes_four_summary_feedback_rows_one_per_field(stub_db):
	at = _open_modal(stub_db)
	_submit_button(at).click().run()
	assert stub_db['insert_feedback'].call_count == 4
	fields_written = {c.args[2] for c in stub_db['insert_feedback'].call_args_list}
	assert fields_written == {'methodology', 'findings', 'relevance', 'limitations'}


def test_form_submit_writes_per_field_rating_value(stub_db):
	at = _open_modal(stub_db)
	at.slider(key='rating_p1_findings').set_value(2)
	_submit_button(at).click().run()
	findings_call = next(c for c in stub_db['insert_feedback'].call_args_list if c.args[2] == 'findings')
	assert findings_call.kwargs['rating'] == 2


def test_form_submit_stores_correction_when_non_empty(stub_db):
	at = _open_modal(stub_db)
	at.text_area(key='correction_p1_relevance').set_value('clarifies the BPD link')
	_submit_button(at).click().run()
	relevance_call = next(c for c in stub_db['insert_feedback'].call_args_list if c.args[2] == 'relevance')
	assert relevance_call.kwargs['correction'] == 'clarifies the BPD link'


def test_form_submit_stores_none_when_correction_is_empty(stub_db):
	at = _open_modal(stub_db)
	_submit_button(at).click().run()
	# Default textarea value is empty; persisted as NULL to keep summary_feedback's correction column meaningful.
	for call in stub_db['insert_feedback'].call_args_list:
		assert call.kwargs['correction'] is None


def test_form_submit_stores_none_when_correction_is_whitespace_only(stub_db):
	at = _open_modal(stub_db)
	at.text_area(key='correction_p1_methodology').set_value('   \n  ')
	_submit_button(at).click().run()
	methodology_call = next(c for c in stub_db['insert_feedback'].call_args_list if c.args[2] == 'methodology')
	assert methodology_call.kwargs['correction'] is None


def test_form_resubmit_appends_new_rows_each_time(stub_db):
	# Append-only event log: each submit appends one ratings row + one summary_feedback row per rendered field. Latest-row semantics live in db.get_latest_*.
	at = _open_modal(stub_db)
	_submit_button(at).click().run()
	# st.dialog closes on the post-submit rerun; re-open it before submitting again.
	next(b for b in at.button if b.label == 'Attention Is All You Need').click().run()
	_submit_button(at).click().run()
	assert stub_db['insert_rating'].call_count == 2
	assert stub_db['insert_feedback'].call_count == 8


def test_form_submit_skips_summary_feedback_for_missing_fields(stub_db):
	# When a paper only has methodology, only one summary_feedback row should be written on submit (not four).
	at = _open_modal(stub_db, paper=_make_paper(findings=None, relevance=None, limitations=None))
	_submit_button(at).click().run()
	assert stub_db['insert_feedback'].call_count == 1
	assert stub_db['insert_feedback'].call_args.args[2] == 'methodology'
	# Overall rating is still recorded even when only one section is rendered.
	stub_db['insert_rating'].assert_called_once()


# -- HTML escaping of upstream strings --

_HOSTILE = '<script>alert(1)</script>'
_HOSTILE_ESCAPED = '&lt;script&gt;alert(1)&lt;/script&gt;'


def _click_title(at, label):
	(title_button,) = [b for b in at.button if b.label == label]
	title_button.click().run()
	return at


def test_modal_escapes_title_html(stub_db):
	stub_db['search'].return_value = [_make_paper(title=_HOSTILE)]
	stub_db['count'].return_value = 1
	at = _click_title(AppTest.from_file(_APP_PATH).run(), _HOSTILE)
	modal_titles = [m.value for m in at.markdown if 'class="modal-title"' in m.value]
	assert any(_HOSTILE_ESCAPED in m for m in modal_titles)
	assert not any(_HOSTILE in m for m in modal_titles)


def test_modal_escapes_authors_abstract_and_summary_fields(stub_db):
	stub_db['search'].return_value = [_make_paper(authors=['<b>Eve</b>'], abstract='<i>abs</i>', methodology=_HOSTILE)]
	stub_db['count'].return_value = 1
	at = _click_title(AppTest.from_file(_APP_PATH).run(), 'Attention Is All You Need')
	authors_blocks = [m.value for m in at.markdown if 'class="modal-authors"' in m.value]
	body_blocks = [m.value for m in at.markdown if 'class="section-body"' in m.value]
	assert any('&lt;b&gt;Eve&lt;/b&gt;' in m for m in authors_blocks)
	assert any('&lt;i&gt;abs&lt;/i&gt;' in m for m in body_blocks)
	assert any(_HOSTILE_ESCAPED in m for m in body_blocks)
	assert not any(_HOSTILE in m for m in body_blocks)


def test_modal_escapes_doi_in_meta(stub_db):
	stub_db['search'].return_value = [_make_paper(doi='10.1/<svg onload=x>')]
	stub_db['count'].return_value = 1
	at = _click_title(AppTest.from_file(_APP_PATH).run(), 'Attention Is All You Need')
	modal_meta = [m.value for m in at.markdown if 'class="modal-meta"' in m.value]
	assert any('10.1/&lt;svg onload=x&gt;' in m for m in modal_meta)


def test_modal_action_links_escape_href_and_add_noopener(stub_db):
	# A quote in the URL must not break out of the href attribute; target=_blank links carry rel to prevent reverse-tabnabbing.
	hostile_url = 'https://example/p1?q="><script>x</script>'
	stub_db['search'].return_value = [_make_paper(url=hostile_url)]
	stub_db['count'].return_value = 1
	at = _click_title(AppTest.from_file(_APP_PATH).run(), 'Attention Is All You Need')
	action_blocks = [m.value for m in at.markdown if 'class="modal-actions"' in m.value]
	assert any('href="https://example/p1?q=&quot;&gt;&lt;script&gt;x&lt;/script&gt;"' in m for m in action_blocks)
	assert any('rel="noopener noreferrer"' in m for m in action_blocks)
	assert not any('<script>' in m for m in action_blocks)


def test_row_escapes_author_names(stub_db):
	stub_db['search'].return_value = [_make_paper(authors=['<b>Eve</b>'])]
	stub_db['count'].return_value = 1
	at = AppTest.from_file(_APP_PATH).run()
	row_meta = [m.value for m in at.markdown if 'class="row-meta"' in m.value]
	assert any('&lt;b&gt;Eve&lt;/b&gt;' in m for m in row_meta)
	assert not any('<b>Eve</b>' in m for m in row_meta)


def test_row_renders_no_authors_placeholder_when_authors_empty(stub_db):
	stub_db['search'].return_value = [_make_paper(authors=[])]
	stub_db['count'].return_value = 1
	at = AppTest.from_file(_APP_PATH).run()
	row_meta = [m.value for m in at.markdown if 'class="row-meta"' in m.value]
	assert any(_COPY['authors']['none'] in m for m in row_meta)


def test_modal_omits_meta_row_when_paper_has_no_meta_fields(stub_db):
	sparse = _make_paper(year=None, citation_count=None, latest_rating=None, doi=None)
	stub_db['search'].return_value = [sparse]
	stub_db['count'].return_value = 1
	at = _click_title(AppTest.from_file(_APP_PATH).run(), 'Attention Is All You Need')
	assert not any('class="modal-meta"' in m.value for m in at.markdown)
