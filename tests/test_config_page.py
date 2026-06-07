from datetime import datetime
from pathlib import Path

import pytest
import yaml
from streamlit.testing.v1 import AppTest

import config

_APP_PATH = 'pages/config.py'
_COPY = yaml.safe_load((Path(__file__).resolve().parents[1] / 'copy.yaml').read_text())
_SEED = config.load_seed()


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
	# db.load_config is stubbed at the db boundary so config.load's bootstrap logic runs for real against the fake table contents.
	mocker.patch('db.connect', return_value=mocker.MagicMock())
	return {
		'load_config': mocker.patch('db.load_config', return_value=dict(_SEED)),
		'update_config': mocker.patch('db.update_config'),
		'list_config_history': mocker.patch('db.list_config_history', return_value=[]),
	}


def _make_history_entry(**overrides):
	base = {
		'id': 1,
		'key': 'relevance_threshold',
		'value': 7,
		'changed_at': datetime(2026, 6, 1, 9, 30),
		'changed_by': 'dashboard',
	}
	base.update(overrides)
	return base


def _submit(at):
	at.button[0].click().run()
	return at


# -- top-level rendering --


def test_page_renders_config_title(stub_db):
	at = AppTest.from_file(_APP_PATH).run()
	assert any(_COPY['config']['page_title'] in m.value for m in at.markdown)


def test_form_prepopulates_queries_one_per_line(stub_db):
	at = AppTest.from_file(_APP_PATH).run()
	assert at.text_area(key='cfg_queries').value == '\n'.join(_SEED['search_queries'])


def test_form_prepopulates_context_numbers_and_model(stub_db):
	at = AppTest.from_file(_APP_PATH).run()
	assert at.text_area(key='cfg_context').value == _SEED['research_context']
	assert at.number_input(key='cfg_threshold').value == _SEED['relevance_threshold']
	assert at.number_input(key='cfg_days_back').value == _SEED['days_back']
	assert at.number_input(key='cfg_max_per_query').value == _SEED['max_per_query']
	assert at.number_input(key='cfg_prefilter_top_n').value == _SEED['prefilter_top_n']
	assert at.number_input(key='cfg_fewshot_min_ratings').value == _SEED['fewshot_min_ratings']
	assert at.number_input(key='cfg_fewshot_min_feedback').value == _SEED['fewshot_min_feedback']
	assert at.number_input(key='cfg_fewshot_neighbours').value == _SEED['fewshot_neighbours']
	assert at.number_input(key='cfg_fewshot_good_rating').value == _SEED['fewshot_good_rating']
	assert at.number_input(key='cfg_fewshot_bad_rating').value == _SEED['fewshot_bad_rating']
	assert at.number_input(key='cfg_calibration_max_examples').value == _SEED['calibration_max_examples']
	assert at.text_input(key='cfg_model').value == _SEED['gemini_model']


def test_form_does_not_expose_infrastructure_keys(stub_db):
	# Base URLs, batch size, call budget and the PDF cap are deliberately not editable from the dashboard.
	at = AppTest.from_file(_APP_PATH).run()
	widget_keys = {w.key for w in [*at.text_area, *at.text_input, *at.number_input]}
	assert widget_keys == {
		'cfg_queries',
		'cfg_context',
		'cfg_threshold',
		'cfg_days_back',
		'cfg_max_per_query',
		'cfg_prefilter_top_n',
		'cfg_fewshot_min_ratings',
		'cfg_fewshot_min_feedback',
		'cfg_fewshot_neighbours',
		'cfg_fewshot_good_rating',
		'cfg_fewshot_bad_rating',
		'cfg_calibration_max_examples',
		'cfg_model',
	}


def test_first_visit_seeds_empty_config_table(stub_db):
	# config.load bootstraps app_config from the seed file when the table is empty, same path as the cron pipeline.
	stub_db['load_config'].return_value = {}
	at = AppTest.from_file(_APP_PATH).run()
	stub_db['update_config'].assert_called_once()
	assert stub_db['update_config'].call_args.args[1] == _SEED
	assert stub_db['update_config'].call_args.kwargs['by'] == 'seed'
	assert at.text_area(key='cfg_queries').value == '\n'.join(_SEED['search_queries'])


# -- submit --


def test_submit_without_changes_writes_nothing_and_shows_notice(stub_db):
	at = _submit(AppTest.from_file(_APP_PATH).run())
	stub_db['update_config'].assert_not_called()
	assert any(_COPY['config']['no_changes'] in m.value for m in at.markdown)


def test_submit_writes_only_changed_keys(stub_db):
	at = AppTest.from_file(_APP_PATH).run()
	at.number_input(key='cfg_threshold').set_value(9)
	_submit(at)
	stub_db['update_config'].assert_called_once()
	assert stub_db['update_config'].call_args.args[1] == {'relevance_threshold': 9}
	assert stub_db['update_config'].call_args.kwargs['by'] == 'dashboard'
	assert any(_COPY['config']['saved'] in m.value for m in at.markdown)


def test_submit_writes_changed_prefilter_top_n(stub_db):
	at = AppTest.from_file(_APP_PATH).run()
	at.number_input(key='cfg_prefilter_top_n').set_value(50)
	_submit(at)
	assert stub_db['update_config'].call_args.args[1] == {'prefilter_top_n': 50}


def test_submit_writes_changed_fewshot_knobs(stub_db):
	at = AppTest.from_file(_APP_PATH).run()
	at.number_input(key='cfg_fewshot_min_ratings').set_value(8)
	at.number_input(key='cfg_calibration_max_examples').set_value(12)
	_submit(at)
	assert stub_db['update_config'].call_args.args[1] == {'fewshot_min_ratings': 8, 'calibration_max_examples': 12}


def test_queries_textarea_parses_one_query_per_line_dropping_blanks(stub_db):
	at = AppTest.from_file(_APP_PATH).run()
	at.text_area(key='cfg_queries').set_value('  first query  \n\n   \nsecond query\n')
	_submit(at)
	assert stub_db['update_config'].call_args.args[1] == {'search_queries': ['first query', 'second query']}


def test_model_value_is_stripped_before_compare_and_write(stub_db):
	at = AppTest.from_file(_APP_PATH).run()
	at.text_input(key='cfg_model').set_value(f'  {_SEED["gemini_model"]}  ')
	_submit(at)
	# Whitespace-only edits normalise back to the stored value, so nothing is written.
	stub_db['update_config'].assert_not_called()


# -- validation --


def test_empty_queries_blocks_save(stub_db):
	at = AppTest.from_file(_APP_PATH).run()
	at.text_area(key='cfg_queries').set_value('\n  \n')
	_submit(at)
	stub_db['update_config'].assert_not_called()
	assert any(_COPY['config']['error_no_queries'] in m.value for m in at.markdown)


def test_empty_context_blocks_save(stub_db):
	at = AppTest.from_file(_APP_PATH).run()
	at.text_area(key='cfg_context').set_value('   ')
	_submit(at)
	stub_db['update_config'].assert_not_called()
	assert any(_COPY['config']['error_no_context'] in m.value for m in at.markdown)


def test_empty_model_blocks_save(stub_db):
	at = AppTest.from_file(_APP_PATH).run()
	at.text_input(key='cfg_model').set_value('  ')
	_submit(at)
	stub_db['update_config'].assert_not_called()
	assert any(_COPY['config']['error_no_model'] in m.value for m in at.markdown)


# -- audit trail --


def test_audit_trail_empty_state(stub_db):
	at = AppTest.from_file(_APP_PATH).run()
	assert any(_COPY['config']['empty_history'] in m.value for m in at.markdown)


def test_audit_trail_lists_recent_changes(stub_db):
	stub_db['list_config_history'].return_value = [
		_make_history_entry(key='relevance_threshold', value=7, changed_by='dashboard'),
		_make_history_entry(id=2, key='days_back', value=30, changed_by='seed'),
	]
	at = AppTest.from_file(_APP_PATH).run()
	rows = [m.value for m in at.markdown if 'class="history-paper-meta"' in m.value]
	assert any('relevance_threshold' in r and 'dashboard' in r and '2026-06-01 09:30' in r for r in rows)
	assert any('days_back' in r and 'seed' in r and '30' in r for r in rows)


def test_audit_trail_truncates_long_values(stub_db):
	stub_db['list_config_history'].return_value = [_make_history_entry(key='research_context', value='x' * 500)]
	at = AppTest.from_file(_APP_PATH).run()
	rows = [m.value for m in at.markdown if 'class="history-paper-meta"' in m.value]
	assert any(_COPY['config']['truncation_suffix'] in r and 'x' * 500 not in r for r in rows)


def test_audit_trail_renders_unknown_actor_glyph_when_changed_by_is_null(stub_db):
	stub_db['list_config_history'].return_value = [_make_history_entry(changed_by=None)]
	at = AppTest.from_file(_APP_PATH).run()
	rows = [m.value for m in at.markdown if 'class="history-paper-meta"' in m.value]
	assert any(_COPY['icons']['date_unknown'] in r for r in rows)


def test_audit_trail_escapes_key_and_changed_by(stub_db):
	# Keys and actors are DB-derived strings rendered under unsafe_allow_html; hostile content must arrive escaped.
	stub_db['list_config_history'].return_value = [
		_make_history_entry(key='<img src=x onerror=y>', changed_by='<script>z</script>')
	]
	at = AppTest.from_file(_APP_PATH).run()
	rows = [m.value for m in at.markdown if 'class="history-paper-meta"' in m.value]
	assert any('&lt;img src=x onerror=y&gt;' in r and '&lt;script&gt;z&lt;/script&gt;' in r for r in rows)
	assert not any('<script>' in r for r in rows)
