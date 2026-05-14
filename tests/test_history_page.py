from datetime import datetime
from pathlib import Path

import pytest
import yaml
from streamlit.testing.v1 import AppTest

_APP_PATH = 'pages/history.py'
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
	mocker.patch('db.connect', return_value=mocker.MagicMock())
	return {
		'list_runs': mocker.patch('db.list_runs', return_value=[]),
		'list_papers_for_run': mocker.patch('db.list_papers_for_run', return_value=[]),
	}


def _make_run(**overrides):
	base = {
		'id': 1,
		'started_at': datetime(2026, 5, 13, 8, 0),
		'finished_at': datetime(2026, 5, 13, 8, 6),
		'papers_fetched': 64,
		'papers_kept': 15,
		'queries_attempted': 8,
		'queries_errored': 0,
	}
	base.update(overrides)
	return base


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
		'methodology': None,
		'findings': None,
		'relevance': None,
		'limitations': None,
		'latest_rating': 4,
		'was_new': True,
	}
	base.update(overrides)
	return base


# -- top-level rendering --


def test_page_renders_history_title(stub_db):
	at = AppTest.from_file(_APP_PATH).run()
	assert any(_COPY['history']['page_title'] in m.value for m in at.markdown)


def test_page_shows_empty_state_when_no_runs(stub_db):
	at = AppTest.from_file(_APP_PATH).run()
	assert any(_COPY['history']['empty_runs'] in m.value for m in at.markdown)


def test_page_does_not_call_list_papers_for_run_when_no_runs(stub_db):
	AppTest.from_file(_APP_PATH).run()
	stub_db['list_papers_for_run'].assert_not_called()


# -- run expanders --


def test_page_renders_one_expander_per_run(stub_db):
	stub_db['list_runs'].return_value = [_make_run(id=1), _make_run(id=2), _make_run(id=3)]
	at = AppTest.from_file(_APP_PATH).run()
	assert len(at.expander) == 3


def test_run_header_includes_started_at_fetched_and_kept(stub_db):
	stub_db['list_runs'].return_value = [_make_run(papers_fetched=64, papers_kept=15)]
	at = AppTest.from_file(_APP_PATH).run()
	(exp,) = at.expander
	assert '2026-05-13 08:00' in exp.label
	assert '64' in exp.label
	assert '15' in exp.label


def test_run_header_includes_query_attempted_and_errored_counts(stub_db):
	stub_db['list_runs'].return_value = [_make_run(queries_attempted=8, queries_errored=2)]
	at = AppTest.from_file(_APP_PATH).run()
	(exp,) = at.expander
	assert '8' in exp.label
	assert '2' in exp.label


def test_run_header_marks_in_progress_when_finished_at_is_none(stub_db):
	# A run that never reached finish_run (cron crash mid-pipeline) shows the in-progress marker rather than implying it completed at the same instant it started.
	stub_db['list_runs'].return_value = [_make_run(finished_at=None)]
	at = AppTest.from_file(_APP_PATH).run()
	(exp,) = at.expander
	assert _COPY['history']['in_progress'] in exp.label


def test_run_header_treats_missing_papers_kept_as_zero(stub_db):
	stub_db['list_runs'].return_value = [_make_run(papers_kept=None)]
	at = AppTest.from_file(_APP_PATH).run()
	(exp,) = at.expander
	assert f'0 {_COPY["history"]["kept_label"]}' in exp.label


# -- per-run paper roster --


def test_expander_loads_papers_for_its_run_id(stub_db):
	stub_db['list_runs'].return_value = [_make_run(id=42)]
	stub_db['list_papers_for_run'].return_value = [_make_paper()]
	AppTest.from_file(_APP_PATH).run()
	stub_db['list_papers_for_run'].assert_called_once()
	assert stub_db['list_papers_for_run'].call_args.args[1] == 42


def test_expander_renders_paper_titles(stub_db):
	stub_db['list_runs'].return_value = [_make_run()]
	stub_db['list_papers_for_run'].return_value = [
		_make_paper(paper_id='p1', title='Paper One'),
		_make_paper(paper_id='p2', title='Paper Two'),
	]
	at = AppTest.from_file(_APP_PATH).run()
	paper_titles = [m.value for m in at.markdown if 'class="history-paper-title"' in m.value]
	assert any('Paper One' in m for m in paper_titles)
	assert any('Paper Two' in m for m in paper_titles)


def test_expander_renders_authors_year_and_latest_rating(stub_db):
	stub_db['list_runs'].return_value = [_make_run()]
	stub_db['list_papers_for_run'].return_value = [_make_paper(authors=['Smith', 'Jones'], year=2024, latest_rating=5)]
	at = AppTest.from_file(_APP_PATH).run()
	meta_blocks = [m.value for m in at.markdown if 'class="history-paper-meta"' in m.value]
	assert any('Smith' in m for m in meta_blocks)
	assert any('2024' in m for m in meta_blocks)
	assert any(f'5 {_COPY["icons"]["rating_star"]}' in m for m in meta_blocks)


def test_expander_renders_unrated_glyph_when_paper_has_no_rating(stub_db):
	stub_db['list_runs'].return_value = [_make_run()]
	stub_db['list_papers_for_run'].return_value = [_make_paper(latest_rating=None)]
	at = AppTest.from_file(_APP_PATH).run()
	meta_blocks = [m.value for m in at.markdown if 'class="history-paper-meta"' in m.value]
	assert any(_COPY['icons']['rating_unrated'] in m for m in meta_blocks)


def test_expander_shows_empty_papers_message_for_run_with_no_roster(stub_db):
	# Pre-migration runs and any future run that crashed before recording its roster both surface this message; tested so the History page never crashes.
	stub_db['list_runs'].return_value = [_make_run()]
	stub_db['list_papers_for_run'].return_value = []
	at = AppTest.from_file(_APP_PATH).run()
	empty_blocks = [m.value for m in at.markdown if _COPY['history']['empty_run_papers'] in m.value]
	assert empty_blocks


def test_paper_first_seen_in_this_run_is_tagged_new(stub_db):
	# was_new=True on the join row means this run is the first to fetch the paper; the History page surfaces that as a "new" tag so operators can spot fresh hits.
	stub_db['list_runs'].return_value = [_make_run()]
	stub_db['list_papers_for_run'].return_value = [_make_paper(was_new=True)]
	at = AppTest.from_file(_APP_PATH).run()
	title_blocks = [m.value for m in at.markdown if 'class="history-paper-title"' in m.value]
	assert any('history-tag-new' in m for m in title_blocks)
	assert any(_COPY['history']['tag_new'] in m for m in title_blocks)
	assert not any('history-tag-repeat' in m for m in title_blocks)


def test_paper_seen_in_an_earlier_run_is_tagged_repeat(stub_db):
	# was_new=False means upsert_paper hit a conflict, so this run is re-encountering an existing paper; tag as repeat so the operator can ignore it at a glance.
	stub_db['list_runs'].return_value = [_make_run()]
	stub_db['list_papers_for_run'].return_value = [_make_paper(was_new=False)]
	at = AppTest.from_file(_APP_PATH).run()
	title_blocks = [m.value for m in at.markdown if 'class="history-paper-title"' in m.value]
	assert any('history-tag-repeat' in m for m in title_blocks)
	assert any(_COPY['history']['tag_repeat'] in m for m in title_blocks)
	assert not any('history-tag-new' in m for m in title_blocks)


def test_page_uses_dark_palette_in_css(stub_db):
	at = AppTest.from_file(_APP_PATH).run()
	css = '\n'.join(m.value for m in at.markdown)
	# History page inherits the same theme tokens as the entrypoint; verifies the palette doesn't drift if styles.css changes shape.
	assert '#131110' in css
	assert '#d99565' in css
