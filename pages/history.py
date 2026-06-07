import html
import os
from pathlib import Path

import streamlit as st
import yaml
from dotenv import load_dotenv

import db

load_dotenv()

_ROOT = Path(__file__).resolve().parents[1]
_STYLES_PATH = _ROOT / 'styles.css'
_COPY_PATH = _ROOT / 'copy.yaml'

with open(_COPY_PATH) as _f:
	COPY = yaml.safe_load(_f)


def _db_url_var():
	# Mirror the entrypoint's switch so the same DASHBOARD_USE_TEST_DB=1 flag works on every page.
	return 'DATABASE_URL_TEST' if os.environ.get('DASHBOARD_USE_TEST_DB') == '1' else 'DATABASE_URL'


@st.cache_resource
def _get_connection():
	conn = db.connect(os.environ[_db_url_var()])
	db.init_schema(conn)
	return conn


def _inject_styles():
	st.markdown(f'<style>{_STYLES_PATH.read_text()}</style>', unsafe_allow_html=True)


_NAV_ITEMS = (('papers', 'streamlit_app.py'), ('history', 'pages/history.py'), ('config', 'pages/config.py'))


def _render_sidebar(current):
	# Mirrored on every page so the nav persists. The active row renders as styled markdown (with the accent border) because Streamlit's st.page_link doesn't expose its current-page state on the DOM for CSS to target.
	with st.sidebar:
		for key, path in _NAV_ITEMS:
			label = COPY['nav'][key]
			if key == current:
				st.markdown(f'<div class="sidebar-nav-active">{label}</div>', unsafe_allow_html=True)
			else:
				st.page_link(path, label=label)


def _format_authors(authors):
	if not authors:
		return COPY['authors']['none']
	if len(authors) <= 3:
		return ', '.join(authors)
	return ', '.join(authors[:3]) + COPY['authors']['etal_suffix']


def _format_rating(rating):
	if not rating:
		return COPY['icons']['rating_unrated']
	return f'{rating} {COPY["icons"]["rating_star"]}'


def _run_header(run):
	# Single mono line matching the rest of the dashboard's editorial style: timestamp, then counts, then in-progress marker if finished_at is NULL.
	started = run['started_at'].strftime('%Y-%m-%d %H:%M') if run.get('started_at') else COPY['icons']['date_unknown']
	parts = [
		started,
		f'{run["papers_fetched"]} {COPY["history"]["fetched_label"]}',
		f'{run.get("papers_kept") or 0} {COPY["history"]["kept_label"]}',
	]
	queries_attempted = run.get('queries_attempted')
	queries_errored = run.get('queries_errored')
	if queries_attempted is not None:
		parts.append(
			f'{queries_attempted} {COPY["history"]["queries_label"]} ({queries_errored or 0} {COPY["history"]["errored_label"]})'
		)
	if run.get('finished_at') is None:
		parts.append(COPY['history']['in_progress'])
	sep = f' {COPY["icons"]["meta_separator"]} '
	return sep.join(parts)


def _render_run_papers(papers):
	if not papers:
		st.markdown(f'<div class="empty">{COPY["history"]["empty_run_papers"]}</div>', unsafe_allow_html=True)
		return
	for paper in papers:
		# Titles and authors come from upstream sources via the DB; escape before interpolating under unsafe_allow_html.
		title = html.escape(paper.get('title') or 'Untitled')
		# Tag every paper as new on first encounter or repeat on subsequent runs; lets operators see at a glance which runs introduced what.
		tag_label = COPY['history']['tag_new'] if paper.get('was_new') else COPY['history']['tag_repeat']
		tag_class = 'history-tag-new' if paper.get('was_new') else 'history-tag-repeat'
		st.markdown(
			f'<div class="history-paper-title"><span class="history-tag {tag_class}">{tag_label}</span>{title}</div>',
			unsafe_allow_html=True,
		)
		meta_parts = [html.escape(_format_authors(paper.get('authors')))]
		if paper.get('year'):
			meta_parts.append(str(paper['year']))
		meta_parts.append(_format_rating(paper.get('latest_rating')))
		sep = f' {COPY["icons"]["meta_separator"]} '
		st.markdown(f'<div class="history-paper-meta">{sep.join(meta_parts)}</div>', unsafe_allow_html=True)


def _render_run_prompts(run):
	# Logged prompts can carry upstream paper text, but st.code renders verbatim and never as HTML, so no escaping is needed here unlike the unsafe_allow_html blocks above.
	for key, label_key in (('scorer_prompt', 'scorer_prompt_label'), ('summariser_prompt', 'summariser_prompt_label')):
		prompt = run.get(key)
		if not prompt:
			continue
		with st.expander(COPY['history'][label_key], expanded=False):
			st.code(prompt)


def main():
	st.set_page_config(page_title=COPY['history']['page_title'], layout='wide', initial_sidebar_state='expanded')
	_inject_styles()
	_render_sidebar(current='history')
	st.markdown(f'<div class="brand">{COPY["history"]["page_title"]}</div>', unsafe_allow_html=True)

	conn = _get_connection()
	runs = db.list_runs(conn)

	if not runs:
		st.markdown(f'<div class="empty">{COPY["history"]["empty_runs"]}</div>', unsafe_allow_html=True)
		return

	for run in runs:
		with st.expander(_run_header(run), expanded=False):
			papers = db.list_papers_for_run(conn, run['id'])
			_render_run_papers(papers)
			_render_run_prompts(run)


main()
