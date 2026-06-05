import html
import json
import os
from pathlib import Path

import streamlit as st
import yaml
from dotenv import load_dotenv

import config
import db

load_dotenv()

_ROOT = Path(__file__).resolve().parents[1]
_STYLES_PATH = _ROOT / 'styles.css'
_COPY_PATH = _ROOT / 'copy.yaml'

with open(_COPY_PATH) as _f:
	COPY = yaml.safe_load(_f)

# The form exposes only the keys an operator should tune; infrastructure values (URLs, batch size, budget, PDF cap) stay seed/DB-managed.
HISTORY_LIMIT = 20
_VALUE_PREVIEW_CHARS = 80
_CHANGED_BY = 'dashboard'


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


def _form_values(cfg):
	# Queries round-trip through the textarea one per line; blank lines are dropped on parse so stray newlines never reach the fetcher.
	queries_text = st.text_area(
		COPY['config']['queries_label'], value='\n'.join(cfg.search_queries), height=220, key='cfg_queries'
	)
	context = st.text_area(COPY['config']['context_label'], value=cfg.research_context, height=320, key='cfg_context')
	cols = st.columns(3)
	threshold = cols[0].number_input(
		COPY['config']['threshold_label'], min_value=1, max_value=10, value=cfg.relevance_threshold, key='cfg_threshold'
	)
	days_back = cols[1].number_input(
		COPY['config']['days_back_label'], min_value=1, max_value=365, value=cfg.days_back, key='cfg_days_back'
	)
	max_per_query = cols[2].number_input(
		COPY['config']['max_per_query_label'],
		min_value=1,
		max_value=100,
		value=cfg.max_per_query,
		key='cfg_max_per_query',
	)
	model = st.text_input(COPY['config']['model_label'], value=cfg.gemini_model, key='cfg_model')
	return {
		'search_queries': [q.strip() for q in queries_text.splitlines() if q.strip()],
		'research_context': context,
		'relevance_threshold': threshold,
		'days_back': days_back,
		'max_per_query': max_per_query,
		'gemini_model': model.strip(),
	}


def _validate(values):
	errors = []
	if not values['search_queries']:
		errors.append(COPY['config']['error_no_queries'])
	if not values['research_context'].strip():
		errors.append(COPY['config']['error_no_context'])
	if not values['gemini_model']:
		errors.append(COPY['config']['error_no_model'])
	return errors


def _changed(cfg, values):
	# Only changed keys are written so the audit trail records edits, not no-op resubmits of the whole form.
	return {key: value for key, value in values.items() if value != getattr(cfg, key)}


def _save(conn, cfg, values):
	errors = _validate(values)
	if errors:
		for error in errors:
			st.markdown(f'<div class="config-error">{error}</div>', unsafe_allow_html=True)
		return
	updates = _changed(cfg, values)
	if not updates:
		st.markdown(f'<div class="config-status">{COPY["config"]["no_changes"]}</div>', unsafe_allow_html=True)
		return
	db.update_config(conn, updates, by=_CHANGED_BY)
	st.markdown(f'<div class="config-status">{COPY["config"]["saved"]}</div>', unsafe_allow_html=True)


def _value_preview(value):
	rendered = json.dumps(value)
	if len(rendered) > _VALUE_PREVIEW_CHARS:
		rendered = rendered[:_VALUE_PREVIEW_CHARS] + COPY['config']['truncation_suffix']
	return html.escape(rendered)


def _render_history(conn):
	st.markdown(
		f'<hr class="section-divider"/><div class="section-label">{COPY["config"]["history_label"]}</div>',
		unsafe_allow_html=True,
	)
	entries = db.list_config_history(conn, limit=HISTORY_LIMIT)
	if not entries:
		st.markdown(f'<div class="empty">{COPY["config"]["empty_history"]}</div>', unsafe_allow_html=True)
		return
	sep = f' {COPY["icons"]["meta_separator"]} '
	for entry in entries:
		changed_at = (
			entry['changed_at'].strftime('%Y-%m-%d %H:%M') if entry.get('changed_at') else COPY['icons']['date_unknown']
		)
		by = entry.get('changed_by') or COPY['icons']['date_unknown']
		line = sep.join([changed_at, html.escape(entry['key']), html.escape(by), _value_preview(entry['value'])])
		st.markdown(f'<div class="history-paper-meta">{line}</div>', unsafe_allow_html=True)


def main():
	st.set_page_config(page_title=COPY['config']['page_title'], layout='wide', initial_sidebar_state='expanded')
	_inject_styles()
	_render_sidebar(current='config')
	st.markdown(f'<div class="brand">{COPY["config"]["page_title"]}</div>', unsafe_allow_html=True)

	conn = _get_connection()
	# config.load seeds app_config from the seed file on first visit against an empty table, same bootstrap path as the cron pipeline.
	cfg = config.load(conn)

	with st.form(key='config_form'):
		values = _form_values(cfg)
		submitted = st.form_submit_button(COPY['config']['submit'])
	if submitted:
		_save(conn, cfg, values)

	_render_history(conn)


main()
