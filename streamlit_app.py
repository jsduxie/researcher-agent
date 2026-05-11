import os
from pathlib import Path

import streamlit as st
import yaml
from dotenv import load_dotenv

import db

# No-op on Streamlit Cloud, where secrets are injected via the UI / secrets.toml.
load_dotenv()

_ROOT = Path(__file__).resolve().parent
_STYLES_PATH = _ROOT / 'styles.css'
_COPY_PATH = _ROOT / 'copy.yaml'

with open(_COPY_PATH) as _f:
	COPY = yaml.safe_load(_f)

PAGE_SIZE = 20
SUMMARY_FIELDS = (
	('methodology', 'METHODOLOGY'),
	('findings', 'FINDINGS'),
	('relevance', 'RELEVANCE'),
	('limitations', 'LIMITATIONS'),
)


def _db_url_var():
	# `DASHBOARD_USE_TEST_DB=1` routes to DATABASE_URL_TEST for local UAT against seeds.
	return 'DATABASE_URL_TEST' if os.environ.get('DASHBOARD_USE_TEST_DB') == '1' else 'DATABASE_URL'


@st.cache_resource
def _get_connection():
	conn = db.connect(os.environ[_db_url_var()])
	db.init_schema(conn)
	return conn


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


def _filters():
	cols = st.columns([3, 1, 1])
	q = cols[0].text_input(COPY['labels']['search'], key='search_q', placeholder=COPY['search_placeholder'])
	since = cols[1].date_input(COPY['labels']['date_since'], value=None, key='date_since', format='YYYY-MM-DD')
	until = cols[2].date_input(COPY['labels']['date_until'], value=None, key='date_until', format='YYYY-MM-DD')
	# Reset to page 1 on any filter change so a tighter filter cannot leave us on an empty page.
	signature = (q, since, until)
	if st.session_state.get('_last_filter') != signature:
		st.session_state.page = 1
		st.session_state._last_filter = signature
	return (q or None), db.DateRange(since=since, until=until)


@st.dialog(' ', width='large')
def _show_paper(paper):
	title = paper.get('title') or 'Untitled'
	st.markdown(f'<div class="modal-title">{title}</div>', unsafe_allow_html=True)
	st.markdown(f'<div class="modal-authors">{_format_authors(paper.get("authors"))}</div>', unsafe_allow_html=True)
	meta = []
	if paper.get('year'):
		meta.append(str(paper['year']))
	if paper.get('citation_count'):
		meta.append(f'{paper["citation_count"]:,} citations')
	if paper.get('latest_rating'):
		meta.append(f'{paper["latest_rating"]} {COPY["icons"]["rating_star"]}')
	if paper.get('doi'):
		meta.append(f'DOI {paper["doi"]}')
	if meta:
		sep = f' {COPY["icons"]["meta_separator"]} '
		st.markdown(f'<div class="modal-meta">{sep.join(meta)}</div>', unsafe_allow_html=True)

	for field, label_text in SUMMARY_FIELDS:
		value = paper.get(field)
		if value:
			st.markdown(
				f'<hr class="section-divider"/>'
				f'<div class="section-label">{label_text}</div>'
				f'<div class="section-body">{value}</div>',
				unsafe_allow_html=True,
			)

	links = []
	if paper.get('url'):
		links.append(f'<a href="{paper["url"]}" target="_blank">{COPY["buttons"]["open_paper"]}</a>')
	if paper.get('pdf_url'):
		links.append(f'<a href="{paper["pdf_url"]}" target="_blank">{COPY["buttons"]["open_pdf"]}</a>')
	if links:
		st.markdown(f'<div class="row-links">{"".join(links)}</div>', unsafe_allow_html=True)


def _render_header():
	cols = st.columns([0.4, 5, 2.5, 1.2, 0.6])
	headers = [
		'',
		COPY['headers']['title'],
		COPY['headers']['authors'],
		COPY['headers']['date'],
		COPY['headers']['rating'],
	]
	for col, label in zip(cols, headers, strict=True):
		col.markdown(f'<div class="col-header">{label}</div>', unsafe_allow_html=True)
	st.markdown('<hr class="row-divider header-divider"/>', unsafe_allow_html=True)


def _render_paper_row(paper, index):
	paper_id = paper['paper_id']
	date_str = paper['fetched_at'].strftime('%Y-%m-%d') if paper.get('fetched_at') else COPY['icons']['date_unknown']

	row = st.columns([0.4, 5, 2.5, 1.2, 0.6])
	row[0].markdown(f'<div class="row-index">{index:02d}</div>', unsafe_allow_html=True)
	if row[1].button(paper.get('title') or 'Untitled', key=f'title_{paper_id}', use_container_width=True):
		_show_paper(paper)
	row[2].markdown(f'<div class="row-meta">{_format_authors(paper.get("authors"))}</div>', unsafe_allow_html=True)
	row[3].markdown(f'<div class="row-meta">{date_str}</div>', unsafe_allow_html=True)
	row[4].markdown(
		f'<div class="row-rating">{_format_rating(paper.get("latest_rating"))}</div>', unsafe_allow_html=True
	)
	st.markdown('<hr class="row-divider"/>', unsafe_allow_html=True)


def _pagination(total):
	pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
	if pages <= 1:
		return
	page = st.session_state.page
	cols = st.columns([1, 2, 1])
	if cols[0].button(COPY['buttons']['prev'], use_container_width=True, disabled=page <= 1, key='page_prev'):
		st.session_state.page = page - 1
		st.rerun()
	cols[1].markdown(f'<div class="page-status">{page} / {pages}</div>', unsafe_allow_html=True)
	if cols[2].button(COPY['buttons']['next'], use_container_width=True, disabled=page >= pages, key='page_next'):
		st.session_state.page = page + 1
		st.rerun()


def _inject_styles():
	st.markdown(f'<style>{_STYLES_PATH.read_text()}</style>', unsafe_allow_html=True)


def main():
	st.set_page_config(page_title=COPY['brand'], layout='wide', initial_sidebar_state='collapsed')
	_inject_styles()
	st.session_state.setdefault('page', 1)

	st.markdown(f'<div class="brand">{COPY["brand"]}</div>', unsafe_allow_html=True)

	q, date_range = _filters()

	conn = _get_connection()
	total = db.count_papers(conn, q=q, date_range=date_range)
	offset = (st.session_state.page - 1) * PAGE_SIZE
	papers = db.search_papers(conn, q=q, date_range=date_range, limit=PAGE_SIZE, offset=offset)

	if not papers:
		st.markdown(f'<div class="empty">{COPY["empty_state"]}</div>', unsafe_allow_html=True)
	else:
		_render_header()
		for i, paper in enumerate(papers, start=offset + 1):
			_render_paper_row(paper, i)

	_pagination(total)


main()
