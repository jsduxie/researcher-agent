import functools
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import psycopg


class DateRange(NamedTuple):
	since: datetime | None = None
	until: datetime | None = None


_SCHEMA_PATH = Path(__file__).parent / 'db' / 'schema.sql'

_UPSERT_PAPER_SQL = """
INSERT INTO papers (paper_id, title, abstract, year, citation_count, url, doi, pdf_url, fetched_at, updated_at)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
ON CONFLICT (paper_id) DO UPDATE
SET title = EXCLUDED.title,
    abstract = EXCLUDED.abstract,
    year = EXCLUDED.year,
    citation_count = EXCLUDED.citation_count,
    url = EXCLUDED.url,
    doi = EXCLUDED.doi,
    pdf_url = EXCLUDED.pdf_url,
    updated_at = NOW()
RETURNING (xmax = 0) AS was_inserted
"""

_UPSERT_SUMMARY_SQL = """
INSERT INTO summaries (paper_id, methodology, findings, relevance, limitations, model_version, created_at, updated_at)
VALUES (%s, %s, %s, %s, %s, %s, NOW(), NOW())
ON CONFLICT (paper_id) DO UPDATE
SET methodology = EXCLUDED.methodology,
    findings = EXCLUDED.findings,
    relevance = EXCLUDED.relevance,
    limitations = EXCLUDED.limitations,
    model_version = EXCLUDED.model_version,
    updated_at = NOW()
"""

_SUMMARY_COLUMNS = ('methodology', 'findings', 'relevance', 'limitations', 'model_version')


def connect(database_url):
	# autocommit and prepare_threshold=None keep us compatible with Neon's PgBouncer pooler.
	return psycopg.connect(database_url, autocommit=True, prepare_threshold=None)


@contextmanager
def session(database_url):
	conn = connect(database_url)
	# Stashed so @database_reconnect can reopen a fresh session on OperationalError.
	conn._database_url = database_url
	try:
		yield conn
	finally:
		conn.close()


def database_reconnect(fn):
	# Catches a single psycopg.OperationalError on the wrapped helper, opens a fresh session from conn._database_url, and retries exactly once.
	@functools.wraps(fn)
	def wrapper(conn, *args, **kwargs):
		try:
			return fn(conn, *args, **kwargs)
		except psycopg.OperationalError:
			url = getattr(conn, '_database_url', None)
			if url is None:
				raise
			with session(url) as fresh_conn:
				return fn(fresh_conn, *args, **kwargs)

	return wrapper


@database_reconnect
def init_schema(conn):
	with conn.cursor() as cur:
		cur.execute(_SCHEMA_PATH.read_text())


@database_reconnect
def upsert_paper(conn, paper):
	paper_id = paper['paperId']
	doi = (paper.get('externalIds') or {}).get('DOI')
	pdf_url = (paper.get('openAccessPdf') or {}).get('url')
	with conn.transaction(), conn.cursor() as cur:
		cur.execute(
			_UPSERT_PAPER_SQL,
			(
				paper_id,
				paper.get('title'),
				paper.get('abstract'),
				paper.get('year'),
				paper.get('citationCount'),
				paper.get('url'),
				doi,
				pdf_url,
			),
		)
		was_inserted = cur.fetchone()[0]
		cur.execute('DELETE FROM paper_authors WHERE paper_id = %s', (paper_id,))
		for position, author in enumerate(paper.get('authors') or []):
			name = author.get('name') if isinstance(author, dict) else None
			if not name:
				continue
			cur.execute(
				'INSERT INTO paper_authors (paper_id, position, name) VALUES (%s, %s, %s)', (paper_id, position, name)
			)
	return was_inserted


def paper_exists(conn, paper_id):
	with conn.cursor() as cur:
		cur.execute('SELECT 1 FROM papers WHERE paper_id = %s', (paper_id,))
		return cur.fetchone() is not None


@database_reconnect
def start_run(conn, papers_fetched):
	with conn.cursor() as cur:
		cur.execute('INSERT INTO runs (papers_fetched) VALUES (%s) RETURNING id', (papers_fetched,))
		return cur.fetchone()[0]


@database_reconnect
def finish_run(conn, run_id, papers_kept, queries_attempted, queries_errored):
	with conn.cursor() as cur:
		cur.execute(
			'UPDATE runs SET finished_at = NOW(), papers_kept = %s, queries_attempted = %s, queries_errored = %s '
			'WHERE id = %s',
			(papers_kept, queries_attempted, queries_errored, run_id),
		)


def needs_scoring(conn, paper_ids):
	if not paper_ids:
		return set()
	with conn.cursor() as cur:
		cur.execute('SELECT paper_id FROM papers WHERE paper_id = ANY(%s) AND scored_at IS NULL', (list(paper_ids),))
		return {row[0] for row in cur.fetchall()}


def get_summary(conn, paper_id):
	with conn.cursor() as cur:
		cur.execute(
			'SELECT methodology, findings, relevance, limitations, model_version FROM summaries WHERE paper_id = %s',
			(paper_id,),
		)
		row = cur.fetchone()
	if row is None:
		return None
	return dict(zip(_SUMMARY_COLUMNS, row, strict=True))


@database_reconnect
def upsert_summary(conn, paper_id, fields, model_version):
	with conn.cursor() as cur:
		cur.execute(
			_UPSERT_SUMMARY_SQL,
			(
				paper_id,
				fields.get('methodology'),
				fields.get('findings'),
				fields.get('relevance'),
				fields.get('limitations'),
				model_version,
			),
		)


@database_reconnect
def mark_scoring_results(conn, attempted, responded):
	if not attempted:
		return
	attempted_list = list(attempted)
	responded_list = list(responded)
	with conn.transaction(), conn.cursor() as cur:
		cur.execute('UPDATE papers SET score_attempts = score_attempts + 1 WHERE paper_id = ANY(%s)', (attempted_list,))
		if responded_list:
			cur.execute('UPDATE papers SET scored_at = NOW() WHERE paper_id = ANY(%s)', (responded_list,))


# -- review writes (append-only event log; readers select the latest row) --


def insert_rating(conn, paper_id, rating):
	with conn.cursor() as cur:
		cur.execute('INSERT INTO ratings (paper_id, rating) VALUES (%s, %s)', (paper_id, rating))


def insert_summary_feedback(conn, paper_id, field, rating=None, correction=None):
	with conn.cursor() as cur:
		cur.execute(
			'INSERT INTO summary_feedback (paper_id, field, rating, correction) VALUES (%s, %s, %s, %s)',
			(paper_id, field, rating, correction),
		)


def get_latest_rating(conn, paper_id):
	with conn.cursor() as cur:
		cur.execute(
			'SELECT rating FROM ratings WHERE paper_id = %s ORDER BY created_at DESC, id DESC LIMIT 1', (paper_id,)
		)
		row = cur.fetchone()
	return row[0] if row else None


def get_latest_field_feedback(conn, paper_id):
	# DISTINCT ON returns the latest row per field; the ORDER BY drives which wins.
	with conn.cursor() as cur:
		cur.execute(
			'SELECT DISTINCT ON (field) field, rating, correction '
			'FROM summary_feedback WHERE paper_id = %s '
			'ORDER BY field, created_at DESC, id DESC',
			(paper_id,),
		)
		rows = cur.fetchall()
	return {field: {'rating': rating, 'correction': correction} for field, rating, correction in rows}


# -- search and history reads --


_SEARCH_PAPERS_COLUMNS = (
	'paper_id',
	'title',
	'abstract',
	'year',
	'citation_count',
	'url',
	'doi',
	'pdf_url',
	'fetched_at',
	'authors',
	'methodology',
	'findings',
	'relevance',
	'limitations',
	'latest_rating',
)

_SEARCH_PAPERS_SQL = """
SELECT
	papers.paper_id,
	papers.title,
	papers.abstract,
	papers.year,
	papers.citation_count,
	papers.url,
	papers.doi,
	papers.pdf_url,
	papers.fetched_at,
	COALESCE(
		(SELECT array_agg(name ORDER BY position) FROM paper_authors WHERE paper_id = papers.paper_id),
		ARRAY[]::TEXT[]
	) AS authors,
	summaries.methodology,
	summaries.findings,
	summaries.relevance,
	summaries.limitations,
	(SELECT rating FROM ratings WHERE paper_id = papers.paper_id ORDER BY created_at DESC, id DESC LIMIT 1) AS latest_rating
FROM papers
LEFT JOIN summaries ON papers.paper_id = summaries.paper_id
{where_clause}
ORDER BY papers.fetched_at DESC
LIMIT %s OFFSET %s
"""

_LIST_RUNS_COLUMNS = (
	'id',
	'started_at',
	'finished_at',
	'papers_fetched',
	'papers_kept',
	'queries_attempted',
	'queries_errored',
)


def _build_search_filters(q, date_range):
	since, until = date_range if date_range else DateRange()
	clauses = []
	params = []
	if q:
		clauses.append('(papers.title ILIKE %s OR papers.abstract ILIKE %s)')
		like = f'%{q}%'
		params.extend([like, like])
	if since is not None:
		clauses.append('papers.fetched_at >= %s')
		params.append(since)
	if until is not None:
		clauses.append('papers.fetched_at <= %s')
		params.append(until)
	return clauses, params


def search_papers(conn, q=None, date_range=None, limit=20, offset=0):
	# date_range is a DateRange (or a (since, until) tuple); either bound may be None.
	clauses, params = _build_search_filters(q, date_range)
	where_clause = ' WHERE ' + ' AND '.join(clauses) if clauses else ''
	sql = _SEARCH_PAPERS_SQL.format(where_clause=where_clause)
	params.extend([limit, offset])
	with conn.cursor() as cur:
		cur.execute(sql, params)
		rows = cur.fetchall()
	return [dict(zip(_SEARCH_PAPERS_COLUMNS, row, strict=True)) for row in rows]


def count_papers(conn, q=None, date_range=None):
	clauses, params = _build_search_filters(q, date_range)
	where_clause = ' WHERE ' + ' AND '.join(clauses) if clauses else ''
	sql = f'SELECT COUNT(*) FROM papers{where_clause}'
	with conn.cursor() as cur:
		cur.execute(sql, params)
		return cur.fetchone()[0]


def list_runs(conn, limit=50):
	with conn.cursor() as cur:
		cur.execute(
			'SELECT id, started_at, finished_at, papers_fetched, papers_kept, queries_attempted, queries_errored '
			'FROM runs ORDER BY started_at DESC LIMIT %s',
			(limit,),
		)
		rows = cur.fetchall()
	return [dict(zip(_LIST_RUNS_COLUMNS, row, strict=True)) for row in rows]
