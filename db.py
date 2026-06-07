import functools
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import psycopg
from psycopg.types.json import Jsonb


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


_LIST_UNSCORED_SQL = """
SELECT
	papers.paper_id,
	papers.title,
	papers.abstract,
	papers.year,
	papers.citation_count,
	papers.url,
	papers.doi,
	papers.pdf_url,
	COALESCE(
		(SELECT array_agg(name ORDER BY position) FROM paper_authors WHERE paper_id = papers.paper_id),
		ARRAY[]::TEXT[]
	) AS authors
FROM papers
WHERE scored_at IS NULL AND score_attempts < %s AND NOT (paper_id = ANY(%s))
ORDER BY fetched_at DESC
LIMIT %s
"""


def list_unscored(conn, exclude_ids, max_attempts, limit):
	# Rows come back in the Semantic Scholar dict shape so swept papers flow through scoring, summarising and rendering unchanged.
	with conn.cursor() as cur:
		cur.execute(_LIST_UNSCORED_SQL, (max_attempts, list(exclude_ids), limit))
		rows = cur.fetchall()
	return [
		{
			'paperId': paper_id,
			'title': title,
			'abstract': abstract,
			'year': year,
			'citationCount': citation_count,
			'url': url,
			'externalIds': {'DOI': doi} if doi else {},
			'openAccessPdf': {'url': pdf_url} if pdf_url else None,
			'authors': [{'name': name} for name in authors],
		}
		for paper_id, title, abstract, year, citation_count, url, doi, pdf_url, authors in rows
	]


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


# -- embeddings (pgvector; serialised as '[f1,f2,...]' text so no pgvector client package is needed) --


@database_reconnect
def set_paper_embedding(conn, paper_id, embedding):
	serialised = '[' + ','.join(str(v) for v in embedding) + ']'
	with conn.cursor() as cur:
		cur.execute('UPDATE papers SET embedding = %s::vector WHERE paper_id = %s', (serialised, paper_id))


def get_paper_embedding(conn, paper_id):
	with conn.cursor() as cur:
		cur.execute('SELECT embedding::text FROM papers WHERE paper_id = %s', (paper_id,))
		row = cur.fetchone()
	if row is None or row[0] is None:
		return None
	return [float(v) for v in row[0].strip('[]').split(',')]


def list_papers_missing_embeddings(conn):
	# Backfill source: papers written before the vector column existed have no embedding yet.
	with conn.cursor() as cur:
		cur.execute('SELECT paper_id, title, abstract FROM papers WHERE embedding IS NULL ORDER BY fetched_at')
		rows = cur.fetchall()
	return [{'paper_id': paper_id, 'title': title, 'abstract': abstract} for paper_id, title, abstract in rows]


# -- review writes (append-only event log; readers select the latest row) --


@database_reconnect
def insert_rating(conn, paper_id, rating):
	with conn.cursor() as cur:
		cur.execute('INSERT INTO ratings (paper_id, rating) VALUES (%s, %s)', (paper_id, rating))


@database_reconnect
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


# -- few-shot retrieval (pgvector cosine similarity over papers carrying feedback) --


_SIMILAR_RATED_PAPERS_SQL = """
SELECT
	papers.paper_id,
	papers.title,
	papers.abstract,
	(SELECT rating FROM ratings WHERE paper_id = papers.paper_id ORDER BY created_at DESC, id DESC LIMIT 1) AS latest_rating
FROM papers
WHERE papers.embedding IS NOT NULL
	AND EXISTS (SELECT 1 FROM ratings WHERE ratings.paper_id = papers.paper_id)
	AND papers.paper_id IS DISTINCT FROM %s
ORDER BY papers.embedding <=> %s::vector
LIMIT %s
"""

_SIMILAR_RATED_PAPERS_COLUMNS = ('paper_id', 'title', 'abstract', 'latest_rating')


def similar_rated_papers(conn, embedding, limit, exclude_id=None):
	# Few-shot calibration source: rated papers ranked by cosine distance (<=>) to the query embedding. IS DISTINCT FROM keeps a paper out of its own example set while a NULL exclude_id excludes nothing.
	serialised = '[' + ','.join(str(v) for v in embedding) + ']'
	with conn.cursor() as cur:
		cur.execute(_SIMILAR_RATED_PAPERS_SQL, (exclude_id, serialised, limit))
		rows = cur.fetchall()
	return [dict(zip(_SIMILAR_RATED_PAPERS_COLUMNS, row, strict=True)) for row in rows]


_SIMILAR_FEEDBACK_PAPERS_SQL = """
SELECT
	papers.paper_id,
	papers.title,
	papers.abstract
FROM papers
WHERE papers.embedding IS NOT NULL
	AND EXISTS (SELECT 1 FROM summary_feedback WHERE summary_feedback.paper_id = papers.paper_id)
	AND papers.paper_id IS DISTINCT FROM %s
ORDER BY papers.embedding <=> %s::vector
LIMIT %s
"""

_SIMILAR_FEEDBACK_PAPERS_COLUMNS = ('paper_id', 'title', 'abstract')


def similar_feedback_papers(conn, embedding, limit, exclude_id=None):
	# Same ranking as similar_rated_papers but gated on summary_feedback; attaches the latest per-field rating+correction so the summariser can pick good (>=4) and bad (<=2) shapes.
	serialised = '[' + ','.join(str(v) for v in embedding) + ']'
	with conn.cursor() as cur:
		cur.execute(_SIMILAR_FEEDBACK_PAPERS_SQL, (exclude_id, serialised, limit))
		rows = cur.fetchall()
	papers = [dict(zip(_SIMILAR_FEEDBACK_PAPERS_COLUMNS, row, strict=True)) for row in rows]
	for paper in papers:
		paper['feedback'] = get_latest_field_feedback(conn, paper['paper_id'])
	return papers


def count_ratings(conn):
	with conn.cursor() as cur:
		cur.execute('SELECT COUNT(*) FROM ratings')
		return cur.fetchone()[0]


def count_summary_feedback(conn):
	with conn.cursor() as cur:
		cur.execute('SELECT COUNT(*) FROM summary_feedback')
		return cur.fetchone()[0]


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

_LIST_PAPERS_FOR_RUN_COLUMNS = (*_SEARCH_PAPERS_COLUMNS, 'was_new')

_LIST_PAPERS_FOR_RUN_SQL = """
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
	(SELECT rating FROM ratings WHERE paper_id = papers.paper_id ORDER BY created_at DESC, id DESC LIMIT 1) AS latest_rating,
	runs_papers.was_new
FROM papers
JOIN runs_papers ON runs_papers.paper_id = papers.paper_id
LEFT JOIN summaries ON papers.paper_id = summaries.paper_id
WHERE runs_papers.run_id = %s
ORDER BY papers.fetched_at DESC
"""


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


@database_reconnect
def record_run_papers(conn, run_id, entries):
	# entries is an iterable of (paper_id, was_new) pairs. was_new comes from upsert_paper's RETURNING (xmax = 0) check so the History page can distinguish fresh fetches from re-encounters.
	rows = [(run_id, paper_id, was_new) for paper_id, was_new in entries]
	if not rows:
		return
	with conn.transaction(), conn.cursor() as cur:
		cur.executemany(
			'INSERT INTO runs_papers (run_id, paper_id, was_new) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING', rows
		)


def list_papers_for_run(conn, run_id):
	with conn.cursor() as cur:
		cur.execute(_LIST_PAPERS_FOR_RUN_SQL, (run_id,))
		rows = cur.fetchall()
	return [dict(zip(_LIST_PAPERS_FOR_RUN_COLUMNS, row, strict=True)) for row in rows]


# -- app config (per-key JSONB) + audit history --


_LIST_CONFIG_HISTORY_COLUMNS = ('id', 'key', 'value', 'changed_at', 'changed_by')

_UPSERT_APP_CONFIG_SQL = """
INSERT INTO app_config (key, value, updated_at, updated_by)
VALUES (%s, %s, NOW(), %s)
ON CONFLICT (key) DO UPDATE
SET value = EXCLUDED.value,
    updated_at = NOW(),
    updated_by = EXCLUDED.updated_by
"""

_INSERT_APP_CONFIG_HISTORY_SQL = (
	'INSERT INTO app_config_history (key, value, changed_at, changed_by) VALUES (%s, %s, NOW(), %s)'
)


def load_config(conn):
	with conn.cursor() as cur:
		cur.execute('SELECT key, value FROM app_config')
		return dict(cur.fetchall())


@database_reconnect
def update_config(conn, updates, by=None):
	# Upserts every key and appends one history row per key in a single transaction so the audit log cannot drift from app_config.
	if not updates:
		return
	with conn.transaction(), conn.cursor() as cur:
		for key, value in updates.items():
			payload = Jsonb(value)
			cur.execute(_UPSERT_APP_CONFIG_SQL, (key, payload, by))
			cur.execute(_INSERT_APP_CONFIG_HISTORY_SQL, (key, payload, by))


def list_config_history(conn, key=None, limit=20):
	with conn.cursor() as cur:
		if key is None:
			cur.execute(
				'SELECT id, key, value, changed_at, changed_by FROM app_config_history '
				'ORDER BY changed_at DESC, id DESC LIMIT %s',
				(limit,),
			)
		else:
			cur.execute(
				'SELECT id, key, value, changed_at, changed_by FROM app_config_history '
				'WHERE key = %s ORDER BY changed_at DESC, id DESC LIMIT %s',
				(key, limit),
			)
		rows = cur.fetchall()
	return [dict(zip(_LIST_CONFIG_HISTORY_COLUMNS, row, strict=True)) for row in rows]
