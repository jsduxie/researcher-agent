import os

import pytest

import db

_DATABASE_URL_TEST = os.environ.get('DATABASE_URL_TEST')

pytestmark = pytest.mark.skipif(
	_DATABASE_URL_TEST is None, reason='DATABASE_URL_TEST not set; integration tests require a live Postgres branch'
)


@pytest.fixture
def conn():
	c = db.connect(_DATABASE_URL_TEST)
	try:
		db.init_schema(c)
		with c.cursor() as cur:
			cur.execute(
				'TRUNCATE TABLE summary_feedback, ratings, summaries, paper_authors, runs, papers RESTART IDENTITY CASCADE'
			)
		yield c
	finally:
		c.close()


def _read_paper(conn, paper_id):
	with conn.cursor() as cur:
		cur.execute(
			'SELECT title, abstract, year, citation_count, url, doi, pdf_url FROM papers WHERE paper_id = %s',
			(paper_id,),
		)
		return cur.fetchone()


def _read_authors(conn, paper_id):
	with conn.cursor() as cur:
		cur.execute('SELECT position, name FROM paper_authors WHERE paper_id = %s ORDER BY position', (paper_id,))
		return cur.fetchall()


# -- schema --


def test_init_schema_is_idempotent(conn):
	# Fixture already ran init_schema once. Calling again must not raise.
	db.init_schema(conn)


# -- upsert_paper --


def test_upsert_paper_inserts_new_paper_and_returns_true(conn):
	paper = {
		'paperId': 'p1',
		'title': 'T',
		'abstract': 'A',
		'year': 2024,
		'citationCount': 5,
		'url': 'https://ss/p1',
		'externalIds': {'DOI': '10.1/p1'},
		'openAccessPdf': {'url': 'https://x/p1.pdf'},
	}
	assert db.upsert_paper(conn, paper) is True
	row = _read_paper(conn, 'p1')
	assert row == ('T', 'A', 2024, 5, 'https://ss/p1', '10.1/p1', 'https://x/p1.pdf')


def test_upsert_paper_returns_false_on_second_call_for_same_id(conn):
	db.upsert_paper(conn, {'paperId': 'p1', 'title': 'first'})
	assert db.upsert_paper(conn, {'paperId': 'p1', 'title': 'second'}) is False


def test_upsert_paper_refreshes_citation_count_on_update(conn):
	db.upsert_paper(conn, {'paperId': 'p1', 'citationCount': 5})
	db.upsert_paper(conn, {'paperId': 'p1', 'citationCount': 15})
	row = _read_paper(conn, 'p1')
	assert row[3] == 15  # citation_count


def test_upsert_paper_refreshes_title_on_update(conn):
	db.upsert_paper(conn, {'paperId': 'p1', 'title': 'first'})
	db.upsert_paper(conn, {'paperId': 'p1', 'title': 'second'})
	row = _read_paper(conn, 'p1')
	assert row[0] == 'second'


def test_upsert_paper_stores_authors_in_order(conn):
	paper = {'paperId': 'p1', 'authors': [{'name': 'Smith J.'}, {'name': 'Doe A.'}, {'name': 'Brown C.'}]}
	db.upsert_paper(conn, paper)
	assert _read_authors(conn, 'p1') == [(0, 'Smith J.'), (1, 'Doe A.'), (2, 'Brown C.')]


def test_upsert_paper_replaces_authors_on_update(conn):
	db.upsert_paper(conn, {'paperId': 'p1', 'authors': [{'name': 'A'}, {'name': 'B'}]})
	db.upsert_paper(conn, {'paperId': 'p1', 'authors': [{'name': 'C'}]})
	assert _read_authors(conn, 'p1') == [(0, 'C')]


def test_upsert_paper_skips_authors_with_missing_name(conn):
	paper = {'paperId': 'p1', 'authors': [{'name': 'A'}, {}, {'name': 'B'}]}
	db.upsert_paper(conn, paper)
	assert _read_authors(conn, 'p1') == [(0, 'A'), (2, 'B')]


def test_upsert_paper_with_no_authors_leaves_paper_authors_empty(conn):
	db.upsert_paper(conn, {'paperId': 'p1', 'title': 'T'})
	assert _read_authors(conn, 'p1') == []


def test_upsert_paper_sets_updated_at_more_recent_on_second_call(conn):
	db.upsert_paper(conn, {'paperId': 'p1'})
	with conn.cursor() as cur:
		cur.execute('SELECT updated_at FROM papers WHERE paper_id = %s', ('p1',))
		first = cur.fetchone()[0]
	db.upsert_paper(conn, {'paperId': 'p1', 'title': 'updated'})
	with conn.cursor() as cur:
		cur.execute('SELECT updated_at FROM papers WHERE paper_id = %s', ('p1',))
		second = cur.fetchone()[0]
	assert second >= first


# -- paper_exists --


def test_paper_exists_returns_false_for_unknown_id(conn):
	assert db.paper_exists(conn, 'never-seen') is False


def test_paper_exists_returns_true_after_upsert(conn):
	db.upsert_paper(conn, {'paperId': 'p1'})
	assert db.paper_exists(conn, 'p1') is True


# -- runs --


def test_start_run_inserts_row_with_papers_fetched_and_null_finished_at(conn):
	run_id = db.start_run(conn, papers_fetched=42)
	with conn.cursor() as cur:
		cur.execute('SELECT papers_fetched, finished_at, papers_kept FROM runs WHERE id = %s', (run_id,))
		assert cur.fetchone() == (42, None, None)


def test_finish_run_sets_finished_at_and_papers_kept(conn):
	run_id = db.start_run(conn, papers_fetched=42)
	db.finish_run(conn, run_id, papers_kept=7)
	with conn.cursor() as cur:
		cur.execute('SELECT finished_at, papers_kept FROM runs WHERE id = %s', (run_id,))
		finished_at, papers_kept = cur.fetchone()
	assert finished_at is not None
	assert papers_kept == 7


def test_each_start_run_gets_a_distinct_id(conn):
	id_a = db.start_run(conn, 1)
	id_b = db.start_run(conn, 2)
	assert id_a != id_b


# -- end-to-end dedup acceptance --


def test_two_consecutive_runs_second_does_not_re_score_existing_paper(conn):
	# Run 1: fetch and upsert 'p1'.
	first = db.upsert_paper(conn, {'paperId': 'p1', 'title': 'T'})
	assert first is True
	# Run 2: fetch same paper. Upsert returns False, so pipeline filters it out of scoring.
	second = db.upsert_paper(conn, {'paperId': 'p1', 'title': 'T'})
	assert second is False


# -- failure modes: real DB errors --


def test_connect_raises_for_unreachable_host():
	import psycopg

	with pytest.raises(psycopg.OperationalError):
		db.connect('postgresql://user:pass@127.0.0.1:1/db?connect_timeout=2')


# -- round-trip: writes match reads byte-for-byte --


def test_upsert_paper_round_trips_every_field(conn):
	paper = {
		'paperId': 'p1',
		'title': 'A round-trip paper',
		'abstract': 'Some abstract.',
		'year': 2024,
		'citationCount': 42,
		'url': 'https://ss/p1',
		'externalIds': {'DOI': '10.1/p1'},
		'openAccessPdf': {'url': 'https://x/p1.pdf'},
		'authors': [{'name': 'Smith J.'}, {'name': 'Doe A.'}],
	}
	db.upsert_paper(conn, paper)
	with conn.cursor() as cur:
		cur.execute(
			'SELECT paper_id, title, abstract, year, citation_count, url, doi, pdf_url FROM papers WHERE paper_id = %s',
			('p1',),
		)
		row = cur.fetchone()
	assert row == (
		'p1',
		'A round-trip paper',
		'Some abstract.',
		2024,
		42,
		'https://ss/p1',
		'10.1/p1',
		'https://x/p1.pdf',
	)
	assert _read_authors(conn, 'p1') == [(0, 'Smith J.'), (1, 'Doe A.')]


def test_write_then_paper_exists_then_read_is_consistent(conn):
	# Sequential round-trip across the three primary read/write functions.
	assert db.paper_exists(conn, 'p1') is False
	db.upsert_paper(conn, {'paperId': 'p1', 'title': 'T', 'citationCount': 5})
	assert db.paper_exists(conn, 'p1') is True
	with conn.cursor() as cur:
		cur.execute('SELECT title, citation_count FROM papers WHERE paper_id = %s', ('p1',))
		assert cur.fetchone() == ('T', 5)
