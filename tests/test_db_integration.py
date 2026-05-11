import os

import pytest

import db

_DATABASE_URL_TEST = os.environ.get('DATABASE_URL_TEST')

# `not _DATABASE_URL_TEST` covers both None and empty string. GitHub Actions sets
# missing secrets to '' (notably on pull_request runs from forks), so an `is None`
# check would let the tests run with an empty URL and fail at db.connect('').
pytestmark = [
	pytest.mark.integration,
	pytest.mark.skipif(
		not _DATABASE_URL_TEST, reason='DATABASE_URL_TEST not set; integration tests require a live Postgres branch'
	),
]


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
	db.finish_run(conn, run_id, papers_kept=7, queries_attempted=0, queries_errored=0)
	with conn.cursor() as cur:
		cur.execute('SELECT finished_at, papers_kept FROM runs WHERE id = %s', (run_id,))
		finished_at, papers_kept = cur.fetchone()
	assert finished_at is not None
	assert papers_kept == 7


def test_finish_run_records_query_outcomes(conn):
	run_id = db.start_run(conn, papers_fetched=42)
	db.finish_run(conn, run_id, papers_kept=3, queries_attempted=8, queries_errored=2)
	with conn.cursor() as cur:
		cur.execute('SELECT queries_attempted, queries_errored FROM runs WHERE id = %s', (run_id,))
		assert cur.fetchone() == (8, 2)


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


# -- needs_scoring --


def test_needs_scoring_returns_paper_ids_with_null_scored_at(conn):
	db.upsert_paper(conn, {'paperId': 'p1'})
	db.upsert_paper(conn, {'paperId': 'p2'})
	db.upsert_paper(conn, {'paperId': 'p3'})
	# p2 is already scored.
	db.mark_scoring_results(conn, attempted=['p2'], responded={'p2'})
	assert db.needs_scoring(conn, ['p1', 'p2', 'p3']) == {'p1', 'p3'}


def test_needs_scoring_skips_paper_ids_not_in_the_input_list(conn):
	db.upsert_paper(conn, {'paperId': 'p1'})
	db.upsert_paper(conn, {'paperId': 'p2'})
	# Only ask about p1.
	assert db.needs_scoring(conn, ['p1']) == {'p1'}


def test_needs_scoring_returns_empty_when_input_list_is_empty(conn):
	db.upsert_paper(conn, {'paperId': 'p1'})
	assert db.needs_scoring(conn, []) == set()


def test_needs_scoring_returns_empty_when_no_matching_rows_exist(conn):
	assert db.needs_scoring(conn, ['never-upserted']) == set()


# -- mark_scoring_results --


def test_mark_scoring_results_sets_scored_at_for_responded_papers(conn):
	db.upsert_paper(conn, {'paperId': 'p1'})
	db.upsert_paper(conn, {'paperId': 'p2'})
	db.mark_scoring_results(conn, attempted=['p1', 'p2'], responded={'p1'})
	with conn.cursor() as cur:
		cur.execute('SELECT paper_id, scored_at FROM papers WHERE paper_id IN (%s, %s) ORDER BY paper_id', ('p1', 'p2'))
		rows = cur.fetchall()
	# p1 was responded: scored_at set. p2 attempted but not responded: scored_at still NULL.
	assert rows[0][0] == 'p1' and rows[0][1] is not None
	assert rows[1][0] == 'p2' and rows[1][1] is None


def test_mark_scoring_results_increments_score_attempts_for_all_attempted(conn):
	db.upsert_paper(conn, {'paperId': 'p1'})
	db.upsert_paper(conn, {'paperId': 'p2'})
	db.mark_scoring_results(conn, attempted=['p1', 'p2'], responded=set())
	with conn.cursor() as cur:
		cur.execute(
			'SELECT paper_id, score_attempts FROM papers WHERE paper_id IN (%s, %s) ORDER BY paper_id', ('p1', 'p2')
		)
		assert cur.fetchall() == [('p1', 1), ('p2', 1)]


def test_mark_scoring_results_accumulates_attempts_across_runs(conn):
	db.upsert_paper(conn, {'paperId': 'p1'})
	db.mark_scoring_results(conn, attempted=['p1'], responded=set())
	db.mark_scoring_results(conn, attempted=['p1'], responded=set())
	db.mark_scoring_results(conn, attempted=['p1'], responded={'p1'})
	with conn.cursor() as cur:
		cur.execute('SELECT score_attempts, scored_at FROM papers WHERE paper_id = %s', ('p1',))
		attempts, scored_at = cur.fetchone()
	assert attempts == 3
	assert scored_at is not None


def test_mark_scoring_results_noops_on_empty_attempted(conn):
	db.upsert_paper(conn, {'paperId': 'p1'})
	db.mark_scoring_results(conn, attempted=[], responded=set())
	with conn.cursor() as cur:
		cur.execute('SELECT score_attempts, scored_at FROM papers WHERE paper_id = %s', ('p1',))
		assert cur.fetchone() == (0, None)


# -- upsert_summary timestamp semantics --


def test_upsert_summary_preserves_created_at_across_re_upsert(conn):
	# A second upsert for the same paper_id must keep the original created_at while
	# advancing updated_at. Verifies the ON CONFLICT clause does not overwrite the
	# creation timestamp.
	db.upsert_paper(conn, {'paperId': 'p1'})

	db.upsert_summary(conn, 'p1', {'methodology': 'first'}, 'v1')
	with conn.cursor() as cur:
		cur.execute('SELECT created_at, updated_at FROM summaries WHERE paper_id = %s', ('p1',))
		first_created, first_updated = cur.fetchone()

	db.upsert_summary(conn, 'p1', {'methodology': 'second'}, 'v2')
	with conn.cursor() as cur:
		cur.execute('SELECT created_at, updated_at FROM summaries WHERE paper_id = %s', ('p1',))
		second_created, second_updated = cur.fetchone()

	assert second_created == first_created
	assert second_updated >= first_updated


# -- orphan-retry acceptance: Gemini-failed papers come back on the next run --


def test_paper_with_failed_gemini_run_is_retried_on_next_run(conn):
	# Run 1: paper is upserted and attempted, but Gemini returned nothing usable
	# (responded set is empty). scored_at stays NULL.
	db.upsert_paper(conn, {'paperId': 'p1', 'title': 'T'})
	db.mark_scoring_results(conn, attempted=['p1'], responded=set())

	# needs_scoring still returns p1 because scored_at IS NULL.
	assert db.needs_scoring(conn, ['p1']) == {'p1'}

	# Run 2: same paper, Gemini now responds. scored_at gets set.
	db.upsert_paper(conn, {'paperId': 'p1', 'title': 'T'})  # citation refresh path
	db.mark_scoring_results(conn, attempted=['p1'], responded={'p1'})

	# Run 3: needs_scoring no longer returns it.
	assert db.needs_scoring(conn, ['p1']) == set()
	with conn.cursor() as cur:
		cur.execute('SELECT score_attempts FROM papers WHERE paper_id = %s', ('p1',))
		assert cur.fetchone()[0] == 2  # incremented once per run
