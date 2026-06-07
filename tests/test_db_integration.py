import os
from datetime import UTC

import pytest

import db

_DATABASE_URL_TEST = os.environ.get('DATABASE_URL_TEST')

# `not _DATABASE_URL_TEST` covers None and empty string. GitHub Actions sets missing secrets to '' on fork PRs, so an `is None` check would run on '' and fail.
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
				'TRUNCATE TABLE summary_feedback, ratings, summaries, paper_authors, runs, papers, '
				'app_config_history, app_config RESTART IDENTITY CASCADE'
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


# -- list_unscored --


def test_list_unscored_sweeps_only_eligible_papers_in_api_shape(conn):
	db.upsert_paper(conn, {'paperId': 'unscored', 'title': 't', 'authors': [{'name': 'A One'}]})
	db.upsert_paper(conn, {'paperId': 'scored'})
	db.upsert_paper(conn, {'paperId': 'exhausted'})
	db.upsert_paper(conn, {'paperId': 'current-fetch'})
	db.mark_scoring_results(conn, attempted=['scored'], responded={'scored'})
	for _ in range(3):
		db.mark_scoring_results(conn, attempted=['exhausted'], responded=set())

	papers = db.list_unscored(conn, exclude_ids=['current-fetch'], max_attempts=3, limit=10)

	assert [p['paperId'] for p in papers] == ['unscored']
	assert papers[0]['title'] == 't'
	assert papers[0]['authors'] == [{'name': 'A One'}]


def test_list_unscored_respects_limit(conn):
	for i in range(3):
		db.upsert_paper(conn, {'paperId': f'p{i}'})
	assert len(db.list_unscored(conn, exclude_ids=[], max_attempts=3, limit=2)) == 2


# -- embeddings --


def test_paper_embedding_round_trips_through_pgvector(conn):
	db.upsert_paper(conn, {'paperId': 'p1'})
	vector = [round(i / 384, 6) for i in range(384)]
	db.set_paper_embedding(conn, 'p1', vector)
	stored = db.get_paper_embedding(conn, 'p1')
	assert stored is not None
	assert len(stored) == 384
	# pgvector stores float4, so compare at its precision rather than exact equality.
	assert all(abs(a - b) < 1e-5 for a, b in zip(stored, vector, strict=True))


def test_get_paper_embedding_none_before_any_embedding_written(conn):
	db.upsert_paper(conn, {'paperId': 'p1'})
	assert db.get_paper_embedding(conn, 'p1') is None


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
	# A second upsert for the same paper_id must keep created_at while advancing updated_at. Verifies the ON CONFLICT clause doesn't overwrite creation.
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
	# Run 1: paper is upserted and attempted, but Gemini returned nothing usable (responded set is empty). scored_at stays NULL.
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


# -- ratings (append-only event log) --


def test_insert_rating_persists_row(conn):
	db.upsert_paper(conn, {'paperId': 'p1'})
	db.insert_rating(conn, 'p1', 4)
	with conn.cursor() as cur:
		cur.execute('SELECT paper_id, rating FROM ratings WHERE paper_id = %s', ('p1',))
		assert cur.fetchone() == ('p1', 4)


def test_insert_rating_appends_rather_than_replaces(conn):
	# Reviews are a log so the user can change their mind without erasing prior signal.
	db.upsert_paper(conn, {'paperId': 'p1'})
	db.insert_rating(conn, 'p1', 2)
	db.insert_rating(conn, 'p1', 5)
	with conn.cursor() as cur:
		cur.execute('SELECT rating FROM ratings WHERE paper_id = %s ORDER BY id', ('p1',))
		assert [r[0] for r in cur.fetchall()] == [2, 5]


def test_insert_rating_rejects_value_outside_one_to_five(conn):
	import psycopg

	db.upsert_paper(conn, {'paperId': 'p1'})
	with pytest.raises(psycopg.errors.CheckViolation):
		db.insert_rating(conn, 'p1', 0)


def test_get_latest_rating_returns_most_recent_value(conn):
	db.upsert_paper(conn, {'paperId': 'p1'})
	db.insert_rating(conn, 'p1', 3)
	db.insert_rating(conn, 'p1', 1)
	db.insert_rating(conn, 'p1', 5)
	assert db.get_latest_rating(conn, 'p1') == 5


def test_get_latest_rating_returns_none_when_no_rows(conn):
	db.upsert_paper(conn, {'paperId': 'p1'})
	assert db.get_latest_rating(conn, 'p1') is None


# -- summary_feedback (per-field rating + correction) --


def test_insert_summary_feedback_persists_rating_and_correction(conn):
	db.upsert_paper(conn, {'paperId': 'p1'})
	db.insert_summary_feedback(conn, 'p1', 'methodology', rating=4, correction='more detail on dataset')
	with conn.cursor() as cur:
		cur.execute('SELECT paper_id, field, rating, correction FROM summary_feedback WHERE paper_id = %s', ('p1',))
		assert cur.fetchone() == ('p1', 'methodology', 4, 'more detail on dataset')


def test_insert_summary_feedback_allows_correction_only(conn):
	# Optional rating: a user can leave a comment without scoring the section.
	db.upsert_paper(conn, {'paperId': 'p1'})
	db.insert_summary_feedback(conn, 'p1', 'findings', correction='cite Smith 2024')
	with conn.cursor() as cur:
		cur.execute('SELECT rating, correction FROM summary_feedback WHERE paper_id = %s', ('p1',))
		assert cur.fetchone() == (None, 'cite Smith 2024')


def test_insert_summary_feedback_rejects_rating_outside_one_to_five(conn):
	import psycopg

	db.upsert_paper(conn, {'paperId': 'p1'})
	with pytest.raises(psycopg.errors.CheckViolation):
		db.insert_summary_feedback(conn, 'p1', 'methodology', rating=6)


def test_get_latest_field_feedback_returns_one_row_per_field(conn):
	db.upsert_paper(conn, {'paperId': 'p1'})
	db.insert_summary_feedback(conn, 'p1', 'methodology', rating=2)
	db.insert_summary_feedback(conn, 'p1', 'methodology', rating=4, correction='better')
	db.insert_summary_feedback(conn, 'p1', 'findings', rating=5)
	result = db.get_latest_field_feedback(conn, 'p1')
	assert result == {
		'methodology': {'rating': 4, 'correction': 'better'},
		'findings': {'rating': 5, 'correction': None},
	}


def test_get_latest_field_feedback_returns_empty_dict_when_no_rows(conn):
	db.upsert_paper(conn, {'paperId': 'p1'})
	assert db.get_latest_field_feedback(conn, 'p1') == {}


# -- few-shot retrieval (real pgvector ordering + latest-row semantics) --


def _vec(*head):
	# papers.embedding is vector(384), so every embedding (query or stored) must have 384 dims; only the leading components carry the test's distance signal.
	return list(head) + [0.0] * (384 - len(head))


def _embed(conn, paper_id, vector):
	db.set_paper_embedding(conn, paper_id, vector)


def test_similar_rated_papers_orders_nearest_first_by_cosine_distance(conn):
	# query is closest to near, then mid, then far; all three are rated and embedded.
	query = _vec(1.0, 0.0, 0.0)
	db.upsert_paper(conn, {'paperId': 'near'})
	db.upsert_paper(conn, {'paperId': 'mid'})
	db.upsert_paper(conn, {'paperId': 'far'})
	_embed(conn, 'near', _vec(0.99, 0.1, 0.0))
	_embed(conn, 'mid', _vec(0.5, 0.5, 0.0))
	_embed(conn, 'far', _vec(-1.0, 0.0, 0.0))
	for pid in ('near', 'mid', 'far'):
		db.insert_rating(conn, pid, 3)
	result = db.similar_rated_papers(conn, query, limit=3)
	assert [p['paper_id'] for p in result] == ['near', 'mid', 'far']


def test_similar_rated_papers_excludes_unrated_and_unembedded_papers(conn):
	query = _vec(1.0, 0.0, 0.0)
	db.upsert_paper(conn, {'paperId': 'rated'})
	db.upsert_paper(conn, {'paperId': 'embedded-no-rating'})
	db.upsert_paper(conn, {'paperId': 'rated-no-embedding'})
	_embed(conn, 'rated', _vec(1.0, 0.0, 0.0))
	_embed(conn, 'embedded-no-rating', _vec(1.0, 0.0, 0.0))
	db.insert_rating(conn, 'rated', 4)
	db.insert_rating(conn, 'rated-no-embedding', 5)
	result = db.similar_rated_papers(conn, query, limit=10)
	assert [p['paper_id'] for p in result] == ['rated']


def test_similar_rated_papers_excludes_given_paper_id(conn):
	query = _vec(1.0, 0.0, 0.0)
	db.upsert_paper(conn, {'paperId': 'self'})
	db.upsert_paper(conn, {'paperId': 'other'})
	_embed(conn, 'self', _vec(1.0, 0.0, 0.0))
	_embed(conn, 'other', _vec(0.9, 0.1, 0.0))
	db.insert_rating(conn, 'self', 3)
	db.insert_rating(conn, 'other', 4)
	result = db.similar_rated_papers(conn, query, limit=10, exclude_id='self')
	assert [p['paper_id'] for p in result] == ['other']


def test_similar_rated_papers_returns_latest_rating_per_paper(conn):
	db.upsert_paper(conn, {'paperId': 'p1', 'title': 'T', 'abstract': 'A'})
	_embed(conn, 'p1', _vec(1.0, 0.0, 0.0))
	db.insert_rating(conn, 'p1', 2)
	db.insert_rating(conn, 'p1', 5)
	(paper,) = db.similar_rated_papers(conn, _vec(1.0, 0.0, 0.0), limit=3)
	assert paper['title'] == 'T'
	assert paper['abstract'] == 'A'
	assert paper['latest_rating'] == 5


def test_similar_rated_papers_respects_limit(conn):
	for i in range(4):
		db.upsert_paper(conn, {'paperId': f'p{i}'})
		_embed(conn, f'p{i}', _vec(1.0, 0.0, 0.0))
		db.insert_rating(conn, f'p{i}', 3)
	assert len(db.similar_rated_papers(conn, _vec(1.0, 0.0, 0.0), limit=2)) == 2


def test_similar_feedback_papers_orders_nearest_first_and_attaches_latest_feedback(conn):
	query = _vec(1.0, 0.0, 0.0)
	db.upsert_paper(conn, {'paperId': 'near', 'title': 'Near'})
	db.upsert_paper(conn, {'paperId': 'far', 'title': 'Far'})
	_embed(conn, 'near', _vec(0.99, 0.1, 0.0))
	_embed(conn, 'far', _vec(-1.0, 0.0, 0.0))
	db.insert_summary_feedback(conn, 'near', 'methodology', rating=2)
	db.insert_summary_feedback(conn, 'near', 'methodology', rating=5, correction='clearer')
	db.insert_summary_feedback(conn, 'far', 'findings', rating=4)
	result = db.similar_feedback_papers(conn, query, limit=3)
	assert [p['paper_id'] for p in result] == ['near', 'far']
	# Latest-row semantics: methodology resolves to the second (rating 5) row.
	assert result[0]['feedback'] == {'methodology': {'rating': 5, 'correction': 'clearer'}}
	assert result[1]['feedback'] == {'findings': {'rating': 4, 'correction': None}}


def test_similar_feedback_papers_excludes_papers_without_feedback_or_embedding(conn):
	query = _vec(1.0, 0.0, 0.0)
	db.upsert_paper(conn, {'paperId': 'with-feedback'})
	db.upsert_paper(conn, {'paperId': 'embedded-no-feedback'})
	db.upsert_paper(conn, {'paperId': 'feedback-no-embedding'})
	_embed(conn, 'with-feedback', _vec(1.0, 0.0, 0.0))
	_embed(conn, 'embedded-no-feedback', _vec(1.0, 0.0, 0.0))
	db.insert_summary_feedback(conn, 'with-feedback', 'methodology', rating=4)
	db.insert_summary_feedback(conn, 'feedback-no-embedding', 'methodology', rating=4)
	result = db.similar_feedback_papers(conn, query, limit=10)
	assert [p['paper_id'] for p in result] == ['with-feedback']


def test_similar_feedback_papers_excludes_given_paper_id(conn):
	query = _vec(1.0, 0.0, 0.0)
	db.upsert_paper(conn, {'paperId': 'self'})
	db.upsert_paper(conn, {'paperId': 'other'})
	_embed(conn, 'self', _vec(1.0, 0.0, 0.0))
	_embed(conn, 'other', _vec(0.9, 0.1, 0.0))
	db.insert_summary_feedback(conn, 'self', 'methodology', rating=3)
	db.insert_summary_feedback(conn, 'other', 'findings', rating=4)
	result = db.similar_feedback_papers(conn, query, limit=10, exclude_id='self')
	assert [p['paper_id'] for p in result] == ['other']


def test_count_ratings_counts_every_appended_row(conn):
	db.upsert_paper(conn, {'paperId': 'p1'})
	db.upsert_paper(conn, {'paperId': 'p2'})
	assert db.count_ratings(conn) == 0
	db.insert_rating(conn, 'p1', 3)
	db.insert_rating(conn, 'p1', 5)
	db.insert_rating(conn, 'p2', 4)
	assert db.count_ratings(conn) == 3


def test_count_summary_feedback_counts_every_appended_row(conn):
	db.upsert_paper(conn, {'paperId': 'p1'})
	assert db.count_summary_feedback(conn) == 0
	db.insert_summary_feedback(conn, 'p1', 'methodology', rating=4)
	db.insert_summary_feedback(conn, 'p1', 'methodology', rating=2, correction='redo')
	assert db.count_summary_feedback(conn) == 2


# -- search_papers --


def test_search_papers_returns_all_when_no_filters(conn):
	db.upsert_paper(conn, {'paperId': 'p1', 'title': 'first'})
	db.upsert_paper(conn, {'paperId': 'p2', 'title': 'second'})
	result = db.search_papers(conn)
	assert {p['paper_id'] for p in result} == {'p1', 'p2'}


def test_search_papers_returns_dict_shape_with_all_columns(conn):
	db.upsert_paper(
		conn,
		{
			'paperId': 'p1',
			'title': 'T',
			'abstract': 'A',
			'year': 2024,
			'citationCount': 7,
			'url': 'https://u',
			'externalIds': {'DOI': '10.1/x'},
			'openAccessPdf': {'url': 'https://x.pdf'},
			'authors': [{'name': 'Smith J.'}, {'name': 'Doe A.'}],
		},
	)
	db.upsert_summary(conn, 'p1', {'methodology': 'm', 'findings': 'f', 'relevance': 'r', 'limitations': 'l'}, 'v1')
	(paper,) = db.search_papers(conn)
	assert paper['paper_id'] == 'p1'
	assert paper['title'] == 'T'
	assert paper['abstract'] == 'A'
	assert paper['year'] == 2024
	assert paper['citation_count'] == 7
	assert paper['url'] == 'https://u'
	assert paper['doi'] == '10.1/x'
	assert paper['pdf_url'] == 'https://x.pdf'
	assert paper['authors'] == ['Smith J.', 'Doe A.']
	assert paper['methodology'] == 'm'
	assert paper['findings'] == 'f'
	assert paper['relevance'] == 'r'
	assert paper['limitations'] == 'l'
	assert paper['latest_rating'] is None  # no rating recorded yet


def test_search_papers_latest_rating_reflects_most_recent_rating(conn):
	db.upsert_paper(conn, {'paperId': 'p1'})
	db.insert_rating(conn, 'p1', 3)
	db.insert_rating(conn, 'p1', 5)
	(paper,) = db.search_papers(conn)
	assert paper['latest_rating'] == 5


def test_search_papers_latest_rating_is_none_when_no_ratings(conn):
	db.upsert_paper(conn, {'paperId': 'p1'})
	(paper,) = db.search_papers(conn)
	assert paper['latest_rating'] is None


def test_search_papers_returns_papers_without_summaries(conn):
	# LEFT JOIN: papers with no summaries row still appear, with summary fields as None.
	db.upsert_paper(conn, {'paperId': 'p1', 'title': 'no summary yet'})
	(paper,) = db.search_papers(conn)
	assert paper['methodology'] is None
	assert paper['findings'] is None


def test_search_papers_returns_empty_authors_list_when_none_stored(conn):
	# array_agg over an empty subquery returns NULL; the COALESCE turns it into [].
	db.upsert_paper(conn, {'paperId': 'p1', 'title': 'lonely'})
	(paper,) = db.search_papers(conn)
	assert paper['authors'] == []


def test_search_papers_filters_by_query_against_title_and_abstract(conn):
	# ILIKE is plain substring with case folding; no synonym handling, so p1's title must literally contain the query.
	db.upsert_paper(conn, {'paperId': 'p1', 'title': 'BPD classification', 'abstract': 'unrelated'})
	db.upsert_paper(conn, {'paperId': 'p2', 'title': 'unrelated', 'abstract': 'discusses BPD detection'})
	db.upsert_paper(conn, {'paperId': 'p3', 'title': 'transformers', 'abstract': 'attention is all you need'})
	result = db.search_papers(conn, q='bpd')
	assert {p['paper_id'] for p in result} == {'p1', 'p2'}


def test_search_papers_query_is_case_insensitive(conn):
	db.upsert_paper(conn, {'paperId': 'p1', 'title': 'Borderline Personality Disorder'})
	assert {p['paper_id'] for p in db.search_papers(conn, q='borderline')} == {'p1'}


def test_search_papers_filters_by_date_range(conn):
	# fetched_at defaults to NOW(), so set it explicitly to test the filter.
	from datetime import datetime

	db.upsert_paper(conn, {'paperId': 'old', 'title': 'old'})
	db.upsert_paper(conn, {'paperId': 'new', 'title': 'new'})
	with conn.cursor() as cur:
		cur.execute('UPDATE papers SET fetched_at = %s WHERE paper_id = %s', (datetime(2026, 1, 1, tzinfo=UTC), 'old'))
		cur.execute('UPDATE papers SET fetched_at = %s WHERE paper_id = %s', (datetime(2026, 5, 1, tzinfo=UTC), 'new'))
	since = datetime(2026, 3, 1, tzinfo=UTC)
	until = datetime(2026, 6, 1, tzinfo=UTC)
	assert {p['paper_id'] for p in db.search_papers(conn, date_range=db.DateRange(since, until))} == {'new'}


def test_search_papers_orders_by_fetched_at_descending(conn):
	from datetime import datetime

	db.upsert_paper(conn, {'paperId': 'a'})
	db.upsert_paper(conn, {'paperId': 'b'})
	with conn.cursor() as cur:
		cur.execute('UPDATE papers SET fetched_at = %s WHERE paper_id = %s', (datetime(2026, 1, 1, tzinfo=UTC), 'a'))
		cur.execute('UPDATE papers SET fetched_at = %s WHERE paper_id = %s', (datetime(2026, 5, 1, tzinfo=UTC), 'b'))
	result = db.search_papers(conn)
	assert [p['paper_id'] for p in result] == ['b', 'a']


def test_search_papers_respects_limit_and_offset(conn):
	for i in range(5):
		db.upsert_paper(conn, {'paperId': f'p{i}'})
	page1 = db.search_papers(conn, limit=2, offset=0)
	page2 = db.search_papers(conn, limit=2, offset=2)
	page3 = db.search_papers(conn, limit=2, offset=4)
	assert len(page1) == 2
	assert len(page2) == 2
	assert len(page3) == 1
	# No overlap across pages.
	assert not ({p['paper_id'] for p in page1} & {p['paper_id'] for p in page2})


# -- count_papers --


def test_count_papers_matches_unfiltered_search(conn):
	for i in range(3):
		db.upsert_paper(conn, {'paperId': f'p{i}'})
	assert db.count_papers(conn) == 3


def test_count_papers_applies_query_filter(conn):
	db.upsert_paper(conn, {'paperId': 'p1', 'title': 'BPD detection'})
	db.upsert_paper(conn, {'paperId': 'p2', 'title': 'unrelated'})
	assert db.count_papers(conn, q='bpd') == 1


# -- list_runs --


def test_list_runs_returns_runs_latest_first(conn):
	r1 = db.start_run(conn, papers_fetched=10)
	r2 = db.start_run(conn, papers_fetched=20)
	r3 = db.start_run(conn, papers_fetched=30)
	result = db.list_runs(conn)
	# BIGSERIAL is monotonically increasing and started_at defaults to NOW(), so the latest run id is the latest run by start time.
	assert [r['id'] for r in result] == [r3, r2, r1]


def test_list_runs_returns_full_row_shape(conn):
	run_id = db.start_run(conn, papers_fetched=42)
	db.finish_run(conn, run_id, papers_kept=7, queries_attempted=8, queries_errored=1)
	(run,) = db.list_runs(conn)
	assert run['id'] == run_id
	assert run['papers_fetched'] == 42
	assert run['papers_kept'] == 7
	assert run['queries_attempted'] == 8
	assert run['queries_errored'] == 1
	assert run['started_at'] is not None
	assert run['finished_at'] is not None


def test_list_runs_respects_limit(conn):
	for _ in range(5):
		db.start_run(conn, papers_fetched=0)
	assert len(db.list_runs(conn, limit=3)) == 3


# -- runs_papers / list_papers_for_run --


def _upsert_paper_minimal(conn, paper_id, title='t'):
	db.upsert_paper(conn, {'paperId': paper_id, 'title': title})


def test_record_run_papers_attaches_papers_to_run(conn):
	_upsert_paper_minimal(conn, 'p1')
	_upsert_paper_minimal(conn, 'p2')
	run_id = db.start_run(conn, papers_fetched=2)

	db.record_run_papers(conn, run_id, [('p1', True), ('p2', True)])

	with conn.cursor() as cur:
		cur.execute('SELECT paper_id FROM runs_papers WHERE run_id = %s ORDER BY paper_id', (run_id,))
		assert [row[0] for row in cur.fetchall()] == ['p1', 'p2']


def test_record_run_papers_persists_the_was_new_flag(conn):
	_upsert_paper_minimal(conn, 'fresh')
	_upsert_paper_minimal(conn, 'repeat')
	run_id = db.start_run(conn, papers_fetched=2)

	db.record_run_papers(conn, run_id, [('fresh', True), ('repeat', False)])

	with conn.cursor() as cur:
		cur.execute('SELECT paper_id, was_new FROM runs_papers WHERE run_id = %s ORDER BY paper_id', (run_id,))
		assert cur.fetchall() == [('fresh', True), ('repeat', False)]


def test_record_run_papers_is_idempotent_within_a_run(conn):
	# Re-recording the same (run_id, paper_id) is a no-op; the PK + ON CONFLICT DO NOTHING absorbs duplicates from a retried run that crashed mid-write.
	_upsert_paper_minimal(conn, 'p1')
	run_id = db.start_run(conn, papers_fetched=1)

	db.record_run_papers(conn, run_id, [('p1', True)])
	db.record_run_papers(conn, run_id, [('p1', True)])

	with conn.cursor() as cur:
		cur.execute('SELECT COUNT(*) FROM runs_papers WHERE run_id = %s AND paper_id = %s', (run_id, 'p1'))
		assert cur.fetchone()[0] == 1


def test_record_run_papers_no_ops_on_empty_input(conn):
	run_id = db.start_run(conn, papers_fetched=0)
	db.record_run_papers(conn, run_id, [])
	with conn.cursor() as cur:
		cur.execute('SELECT COUNT(*) FROM runs_papers WHERE run_id = %s', (run_id,))
		assert cur.fetchone()[0] == 0


def test_record_run_papers_supports_same_paper_appearing_in_multiple_runs(conn):
	# A paper re-encountered in a later run must show up under both runs' rosters with was_new=False, which is the whole reason this commit picked a join table over the fetched_at window.
	_upsert_paper_minimal(conn, 'p1')
	run_a = db.start_run(conn, papers_fetched=1)
	run_b = db.start_run(conn, papers_fetched=1)

	db.record_run_papers(conn, run_a, [('p1', True)])
	db.record_run_papers(conn, run_b, [('p1', False)])

	with conn.cursor() as cur:
		cur.execute('SELECT run_id, was_new FROM runs_papers WHERE paper_id = %s ORDER BY run_id', ('p1',))
		assert cur.fetchall() == [(run_a, True), (run_b, False)]


def test_list_papers_for_run_returns_only_rosters_papers(conn):
	_upsert_paper_minimal(conn, 'in-run', title='Kept')
	_upsert_paper_minimal(conn, 'not-in-run', title='Skipped')
	run_id = db.start_run(conn, papers_fetched=1)
	db.record_run_papers(conn, run_id, [('in-run', True)])

	papers = db.list_papers_for_run(conn, run_id)

	assert [p['paper_id'] for p in papers] == ['in-run']


def test_list_papers_for_run_returns_empty_list_when_run_has_no_roster(conn):
	# Historical pre-migration runs and any future run that crashed before recording its roster both surface as empty here; History page renders an empty-state message.
	run_id = db.start_run(conn, papers_fetched=0)
	assert db.list_papers_for_run(conn, run_id) == []


def test_list_papers_for_run_returns_rich_paper_shape_for_dashboard(conn):
	# History page reuses the dashboard's modal renderer; the shape returned here must match what search_papers returns so the modal works without branching.
	_upsert_paper_minimal(conn, 'p1', title='Paper One')
	db.upsert_summary(conn, 'p1', {'methodology': 'm', 'findings': 'f', 'relevance': 'r', 'limitations': 'l'}, 'v1')
	db.insert_rating(conn, 'p1', 4)
	run_id = db.start_run(conn, papers_fetched=1)
	db.record_run_papers(conn, run_id, [('p1', True)])

	(paper,) = db.list_papers_for_run(conn, run_id)

	assert paper['title'] == 'Paper One'
	assert paper['methodology'] == 'm'
	assert paper['latest_rating'] == 4
	assert paper['was_new'] is True


def test_list_papers_for_run_returns_was_new_flag_from_join_row(conn):
	_upsert_paper_minimal(conn, 'fresh')
	_upsert_paper_minimal(conn, 'repeat')
	run_id = db.start_run(conn, papers_fetched=2)
	db.record_run_papers(conn, run_id, [('fresh', True), ('repeat', False)])

	papers = db.list_papers_for_run(conn, run_id)

	by_id = {p['paper_id']: p['was_new'] for p in papers}
	assert by_id == {'fresh': True, 'repeat': False}


def test_truncating_run_cascades_to_runs_papers(conn):
	# ON DELETE CASCADE on the runs(id) FK keeps the roster aligned with the runs table when a run row is removed.
	_upsert_paper_minimal(conn, 'p1')
	run_id = db.start_run(conn, papers_fetched=1)
	db.record_run_papers(conn, run_id, [('p1', True)])

	with conn.cursor() as cur:
		cur.execute('DELETE FROM runs WHERE id = %s', (run_id,))
		cur.execute('SELECT COUNT(*) FROM runs_papers WHERE run_id = %s', (run_id,))
		assert cur.fetchone()[0] == 0


def test_deleting_paper_cascades_to_runs_papers(conn):
	# Symmetric to the runs cascade; orphan roster entries shouldn't survive a paper deletion.
	_upsert_paper_minimal(conn, 'p1')
	run_id = db.start_run(conn, papers_fetched=1)
	db.record_run_papers(conn, run_id, [('p1', True)])

	with conn.cursor() as cur:
		cur.execute('DELETE FROM papers WHERE paper_id = %s', ('p1',))
		cur.execute('SELECT COUNT(*) FROM runs_papers WHERE paper_id = %s', ('p1',))
		assert cur.fetchone()[0] == 0


# -- app_config (per-key JSONB) + audit history --


def test_load_config_returns_empty_dict_when_table_empty(conn):
	assert db.load_config(conn) == {}


def test_update_config_round_trips_a_single_key(conn):
	db.update_config(conn, {'days_back': 14})
	assert db.load_config(conn) == {'days_back': 14}


def test_update_config_preserves_jsonb_types_across_keys(conn):
	updates = {'search_queries': ['a', 'b', 'c'], 'relevance_threshold': 6, 'research_context': 'multi\nline\ntext'}
	db.update_config(conn, updates)
	assert db.load_config(conn) == updates


def test_update_config_replaces_value_for_existing_key(conn):
	db.update_config(conn, {'days_back': 7})
	db.update_config(conn, {'days_back': 14})
	assert db.load_config(conn) == {'days_back': 14}


def test_update_config_persists_updated_by_when_provided(conn):
	db.update_config(conn, {'days_back': 14}, by='alice')
	with conn.cursor() as cur:
		cur.execute('SELECT updated_by FROM app_config WHERE key = %s', ('days_back',))
		assert cur.fetchone()[0] == 'alice'


def test_update_config_persists_updated_by_as_null_by_default(conn):
	db.update_config(conn, {'days_back': 14})
	with conn.cursor() as cur:
		cur.execute('SELECT updated_by FROM app_config WHERE key = %s', ('days_back',))
		assert cur.fetchone()[0] is None


def test_update_config_appends_history_on_every_write_including_overwrite(conn):
	# Two writes to the same key produce two history rows; the audit log is append-only.
	db.update_config(conn, {'days_back': 7})
	db.update_config(conn, {'days_back': 14})
	history = db.list_config_history(conn, key='days_back')
	assert [row['value'] for row in history] == [14, 7]


def test_update_config_appends_one_history_row_per_key_per_write(conn):
	db.update_config(conn, {'days_back': 7, 'relevance_threshold': 6})
	history = db.list_config_history(conn)
	assert sorted(row['key'] for row in history) == ['days_back', 'relevance_threshold']


def test_list_config_history_orders_newest_first(conn):
	db.update_config(conn, {'days_back': 7})
	db.update_config(conn, {'days_back': 14})
	db.update_config(conn, {'days_back': 21})
	history = db.list_config_history(conn, key='days_back')
	assert [row['value'] for row in history] == [21, 14, 7]


def test_list_config_history_filters_by_key(conn):
	db.update_config(conn, {'days_back': 7})
	db.update_config(conn, {'relevance_threshold': 6})
	history = db.list_config_history(conn, key='days_back')
	assert [row['key'] for row in history] == ['days_back']


def test_list_config_history_respects_limit(conn):
	for i in range(5):
		db.update_config(conn, {'days_back': i})
	assert len(db.list_config_history(conn, key='days_back', limit=3)) == 3


def test_list_config_history_records_changed_by(conn):
	db.update_config(conn, {'days_back': 14}, by='alice')
	(row,) = db.list_config_history(conn, key='days_back')
	assert row['changed_by'] == 'alice'


def test_update_config_no_op_does_not_touch_either_table(conn):
	db.update_config(conn, {})
	with conn.cursor() as cur:
		cur.execute('SELECT COUNT(*) FROM app_config')
		assert cur.fetchone()[0] == 0
		cur.execute('SELECT COUNT(*) FROM app_config_history')
		assert cur.fetchone()[0] == 0
