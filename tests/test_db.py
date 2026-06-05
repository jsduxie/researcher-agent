import pytest

import db


@pytest.fixture
def mock_conn(mocker):
	conn = mocker.MagicMock()
	# conn.cursor() is used as a context manager; conn.transaction() too. MagicMock supports both protocols out of the box.
	return conn


def _cursor(mock_conn):
	return mock_conn.cursor.return_value.__enter__.return_value


# -- connect --


def test_connect_passes_autocommit_and_disables_prepared_statements(mocker):
	mock_psycopg_connect = mocker.patch('db.psycopg.connect')
	db.connect('postgresql://x')
	mock_psycopg_connect.assert_called_once_with('postgresql://x', autocommit=True, prepare_threshold=None)


def test_connect_returns_psycopg_connection(mocker):
	mock_psycopg_connect = mocker.patch('db.psycopg.connect')
	result = db.connect('postgresql://x')
	assert result is mock_psycopg_connect.return_value


# -- session --


def test_session_yields_connection_from_connect(mocker):
	mock_psycopg_connect = mocker.patch('db.psycopg.connect')
	with db.session('postgresql://x') as conn:
		assert conn is mock_psycopg_connect.return_value
	mock_psycopg_connect.assert_called_once_with('postgresql://x', autocommit=True, prepare_threshold=None)


def test_session_closes_connection_on_normal_exit(mocker):
	mock_conn = mocker.patch('db.psycopg.connect').return_value
	with db.session('postgresql://x'):
		pass
	mock_conn.close.assert_called_once()


def test_session_closes_and_propagates_on_body_exception(mocker):
	mock_conn = mocker.patch('db.psycopg.connect').return_value
	with pytest.raises(RuntimeError, match='boom'), db.session('postgresql://x'):
		raise RuntimeError('boom')
	mock_conn.close.assert_called_once()


def test_session_propagates_connect_failure(mocker):
	import psycopg

	mocker.patch('db.psycopg.connect', side_effect=psycopg.OperationalError('refused'))
	with pytest.raises(psycopg.OperationalError, match='refused'), db.session('postgresql://bad'):
		pass


def test_session_stashes_database_url_on_conn_for_reconnect(mocker):
	# @database_reconnect reads conn._database_url to open a fresh session on OperationalError.
	mocker.patch('db.psycopg.connect')
	with db.session('postgresql://x') as conn:
		assert conn._database_url == 'postgresql://x'


# -- init_schema --


def test_init_schema_executes_schema_sql(mock_conn):
	db.init_schema(mock_conn)
	executed_sql = _cursor(mock_conn).execute.call_args.args[0]
	assert 'CREATE TABLE IF NOT EXISTS papers' in executed_sql
	assert 'CREATE TABLE IF NOT EXISTS paper_authors' in executed_sql
	assert 'CREATE TABLE IF NOT EXISTS summaries' in executed_sql
	assert 'CREATE TABLE IF NOT EXISTS runs' in executed_sql
	assert 'CREATE TABLE IF NOT EXISTS ratings' in executed_sql
	assert 'CREATE TABLE IF NOT EXISTS summary_feedback' in executed_sql


def test_init_schema_includes_idempotent_migration_for_runs_query_columns(mock_conn):
	db.init_schema(mock_conn)
	executed_sql = _cursor(mock_conn).execute.call_args.args[0]
	assert 'ALTER TABLE runs ADD COLUMN IF NOT EXISTS queries_attempted' in executed_sql
	assert 'ALTER TABLE runs ADD COLUMN IF NOT EXISTS queries_errored' in executed_sql


def test_init_schema_includes_summary_feedback_thumbs_to_rating_migration(mock_conn):
	db.init_schema(mock_conn)
	executed_sql = _cursor(mock_conn).execute.call_args.args[0]
	assert 'ALTER TABLE summary_feedback ADD COLUMN IF NOT EXISTS rating' in executed_sql
	assert 'ALTER TABLE summary_feedback DROP COLUMN IF EXISTS thumbs' in executed_sql


def test_init_schema_includes_app_config_and_history_tables(mock_conn):
	db.init_schema(mock_conn)
	executed_sql = _cursor(mock_conn).execute.call_args.args[0]
	assert 'CREATE TABLE IF NOT EXISTS app_config' in executed_sql
	assert 'CREATE TABLE IF NOT EXISTS app_config_history' in executed_sql
	assert 'CREATE INDEX IF NOT EXISTS app_config_history_changed_at_idx' in executed_sql


# -- upsert_paper --


def test_upsert_paper_uses_insert_on_conflict_with_returning(mock_conn):
	_cursor(mock_conn).fetchone.return_value = (True,)
	db.upsert_paper(mock_conn, {'paperId': 'abc'})
	insert_sql = _cursor(mock_conn).execute.call_args_list[0].args[0]
	assert 'INSERT INTO papers' in insert_sql
	assert 'ON CONFLICT (paper_id) DO UPDATE' in insert_sql
	assert 'RETURNING (xmax = 0) AS was_inserted' in insert_sql


def test_upsert_paper_returns_true_when_newly_inserted(mock_conn):
	_cursor(mock_conn).fetchone.return_value = (True,)
	assert db.upsert_paper(mock_conn, {'paperId': 'abc'}) is True


def test_upsert_paper_returns_false_when_updated(mock_conn):
	_cursor(mock_conn).fetchone.return_value = (False,)
	assert db.upsert_paper(mock_conn, {'paperId': 'abc'}) is False


def test_upsert_paper_binds_all_fields_in_order(mock_conn):
	_cursor(mock_conn).fetchone.return_value = (True,)
	paper = {
		'paperId': 'abc',
		'title': 'T',
		'abstract': 'A',
		'year': 2024,
		'citationCount': 5,
		'url': 'https://ss/abc',
		'externalIds': {'DOI': '10.1/abc'},
		'openAccessPdf': {'url': 'https://x/p.pdf'},
	}
	db.upsert_paper(mock_conn, paper)
	params = _cursor(mock_conn).execute.call_args_list[0].args[1]
	assert params == ('abc', 'T', 'A', 2024, 5, 'https://ss/abc', '10.1/abc', 'https://x/p.pdf')


def test_upsert_paper_binds_none_for_missing_optional_fields(mock_conn):
	_cursor(mock_conn).fetchone.return_value = (True,)
	db.upsert_paper(mock_conn, {'paperId': 'abc'})
	params = _cursor(mock_conn).execute.call_args_list[0].args[1]
	assert params == ('abc', None, None, None, None, None, None, None)


def test_upsert_paper_handles_external_ids_none(mock_conn):
	_cursor(mock_conn).fetchone.return_value = (True,)
	db.upsert_paper(mock_conn, {'paperId': 'abc', 'externalIds': None, 'openAccessPdf': None})
	params = _cursor(mock_conn).execute.call_args_list[0].args[1]
	assert params[6] is None  # doi
	assert params[7] is None  # pdf_url


def test_upsert_paper_replaces_authors_via_delete_then_insert(mock_conn):
	_cursor(mock_conn).fetchone.return_value = (True,)
	paper = {'paperId': 'abc', 'authors': [{'name': 'Smith J.'}, {'name': 'Doe A.'}]}
	db.upsert_paper(mock_conn, paper)
	calls = _cursor(mock_conn).execute.call_args_list
	# 0: INSERT papers, 1: DELETE authors, 2: INSERT author 0, 3: INSERT author 1
	assert 'DELETE FROM paper_authors' in calls[1].args[0]
	assert calls[1].args[1] == ('abc',)
	assert 'INSERT INTO paper_authors' in calls[2].args[0]
	assert calls[2].args[1] == ('abc', 0, 'Smith J.')
	assert calls[3].args[1] == ('abc', 1, 'Doe A.')


def test_upsert_paper_skips_authors_missing_name(mock_conn):
	_cursor(mock_conn).fetchone.return_value = (True,)
	paper = {'paperId': 'abc', 'authors': [{'name': 'A'}, {}, {'name': None}, {'name': 'B'}]}
	db.upsert_paper(mock_conn, paper)
	author_inserts = [c for c in _cursor(mock_conn).execute.call_args_list if 'INSERT INTO paper_authors' in c.args[0]]
	assert len(author_inserts) == 2
	# Positions preserved from the original list (0 and 3), so downstream order is intact.
	assert author_inserts[0].args[1] == ('abc', 0, 'A')
	assert author_inserts[1].args[1] == ('abc', 3, 'B')


def test_upsert_paper_skips_non_dict_author_entries(mock_conn):
	_cursor(mock_conn).fetchone.return_value = (True,)
	paper = {'paperId': 'abc', 'authors': [{'name': 'A'}, 'not a dict', None, {'name': 'B'}]}
	db.upsert_paper(mock_conn, paper)
	author_inserts = [c for c in _cursor(mock_conn).execute.call_args_list if 'INSERT INTO paper_authors' in c.args[0]]
	assert [c.args[1] for c in author_inserts] == [('abc', 0, 'A'), ('abc', 3, 'B')]


def test_upsert_paper_with_no_authors_only_deletes(mock_conn):
	_cursor(mock_conn).fetchone.return_value = (True,)
	db.upsert_paper(mock_conn, {'paperId': 'abc'})
	calls = _cursor(mock_conn).execute.call_args_list
	# INSERT papers, then DELETE authors. No INSERT into paper_authors.
	assert len(calls) == 2
	assert 'DELETE FROM paper_authors' in calls[1].args[0]


def test_upsert_paper_runs_inside_a_transaction(mock_conn):
	_cursor(mock_conn).fetchone.return_value = (True,)
	db.upsert_paper(mock_conn, {'paperId': 'abc', 'authors': [{'name': 'A'}]})
	mock_conn.transaction.assert_called_once()


# -- paper_exists --


def test_paper_exists_returns_true_when_row_present(mock_conn):
	_cursor(mock_conn).fetchone.return_value = (1,)
	assert db.paper_exists(mock_conn, 'abc') is True
	call = _cursor(mock_conn).execute.call_args
	assert 'SELECT 1 FROM papers WHERE paper_id = %s' in call.args[0]
	assert call.args[1] == ('abc',)


def test_paper_exists_returns_false_when_row_absent(mock_conn):
	_cursor(mock_conn).fetchone.return_value = None
	assert db.paper_exists(mock_conn, 'abc') is False


# -- start_run / finish_run --


def test_start_run_inserts_and_returns_new_id(mock_conn):
	_cursor(mock_conn).fetchone.return_value = (42,)
	assert db.start_run(mock_conn, 10) == 42
	call = _cursor(mock_conn).execute.call_args
	assert 'INSERT INTO runs' in call.args[0]
	assert 'RETURNING id' in call.args[0]
	assert call.args[1] == (10,)


def test_start_run_leaves_papers_kept_null_until_finish(mock_conn):
	_cursor(mock_conn).fetchone.return_value = (1,)
	db.start_run(mock_conn, 10)
	sql = _cursor(mock_conn).execute.call_args.args[0]
	assert 'papers_kept' not in sql
	assert 'finished_at' not in sql


def test_finish_run_updates_finished_at_and_papers_kept(mock_conn):
	db.finish_run(mock_conn, run_id=42, papers_kept=7, queries_attempted=0, queries_errored=0)
	call = _cursor(mock_conn).execute.call_args
	assert 'UPDATE runs' in call.args[0]
	assert 'finished_at = NOW()' in call.args[0]
	assert 'papers_kept = %s' in call.args[0]
	assert call.args[1][0] == 7
	assert call.args[1][-1] == 42


def test_finish_run_records_query_outcomes(mock_conn):
	db.finish_run(mock_conn, run_id=42, papers_kept=3, queries_attempted=8, queries_errored=2)
	call = _cursor(mock_conn).execute.call_args
	assert 'queries_attempted = %s' in call.args[0]
	assert 'queries_errored = %s' in call.args[0]
	assert call.args[1] == (3, 8, 2, 42)


# -- error propagation: db.py never swallows database errors --


def test_connect_propagates_operational_error(mocker):
	import psycopg

	mocker.patch('db.psycopg.connect', side_effect=psycopg.OperationalError('connection refused'))
	with pytest.raises(psycopg.OperationalError, match='connection refused'):
		db.connect('postgresql://bad')


def test_init_schema_propagates_database_error(mock_conn):
	_cursor(mock_conn).execute.side_effect = RuntimeError('DDL boom')
	with pytest.raises(RuntimeError, match='DDL boom'):
		db.init_schema(mock_conn)


def test_upsert_paper_propagates_database_error(mock_conn):
	_cursor(mock_conn).execute.side_effect = RuntimeError('write boom')
	with pytest.raises(RuntimeError, match='write boom'):
		db.upsert_paper(mock_conn, {'paperId': 'abc'})


def test_paper_exists_propagates_database_error(mock_conn):
	_cursor(mock_conn).execute.side_effect = RuntimeError('read boom')
	with pytest.raises(RuntimeError, match='read boom'):
		db.paper_exists(mock_conn, 'abc')


def test_start_run_propagates_database_error(mock_conn):
	_cursor(mock_conn).execute.side_effect = RuntimeError('start boom')
	with pytest.raises(RuntimeError, match='start boom'):
		db.start_run(mock_conn, 10)


def test_finish_run_propagates_database_error(mock_conn):
	_cursor(mock_conn).execute.side_effect = RuntimeError('finish boom')
	with pytest.raises(RuntimeError, match='finish boom'):
		db.finish_run(mock_conn, run_id=1, papers_kept=0, queries_attempted=0, queries_errored=0)


# -- needs_scoring --


def test_needs_scoring_returns_empty_set_for_empty_input(mock_conn):
	assert db.needs_scoring(mock_conn, []) == set()
	_cursor(mock_conn).execute.assert_not_called()


def test_needs_scoring_selects_paper_id_where_scored_at_is_null(mock_conn):
	_cursor(mock_conn).fetchall.return_value = [('p1',), ('p3',)]
	result = db.needs_scoring(mock_conn, ['p1', 'p2', 'p3'])
	call = _cursor(mock_conn).execute.call_args
	assert 'SELECT paper_id FROM papers' in call.args[0]
	assert 'scored_at IS NULL' in call.args[0]
	assert 'paper_id = ANY(%s)' in call.args[0]
	assert call.args[1] == (['p1', 'p2', 'p3'],)
	assert result == {'p1', 'p3'}


def test_needs_scoring_returns_empty_set_when_no_rows_match(mock_conn):
	_cursor(mock_conn).fetchall.return_value = []
	assert db.needs_scoring(mock_conn, ['p1']) == set()


def test_needs_scoring_propagates_database_error(mock_conn):
	_cursor(mock_conn).execute.side_effect = RuntimeError('select boom')
	with pytest.raises(RuntimeError, match='select boom'):
		db.needs_scoring(mock_conn, ['p1'])


# -- list_unscored --


def _unscored_row(paper_id='p1', doi='10.1/x', pdf_url='http://pdf', authors=('A One',)):
	return (paper_id, 't', 'a', 2024, 5, 'http://u', doi, pdf_url, list(authors))


def test_list_unscored_filters_on_attempts_exclusions_and_limit(mock_conn):
	_cursor(mock_conn).fetchall.return_value = []
	db.list_unscored(mock_conn, exclude_ids=['p9'], max_attempts=3, limit=20)
	sql, params = _cursor(mock_conn).execute.call_args.args
	assert 'scored_at IS NULL' in sql
	assert 'score_attempts < %s' in sql
	assert 'NOT (paper_id = ANY(%s))' in sql
	assert 'LIMIT %s' in sql
	assert params == (3, ['p9'], 20)


def test_list_unscored_maps_rows_to_semantic_scholar_shape(mock_conn):
	# The scoring pipeline expects fetcher-shaped dicts; swept rows must be indistinguishable from fresh fetches.
	_cursor(mock_conn).fetchall.return_value = [_unscored_row()]
	[paper] = db.list_unscored(mock_conn, exclude_ids=[], max_attempts=3, limit=20)
	assert paper == {
		'paperId': 'p1',
		'title': 't',
		'abstract': 'a',
		'year': 2024,
		'citationCount': 5,
		'url': 'http://u',
		'externalIds': {'DOI': '10.1/x'},
		'openAccessPdf': {'url': 'http://pdf'},
		'authors': [{'name': 'A One'}],
	}


def test_list_unscored_omits_doi_and_pdf_when_absent(mock_conn):
	_cursor(mock_conn).fetchall.return_value = [_unscored_row(doi=None, pdf_url=None, authors=())]
	[paper] = db.list_unscored(mock_conn, exclude_ids=[], max_attempts=3, limit=20)
	assert paper['externalIds'] == {}
	assert paper['openAccessPdf'] is None
	assert paper['authors'] == []


def test_list_unscored_propagates_database_error(mock_conn):
	_cursor(mock_conn).execute.side_effect = RuntimeError('sweep boom')
	with pytest.raises(RuntimeError, match='sweep boom'):
		db.list_unscored(mock_conn, exclude_ids=[], max_attempts=3, limit=20)


# -- embeddings --


def test_set_paper_embedding_serialises_vector_with_cast(mock_conn):
	db.set_paper_embedding(mock_conn, 'p1', [0.1, 0.2, 0.3])
	sql, params = _cursor(mock_conn).execute.call_args.args
	assert '%s::vector' in sql
	assert params == ('[0.1,0.2,0.3]', 'p1')


def test_get_paper_embedding_parses_vector_text(mock_conn):
	_cursor(mock_conn).fetchone.return_value = ('[0.1,0.2,0.3]',)
	assert db.get_paper_embedding(mock_conn, 'p1') == [0.1, 0.2, 0.3]


def test_get_paper_embedding_returns_none_when_paper_missing(mock_conn):
	_cursor(mock_conn).fetchone.return_value = None
	assert db.get_paper_embedding(mock_conn, 'p1') is None


def test_get_paper_embedding_returns_none_when_not_embedded(mock_conn):
	_cursor(mock_conn).fetchone.return_value = (None,)
	assert db.get_paper_embedding(mock_conn, 'p1') is None


# -- mark_scoring_results --


def test_mark_scoring_results_returns_early_when_attempted_is_empty(mock_conn):
	db.mark_scoring_results(mock_conn, attempted=[], responded=set())
	_cursor(mock_conn).execute.assert_not_called()


def test_mark_scoring_results_increments_score_attempts_for_attempted(mock_conn):
	db.mark_scoring_results(mock_conn, attempted=['p1', 'p2'], responded=set())
	# Only one UPDATE statement when responded is empty.
	calls = _cursor(mock_conn).execute.call_args_list
	assert len(calls) == 1
	assert 'UPDATE papers SET score_attempts = score_attempts + 1' in calls[0].args[0]
	assert 'paper_id = ANY(%s)' in calls[0].args[0]
	assert calls[0].args[1] == (['p1', 'p2'],)


def test_mark_scoring_results_sets_scored_at_for_responded(mock_conn):
	db.mark_scoring_results(mock_conn, attempted=['p1', 'p2'], responded={'p1'})
	calls = _cursor(mock_conn).execute.call_args_list
	assert len(calls) == 2
	# First UPDATE: increment score_attempts for all attempted.
	assert 'score_attempts = score_attempts + 1' in calls[0].args[0]
	assert sorted(calls[0].args[1][0]) == ['p1', 'p2']
	# Second UPDATE: set scored_at for responded only.
	assert 'scored_at = NOW()' in calls[1].args[0]
	assert calls[1].args[1] == (['p1'],)


def test_mark_scoring_results_wraps_updates_in_a_transaction(mock_conn):
	db.mark_scoring_results(mock_conn, attempted=['p1'], responded={'p1'})
	mock_conn.transaction.assert_called_once()


def test_mark_scoring_results_propagates_database_error(mock_conn):
	_cursor(mock_conn).execute.side_effect = RuntimeError('update boom')
	with pytest.raises(RuntimeError, match='update boom'):
		db.mark_scoring_results(mock_conn, attempted=['p1'], responded={'p1'})


# -- get_summary --


def test_get_summary_returns_dict_when_row_present(mock_conn):
	_cursor(mock_conn).fetchone.return_value = ('m', 'f', 'r', 'l', 'test-model-v1')
	result = db.get_summary(mock_conn, 'abc')
	call = _cursor(mock_conn).execute.call_args
	assert 'SELECT methodology, findings, relevance, limitations, model_version FROM summaries' in call.args[0]
	assert 'paper_id = %s' in call.args[0]
	assert call.args[1] == ('abc',)
	assert result == {
		'methodology': 'm',
		'findings': 'f',
		'relevance': 'r',
		'limitations': 'l',
		'model_version': 'test-model-v1',
	}


def test_get_summary_returns_none_when_no_row(mock_conn):
	_cursor(mock_conn).fetchone.return_value = None
	assert db.get_summary(mock_conn, 'abc') is None


def test_get_summary_propagates_database_error(mock_conn):
	_cursor(mock_conn).execute.side_effect = RuntimeError('select boom')
	with pytest.raises(RuntimeError, match='select boom'):
		db.get_summary(mock_conn, 'abc')


# -- upsert_summary --


def test_upsert_summary_uses_insert_on_conflict(mock_conn):
	fields = {'methodology': 'm', 'findings': 'f', 'relevance': 'r', 'limitations': 'l'}
	db.upsert_summary(mock_conn, 'abc', fields, 'test-model-v1')
	sql = _cursor(mock_conn).execute.call_args.args[0]
	assert 'INSERT INTO summaries' in sql
	assert 'ON CONFLICT (paper_id) DO UPDATE' in sql


def test_upsert_summary_preserves_created_at_on_conflict(mock_conn):
	# The ON CONFLICT clause must only touch updated_at; otherwise the column literally named created_at stops representing creation time.
	db.upsert_summary(mock_conn, 'abc', {}, 'test-model-v1')
	sql = _cursor(mock_conn).execute.call_args.args[0]
	do_update_clause = sql.split('DO UPDATE')[1]
	assert 'updated_at = NOW()' in do_update_clause
	assert 'created_at' not in do_update_clause


def test_upsert_summary_binds_all_fields_in_order(mock_conn):
	fields = {'methodology': 'm', 'findings': 'f', 'relevance': 'r', 'limitations': 'l'}
	db.upsert_summary(mock_conn, 'abc', fields, 'test-model-v1')
	params = _cursor(mock_conn).execute.call_args.args[1]
	assert params == ('abc', 'm', 'f', 'r', 'l', 'test-model-v1')


def test_upsert_summary_binds_none_for_missing_field_keys(mock_conn):
	# A field absent from the dict should bind NULL. Callers using a placeholder ('Not available') always provide the key; this guards a partial Gemini response.
	db.upsert_summary(mock_conn, 'abc', {'methodology': 'm'}, 'test-model-v1')
	params = _cursor(mock_conn).execute.call_args.args[1]
	assert params == ('abc', 'm', None, None, None, 'test-model-v1')


def test_upsert_summary_accepts_none_model_version(mock_conn):
	db.upsert_summary(mock_conn, 'abc', {}, None)
	params = _cursor(mock_conn).execute.call_args.args[1]
	assert params == ('abc', None, None, None, None, None)


def test_upsert_summary_propagates_database_error(mock_conn):
	_cursor(mock_conn).execute.side_effect = RuntimeError('write boom')
	with pytest.raises(RuntimeError, match='write boom'):
		db.upsert_summary(mock_conn, 'abc', {}, 'test-model-v1')


# -- database_reconnect (decorator in isolation) --


def test_database_reconnect_retries_once_on_operational_error_then_succeeds(mocker):
	import psycopg

	calls = []

	@db.database_reconnect
	def op(conn, x):
		calls.append(conn)
		if len(calls) == 1:
			raise psycopg.OperationalError('reaped')
		return x * 2

	original_conn = mocker.MagicMock()
	original_conn._database_url = 'postgresql://x'
	fresh_conn = mocker.MagicMock()
	mock_connect = mocker.patch('db.psycopg.connect', return_value=fresh_conn)

	result = op(original_conn, 5)

	assert result == 10
	assert calls == [original_conn, fresh_conn]
	mock_connect.assert_called_once_with('postgresql://x', autocommit=True, prepare_threshold=None)


def test_database_reconnect_propagates_when_second_attempt_also_fails(mocker):
	import psycopg

	@db.database_reconnect
	def op(conn):
		raise psycopg.OperationalError('always')

	original_conn = mocker.MagicMock()
	original_conn._database_url = 'postgresql://x'
	mocker.patch('db.psycopg.connect', return_value=mocker.MagicMock())

	with pytest.raises(psycopg.OperationalError, match='always'):
		op(original_conn)


def test_database_reconnect_does_not_retry_on_non_operational_error(mocker):
	import psycopg

	calls = []

	@db.database_reconnect
	def op(conn):
		calls.append(conn)
		raise psycopg.DataError('bad data')

	original_conn = mocker.MagicMock()
	original_conn._database_url = 'postgresql://x'
	mock_connect = mocker.patch('db.psycopg.connect')

	with pytest.raises(psycopg.DataError, match='bad data'):
		op(original_conn)

	assert len(calls) == 1
	mock_connect.assert_not_called()


def test_database_reconnect_propagates_when_conn_has_no_database_url(mocker):
	import psycopg

	@db.database_reconnect
	def op(conn):
		raise psycopg.OperationalError('reaped')

	# A conn with no _database_url attribute (e.g., a code path that bypassed db.session).
	class BareConn:
		pass

	conn = BareConn()
	mock_connect = mocker.patch('db.psycopg.connect')

	with pytest.raises(psycopg.OperationalError, match='reaped'):
		op(conn)

	mock_connect.assert_not_called()


# -- reconnect on real wrapped helpers --


def _setup_reconnect(mock_conn, mocker):
	# Common setup: original conn's cursor raises OperationalError on every execute; a fresh conn returned by db.psycopg.connect for the reconnect.
	import psycopg

	mock_conn._database_url = 'postgresql://x'
	_cursor(mock_conn).execute.side_effect = psycopg.OperationalError('reaped')
	fresh_conn = mocker.MagicMock()
	mock_connect = mocker.patch('db.psycopg.connect', return_value=fresh_conn)
	return fresh_conn, mock_connect


def test_init_schema_retries_on_operational_error(mock_conn, mocker):
	fresh_conn, mock_connect = _setup_reconnect(mock_conn, mocker)
	db.init_schema(mock_conn)
	_cursor(fresh_conn).execute.assert_called_once()
	mock_connect.assert_called_once_with('postgresql://x', autocommit=True, prepare_threshold=None)


def test_upsert_paper_retries_on_operational_error(mock_conn, mocker):
	fresh_conn, mock_connect = _setup_reconnect(mock_conn, mocker)
	_cursor(fresh_conn).fetchone.return_value = (True,)
	result = db.upsert_paper(mock_conn, {'paperId': 'abc'})
	assert result is True
	mock_connect.assert_called_once()


def test_start_run_retries_on_operational_error(mock_conn, mocker):
	fresh_conn, mock_connect = _setup_reconnect(mock_conn, mocker)
	_cursor(fresh_conn).fetchone.return_value = (42,)
	result = db.start_run(mock_conn, 10)
	assert result == 42
	mock_connect.assert_called_once()


def test_finish_run_retries_on_operational_error(mock_conn, mocker):
	fresh_conn, mock_connect = _setup_reconnect(mock_conn, mocker)
	db.finish_run(mock_conn, run_id=1, papers_kept=0, queries_attempted=0, queries_errored=0)
	_cursor(fresh_conn).execute.assert_called_once()
	mock_connect.assert_called_once()


def test_upsert_summary_retries_on_operational_error(mock_conn, mocker):
	fresh_conn, mock_connect = _setup_reconnect(mock_conn, mocker)
	db.upsert_summary(mock_conn, 'abc', {}, 'test-model-v1')
	_cursor(fresh_conn).execute.assert_called_once()
	mock_connect.assert_called_once()


def test_mark_scoring_results_retries_on_operational_error(mock_conn, mocker):
	fresh_conn, mock_connect = _setup_reconnect(mock_conn, mocker)
	db.mark_scoring_results(mock_conn, attempted=['p1'], responded={'p1'})
	# Two executes on the fresh conn: increment + scored_at update.
	assert _cursor(fresh_conn).execute.call_count == 2
	mock_connect.assert_called_once()


def test_insert_rating_retries_on_operational_error(mock_conn, mocker):
	fresh_conn, mock_connect = _setup_reconnect(mock_conn, mocker)
	db.insert_rating(mock_conn, 'abc', 4)
	_cursor(fresh_conn).execute.assert_called_once()
	mock_connect.assert_called_once()


def test_insert_summary_feedback_retries_on_operational_error(mock_conn, mocker):
	fresh_conn, mock_connect = _setup_reconnect(mock_conn, mocker)
	db.insert_summary_feedback(mock_conn, 'abc', 'methodology', rating=4)
	_cursor(fresh_conn).execute.assert_called_once()
	mock_connect.assert_called_once()


# -- insert_rating --


def test_insert_rating_appends_row(mock_conn):
	db.insert_rating(mock_conn, 'abc', 4)
	call = _cursor(mock_conn).execute.call_args
	assert 'INSERT INTO ratings' in call.args[0]
	assert '(paper_id, rating)' in call.args[0]
	assert call.args[1] == ('abc', 4)


def test_insert_rating_propagates_database_error(mock_conn):
	_cursor(mock_conn).execute.side_effect = RuntimeError('write boom')
	with pytest.raises(RuntimeError, match='write boom'):
		db.insert_rating(mock_conn, 'abc', 4)


# -- insert_summary_feedback --


def test_insert_summary_feedback_appends_row_with_all_fields(mock_conn):
	db.insert_summary_feedback(mock_conn, 'abc', 'methodology', rating=5, correction='clearer please')
	call = _cursor(mock_conn).execute.call_args
	assert 'INSERT INTO summary_feedback' in call.args[0]
	assert call.args[1] == ('abc', 'methodology', 5, 'clearer please')


def test_insert_summary_feedback_defaults_rating_and_correction_to_none(mock_conn):
	db.insert_summary_feedback(mock_conn, 'abc', 'findings')
	assert _cursor(mock_conn).execute.call_args.args[1] == ('abc', 'findings', None, None)


def test_insert_summary_feedback_propagates_database_error(mock_conn):
	_cursor(mock_conn).execute.side_effect = RuntimeError('write boom')
	with pytest.raises(RuntimeError, match='write boom'):
		db.insert_summary_feedback(mock_conn, 'abc', 'findings', rating=3)


# -- get_latest_rating --


def test_get_latest_rating_selects_latest_row(mock_conn):
	_cursor(mock_conn).fetchone.return_value = (5,)
	assert db.get_latest_rating(mock_conn, 'abc') == 5
	call = _cursor(mock_conn).execute.call_args
	# ORDER BY created_at DESC + id DESC tiebreak keeps ordering stable across rows inserted in the same millisecond (important for UI prefill).
	assert 'ORDER BY created_at DESC, id DESC LIMIT 1' in call.args[0]
	assert call.args[1] == ('abc',)


def test_get_latest_rating_returns_none_when_no_rows(mock_conn):
	_cursor(mock_conn).fetchone.return_value = None
	assert db.get_latest_rating(mock_conn, 'abc') is None


def test_get_latest_rating_propagates_database_error(mock_conn):
	_cursor(mock_conn).execute.side_effect = RuntimeError('select boom')
	with pytest.raises(RuntimeError, match='select boom'):
		db.get_latest_rating(mock_conn, 'abc')


# -- get_latest_field_feedback --


def test_get_latest_field_feedback_returns_dict_keyed_by_field(mock_conn):
	_cursor(mock_conn).fetchall.return_value = [('methodology', 5, None), ('findings', 3, 'expand more')]
	result = db.get_latest_field_feedback(mock_conn, 'abc')
	assert result == {
		'methodology': {'rating': 5, 'correction': None},
		'findings': {'rating': 3, 'correction': 'expand more'},
	}


def test_get_latest_field_feedback_uses_distinct_on_field(mock_conn):
	_cursor(mock_conn).fetchall.return_value = []
	db.get_latest_field_feedback(mock_conn, 'abc')
	sql = _cursor(mock_conn).execute.call_args.args[0]
	assert 'DISTINCT ON (field)' in sql
	assert 'ORDER BY field, created_at DESC, id DESC' in sql


def test_get_latest_field_feedback_returns_empty_dict_when_no_rows(mock_conn):
	_cursor(mock_conn).fetchall.return_value = []
	assert db.get_latest_field_feedback(mock_conn, 'abc') == {}


def test_get_latest_field_feedback_propagates_database_error(mock_conn):
	_cursor(mock_conn).execute.side_effect = RuntimeError('select boom')
	with pytest.raises(RuntimeError, match='select boom'):
		db.get_latest_field_feedback(mock_conn, 'abc')


# -- search_papers --


def test_search_papers_with_no_filters_omits_where(mock_conn):
	_cursor(mock_conn).fetchall.return_value = []
	db.search_papers(mock_conn)
	sql = _cursor(mock_conn).execute.call_args.args[0]
	# The inner authors subquery legitimately uses WHERE; the outer filter references papers.fetched_at / .title / .abstract, so absence of those is the signal.
	assert 'papers.fetched_at >=' not in sql
	assert 'papers.fetched_at <=' not in sql
	assert 'papers.title ILIKE' not in sql
	assert 'ORDER BY papers.fetched_at DESC' in sql
	assert 'LIMIT %s OFFSET %s' in sql


def test_search_papers_passes_limit_and_offset_as_trailing_params(mock_conn):
	_cursor(mock_conn).fetchall.return_value = []
	db.search_papers(mock_conn, limit=10, offset=20)
	params = _cursor(mock_conn).execute.call_args.args[1]
	assert params[-2:] == [10, 20]


def test_search_papers_with_query_adds_ilike_clauses_for_title_and_abstract(mock_conn):
	_cursor(mock_conn).fetchall.return_value = []
	db.search_papers(mock_conn, q='bpd')
	call = _cursor(mock_conn).execute.call_args
	assert 'ILIKE' in call.args[0]
	assert 'papers.title ILIKE' in call.args[0]
	assert 'papers.abstract ILIKE' in call.args[0]
	# Two %bpd% bindings for title and abstract.
	assert call.args[1][0] == '%bpd%'
	assert call.args[1][1] == '%bpd%'


def test_search_papers_with_date_range_adds_filter_clauses(mock_conn):
	from datetime import date

	_cursor(mock_conn).fetchall.return_value = []
	db.search_papers(mock_conn, date_range=db.DateRange(since=date(2026, 1, 1), until=date(2026, 5, 1)))
	call = _cursor(mock_conn).execute.call_args
	assert 'papers.fetched_at >= %s' in call.args[0]
	assert 'papers.fetched_at <= %s' in call.args[0]
	assert date(2026, 1, 1) in call.args[1]
	assert date(2026, 5, 1) in call.args[1]


def test_search_papers_with_only_lower_bound_adds_since_clause(mock_conn):
	from datetime import date

	_cursor(mock_conn).fetchall.return_value = []
	db.search_papers(mock_conn, date_range=db.DateRange(since=date(2026, 1, 1)))
	sql = _cursor(mock_conn).execute.call_args.args[0]
	assert 'papers.fetched_at >= %s' in sql
	assert 'papers.fetched_at <= %s' not in sql


def test_search_papers_with_only_upper_bound_adds_until_clause(mock_conn):
	from datetime import date

	_cursor(mock_conn).fetchall.return_value = []
	db.search_papers(mock_conn, date_range=db.DateRange(until=date(2026, 5, 1)))
	sql = _cursor(mock_conn).execute.call_args.args[0]
	assert 'papers.fetched_at <= %s' in sql
	assert 'papers.fetched_at >= %s' not in sql


def test_search_papers_returns_list_of_dicts_with_expected_columns(mock_conn):
	from datetime import datetime

	_cursor(mock_conn).fetchall.return_value = [
		(
			'p1',
			'T',
			'A',
			2024,
			5,
			'https://u',
			'10.1/x',
			'https://x/p.pdf',
			datetime(2026, 5, 1),
			['Smith J.'],
			'm',
			'f',
			'r',
			'l',
			4,
		)
	]
	result = db.search_papers(mock_conn)
	assert result == [
		{
			'paper_id': 'p1',
			'title': 'T',
			'abstract': 'A',
			'year': 2024,
			'citation_count': 5,
			'url': 'https://u',
			'doi': '10.1/x',
			'pdf_url': 'https://x/p.pdf',
			'fetched_at': datetime(2026, 5, 1),
			'authors': ['Smith J.'],
			'methodology': 'm',
			'findings': 'f',
			'relevance': 'r',
			'limitations': 'l',
			'latest_rating': 4,
		}
	]


def test_search_papers_sql_includes_latest_rating_subquery(mock_conn):
	_cursor(mock_conn).fetchall.return_value = []
	db.search_papers(mock_conn)
	sql = _cursor(mock_conn).execute.call_args.args[0]
	assert 'SELECT rating FROM ratings' in sql
	assert 'AS latest_rating' in sql


def test_search_papers_propagates_database_error(mock_conn):
	_cursor(mock_conn).execute.side_effect = RuntimeError('select boom')
	with pytest.raises(RuntimeError, match='select boom'):
		db.search_papers(mock_conn)


# -- count_papers --


def test_count_papers_returns_total_count(mock_conn):
	_cursor(mock_conn).fetchone.return_value = (42,)
	assert db.count_papers(mock_conn) == 42
	sql = _cursor(mock_conn).execute.call_args.args[0]
	assert 'SELECT COUNT(*) FROM papers' in sql
	assert 'WHERE' not in sql


def test_count_papers_applies_same_filters_as_search(mock_conn):
	_cursor(mock_conn).fetchone.return_value = (3,)
	db.count_papers(mock_conn, q='bpd')
	call = _cursor(mock_conn).execute.call_args
	assert 'papers.title ILIKE' in call.args[0]
	assert call.args[1][0] == '%bpd%'


# -- list_runs --


def test_list_runs_returns_rows_ordered_by_started_at_desc(mock_conn):
	from datetime import datetime

	_cursor(mock_conn).fetchall.return_value = [
		(2, datetime(2026, 5, 2), datetime(2026, 5, 2), 50, 5, 8, 0),
		(1, datetime(2026, 5, 1), datetime(2026, 5, 1), 40, 3, 8, 1),
	]
	result = db.list_runs(mock_conn, limit=10)
	call = _cursor(mock_conn).execute.call_args
	assert 'ORDER BY started_at DESC' in call.args[0]
	assert call.args[1] == (10,)
	assert len(result) == 2
	assert result[0]['id'] == 2
	assert result[0]['papers_kept'] == 5
	assert result[1]['queries_errored'] == 1


def test_list_runs_propagates_database_error(mock_conn):
	_cursor(mock_conn).execute.side_effect = RuntimeError('select boom')
	with pytest.raises(RuntimeError, match='select boom'):
		db.list_runs(mock_conn)


# -- record_run_papers --


def test_record_run_papers_inserts_one_row_per_entry_with_was_new_flag(mock_conn):
	db.record_run_papers(mock_conn, 42, [('p1', True), ('p2', False), ('p3', True)])
	call = _cursor(mock_conn).executemany.call_args
	assert 'INSERT INTO runs_papers' in call.args[0]
	assert 'was_new' in call.args[0]
	assert 'ON CONFLICT DO NOTHING' in call.args[0]
	assert call.args[1] == [(42, 'p1', True), (42, 'p2', False), (42, 'p3', True)]


def test_record_run_papers_no_ops_on_empty_input(mock_conn):
	db.record_run_papers(mock_conn, 42, [])
	_cursor(mock_conn).executemany.assert_not_called()


def test_record_run_papers_propagates_database_error(mock_conn):
	_cursor(mock_conn).executemany.side_effect = RuntimeError('insert boom')
	with pytest.raises(RuntimeError, match='insert boom'):
		db.record_run_papers(mock_conn, 42, [('p1', True)])


# -- list_papers_for_run --


def test_list_papers_for_run_filters_by_run_id_via_join(mock_conn):
	_cursor(mock_conn).fetchall.return_value = []
	db.list_papers_for_run(mock_conn, 42)
	call = _cursor(mock_conn).execute.call_args
	assert 'JOIN runs_papers' in call.args[0]
	assert 'WHERE runs_papers.run_id = %s' in call.args[0]
	assert call.args[1] == (42,)


def test_list_papers_for_run_returns_list_of_dicts_with_expected_columns(mock_conn):
	from datetime import datetime

	_cursor(mock_conn).fetchall.return_value = [
		(
			'p1',
			'title',
			'abstract',
			2024,
			10,
			'https://x/p1',
			'10.1/p1',
			'https://x/p1.pdf',
			datetime(2026, 5, 1),
			['Author One'],
			'methodology text',
			'findings text',
			'relevance text',
			'limitations text',
			4,
			True,
		)
	]
	result = db.list_papers_for_run(mock_conn, 42)
	assert len(result) == 1
	assert result[0]['paper_id'] == 'p1'
	assert result[0]['authors'] == ['Author One']
	assert result[0]['latest_rating'] == 4
	assert result[0]['methodology'] == 'methodology text'
	assert result[0]['was_new'] is True


def test_list_papers_for_run_returns_empty_list_when_run_has_no_papers(mock_conn):
	_cursor(mock_conn).fetchall.return_value = []
	assert db.list_papers_for_run(mock_conn, 42) == []


def test_list_papers_for_run_propagates_database_error(mock_conn):
	_cursor(mock_conn).execute.side_effect = RuntimeError('select boom')
	with pytest.raises(RuntimeError, match='select boom'):
		db.list_papers_for_run(mock_conn, 42)


# -- load_config --


def test_load_config_returns_dict_of_key_to_value(mock_conn):
	_cursor(mock_conn).fetchall.return_value = [('relevance_threshold', 6), ('search_queries', ['a', 'b'])]
	assert db.load_config(mock_conn) == {'relevance_threshold': 6, 'search_queries': ['a', 'b']}
	sql = _cursor(mock_conn).execute.call_args.args[0]
	assert 'SELECT key, value FROM app_config' in sql


def test_load_config_returns_empty_dict_when_no_rows(mock_conn):
	_cursor(mock_conn).fetchall.return_value = []
	assert db.load_config(mock_conn) == {}


def test_load_config_propagates_database_error(mock_conn):
	_cursor(mock_conn).execute.side_effect = RuntimeError('select boom')
	with pytest.raises(RuntimeError, match='select boom'):
		db.load_config(mock_conn)


# -- update_config --


def test_update_config_no_op_when_updates_empty(mock_conn):
	db.update_config(mock_conn, {})
	_cursor(mock_conn).execute.assert_not_called()
	mock_conn.transaction.assert_not_called()


def test_update_config_writes_both_tables_for_each_key(mock_conn):
	db.update_config(mock_conn, {'days_back': 14, 'relevance_threshold': 6})
	calls = _cursor(mock_conn).execute.call_args_list
	# Two keys, each producing an upsert plus a history append, gives 4 cursor calls.
	assert len(calls) == 4
	assert 'INSERT INTO app_config' in calls[0].args[0]
	assert 'ON CONFLICT (key) DO UPDATE' in calls[0].args[0]
	assert 'INSERT INTO app_config_history' in calls[1].args[0]
	assert 'INSERT INTO app_config' in calls[2].args[0]
	assert 'INSERT INTO app_config_history' in calls[3].args[0]


def test_update_config_wraps_in_a_single_transaction(mock_conn):
	db.update_config(mock_conn, {'days_back': 14, 'relevance_threshold': 6})
	mock_conn.transaction.assert_called_once()


def test_update_config_wraps_values_in_jsonb(mock_conn):
	from psycopg.types.json import Jsonb

	db.update_config(mock_conn, {'search_queries': ['x', 'y']})
	calls = _cursor(mock_conn).execute.call_args_list
	upsert_value = calls[0].args[1][1]
	history_value = calls[1].args[1][1]
	assert isinstance(upsert_value, Jsonb)
	assert isinstance(history_value, Jsonb)
	assert upsert_value.obj == ['x', 'y']
	assert history_value.obj == ['x', 'y']


def test_update_config_threads_by_to_both_inserts(mock_conn):
	db.update_config(mock_conn, {'days_back': 14}, by='alice')
	calls = _cursor(mock_conn).execute.call_args_list
	# (key, value, by) for upsert; (key, value, by) for history insert.
	assert calls[0].args[1][0] == 'days_back'
	assert calls[0].args[1][2] == 'alice'
	assert calls[1].args[1][0] == 'days_back'
	assert calls[1].args[1][2] == 'alice'


def test_update_config_defaults_by_to_none(mock_conn):
	db.update_config(mock_conn, {'days_back': 14})
	calls = _cursor(mock_conn).execute.call_args_list
	assert calls[0].args[1][2] is None
	assert calls[1].args[1][2] is None


def test_update_config_propagates_database_error(mock_conn):
	_cursor(mock_conn).execute.side_effect = RuntimeError('write boom')
	with pytest.raises(RuntimeError, match='write boom'):
		db.update_config(mock_conn, {'days_back': 14})


def test_update_config_retries_on_operational_error(mock_conn, mocker):
	fresh_conn, mock_connect = _setup_reconnect(mock_conn, mocker)
	db.update_config(mock_conn, {'days_back': 14})
	# Two executes on the fresh conn: upsert + history append.
	assert _cursor(fresh_conn).execute.call_count == 2
	mock_connect.assert_called_once()


# -- list_config_history --


def test_list_config_history_unfiltered_orders_by_changed_at_desc(mock_conn):
	_cursor(mock_conn).fetchall.return_value = []
	db.list_config_history(mock_conn)
	call = _cursor(mock_conn).execute.call_args
	sql = call.args[0]
	assert 'FROM app_config_history' in sql
	assert 'ORDER BY changed_at DESC, id DESC' in sql
	assert 'WHERE key' not in sql
	assert call.args[1] == (20,)


def test_list_config_history_with_key_adds_where_clause(mock_conn):
	_cursor(mock_conn).fetchall.return_value = []
	db.list_config_history(mock_conn, key='days_back', limit=5)
	call = _cursor(mock_conn).execute.call_args
	sql = call.args[0]
	assert 'WHERE key = %s' in sql
	assert call.args[1] == ('days_back', 5)


def test_list_config_history_returns_list_of_dicts_with_expected_columns(mock_conn):
	from datetime import datetime

	_cursor(mock_conn).fetchall.return_value = [
		(2, 'days_back', 14, datetime(2026, 5, 16), 'alice'),
		(1, 'relevance_threshold', 6, datetime(2026, 5, 15), None),
	]
	result = db.list_config_history(mock_conn)
	assert result == [
		{'id': 2, 'key': 'days_back', 'value': 14, 'changed_at': datetime(2026, 5, 16), 'changed_by': 'alice'},
		{'id': 1, 'key': 'relevance_threshold', 'value': 6, 'changed_at': datetime(2026, 5, 15), 'changed_by': None},
	]


def test_list_config_history_returns_empty_list_when_no_rows(mock_conn):
	_cursor(mock_conn).fetchall.return_value = []
	assert db.list_config_history(mock_conn) == []


def test_list_config_history_propagates_database_error(mock_conn):
	_cursor(mock_conn).execute.side_effect = RuntimeError('select boom')
	with pytest.raises(RuntimeError, match='select boom'):
		db.list_config_history(mock_conn)
