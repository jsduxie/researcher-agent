import pytest

import db


@pytest.fixture
def mock_conn(mocker):
	conn = mocker.MagicMock()
	# conn.cursor() is used as a context manager; conn.transaction() too.
	# MagicMock supports both protocols out of the box.
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
	db.finish_run(mock_conn, run_id=42, papers_kept=7)
	call = _cursor(mock_conn).execute.call_args
	assert 'UPDATE runs' in call.args[0]
	assert 'finished_at = NOW()' in call.args[0]
	assert 'papers_kept = %s' in call.args[0]
	assert call.args[1] == (7, 42)


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
		db.finish_run(mock_conn, run_id=1, papers_kept=0)


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
	_cursor(mock_conn).fetchone.return_value = ('m', 'f', 'r', 'l', 'gemini-2.5-flash')
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
		'model_version': 'gemini-2.5-flash',
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
	db.upsert_summary(mock_conn, 'abc', fields, 'gemini-2.5-flash')
	sql = _cursor(mock_conn).execute.call_args.args[0]
	assert 'INSERT INTO summaries' in sql
	assert 'ON CONFLICT (paper_id) DO UPDATE' in sql


def test_upsert_summary_preserves_created_at_on_conflict(mock_conn):
	# The ON CONFLICT clause must only touch updated_at; otherwise the column
	# literally named created_at stops representing creation time.
	db.upsert_summary(mock_conn, 'abc', {}, 'gemini-2.5-flash')
	sql = _cursor(mock_conn).execute.call_args.args[0]
	do_update_clause = sql.split('DO UPDATE')[1]
	assert 'updated_at = NOW()' in do_update_clause
	assert 'created_at' not in do_update_clause


def test_upsert_summary_binds_all_fields_in_order(mock_conn):
	fields = {'methodology': 'm', 'findings': 'f', 'relevance': 'r', 'limitations': 'l'}
	db.upsert_summary(mock_conn, 'abc', fields, 'gemini-2.5-flash')
	params = _cursor(mock_conn).execute.call_args.args[1]
	assert params == ('abc', 'm', 'f', 'r', 'l', 'gemini-2.5-flash')


def test_upsert_summary_binds_none_for_missing_field_keys(mock_conn):
	# A field absent from the dict should bind NULL rather than raise. Callers using a
	# placeholder string ("Not available from this source.") will provide the key; this
	# guards against a partial Gemini response with keys missing entirely.
	db.upsert_summary(mock_conn, 'abc', {'methodology': 'm'}, 'gemini-2.5-flash')
	params = _cursor(mock_conn).execute.call_args.args[1]
	assert params == ('abc', 'm', None, None, None, 'gemini-2.5-flash')


def test_upsert_summary_accepts_none_model_version(mock_conn):
	db.upsert_summary(mock_conn, 'abc', {}, None)
	params = _cursor(mock_conn).execute.call_args.args[1]
	assert params == ('abc', None, None, None, None, None)


def test_upsert_summary_propagates_database_error(mock_conn):
	_cursor(mock_conn).execute.side_effect = RuntimeError('write boom')
	with pytest.raises(RuntimeError, match='write boom'):
		db.upsert_summary(mock_conn, 'abc', {}, 'gemini-2.5-flash')
