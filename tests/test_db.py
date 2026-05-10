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
