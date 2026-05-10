import pytest

import main


@pytest.fixture(autouse=True)
def env(monkeypatch):
	monkeypatch.setenv('DATABASE_URL', 'postgresql://fake')
	monkeypatch.setenv('GEMINI_API_KEY', 'fake')
	monkeypatch.setenv('GMAIL_USER', 'u')
	monkeypatch.setenv('GMAIL_APP_PASSWORD', 'p')
	monkeypatch.setenv('EMAIL_TO', 'to@x')


@pytest.fixture
def mock_db(mocker):
	conn = mocker.MagicMock()
	return {
		'conn': conn,
		'connect': mocker.patch('main.db.connect', return_value=conn),
		'init_schema': mocker.patch('main.db.init_schema'),
		'start_run': mocker.patch('main.db.start_run', return_value=42),
		'upsert_paper': mocker.patch('main.db.upsert_paper', return_value=True),
		'finish_run': mocker.patch('main.db.finish_run'),
	}


@pytest.fixture
def mock_io(mocker):
	return {
		'fetch': mocker.patch('main.fetch_papers', return_value=[{'paperId': 'p1'}]),
		'score': mocker.patch('main.scorer.score_and_summarise', return_value=[{'paperId': 'p1', 'ai_score': 8}]),
		'send': mocker.patch('main.emailer.send_email'),
	}


def test_live_run_opens_db_initialises_schema_and_brackets_with_run_lifecycle(mock_db, mock_io):
	main.main([])
	mock_db['connect'].assert_called_once_with('postgresql://fake')
	mock_db['init_schema'].assert_called_once_with(mock_db['conn'])
	mock_db['start_run'].assert_called_once_with(mock_db['conn'], 1)
	mock_db['finish_run'].assert_called_once_with(mock_db['conn'], 42, 1)


def test_live_run_upserts_every_unique_paper(mock_db, mock_io, mocker):
	mocker.patch('main.fetch_papers', return_value=[{'paperId': 'p1'}, {'paperId': 'p2'}, {'paperId': 'p3'}])
	main.main([])
	assert mock_db['upsert_paper'].call_count == 3
	upserted = [c.args[1]['paperId'] for c in mock_db['upsert_paper'].call_args_list]
	assert upserted == ['p1', 'p2', 'p3']


def test_live_run_only_passes_newly_inserted_papers_to_scorer(mock_db, mock_io, mocker):
	mocker.patch('main.fetch_papers', return_value=[{'paperId': 'p1'}, {'paperId': 'p2'}, {'paperId': 'p3'}])
	mock_db['upsert_paper'].side_effect = [True, False, True]
	main.main([])
	scored = mock_io['score'].call_args.args[0]
	assert [p['paperId'] for p in scored] == ['p1', 'p3']


def test_live_run_finishes_run_and_skips_email_when_no_papers_kept(mock_db, mock_io):
	mock_io['score'].return_value = []
	main.main([])
	mock_db['finish_run'].assert_called_once_with(mock_db['conn'], 42, 0)
	mock_io['send'].assert_not_called()


def test_live_run_sends_email_when_papers_kept(mock_db, mock_io):
	main.main([])
	mock_io['send'].assert_called_once()
	html, count, creds = mock_io['send'].call_args.args
	assert count == 1
	assert creds.user == 'u'
	assert creds.to == 'to@x'


def test_live_run_sends_email_before_finishing_the_run(mock_db, mock_io):
	# Logging is secondary to delivery; finish_run must run after _send so a logging
	# failure can never silently swallow an email that was meant to go out.
	call_order = []
	mock_io['send'].side_effect = lambda *a, **kw: call_order.append('send')
	mock_db['finish_run'].side_effect = lambda *a, **kw: call_order.append('finish_run')

	main.main([])

	assert call_order == ['send', 'finish_run']


# -- failure modes --


def test_live_run_propagates_db_connect_failure(mocker, mock_io):
	import psycopg

	mocker.patch('main.db.connect', side_effect=psycopg.OperationalError('connection refused'))
	with pytest.raises(psycopg.OperationalError, match='connection refused'):
		main.main([])
	# Nothing downstream of the connect should run.
	mock_io['fetch'].assert_not_called()
	mock_io['score'].assert_not_called()
	mock_io['send'].assert_not_called()


def test_live_run_does_not_finish_run_when_upsert_crashes_mid_pipeline(mock_db, mock_io, mocker):
	mocker.patch('main.fetch_papers', return_value=[{'paperId': 'p1'}, {'paperId': 'p2'}, {'paperId': 'p3'}])
	mock_db['upsert_paper'].side_effect = [True, RuntimeError('upsert boom'), True]

	with pytest.raises(RuntimeError, match='upsert boom'):
		main.main([])

	# Run row stays with NULL finished_at so the failed run is visible in diagnostics.
	mock_db['finish_run'].assert_not_called()
	mock_io['score'].assert_not_called()
	mock_io['send'].assert_not_called()


def test_live_run_upserts_papers_even_when_scorer_returns_empty(mock_db, mock_io):
	# Simulates Gemini failure: _score_chunk catches the exception and returns []. The
	# pipeline must still record the run cleanly and persist the papers that were fetched.
	mock_io['score'].return_value = []

	main.main([])

	assert mock_db['upsert_paper'].call_count == 1
	mock_db['finish_run'].assert_called_once_with(mock_db['conn'], 42, 0)
	mock_io['send'].assert_not_called()


def test_live_run_propagates_when_scorer_raises_unhandled(mock_db, mock_io):
	mock_io['score'].side_effect = RuntimeError('scorer boom')

	with pytest.raises(RuntimeError, match='scorer boom'):
		main.main([])

	# Upserts ran before scoring, finish_run did not.
	assert mock_db['upsert_paper'].call_count == 1
	mock_db['finish_run'].assert_not_called()
	mock_io['send'].assert_not_called()
