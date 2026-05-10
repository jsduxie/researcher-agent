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
	# By default, needs_scoring claims every paper passed in needs scoring.
	return {
		'conn': conn,
		'connect': mocker.patch('main.db.connect', return_value=conn),
		'init_schema': mocker.patch('main.db.init_schema'),
		'start_run': mocker.patch('main.db.start_run', return_value=42),
		'upsert_paper': mocker.patch('main.db.upsert_paper'),
		'needs_scoring': mocker.patch('main.db.needs_scoring', side_effect=lambda conn, ids: set(ids)),
		'mark_scoring_results': mocker.patch('main.db.mark_scoring_results'),
		'finish_run': mocker.patch('main.db.finish_run'),
	}


@pytest.fixture
def mock_io(mocker):
	return {
		'fetch': mocker.patch('main.fetch_papers', return_value=[{'paperId': 'p1'}]),
		'score': mocker.patch(
			'main.scorer.score_and_summarise', return_value=([{'paperId': 'p1', 'ai_score': 8}], {'p1'})
		),
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


def test_live_run_only_scores_papers_needs_scoring_reports_as_unscored(mock_db, mock_io, mocker):
	# needs_scoring is what now controls which papers reach the scorer (replaces
	# the was_inserted filter from commit 2).
	mocker.patch('main.fetch_papers', return_value=[{'paperId': 'p1'}, {'paperId': 'p2'}, {'paperId': 'p3'}])
	mock_db['needs_scoring'].side_effect = lambda conn, ids: {'p1', 'p3'}
	main.main([])
	scored = mock_io['score'].call_args.args[0]
	assert sorted(p['paperId'] for p in scored) == ['p1', 'p3']


def test_live_run_marks_scoring_results_with_attempted_and_responded(mock_db, mock_io):
	mock_io['score'].return_value = ([{'paperId': 'p1', 'ai_score': 8}], {'p1'})
	main.main([])
	mock_db['mark_scoring_results'].assert_called_once()
	call = mock_db['mark_scoring_results'].call_args
	assert call.args[0] is mock_db['conn']
	assert call.kwargs['attempted'] == ['p1']
	assert call.kwargs['responded'] == {'p1'}


def test_live_run_does_not_call_mark_scoring_results_when_nothing_needs_scoring(mock_db, mock_io):
	mock_db['needs_scoring'].side_effect = lambda conn, ids: set()
	main.main([])
	mock_db['mark_scoring_results'].assert_not_called()
	# Scorer also gets nothing since nothing needed scoring.
	scored = mock_io['score'].call_args.args[0]
	assert scored == []


def test_live_run_finishes_run_and_skips_email_when_no_papers_kept(mock_db, mock_io):
	mock_io['score'].return_value = ([], set())
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
	mock_db['upsert_paper'].side_effect = [None, RuntimeError('upsert boom'), None]

	with pytest.raises(RuntimeError, match='upsert boom'):
		main.main([])

	mock_db['finish_run'].assert_not_called()
	mock_db['mark_scoring_results'].assert_not_called()
	mock_io['score'].assert_not_called()
	mock_io['send'].assert_not_called()


def test_live_run_marks_scoring_results_even_when_scorer_returns_empty(mock_db, mock_io):
	# Simulates a Gemini batch failure: scorer returns ([], set()). Papers were attempted
	# but none responded. mark_scoring_results still records the attempts so we can see
	# them in score_attempts. scored_at stays NULL, so a later run re-tries them.
	mock_io['score'].return_value = ([], set())

	main.main([])

	mock_db['mark_scoring_results'].assert_called_once()
	call = mock_db['mark_scoring_results'].call_args
	assert call.kwargs['attempted'] == ['p1']
	assert call.kwargs['responded'] == set()
	mock_db['finish_run'].assert_called_once_with(mock_db['conn'], 42, 0)
	mock_io['send'].assert_not_called()


def test_live_run_propagates_when_scorer_raises_unhandled(mock_db, mock_io):
	mock_io['score'].side_effect = RuntimeError('scorer boom')

	with pytest.raises(RuntimeError, match='scorer boom'):
		main.main([])

	# Upserts ran before scoring; mark_scoring_results and finish_run did not.
	assert mock_db['upsert_paper'].call_count == 1
	mock_db['mark_scoring_results'].assert_not_called()
	mock_db['finish_run'].assert_not_called()
	mock_io['send'].assert_not_called()
