import pytest

import main
from config import SEARCH_QUERIES


@pytest.fixture(autouse=True)
def env(monkeypatch):
	monkeypatch.setenv('DATABASE_URL', 'postgresql://fake')
	monkeypatch.setenv('GEMINI_API_KEY', 'fake')
	monkeypatch.setenv('SEMANTIC_SCHOLAR_API_KEY', 'fake-ss')
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
		'summarise': mocker.patch(
			'main.summariser.summarise_paper',
			return_value={'methodology': 'm', 'findings': 'f', 'relevance': 'r', 'limitations': 'l'},
		),
		'send': mocker.patch('main.emailer.send_email'),
	}


def test_live_run_opens_db_initialises_schema_and_brackets_with_run_lifecycle(mock_db, mock_io):
	main.main([])
	mock_db['connect'].assert_called_once_with('postgresql://fake')
	mock_db['init_schema'].assert_called_once_with(mock_db['conn'])
	mock_db['start_run'].assert_called_once_with(mock_db['conn'], 1)
	mock_db['finish_run'].assert_called_once_with(mock_db['conn'], 42, 1, len(SEARCH_QUERIES), 0)


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
	mock_db['finish_run'].assert_called_once_with(mock_db['conn'], 42, 0, len(SEARCH_QUERIES), 0)
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
	mock_db['finish_run'].assert_called_once_with(mock_db['conn'], 42, 0, len(SEARCH_QUERIES), 0)
	mock_io['send'].assert_not_called()


def test_live_run_exits_nonzero_when_all_queries_error(mock_db, mock_io, mocker, capsys):
	from fetcher import FetchError

	mocker.patch('main.fetch_papers', side_effect=FetchError('rate limited'))
	mock_io['score'].return_value = ([], set())

	with pytest.raises(SystemExit) as exc:
		main.main([])

	assert exc.value.code != 0
	mock_io['send'].assert_not_called()
	# finish_run runs before the exit so the run row reflects what happened.
	mock_db['finish_run'].assert_called_once()
	args = mock_db['finish_run'].call_args.args
	assert args[3] > 0  # queries_attempted
	assert args[3] == args[4]  # all attempted == errored
	assert 'Error fetching' in capsys.readouterr().out


def test_live_run_proceeds_when_some_queries_succeed(mock_db, mock_io, mocker):
	from config import SEARCH_QUERIES
	from fetcher import FetchError

	side_effects = [FetchError('rate limited')] * (len(SEARCH_QUERIES) - 1) + [[{'paperId': 'p1'}]]
	mocker.patch('main.fetch_papers', side_effect=side_effects)

	main.main([])

	mock_db['finish_run'].assert_called_once()
	args = mock_db['finish_run'].call_args.args
	assert args[3] == len(SEARCH_QUERIES)
	assert args[4] == len(SEARCH_QUERIES) - 1


def test_live_run_exits_zero_when_no_results_but_no_errors(mock_db, mock_io):
	mock_io['fetch'].return_value = []
	mock_io['score'].return_value = ([], set())

	main.main([])  # no SystemExit

	mock_io['send'].assert_not_called()
	mock_db['finish_run'].assert_called_once()
	args = mock_db['finish_run'].call_args.args
	assert args[3] > 0
	assert args[4] == 0


def test_live_run_exits_when_semantic_scholar_api_key_missing(monkeypatch, mock_db, mock_io):
	monkeypatch.delenv('SEMANTIC_SCHOLAR_API_KEY', raising=False)

	with pytest.raises(SystemExit) as exc:
		main.main([])

	assert 'SEMANTIC_SCHOLAR_API_KEY' in str(exc.value)
	mock_db['connect'].assert_not_called()
	mock_io['fetch'].assert_not_called()


def test_live_run_passes_api_key_to_fetch_papers(mock_db, mock_io):
	main.main([])
	# Every call to fetch_papers must carry the key sourced from the env.
	for call in mock_io['fetch'].call_args_list:
		assert call.args[1] == 'fake-ss'


def test_live_run_propagates_when_scorer_raises_unhandled(mock_db, mock_io):
	mock_io['score'].side_effect = RuntimeError('scorer boom')

	with pytest.raises(RuntimeError, match='scorer boom'):
		main.main([])

	# Upserts ran before scoring; mark_scoring_results and finish_run did not.
	assert mock_db['upsert_paper'].call_count == 1
	mock_db['mark_scoring_results'].assert_not_called()
	mock_db['finish_run'].assert_not_called()
	mock_io['send'].assert_not_called()


# -- paperId guard --


def test_live_run_drops_papers_without_paper_id_before_persisting(mock_db, mock_io, mocker, capsys):
	mocker.patch('main.fetch_papers', return_value=[{'paperId': 'p1', 'title': 'with id'}, {'title': 'no id'}])

	main.main([])

	# Only the paper with paperId reaches upsert_paper and start_run's count.
	mock_db['upsert_paper'].assert_called_once()
	upserted = mock_db['upsert_paper'].call_args.args[1]
	assert upserted['paperId'] == 'p1'
	mock_db['start_run'].assert_called_once_with(mock_db['conn'], 1)
	# Log line confirms the drop.
	assert 'Dropped 1 paper(s) without paperId' in capsys.readouterr().out


def test_live_run_keeps_running_when_every_paper_lacks_paper_id(mock_db, mock_io, mocker, capsys):
	# Edge case: all fetched papers are missing paperId. The pipeline should drop them,
	# log, and finish cleanly without ever calling upsert.
	mocker.patch('main.fetch_papers', return_value=[{'title': 'one'}, {'title': 'two'}])
	mock_io['score'].return_value = ([], set())

	main.main([])

	mock_db['upsert_paper'].assert_not_called()
	mock_db['start_run'].assert_called_once_with(mock_db['conn'], 0)
	mock_db['finish_run'].assert_called_once_with(mock_db['conn'], 42, 0, len(SEARCH_QUERIES), 0)
	assert 'Dropped 2 paper(s) without paperId' in capsys.readouterr().out


# -- gemini wrappers --


def test_gemini_score_wrapper_calls_scorer_gemini_in_live_mode(mocker, monkeypatch):
	# Orchestration tests mock scorer.score_and_summarise wholesale, so the live-mode
	# branch of main._gemini_score (the callable passed to the scorer) is never
	# exercised end-to-end. Cover it directly.
	monkeypatch.setattr(main, 'DRY_RUN', False)
	mock_scorer_gemini = mocker.patch('main.scorer.gemini', return_value='gemini json')

	result = main._gemini_score('prompt text')

	mock_scorer_gemini.assert_called_once_with('prompt text', 'fake', 3)
	assert result == 'gemini json'


def test_gemini_summarise_wrapper_calls_scorer_gemini_in_live_mode(mocker, monkeypatch):
	# Same as above for the summariser-facing wrapper, which routes prompt-only
	# summariser calls through the same generateContent endpoint.
	monkeypatch.setattr(main, 'DRY_RUN', False)
	mock_scorer_gemini = mocker.patch('main.scorer.gemini', return_value='summary json')

	result = main._gemini_summarise('prompt text')

	mock_scorer_gemini.assert_called_once_with('prompt text', 'fake', 3)
	assert result == 'summary json'


def test_gemini_score_wrapper_increments_call_count(monkeypatch):
	monkeypatch.setattr(main, 'DRY_RUN', True)
	main.GEMINI_CALL_COUNT = 0
	main._gemini_score('prompt')
	main._gemini_score('prompt')
	assert main.GEMINI_CALL_COUNT == 2


def test_gemini_score_wrapper_does_not_increment_when_scorer_raises(monkeypatch, mocker):
	# Transport errors must not be counted. If they were, a flapping endpoint could
	# trip the call-count warning even though no quota was consumed.
	monkeypatch.setattr(main, 'DRY_RUN', False)
	mocker.patch('main.scorer.gemini', side_effect=Exception('boom'))
	main.GEMINI_CALL_COUNT = 0
	with pytest.raises(Exception, match='boom'):
		main._gemini_score('prompt')
	assert main.GEMINI_CALL_COUNT == 0


def test_record_gemini_call_increments_count(monkeypatch):
	main.GEMINI_CALL_COUNT = 0
	main._record_gemini_call()
	main._record_gemini_call()
	main._record_gemini_call()
	assert main.GEMINI_CALL_COUNT == 3


# -- summariser orchestration --


def test_live_run_summarises_each_enriched_paper(mock_db, mock_io):
	main.main([])
	mock_io['summarise'].assert_called_once()
	call = mock_io['summarise'].call_args
	assert call.args[0]['paperId'] == 'p1'
	# Wired with the summariser-specific gemini wrapper, db connection, api_key, and
	# the call-count recorder.
	assert call.args[1] is main._gemini_summarise
	assert call.kwargs['conn'] is mock_db['conn']
	assert call.kwargs['api_key'] == 'fake'
	assert call.kwargs['on_gemini_call'] is main._record_gemini_call


def test_live_run_summarises_only_papers_above_threshold(mock_db, mock_io):
	# Scorer returns a single enriched paper; even if more papers were scored,
	# only those that made it past the relevance filter should be summarised.
	mock_io['score'].return_value = ([{'paperId': 'kept', 'ai_score': 8}], {'kept', 'dropped'})
	main.main([])
	mock_io['summarise'].assert_called_once()
	assert mock_io['summarise'].call_args.args[0]['paperId'] == 'kept'


def test_live_run_does_not_summarise_when_no_papers_kept(mock_db, mock_io):
	mock_io['score'].return_value = ([], set())
	main.main([])
	mock_io['summarise'].assert_not_called()


def test_live_run_attaches_summary_fields_to_paper(mock_db, mock_io):
	main.main([])
	# After summarise_paper returns the four-field dict, the paper passed to the
	# email builder should carry those fields. Verify via the dict that was emailed.
	emailed_paper = mock_io['summarise'].call_args.args[0]
	assert emailed_paper['methodology'] == 'm'
	assert emailed_paper['findings'] == 'f'
	assert emailed_paper['relevance'] == 'r'
	assert emailed_paper['limitations'] == 'l'


def test_live_run_handles_summariser_returning_none(mock_db, mock_io):
	# A complete summariser failure for a paper must not break the run; the email
	# still goes out and the run still finishes.
	mock_io['summarise'].return_value = None
	main.main([])
	mock_io['send'].assert_called_once()
	mock_db['finish_run'].assert_called_once()


# -- gemini call count + warning --


def test_live_run_prints_total_gemini_call_count(mock_db, mock_io, capsys):
	# The on_gemini_call callback fires inside summariser; simulate it firing on each
	# summarise_paper call to confirm main accumulates the count.
	def fake_summarise(paper, gemini_fn, conn, api_key, on_gemini_call):
		on_gemini_call()
		return {'methodology': 'm', 'findings': 'f', 'relevance': 'r', 'limitations': 'l'}

	mock_io['summarise'].side_effect = fake_summarise
	main.main([])
	out = capsys.readouterr().out
	assert 'Total Gemini calls this run:' in out


def test_live_run_warns_when_gemini_call_count_exceeds_threshold(mock_db, mock_io, mocker, capsys):
	mocker.patch('main.GEMINI_CALL_WARN_THRESHOLD', 0)

	def fake_summarise(paper, gemini_fn, conn, api_key, on_gemini_call):
		on_gemini_call()

	mock_io['summarise'].side_effect = fake_summarise
	main.main([])
	assert 'WARNING: Gemini call count' in capsys.readouterr().out


def test_live_run_does_not_warn_when_gemini_call_count_under_threshold(mock_db, mock_io, mocker, capsys):
	mocker.patch('main.GEMINI_CALL_WARN_THRESHOLD', 1_000_000)
	main.main([])
	assert 'WARNING' not in capsys.readouterr().out


def test_live_run_resets_gemini_call_count_between_runs(mock_db, mock_io):
	# A stale counter from a prior run would cause spurious warnings on the next.
	main.GEMINI_CALL_COUNT = 999
	main.main([])
	# After the run, count reflects only this run's calls (scorer is mocked so it
	# didn't go through _gemini_score; summariser was mocked so no on_gemini_call
	# fired). Net result: count is back to 0 plus whatever the mocks triggered.
	assert main.GEMINI_CALL_COUNT < 999
