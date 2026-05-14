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
	# `db.session` is the scoped context manager used by every DB boundary in main. Default: each session yields the same mock conn.
	conn = mocker.MagicMock()
	session_cm = mocker.MagicMock()
	session_cm.__enter__.return_value = conn
	session_cm.__exit__.return_value = False
	# By default, needs_scoring claims every paper passed in needs scoring.
	return {
		'conn': conn,
		'session': mocker.patch('main.db.session', return_value=session_cm),
		'init_schema': mocker.patch('main.db.init_schema'),
		'start_run': mocker.patch('main.db.start_run', return_value=42),
		'upsert_paper': mocker.patch('main.db.upsert_paper', return_value=True),
		'record_run_papers': mocker.patch('main.db.record_run_papers'),
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
	# Every session call carries the DATABASE_URL value; multiple opens happen across boundaries.
	for call in mock_db['session'].call_args_list:
		assert call.args == ('postgresql://fake',)
	mock_db['init_schema'].assert_called_once_with(mock_db['conn'])
	mock_db['start_run'].assert_called_once_with(mock_db['conn'], 1)
	mock_db['finish_run'].assert_called_once_with(mock_db['conn'], 42, 1, len(SEARCH_QUERIES), 0)


# -- visibility logging (env-divergence diagnosis) --


def test_live_run_logs_db_target_host_at_startup(mock_db, mock_io, monkeypatch, capsys):
	# A misconfigured DATABASE_URL secret in CI silently writes to the wrong Neon endpoint; logging the host makes the divergence visible in 5 seconds.
	monkeypatch.setenv('DATABASE_URL', 'postgresql://user:secret@host.neon.tech:5432/neondb?sslmode=require')
	main.main([])
	out = capsys.readouterr().out
	assert 'DB target: host.neon.tech:5432' in out
	assert 'secret' not in out


def test_live_run_does_not_log_db_credentials(mock_db, mock_io, monkeypatch, capsys):
	monkeypatch.setenv('DATABASE_URL', 'postgresql://user:supersecretpassword@host/db')
	main.main([])
	assert 'supersecretpassword' not in capsys.readouterr().out


def test_db_host_for_log_returns_unknown_when_url_lacks_credentials_block():
	# Defensive: don't crash if someone passes a URL without an '@' marker; surface a sentinel rather than the raw URL.
	assert main._db_host_for_log('postgresql://host/db') == '<unknown>'


def test_live_run_logs_persisted_fetched_papers_summary(mock_db, mock_io, mocker, capsys):
	mocker.patch('main.fetch_papers', return_value=[{'paperId': 'p1'}, {'paperId': 'p2'}, {'paperId': 'p3'}])
	main.main([])
	out = capsys.readouterr().out
	# Operators can confirm at a glance that the upserts actually wrote and how many papers reached the scorer.
	assert 'Persisted fetched papers: run_id=42, upserted=3, needs_scoring=3' in out


def test_live_run_logs_recorded_scoring_attempts_summary(mock_db, mock_io, capsys):
	main.main([])
	out = capsys.readouterr().out
	assert 'Recorded scoring attempts: attempted=1, responded=1' in out


def test_live_run_logs_finalised_run_summary(mock_db, mock_io, capsys):
	main.main([])
	out = capsys.readouterr().out
	assert 'Finalised run: run_id=42, papers_kept=1' in out


def test_live_run_does_not_log_scoring_attempts_when_nothing_needs_scoring(mock_db, mock_io, capsys):
	mock_db['needs_scoring'].side_effect = lambda conn, ids: set()
	main.main([])
	# The scoring-attempts line is skipped when there's nothing to mark, so the summary lines stay truthful.
	assert 'Recorded scoring attempts' not in capsys.readouterr().out


def test_live_run_upserts_every_unique_paper(mock_db, mock_io, mocker):
	mocker.patch('main.fetch_papers', return_value=[{'paperId': 'p1'}, {'paperId': 'p2'}, {'paperId': 'p3'}])
	main.main([])
	assert mock_db['upsert_paper'].call_count == 3
	upserted = [c.args[1]['paperId'] for c in mock_db['upsert_paper'].call_args_list]
	assert upserted == ['p1', 'p2', 'p3']


def test_live_run_records_run_to_paper_roster_once_per_run(mock_db, mock_io, mocker):
	# runs_papers gives each run a precise paper roster so the History page can show "which papers belonged to this run" without an inferred fetched_at window.
	mocker.patch('main.fetch_papers', return_value=[{'paperId': 'p1'}, {'paperId': 'p2'}, {'paperId': 'p3'}])
	main.main([])
	mock_db['record_run_papers'].assert_called_once()
	call = mock_db['record_run_papers'].call_args
	assert call.args[1] == 42
	assert sorted(call.args[2]) == [('p1', True), ('p2', True), ('p3', True)]


def test_live_run_marks_returning_papers_as_repeats_in_the_roster(mock_db, mock_io, mocker):
	# upsert_paper returns False on conflict (paper was already in the table). The history roster carries that through so re-encountered papers are tagged "repeat".
	mocker.patch('main.fetch_papers', return_value=[{'paperId': 'fresh'}, {'paperId': 'repeat'}])
	mock_db['upsert_paper'].side_effect = [True, False]
	main.main([])
	entries = mock_db['record_run_papers'].call_args.args[2]
	assert sorted(entries) == [('fresh', True), ('repeat', False)]


def test_live_run_skips_run_paper_roster_when_no_papers_have_paper_id(mock_db, mock_io, mocker, capsys):
	# Papers without paperId are dropped before Phase A; record_run_papers still runs with an empty list, which db.record_run_papers no-ops on.
	mocker.patch('main.fetch_papers', return_value=[{'title': 'no id'}])
	main.main([])
	mock_db['record_run_papers'].assert_called_once()
	assert mock_db['record_run_papers'].call_args.args[2] == []
	assert 'Dropped 1 paper(s) without paperId' in capsys.readouterr().out


def test_live_run_only_scores_papers_needs_scoring_reports_as_unscored(mock_db, mock_io, mocker):
	# needs_scoring now controls which papers reach the scorer (replaces the was_inserted filter from commit 2).
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
	# Logging is secondary to delivery; finish_run must run after _send so a logging failure can never silently swallow an email that was meant to go out.
	call_order = []
	mock_io['send'].side_effect = lambda *a, **kw: call_order.append('send')
	mock_db['finish_run'].side_effect = lambda *a, **kw: call_order.append('finish_run')

	main.main([])

	assert call_order == ['send', 'finish_run']


# -- failure modes --


def test_live_run_propagates_db_session_failure(mock_db, mock_io):
	import psycopg

	mock_db['session'].side_effect = psycopg.OperationalError('connection refused')
	with pytest.raises(psycopg.OperationalError, match='connection refused'):
		main.main([])
	# Phase A opens the first session after fetch; fetch ran, but scoring and email did not.
	mock_io['fetch'].assert_called()
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
	# Simulates a Gemini batch failure: scorer returns ([], set()). Attempts are still recorded in score_attempts; scored_at stays NULL so a later run re-tries.
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


@pytest.mark.parametrize('var', main.REQUIRED_ENV_VARS)
def test_live_run_exits_when_required_env_var_missing(monkeypatch, mock_db, mock_io, var):
	monkeypatch.delenv(var, raising=False)

	with pytest.raises(SystemExit) as exc:
		main.main([])

	assert var in str(exc.value)
	mock_db['session'].assert_not_called()
	mock_io['fetch'].assert_not_called()


def test_live_run_exit_message_lists_every_missing_required_env_var(monkeypatch, mock_db, mock_io):
	# Listed together so operators fix all missing keys in one go, not one CI run at a time.
	for var in main.REQUIRED_ENV_VARS:
		monkeypatch.delenv(var, raising=False)

	with pytest.raises(SystemExit) as exc:
		main.main([])

	for var in main.REQUIRED_ENV_VARS:
		assert var in str(exc.value)


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
	# Edge case: all fetched papers are missing paperId. The pipeline drops them, logs, and finishes cleanly without ever calling upsert.
	mocker.patch('main.fetch_papers', return_value=[{'title': 'one'}, {'title': 'two'}])
	mock_io['score'].return_value = ([], set())

	main.main([])

	mock_db['upsert_paper'].assert_not_called()
	mock_db['start_run'].assert_called_once_with(mock_db['conn'], 0)
	mock_db['finish_run'].assert_called_once_with(mock_db['conn'], 42, 0, len(SEARCH_QUERIES), 0)
	assert 'Dropped 2 paper(s) without paperId' in capsys.readouterr().out


# -- gemini wrappers --


def test_gemini_score_wrapper_calls_scorer_gemini_in_live_mode(mocker, monkeypatch):
	# Orchestration tests mock scorer.score_and_summarise wholesale; cover the live-mode branch of _gemini_score directly.
	monkeypatch.setattr(main, 'DRY_RUN', False)
	mock_scorer_gemini = mocker.patch('main.scorer.gemini', return_value='gemini json')

	result = main._gemini_score('prompt text')

	mock_scorer_gemini.assert_called_once_with('prompt text', 'fake', 3, on_attempt=main._record_gemini_call)
	assert result == 'gemini json'


def test_gemini_summarise_wrapper_calls_scorer_gemini_in_live_mode(mocker, monkeypatch):
	# Same as above for the summariser-facing wrapper.
	monkeypatch.setattr(main, 'DRY_RUN', False)
	mock_scorer_gemini = mocker.patch('main.scorer.gemini', return_value='summary json')

	result = main._gemini_summarise('prompt text')

	mock_scorer_gemini.assert_called_once_with('prompt text', 'fake', 3, on_attempt=main._record_gemini_call)
	assert result == 'summary json'


def test_gemini_score_wrapper_increments_call_count(monkeypatch):
	monkeypatch.setattr(main, 'DRY_RUN', True)
	main.GEMINI_CALL_COUNT = 0
	main._gemini_score('prompt')
	main._gemini_score('prompt')
	assert main.GEMINI_CALL_COUNT == 2


def test_gemini_score_wrapper_does_not_increment_when_scorer_is_mocked(monkeypatch, mocker):
	# When scorer.gemini is mocked, the new on_attempt callback never fires inside the (replaced) retry loop; real per-attempt firing is in test_scorer.py.
	monkeypatch.setattr(main, 'DRY_RUN', False)
	mocker.patch('main.scorer.gemini', side_effect=Exception('boom'))
	main.GEMINI_CALL_COUNT = 0
	with pytest.raises(Exception, match='boom'):
		main._gemini_score('prompt')
	assert main.GEMINI_CALL_COUNT == 0


def test_gemini_score_wrapper_passes_record_callback_as_on_attempt(monkeypatch, mocker):
	monkeypatch.setattr(main, 'DRY_RUN', False)
	mock_gemini = mocker.patch('main.scorer.gemini', return_value='ok')
	main._gemini_score('prompt')
	assert mock_gemini.call_args.kwargs.get('on_attempt') is main._record_gemini_call


def test_gemini_summarise_wrapper_passes_record_callback_as_on_attempt(monkeypatch, mocker):
	monkeypatch.setattr(main, 'DRY_RUN', False)
	mock_gemini = mocker.patch('main.scorer.gemini', return_value='ok')
	main._gemini_summarise('prompt')
	assert mock_gemini.call_args.kwargs.get('on_attempt') is main._record_gemini_call


def test_record_gemini_call_increments_count(monkeypatch):
	main.GEMINI_CALL_COUNT = 0
	main._record_gemini_call()
	main._record_gemini_call()
	main._record_gemini_call()
	assert main.GEMINI_CALL_COUNT == 3


def test_record_gemini_call_raises_when_at_cap_and_does_not_increment(mocker):
	# Per-attempt check happens inside _record_gemini_call so a 429 retry storm trips the budget on the next attempt rather than several past it.
	mocker.patch('main.GEMINI_CALL_BUDGET', 3)
	main.GEMINI_CALL_COUNT = 3

	with pytest.raises(main.GeminiBudgetExhausted, match='budget 3'):
		main._record_gemini_call()

	assert main.GEMINI_CALL_COUNT == 3


# -- summariser orchestration --


def test_live_run_summarises_each_enriched_paper(mock_db, mock_io):
	main.main([])
	mock_io['summarise'].assert_called_once()
	call = mock_io['summarise'].call_args
	assert call.args[0]['paperId'] == 'p1'
	# Wired with summariser's gemini wrapper, database_url (summariser manages its own scoped sessions), api_key, and the call-count recorder.
	assert call.args[1] is main._gemini_summarise
	assert call.kwargs['database_url'] == 'postgresql://fake'
	assert call.kwargs['api_key'] == 'fake'
	assert call.kwargs['on_gemini_call'] is main._record_gemini_call


def test_live_run_summarises_only_papers_above_threshold(mock_db, mock_io):
	# Scorer returns a single enriched paper; even if more were scored, only those past the relevance filter should be summarised.
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
	# After summarise_paper returns the four-field dict, the paper passed to the email builder should carry those fields. Verify via the emailed dict.
	emailed_paper = mock_io['summarise'].call_args.args[0]
	assert emailed_paper['methodology'] == 'm'
	assert emailed_paper['findings'] == 'f'
	assert emailed_paper['relevance'] == 'r'
	assert emailed_paper['limitations'] == 'l'


def test_live_run_handles_summariser_returning_none(mock_db, mock_io):
	# A complete summariser failure for a paper must not break the run; the email still goes out and the run still finishes.
	mock_io['summarise'].return_value = None
	main.main([])
	mock_io['send'].assert_called_once()
	mock_db['finish_run'].assert_called_once()


# -- gemini call count + budget --


def test_live_run_prints_total_gemini_call_count(mock_db, mock_io, capsys):
	# The on_gemini_call callback fires inside summariser; simulate it firing on each summarise_paper call to confirm main accumulates the count.
	def fake_summarise(paper, gemini_fn, database_url, api_key, on_gemini_call):
		on_gemini_call()
		return {'methodology': 'm', 'findings': 'f', 'relevance': 'r', 'limitations': 'l'}

	mock_io['summarise'].side_effect = fake_summarise
	main.main([])
	out = capsys.readouterr().out
	assert 'Total Gemini calls this run:' in out


def test_live_run_logs_budget_exhausted_message_when_count_reaches_cap(mock_db, mock_io, mocker, capsys):
	mocker.patch('main.GEMINI_CALL_BUDGET', 1)

	def fake_summarise(paper, gemini_fn, database_url, api_key, on_gemini_call):
		on_gemini_call()
		return {'methodology': 'm', 'findings': 'f', 'relevance': 'r', 'limitations': 'l'}

	mock_io['summarise'].side_effect = fake_summarise
	main.main([])
	assert 'Gemini budget exhausted' in capsys.readouterr().out


def test_live_run_does_not_log_budget_exhausted_when_under_cap(mock_db, mock_io, mocker, capsys):
	mocker.patch('main.GEMINI_CALL_BUDGET', 1_000_000)
	main.main([])
	assert 'Gemini budget exhausted' not in capsys.readouterr().out


def test_live_run_resets_gemini_call_count_between_runs(mock_db, mock_io):
	# A stale counter from a prior run would cause the budget to misfire on the next.
	main.GEMINI_CALL_COUNT = 999
	main.main([])
	# After the run, count reflects only this run's calls (scorer is mocked so _gemini_score wasn't called; summariser was mocked so on_gemini_call didn't fire).
	assert main.GEMINI_CALL_COUNT < 999


# -- db session lifecycle --


def test_main_opens_a_session_per_pipeline_boundary(mock_db, mock_io):
	# Phase D sessions are now owned by summariser.summarise_paper (mocked here); main opens A (init+upserts+needs_scoring) + C (mark) + E (finish) = 3.
	main.main([])
	assert mock_db['session'].call_count == 3


def test_main_skips_phase_c_session_when_nothing_needs_scoring(mock_db, mock_io):
	# needs_scoring returns empty so new_papers=[] (Phase C skipped); scorer also receives [] so nothing is kept and Phase D's summariser sessions don't run.
	mock_db['needs_scoring'].side_effect = lambda conn, ids: set()
	mock_io['score'].return_value = ([], set())
	main.main([])
	# 1 (A) + 0 (C) + 1 (E) = 2.
	assert mock_db['session'].call_count == 2


def test_no_db_session_is_open_during_scoring(mock_db, mock_io):
	# Replace the default session mock with a tracker that counts active opens; snapshot active count at the moment scorer.score_and_summarise is invoked.
	from contextlib import contextmanager

	active = [0]
	active_during_scoring = []

	@contextmanager
	def tracking_session(url):
		active[0] += 1
		try:
			yield mock_db['conn']
		finally:
			active[0] -= 1

	mock_db['session'].side_effect = tracking_session

	score_return = mock_io['score'].return_value

	def snap(*args, **kwargs):
		active_during_scoring.append(active[0])
		return score_return

	mock_io['score'].side_effect = snap

	main.main([])

	# Phase B must hold no DB session, otherwise we burn Neon compute during Gemini retries.
	assert active_during_scoring == [0]


def test_main_propagates_when_a_later_phase_session_fails(mock_db, mock_io):
	# Phase A succeeds; the next session open (Phase C) fails. Verifies the pipeline completes the phases it can and surfaces the failure cleanly.
	from contextlib import contextmanager

	import psycopg

	call_count = [0]

	@contextmanager
	def maybe_failing(url):
		call_count[0] += 1
		if call_count[0] == 1:
			yield mock_db['conn']
		else:
			raise psycopg.OperationalError('reaped')

	mock_db['session'].side_effect = maybe_failing

	with pytest.raises(psycopg.OperationalError, match='reaped'):
		main.main([])

	mock_db['upsert_paper'].assert_called()
	mock_io['score'].assert_called_once()
	mock_db['mark_scoring_results'].assert_not_called()
	mock_io['summarise'].assert_not_called()
	mock_db['finish_run'].assert_not_called()
	mock_io['send'].assert_not_called()


# -- quota exhaustion (RESOURCE_EXHAUSTED) --


def test_summarise_kept_papers_halts_on_quota_exhausted_and_preserves_earlier_fields(mock_db, mock_io, mocker):
	# 3 kept papers; summarise_paper raises GeminiQuotaExhausted on the 2nd.
	import scorer as scorer_module

	# Scores descending so the sort in main() preserves p1/p2/p3 order.
	p1 = {'paperId': 'p1', 'ai_score': 9}
	p2 = {'paperId': 'p2', 'ai_score': 8}
	p3 = {'paperId': 'p3', 'ai_score': 7}
	mock_io['score'].return_value = ([p1, p2, p3], {'p1', 'p2', 'p3'})

	summary = {'methodology': 'm', 'findings': 'f', 'relevance': 'r', 'limitations': 'l'}
	mock_io['summarise'].side_effect = [summary, scorer_module.GeminiQuotaExhausted('quota'), summary]

	main.main([])

	# p1 succeeded, p2 raised, p3 not attempted.
	assert mock_io['summarise'].call_count == 2
	assert p1.get('methodology') == 'm'
	assert 'methodology' not in p2
	assert 'methodology' not in p3


# -- gemini budget enforcement --


def test_gemini_score_raises_budget_exhausted_when_count_at_cap(mocker, monkeypatch):
	# _record_gemini_call fires inside scorer.gemini's retry loop via on_attempt and raises before any post when the counter already sits at the cap.
	monkeypatch.setattr(main, 'DRY_RUN', False)
	mocker.patch('main.GEMINI_CALL_BUDGET', 2)

	def fake_gemini(*args, **kwargs):
		if kwargs.get('on_attempt'):
			kwargs['on_attempt']()
		return 'ok'

	mock_gemini = mocker.patch('main.scorer.gemini', side_effect=fake_gemini)
	main.GEMINI_CALL_COUNT = 2

	with pytest.raises(main.GeminiBudgetExhausted, match='budget 2'):
		main._gemini_score('prompt')

	# scorer.gemini was reached, but on_attempt raised before any work happened in its retry loop.
	mock_gemini.assert_called_once()


def test_gemini_score_does_not_increment_when_budget_raises(mocker, monkeypatch):
	monkeypatch.setattr(main, 'DRY_RUN', False)
	mocker.patch('main.GEMINI_CALL_BUDGET', 1)

	def fake_gemini(*args, **kwargs):
		if kwargs.get('on_attempt'):
			kwargs['on_attempt']()
		return 'ok'

	mocker.patch('main.scorer.gemini', side_effect=fake_gemini)
	main.GEMINI_CALL_COUNT = 1

	with pytest.raises(main.GeminiBudgetExhausted):
		main._gemini_score('prompt')

	# _record_gemini_call raises before its own increment, so the counter must stay put.
	assert main.GEMINI_CALL_COUNT == 1


def test_gemini_score_permits_calls_until_count_reaches_cap(mocker, monkeypatch):
	# Budget N means exactly N successful attempts before the next one is refused; on_attempt fires per attempt inside scorer.gemini.
	monkeypatch.setattr(main, 'DRY_RUN', False)
	mocker.patch('main.GEMINI_CALL_BUDGET', 3)

	def fake_gemini(*args, **kwargs):
		if kwargs.get('on_attempt'):
			kwargs['on_attempt']()
		return 'ok'

	mocker.patch('main.scorer.gemini', side_effect=fake_gemini)
	main.GEMINI_CALL_COUNT = 0

	main._gemini_score('p')
	main._gemini_score('p')
	main._gemini_score('p')

	assert main.GEMINI_CALL_COUNT == 3
	with pytest.raises(main.GeminiBudgetExhausted):
		main._gemini_score('p')


def test_summarise_kept_papers_breaks_when_budget_already_reached(mock_db, mock_io, mocker, capsys):
	# Three enriched papers, budget=1 already consumed by scoring; first summarise attempt detects via on_gemini_call and raises before any Gemini work.
	mocker.patch('main.GEMINI_CALL_BUDGET', 1)

	def consume_then_score(papers, gemini_fn):
		main.GEMINI_CALL_COUNT = 1
		return (
			[{'paperId': 'a', 'ai_score': 8}, {'paperId': 'b', 'ai_score': 7}, {'paperId': 'c', 'ai_score': 6}],
			{'a', 'b', 'c'},
		)

	mock_io['score'].side_effect = consume_then_score

	def fake_summarise(paper, gemini_fn, database_url, api_key, on_gemini_call):
		on_gemini_call()
		return {'methodology': 'm', 'findings': 'f', 'relevance': 'r', 'limitations': 'l'}

	mock_io['summarise'].side_effect = fake_summarise

	main.main([])

	# Paper 'a' was attempted; on_gemini_call detected the cap and raised before any summary work. Papers 'b' and 'c' never reached the summariser.
	mock_io['summarise'].assert_called_once()
	mock_io['send'].assert_called_once()
	assert 'Gemini budget (1) reached; 3 paper(s) will be emailed without full summaries' in capsys.readouterr().out


def test_summarise_kept_papers_stops_mid_loop_when_count_crosses_cap(mock_db, mock_io, mocker, capsys):
	# Budget=2, scoring burned 0, summariser fires on_gemini_call once per paper; third call raises in-place before doing any work.
	mocker.patch('main.GEMINI_CALL_BUDGET', 2)
	mock_io['score'].return_value = (
		[{'paperId': 'a', 'ai_score': 8}, {'paperId': 'b', 'ai_score': 7}, {'paperId': 'c', 'ai_score': 6}],
		{'a', 'b', 'c'},
	)

	def fake_summarise(paper, gemini_fn, database_url, api_key, on_gemini_call):
		on_gemini_call()
		return {'methodology': 'm', 'findings': 'f', 'relevance': 'r', 'limitations': 'l'}

	mock_io['summarise'].side_effect = fake_summarise

	main.main([])

	# All three papers reach the summariser; the third call's on_gemini_call raises before producing a summary.
	assert mock_io['summarise'].call_count == 3
	mock_io['send'].assert_called_once()
	assert '1 paper(s) will be emailed without full summaries' in capsys.readouterr().out


def test_live_run_does_not_crash_when_budget_zero(mock_db, mock_io, mocker):
	# Defensive: if the budget is mis-set to 0, scorer batches get refused, summariser never runs, the pipeline still finishes the run cleanly.
	mocker.patch('main.GEMINI_CALL_BUDGET', 0)
	mock_io['score'].return_value = ([], set())

	main.main([])

	mock_db['finish_run'].assert_called_once()
	mock_io['send'].assert_not_called()
