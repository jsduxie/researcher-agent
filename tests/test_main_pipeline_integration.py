import json
import os

import pytest
import responses

import config
import db
import main
import scorer

# main seeds app_config from the seed file on a clean DB, so the URL its Gemini calls hit is the seed-derived one.
GEMINI_URL = scorer.gemini_url(config.Config(**config.load_seed()))

_DATABASE_URL_TEST = os.environ.get('DATABASE_URL_TEST')

pytestmark = [
	pytest.mark.integration,
	pytest.mark.skipif(
		not _DATABASE_URL_TEST,
		reason='DATABASE_URL_TEST not set; pipeline integration test requires a live Postgres branch',
	),
]


@pytest.fixture
def clean_db():
	# Truncate before each test so state assertions are unambiguous.
	with db.session(_DATABASE_URL_TEST) as conn:
		db.init_schema(conn)
		with conn.cursor() as cur:
			cur.execute(
				'TRUNCATE TABLE summary_feedback, ratings, summaries, paper_authors, runs, papers, '
				'app_config, app_config_history RESTART IDENTITY CASCADE'
			)


@pytest.fixture
def env(monkeypatch):
	# Every var main checks must be present, but only DATABASE_URL needs to be real.
	monkeypatch.setenv('DATABASE_URL', _DATABASE_URL_TEST)
	monkeypatch.setenv('SEMANTIC_SCHOLAR_API_KEY', 'fake-ss-key')
	monkeypatch.setenv('GEMINI_API_KEY', 'fake-gemini-key')
	monkeypatch.setenv('GMAIL_USER', 'fake@example.com')
	monkeypatch.setenv('GMAIL_APP_PASSWORD', 'fake')
	monkeypatch.setenv('EMAIL_TO', 'recipient@example.com')


def _gemini_envelope(text):
	return {'candidates': [{'content': {'parts': [{'text': text}]}}]}


def _gemini_callback(request):
	# Route by prompt content: scorer wants relevance_score; summariser wants the four content fields.
	body = json.loads(request.body)
	prompt = body['contents'][0]['parts'][0]['text']
	if 'relevance_score' in prompt:
		payload = json.dumps(
			[
				{'index': 0, 'relevance_score': 8, 'relevance_reason': 'reason1'},
				{'index': 1, 'relevance_score': 9, 'relevance_reason': 'reason2'},
			]
		)
	else:
		payload = json.dumps({'methodology': 'm', 'findings': 'f', 'relevance_to_research': 'r', 'limitations': 'l'})
	return (200, {}, json.dumps(_gemini_envelope(payload)))


@responses.activate
def test_main_runs_end_to_end_with_real_db_and_http_mocked_gemini(clean_db, env, mocker):
	# Two papers with no openAccessPdf so the summariser stays on the abstract-only path and every Gemini hit lands on GEMINI_URL (not the Files API).
	papers = [
		{'paperId': 'p1', 'title': 'Paper 1', 'abstract': 'Abstract one.', 'citationCount': 5, 'authors': []},
		{'paperId': 'p2', 'title': 'Paper 2', 'abstract': 'Abstract two.', 'citationCount': 3, 'authors': []},
	]
	mocker.patch('main.fetch_papers', return_value=papers)
	mocker.patch('main.emailer.send_email')
	responses.add_callback(responses.POST, GEMINI_URL, callback=_gemini_callback)

	main.main([])

	with db.session(_DATABASE_URL_TEST) as conn, conn.cursor() as cur:
		cur.execute('SELECT COUNT(*) FROM papers')
		assert cur.fetchone()[0] == 2
		cur.execute('SELECT COUNT(*) FROM summaries')
		assert cur.fetchone()[0] == 2
		cur.execute('SELECT COUNT(*) FROM runs WHERE finished_at IS NOT NULL')
		assert cur.fetchone()[0] == 1
		cur.execute('SELECT scored_at FROM papers ORDER BY paper_id')
		assert all(row[0] is not None for row in cur.fetchall())


@responses.activate
def test_main_survives_when_every_gemini_call_returns_429(clean_db, env, mocker):
	# Reproduces the 2026-05-12 failure (every Gemini batch 429s); the scoped-session pipeline must complete cleanly with papers_kept=0 instead of crashing.
	papers = [{'paperId': 'p1', 'title': 'Paper 1', 'abstract': 'Abstract one.', 'citationCount': 5, 'authors': []}]
	mocker.patch('main.fetch_papers', return_value=papers)
	mocker.patch('main.emailer.send_email')
	# Speed: zero out the retry sleeps so the test doesn't spend ~75s in backoff.
	mocker.patch('scorer.time.sleep')
	responses.add(responses.POST, GEMINI_URL, json={'error': 'rate limited'}, status=429)

	# Should not raise.
	main.main([])

	with db.session(_DATABASE_URL_TEST) as conn, conn.cursor() as cur:
		cur.execute('SELECT COUNT(*) FROM papers')
		assert cur.fetchone()[0] == 1
		cur.execute('SELECT COUNT(*) FROM runs WHERE finished_at IS NOT NULL AND papers_kept = 0')
		assert cur.fetchone()[0] == 1
		# Scorer was attempted but nothing responded; score_attempts bumped, scored_at still NULL.
		cur.execute('SELECT score_attempts, scored_at FROM papers WHERE paper_id = %s', ('p1',))
		attempts, scored_at = cur.fetchone()
		assert attempts == 1
		assert scored_at is None
