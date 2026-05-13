import argparse
import json
import os
import sys
from pathlib import Path

import db
import emailer
import scorer
import summariser
from config import GEMINI_CALL_BUDGET, RELEVANCE_THRESHOLD, SEARCH_QUERIES
from fetcher import FetchError, dedup_papers, fetch_papers
from render import build_email
from scorer import GeminiBudgetExhausted

_FIXTURES = Path(__file__).parent / 'tests' / 'fixtures'
DRY_RUN = False
GEMINI_CALL_COUNT = 0

# Contract enforced by tests/test_workflow_env.py against the send-digest step.
REQUIRED_ENV_VARS = (
	'SEMANTIC_SCHOLAR_API_KEY',
	'DATABASE_URL',
	'GEMINI_API_KEY',
	'GMAIL_USER',
	'GMAIL_APP_PASSWORD',
	'EMAIL_TO',
)


def _fetch(query, api_key):
	if DRY_RUN:
		return json.loads((_FIXTURES / 'papers.json').read_text())
	return fetch_papers(query, api_key)


def _collect_papers(api_key):
	all_papers = []
	attempted = 0
	errored = 0
	for query in SEARCH_QUERIES:
		print(f'\nSearching: {query}')
		attempted += 1
		try:
			all_papers.extend(_fetch(query, api_key))
		except FetchError as e:
			print(f'Error fetching "{query}": {e}')
			errored += 1
	return all_papers, attempted, errored


def _gemini_score(prompt, retries=3):
	if DRY_RUN:
		_record_gemini_call()
		return (_FIXTURES / 'gemini_score.json').read_text()
	return scorer.gemini(prompt, os.environ['GEMINI_API_KEY'], retries, on_attempt=_record_gemini_call)


def _gemini_summarise(prompt, retries=3):
	if DRY_RUN:
		_record_gemini_call()
		return (_FIXTURES / 'gemini_summary.json').read_text()
	return scorer.gemini(prompt, os.environ['GEMINI_API_KEY'], retries, on_attempt=_record_gemini_call)


def _record_gemini_call():
	# Raise per-attempt before incrementing so a 429 retry storm trips the budget on its next attempt, not three or four attempts past it.
	global GEMINI_CALL_COUNT
	if GEMINI_CALL_COUNT >= GEMINI_CALL_BUDGET:
		raise GeminiBudgetExhausted(f'budget {GEMINI_CALL_BUDGET} reached after {GEMINI_CALL_COUNT} calls')
	GEMINI_CALL_COUNT += 1


def _summarise_kept_papers(enriched, database_url):
	# summariser.summarise_paper manages two short DB sessions (cache check, then persist); no DB open during the Gemini work in between.
	api_key = None if DRY_RUN else os.environ.get('GEMINI_API_KEY')
	skipped = 0
	for i, paper in enumerate(enriched):
		try:
			fields = summariser.summarise_paper(
				paper, _gemini_summarise, database_url=database_url, api_key=api_key, on_gemini_call=_record_gemini_call
			)
		except GeminiBudgetExhausted as e:
			# _record_gemini_call raises this when the next attempt would exceed the cap; remaining papers are emailed without summaries.
			skipped = len(enriched) - i
			print(f'Budget exhausted, halting summarisation: {e}')
			break
		except scorer.GeminiQuotaExhausted as e:
			print(f'Quota exhausted, halting summarisation: {e}')
			break
		if fields:
			paper.update(fields)
	if skipped:
		print(
			f'Gemini budget ({GEMINI_CALL_BUDGET}) reached; {skipped} paper(s) will be emailed without full summaries'
		)


def _report_gemini_usage():
	print(f'Total Gemini calls this run: {GEMINI_CALL_COUNT}/{GEMINI_CALL_BUDGET}')
	if GEMINI_CALL_COUNT >= GEMINI_CALL_BUDGET:
		print(f'Gemini budget exhausted (cap {GEMINI_CALL_BUDGET})')


def _send(html, paper_count):
	if DRY_RUN:
		print(html)
		return
	creds = emailer.SmtpCredentials(os.environ['GMAIL_USER'], os.environ['GMAIL_APP_PASSWORD'], os.environ['EMAIL_TO'])
	emailer.send_email(html, paper_count, creds)


def main(argv=None):
	parser = argparse.ArgumentParser()
	parser.add_argument('--dry-run', action='store_true', help='use fixtures and print HTML, no network or email')
	global DRY_RUN, GEMINI_CALL_COUNT
	DRY_RUN = parser.parse_args(argv).dry_run
	GEMINI_CALL_COUNT = 0

	run_id = None
	api_key = None
	database_url = None
	if not DRY_RUN:
		missing = [v for v in REQUIRED_ENV_VARS if not os.environ.get(v)]
		if missing:
			sys.exit(f'{", ".join(missing)} required; set in the environment or GitHub Actions secrets.')
		api_key = os.environ['SEMANTIC_SCHOLAR_API_KEY']
		database_url = os.environ['DATABASE_URL']

	all_papers, queries_attempted, queries_errored = _collect_papers(api_key)

	unique = dedup_papers(all_papers)
	# Papers without paperId can't be persisted (paper_id is the PK) or deduplicated across runs; drop up front rather than crashing in upsert.
	missing_id = [p for p in unique if not p.get('paperId')]
	if missing_id:
		print(f'Dropped {len(missing_id)} paper(s) without paperId (cannot persist)')
	unique = [p for p in unique if p.get('paperId')]

	# Phase A: schema, run row, paper upserts, needs_scoring filter. One scoped session.
	if database_url is not None:
		with db.session(database_url) as conn:
			db.init_schema(conn)
			run_id = db.start_run(conn, len(unique))
			for p in unique:
				db.upsert_paper(conn, p)
			unscored = db.needs_scoring(conn, [p['paperId'] for p in unique])
			new_papers = [p for p in unique if p['paperId'] in unscored]
	else:
		new_papers = unique

	print(f'{len(unique)} unique papers found, {len(new_papers)} need scoring. Scoring\n')

	# Phase B: scoring. No DB connection held during the Gemini work.
	enriched, responded = scorer.score_and_summarise(new_papers, _gemini_score)
	enriched.sort(key=lambda p: (p.get('ai_score') or 0, p.get('citationCount') or 0), reverse=True)
	print(f'{len(enriched)} papers passed the relevance filter (>={RELEVANCE_THRESHOLD}/10).')

	# Phase C: record what was attempted/responded.
	if database_url is not None and new_papers:
		with db.session(database_url) as conn:
			db.mark_scoring_results(conn, attempted=[p['paperId'] for p in new_papers], responded=responded)

	# Phase D: per-paper summarisation, each in its own session.
	_summarise_kept_papers(enriched, database_url)
	_report_gemini_usage()

	if not enriched:
		print('No relevant papers, skipping email.')
	else:
		_send(build_email(enriched), len(enriched))

	# Phase E: finish_run last so a logging failure never silently skips the email.
	if database_url is not None:
		with db.session(database_url) as conn:
			db.finish_run(conn, run_id, len(enriched), queries_attempted, queries_errored)

	if queries_attempted > 0 and queries_attempted == queries_errored:
		sys.exit(f'All {queries_attempted} queries errored; no papers retrieved this run.')


if __name__ == '__main__':
	main()
