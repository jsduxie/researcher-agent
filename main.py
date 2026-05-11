import argparse
import json
import os
import sys
from pathlib import Path

import db
import emailer
import scorer
import summariser
from config import GEMINI_CALL_WARN_THRESHOLD, RELEVANCE_THRESHOLD, SEARCH_QUERIES
from fetcher import FetchError, dedup_papers, fetch_papers
from render import build_email

_FIXTURES = Path(__file__).parent / 'tests' / 'fixtures'
DRY_RUN = False
GEMINI_CALL_COUNT = 0


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
	global GEMINI_CALL_COUNT
	if DRY_RUN:
		result = (_FIXTURES / 'gemini_score.json').read_text()
	else:
		result = scorer.gemini(prompt, os.environ['GEMINI_API_KEY'], retries)
	# Mirror the summariser's on_gemini_call semantics: increment only after a
	# response body returned, so transport errors do not inflate the counter.
	GEMINI_CALL_COUNT += 1
	return result


def _gemini_summarise(prompt, retries=3):
	if DRY_RUN:
		return (_FIXTURES / 'gemini_summary.json').read_text()
	return scorer.gemini(prompt, os.environ['GEMINI_API_KEY'], retries)


def _record_gemini_call():
	global GEMINI_CALL_COUNT
	GEMINI_CALL_COUNT += 1


def _summarise_kept_papers(enriched, conn):
	# Summariser handles its own cache check against the summaries table, so a paper
	# we already summarised on a prior run short-circuits here. PDF path is skipped
	# in dry-run because api_key resolves to None.
	api_key = None if DRY_RUN else os.environ.get('GEMINI_API_KEY')
	for paper in enriched:
		fields = summariser.summarise_paper(
			paper, _gemini_summarise, conn=conn, api_key=api_key, on_gemini_call=_record_gemini_call
		)
		if fields:
			paper.update(fields)


def _report_gemini_usage():
	print(f'Total Gemini calls this run: {GEMINI_CALL_COUNT}')
	if GEMINI_CALL_COUNT > GEMINI_CALL_WARN_THRESHOLD:
		print(f'WARNING: Gemini call count ({GEMINI_CALL_COUNT}) exceeds threshold ({GEMINI_CALL_WARN_THRESHOLD})')


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

	conn = None
	run_id = None
	api_key = None
	if not DRY_RUN:
		api_key = os.environ.get('SEMANTIC_SCHOLAR_API_KEY')
		if not api_key:
			sys.exit('SEMANTIC_SCHOLAR_API_KEY is required; set it in the environment or GitHub Actions secrets.')
		conn = db.connect(os.environ['DATABASE_URL'])
		db.init_schema(conn)

	all_papers, queries_attempted, queries_errored = _collect_papers(api_key)

	unique = dedup_papers(all_papers)
	# Papers without paperId cannot be persisted (paper_id is the PK) and cannot be
	# deduplicated across runs, so drop them up front rather than crashing in upsert.
	missing_id = [p for p in unique if not p.get('paperId')]
	if missing_id:
		print(f'Dropped {len(missing_id)} paper(s) without paperId (cannot persist)')
	unique = [p for p in unique if p.get('paperId')]

	if conn is not None:
		run_id = db.start_run(conn, len(unique))
		# Always upsert so existing rows refresh metadata (citation counts).
		for p in unique:
			db.upsert_paper(conn, p)
		unscored = db.needs_scoring(conn, [p['paperId'] for p in unique])
		new_papers = [p for p in unique if p['paperId'] in unscored]
	else:
		new_papers = unique

	print(f'{len(unique)} unique papers found, {len(new_papers)} need scoring. Scoring\n')

	enriched, responded = scorer.score_and_summarise(new_papers, _gemini_score)
	enriched.sort(key=lambda p: (p.get('ai_score') or 0, p.get('citationCount') or 0), reverse=True)
	print(f'{len(enriched)} papers passed the relevance filter (>={RELEVANCE_THRESHOLD}/10).')

	if conn is not None and new_papers:
		db.mark_scoring_results(conn, attempted=[p['paperId'] for p in new_papers], responded=responded)

	_summarise_kept_papers(enriched, conn)
	_report_gemini_usage()

	if not enriched:
		print('No relevant papers, skipping email.')
	else:
		_send(build_email(enriched), len(enriched))

	# finish_run last so a logging failure never silently skips the email.
	if conn is not None:
		db.finish_run(conn, run_id, len(enriched), queries_attempted, queries_errored)

	if queries_attempted > 0 and queries_attempted == queries_errored:
		sys.exit(f'All {queries_attempted} queries errored; no papers retrieved this run.')


if __name__ == '__main__':
	main()
