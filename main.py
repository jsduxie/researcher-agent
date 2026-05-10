import argparse
import json
import os
from pathlib import Path

import db
import emailer
import scorer
from config import RELEVANCE_THRESHOLD, SEARCH_QUERIES
from fetcher import dedup_papers, fetch_papers
from render import build_email

_FIXTURES = Path(__file__).parent / 'tests' / 'fixtures'
DRY_RUN = False


def _fetch(query):
	if DRY_RUN:
		return json.loads((_FIXTURES / 'papers.json').read_text())
	return fetch_papers(query)


def _gemini(prompt, retries=3):
	if DRY_RUN:
		return (_FIXTURES / 'gemini_score.json').read_text()
	return scorer.gemini(prompt, os.environ['GEMINI_API_KEY'], retries)


def _send(html, paper_count):
	if DRY_RUN:
		print(html)
		return
	creds = emailer.SmtpCredentials(os.environ['GMAIL_USER'], os.environ['GMAIL_APP_PASSWORD'], os.environ['EMAIL_TO'])
	emailer.send_email(html, paper_count, creds)


def main(argv=None):
	parser = argparse.ArgumentParser()
	parser.add_argument('--dry-run', action='store_true', help='use fixtures and print HTML, no network or email')
	global DRY_RUN
	DRY_RUN = parser.parse_args(argv).dry_run

	conn = None
	run_id = None
	if not DRY_RUN:
		conn = db.connect(os.environ['DATABASE_URL'])
		db.init_schema(conn)

	all_papers = []
	for query in SEARCH_QUERIES:
		print(f'\nSearching: {query}')
		all_papers.extend(_fetch(query))

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

	enriched, responded = scorer.score_and_summarise(new_papers, _gemini)
	enriched.sort(key=lambda p: (p.get('ai_score') or 0, p.get('citationCount') or 0), reverse=True)
	print(f'{len(enriched)} papers passed the relevance filter (>={RELEVANCE_THRESHOLD}/10).')

	if conn is not None and new_papers:
		db.mark_scoring_results(conn, attempted=[p['paperId'] for p in new_papers], responded=responded)

	if not enriched:
		print('No relevant papers, skipping email.')
	else:
		_send(build_email(enriched), len(enriched))

	# finish_run last so a logging failure never silently skips the email.
	if conn is not None:
		db.finish_run(conn, run_id, len(enriched))


if __name__ == '__main__':
	main()
