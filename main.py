import argparse
import json
import os
from pathlib import Path

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

	all_papers = []
	for query in SEARCH_QUERIES:
		print(f'\nSearching: {query}')
		all_papers.extend(_fetch(query))

	unique = dedup_papers(all_papers)
	print(f'{len(unique)} unique papers found. Scoring\n')

	enriched = scorer.score_and_summarise(unique, _gemini)
	enriched.sort(key=lambda p: (p.get('ai_score') or 0, p.get('citationCount') or 0), reverse=True)
	print(f'{len(enriched)} papers passed the relevance filter (>={RELEVANCE_THRESHOLD}/10).')

	if not enriched:
		print('No relevant papers, skipping email.')
		return

	_send(build_email(enriched), len(enriched))


if __name__ == '__main__':
	main()
