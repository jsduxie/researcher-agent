import argparse
import json
import os
from pathlib import Path

import emailer
import scorer
from config import RELEVANCE_THRESHOLD, SEARCH_QUERIES
from fetcher import dedup_papers
from fetcher import fetch_papers as _live_fetch_papers
from render import build_email
from scorer import apply_scores, parse_gemini_scores  # noqa: F401  (re-exported for test_score_parsing)

DRY_RUN = False
FIXTURES_DIR = Path(__file__).parent / 'tests' / 'fixtures'


def fetch_papers(query):
	if DRY_RUN:
		with open(FIXTURES_DIR / 'papers.json') as f:
			return json.load(f)
	return _live_fetch_papers(query)


def gemini(prompt, retries=3):
	if DRY_RUN:
		with open(FIXTURES_DIR / 'gemini_score.json') as f:
			return json.dumps(json.load(f))
	return scorer.gemini(prompt, os.environ['GEMINI_API_KEY'], retries)


def score_and_summarise(papers):
	return scorer.score_and_summarise(papers, gemini)


def send_email(html, paper_count):
	if DRY_RUN:
		print(html)
		return
	creds = emailer.SmtpCredentials(
		user=os.environ['GMAIL_USER'], password=os.environ['GMAIL_APP_PASSWORD'], to=os.environ['EMAIL_TO']
	)
	emailer.send_email(html, paper_count, creds)


def main(argv=None):
	parser = argparse.ArgumentParser()
	parser.add_argument('--dry-run', action='store_true', help='use fixtures and print HTML, no network or email')
	args = parser.parse_args(argv)

	global DRY_RUN
	DRY_RUN = args.dry_run

	all_papers = []
	for query in SEARCH_QUERIES:
		print(f'\nSearching: {query}')
		all_papers.extend(fetch_papers(query))

	unique = dedup_papers(all_papers)

	print(f'{len(unique)} unique papers found. Scoring\n')

	enriched = score_and_summarise(unique)
	enriched.sort(key=lambda p: (p.get('ai_score') or 0, p.get('citationCount') or 0), reverse=True)

	print(f'{len(enriched)} papers passed the relevance filter (>={RELEVANCE_THRESHOLD}/10).')

	if not enriched:
		print('No relevant papers, skipping email.')
		return

	html = build_email(enriched)
	send_email(html, len(enriched))


if __name__ == '__main__':
	main()
