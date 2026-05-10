import argparse
import json
import os
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import scorer
from config import EMAIL_TEMPLATE, RELEVANCE_THRESHOLD, SEARCH_QUERIES
from fetcher import dedup_papers
from fetcher import fetch_papers as _live_fetch_papers
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


SCORE_COLOURS = {9: '#059669', 7: '#2563eb', 0: '#64748b'}


def score_colour(score):
	if not isinstance(score, int):
		return '#94a3b8'
	for threshold, colour in sorted(SCORE_COLOURS.items(), reverse=True):
		if score >= threshold:
			return colour
	return '#94a3b8'


def _format_authors(authors):
	names = ', '.join(a['name'] for a in authors[:3])
	if len(authors) > 3:
		names += ' et al.'
	return names


def _build_links(paper):
	links = []
	url = paper.get('url', '')
	doi = (paper.get('externalIds') or {}).get('DOI')
	pdf = (paper.get('openAccessPdf') or {}).get('url', '')
	if url:
		links.append({'url': url, 'label': 'Semantic Scholar'})
	if doi:
		links.append({'url': f'https://doi.org/{doi}', 'label': 'DOI'})
	if pdf:
		links.append({'url': pdf, 'label': 'PDF'})
	return links


def _paper_render_data(paper):
	score = paper.get('ai_score', 'N/A')
	return {
		'title': paper.get('title', 'No title'),
		'authors_display': _format_authors(paper.get('authors') or []),
		'pub_date': paper.get('publicationDate') or str(paper.get('year') or 'N/A'),
		'citation_count': paper.get('citationCount', 0),
		'score_colour_value': score_colour(score),
		'score_display': score if isinstance(score, int) else '?',
		'ai_summary': paper.get('ai_summary', ''),
		'ai_contribution': paper.get('ai_contribution', ''),
		'ai_reason': paper.get('ai_reason', ''),
		'links': _build_links(paper),
	}


def build_email(papers):
	today = datetime.now().strftime('%B %d, %Y')
	rendered = [_paper_render_data(p) for p in papers]
	return EMAIL_TEMPLATE.render(papers=rendered, count=len(papers), today=today)


def send_email(html, paper_count):
	if DRY_RUN:
		print(html)
		return

	msg = MIMEMultipart('alternative')
	msg['Subject'] = f'Research Digest: {paper_count} relevant papers — {datetime.now().strftime("%b %d")}'
	msg['From'] = os.environ['GMAIL_USER']
	msg['To'] = os.environ['EMAIL_TO']
	msg.attach(MIMEText(html, 'html'))

	with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
		server.login(os.environ['GMAIL_USER'], os.environ['GMAIL_APP_PASSWORD'])
		server.sendmail(os.environ['GMAIL_USER'], os.environ['EMAIL_TO'], msg.as_string())

	print(f'Email sent with {paper_count} papers.')


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
