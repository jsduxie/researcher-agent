import argparse
import json
import os
import smtplib
import time
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import requests

from config import (
	BATCH_SIZE,
	DAYS_BACK,
	EMAIL_TEMPLATE,
	MAX_PER_QUERY,
	RELEVANCE_THRESHOLD,
	RESEARCH_CONTEXT,
	SCORER_PROMPT,
	SEARCH_QUERIES,
)

SEMANTIC_SCHOLAR_URL = 'https://api.semanticscholar.org/graph/v1/paper/search'
GEMINI_URL = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent'

DRY_RUN = False
FIXTURES_DIR = Path(__file__).parent / 'tests' / 'fixtures'


def fetch_papers(query):
	if DRY_RUN:
		with open(FIXTURES_DIR / 'papers.json') as f:
			return json.load(f)

	cutoff = (datetime.now() - timedelta(days=DAYS_BACK)).year
	params = {
		'query': query,
		'limit': MAX_PER_QUERY,
		'fields': 'title,abstract,authors,year,citationCount,externalIds,openAccessPdf,url,publicationDate',
		'publicationDateOrYear': f'{cutoff}-',
	}
	try:
		time.sleep(2)
		r = requests.get(SEMANTIC_SCHOLAR_URL, params=params, timeout=15)
		r.raise_for_status()
		return r.json().get('data', [])
	except Exception as e:
		print(f'Error fetching "{query}": {e}')
		return []


def gemini(prompt, retries=3):
	if DRY_RUN:
		with open(FIXTURES_DIR / 'gemini_score.json') as f:
			return json.dumps(json.load(f))

	headers = {'Content-Type': 'application/json'}
	body = {'contents': [{'parts': [{'text': prompt}]}]}

	for attempt in range(retries):
		time.sleep(5)
		r = requests.post(f'{GEMINI_URL}?key={os.environ["GEMINI_API_KEY"]}', headers=headers, json=body, timeout=120)
		if r.status_code == 429:
			wait = 15 * (attempt + 1)
			print(f'Gemini rate limited, waiting {wait}s')
			time.sleep(wait)
			continue
		r.raise_for_status()
		return r.json()['candidates'][0]['content']['parts'][0]['text'].strip()

	raise Exception('Gemini failed after retries')


def dedup_papers(papers):
	seen, unique = set(), []
	for p in papers:
		pid = p.get('paperId') or p.get('title', '')
		if pid not in seen:
			seen.add(pid)
			unique.append(p)
	return unique


def parse_gemini_scores(response_text):
	cleaned = response_text.replace('```json', '').replace('```', '').strip()
	return json.loads(cleaned)


def apply_scores(papers, scores, threshold):
	results_by_index = {r['index']: r for r in scores}
	enriched = []
	for i, paper in enumerate(papers):
		title = paper.get('title', '')[:60]
		data = results_by_index.get(i)
		if data is None:
			print(f'Missing result for [{i}]: {title}')
			continue
		score = data.get('relevance_score', 0)
		if score < threshold:
			print(f' Dropped (score {score}/10): {title}')
			continue
		paper['ai_score'] = score
		paper['ai_reason'] = data['relevance_reason']
		paper['ai_summary'] = data['summary']
		paper['ai_contribution'] = data['key_contribution']
		print(f'Kept (score {score}/10): {title}')
		enriched.append(paper)
	return enriched


def score_and_summarise(papers):
	if not papers:
		return []

	enriched = []
	for chunk_start in range(0, len(papers), BATCH_SIZE):
		chunk = papers[chunk_start : chunk_start + BATCH_SIZE]
		print(f'Scoring batch {chunk_start // BATCH_SIZE + 1} ({len(chunk)} papers)...')
		enriched.extend(_score_chunk(chunk))

	return enriched


def _score_chunk(papers):
	if not papers:
		return []

	paper_entries = []
	for i, p in enumerate(papers):
		title = p.get('title', '')
		abstract = p.get('abstract') or 'No abstract available.'
		paper_entries.append(f'[{i}] Title: {title}\nAbstract: {abstract}')

	papers_block = '\n\n'.join(paper_entries)

	prompt = SCORER_PROMPT.format(research_context=RESEARCH_CONTEXT, papers_block=papers_block)

	try:
		response = gemini(prompt)
		results = parse_gemini_scores(response)
	except Exception as e:
		print(f'Batch Gemini error: {e}')
		return []

	return apply_scores(papers, results, RELEVANCE_THRESHOLD)


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
