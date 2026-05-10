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


S = {
	'wrapper': 'max-width:640px; margin:0 auto; padding:32px 16px; font-family:Roboto,system-ui,-apple-system,sans-serif; background:#f0f4f8; color:#1e293b;',
	'header': 'background:#1e3a5f; border-radius:12px 12px 0 0; padding:32px 28px 24px 28px;',
	'header_label': 'font-size:11px; font-weight:600; letter-spacing:1.5px; text-transform:uppercase; color:#93c5fd; margin:0 0 8px 0;',
	'header_title': 'font-size:22px; font-weight:700; color:#ffffff; line-height:1.3; margin:0;',
	'header_count_box': 'background:rgba(255,255,255,0.15); border-radius:8px; padding:10px 14px; display:inline-block; text-align:center;',
	'header_count_num': 'font-size:22px; font-weight:700; color:#ffffff; line-height:1;',
	'header_count_label': 'font-size:10px; color:#93c5fd; text-transform:uppercase; letter-spacing:0.5px; margin-top:2px;',
	'subheader': 'background:#ffffff; border-left:1px solid #e2e8f0; border-right:1px solid #e2e8f0; padding:16px 28px; border-bottom:1px solid #e2e8f0;',
	'subheader_text': 'font-size:12px; color:#64748b; margin:0;',
	'card': 'background:#ffffff; border-left:1px solid #e2e8f0; border-right:1px solid #e2e8f0; padding:24px 28px; border-bottom:1px solid #e2e8f0;',
	'card_last': 'background:#ffffff; border-left:1px solid #e2e8f0; border-right:1px solid #e2e8f0; border-bottom:1px solid #e2e8f0; border-radius:0 0 12px 12px; padding:24px 28px;',
	'score_badge': 'color:#ffffff; font-size:13px; font-weight:700; text-align:center; border-radius:8px; padding:6px 0; width:40px;',
	'card_title': 'font-size:15px; font-weight:600; color:#0f172a; line-height:1.45; margin:0;',
	'card_meta': 'font-size:12px; color:#94a3b8; margin-top:5px;',
	'summary_box': 'background:#f8fafc; border-radius:8px; padding:14px 16px; margin-top:14px; border:1px solid #e2e8f0;',
	'summary_text': 'font-size:13px; color:#334155; line-height:1.65; margin:0;',
	'summary_text_mt': 'font-size:13px; color:#334155; line-height:1.65; margin:10px 0 0 0;',
	'summary_label': 'font-weight:600; color:#1e3a5f;',
	'reason': 'font-size:12px; color:#94a3b8; margin:10px 0 0 0; font-style:italic;',
	'link_pill': 'display:inline-block; font-size:12px; font-weight:500; color:#2563eb; text-decoration:none; padding:4px 10px; border:1px solid #bfdbfe; border-radius:6px; margin-right:6px; margin-top:4px;',
	'links_row': 'margin-top:12px;',
	'footer': 'text-align:center; padding:20px 0 8px 0;',
	'footer_text': 'font-size:11px; color:#94a3b8;',
	'empty': 'background:#ffffff; border:1px solid #e2e8f0; border-radius:0 0 12px 12px; padding:40px 28px; text-align:center; color:#94a3b8; font-size:14px;',
}

SCORE_COLOURS = {9: '#059669', 7: '#2563eb', 0: '#64748b'}


def score_colour(score):
	if not isinstance(score, int):
		return '#94a3b8'
	for threshold, colour in sorted(SCORE_COLOURS.items(), reverse=True):
		if score >= threshold:
			return colour
	return '#94a3b8'


def paper_links_html(paper):
	url = paper.get('url', '')
	doi = (paper.get('externalIds') or {}).get('DOI')
	pdf = (paper.get('openAccessPdf') or {}).get('url', '')

	links = []
	if url:
		links.append(f'<a href="{url}" style="{S["link_pill"]}">Semantic Scholar</a>')
	if doi:
		links.append(f'<a href="https://doi.org/{doi}" style="{S["link_pill"]}">DOI</a>')
	if pdf:
		links.append(f'<a href="{pdf}" style="{S["link_pill"]}">PDF</a>')

	if not links:
		return ''
	return f'<div style="{S["links_row"]}">{"".join(links)}</div>'


def paper_card_html(paper, is_last=False):
	title = paper.get('title', 'No title')
	authors = ', '.join(a['name'] for a in (paper.get('authors') or [])[:3])
	if len(paper.get('authors') or []) > 3:
		authors += ' et al.'
	pub_date = paper.get('publicationDate') or str(paper.get('year') or 'N/A')
	citations = paper.get('citationCount', 0)
	score = paper.get('ai_score', 'N/A')
	colour = score_colour(score)
	score_display = score if isinstance(score, int) else '?'

	card_style = S['card_last'] if is_last else S['card']
	badge_style = f'{S["score_badge"]} background:{colour};'

	summary = paper.get('ai_summary', '')
	contribution = paper.get('ai_contribution', '')
	reason = paper.get('ai_reason', '')
	links = paper_links_html(paper)

	return f"""<div style="{card_style}">
  <table width="100%" cellpadding="0" cellspacing="0" border="0"><tr>
    <td style="width:40px; vertical-align:top; padding-right:14px;">
      <div style="{badge_style}">{score_display}</div>
    </td>
    <td style="vertical-align:top;">
      <div style="{S['card_title']}">{title}</div>
      <div style="{S['card_meta']}">{authors} &nbsp;&middot;&nbsp; {pub_date} &nbsp;&middot;&nbsp; {citations} citations</div>
    </td>
  </tr></table>
  <div style="{S['summary_box']}">
    <div style="{S['summary_text']}"><span style="{S['summary_label']}">Summary</span> &mdash; {summary}</div>
    <div style="{S['summary_text_mt']}"><span style="{S['summary_label']}">Key takeaway</span> &mdash; {contribution}</div>
  </div>
  <div style="{S['reason']}">{reason}</div>
  {links}
</div>"""


def build_email(papers):
	today = datetime.now().strftime('%B %d, %Y')
	count = len(papers)

	cards_html = ''
	if papers:
		for i, p in enumerate(papers):
			cards_html += paper_card_html(p, is_last=(i == len(papers) - 1))
	else:
		cards_html = f'<div style="{S["empty"]}">No relevant papers found this period.</div>'

	return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Research Digest</title>
<link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;600;700&display=swap" rel="stylesheet">
</head>
<body style="margin:0; padding:0; background:#f0f4f8;">
<div style="{S['wrapper']}">

  <div style="{S['header']}">
    <table width="100%" cellpadding="0" cellspacing="0" border="0"><tr>
      <td>
        <div style="{S['header_label']}">Research Digest</div>
        <div style="{S['header_title']}">BPD Detection &amp; Mental Health NLP</div>
      </td>
      <td style="text-align:right; vertical-align:top;">
        <div style="{S['header_count_box']}">
          <div style="{S['header_count_num']}">{count}</div>
          <div style="{S['header_count_label']}">papers</div>
        </div>
      </td>
    </tr></table>
  </div>

  <div style="{S['subheader']}">
    <p style="{S['subheader_text']}">{today} &nbsp;&middot;&nbsp; Scored &amp; filtered &ge;{RELEVANCE_THRESHOLD}/10 &nbsp;&middot;&nbsp; Summaries by Gemini 2.5 Flash</p>
  </div>

  {cards_html}

  <div style="{S['footer']}">
    <span style="{S['footer_text']}">Semantic Scholar &nbsp;&middot;&nbsp; Gemini 2.5 Flash &nbsp;&middot;&nbsp; GitHub Actions</span>
  </div>

</div>
</body></html>"""


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
