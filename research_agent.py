import requests
import smtplib
import os
import json
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime, timedelta
from textwrap import dedent

import dominate
from dominate import tags
from dominate.util import raw

SEARCH_QUERIES = [
    'Borderline Personality Disorder detection social media',
    'BPD natural language processing Reddit',
    'mental health NLP transformer BERT classification',
    'explainable AI mental health NLP SHAP LIME',
    'personality disorder machine learning detection',
    'Hierarchical Attention Network',
    'Feature Guided Attention',
    'Attention mechanisms text classification',
]

RESEARCH_CONTEXT = dedent('''
    I am completing a research project surrounding the screening of Borderline Personality
    Disorder (BPD/EUPD) from social media data.

    My work involves:
    - Evaluating a variety of machine learning (logistic regression, random forest) and deep
      learning (CNN, BiLSTM, Hierarchical Attention Network, BERT transformers, Hierarchical
      transformers) on the detection of BPD from Reddit posts
    - Compiling an extended dataset of tough classification samples from similar conditions,
      including depression, other personality disorders and CPTSD
    - Evaluating the explainability of model-agnostic (SHAP, LIME) and model-specific methods
      (attention, logistic regression coefficients, gradients) across model architectures
    - Handling severe class imbalance
    - Extending the current work on Hierarchical Attention Networks in line with BPD, by
      incorporating feature-guided attention from NRC-VAD, LIWC and Empath features as an
      additive bias for word- and sentence-level attention to improve performance and
      explainability. These are divided into three heads mirroring core traits identified by
      Sainslow and Southward: Interpersonal/Loneliness, Emotional instability/intensity,
      Impulsive behaviours

    My HAN implementation uses a MentalBERT word-level encoder and a transformer
    sentence-level encoder in place of BiGRUs.

    I am interested in papers relevant to: BPD or other personality disorder detection,
    mental health NLP broadly, transformer architectures for text classification,
    explainability/interpretability in clinical NLP, Reddit/social media mental health
    analysis, hierarchical attention networks and hierarchical transformers.
''')

RELEVANCE_THRESHOLD = 6
DAYS_BACK = 14
MAX_PER_QUERY = 8
BATCH_SIZE = 10

SEMANTIC_SCHOLAR_URL = 'https://api.semanticscholar.org/graph/v1/paper/search'
GEMINI_URL = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent'

def fetch_papers(query):
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
    headers = {'Content-Type': 'application/json'}
    body = {'contents': [{'parts': [{'text': prompt}]}]}
    
    for attempt in range(retries):
        time.sleep(5)
        r = requests.post(
            f'{GEMINI_URL}?key={os.environ["GEMINI_API_KEY"]}',
            headers=headers,
            json=body,
            timeout=120,
        )
        if r.status_code == 429:
            wait = 15 * (attempt + 1)
            print(f'Gemini rate limited, waiting {wait}s')
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r.json()['candidates'][0]['content']['parts'][0]['text'].strip()
    
    raise Exception('Gemini failed after retries')

def score_and_summarise(papers):
    if not papers:
        return []

    enriched = []
    for chunk_start in range(0, len(papers), BATCH_SIZE):
        chunk = papers[chunk_start:chunk_start + BATCH_SIZE]
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

    prompt = dedent(f'''
        You are a research assistant helping a masters student assess papers.

        RESEARCH CONTEXT:
        {RESEARCH_CONTEXT}

        PAPERS:
        {papers_block}

        For EACH paper, assess its relevance to the student's research.
        Respond with a JSON array (no markdown, no extra text) where each element has:
        {{
          "index": <integer matching the [i] label>,
          "relevance_score": <integer 1-10>,
          "relevance_reason": "<one sentence on why it is or is not relevant>",
          "summary": "<2-3 sentences summarising the paper from the angle of the student's research>",
          "key_contribution": "<the single most important takeaway for this student>"
        }}

        Return ONLY the JSON array.
    ''')

    try:
        response = gemini(prompt)
        response = response.replace('```json', '').replace('```', '').strip()
        results = json.loads(response)
    except Exception as e:
        print(f'Batch Gemini error: {e}')
        return []

    enriched = []
    results_by_index = {r['index']: r for r in results}

    for i, paper in enumerate(papers):
        title = paper.get('title', '')[:60]
        data = results_by_index.get(i)

        if data is None:
            print(f'Missing result for [{i}]: {title}')
            continue

        score = data.get('relevance_score', 0)
        if score < RELEVANCE_THRESHOLD:
            print(f' Dropped (score {score}/10): {title}')
            continue

        paper['ai_score'] = score
        paper['ai_reason'] = data['relevance_reason']
        paper['ai_summary'] = data['summary']
        paper['ai_contribution'] = data['key_contribution']
        print(f'Kept (score {score}/10): {title}')
        enriched.append(paper)

    return enriched

S = {
    'body': 'max-width:680px; margin:auto; padding:24px; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; background:#0d1117; color:#e6edf3;',
    'banner': 'border-bottom:1px solid #30363d; padding-bottom:20px; margin-bottom:24px;',
    'banner_h1': 'margin:0 0 4px 0; font-size:20px; font-weight:600; color:#e6edf3; letter-spacing:-0.3px;',
    'banner_sub': 'margin:0; font-size:13px; color:#7d8590;',
    'intro': 'font-size:13px; color:#7d8590; margin-bottom:24px;',
    'footer': 'font-size:12px; color:#484f58; text-align:center;',
    'hr': 'border:none; border-top:1px solid #21262d; margin:24px 0;',
    'card': 'border:1px solid #30363d; border-radius:6px; padding:16px; margin-bottom:16px; background:#161b22;',
    'card_header': 'display:flex; justify-content:space-between; align-items:flex-start; gap:12px;',
    'card_title': 'margin:0; font-size:15px; font-weight:600; color:#e6edf3; flex:1; line-height:1.4;',
    'card_meta': 'margin:6px 0 0 0; font-size:12px; color:#7d8590;',
    'card_body': 'margin:12px 0 10px 0; padding:12px; background:#0d1117; border-radius:4px; border:1px solid #21262d;',
    'card_text': 'margin:0 0 8px 0; font-size:13px; color:#c9d1d9; line-height:1.6;',
    'card_text_last': 'margin:0; font-size:13px; color:#c9d1d9; line-height:1.6;',
    'card_reason': 'margin:0 0 10px 0; font-size:12px; color:#484f58;',
    'card_footer': 'display:flex; gap:10px; align-items:center; flex-wrap:wrap;',
    'badge': 'background:#21262d; border:1px solid #30363d; padding:2px 8px; border-radius:4px; font-size:12px; color:#7d8590;',
    'link': 'color:#58a6ff; text-decoration:none;',
    'links': 'font-size:12px;',
    'score_badge': 'padding:2px 8px; border-radius:4px; font-size:12px; font-weight:600; color:#0d1117;',
}

SCORE_COLOURS = {9: '#3fb950', 7: '#58a6ff', 0: '#d29922'}

SCORE_COLOURS = {9: '#22c55e', 7: '#3b82f6', 0: '#f59e0b'}


def score_colour(score):
    if not isinstance(score, int):
        return '#888888'
    for threshold, colour in sorted(SCORE_COLOURS.items(), reverse=True):
        if score >= threshold:
            return colour
    return '#888888'


def paper_links_tag(paper):
    url = paper.get('url', '')
    doi = (paper.get('externalIds') or {}).get('DOI')
    pdf = (paper.get('openAccessPdf') or {}).get('url', '')

    parts = []
    if url:
        parts.append(tags.a('Semantic Scholar', href=url, style=S['link']))
    if doi:
        parts.append(tags.a('DOI', href=f'https://doi.org/{doi}', style=S['link']))
    if pdf:
        parts.append(tags.a('PDF', href=pdf, style=S['link']))

    container = tags.span(style=S['links'])
    for i, link in enumerate(parts):
        container += link
        if i < len(parts) - 1:
            container += raw(' &middot; ')

    if not parts:
        container += 'No link'

    return container


def paper_card(paper):
    title = paper.get('title', 'No title')
    authors = ', '.join(a['name'] for a in (paper.get('authors') or [])[:3])
    if len(paper.get('authors') or []) > 3:
        authors += ' et al.'
    pub_date = paper.get('publicationDate') or str(paper.get('year') or 'N/A')
    citations = paper.get('citationCount', 0)
    score = paper.get('ai_score', 'N/A')
    colour = score_colour(score)

    card_style = S['card'] + f'border-left:4px solid {colour};'
    badge_style = S['score_badge'] + f'background:{colour};'

    card = tags.div(style=card_style)

    with card:
        with tags.div(style=S['card_header']):
            tags.h3(title, style=S['card_title'])
            tags.span(f'{score}/10', style=badge_style)

        tags.p(f'{authors} · {pub_date}', style=S['card_meta'])

        with tags.div(style=S['card_body']):
            with tags.p(style=S['card_text']):
                tags.strong('Summary: ')
                raw(paper.get('ai_summary', ''))
            with tags.p(style=S['card_text_last']):
                tags.strong('Key takeaway: ')
                raw(paper.get('ai_contribution', ''))

        tags.p(f'Relevance: {paper.get("ai_reason", "")}', style=S['card_reason'])

        with tags.div(style=S['card_footer']):
            tags.span(f'{citations} citations', style=S['badge'])
            paper_links_tag(paper)

    return card


def build_email(papers):
    today = datetime.now().strftime('%B %d, %Y')

    doc = dominate.document(title='Research Digest')

    with doc:
        with tags.body(style=S['body']):
            with tags.div(style=S['banner']):
                tags.h1('Research Digest', style=S['banner_h1'])
                tags.p(
                    raw(f'{today} &nbsp;&middot;&nbsp; {len(papers)} paper(s)'),
                    style=S['banner_sub'],
                )

            tags.p(
                f'Papers scored 1-10 for relevance to your BPD detection / mental health NLP '
                f'research and filtered to >={RELEVANCE_THRESHOLD}/10. '
                f'Summaries generated by Gemini 1.5 Flash.',
                style=S['intro'],
            )

            if papers:
                for p in papers:
                    paper_card(p)
            else:
                tags.p('No relevant papers found this period.', style='color:#888;')

            tags.hr(style=S['hr'])
            tags.p(raw('Semantic Scholar &middot; Gemini 2.0 Flash &middot; GitHub Actions'), style=S['footer'])

    return str(doc)

def send_email(html, paper_count):
    msg = MIMEMultipart('alternative')
    msg['Subject'] = f'Research Digest: {paper_count} relevant papers — {datetime.now().strftime("%b %d")}'
    msg['From'] = os.environ['GMAIL_USER']
    msg['To'] = os.environ['EMAIL_TO']
    msg.attach(MIMEText(html, 'html'))

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        server.login(os.environ['GMAIL_USER'], os.environ['GMAIL_APP_PASSWORD'])
        server.sendmail(os.environ['GMAIL_USER'], os.environ['EMAIL_TO'], msg.as_string())

    print(f'Email sent with {paper_count} papers.')


def main():
    all_papers = []
    for query in SEARCH_QUERIES:
        print(f'\nSearching: {query}')
        all_papers.extend(fetch_papers(query))

    seen, unique = set(), []
    for p in all_papers:
        pid = p.get('paperId') or p.get('title', '')
        if pid not in seen:
            seen.add(pid)
            unique.append(p)

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