from datetime import datetime

from config import EMAIL_TEMPLATE

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
