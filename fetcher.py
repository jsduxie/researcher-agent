import time
from datetime import datetime, timedelta

import requests

from config import DAYS_BACK, MAX_PER_QUERY

SEMANTIC_SCHOLAR_URL = 'https://api.semanticscholar.org/graph/v1/paper/search'
PAPER_FIELDS = 'title,abstract,authors,year,citationCount,externalIds,openAccessPdf,url,publicationDate'


def _cutoff_year(today, days_back):
	return (today - timedelta(days=days_back)).year


def fetch_papers(query):
	cutoff = _cutoff_year(datetime.now(), DAYS_BACK)
	params = {'query': query, 'limit': MAX_PER_QUERY, 'fields': PAPER_FIELDS, 'publicationDateOrYear': f'{cutoff}-'}
	try:
		time.sleep(2)
		r = requests.get(SEMANTIC_SCHOLAR_URL, params=params, timeout=15)
		r.raise_for_status()
		return r.json().get('data', [])
	except Exception as e:
		print(f'Error fetching "{query}": {e}')
		return []


def dedup_papers(papers):
	seen, unique = set(), []
	for p in papers:
		pid = p.get('paperId')
		key = ('id', pid) if pid else ('title', p.get('title', ''))
		if key not in seen:
			seen.add(key)
			unique.append(p)
	return unique
