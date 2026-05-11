import time
from datetime import datetime, timedelta

import requests

from config import DAYS_BACK, MAX_PER_QUERY

SEMANTIC_SCHOLAR_URL = 'https://api.semanticscholar.org/graph/v1/paper/search'
PAPER_FIELDS = 'title,abstract,authors,year,citationCount,externalIds,openAccessPdf,url,publicationDate'

RETRY_STATUS_CODES = (429, 500, 502, 503, 504)
BACKOFF_DELAYS = (5, 15, 45)


class FetchError(Exception):
	pass


def _cutoff_year(today, days_back):
	return (today - timedelta(days=days_back)).year


def _retry_after_seconds(response):
	# HTTP-date form (RFC 7231) is ignored; we fall back to exponential backoff
	# in that case rather than parse dates the API never sends.
	header = response.headers.get('Retry-After')
	if header is None:
		return None
	try:
		return int(header)
	except (TypeError, ValueError):
		return None


def fetch_papers(query, api_key):
	cutoff = _cutoff_year(datetime.now(), DAYS_BACK)
	params = {'query': query, 'limit': MAX_PER_QUERY, 'fields': PAPER_FIELDS, 'publicationDateOrYear': f'{cutoff}-'}
	headers = {'x-api-key': api_key}
	time.sleep(2)
	for attempt in range(len(BACKOFF_DELAYS) + 1):
		try:
			r = requests.get(SEMANTIC_SCHOLAR_URL, params=params, headers=headers, timeout=15)
		except requests.RequestException as e:
			if attempt == len(BACKOFF_DELAYS):
				raise FetchError(f'{query}: gave up after {attempt + 1} attempts: {e}') from e
			time.sleep(BACKOFF_DELAYS[attempt])
			continue
		if r.status_code in RETRY_STATUS_CODES:
			if attempt == len(BACKOFF_DELAYS):
				raise FetchError(f'{query}: gave up after {attempt + 1} attempts: HTTP {r.status_code}')
			delay = _retry_after_seconds(r) or BACKOFF_DELAYS[attempt]
			time.sleep(delay)
			continue
		if not r.ok:
			raise FetchError(f'{query}: HTTP {r.status_code}, not retried')
		return r.json().get('data', [])
	raise FetchError(f'{query}: retry loop fell through unexpectedly')


def dedup_papers(papers):
	seen, unique = set(), []
	for p in papers:
		pid = p.get('paperId')
		key = ('id', pid) if pid else ('title', p.get('title', ''))
		if key not in seen:
			seen.add(key)
			unique.append(p)
	return unique
