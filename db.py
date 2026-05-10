from pathlib import Path

import psycopg

_SCHEMA_PATH = Path(__file__).parent / 'db' / 'schema.sql'

_UPSERT_PAPER_SQL = """
INSERT INTO papers (paper_id, title, abstract, year, citation_count, url, doi, pdf_url, fetched_at, updated_at)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
ON CONFLICT (paper_id) DO UPDATE
SET title = EXCLUDED.title,
    abstract = EXCLUDED.abstract,
    year = EXCLUDED.year,
    citation_count = EXCLUDED.citation_count,
    url = EXCLUDED.url,
    doi = EXCLUDED.doi,
    pdf_url = EXCLUDED.pdf_url,
    updated_at = NOW()
RETURNING (xmax = 0) AS was_inserted
"""


def connect(database_url):
	# autocommit and prepare_threshold=None keep us compatible with Neon's PgBouncer pooler.
	return psycopg.connect(database_url, autocommit=True, prepare_threshold=None)


def init_schema(conn):
	with conn.cursor() as cur:
		cur.execute(_SCHEMA_PATH.read_text())


def upsert_paper(conn, paper):
	paper_id = paper['paperId']
	doi = (paper.get('externalIds') or {}).get('DOI')
	pdf_url = (paper.get('openAccessPdf') or {}).get('url')
	with conn.transaction(), conn.cursor() as cur:
		cur.execute(
			_UPSERT_PAPER_SQL,
			(
				paper_id,
				paper.get('title'),
				paper.get('abstract'),
				paper.get('year'),
				paper.get('citationCount'),
				paper.get('url'),
				doi,
				pdf_url,
			),
		)
		was_inserted = cur.fetchone()[0]
		cur.execute('DELETE FROM paper_authors WHERE paper_id = %s', (paper_id,))
		for position, author in enumerate(paper.get('authors') or []):
			name = author.get('name') if isinstance(author, dict) else None
			if not name:
				continue
			cur.execute(
				'INSERT INTO paper_authors (paper_id, position, name) VALUES (%s, %s, %s)', (paper_id, position, name)
			)
	return was_inserted


def paper_exists(conn, paper_id):
	with conn.cursor() as cur:
		cur.execute('SELECT 1 FROM papers WHERE paper_id = %s', (paper_id,))
		return cur.fetchone() is not None


def start_run(conn, papers_fetched):
	with conn.cursor() as cur:
		cur.execute('INSERT INTO runs (papers_fetched) VALUES (%s) RETURNING id', (papers_fetched,))
		return cur.fetchone()[0]


def finish_run(conn, run_id, papers_kept):
	with conn.cursor() as cur:
		cur.execute('UPDATE runs SET finished_at = NOW(), papers_kept = %s WHERE id = %s', (papers_kept, run_id))
