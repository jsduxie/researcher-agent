import os
import sys

from dotenv import load_dotenv

import db
import embedder

# Run from the repo root: `python -m scripts.backfill_embeddings`. One-off: embeds papers stored before the vector column existed.
load_dotenv()


def backfill():
	url = os.environ.get('DATABASE_URL')
	if not url:
		sys.exit('DATABASE_URL is not set; export it (or add to .env) before backfilling.')

	conn = db.connect(url)
	papers = db.list_papers_missing_embeddings(conn)
	if not papers:
		print('No papers missing embeddings; nothing to backfill.')
		return

	print(f'Backfilling embeddings for {len(papers)} papers...')
	# Same text shape the live pipeline uses in main._prefilter_scoring_queue: title, blank line, abstract, with empty-string fallbacks.
	texts = [f'{p["title"] or ""}\n\n{p["abstract"] or ""}' for p in papers]
	vectors = embedder.embed_texts(texts)
	for paper, vector in zip(papers, vectors, strict=True):
		db.set_paper_embedding(conn, paper['paper_id'], vector)

	print(f'Backfilled embeddings for {len(papers)} papers.')


if __name__ == '__main__':
	backfill()
