import argparse
import json
import os
import sys
from pathlib import Path

import config
import db
import emailer
import embedder
import scorer
import summariser
from fetcher import FetchError, dedup_papers, fetch_papers
from gemini import GeminiBudgetExhausted, GeminiQuotaExhausted
from render import build_email

_FIXTURES = Path(__file__).parent / 'tests' / 'fixtures'
DRY_RUN = False
GEMINI_CALL_COUNT = 0
# Populated from cfg.gemini_call_budget at the top of main() so tests can still patch this module attribute directly without touching cfg.
GEMINI_CALL_BUDGET = 0
_CFG = None
# Papers swept from the backlog stop being retried after this many real Gemini attempts.
SWEEP_MAX_ATTEMPTS = 3

# Contract enforced by tests/test_workflow_env.py against the send-digest step.
REQUIRED_ENV_VARS = (
	'SEMANTIC_SCHOLAR_API_KEY',
	'DATABASE_URL',
	'GEMINI_API_KEY',
	'GMAIL_USER',
	'GMAIL_APP_PASSWORD',
	'EMAIL_TO',
)


def _db_host_for_log(url):
	# Strip credentials and trailing path/query; the host portion is the actionable signal when diagnosing env-divergence in cron logs.
	if '@' not in url:
		return '<unknown>'
	return url.split('@', 1)[1].split('/', 1)[0]


def _fetch(query, api_key):
	if DRY_RUN:
		return json.loads((_FIXTURES / 'papers.json').read_text())
	return fetch_papers(query, api_key, _CFG)


def _collect_papers(api_key):
	all_papers = []
	attempted = 0
	errored = 0
	for query in _CFG.search_queries:
		print(f'\nSearching: {query}')
		attempted += 1
		try:
			all_papers.extend(_fetch(query, api_key))
		except FetchError as e:
			print(f'Error fetching "{query}": {e}')
			errored += 1
	return all_papers, attempted, errored


def _gemini_score(prompt, retries=3):
	if DRY_RUN:
		_record_gemini_call()
		return (_FIXTURES / 'gemini_score.json').read_text()
	return scorer.gemini(prompt, os.environ['GEMINI_API_KEY'], _CFG, retries, on_attempt=_record_gemini_call)


def _gemini_summarise(prompt, retries=3):
	if DRY_RUN:
		_record_gemini_call()
		return (_FIXTURES / 'gemini_summary.json').read_text()
	return scorer.gemini(prompt, os.environ['GEMINI_API_KEY'], _CFG, retries, on_attempt=_record_gemini_call)


def _record_gemini_call():
	# Raise per-attempt before incrementing so a 429 retry storm trips the budget on its next attempt, not three or four attempts past it.
	global GEMINI_CALL_COUNT
	if GEMINI_CALL_COUNT >= GEMINI_CALL_BUDGET:
		raise GeminiBudgetExhausted(f'budget {GEMINI_CALL_BUDGET} reached after {GEMINI_CALL_COUNT} calls')
	GEMINI_CALL_COUNT += 1


def _summarise_kept_papers(enriched, database_url):
	# summariser.summarise_paper manages two short DB sessions (cache check, then persist); no DB open during the Gemini work in between.
	api_key = None if DRY_RUN else os.environ.get('GEMINI_API_KEY')
	ctx = summariser.SummariserContext(
		cfg=_CFG, database_url=database_url, api_key=api_key, on_gemini_call=_record_gemini_call
	)
	skipped = 0
	for i, paper in enumerate(enriched):
		try:
			fields = summariser.summarise_paper(paper, _gemini_summarise, ctx)
		except GeminiBudgetExhausted as e:
			# _record_gemini_call raises this when the next attempt would exceed the cap; remaining papers are emailed without summaries.
			skipped = len(enriched) - i
			print(f'Budget exhausted, halting summarisation: {e}')
			break
		except GeminiQuotaExhausted as e:
			print(f'Quota exhausted, halting summarisation: {e}')
			break
		if fields:
			paper.update(fields)
	if skipped:
		print(
			f'Gemini budget ({GEMINI_CALL_BUDGET}) reached; {skipped} paper(s) will be emailed without full summaries'
		)


def _report_gemini_usage():
	print(f'Total Gemini calls this run: {GEMINI_CALL_COUNT}/{GEMINI_CALL_BUDGET}')
	if GEMINI_CALL_COUNT >= GEMINI_CALL_BUDGET:
		print(f'Gemini budget exhausted (cap {GEMINI_CALL_BUDGET})')


def _send(html, paper_count):
	if DRY_RUN:
		print(html)
		return
	creds = emailer.SmtpCredentials(os.environ['GMAIL_USER'], os.environ['GMAIL_APP_PASSWORD'], os.environ['EMAIL_TO'])
	emailer.send_email(html, paper_count, creds)


def _persist_fetched_papers(database_url, unique):
	# Run row, paper upserts, runs_papers roster (with was_new flag), needs_scoring filter in one scoped session; schema was initialised in the config-load session. The summary log proves writes landed; if it's missing from the cron output, env-divergence is the first thing to check.
	paper_ids = [p['paperId'] for p in unique]
	with db.session(database_url) as conn:
		run_id = db.start_run(conn, len(unique))
		roster = [(p['paperId'], db.upsert_paper(conn, p)) for p in unique]
		db.record_run_papers(conn, run_id, roster)
		unscored = db.needs_scoring(conn, paper_ids)
		# Two batches a run drains the unscored backlog steadily without dominating the call budget.
		backlog = db.list_unscored(
			conn, exclude_ids=paper_ids, max_attempts=SWEEP_MAX_ATTEMPTS, limit=2 * _CFG.batch_size
		)
		new_papers = [p for p in unique if p['paperId'] in unscored]
	print(
		f'Persisted fetched papers: run_id={run_id}, upserted={len(unique)}, '
		f'needs_scoring={len(new_papers)}, backlog={len(backlog)}'
	)
	return run_id, new_papers + backlog


def _record_scoring_attempts(database_url, attempted, responded):
	with db.session(database_url) as conn:
		db.mark_scoring_results(conn, attempted=list(attempted), responded=responded)
	print(f'Recorded scoring attempts: attempted={len(attempted)}, responded={len(responded)}')


def _finalise_run(database_url, run_id, papers_kept, queries_attempted, queries_errored):
	with db.session(database_url) as conn:
		db.finish_run(conn, run_id, papers_kept, queries_attempted, queries_errored)
	print(f'Finalised run: run_id={run_id}, papers_kept={papers_kept}')


def _prefilter_scoring_queue(papers, database_url):
	# Local embedding rank against research_context so only the most plausible top N spend Gemini quota; vectors persist for later similarity retrieval.
	if not papers:
		return papers
	[context_vector] = embedder.embed_texts([_CFG.research_context])
	texts = [f'{p.get("title") or ""}\n\n{p.get("abstract") or ""}' for p in papers]
	vectors = embedder.embed_texts(texts)
	if database_url is not None:
		with db.session(database_url) as conn:
			for paper, vector in zip(papers, vectors, strict=True):
				db.set_paper_embedding(conn, paper['paperId'], vector)
	order = embedder.rank_by_similarity(context_vector, vectors)
	kept = []
	for position, idx in enumerate(order[: _CFG.prefilter_top_n], start=1):
		# Rank rides on the paper so the post-scoring log can report how well the pre-filter predicted the LLM's keeps.
		papers[idx]['_prefilter_rank'] = position
		kept.append(papers[idx])
	print(
		f'Pre-filter: {len(papers)} candidates considered, {len(kept)} passed to the scorer (top_n={_CFG.prefilter_top_n})'
	)
	return kept


def _bootstrap(argv):
	# Parses args, validates env, and loads cfg before any work runs so a corrupt config table fails fast.
	parser = argparse.ArgumentParser()
	parser.add_argument('--dry-run', action='store_true', help='use fixtures and print HTML, no network or email')
	parser.add_argument(
		'--no-prefilter', action='store_true', help='ablation: score the full queue without the embedding pre-filter'
	)
	global DRY_RUN, GEMINI_CALL_COUNT, GEMINI_CALL_BUDGET, _CFG
	args = parser.parse_args(argv)
	DRY_RUN = args.dry_run
	GEMINI_CALL_COUNT = 0

	api_key = None
	database_url = None
	if not DRY_RUN:
		missing = [v for v in REQUIRED_ENV_VARS if not os.environ.get(v)]
		if missing:
			sys.exit(f'{", ".join(missing)} required; set in the environment or GitHub Actions secrets.')
		api_key = os.environ['SEMANTIC_SCHOLAR_API_KEY']
		database_url = os.environ['DATABASE_URL']
		print(f'DB target: {_db_host_for_log(database_url)}')
		with db.session(database_url) as conn:
			db.init_schema(conn)
			_CFG = config.load(conn)
	else:
		_CFG = config.Config(**config.load_seed())
	GEMINI_CALL_BUDGET = _CFG.gemini_call_budget
	return args, api_key, database_url


def _dedupe_fetched(all_papers):
	unique = dedup_papers(all_papers)
	# Papers without paperId can't be persisted (paper_id is the PK) or deduplicated across runs; drop up front rather than crashing in upsert.
	missing_id = [p for p in unique if not p.get('paperId')]
	if missing_id:
		print(f'Dropped {len(missing_id)} paper(s) without paperId (cannot persist)')
	return [p for p in unique if p.get('paperId')]


def main(argv=None):
	args, api_key, database_url = _bootstrap(argv)
	run_id = None

	all_papers, queries_attempted, queries_errored = _collect_papers(api_key)
	unique = _dedupe_fetched(all_papers)

	if database_url is not None:
		run_id, new_papers = _persist_fetched_papers(database_url, unique)
	else:
		new_papers = unique

	print(f'{len(unique)} unique papers found, {len(new_papers)} need scoring. Scoring\n')

	# Dry-run stays offline (no model load); --no-prefilter is the live ablation path for A/B comparison.
	if DRY_RUN or args.no_prefilter:
		print('Pre-filter disabled; scoring the full queue')
	else:
		new_papers = _prefilter_scoring_queue(new_papers, database_url)

	# Scoring runs without a DB connection held so Neon's pooler can auto-suspend during the Gemini calls.
	enriched, responded, attempted = scorer.score_and_summarise(new_papers, _gemini_score, _CFG)
	enriched.sort(key=lambda p: (p.get('ai_score') or 0, p.get('citationCount') or 0), reverse=True)
	print(f'{len(enriched)} papers passed the relevance filter (>={_CFG.relevance_threshold}/10).')
	ranks = [p['_prefilter_rank'] for p in enriched if '_prefilter_rank' in p]
	if ranks:
		print(f'Average pre-filter rank of threshold-passing papers: {sum(ranks) / len(ranks):.1f}')

	if database_url is not None and new_papers:
		_record_scoring_attempts(database_url, attempted, responded)

	# Per-paper summarisation. Each call manages its own short DB sessions internally.
	_summarise_kept_papers(enriched, database_url)
	_report_gemini_usage()

	if not enriched:
		print('No relevant papers, skipping email.')
	else:
		_send(build_email(enriched, _CFG), len(enriched))

	# Finalising the run last so a logging failure never silently skips the email.
	if database_url is not None:
		_finalise_run(database_url, run_id, len(enriched), queries_attempted, queries_errored)

	if queries_attempted > 0 and queries_attempted == queries_errored:
		sys.exit(f'All {queries_attempted} queries errored; no papers retrieved this run.')


if __name__ == '__main__':
	main()
