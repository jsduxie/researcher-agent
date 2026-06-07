import json
import re
from collections.abc import Callable
from dataclasses import dataclass

import requests

import db
from config import (
	SUMMARISER_FEWSHOT_BAD,
	SUMMARISER_FEWSHOT_GOOD,
	SUMMARISER_FEWSHOT_GOOD_NOTE,
	SUMMARISER_FEWSHOT_PROMPT,
	SUMMARISER_PROMPT,
	Config,
)
from gemini import GeminiBudgetExhausted, GeminiQuotaExhausted, generate_with_file, upload_pdf_to_gemini

MISSING_FIELD_PLACEHOLDER = 'Not available from this source.'

_FENCE_RE = re.compile(r'```(?:json)?', re.IGNORECASE)
_FIELDS = ('methodology', 'findings', 'relevance', 'limitations')
# Some publishers 403 a bare requests download; a browser-like header pair gets the real PDF for several of them.
_PDF_REQUEST_HEADERS = {
	'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36',
	'Accept': 'application/pdf,*/*',
}
# Prompt uses `relevance_to_research` for clarity; DB column is `relevance`. Map at this boundary so downstream code aligns with the schema.
_PROMPT_KEY_BY_COLUMN = {'relevance': 'relevance_to_research'}


@dataclass
class SummariserContext:
	# Bundles cfg + per-run optional plumbing so summarise_paper stays under ruff's max-args=5 cap once cfg is added.
	cfg: Config
	database_url: str | None = None
	api_key: str | None = None
	on_gemini_call: Callable | None = None
	# Fires with each rendered summariser prompt so the caller can log a representative one to the run row.
	on_prompt: Callable | None = None
	# False disables few-shot regardless of the feedback gate; the --no-fewshot ablation path sets this.
	fewshot_enabled: bool = True


def summarise_paper(paper, gemini_fn, ctx):
	# Two short DB sessions (cache check, then persist); Gemini work in between runs with no connection open so Neon can auto-suspend.
	paper_id = paper.get('paperId')
	title = (paper.get('title') or '')[:60]

	if ctx.database_url and paper_id:
		with db.session(ctx.database_url) as conn:
			cached = db.get_summary(conn, paper_id)
		if cached is not None:
			print(f'Cache hit, skipping summarisation: {title}')
			return cached

	# Built in its own short session so no connection is held during the Gemini calls below; empty unless the feedback gate is met and the paper has an embedding.
	fewshot_block = _build_fewshot_block(paper_id, ctx)

	fields = None
	pdf_url = (paper.get('openAccessPdf') or {}).get('url')
	if pdf_url and ctx.api_key:
		fields = _summarise_via_pdf(title, pdf_url, ctx, fewshot_block)

	if fields is None:
		fields = _summarise_via_abstract(paper, gemini_fn, ctx, fewshot_block)

	if fields is None:
		return None

	if ctx.database_url and paper_id:
		with db.session(ctx.database_url) as conn:
			db.upsert_summary(conn, paper_id, fields, ctx.cfg.gemini_model)

	return fields


def _build_fewshot_block(paper_id, ctx):
	# Returns '' below the gate, with no DB, or when the paper has no embedding, keeping the prompt byte-identical.
	if not ctx.fewshot_enabled or not ctx.database_url or not paper_id:
		return ''
	with db.session(ctx.database_url) as conn:
		if db.count_summary_feedback(conn) < ctx.cfg.fewshot_min_feedback:
			return ''
		embedding = db.get_paper_embedding(conn, paper_id)
		if embedding is None:
			return ''
		neighbours = db.similar_feedback_papers(conn, embedding, ctx.cfg.fewshot_neighbours, exclude_id=paper_id)
	block = build_fewshot_block(neighbours, ctx.cfg)
	if block:
		print('Few-shot summariser: injected field-feedback examples into the prompt')
	return block


def build_fewshot_block(feedback_papers, cfg):
	# Fields at or above the good cutoff become examples to emulate; any field below it carrying a correction becomes a pattern to avoid.
	good_lines = []
	bad_lines = []
	for paper in feedback_papers:
		title = paper.get('title') or ''
		for field, entry in (paper.get('feedback') or {}).items():
			rating = entry.get('rating')
			if rating is None:
				continue
			correction = entry.get('correction')
			if rating >= cfg.fewshot_good_rating:
				# A note on a good field explains why it works (or the small refinement for a 4); carry it so Gemini learns from positive examples too.
				template = SUMMARISER_FEWSHOT_GOOD_NOTE if correction else SUMMARISER_FEWSHOT_GOOD
				good_lines.append(template.format(field=field, title=title, correction=correction).strip())
			elif correction:
				bad_lines.append(SUMMARISER_FEWSHOT_BAD.format(field=field, title=title, correction=correction).strip())
	if not good_lines and not bad_lines:
		return ''
	return SUMMARISER_FEWSHOT_PROMPT.format(good='\n'.join(good_lines), bad='\n'.join(bad_lines))


def _summarise_via_pdf(title, pdf_url, ctx, fewshot_block=''):
	# on_gemini_call fires per attempt of each Gemini API call (Files API upload-init and generateContent), not per success. Transport errors count too.
	try:
		max_bytes = ctx.cfg.pdf_max_size_mb * 1024 * 1024
		pdf_bytes = download_pdf(pdf_url, max_bytes)
		display_name = f'{(title or "paper")[:50]}.pdf'
		file_uri = upload_pdf_to_gemini(pdf_bytes, display_name, ctx.api_key, ctx.cfg, on_attempt=ctx.on_gemini_call)
		prompt = fewshot_block + SUMMARISER_PROMPT.format(
			research_context=ctx.cfg.research_context, source_material='See the attached PDF.'
		)
		if ctx.on_prompt:
			ctx.on_prompt(prompt)
		response = generate_with_file(prompt, file_uri, ctx.api_key, ctx.cfg, on_attempt=ctx.on_gemini_call)
		return parse_summary_response(response)
	except (GeminiQuotaExhausted, GeminiBudgetExhausted):
		raise
	except Exception as e:
		print(f'PDF summariser failed for "{title}": {e}; falling back to abstract')
		return None


def _summarise_via_abstract(paper, gemini_fn, ctx, fewshot_block=''):
	# Abstract-path counter is fired inside gemini_fn (main wires scorer.gemini's on_attempt directly); nothing to fire here.
	title = (paper.get('title') or '')[:60]
	abstract = paper.get('abstract')
	if not abstract:
		print(f'No abstract available, cannot summarise: {title}')
		return None
	source_material = f'Abstract:\n{abstract}'
	prompt = fewshot_block + SUMMARISER_PROMPT.format(
		research_context=ctx.cfg.research_context, source_material=source_material
	)
	if ctx.on_prompt:
		ctx.on_prompt(prompt)
	try:
		response = gemini_fn(prompt)
		return parse_summary_response(response)
	except (GeminiQuotaExhausted, GeminiBudgetExhausted):
		raise
	except Exception as e:
		print(f'Summariser Gemini error for "{title}": {e}')
		return None


def download_pdf(url, max_size_bytes):
	with requests.get(url, stream=True, timeout=60, headers=_PDF_REQUEST_HEADERS) as r:
		r.raise_for_status()
		declared = r.headers.get('Content-Length')
		if declared and declared.isdigit() and int(declared) > max_size_bytes:
			raise ValueError(f'PDF declared size {declared} exceeds cap {max_size_bytes}')
		buffer = bytearray()
		for chunk in r.iter_content(chunk_size=64 * 1024):
			buffer.extend(chunk)
			if len(buffer) > max_size_bytes:
				raise ValueError(f'PDF stream exceeded cap {max_size_bytes}')
	body = bytes(buffer)
	# Many 'pdf' URLs are really DOI redirects to an HTML landing page; uploading that to Gemini 400s, so reject anything without the PDF magic header.
	if not body.startswith(b'%PDF'):
		raise ValueError(f'downloaded content is not a PDF (starts with {body[:8]!r})')
	return body


def parse_summary_response(response_text):
	cleaned = _FENCE_RE.sub('', response_text).strip()
	parsed = json.loads(cleaned)
	if not isinstance(parsed, dict):
		raise ValueError(f'Expected JSON object, got {type(parsed).__name__}')

	fields = {}
	for column in _FIELDS:
		prompt_key = _PROMPT_KEY_BY_COLUMN.get(column, column)
		value = parsed.get(prompt_key)
		if not isinstance(value, str) or not value.strip():
			fields[column] = MISSING_FIELD_PLACEHOLDER
		else:
			fields[column] = value.strip()
	return fields
