import json
import re

import db
from config import RESEARCH_CONTEXT, SUMMARISER_PROMPT

MODEL_VERSION = 'gemini-2.5-flash'
MISSING_FIELD_PLACEHOLDER = 'Not available from this source.'

_FENCE_RE = re.compile(r'```(?:json)?', re.IGNORECASE)
_FIELDS = ('methodology', 'findings', 'relevance', 'limitations')
# The prompt asks for `relevance_to_research` to match the human-readable spec, but
# the DB column is `relevance`. Map at this boundary so downstream code stays aligned
# with the schema rather than the prompt.
_PROMPT_KEY_BY_COLUMN = {'relevance': 'relevance_to_research'}


def summarise_paper(paper, gemini_fn, conn=None):
	paper_id = paper.get('paperId')
	title = (paper.get('title') or '')[:60]

	if conn is not None and paper_id:
		cached = db.get_summary(conn, paper_id)
		if cached is not None:
			print(f'Cache hit, skipping summarisation: {title}')
			return cached

	abstract = paper.get('abstract')
	if not abstract:
		print(f'No abstract available, cannot summarise: {title}')
		return None

	source_material = f'Abstract:\n{abstract}'
	prompt = SUMMARISER_PROMPT.format(research_context=RESEARCH_CONTEXT, source_material=source_material)

	try:
		response = gemini_fn(prompt)
		fields = parse_summary_response(response)
	except Exception as e:
		print(f'Summariser Gemini error for "{title}": {e}')
		return None

	if conn is not None and paper_id:
		db.upsert_summary(conn, paper_id, fields, MODEL_VERSION)

	return fields


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
