from pathlib import Path

import yaml
from jinja2 import Environment, FileSystemLoader

_ROOT = Path(__file__).parent
_CONFIG_PATH = _ROOT / 'config' / 'digest.yaml'
_PROMPTS_DIR = _ROOT / 'prompts'
_TEMPLATES_DIR = _ROOT / 'templates'

with open(_CONFIG_PATH) as f:
	_digest = yaml.safe_load(f)

with open(_TEMPLATES_DIR / 'style.yaml') as f:
	S = yaml.safe_load(f)


def _load_prompt(name):
	with open(_PROMPTS_DIR / f'{name}.md') as f:
		return f.read()


SEARCH_QUERIES = _digest['search_queries']
RESEARCH_CONTEXT = _digest['research_context']
RELEVANCE_THRESHOLD = _digest['relevance_threshold']
DAYS_BACK = _digest['days_back']
MAX_PER_QUERY = _digest['max_per_query']
BATCH_SIZE = _digest['batch_size']
GEMINI_MODEL = _digest['gemini_model']
GEMINI_BASE_URL = _digest['gemini_base_url']
GEMINI_UPLOAD_BASE_URL = _digest['gemini_upload_base_url']
GEMINI_CALL_BUDGET = _digest['gemini_call_budget']
PDF_MAX_SIZE_MB = _digest['pdf_max_size_mb']

SCORER_PROMPT = _load_prompt('scorer')
SUMMARISER_PROMPT = _load_prompt('summariser')

_env = Environment(loader=FileSystemLoader(str(_TEMPLATES_DIR)), trim_blocks=True, lstrip_blocks=True)
_env.globals['S'] = S
_env.globals['relevance_threshold'] = RELEVANCE_THRESHOLD
EMAIL_TEMPLATE = _env.get_template('email.html')
