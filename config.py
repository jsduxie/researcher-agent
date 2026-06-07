from dataclasses import dataclass
from pathlib import Path

import yaml
from jinja2 import Environment, FileSystemLoader

import db

_ROOT = Path(__file__).parent
_SEED_PATH = _ROOT / 'config' / 'digest.example.yaml'
_PROMPTS_DIR = _ROOT / 'prompts'
_TEMPLATES_DIR = _ROOT / 'templates'

# config keys expected in db
DIGEST_KEYS = (
	'search_queries',
	'research_context',
	'relevance_threshold',
	'days_back',
	'max_per_query',
	'batch_size',
	'gemini_model',
	'gemini_base_url',
	'gemini_upload_base_url',
	'gemini_call_budget',
	'pdf_max_size_mb',
	'prefilter_top_n',
)


@dataclass(frozen=True)
class Config:
	search_queries: list[str]
	research_context: str
	relevance_threshold: int
	days_back: int
	max_per_query: int
	batch_size: int
	gemini_model: str
	gemini_base_url: str
	gemini_upload_base_url: str
	gemini_call_budget: int
	pdf_max_size_mb: int
	prefilter_top_n: int


def _load_prompt(name):
	with open(_PROMPTS_DIR / f'{name}.md') as f:
		return f.read()


with open(_TEMPLATES_DIR / 'style.yaml') as _f:
	S = yaml.safe_load(_f)

SCORER_PROMPT = _load_prompt('scorer')
SCORER_CALIBRATION_PROMPT = _load_prompt('scorer_calibration')
SCORER_CALIBRATION_EXAMPLE = _load_prompt('scorer_calibration_example')
SUMMARISER_PROMPT = _load_prompt('summariser')
SUMMARISER_FEWSHOT_PROMPT = _load_prompt('summariser_fewshot')
SUMMARISER_FEWSHOT_GOOD = _load_prompt('summariser_fewshot_good')
SUMMARISER_FEWSHOT_BAD = _load_prompt('summariser_fewshot_bad')

_env = Environment(loader=FileSystemLoader(str(_TEMPLATES_DIR)), trim_blocks=True, lstrip_blocks=True)
_env.globals['S'] = S
EMAIL_TEMPLATE = _env.get_template('email.html')


def load_seed():
	# Generic example values; they bootstrap an empty app_config table, after which Neon is the source of truth (edit via the dashboard Config page).
	with open(_SEED_PATH) as f:
		return yaml.safe_load(f)


def load(conn):
	# Bootstraps app_config from the seed on first call, and backfills keys added after the table was seeded so Config never sees a missing field.
	data = db.load_config(conn)
	missing_keys = [k for k in DIGEST_KEYS if k not in data]
	if missing_keys:
		seed = load_seed()
		missing = {k: seed[k] for k in missing_keys}
		db.update_config(conn, missing, by='seed')
		data.update(missing)
	return Config(**data)
