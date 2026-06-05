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


def _load_prompt(name):
	with open(_PROMPTS_DIR / f'{name}.md') as f:
		return f.read()


with open(_TEMPLATES_DIR / 'style.yaml') as _f:
	S = yaml.safe_load(_f)

SCORER_PROMPT = _load_prompt('scorer')
SUMMARISER_PROMPT = _load_prompt('summariser')

_env = Environment(loader=FileSystemLoader(str(_TEMPLATES_DIR)), trim_blocks=True, lstrip_blocks=True)
_env.globals['S'] = S
EMAIL_TEMPLATE = _env.get_template('email.html')


def load_seed():
	# Generic example values; they bootstrap an empty app_config table, after which Neon is the source of truth (edit via the dashboard Config page).
	with open(_SEED_PATH) as f:
		return yaml.safe_load(f)


def load(conn):
	# Bootstraps app_config from the seed file on first call against an empty table, then returns the current config every time.
	data = db.load_config(conn)
	if not data:
		seed = load_seed()
		db.update_config(conn, seed, by='seed')
		data = seed
	return Config(**data)
