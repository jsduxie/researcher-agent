from pathlib import Path

import yaml

_ROOT = Path(__file__).parent
_CONFIG_PATH = _ROOT / 'config' / 'digest.yaml'
_PROMPTS_DIR = _ROOT / 'prompts'

with open(_CONFIG_PATH) as f:
	_digest = yaml.safe_load(f)


def _load_prompt(name):
	with open(_PROMPTS_DIR / f'{name}.md') as f:
		return f.read()


SEARCH_QUERIES = _digest['search_queries']
RESEARCH_CONTEXT = _digest['research_context']
RELEVANCE_THRESHOLD = _digest['relevance_threshold']
DAYS_BACK = _digest['days_back']
MAX_PER_QUERY = _digest['max_per_query']
BATCH_SIZE = _digest['batch_size']

SCORER_PROMPT = _load_prompt('scorer')
