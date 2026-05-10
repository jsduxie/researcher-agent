from pathlib import Path

import yaml

_CONFIG_PATH = Path(__file__).parent / 'config' / 'digest.yaml'

with open(_CONFIG_PATH) as f:
	_digest = yaml.safe_load(f)

SEARCH_QUERIES = _digest['search_queries']
RESEARCH_CONTEXT = _digest['research_context']
RELEVANCE_THRESHOLD = _digest['relevance_threshold']
DAYS_BACK = _digest['days_back']
MAX_PER_QUERY = _digest['max_per_query']
BATCH_SIZE = _digest['batch_size']
