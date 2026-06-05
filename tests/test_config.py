import pytest

import config

# Shape and value tests run against the seed file, which is also what an empty app_config table gets bootstrapped from.
_SEED = config.load_seed()
_CFG = config.Config(**_SEED)

# -- SUMMARISER_PROMPT --


def test_summariser_prompt_loaded_non_empty():
	assert config.SUMMARISER_PROMPT.strip()


def test_summariser_prompt_exposes_required_placeholders():
	assert '{research_context}' in config.SUMMARISER_PROMPT
	assert '{source_material}' in config.SUMMARISER_PROMPT


def test_summariser_prompt_requests_all_four_fields():
	for field in ('methodology', 'findings', 'relevance_to_research', 'limitations'):
		assert field in config.SUMMARISER_PROMPT


def test_summariser_prompt_formats_without_keyerror():
	# JSON braces in the template must be escaped so str.format doesn't treat them as placeholders; if escaping breaks this raises KeyError or IndexError.
	rendered = config.SUMMARISER_PROMPT.format(research_context='ctx', source_material='abc')
	assert 'ctx' in rendered
	assert 'abc' in rendered


# -- seed file shape --


def test_seed_contains_exactly_the_digest_keys():
	# The seed must stay in lockstep with Config's fields; a drifted key fails Config(**seed) at bootstrap, this pins the contract explicitly.
	assert set(_SEED) == set(config.DIGEST_KEYS)


def test_search_queries_is_non_empty_list_of_strings():
	assert isinstance(_CFG.search_queries, list)
	assert _CFG.search_queries
	assert all(isinstance(q, str) and q.strip() for q in _CFG.search_queries)


def test_gemini_call_budget_is_positive_int():
	assert isinstance(_CFG.gemini_call_budget, int)
	assert not isinstance(_CFG.gemini_call_budget, bool)
	assert _CFG.gemini_call_budget > 0


def test_gemini_model_is_non_empty_string():
	assert isinstance(_CFG.gemini_model, str)
	assert _CFG.gemini_model.strip()


def test_gemini_base_url_is_absolute_https():
	assert _CFG.gemini_base_url.startswith('https://')


def test_gemini_upload_base_url_is_absolute_https():
	assert _CFG.gemini_upload_base_url.startswith('https://')


def test_pdf_max_size_mb_is_positive_int():
	assert isinstance(_CFG.pdf_max_size_mb, int)
	assert not isinstance(_CFG.pdf_max_size_mb, bool)
	assert _CFG.pdf_max_size_mb > 0


@pytest.mark.parametrize(
	'field', ['relevance_threshold', 'days_back', 'max_per_query', 'batch_size', 'prefilter_top_n']
)
def test_numeric_digest_values_are_positive_ints(field):
	value = getattr(_CFG, field)
	assert isinstance(value, int)
	assert not isinstance(value, bool)
	assert value > 0


# -- config.load --


def test_load_returns_config_from_populated_table_without_seeding(mocker):
	mocker.patch('config.db.load_config', return_value=dict(_SEED))
	update = mocker.patch('config.db.update_config')
	assert config.load(mocker.Mock()) == _CFG
	update.assert_not_called()


def test_load_seeds_empty_table_then_returns_seed_values(mocker):
	mocker.patch('config.db.load_config', return_value={})
	update = mocker.patch('config.db.update_config')
	conn = mocker.Mock()
	assert config.load(conn) == _CFG
	update.assert_called_once_with(conn, _SEED, by='seed')


def test_load_raises_on_unexpected_db_key(mocker):
	# A rogue key in app_config means the table and Config have drifted; fail the run loudly rather than silently dropping the value.
	mocker.patch('config.db.load_config', return_value={**_SEED, 'rogue_key': 1})
	with pytest.raises(TypeError):
		config.load(mocker.Mock())


def test_load_backfills_missing_keys_from_seed(mocker):
	# Keys added to Config after a table was seeded must self-migrate on the next load rather than crash the run.
	partial = dict(_SEED)
	del partial['prefilter_top_n']
	mocker.patch('config.db.load_config', return_value=partial)
	update = mocker.patch('config.db.update_config')
	conn = mocker.Mock()
	assert config.load(conn) == _CFG
	update.assert_called_once_with(conn, {'prefilter_top_n': _SEED['prefilter_top_n']}, by='seed')
