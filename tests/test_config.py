import config

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
	# JSON braces in the template must be escaped so str.format does not treat them
	# as placeholders. If escaping breaks, this will raise KeyError or IndexError.
	rendered = config.SUMMARISER_PROMPT.format(research_context='ctx', source_material='abc')
	assert 'ctx' in rendered
	assert 'abc' in rendered


# -- GEMINI_CALL_WARN_THRESHOLD --


def test_gemini_call_warn_threshold_is_positive_int():
	assert isinstance(config.GEMINI_CALL_WARN_THRESHOLD, int)
	assert not isinstance(config.GEMINI_CALL_WARN_THRESHOLD, bool)
	assert config.GEMINI_CALL_WARN_THRESHOLD > 0
