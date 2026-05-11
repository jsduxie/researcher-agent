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
	# JSON braces in the template must be escaped so str.format doesn't treat them as placeholders; if escaping breaks this raises KeyError or IndexError.
	rendered = config.SUMMARISER_PROMPT.format(research_context='ctx', source_material='abc')
	assert 'ctx' in rendered
	assert 'abc' in rendered


# -- GEMINI_CALL_BUDGET --


def test_gemini_call_budget_is_positive_int():
	assert isinstance(config.GEMINI_CALL_BUDGET, int)
	assert not isinstance(config.GEMINI_CALL_BUDGET, bool)
	assert config.GEMINI_CALL_BUDGET > 0


# -- GEMINI_MODEL --


def test_gemini_model_is_non_empty_string():
	assert isinstance(config.GEMINI_MODEL, str)
	assert config.GEMINI_MODEL.strip()


# -- GEMINI_BASE_URL --


def test_gemini_base_url_is_non_empty_string():
	assert isinstance(config.GEMINI_BASE_URL, str)
	assert config.GEMINI_BASE_URL.strip()


def test_gemini_base_url_is_absolute_https():
	assert config.GEMINI_BASE_URL.startswith('https://')


# -- GEMINI_UPLOAD_BASE_URL --


def test_gemini_upload_base_url_is_non_empty_string():
	assert isinstance(config.GEMINI_UPLOAD_BASE_URL, str)
	assert config.GEMINI_UPLOAD_BASE_URL.strip()


def test_gemini_upload_base_url_is_absolute_https():
	assert config.GEMINI_UPLOAD_BASE_URL.startswith('https://')


# -- PDF_MAX_SIZE_MB --


def test_pdf_max_size_mb_is_positive_int():
	assert isinstance(config.PDF_MAX_SIZE_MB, int)
	assert not isinstance(config.PDF_MAX_SIZE_MB, bool)
	assert config.PDF_MAX_SIZE_MB > 0
