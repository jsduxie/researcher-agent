import json

import pytest

from research_agent import apply_scores, parse_gemini_scores


def test_parse_gemini_scores_returns_list_for_valid_json():
	text = '[{"index": 0, "relevance_score": 8}]'
	assert parse_gemini_scores(text) == [{'index': 0, 'relevance_score': 8}]


def test_parse_gemini_scores_strips_markdown_fences():
	text = '```json\n[{"index": 0, "relevance_score": 8}]\n```'
	assert parse_gemini_scores(text) == [{'index': 0, 'relevance_score': 8}]


def test_parse_gemini_scores_strips_bare_code_fences():
	text = '```\n[{"index": 0, "relevance_score": 8}]\n```'
	assert parse_gemini_scores(text) == [{'index': 0, 'relevance_score': 8}]


def test_parse_gemini_scores_raises_on_malformed_json():
	with pytest.raises(json.JSONDecodeError):
		parse_gemini_scores('not valid json')


def test_apply_scores_drops_papers_below_threshold():
	papers = [{'title': 'low'}, {'title': 'high'}]
	scores = [
		{'index': 0, 'relevance_score': 5, 'relevance_reason': 'r', 'summary': 's', 'key_contribution': 'k'},
		{'index': 1, 'relevance_score': 8, 'relevance_reason': 'r', 'summary': 's', 'key_contribution': 'k'},
	]
	result = apply_scores(papers, scores, threshold=6)
	assert len(result) == 1
	assert result[0]['title'] == 'high'
	assert result[0]['ai_score'] == 8


def test_apply_scores_skips_papers_with_no_matching_index():
	papers = [{'title': 'p1'}, {'title': 'p2'}]
	scores = [{'index': 0, 'relevance_score': 8, 'relevance_reason': 'r', 'summary': 's', 'key_contribution': 'k'}]
	result = apply_scores(papers, scores, threshold=6)
	assert len(result) == 1
	assert result[0]['title'] == 'p1'


def test_apply_scores_attaches_all_ai_fields():
	papers = [{'title': 'p1'}]
	scores = [
		{
			'index': 0,
			'relevance_score': 9,
			'relevance_reason': 'because',
			'summary': 'a summary',
			'key_contribution': 'main takeaway',
		}
	]
	result = apply_scores(papers, scores, threshold=6)
	assert result[0]['ai_score'] == 9
	assert result[0]['ai_reason'] == 'because'
	assert result[0]['ai_summary'] == 'a summary'
	assert result[0]['ai_contribution'] == 'main takeaway'
