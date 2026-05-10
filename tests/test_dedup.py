import json
from pathlib import Path

from research_agent import dedup_papers

FIXTURES = Path(__file__).parent / 'fixtures'


def test_dedup_collapses_duplicate_paper_ids():
	with open(FIXTURES / 'papers.json') as f:
		papers = json.load(f)
	unique = dedup_papers(papers)
	assert len(unique) == 2
	assert {p['paperId'] for p in unique} == {'abc123', 'def456'}


def test_dedup_keeps_first_occurrence():
	papers = [{'paperId': 'a', 'title': 'first'}, {'paperId': 'a', 'title': 'second'}]
	unique = dedup_papers(papers)
	assert unique == [{'paperId': 'a', 'title': 'first'}]


def test_dedup_falls_back_to_title_when_paper_id_missing():
	papers = [{'title': 'shared title'}, {'title': 'shared title'}, {'title': 'unique title'}]
	unique = dedup_papers(papers)
	assert len(unique) == 2
