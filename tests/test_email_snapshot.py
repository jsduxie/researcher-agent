import json
from datetime import datetime
from pathlib import Path

from research_agent import build_email

FIXTURES = Path(__file__).parent / 'fixtures'


def _freeze_datetime(mocker, when):
	mock_dt = mocker.MagicMock()
	mock_dt.now.return_value = when
	mocker.patch('render.datetime', mock_dt)


def test_email_matches_snapshot(mocker):
	_freeze_datetime(mocker, datetime(2026, 1, 15))

	with open(FIXTURES / 'enriched_papers.json') as f:
		papers = json.load(f)

	expected = (FIXTURES / 'email_snapshot.html').read_text()
	assert build_email(papers) == expected


def test_email_renders_empty_state_with_no_papers(mocker):
	_freeze_datetime(mocker, datetime(2026, 1, 15))

	html = build_email([])
	assert 'No relevant papers found this period.' in html
	assert '>0<' in html
