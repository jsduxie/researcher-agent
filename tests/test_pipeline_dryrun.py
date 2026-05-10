import research_agent


def test_dry_run_uses_fixtures_and_makes_no_network_calls(mocker, capsys):
	mocker.patch.object(research_agent, 'DRY_RUN', False)
	mock_get = mocker.patch('fetcher.requests.get')
	mock_post = mocker.patch('scorer.requests.post')
	mock_smtp = mocker.patch('research_agent.smtplib.SMTP_SSL')

	research_agent.main(['--dry-run'])

	mock_get.assert_not_called()
	mock_post.assert_not_called()
	mock_smtp.assert_not_called()

	captured = capsys.readouterr()
	assert '<!DOCTYPE html>' in captured.out
	assert 'Research Digest' in captured.out


def test_dry_run_short_circuits_time_sleep(mocker):
	mocker.patch.object(research_agent, 'DRY_RUN', False)
	mock_fetcher_sleep = mocker.patch('fetcher.time.sleep')
	mock_scorer_sleep = mocker.patch('scorer.time.sleep')

	research_agent.main(['--dry-run'])

	mock_fetcher_sleep.assert_not_called()
	mock_scorer_sleep.assert_not_called()
