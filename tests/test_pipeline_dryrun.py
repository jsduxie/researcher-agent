import main


def test_dry_run_uses_fixtures_and_makes_no_network_calls(mocker, capsys):
	mocker.patch.object(main, 'DRY_RUN', False)
	mock_get = mocker.patch('fetcher.requests.get')
	mock_post = mocker.patch('gemini.requests.post')
	mock_summariser_get = mocker.patch('summariser.requests.get')
	mock_summariser_post = mocker.patch('summariser.requests.post')
	mock_smtp = mocker.patch('emailer.smtplib.SMTP_SSL')
	mock_db_connect = mocker.patch('db.psycopg.connect')

	main.main(['--dry-run'])

	mock_get.assert_not_called()
	mock_post.assert_not_called()
	# Summariser must stay off the network in dry-run too: PDF path is skipped because api_key is None; abstract path reads the gemini_summary fixture instead.
	mock_summariser_get.assert_not_called()
	mock_summariser_post.assert_not_called()
	mock_smtp.assert_not_called()
	mock_db_connect.assert_not_called()

	captured = capsys.readouterr()
	assert '<!DOCTYPE html>' in captured.out
	assert 'Research Digest' in captured.out


def test_dry_run_short_circuits_time_sleep(mocker):
	mocker.patch.object(main, 'DRY_RUN', False)
	mock_fetcher_sleep = mocker.patch('fetcher.time.sleep')
	mock_gemini_sleep = mocker.patch('gemini.time.sleep')

	main.main(['--dry-run'])

	mock_fetcher_sleep.assert_not_called()
	mock_gemini_sleep.assert_not_called()


def test_dry_run_skips_the_embedding_prefilter(mocker, capsys):
	# Dry-run stays offline: no model load, no embedding.
	mocker.patch.object(main, 'DRY_RUN', False)
	embed = mocker.patch('main.embedder.embed_texts')
	main.main(['--dry-run'])
	embed.assert_not_called()
	assert 'Pre-filter disabled' in capsys.readouterr().out
