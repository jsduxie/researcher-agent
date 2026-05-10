from dataclasses import FrozenInstanceError
from datetime import datetime

import pytest

from emailer import SmtpCredentials, send_email


@pytest.fixture
def creds():
	return SmtpCredentials(user='from@example.com', password='secret', to='to@example.com')


@pytest.fixture
def mock_smtp(mocker):
	return mocker.patch('emailer.smtplib.SMTP_SSL')


def _server(mock_smtp):
	return mock_smtp.return_value.__enter__.return_value


def test_send_email_invokes_smtp_ssl_with_gmail_host_and_port(mock_smtp, creds):
	send_email('<html/>', 1, creds)
	mock_smtp.assert_called_once_with('smtp.gmail.com', 465)


def test_send_email_logs_in_with_user_and_password(mock_smtp, creds):
	send_email('<html/>', 1, creds)
	_server(mock_smtp).login.assert_called_once_with('from@example.com', 'secret')


def test_send_email_sends_from_user_to_recipient(mock_smtp, creds):
	send_email('<html/>', 1, creds)
	args = _server(mock_smtp).sendmail.call_args.args
	assert args[0] == 'from@example.com'
	assert args[1] == 'to@example.com'


def test_send_email_subject_uses_paren_format_with_count_and_date(mock_smtp, creds, mocker):
	mock_dt = mocker.patch('emailer.datetime')
	mock_dt.now.return_value = datetime(2026, 5, 10)

	send_email('<html/>', 3, creds)

	msg_str = _server(mock_smtp).sendmail.call_args.args[2]
	assert 'Subject: Research Digest: 3 relevant papers (May 10)' in msg_str


def test_send_email_message_carries_from_and_to_headers(mock_smtp, creds):
	send_email('<html/>', 1, creds)
	msg_str = _server(mock_smtp).sendmail.call_args.args[2]
	assert 'From: from@example.com' in msg_str
	assert 'To: to@example.com' in msg_str


def test_send_email_message_is_html_content_type(mock_smtp, creds):
	send_email('<html/>', 1, creds)
	msg_str = _server(mock_smtp).sendmail.call_args.args[2]
	assert 'Content-Type: text/html' in msg_str


def test_send_email_prints_success_line(mock_smtp, creds, capsys):
	send_email('<html/>', 7, creds)
	assert 'Email sent with 7 papers.' in capsys.readouterr().out


def test_smtp_credentials_is_frozen():
	creds = SmtpCredentials(user='u', password='p', to='to@x')
	with pytest.raises(FrozenInstanceError):
		creds.user = 'other'  # type: ignore[misc]
