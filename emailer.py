import smtplib
from dataclasses import dataclass
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


@dataclass(frozen=True)
class SmtpCredentials:
	user: str
	password: str
	to: str


def send_email(html, paper_count, smtp_credentials):
	subject_date = datetime.now().strftime('%b %d')
	msg = MIMEMultipart('alternative')
	msg['Subject'] = f'Research Digest: {paper_count} relevant papers ({subject_date})'
	msg['From'] = smtp_credentials.user
	msg['To'] = smtp_credentials.to
	msg.attach(MIMEText(html, 'html'))

	with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
		server.login(smtp_credentials.user, smtp_credentials.password)
		server.sendmail(smtp_credentials.user, smtp_credentials.to, msg.as_string())

	print(f'Email sent with {paper_count} papers.')
