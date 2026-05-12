import os
import time

import pytest

import db

_DATABASE_URL_TEST = os.environ.get('DATABASE_URL_TEST')
_RUN_LONG = os.environ.get('RUN_LONG_INTEGRATION') == '1'

pytestmark = [
	pytest.mark.integration,
	pytest.mark.skipif(not _DATABASE_URL_TEST, reason='DATABASE_URL_TEST not set'),
	pytest.mark.skipif(not _RUN_LONG, reason='RUN_LONG_INTEGRATION != 1; this test sleeps ~6 minutes'),
]


def test_fresh_session_works_after_long_idle_gap():
	# Open a session and clear any stale row from a prior run.
	with db.session(_DATABASE_URL_TEST) as conn:
		db.init_schema(conn)
		with conn.cursor() as cur:
			cur.execute('DELETE FROM papers WHERE paper_id = %s', ('long-idle-test',))

	# Sleep past Neon's PgBouncer reap window (~5 min).
	time.sleep(360)

	# A fresh session must succeed regardless of what was reaped during the gap.
	with db.session(_DATABASE_URL_TEST) as conn:
		db.upsert_paper(conn, {'paperId': 'long-idle-test', 'title': 'roundtrip'})
		with conn.cursor() as cur:
			cur.execute('SELECT title FROM papers WHERE paper_id = %s', ('long-idle-test',))
			assert cur.fetchone() == ('roundtrip',)
