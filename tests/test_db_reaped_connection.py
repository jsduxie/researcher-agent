import os

import pytest

import db

_DATABASE_URL_TEST = os.environ.get('DATABASE_URL_TEST')

pytestmark = [pytest.mark.integration, pytest.mark.skipif(not _DATABASE_URL_TEST, reason='DATABASE_URL_TEST not set')]

# Terminating a specific backend is only deterministic off the pooler; transaction pooling rebinds the server between statements.
_DIRECT_URL = (_DATABASE_URL_TEST or '').replace('-pooler', '')


def test_reaped_connection_recovers_via_database_reconnect():
	# Simulates a PgBouncer/Neon reap in seconds: kill the backend, then prove @database_reconnect reopens a fresh session and retries.
	with db.session(_DIRECT_URL) as conn:
		db.init_schema(conn)
		with conn.cursor() as cur:
			cur.execute('DELETE FROM papers WHERE paper_id = %s', ('reaped-conn-test',))
			cur.execute('SELECT pg_backend_pid()')
			pid = cur.fetchone()[0]

		# Kill conn's backend from a separate session; conn's next statement now hits a dropped server.
		with db.session(_DIRECT_URL) as killer, killer.cursor() as cur:
			cur.execute('SELECT pg_terminate_backend(%s)', (pid,))

		# upsert_paper is @database_reconnect-wrapped: the first execute raises OperationalError, the retry runs on a fresh session.
		db.upsert_paper(conn, {'paperId': 'reaped-conn-test', 'title': 'roundtrip'})

	with db.session(_DIRECT_URL) as conn, conn.cursor() as cur:
		cur.execute('SELECT title FROM papers WHERE paper_id = %s', ('reaped-conn-test',))
		assert cur.fetchone() == ('roundtrip',)
