import pytest

from scripts import seed_dashboard_test


def test_seed_exits_when_database_url_test_is_unset(monkeypatch):
	monkeypatch.delenv('DATABASE_URL_TEST', raising=False)
	with pytest.raises(SystemExit, match='DATABASE_URL_TEST'):
		seed_dashboard_test.seed()


def test_seed_upserts_every_paper_and_summary_then_writes_rating_rows(mocker, monkeypatch):
	# Every paper in the seed list must reach upsert_paper + upsert_summary; every (paper_id, rating) pair must reach insert_rating.
	monkeypatch.setenv('DATABASE_URL_TEST', 'postgresql://fake')
	conn = mocker.MagicMock()
	connect = mocker.patch('scripts.seed_dashboard_test.db.connect', return_value=conn)
	init_schema = mocker.patch('scripts.seed_dashboard_test.db.init_schema')
	upsert_paper = mocker.patch('scripts.seed_dashboard_test.db.upsert_paper')
	upsert_summary = mocker.patch('scripts.seed_dashboard_test.db.upsert_summary')
	insert_rating = mocker.patch('scripts.seed_dashboard_test.db.insert_rating')

	seed_dashboard_test.seed()

	connect.assert_called_once_with('postgresql://fake')
	init_schema.assert_called_once_with(conn)
	assert upsert_paper.call_count == len(seed_dashboard_test._PAPERS)
	assert upsert_summary.call_count == len(seed_dashboard_test._PAPERS)
	assert insert_rating.call_count == len(seed_dashboard_test._SEED_RATINGS)


def test_seed_passes_model_version_sentinel_for_every_summary(mocker, monkeypatch):
	# The 'test-model-v1' sentinel keeps the seed data out of the no-hardcoded-gemini-model guard's reach; verify every persisted summary carries it.
	monkeypatch.setenv('DATABASE_URL_TEST', 'postgresql://fake')
	mocker.patch('scripts.seed_dashboard_test.db.connect', return_value=mocker.MagicMock())
	mocker.patch('scripts.seed_dashboard_test.db.init_schema')
	mocker.patch('scripts.seed_dashboard_test.db.upsert_paper')
	upsert_summary = mocker.patch('scripts.seed_dashboard_test.db.upsert_summary')
	mocker.patch('scripts.seed_dashboard_test.db.insert_rating')

	seed_dashboard_test.seed()

	for call in upsert_summary.call_args_list:
		assert call.args[3] == 'test-model-v1'


def test_seed_writes_each_rating_with_the_configured_paper_id_and_value(mocker, monkeypatch):
	monkeypatch.setenv('DATABASE_URL_TEST', 'postgresql://fake')
	mocker.patch('scripts.seed_dashboard_test.db.connect', return_value=mocker.MagicMock())
	mocker.patch('scripts.seed_dashboard_test.db.init_schema')
	mocker.patch('scripts.seed_dashboard_test.db.upsert_paper')
	mocker.patch('scripts.seed_dashboard_test.db.upsert_summary')
	insert_rating = mocker.patch('scripts.seed_dashboard_test.db.insert_rating')

	seed_dashboard_test.seed()

	written = [(c.args[1], c.args[2]) for c in insert_rating.call_args_list]
	assert written == list(seed_dashboard_test._SEED_RATINGS)
