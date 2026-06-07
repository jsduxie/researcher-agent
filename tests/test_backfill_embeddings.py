import pytest

from scripts import backfill_embeddings


def test_backfill_exits_when_database_url_is_unset(monkeypatch):
	monkeypatch.delenv('DATABASE_URL', raising=False)
	with pytest.raises(SystemExit, match='DATABASE_URL'):
		backfill_embeddings.backfill()


def test_backfill_does_nothing_when_no_papers_missing_embeddings(mocker, monkeypatch):
	monkeypatch.setenv('DATABASE_URL', 'postgresql://fake')
	mocker.patch('scripts.backfill_embeddings.db.connect', return_value=mocker.MagicMock())
	mocker.patch('scripts.backfill_embeddings.db.list_papers_missing_embeddings', return_value=[])
	embed_texts = mocker.patch('scripts.backfill_embeddings.embedder.embed_texts')
	set_embedding = mocker.patch('scripts.backfill_embeddings.db.set_paper_embedding')

	backfill_embeddings.backfill()

	embed_texts.assert_not_called()
	set_embedding.assert_not_called()


def test_backfill_embeds_pipeline_text_shape_and_persists_each_vector(mocker, monkeypatch):
	monkeypatch.setenv('DATABASE_URL', 'postgresql://fake')
	conn = mocker.MagicMock()
	connect = mocker.patch('scripts.backfill_embeddings.db.connect', return_value=conn)
	papers = [{'paper_id': 'p1', 'title': 't1', 'abstract': 'a1'}, {'paper_id': 'p2', 'title': None, 'abstract': None}]
	mocker.patch('scripts.backfill_embeddings.db.list_papers_missing_embeddings', return_value=papers)
	embed_texts = mocker.patch('scripts.backfill_embeddings.embedder.embed_texts', return_value=[[0.1], [0.2]])
	set_embedding = mocker.patch('scripts.backfill_embeddings.db.set_paper_embedding')

	backfill_embeddings.backfill()

	connect.assert_called_once_with('postgresql://fake')
	# Title, blank line, abstract, with empty-string fallbacks for missing fields, matching main._prefilter_scoring_queue.
	embed_texts.assert_called_once_with(['t1\n\na1', '\n\n'])
	assert [(c.args[1], c.args[2]) for c in set_embedding.call_args_list] == [('p1', [0.1]), ('p2', [0.2])]
	assert all(c.args[0] is conn for c in set_embedding.call_args_list)
