import os
import sys
import types

import pytest

import embedder


@pytest.fixture(autouse=True)
def _clear_model_cache():
	embedder._model.cache_clear()
	yield
	embedder._model.cache_clear()


@pytest.fixture
def fake_sentence_transformers(mocker):
	# Injects a stub package so unit tests never import torch or download the model.
	encoder = mocker.Mock()
	encoder.encode.return_value = [[0.6, 0.8], [1.0, 0.0]]
	module = types.ModuleType('sentence_transformers')
	module.SentenceTransformer = mocker.Mock(return_value=encoder)
	mocker.patch.dict(sys.modules, {'sentence_transformers': module})
	return module


# -- embed_texts --


def test_embed_texts_returns_float_lists(fake_sentence_transformers):
	assert embedder.embed_texts(['a', 'b']) == [[0.6, 0.8], [1.0, 0.0]]


def test_embed_texts_requests_normalised_embeddings(fake_sentence_transformers):
	embedder.embed_texts(['a'])
	encoder = fake_sentence_transformers.SentenceTransformer.return_value
	assert encoder.encode.call_args.kwargs['normalize_embeddings'] is True


def test_embed_texts_loads_the_pinned_model_once_across_calls(fake_sentence_transformers):
	embedder.embed_texts(['a'])
	embedder.embed_texts(['b'])
	fake_sentence_transformers.SentenceTransformer.assert_called_once_with(embedder.EMBEDDING_MODEL)


# -- similarity --


def test_similarity_is_dot_product():
	assert embedder.similarity([1.0, 0.0], [0.6, 0.8]) == pytest.approx(0.6)


def test_similarity_identical_normalised_vectors_is_one():
	assert embedder.similarity([0.6, 0.8], [0.6, 0.8]) == pytest.approx(1.0)


def test_similarity_rejects_mismatched_dimensions():
	with pytest.raises(ValueError):
		embedder.similarity([1.0], [1.0, 0.0])


# -- rank_by_similarity --


def test_rank_by_similarity_orders_most_similar_first():
	query = [1.0, 0.0]
	vectors = [[0.0, 1.0], [1.0, 0.0], [0.7, 0.7]]
	assert embedder.rank_by_similarity(query, vectors) == [1, 2, 0]


def test_rank_by_similarity_is_stable_for_ties():
	query = [1.0, 0.0]
	vectors = [[0.5, 0.5], [0.5, 0.5]]
	assert embedder.rank_by_similarity(query, vectors) == [0, 1]


def test_rank_by_similarity_empty_input():
	assert embedder.rank_by_similarity([1.0, 0.0], []) == []


# -- live model (integration) --


@pytest.mark.integration
@pytest.mark.skipif(
	os.environ.get('CI') is None and not os.environ.get('DATABASE_URL_TEST'),
	reason='live model load follows the integration-run gates',
)
def test_real_model_embeds_to_normalised_384_dims():
	[vector] = embedder.embed_texts(['transformer text classification'])
	assert len(vector) == embedder.EMBEDDING_DIM
	assert sum(v * v for v in vector) == pytest.approx(1.0, abs=1e-3)
