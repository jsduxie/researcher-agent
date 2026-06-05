import functools

# The dimension is baked into the papers.embedding vector(384) column, so the model is a code constant rather than app config.
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
EMBEDDING_DIM = 384


@functools.lru_cache(maxsize=1)
def _model():
	# Imported lazily: torch costs seconds to import and most of the test suite never touches it.
	from sentence_transformers import SentenceTransformer

	return SentenceTransformer(EMBEDDING_MODEL)


def embed_texts(texts):
	# normalize_embeddings makes cosine similarity a plain dot product downstream.
	return [[float(v) for v in row] for row in _model().encode(list(texts), normalize_embeddings=True)]


def similarity(a, b):
	# Vectors arrive normalised, so the dot product is the cosine similarity.
	return sum(x * y for x, y in zip(a, b, strict=True))


def rank_by_similarity(query_vector, vectors):
	# Returns indices into vectors, most similar first; stable for ties.
	sims = [similarity(query_vector, v) for v in vectors]
	return sorted(range(len(vectors)), key=lambda i: -sims[i])
