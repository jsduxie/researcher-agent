CREATE TABLE IF NOT EXISTS papers (
	paper_id TEXT PRIMARY KEY,
	title TEXT,
	abstract TEXT,
	year INTEGER,
	citation_count INTEGER,
	url TEXT,
	doi TEXT,
	pdf_url TEXT,
	fetched_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
	updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
	scored_at TIMESTAMPTZ,
	score_attempts INTEGER NOT NULL DEFAULT 0
);

-- Idempotent migration for databases created before these columns existed.
ALTER TABLE papers ADD COLUMN IF NOT EXISTS scored_at TIMESTAMPTZ;
ALTER TABLE papers ADD COLUMN IF NOT EXISTS score_attempts INTEGER NOT NULL DEFAULT 0;

CREATE TABLE IF NOT EXISTS paper_authors (
	paper_id TEXT NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
	position INTEGER NOT NULL,
	name TEXT NOT NULL,
	PRIMARY KEY (paper_id, position)
);

CREATE TABLE IF NOT EXISTS summaries (
	paper_id TEXT PRIMARY KEY REFERENCES papers(paper_id) ON DELETE CASCADE,
	methodology TEXT,
	findings TEXT,
	relevance TEXT,
	limitations TEXT,
	model_version TEXT,
	created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS runs (
	id BIGSERIAL PRIMARY KEY,
	started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
	finished_at TIMESTAMPTZ,
	papers_fetched INTEGER NOT NULL DEFAULT 0,
	papers_kept INTEGER
);

CREATE TABLE IF NOT EXISTS ratings (
	id BIGSERIAL PRIMARY KEY,
	paper_id TEXT NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
	rating INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
	created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS summary_feedback (
	id BIGSERIAL PRIMARY KEY,
	paper_id TEXT NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
	field TEXT NOT NULL,
	thumbs INTEGER NOT NULL CHECK (thumbs IN (-1, 1)),
	correction TEXT,
	created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
