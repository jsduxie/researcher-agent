# Researcher Agent

[![codecov](https://codecov.io/gh/jsduxie/researcher-agent/graph/badge.svg)](https://codecov.io/gh/jsduxie/researcher-agent)

Daily research-paper digest. Fetches Semantic Scholar results, scores and summarises them with Gemini, emails the digest. Portfolio project tied to an MEng thesis on mental health classification and explainability.

## Environment variables

Copy `.env.example` to `.env` for local use. Every variable except `DATABASE_URL_TEST` and `CODECOV_TOKEN` is required by the cron run.

| Variable | Required | Where it's used | Notes |
|---|---|---|---|
| `SEMANTIC_SCHOLAR_API_KEY` | yes | `fetcher.py` | Sent as `x-api-key` on every Semantic Scholar request. Free; request at https://www.semanticscholar.org/product/api |
| `GEMINI_API_KEY` | yes | `scorer.py`, `summariser.py` | Free tier of Gemini Flash |
| `GMAIL_USER` | yes | `emailer.py` | Sender Gmail address |
| `GMAIL_APP_PASSWORD` | yes | `emailer.py` | Google account app password (not the login password) |
| `EMAIL_TO` | yes | `emailer.py` | Recipient address |
| `DATABASE_URL` | yes | `db.py`, `main.py` | Neon Postgres connection string |
| `DATABASE_URL_TEST` | tests only | `tests/` | Separate Neon branch for `pytest -m integration` |
| `CODECOV_TOKEN` | CI only | GitHub Actions | Upload step in the workflow |

## Configuration

Pipeline behaviour lives in the `app_config` table in Neon, edited from the dashboard's Configuration page. On a fresh database the table is seeded from `config/digest.example.yaml`; after that, the file is only an example and Neon is the source of truth. Gemini-facing keys:

| Key | Default | Purpose |
|---|---|---|
| `gemini_model` | `gemini-3.1-flash-lite` | Model used by scorer and summariser |
| `gemini_base_url` | `https://generativelanguage.googleapis.com/v1beta` | Base URL for `generateContent` |
| `gemini_upload_base_url` | `https://generativelanguage.googleapis.com/upload/v1beta` | Base URL for the Files API resumable upload |

Swap the model by editing `gemini_model` on the Configuration page; the scorer and summariser pick it up on the next run with no code change.

## Running tests with coverage

Install dev dependencies:

```bash
pip install -r requirements-dev.txt
```

Run unit tests with coverage (the default local loop):

```bash
pytest -m 'not integration' --cov --cov-branch --cov-report=term
```

Run integration tests against a live Postgres branch (the secret `DATABASE_URL_TEST` points at it in CI):

```bash
DATABASE_URL_TEST='postgresql://...' pytest -m integration --cov-append --cov-branch --cov-report=term
```

To run the same gates CI enforces, after one or both pytest runs above:

```bash
coverage report --fail-under=95 # total coverage gate
python scripts/check_coverage.py # per-module floor (90%)
```

Codecov posts a per-file coverage comment on every pull request and a project status check that mirrors the total gate. The badge above tracks the latest `main` figure.
