# Researcher Agent

[![codecov](https://codecov.io/gh/jsduxie/researcher-agent/graph/badge.svg)](https://codecov.io/gh/jsduxie/researcher-agent)

Daily research-paper digest. Fetches Semantic Scholar results, scores and summarises them with Gemini, emails the digest. Portfolio project tied to an MEng thesis on mental health classification and explainability.

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
