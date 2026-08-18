# Contributing

Thanks for taking an interest in Researcher Agent. It started as a companion tool for my MEng thesis. It is a real, tested project, and contributions are welcome, whether that is a bug fix, a docs improvement or a new feature.

This guide covers local setup, how the tests work, and the conventions the repo follows.

## Ways to contribute

- For a bug or an idea, open an issue. For a bug, include the exact error and the steps that triggered it.
- Anything labelled [`good first issue`](https://github.com/jsduxie/researcher-agent/labels/good%20first%20issue) is scoped as an entry point.
- For code or docs, fork, branch and open a pull request against `main`. One focused change is reviewed and merged faster than several bundled together.

## Local setup

The project targets Python 3.11, the version CI runs and the floor in `pyproject.toml`.

```bash
python -m venv .venv && source .venv/bin/activate
```

Install PyTorch first, from the CPU-only index:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

`torch` arrives transitively via `sentence-transformers`, but the default Linux wheel bundles CUDA and is roughly 2GB. Installing the CPU wheel explicitly first keeps the environment small and matches what CI does. This is the step people most often skip.

Then the project and development dependencies:

```bash
pip install -r requirements.txt -r requirements-dev.txt
```

Finally, copy the environment template and fill in what you need:

```bash
cp .env.example .env
```

You do not need a local Postgres. Unit tests mock the database, and the integration tests run against a real Neon branch via `DATABASE_URL_TEST` (used in CI). If you are only changing logic covered by unit tests, leave the database variables blank.

## Running the agent

The fastest way to see a full run without any network calls, API keys or emails is dry-run mode, which reads from fixtures and prints the rendered HTML to stdout:

```bash
python main.py --dry-run
```

Two ablation flags disable a single stage so you can compare behaviour:

- `--no-prefilter` scores the full queue without the embedding pre-filter.
- `--no-fewshot` disables few-shot calibration regardless of how many ratings exist.

## Tests and coverage

Unit tests are the default local loop:

```bash
pytest -m 'not integration' --cov --cov-branch --cov-report=term
```

Integration tests need a throwaway Postgres branch:

```bash
DATABASE_URL_TEST='postgresql://...' pytest -m integration --cov-append --cov-branch --cov-report=term
```

CI enforces two coverage floors. A pull request will not pass until both are met:

```bash
coverage report --fail-under=95 # total coverage floor
python scripts/check_coverage.py # per-module floor (90%)
```

New code is expected to come with tests. For a bug fix, a test that fails before the change and passes after it is the clearest way to show the fix works.

## Snapshot tests

The email digest is asserted byte-for-byte against `tests/fixtures/email_snapshot.html` in `test_build_email_renders_snapshot_byte_for_byte`. This is deliberate: it makes any change to the rendered HTML visible in review rather than silent.

If you intentionally change the email rendering, the snapshot will fail. Regenerate the fixture and commit it in the same pull request, so the diff shows exactly how the output changed. The test builds its input from `tests/fixtures/enriched_papers.json`, the seed-derived config, and a frozen date of `2026-01-15`. Reproduce those same inputs to regenerate:

```python
import json
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import config
from render import build_email

fixtures = Path('tests/fixtures')
papers = json.loads((fixtures / 'enriched_papers.json').read_text())
cfg = config.Config(**config.load_seed())

with patch('render.datetime') as dt:
	dt.now.return_value = datetime(2026, 1, 15)
	(fixtures / 'email_snapshot.html').write_text(build_email(papers, cfg))
```

Read the diff before committing a regenerated snapshot. The test exists to catch rendering changes you did not intend.

## Code style and conventions

- British English in code, comments, commits and docs (summarise, behaviour, colour).
- `ruff` handles linting and formatting. Install the hooks once and they run on every commit:

  ```bash
  pre-commit install
  ```

  To check the whole tree manually: `ruff check .` and `ruff format --check .`.
- Comments explain the non-obvious why, not the what. Match the style of the surrounding code.
- Commits follow [Conventional Commits](https://www.conventionalcommits.org/), as do pull request titles, for example `fix(email): enable Jinja2 autoescape` or `docs(readme): add architecture overview`.

## Pull request workflow

1. Branch from `main` as `<issue>-<type>-<description>`, for example `48-fix-jinja2-autoescape` or `49-docs-readme-overview`. The issue number first makes the branch and the discussion easy to line up.
2. Keep the pull request focused on one change. If you spot something unrelated, a separate pull request is easier to review and merge.
3. Make sure the full CI suite passes: lint, unit, integration and both coverage floors. The commands are listed above, so run them locally first.
4. Open the pull request against `main` with a short description of what changed and why. Link the issue it closes with `Fixes #NN`.

I review pull requests as soon as I can.

## Licence

By contributing, you agree that your contributions are licensed under the [MIT Licence](LICENSE) that covers this project.
