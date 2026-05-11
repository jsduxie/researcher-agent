from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _project_python_files():
	for path in _PROJECT_ROOT.rglob('*.py'):
		if any(part.startswith('.') for part in path.parts):
			continue
		yield path


_PY_FILES = sorted(_project_python_files())


@pytest.mark.parametrize('path', _PY_FILES, ids=lambda p: str(p.relative_to(_PROJECT_ROOT)))
def test_python_source_is_ascii_only(path):
	text = path.read_text(encoding='utf-8')
	try:
		text.encode('ascii')
	except UnicodeEncodeError as e:
		line_no = text[: e.start].count('\n') + 1
		offending = text[e.start]
		line = text.splitlines()[line_no - 1].strip()
		pytest.fail(f'{path.relative_to(_PROJECT_ROOT)}:{line_no} non-ASCII {offending!r} in: {line!r}')
