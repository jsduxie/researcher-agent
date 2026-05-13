import json
from pathlib import Path

from scripts.check_coverage import PER_MODULE_FLOOR, main


def _write_coverage_json(path: Path, files: dict) -> None:
	path.write_text(json.dumps({'files': files}))


def test_passes_when_every_module_meets_the_floor(tmp_path, capsys):
	cov = tmp_path / 'coverage.json'
	# b.py sits exactly on the floor; the gate uses `<` so equality passes.
	_write_coverage_json(
		cov,
		{
			'a.py': {'summary': {'num_statements': 10, 'percent_covered': 100.0}},
			'b.py': {'summary': {'num_statements': 20, 'percent_covered': PER_MODULE_FLOOR}},
		},
	)

	assert main(cov) == 0
	assert 'passed' in capsys.readouterr().out


def test_fails_and_reports_each_module_below_the_floor(tmp_path, capsys):
	cov = tmp_path / 'coverage.json'
	_write_coverage_json(
		cov,
		{
			'good.py': {'summary': {'num_statements': 10, 'percent_covered': 100.0}},
			'bad.py': {'summary': {'num_statements': 10, 'percent_covered': 80.0}},
			'worse.py': {'summary': {'num_statements': 50, 'percent_covered': 42.5}},
		},
	)

	assert main(cov) == 1

	out = capsys.readouterr().out
	assert 'failed' in out
	assert 'bad.py: 80.0%' in out
	assert 'worse.py: 42.5%' in out
	# Modules above the floor must not appear in the failure report.
	assert 'good.py' not in out


def test_skips_empty_modules_regardless_of_reported_percentage(tmp_path, capsys):
	# A 0-statement file should be ignored even if its percent_covered is below the floor (coverage.py can emit 0.0 here); without the skip this would fail the gate.
	cov = tmp_path / 'coverage.json'
	_write_coverage_json(
		cov,
		{
			'covered.py': {'summary': {'num_statements': 10, 'percent_covered': 100.0}},
			'empty.py': {'summary': {'num_statements': 0, 'percent_covered': 0.0}},
		},
	)

	assert main(cov) == 0
	assert 'passed' in capsys.readouterr().out
