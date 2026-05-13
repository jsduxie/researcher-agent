import json
import sys
from pathlib import Path

PER_MODULE_FLOOR = 90.0
DEFAULT_COVERAGE_JSON = Path('coverage.json')


def main(coverage_path: Path = DEFAULT_COVERAGE_JSON) -> int:
	data = json.loads(coverage_path.read_text())
	failures = []
	for filename, info in sorted(data['files'].items()):
		summary = info['summary']
		# Empty source files (0 statements) have a meaningless percent_covered; skip. Belt-and-braces with coverage's skip_empty in pyproject.
		if summary['num_statements'] == 0:
			continue
		pct = summary['percent_covered']
		if pct < PER_MODULE_FLOOR:
			failures.append((filename, pct))

	if failures:
		print(f'Per-module coverage gate failed (floor {PER_MODULE_FLOOR:.0f}%):')
		for filename, pct in failures:
			print(f'  {filename}: {pct:.1f}%')
		return 1

	print(f'Per-module coverage gate passed: all source modules >= {PER_MODULE_FLOOR:.0f}%.')
	return 0


if __name__ == '__main__':
	sys.exit(main())
