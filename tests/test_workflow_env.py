from pathlib import Path

import pytest
import yaml

import main

_WORKFLOW_PATH = Path(__file__).resolve().parents[1] / '.github' / 'workflows' / 'research_agent.yaml'


def _send_digest_run_step_env():
	workflow = yaml.safe_load(_WORKFLOW_PATH.read_text())
	send_digest = workflow['jobs']['send-digest']
	step = next(s for s in send_digest['steps'] if s.get('name') == 'Run research agent')
	return step['env']


@pytest.mark.parametrize('var', main.REQUIRED_ENV_VARS)
def test_send_digest_run_step_maps_every_required_env_var(var):
	# Regression guard for #19: a secret in repo settings is not exposed to the runner unless the step's env block lists it.
	env = _send_digest_run_step_env()
	assert var in env, (
		f'{var} is required by main.REQUIRED_ENV_VARS but is not mapped in the '
		f'send-digest "Run research agent" step. Add it to the env block so the '
		f'GitHub Actions runner exposes it to main.py.'
	)


@pytest.mark.parametrize('var', main.REQUIRED_ENV_VARS)
def test_send_digest_run_step_sources_required_var_from_secrets(var):
	env = _send_digest_run_step_env()
	expression = env[var]
	assert f'secrets.{var}' in expression, (
		f'{var} in the send-digest step does not reference secrets.{var}; '
		f'got {expression!r}. A hardcoded value would leak in workflow logs.'
	)
