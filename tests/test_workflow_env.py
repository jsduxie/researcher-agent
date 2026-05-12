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
	# Regression guard for #19: a secret in repo settings is not exposed to the
	# runner unless the step's env block lists it.
	env = _send_digest_run_step_env()
	assert var in env, f'{var} is in main.REQUIRED_ENV_VARS but missing from the send-digest step env block.'


@pytest.mark.parametrize('var', main.REQUIRED_ENV_VARS)
def test_send_digest_run_step_sources_required_var_from_secrets(var):
	env = _send_digest_run_step_env()
	expression = env[var]
	assert f'secrets.{var}' in expression, f'{var} must reference secrets.{var}, got {expression!r}.'
