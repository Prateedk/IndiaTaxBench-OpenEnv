"""Tests for scripts/notebook_env_helpers.py."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HELPERS = ROOT / "scripts" / "notebook_env_helpers.py"


@pytest.fixture(scope="module")
def nbh():
    spec = importlib.util.spec_from_file_location("notebook_env_helpers", HELPERS)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(mod)
    return mod


def test_parse_prediction_roundtrip(nbh):
    raw = '{"total": 1.5, "initial_tax": 2, "surcharge": 0, "cess": 0.25}'
    d = nbh.parse_prediction(raw)
    assert d == {"total": 1.5, "initial_tax": 2.0, "surcharge": 0.0, "cess": 0.25}


def test_parse_prediction_fenced(nbh):
    raw = "```json\n{\"total\": 1, \"initial_tax\": 0, \"surcharge\": 0, \"cess\": 0}\n```"
    d = nbh.parse_prediction(raw)
    assert d["total"] == 1.0


def test_unwrap_observation(nbh):
    obs = {"task_id": "x", "reward": 0.5}
    assert nbh.unwrap_observation({"observation": obs}) == obs
    assert nbh.unwrap_observation(obs) == obs


def test_advisor_parse_and_build(nbh):
    raw = (
        '{"filing_profile_summary": "'
        + ("x" * 25)
        + '", "next_year_actions": ['
        '{"action": "use 80C", "rationale": "maximize"}, '
        '{"action": "document HRA", "rationale": "rent proof"}], '
        '"cautions": ["get professional help if needed."]}'
    )
    d = nbh.parse_advice_text(raw)
    assert "filing_profile_summary" in d
    m = nbh.build_advisor_messages("{}", "salary_metro_80c_fy2425", task_description="d")
    assert m[0]["role"] == "system"
    assert "Task id:" in m[1]["content"]
