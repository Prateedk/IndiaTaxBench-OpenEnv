"""HTTP-level advisor flow: /reset + /step must parse advice_text and set parsed_ok (regression for curl/Space)."""

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from india_tax_bench.server import app as app_mod  # noqa: E402


@pytest.fixture
def client():
    return TestClient(app_mod.app)


def test_submit_tax_advice_parses_curl_shaped_json(client: TestClient):
    r0 = client.post(
        "/reset", json={"task": "salary_metro_80c_fy2425", "advisor": True}
    )
    assert r0.status_code == 200
    assert r0.json()["observation"]["episode_mode"] == "advisor"

    # Hard-rubric: continuous partial credit on summary length, 4 action slots, cautions, keyphrases
    advice = {
        "filing_profile_summary": (
            "Metro private filer: optimize Chapter VI-A, 80C, and 80CCD NPS; document HRA and payroll."
        ),
        "next_year_actions": [
            {"action": "Maximize 80C and review Chapter VI-A basket", "rationale": "NPS/ELSS/EPF under caps"},
            {"action": "Maintain rent proofs for HRA in metro", "rationale": "Exemption needs employer alignment"},
            {"action": "Plan 80D and cash-flow for advance tax on bonus", "rationale": "Avoid year-end underpayment fees"},
        ],
        "cautions": [
            "Verify ITR and Form 12BB with current rules.",
            "Get CA help if salary and bonus change materially next year.",
        ],
    }
    r1 = client.post(
        "/step",
        json={
            "action": {
                "action_type": "submit_tax_advice",
                "advice_text": json.dumps(advice),
            }
        },
    )
    assert r1.status_code == 200
    obs = r1.json()["observation"]
    sa = obs["submitted_advice"]
    assert len(sa) == 1
    assert sa[0]["parsed_ok"] is True
    assert sa[0]["rubric"] > 0.5
    assert "structured JSON=yes" in obs["feedback"]

