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

    advice = {
        "filing_profile_summary": "Sample summary text for a metro filer with salary and 80C.",
        "next_year_actions": [
            {"action": "Maximize 80C within 1.5L", "rationale": "ELSS/PPF"},
            {"action": "HRA rent proofs", "rationale": "Documentation"},
        ],
        "cautions": ["Engage a CA for large changes."],
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

