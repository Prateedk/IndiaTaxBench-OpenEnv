"""Tests for server.advisor_rubric."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.advisor_rubric import parse_advice_json, score_advice_object


def test_score_reasonable_for_valid_json():
    obj = {
        "filing_profile_summary": "x" * 30,
        "next_year_actions": [
            {"action": "maximize 80C basket", "rationale": "r1"},
            {"action": "plan HRA proofs", "rationale": "r2"},
        ],
        "cautions": ["c1"],
    }
    s = score_advice_object(obj, "salary_metro_80c_fy2425")
    assert 0.0 <= s <= 1.0
    assert s > 0.3


def test_parse_advice_json():
    t = '{"filing_profile_summary": "' + "y" * 22 + '", "next_year_actions": [], "cautions": []}'
    o = parse_advice_json(t)
    assert o is not None
    assert "filing_profile_summary" in o


def test_parse_advice_json_double_encoded_string():
    import json

    inner = {
        "filing_profile_summary": "x" * 30,
        "next_year_actions": [{"action": "a1", "rationale": "r1"}],
        "cautions": ["c1"],
    }
    t = json.dumps(json.dumps(inner))
    o = parse_advice_json(t)
    assert o is not None
    assert o.get("filing_profile_summary") == "x" * 30
