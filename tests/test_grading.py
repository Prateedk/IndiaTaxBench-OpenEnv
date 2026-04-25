"""Grading helper tests."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.india_tax_bench_environment import _field_score, _score_prediction


def test_field_score_exact():
    assert _field_score(100.0, 100.0) == 1.0


def test_field_score_zero_gold():
    assert _field_score(0.0, 0.0) == 1.0


def test_score_prediction_perfect():
    oracle = {
        "total": 242580.0,
        "initial_tax": 233250.0,
        "surcharge": 0.0,
        "cess": 9330.0,
    }
    mean_s, bd = _score_prediction(oracle, oracle)
    assert mean_s == 1.0
    assert all(v == 1.0 for v in bd.values())
