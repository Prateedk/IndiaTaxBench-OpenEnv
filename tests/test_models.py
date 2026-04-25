"""Model validation tests."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import IndiaTaxBenchAction, IndiaTaxBenchObservation


def test_action_roundtrip():
    a = IndiaTaxBenchAction(
        action_type="submit_prediction",
        predicted_total=1.0,
        predicted_initial_tax=2.0,
        predicted_surcharge=0.0,
        predicted_cess=3.0,
    )
    d = a.model_dump(exclude_none=True)
    assert d["action_type"] == "submit_prediction"
    assert d["predicted_total"] == 1.0


def test_observation_defaults():
    o = IndiaTaxBenchObservation(
        scenario_json="{}",
        task_id="t1",
        task_description="d",
        feedback="f",
        done=False,
        reward=0.5,
    )
    assert o.valid_actions == []
