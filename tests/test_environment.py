"""Integration tests for IndiaTaxBenchEnvironment."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import IndiaTaxBenchAction, IndiaTaxBenchObservation
from server.india_tax_bench_environment import (
    IndiaTaxBenchEnvironment,
    MAX_STEPS,
    REWARD_MAX,
    REWARD_MIN,
)
from server.tasks import ALL_TASK_IDS, TASKS


class TestReset:
    def test_reset_returns_observation(self):
        env = IndiaTaxBenchEnvironment()
        tid = ALL_TASK_IDS[0]
        obs = env.reset(task=tid)
        assert isinstance(obs, IndiaTaxBenchObservation)
        assert obs.task_id == tid
        assert obs.done is False
        assert obs.task_difficulty in ("easy", "medium", "hard")
        assert "tax_settings" in obs.scenario_json

    def test_reset_random(self):
        env = IndiaTaxBenchEnvironment()
        obs = env.reset()
        assert obs.task_id in ALL_TASK_IDS

    def test_reset_valid_actions(self):
        env = IndiaTaxBenchEnvironment()
        obs = env.reset(task=ALL_TASK_IDS[0])
        assert "submit_prediction" in obs.valid_actions
        assert "finalize" in obs.valid_actions


class TestSubmitAndFinalize:
    def test_perfect_oracle_high_reward(self):
        tid = ALL_TASK_IDS[0]
        o = TASKS[tid]["oracle"]
        env = IndiaTaxBenchEnvironment()
        env.reset(task=tid)
        obs = env.step(
            IndiaTaxBenchAction(
                action_type="submit_prediction",
                predicted_total=o["total"],
                predicted_initial_tax=o["initial_tax"],
                predicted_surcharge=o["surcharge"],
                predicted_cess=o["cess"],
            )
        )
        assert obs.done is False
        assert obs.submitted_predictions[-1]["score"] > 0.99
        fin = env.step(IndiaTaxBenchAction(action_type="finalize"))
        assert fin.done is True
        assert float(fin.reward or 0) > 0.5

    def test_bad_prediction_low_finalize(self):
        tid = ALL_TASK_IDS[0]
        env = IndiaTaxBenchEnvironment()
        env.reset(task=tid)
        env.step(
            IndiaTaxBenchAction(
                action_type="submit_prediction",
                predicted_total=0.0,
                predicted_initial_tax=0.0,
                predicted_surcharge=0.0,
                predicted_cess=0.0,
            )
        )
        fin = env.step(IndiaTaxBenchAction(action_type="finalize"))
        assert fin.done is True
        assert float(fin.reward or 0) < float(REWARD_MAX)


class TestAdvisorEpisode:
    def test_advisor_reset_and_finalize(self):
        tid = ALL_TASK_IDS[0]
        env = IndiaTaxBenchEnvironment()
        obs = env.reset(task=tid, advisor=True)
        assert obs.episode_mode == "advisor"
        assert "submit_tax_advice" in obs.valid_actions
        sample = json.dumps(
            {
                "filing_profile_summary": (
                    "Test filer: metro, private, Chapter VI-A and 80C / NPS planning; "
                    "FY 2024–25 baseline with documented payroll deductions. "
                    "Long enough to satisfy the hard-rubric length band."
                ),
                "next_year_actions": [
                    {
                        "action": "Review 80C and 80CCD NPS room",
                        "rationale": "Chapter VI-A cap",
                    },
                    {
                        "action": "HRA rent proofs vs salary structure",
                        "rationale": "exemption support",
                    },
                    {
                        "action": "80D and advance-tax timing for bonus",
                        "rationale": "avoid cashflow gaps",
                    },
                ],
                "cautions": [
                    "Get CA review for large life changes next year.",
                    "Re-verify Form 12BB and ITR with current law.",
                ],
            }
        )
        o1 = env.step(
            IndiaTaxBenchAction(
                action_type="submit_tax_advice",
                advice_text=sample,
            )
        )
        assert o1.done is False
        assert o1.submitted_advice
        fin = env.step(IndiaTaxBenchAction(action_type="finalize_advice"))
        assert fin.done is True
        assert REWARD_MIN <= float(fin.reward or 0) <= REWARD_MAX

    def test_advisor_revise(self):
        tid = ALL_TASK_IDS[0]
        env = IndiaTaxBenchEnvironment()
        env.reset(task=tid, advisor=True)
        w1 = (
            '{"filing_profile_summary": "Short.", "next_year_actions": [{"action": "a", "rationale": "b"}], '
            '"cautions": ["c"]}'
        )
        env.step(IndiaTaxBenchAction(action_type="submit_tax_advice", advice_text=w1))
        w2 = (
            '{"filing_profile_summary": "A longer profile summary for the filer with enough text here.", '
            '"next_year_actions": [{"action": "Maximize 80C", "rationale": "Use NPS 80CCD"}, '
            '{"action": "HRA rent receipts", "rationale": "Proof for exemption"}], '
            '"cautions": ["Get CA review if business income grows."]}'
        )
        o2 = env.step(
            IndiaTaxBenchAction(
                action_type="revise_tax_advice",
                item_index=0,
                advice_text=w2,
            )
        )
        assert o2.done is False
        assert o2.submitted_advice[0].get("rubric", 0) >= 0.0
