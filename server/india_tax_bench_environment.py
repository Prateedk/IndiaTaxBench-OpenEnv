"""India income tax old-regime prediction: multi-step OpenEnv environment."""

from __future__ import annotations

import json
import math
import random
from typing import Any, Dict, List, Optional, Tuple, cast
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import (
        IndiaTaxBenchAction,
        IndiaTaxBenchObservation,
        TaskDifficultyLabel,
    )
except ImportError:
    from models import (
        IndiaTaxBenchAction,
        IndiaTaxBenchObservation,
        TaskDifficultyLabel,
    )

try:
    from .tasks import ALL_TASK_IDS, TASKS
except ImportError:
    from server.tasks import ALL_TASK_IDS, TASKS

REWARD_MIN = 0.0
REWARD_MAX = 1.0
MAX_STEPS = 15
MAX_HINTS = 3
HINT_PENALTY = 0.03
STEP_REWARD_SCALE = 0.10
EFFICIENCY_BONUS_SCALE = 0.05
AUTO_FINALIZE_PENALTY = 0.05

ACTION_TYPES = ("submit_prediction", "revise_prediction", "request_context", "finalize")

_ORACLE_KEYS = ("total", "initial_tax", "surcharge", "cess")

_HINTS = [
    "Under the old regime, taxable salary income often reflects standard deduction and Chapter VI-A caps—recompute gross to taxable carefully.",
    "Cess is applied on tax including surcharge; ensure you did not apply cess to a base that already included cess.",
    "HRA exemption interacts with metro vs non-metro rules; verify `is_metro_resident` when HRA and rent are both present.",
    "When several deductions apply, confirm which ones the scenario actually uses before summing eligible amounts.",
]


def _clamp_reward(r: float) -> float:
    return max(REWARD_MIN, min(REWARD_MAX, r))


def _field_score(pred: float, gold: float) -> float:
    """Score in [0, 1] using relative error with an absolute INR floor."""
    if not (math.isfinite(pred) and math.isfinite(gold)):
        return 0.0
    diff = abs(pred - gold)
    if abs(gold) < 1e-9:
        return 1.0 if diff < 1.0 else max(0.0, 1.0 - diff / 500.0)
    denom = max(500.0, 0.02 * abs(gold))
    return max(0.0, min(1.0, 1.0 - diff / denom))


def _score_prediction(
    pred: Dict[str, float], oracle: Dict[str, float]
) -> Tuple[float, Dict[str, float]]:
    """Mean field score and per-field scores."""
    breakdown: Dict[str, float] = {}
    scores: List[float] = []
    for k in _ORACLE_KEYS:
        ps = _field_score(float(pred.get(k, 0.0)), float(oracle[k]))
        breakdown[k] = round(ps, 4)
        scores.append(ps)
    mean_s = sum(scores) / len(scores) if scores else 0.0
    return mean_s, breakdown


def _best_submission_score(submitted: List[Dict[str, Any]]) -> Tuple[float, Dict[str, Any]]:
    if not submitted:
        return 0.0, {}
    best = max(submitted, key=lambda x: float(x.get("score", 0.0)))
    return float(best.get("score", 0.0)), best


def _grade_finalize(
    submitted: List[Dict[str, Any]], oracle: Dict[str, float]
) -> Tuple[float, str]:
    """Base reward in ~[0,1] from best submission vs oracle."""
    if not submitted:
        return 0.0, "No prediction submitted before finalize."
    score, best = _best_submission_score(submitted)
    bd = best.get("field_scores", {})
    parts = [f"{k}={bd.get(k, 0):.3f}" for k in _ORACLE_KEYS]
    feedback = f"Best submission score={score:.4f} (" + ", ".join(parts) + ")"
    return score, feedback


class IndiaTaxBenchEnvironment(Environment):
    """Multi-step prediction of old-regime tax components from JSON scenarios."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._current_task_id: str = ""
        self._done: bool = False
        self._submitted: List[Dict[str, Any]] = []
        self._hints_used: int = 0
        self._cumulative_reward: float = 0.0

    def _valid_actions(self) -> List[str]:
        if self._done:
            return []
        out = ["submit_prediction", "finalize"]
        if self._submitted:
            out.append("revise_prediction")
        if self._hints_used < MAX_HINTS:
            out.append("request_context")
        return sorted(out)

    def _oracle(self) -> Dict[str, float]:
        t = TASKS[self._current_task_id]
        return dict(t["oracle"])

    def _make_obs(
        self,
        feedback: str,
        reward: float,
        done: bool,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> IndiaTaxBenchObservation:
        t = TASKS.get(self._current_task_id, {})
        scenario = t.get("scenario", {})
        meta: Dict[str, Any] = {"task": self._current_task_id}
        if extra_meta:
            meta.update(extra_meta)

        diff_raw = t.get("difficulty", "medium")
        if diff_raw not in ("easy", "medium", "hard"):
            diff_raw = "medium"
        diff = cast(TaskDifficultyLabel, diff_raw)

        return IndiaTaxBenchObservation(
            scenario_json=json.dumps(scenario, ensure_ascii=False, indent=2),
            task_id=self._current_task_id,
            task_description=str(t.get("description", "")),
            task_difficulty=diff,
            feedback=feedback,
            submitted_predictions=list(self._submitted),
            steps_remaining=MAX_STEPS - self._state.step_count,
            hints_used=self._hints_used,
            valid_actions=self._valid_actions() if not done else [],
            done=done,
            reward=_clamp_reward(reward),
            metadata=meta,
        )

    def reset(
        self,
        seed: Optional[int] = None,
        task: Optional[str] = None,
        **kwargs: Any,
    ) -> IndiaTaxBenchObservation:
        if not ALL_TASK_IDS:
            raise RuntimeError("No tasks loaded; check india_tax_capture/data/india_tax_rows.jsonl")

        if task and task in TASKS:
            self._current_task_id = task
        else:
            self._current_task_id = random.choice(ALL_TASK_IDS)

        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._done = False
        self._submitted = []
        self._hints_used = 0
        self._cumulative_reward = 0.0

        return self._make_obs(
            feedback=(
                "Predict old-regime `total`, `initial_tax`, `surcharge`, `cess` (INR). "
                "Use submit_prediction, optionally revise_prediction or request_context, then finalize."
            ),
            reward=REWARD_MIN,
            done=False,
        )

    def step(self, action: IndiaTaxBenchAction) -> IndiaTaxBenchObservation:  # type: ignore[override]
        self._state.step_count += 1

        if self._done:
            return self._make_obs(
                feedback="Episode already done. Call reset().",
                reward=REWARD_MIN,
                done=True,
                extra_meta={"error": "episode_done"},
            )

        if not self._current_task_id:
            self.reset()

        at = action.action_type
        if at not in ACTION_TYPES:
            return self._make_obs(
                feedback=f"Unknown action_type '{at}'. Valid: {', '.join(ACTION_TYPES)}",
                reward=_clamp_reward(self._cumulative_reward * 0.01),
                done=False,
                extra_meta={"error": f"unknown action_type: {at}"},
            )

        oracle = self._oracle()

        if at == "submit_prediction":
            return self._handle_submit(action, oracle)
        if at == "revise_prediction":
            return self._handle_revise(action, oracle)
        if at == "request_context":
            return self._handle_hint()
        if at == "finalize":
            return self._handle_finalize(oracle, voluntary=True)

        return self._make_obs(feedback="Unhandled action.", reward=REWARD_MIN, done=False)

    def _prediction_dict(self, action: IndiaTaxBenchAction) -> Dict[str, float]:
        return {
            "total": float(action.predicted_total if action.predicted_total is not None else 0.0),
            "initial_tax": float(
                action.predicted_initial_tax if action.predicted_initial_tax is not None else 0.0
            ),
            "surcharge": float(
                action.predicted_surcharge if action.predicted_surcharge is not None else 0.0
            ),
            "cess": float(action.predicted_cess if action.predicted_cess is not None else 0.0),
        }

    def _handle_submit(
        self, action: IndiaTaxBenchAction, oracle: Dict[str, float]
    ) -> IndiaTaxBenchObservation:
        pred = self._prediction_dict(action)
        score, breakdown = _score_prediction(pred, oracle)
        entry = {
            **pred,
            "score": round(score, 4),
            "field_scores": breakdown,
        }
        self._submitted.append(entry)
        step_reward = score * STEP_REWARD_SCALE
        self._cumulative_reward += step_reward

        feedback = (
            f"Submitted prediction {len(self._submitted)}: mean_field_score={score:.4f} "
            f"(total={breakdown['total']}, initial_tax={breakdown['initial_tax']}, "
            f"surcharge={breakdown['surcharge']}, cess={breakdown['cess']})"
        )

        if self._state.step_count >= MAX_STEPS:
            return self._handle_finalize(oracle, voluntary=False)

        return self._make_obs(
            feedback=feedback,
            reward=_clamp_reward(step_reward),
            done=False,
            extra_meta={"prediction_score": score, "prediction_index": len(self._submitted) - 1},
        )

    def _handle_revise(
        self, action: IndiaTaxBenchAction, oracle: Dict[str, float]
    ) -> IndiaTaxBenchObservation:
        idx = action.item_index if action.item_index is not None else len(self._submitted) - 1
        if idx is None or idx < 0 or idx >= len(self._submitted):
            return self._make_obs(
                feedback=f"Invalid item_index={idx}. You have {len(self._submitted)} predictions (0-indexed).",
                reward=_clamp_reward(self._cumulative_reward * 0.01),
                done=False,
                extra_meta={"error": "invalid item_index"},
            )

        old = self._submitted[idx]
        old_score = float(old.get("score", 0.0))
        pred = self._prediction_dict(action)
        score, breakdown = _score_prediction(pred, oracle)
        new_entry = {**pred, "score": round(score, 4), "field_scores": breakdown}
        self._submitted[idx] = new_entry

        improvement = score - old_score
        step_reward = max(0.0, improvement) * STEP_REWARD_SCALE
        self._cumulative_reward += step_reward

        feedback = (
            f"Revised prediction {idx}: {old_score:.4f} → {score:.4f} (delta={improvement:+.4f})"
        )

        if self._state.step_count >= MAX_STEPS:
            return self._handle_finalize(oracle, voluntary=False)

        return self._make_obs(
            feedback=feedback,
            reward=_clamp_reward(step_reward),
            done=False,
            extra_meta={"revision_delta": improvement, "item_index": idx},
        )

    def _handle_hint(self) -> IndiaTaxBenchObservation:
        if self._hints_used >= MAX_HINTS:
            return self._make_obs(
                feedback="No hints remaining.",
                reward=_clamp_reward(-HINT_PENALTY),
                done=False,
            )
        hint = _HINTS[self._hints_used % len(_HINTS)]
        self._hints_used += 1
        self._cumulative_reward -= HINT_PENALTY

        if self._state.step_count >= MAX_STEPS:
            return self._handle_finalize(self._oracle(), voluntary=False)

        return self._make_obs(
            feedback=f"Hint ({self._hints_used}/{MAX_HINTS}): {hint}",
            reward=_clamp_reward(-HINT_PENALTY),
            done=False,
            extra_meta={"hint": hint},
        )

    def _handle_finalize(
        self, oracle: Dict[str, float], *, voluntary: bool
    ) -> IndiaTaxBenchObservation:
        base_reward, fb = _grade_finalize(self._submitted, oracle)
        steps_used = self._state.step_count
        efficiency_bonus = EFFICIENCY_BONUS_SCALE * (MAX_STEPS - steps_used) / MAX_STEPS
        hint_penalty = HINT_PENALTY * self._hints_used
        auto_penalty = 0.0 if voluntary else AUTO_FINALIZE_PENALTY

        final_reward = base_reward + efficiency_bonus - hint_penalty - auto_penalty
        final_reward = _clamp_reward(final_reward)

        suffix = " [auto-finalized: max steps reached]" if not voluntary else ""
        full_feedback = (
            f"{fb} | efficiency_bonus={efficiency_bonus:.3f}, "
            f"hint_penalty={hint_penalty:.3f}, auto_penalty={auto_penalty:.3f}{suffix}"
        )

        self._done = True
        return self._make_obs(
            feedback=full_feedback,
            reward=final_reward,
            done=True,
            extra_meta={
                "final_reward": final_reward,
                "base_reward": round(base_reward, 4),
                "efficiency_bonus": round(efficiency_bonus, 4),
                "hint_penalty": round(hint_penalty, 4),
                "auto_finalize_penalty": round(auto_penalty, 4),
                "voluntary_finalize": voluntary,
                "steps_used": steps_used,
                "predictions_submitted": len(self._submitted),
            },
        )

    @property
    def state(self) -> State:
        return self._state
