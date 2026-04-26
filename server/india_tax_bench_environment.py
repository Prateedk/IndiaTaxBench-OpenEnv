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
    from .advisor_rubric import parse_advice_json, score_advice_object
    from .tasks import ALL_TASK_IDS, TASKS
except ImportError:
    from server.advisor_rubric import parse_advice_json, score_advice_object
    from server.tasks import ALL_TASK_IDS, TASKS

REWARD_MIN = 0.01
REWARD_MAX = 0.99
MAX_STEPS = 15
MAX_HINTS = 3
HINT_PENALTY = 0.03
STEP_REWARD_SCALE = 0.10
EFFICIENCY_BONUS_SCALE = 0.05
AUTO_FINALIZE_PENALTY = 0.05

NUMERIC_ACTION_TYPES = (
    "submit_prediction",
    "revise_prediction",
    "request_context",
    "finalize",
)
ADVISOR_ACTION_TYPES = (
    "submit_tax_advice",
    "revise_tax_advice",
    "request_context",
    "finalize_advice",
)
ALL_ACTION_TYPE_NAMES = set(NUMERIC_ACTION_TYPES) | set(ADVISOR_ACTION_TYPES)
ACTION_TYPES = tuple(sorted(ALL_ACTION_TYPE_NAMES))

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


def _best_advice_rubric(submitted: List[Dict[str, Any]]) -> Tuple[float, Dict[str, Any]]:
    if not submitted:
        return 0.0, {}
    best = max(submitted, key=lambda x: float(x.get("rubric", 0.0)))
    return float(best.get("rubric", 0.0)), best


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
    """Multi-step prediction of old-regime tax components from JSON scenarios.

    NOTE: OpenEnv's REST endpoints recreate the env on every request, so true
    multi-session concurrency requires the MCP transport. For REST-based
    multi-step episodes we publish this env as a process singleton in
    ``server/app.py``; concurrent users then share state, which is acceptable
    for the hackathon-style benchmark.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = False

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._current_task_id: str = ""
        self._done: bool = False
        self._submitted: List[Dict[str, Any]] = []
        self._advisor_mode: bool = False
        self._advice_submissions: List[Dict[str, Any]] = []
        self._hints_used: int = 0
        self._cumulative_reward: float = 0.0

    def _valid_actions(self) -> List[str]:
        if self._done:
            return []
        if self._advisor_mode:
            out = ["submit_tax_advice", "finalize_advice"]
            if self._advice_submissions:
                out.append("revise_tax_advice")
            if self._hints_used < MAX_HINTS:
                out.append("request_context")
            return sorted(out)
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

        desc = str(t.get("description", ""))
        if self._advisor_mode:
            desc = str(t.get("description_advisor", t.get("description", "")))
        return IndiaTaxBenchObservation(
            scenario_json=json.dumps(scenario, ensure_ascii=False, indent=2),
            task_id=self._current_task_id,
            task_description=desc,
            task_difficulty=diff,
            feedback=feedback,
            submitted_predictions=list(self._submitted),
            submitted_advice=list(self._advice_submissions),
            episode_mode="advisor" if self._advisor_mode else "numeric",
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
        advisor: bool = False,
        **kwargs: Any,
    ) -> IndiaTaxBenchObservation:
        if not ALL_TASK_IDS:
            raise RuntimeError("No tasks loaded; check india_tax_capture/data/india_tax_rows.jsonl")

        if "advisor" in kwargs:
            self._advisor_mode = bool(kwargs["advisor"])
        else:
            self._advisor_mode = bool(advisor)

        if task and task in TASKS:
            self._current_task_id = task
        else:
            self._current_task_id = random.choice(ALL_TASK_IDS)

        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._done = False
        self._submitted = []
        self._advice_submissions = []
        self._hints_used = 0
        self._cumulative_reward = 0.0

        if self._advisor_mode:
            fb = (
                "Propose **next-year (upcoming FY / next assessment) tax-saving steps** for this "
                "filer as JSON. Required keys: `filing_profile_summary` (string), `next_year_actions` "
                "(list of {action, rationale, indicative_section?}), `cautions` (list of strings). "
                "Use submit_tax_advice with `advice_text` set to one JSON object (string). "
                "Optionally request_context, revise_tax_advice, then finalize_advice."
            )
        else:
            fb = (
                "Predict old-regime `total`, `initial_tax`, `surcharge`, `cess` (INR). "
                "Use submit_prediction, optionally revise_prediction or request_context, then finalize."
            )
        return self._make_obs(feedback=fb, reward=REWARD_MIN, done=False)

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
        allowed = ADVISOR_ACTION_TYPES if self._advisor_mode else NUMERIC_ACTION_TYPES
        if at not in allowed:
            return self._make_obs(
                feedback=(
                    f"Invalid action for this episode mode. Got '{at}'. "
                    f"Valid: {', '.join(sorted(allowed))}"
                ),
                reward=_clamp_reward(self._cumulative_reward * 0.01),
                done=False,
                extra_meta={"error": f"invalid action for mode: {at}"},
            )

        if self._advisor_mode:
            if at == "submit_tax_advice":
                return self._handle_advice_submit(action)
            if at == "revise_tax_advice":
                return self._handle_advice_revise(action)
            if at == "request_context":
                return self._handle_hint()
            if at == "finalize_advice":
                return self._handle_advice_finalize(voluntary=True)
            return self._make_obs(feedback="Unhandled advisor action.", reward=REWARD_MIN, done=False)

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
            if self._advisor_mode:
                return self._handle_advice_finalize(voluntary=False)
            return self._handle_finalize(self._oracle(), voluntary=False)

        return self._make_obs(
            feedback=f"Hint ({self._hints_used}/{MAX_HINTS}): {hint}",
            reward=_clamp_reward(-HINT_PENALTY),
            done=False,
            extra_meta={"hint": hint},
        )

    def _handle_advice_submit(self, action: IndiaTaxBenchAction) -> IndiaTaxBenchObservation:
        raw = action.advice_text or ""
        obj = parse_advice_json(raw)
        rubric = score_advice_object(obj, self._current_task_id) if obj else 0.0
        entry = {
            "rubric": round(rubric, 4),
            "raw": raw[:12000],
            "parsed_ok": obj is not None,
        }
        self._advice_submissions.append(entry)
        step_reward = rubric * STEP_REWARD_SCALE
        self._cumulative_reward += step_reward
        feedback = (
            f"Submitted advice {len(self._advice_submissions)}: rubric={rubric:.4f} "
            f"(structured JSON={'yes' if obj else 'no'})"
        )
        if self._state.step_count >= MAX_STEPS:
            return self._handle_advice_finalize(voluntary=False)
        return self._make_obs(
            feedback=feedback,
            reward=_clamp_reward(step_reward),
            done=False,
            extra_meta={"advice_rubric": rubric, "advice_index": len(self._advice_submissions) - 1},
        )

    def _handle_advice_revise(self, action: IndiaTaxBenchAction) -> IndiaTaxBenchObservation:
        idx = action.item_index if action.item_index is not None else len(self._advice_submissions) - 1
        if idx is None or idx < 0 or idx >= len(self._advice_submissions):
            return self._make_obs(
                feedback=f"Invalid item_index={idx} for advice list (0..{len(self._advice_submissions)-1}).",
                reward=_clamp_reward(self._cumulative_reward * 0.01),
                done=False,
                extra_meta={"error": "invalid item_index"},
            )
        old = self._advice_submissions[idx]
        old_r = float(old.get("rubric", 0.0))
        raw = action.advice_text or ""
        obj = parse_advice_json(raw)
        rubric = score_advice_object(obj, self._current_task_id) if obj else 0.0
        new_entry = {
            "rubric": round(rubric, 4),
            "raw": raw[:12000],
            "parsed_ok": obj is not None,
        }
        self._advice_submissions[idx] = new_entry
        improvement = rubric - old_r
        step_reward = max(0.0, improvement) * STEP_REWARD_SCALE
        self._cumulative_reward += step_reward
        feedback = f"Revised advice {idx}: rubric {old_r:.4f} → {rubric:.4f} (delta={improvement:+.4f})"
        if self._state.step_count >= MAX_STEPS:
            return self._handle_advice_finalize(voluntary=False)
        return self._make_obs(
            feedback=feedback,
            reward=_clamp_reward(step_reward),
            done=False,
            extra_meta={"advice_revision_delta": improvement, "item_index": idx},
        )

    def _handle_advice_finalize(self, *, voluntary: bool) -> IndiaTaxBenchObservation:
        base_r, best = _best_advice_rubric(self._advice_submissions)
        if not self._advice_submissions:
            fb = "No tax advice submitted before finalize."
        else:
            fb = f"Best advice rubric={base_r:.4f} (submissions={len(self._advice_submissions)})"
        steps_used = self._state.step_count
        efficiency_bonus = EFFICIENCY_BONUS_SCALE * (MAX_STEPS - steps_used) / MAX_STEPS
        hint_penalty = HINT_PENALTY * self._hints_used
        auto_penalty = 0.0 if voluntary else AUTO_FINALIZE_PENALTY
        final_reward = base_r + efficiency_bonus - hint_penalty - auto_penalty
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
                "advisor_rubric": round(base_r, 4),
                "best_submission": best,
                "efficiency_bonus": round(efficiency_bonus, 4),
                "hint_penalty": round(hint_penalty, 4),
                "auto_finalize_penalty": round(auto_penalty, 4),
                "voluntary_finalize": voluntary,
                "steps_used": steps_used,
                "advice_submissions": len(self._advice_submissions),
            },
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
