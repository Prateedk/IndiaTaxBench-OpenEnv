"""Data models for IndiaTaxBench (FY 2024–25 old regime tax prediction)."""

from typing import Any, Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field

TaskDifficultyLabel = Literal["easy", "medium", "hard"]


class IndiaTaxBenchReward(BaseModel):
    """Scalar reward in [0.0, 1.0] aligned with OpenEnv grading rubrics."""

    value: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Per-step or terminal reward after clamping.",
    )


class IndiaTaxBenchAction(Action):
    """Typed action for multi-step old-regime tax prediction."""

    action_type: str = Field(
        ...,
        description=(
            "One of: submit_prediction, revise_prediction, finalize, request_context; "
            "or advisor: submit_tax_advice, revise_tax_advice, finalize_advice, request_context"
        ),
    )
    item_index: Optional[int] = Field(
        default=None,
        description="Index of a previously submitted prediction (revise_prediction only)",
    )
    predicted_total: Optional[float] = Field(
        default=None,
        description="Predicted old-regime total tax (INR)",
    )
    predicted_initial_tax: Optional[float] = Field(
        default=None,
        description="Predicted initial tax before surcharge/cess (INR)",
    )
    predicted_surcharge: Optional[float] = Field(
        default=None,
        description="Predicted surcharge (INR)",
    )
    predicted_cess: Optional[float] = Field(
        default=None,
        description="Predicted cess (INR)",
    )
    advice_text: Optional[str] = Field(
        default=None,
        description="JSON string of next-year tax-saving advice (advisor episode only)",
    )


class IndiaTaxBenchObservation(Observation):
    """Observation for India income tax prediction episodes."""

    scenario_json: str = Field(
        default="",
        description="JSON string of the public scenario (no oracle labels)",
    )
    task_id: str = Field(default="", description="Task id from capture JSONL")
    task_description: str = Field(
        default="",
        description="Human-readable task goal",
    )
    task_difficulty: TaskDifficultyLabel = Field(
        default="medium",
        description="Curriculum label for the episode (easy / medium / hard).",
    )
    feedback: str = Field(default="", description="Per-step grading feedback")
    submitted_predictions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Predictions so far with scores and field breakdowns",
    )
    steps_remaining: int = Field(default=0, description="Steps left before auto-finalize")
    hints_used: int = Field(default=0, description="Hints consumed")
    valid_actions: List[str] = Field(
        default_factory=list,
        description="Legal action_type values at this step",
    )
    submitted_advice: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Advisor-mode submissions with rubric scores",
    )
    episode_mode: str = Field(
        default="numeric",
        description="numeric: liability prediction; advisor: next-year tax-saving advice",
    )
