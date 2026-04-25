"""Data models for IndiaTaxBench (FY 2024–25 old regime tax prediction)."""

from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class IndiaTaxBenchAction(Action):
    """Typed action for multi-step old-regime tax prediction."""

    action_type: str = Field(
        ...,
        description=(
            "One of: submit_prediction, revise_prediction, finalize, request_context"
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
