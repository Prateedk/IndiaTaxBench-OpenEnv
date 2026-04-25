"""IndiaTaxBench environment client."""

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import IndiaTaxBenchAction, IndiaTaxBenchObservation
except ImportError:
    from models import IndiaTaxBenchAction, IndiaTaxBenchObservation


class IndiaTaxBenchEnvClient(
    EnvClient[IndiaTaxBenchAction, IndiaTaxBenchObservation, State]
):
    """HTTP client for IndiaTaxBench."""

    def _step_payload(self, action: IndiaTaxBenchAction) -> Dict[str, Any]:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[IndiaTaxBenchObservation]:
        obs_data = payload.get("observation", {})
        td_raw = obs_data.get("task_difficulty", "medium")
        if td_raw not in ("easy", "medium", "hard"):
            td_raw = "medium"

        observation = IndiaTaxBenchObservation(
            scenario_json=obs_data.get("scenario_json", ""),
            task_id=obs_data.get("task_id", ""),
            task_description=obs_data.get("task_description", ""),
            task_difficulty=td_raw,
            feedback=obs_data.get("feedback", ""),
            submitted_predictions=obs_data.get("submitted_predictions", []),
            steps_remaining=obs_data.get("steps_remaining", 0),
            hints_used=obs_data.get("hints_used", 0),
            valid_actions=obs_data.get("valid_actions", []),
            done=payload.get("done", obs_data.get("done", False)),
            reward=payload.get("reward", obs_data.get("reward")),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
