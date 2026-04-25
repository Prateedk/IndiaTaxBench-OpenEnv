"""Pure helpers for IndiaTaxBench training notebook (no OpenAI client at import time)."""

from __future__ import annotations

import ast
import json
import re
from typing import Any, Dict, List

PREDICT_PROMPT = """\
You are an India income tax assistant. The user gives a JSON scenario for FY 2024–25 \
(Assessment Year style: financial_year=2025 in the JSON). \
Predict **old tax regime** liability with exactly these numeric keys (INR, numbers only):
- total
- initial_tax
- surcharge
- cess

Return ONLY one JSON object with those four keys. No markdown, no explanation."""

REVISE_PROMPT = """\
Your previous numeric prediction scored poorly against internal checks. \
Revise all four fields using the same JSON schema.

Scenario JSON:
{scenario}

Previous prediction JSON:
{previous}

Feedback:
{feedback}

Return ONLY one JSON object with keys: total, initial_tax, surcharge, cess (numbers)."""


def unwrap_observation(payload: Dict[str, Any]) -> Dict[str, Any]:
    return payload.get("observation", payload)


def step_reward_done(step_payload: Dict[str, Any]) -> tuple[float, bool, Dict[str, Any]]:
    obs = unwrap_observation(step_payload)
    r = float(step_payload.get("reward", obs.get("reward", 0.0)) or 0.0)
    d = bool(step_payload.get("done", obs.get("done", False)))
    return r, d, obs


def clean_json(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        raw = "\n".join(lines).strip()
    return raw


_BRACE_RE = re.compile(r"\{.*\}", re.DOTALL)


def _coerce_obj(raw: str) -> Dict[str, Any]:
    """Parse the model's JSON-ish output into a dict.

    Tries (in order):
      1. strict json.loads
      2. regex first {...} block + json.loads
      3. ast.literal_eval (handles single quotes, trailing commas, Python dicts)
    Returns {} when all fail; never raises.
    """
    candidates: List[str] = []
    cleaned = clean_json(raw)
    if cleaned:
        candidates.append(cleaned)
    m = _BRACE_RE.search(cleaned or raw or "")
    if m:
        candidates.append(m.group(0))

    for cand in candidates:
        try:
            data = json.loads(cand)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
        try:
            data = ast.literal_eval(cand)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {}


def parse_prediction(raw: str) -> Dict[str, float]:
    data = _coerce_obj(raw)
    out: Dict[str, float] = {}
    for k in ("total", "initial_tax", "surcharge", "cess"):
        v = data.get(k)
        if v is None:
            out[k] = 0.0
            continue
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            out[k] = 0.0
    return out


def build_predict_messages(scenario_json: str) -> List[Dict[str, str]]:
    user_msg = f"Scenario JSON:\n{scenario_json}\n"
    return [
        {"role": "system", "content": PREDICT_PROMPT},
        {"role": "user", "content": user_msg},
    ]


def build_revise_messages(scenario_json: str, previous_json: str, feedback: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": "You are a precise tax numeric estimator."},
        {
            "role": "user",
            "content": REVISE_PROMPT.format(
                scenario=scenario_json,
                previous=previous_json,
                feedback=feedback,
            ),
        },
    ]
