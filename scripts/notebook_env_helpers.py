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

ADVISOR_SYSTEM_PROMPT = """\
You are a compliant India income-tax planning assistant. The user provides a **FY 2024–25** \
(financial_year=2025) JSON scenario and a `task_id`. Your job is **not** to replace the tax engine, \
but to suggest **concrete, legal next-year and next–assessment-cycle tax-saving levers** this filer \
can consider (documentation, 80C/80D/NPS, HRA, business timing, capital-gains planning, record-keeping).

Return **only one JSON object** (no markdown) with exactly these keys:
- `filing_profile_summary` (string, 2–4 sentences, grounded in the scenario)
- `next_year_actions` (list of at least 2 objects, each with `action` and `rationale`; optional `indicative_section` like "80C")
- `cautions` (list of at least 1 string: limits, or when to seek a professional)

Do not suggest evasion, hiding income, or false deductions."""

REVISE_ADVISOR_PROMPT = """\
The environment gave feedback on your previous JSON advice. Improve the **same schema** (filing_profile_summary, next_year_actions, cautions) while staying legal and specific.

Scenario JSON:
{scenario}

Previous advice JSON:
{previous}

Environment feedback:
{feedback}

Return ONLY the improved single JSON object."""


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


def parse_advice_text(raw: str) -> Dict[str, Any]:
    """Parse model output into a dict; uses server rubric module when `server` is importable."""
    try:
        from server.advisor_rubric import parse_advice_json
    except ImportError:
        parse_advice_json = None  # type: ignore[assignment]
    if parse_advice_json is not None:
        out = parse_advice_json(raw)
        return out if out else {}
    d = _coerce_obj(raw)
    return d if d else {}


def build_advisor_messages(
    scenario_json: str,
    task_id: str,
    *,
    task_description: str = "",
) -> List[Dict[str, str]]:
    u = f"Task id: {task_id}\n"
    if task_description:
        u += f"Task focus: {task_description}\n"
    u += f"Scenario JSON:\n{scenario_json}\n"
    return [
        {"role": "system", "content": ADVISOR_SYSTEM_PROMPT},
        {"role": "user", "content": u},
    ]


def build_revise_advisor_messages(
    scenario_json: str, previous: str, feedback: str
) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": ADVISOR_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": REVISE_ADVISOR_PROMPT.format(
                scenario=scenario_json, previous=previous, feedback=feedback
            ),
        },
    ]


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
