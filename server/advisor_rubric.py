"""Rubric scoring for `submit_tax_advice` / `revise_tax_advice` episodes (next-year tax-saving advice)."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from server.tasks import advisor_keyphrases_for_task

# Penalize if the model output suggests non-compliance.
_BAD_PHRASES: tuple[str, ...] = (
    "donotdeclare",
    "unreported",
    "cash",
    "without receipt",
    "bribe",
    "evasion",
    "fraud",
)


def _loads_advice_object(text: str) -> Any:
    """Parse JSON; on failure, take the first top-level {...} block (tolerant of markdown noise)."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            raise
        return json.loads(m.group(0))


def parse_advice_json(raw: Optional[str]) -> Optional[Dict[str, Any]]:
    if not raw or not str(raw).strip():
        return None
    text = str(raw).strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(
            ln for ln in lines if not ln.strip().startswith("```")
        ).strip()
    # Unwrap double-encoded / string-wrapped JSON: load until we get a dict
    # (e.g. "\"{...}\"" or a model returning json.dumps(s) of an object string).
    obj: Any
    for _ in range(4):
        if not text:
            return None
        try:
            obj = _loads_advice_object(text)
        except json.JSONDecodeError:
            return None
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, str):
            text = obj.strip()
            continue
        return None
    return None


def _flatten_lower(obj: Any) -> str:
    if isinstance(obj, str):
        return obj.lower()
    if isinstance(obj, dict):
        return " ".join(_flatten_lower(v) for v in obj.values())
    if isinstance(obj, list):
        return " ".join(_flatten_lower(x) for x in obj)
    return str(obj).lower()


def score_advice_object(obj: Optional[Dict[str, Any]], task_id: str) -> float:
    """Return 0.0-1.0 quality score (before env-specific scaling)."""
    if not obj:
        return 0.0
    parts: List[float] = []

    summary = obj.get("filing_profile_summary")
    if isinstance(summary, str) and 20 <= len(summary) <= 4000:
        parts.append(0.25)
    else:
        parts.append(0.0)

    actions = obj.get("next_year_actions")
    if isinstance(actions, list) and len(actions) >= 1:
        good = 0
        for a in actions[:10]:
            if not isinstance(a, dict):
                continue
            if isinstance(a.get("action"), str) and len(a["action"]) > 3:
                good += 1
        if good >= 2 or (good >= 1 and len(actions) >= 3):
            parts.append(0.3)
        elif good == 1:
            parts.append(0.15)
        else:
            parts.append(0.0)
    else:
        parts.append(0.0)

    cautions = obj.get("cautions")
    if isinstance(cautions, list) and len(cautions) >= 1:
        parts.append(0.2)
    elif isinstance(cautions, str) and len(cautions) > 2:
        parts.append(0.1)
    else:
        parts.append(0.0)

    comb = _flatten_lower(obj)
    phrases = advisor_keyphrases_for_task(task_id)
    if any(p in comb for p in phrases):
        parts.append(0.25)
    elif len(comb) > 200:
        parts.append(0.1)
    else:
        parts.append(0.0)

    base = min(1.0, sum(parts))
    for bad in _BAD_PHRASES:
        if bad in comb:
            return max(0.0, base * 0.2)
    return max(0.0, min(1.0, base))
