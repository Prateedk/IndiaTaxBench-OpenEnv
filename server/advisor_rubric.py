"""Rubric scoring for `submit_tax_advice` / `revise_tax_advice` episodes (next-year tax-saving advice)."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from server.tasks import advisor_keyphrases_for_task, advisor_task_difficulty

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


def _bad_phrase_hit(bad: str, comb: str) -> bool:
    """Match whole word / phrase; avoid e.g. 'cash' in 'cashflow' or 'cash-flow' as false positives."""
    if bad in ("without receipt",) or " " in bad:
        return bad in comb
    if bad == "donotdeclare":
        return bad in comb
    if bad == "cash":
        # "cash" as a standalone term, not a prefix of cashflow / cash-flow / cashback
        m = re.search(
            r"(?<![-a-z0-9])cash(?![-a-z0-9])",
            comb,
            re.IGNORECASE,
        )
        return m is not None
    return re.search(
        r"(?<![a-z0-9])" + re.escape(bad) + r"(?![a-z0-9])", comb, re.IGNORECASE
    ) is not None


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


def _count_good_actions(actions: Any) -> int:
    if not isinstance(actions, list):
        return 0
    n = 0
    for a in actions[:10]:
        if not isinstance(a, dict):
            continue
        if isinstance(a.get("action"), str) and len(a["action"].strip()) > 3:
            n += 1
    return n


def _summary_score(summary: Any, d: str) -> float:
    if not isinstance(summary, str):
        return 0.0
    n = len(summary.strip())
    if d == "hard":
        if 50 <= n <= 4000:
            return 0.25
        if 28 <= n < 50:
            return 0.1
        return 0.0
    if d == "medium":
        if 30 <= n <= 4000:
            return 0.25
        if 20 <= n < 30:
            return 0.1
        return 0.0
    if 20 <= n <= 4000:
        return 0.25
    if 10 <= n < 20:
        return 0.1
    return 0.0


def _actions_score(actions: Any, d: str) -> float:
    good = _count_good_actions(actions)
    if d == "hard":
        if good >= 3:
            return 0.3
        if good == 2:
            return 0.16
        if good == 1:
            return 0.08
        return 0.0
    if d == "medium":
        alen = len(actions) if isinstance(actions, list) else 0
        if good >= 2 or (good >= 1 and alen >= 3):
            return 0.3
        if good == 1:
            return 0.15
        return 0.0
    if good >= 2 or (good >= 1 and (isinstance(actions, list) and len(actions) >= 3)):
        return 0.3
    if good == 1:
        return 0.15
    return 0.0


def _cautions_score(cautions: Any, d: str) -> float:
    if isinstance(cautions, list):
        n = len([c for c in cautions if isinstance(c, str) and len(c.strip()) > 0])
        if d == "hard":
            if n >= 2:
                return 0.2
            if n == 1:
                return 0.1
            return 0.0
        if n >= 1:
            return 0.2
        return 0.0
    if isinstance(cautions, str) and len(cautions) > 2:
        if d == "hard":
            return 0.1
        return 0.1
    return 0.0


def _keyphrase_score(comb: str, phrases: tuple[str, ...], d: str) -> float:
    m = sum(1 for p in phrases if p in comb)
    if d == "hard":
        if m >= 3:
            return 0.25
        if m == 2:
            return 0.12
        if m == 1:
            return 0.04
        if len(comb) > 350:
            return 0.03
        return 0.0
    if d == "medium":
        if m >= 1:
            return 0.25
        if len(comb) > 200:
            return 0.1
        return 0.0
    if m >= 1:
        return 0.25
    if len(comb) > 200:
        return 0.1
    return 0.0


def score_advice_object(obj: Optional[Dict[str, Any]], task_id: str) -> float:
    """Return 0.0-1.0 quality score (before env-specific scaling)."""
    if not obj:
        return 0.0
    d = advisor_task_difficulty(task_id)
    comb = _flatten_lower(obj)
    parts: List[float] = []

    parts.append(_summary_score(obj.get("filing_profile_summary"), d))
    parts.append(_actions_score(obj.get("next_year_actions"), d))
    parts.append(_cautions_score(obj.get("cautions"), d))
    parts.append(_keyphrase_score(comb, advisor_keyphrases_for_task(task_id), d))

    base = min(1.0, sum(parts))
    for bad in _BAD_PHRASES:
        if _bad_phrase_hit(bad, comb):
            return max(0.0, base * 0.2)
    return max(0.0, min(1.0, base))
