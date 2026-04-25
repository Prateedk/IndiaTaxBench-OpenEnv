"""Task definitions loaded from india_tax_capture JSONL capture rows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

_REPO_ROOT = Path(__file__).resolve().parents[1]
_JSONL_PATH = _REPO_ROOT / "india_tax_capture" / "data" / "india_tax_rows.jsonl"


def _load_tasks() -> Dict[str, Dict[str, Any]]:
    tasks: Dict[str, Dict[str, Any]] = {}
    if not _JSONL_PATH.is_file():
        return tasks
    for line in _JSONL_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        tid = row.get("id")
        if not tid:
            continue
        resp = row.get("response")
        if not isinstance(resp, dict) or resp.get("_error"):
            continue
        old = resp.get("tax_liability", {}).get("old_regime")
        if not isinstance(old, dict):
            continue
        comp = old.get("components") or {}
        try:
            oracle = {
                "total": float(old["total"]),
                "initial_tax": float(comp["initial_tax"]),
                "surcharge": float(comp["surcharge"]),
                "cess": float(comp["cess"]),
            }
        except (KeyError, TypeError, ValueError):
            continue
        req = row.get("request") or {}
        scenario = req.get("scenario") if isinstance(req, dict) else {}
        tasks[str(tid)] = {
            "description": (
                "Predict India FY 2024–25 **old tax regime** liability: "
                "`total`, `initial_tax`, `surcharge`, `cess` (INR), "
                "matching the reference calculator output for this scenario."
            ),
            "scenario": scenario if isinstance(scenario, dict) else {},
            "oracle": oracle,
        }
    return tasks


TASKS: Dict[str, Dict[str, Any]] = _load_tasks()
ALL_TASK_IDS = list(TASKS.keys())
