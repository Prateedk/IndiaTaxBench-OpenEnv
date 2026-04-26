"""Task definitions loaded from india_tax_capture JSONL capture rows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Literal

_REPO_ROOT = Path(__file__).resolve().parents[1]
_JSONL_PATH = _REPO_ROOT / "india_tax_capture" / "data" / "india_tax_rows.jsonl"

TaskDifficulty = Literal["easy", "medium", "hard"]

# Curated label for the UI; for advisor episodes this maps to stricter rubric
# (more keyphrases, longer summary, more actions) so untrained baselines are not at ceiling.
_TASK_DIFFICULTY: Dict[str, TaskDifficulty] = {
    "senior_salary_deductions_fy2425": "hard",
    "salary_non_metro_hra_fy2425": "hard",
    "business_only_fy2425": "hard",
    "salary_metro_80c_fy2425": "hard",
    "salary_capital_gains_fy2425": "hard",
    "salary_business_mixed_fy2425": "hard",
    "senior_80c_80tta_deductions_fy2425": "hard",
    "stcgt_ltcg_weighted_fy2425": "hard",
}

# At least one phrase (lowercased match) should appear in advisor text for a full rubric bonus.
_TASK_ADVISOR_KEYPHRASES: Dict[str, tuple[str, ...]] = {
    "senior_salary_deductions_fy2425": (
        "senior",
        "80d",
        "80tt",
        "deduction",
    ),
    "salary_non_metro_hra_fy2425": (
        "hra",
        "rent",
        "exemption",
        "80gg",
    ),
    "business_only_fy2425": (
        "business",
        "40",
        "expense",
    ),
    "salary_metro_80c_fy2425": (
        "80c",
        "chapter vi",
        "deduction",
        "nps",
    ),
    "salary_capital_gains_fy2425": (
        "capital gain",
        "stcg",
        "ltcg",
        "loss",
    ),
    "salary_business_mixed_fy2425": (
        "business",
        "salary",
        "income",
        "advance tax",
    ),
    "senior_80c_80tta_deductions_fy2425": (
        "80tta",
        "80c",
        "interest",
        "senior",
    ),
    "stcgt_ltcg_weighted_fy2425": (
        "stcg",
        "ltcg",
        "holding",
        "exemption",
    ),
}


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
        diff: TaskDifficulty = _TASK_DIFFICULTY.get(str(tid), "hard")
        tasks[str(tid)] = {
            "description": (
                "Predict India FY 2024–25 **old tax regime** liability: "
                "`total`, `initial_tax`, `surcharge`, `cess` (INR), "
                "matching the reference calculator output for this scenario."
            ),
            "description_advisor": (
                "Given this FY 2024–25 (financial_year=2025) old-regime **scenario** for task "
                f"`{tid}`, list practical **next-year (next FY / next assessment) tax-saving steps** a filer can take—"
                "compliant, realistic, and tied to this income and deduction mix (e.g. 80C, NPS, HRA, health insurance, record-keeping, business expenses, capital-gains planning)."
            ),
            "scenario": scenario if isinstance(scenario, dict) else {},
            "oracle": oracle,
            "difficulty": diff,
        }
    return tasks


TASKS: Dict[str, Dict[str, Any]] = _load_tasks()
ALL_TASK_IDS = list(TASKS.keys())


def advisor_task_difficulty(task_id: str) -> TaskDifficulty:
    """Difficulty used by advisor rubric; defaults to hard (strictest) for unknown task ids."""
    return _TASK_DIFFICULTY.get(str(task_id), "hard")


def advisor_keyphrases_for_task(task_id: str) -> tuple[str, ...]:
    """Key phrases that advisor-style answers should ground in for this task."""
    return _TASK_ADVISOR_KEYPHRASES.get(
        str(task_id),
        ("deduction", "income", "assessment", "compliance"),
    )
