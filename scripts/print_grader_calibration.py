#!/usr/bin/env python3
"""Deterministic oracle scores for README baseline tables.

Run from repo root:
  uv run python scripts/print_grader_calibration.py
  uv run python scripts/print_grader_calibration.py --markdown
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import IndiaTaxBenchAction  # noqa: E402
from server.india_tax_bench_environment import IndiaTaxBenchEnvironment  # noqa: E402
from server.tasks import TASKS  # noqa: E402


def _sorted_task_ids() -> list[str]:
    return sorted(TASKS.keys())


def _oracle_finalize_reward(task_id: str) -> tuple[float, float]:
    """Returns (base_reward, finalize_reward) for submitting exact oracle then finalize."""
    o = TASKS[task_id]["oracle"]
    env = IndiaTaxBenchEnvironment()
    env.reset(task=task_id)
    env.step(
        IndiaTaxBenchAction(
            action_type="submit_prediction",
            predicted_total=o["total"],
            predicted_initial_tax=o["initial_tax"],
            predicted_surcharge=o["surcharge"],
            predicted_cess=o["cess"],
        )
    )
    obs = env.step(IndiaTaxBenchAction(action_type="finalize"))
    base = float(obs.metadata.get("base_reward", 0.0))
    final = float(obs.reward or 0.0)
    return base, final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Print Markdown tables for pasting into README",
    )
    args = parser.parse_args()

    if not TASKS:
        print("No tasks loaded (empty india_tax_capture/data/india_tax_rows.jsonl).")
        return

    if not args.markdown:
        print("task_id | difficulty | oracle_base_reward | oracle_finalize_reward")
        print("--------|------------|--------------------|------------------------")
        for tid in _sorted_task_ids()[:3]:
            b, f = _oracle_finalize_reward(tid)
            d = TASKS[tid].get("difficulty", "?")
            print(f"{tid} | {d} | {b:.4f} | {f:.4f}")
        return

    print("### Oracle path (exact old-regime components from JSONL)")
    print()
    print(
        "Procedure: `reset(task)`, `submit_prediction` with oracle totals/components, "
        "then voluntary `finalize`. `oracle_base_reward` is `metadata.base_reward` on finalize; "
        "`oracle_finalize_reward` is the step reward (includes efficiency and penalties)."
    )
    print()
    print("| task_id | difficulty | oracle_base_reward | oracle_finalize_reward |")
    print("|---------|------------|--------------------|-------------------------|")
    for tid in _sorted_task_ids():
        b, f = _oracle_finalize_reward(tid)
        d = TASKS[tid].get("difficulty", "?")
        print(f"| `{tid}` | {d} | {b:.4f} | {f:.4f} |")


if __name__ == "__main__":
    main()
