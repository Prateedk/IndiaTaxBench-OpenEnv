#!/usr/bin/env python3
"""Capture taxcalcindia request/response pairs for offline training (local engine only).

This targets **India FY 2024–25 (old regime)** using the ``taxcalcindia`` library, which
represents that year as ``TaxSettings(financial_year=2025, ...)`` (AY 2025–26 style
labeling). See README for the mapping.

Output JSONL rows (one JSON object per line)
--------------------------------------------
Each row includes ``id``, ``mode`` (``local``), ``package_versions``, ``request``
(normalized scenario + ``calculate_tax`` flags), and ``response`` (full engine dict
from ``IncomeTaxCalculator.calculate_tax``, or ``{"_error": "..."}`` on failure).

Examples
--------
  uv pip install -e .
  python capture_india_tax_dataset.py --manifest scenarios/manifest.json --out data/india_tax_rows.jsonl

  capture-india-tax --manifest scenarios/manifest.json --out data/india_tax_rows.jsonl
"""

from __future__ import annotations

import argparse
import json
from enum import Enum
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, List, Optional

from taxcalcindia.calculator import IncomeTaxCalculator
from taxcalcindia.exceptions import TaxCalculationException
from taxcalcindia.models import (
    BusinessIncome,
    CapitalGainsIncome,
    Deductions,
    EmploymentType,
    OtherIncome,
    SalaryIncome,
    TaxSettings,
)


def _pkg_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unknown"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_path(manifest_path: Path, rel: str) -> Path:
    p = (manifest_path.parent / rel).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Missing file: {p}")
    return p


def _employment_type(raw: str) -> EmploymentType:
    s = str(raw).strip().lower()
    for m in EmploymentType:
        if m.value == s:
            return m
    raise ValueError(f"Unknown employment_type {raw!r}; expected one of {[e.value for e in EmploymentType]}")


def _build_tax_settings(data: Dict[str, Any]) -> TaxSettings:
    ts = dict(data)
    et = ts.pop("employment_type", EmploymentType.PRIVATE)
    if isinstance(et, str):
        et = _employment_type(et)
    if not isinstance(et, EmploymentType):
        raise TypeError("employment_type must be a string or EmploymentType")
    return TaxSettings(employment_type=et, **ts)


def _build_salary(data: Optional[Dict[str, Any]]) -> Optional[SalaryIncome]:
    if not data:
        return None
    return SalaryIncome(**data)


def _build_business(data: Optional[Dict[str, Any]]) -> Optional[BusinessIncome]:
    if not data:
        return None
    return BusinessIncome(**data)


def _build_capital_gains(data: Optional[Dict[str, Any]]) -> Optional[CapitalGainsIncome]:
    if not data:
        return None
    return CapitalGainsIncome(**data)


def _build_other_income(data: Optional[Dict[str, Any]]) -> Optional[OtherIncome]:
    if not data:
        return None
    return OtherIncome(**data)


def _build_deductions(data: Optional[Dict[str, Any]]) -> Optional[Deductions]:
    if not data:
        return None
    d = dict(data)
    tta = d.pop("section_80tta", None)
    ttb = d.pop("section_80ttb", None)
    obj = Deductions(**d)
    if tta is not None:
        obj.section_80tta = tta
    if ttb is not None:
        obj.section_80ttb = ttb
    return obj


def _normalize_request_scenario(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Return a JSON-serializable copy of the scenario (strings for enums)."""
    out: Dict[str, Any] = {}
    for key in ("tax_settings", "salary", "business", "capital_gains", "other_income", "deductions"):
        if key not in raw:
            continue
        val = raw[key]
        if val is None:
            continue
        if key == "tax_settings" and isinstance(val, dict):
            ts = dict(val)
            et = ts.get("employment_type")
            if isinstance(et, Enum):
                ts["employment_type"] = et.value
            out[key] = ts
        else:
            out[key] = val
    return out


def _jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, "item"):
        try:
            return _jsonable(obj.item())
        except Exception:
            pass
    if isinstance(obj, (bytes, bytearray)):
        return obj.decode("utf-8", errors="replace")
    return str(obj)


def _run_scenario(
    scenario: Dict[str, Any],
    is_comparision_needed: bool,
    is_tax_per_slab_needed: bool,
) -> Dict[str, Any]:
    if "tax_settings" not in scenario:
        raise ValueError("scenario must include 'tax_settings'")

    settings = _build_tax_settings(scenario["tax_settings"])
    salary = _build_salary(scenario.get("salary"))
    business = _build_business(scenario.get("business"))
    capital_gains = _build_capital_gains(scenario.get("capital_gains"))
    other_income = _build_other_income(scenario.get("other_income"))
    deductions = _build_deductions(scenario.get("deductions"))

    calc = IncomeTaxCalculator(
        settings,
        salary=salary,
        capital_gains=capital_gains,
        business=business,
        other_income=other_income,
        deductions=deductions,
    )
    result = calc.calculate_tax(
        is_comparision_needed=is_comparision_needed,
        is_tax_per_slab_needed=is_tax_per_slab_needed,
        display_result=False,
    )
    return result  # type: ignore[return-value]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("scenarios/manifest.json"),
        help="Path to manifest (see scenarios/manifest.json)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/india_tax_rows.jsonl"),
        help="Output JSONL path",
    )
    parser.add_argument(
        "--is-comparision-needed",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override manifest: include tax_regime_comparison in response (default: from manifest)",
    )
    parser.add_argument(
        "--is-tax-per-slab-needed",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override manifest: include tax_per_slabs in response (default: from manifest)",
    )
    args = parser.parse_args(argv)

    manifest_path = args.manifest.resolve()
    manifest = _load_json(manifest_path)
    defaults: Dict[str, Any] = manifest.get("defaults", {})

    def eff_comparison(item: Dict[str, Any]) -> bool:
        if args.is_comparision_needed is not None:
            return bool(args.is_comparision_needed)
        if "is_comparision_needed" in item:
            return bool(item["is_comparision_needed"])
        return bool(defaults.get("is_comparision_needed", True))

    def eff_per_slab(item: Dict[str, Any]) -> bool:
        if args.is_tax_per_slab_needed is not None:
            return bool(args.is_tax_per_slab_needed)
        if "is_tax_per_slab_needed" in item:
            return bool(item["is_tax_per_slab_needed"])
        return bool(defaults.get("is_tax_per_slab_needed", False))

    versions = {"taxcalcindia": _pkg_version("taxcalcindia")}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0

    with args.out.open("w", encoding="utf-8") as sink:
        for item in manifest.get("items", []):
            sid = item["id"]
            scenario_path = _resolve_path(manifest_path, item["scenario_file"])
            scenario_raw = _load_json(scenario_path)
            if not isinstance(scenario_raw, dict):
                raise TypeError(f"Scenario {scenario_path} must be a JSON object")

            cmp_flag = eff_comparison(item)
            slab_flag = eff_per_slab(item)

            request: Dict[str, Any] = {
                "scenario": _normalize_request_scenario(scenario_raw),
                "calculate_tax": {
                    "is_comparision_needed": cmp_flag,
                    "is_tax_per_slab_needed": slab_flag,
                },
            }

            try:
                response = _run_scenario(scenario_raw, cmp_flag, slab_flag)
                response = _jsonable(response)
            except (TaxCalculationException, ValueError, TypeError) as exc:
                response = {"_error": str(exc)}
            except Exception as exc:  # noqa: BLE001 — surface engine errors in dataset
                response = {"_error": str(exc)}

            row = {
                "id": sid,
                "mode": "local",
                "package_versions": versions,
                "request": request,
                "response": response,
            }
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows_written += 1

    print(f"Wrote {rows_written} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
