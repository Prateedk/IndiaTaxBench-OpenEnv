# india_tax_capture

Capture **`taxcalcindia`** request/response pairs as **JSONL** for offline model training, in the same spirit as [`openfisca_capture`](../openfisca_capture/).

## Financial year mapping

Indian **FY 2024–25** (April 2024–March 2025), often discussed alongside **AY 2025–26**, is represented in `taxcalcindia` as:

```text
TaxSettings(..., financial_year=2025)
```

The library rejects `financial_year` values below **2025**. Do **not** use `2024` in scenario JSON.

## Old tax regime

`IncomeTaxCalculator.calculate_tax()` always computes **both** new and old regimes. Each JSONL row’s `response` contains the full engine output. For FY 2024–25 **old regime** training targets, use:

- `response["tax_liability"]["old_regime"]`
- `response["income_summary"]["old_regime_taxable_income"]`

when `response` is not an `_error` object.

## Install

```bash
cd india_tax_capture
uv pip install -e .
# or: pip install -e .
```

## Run

```bash
python capture_india_tax_dataset.py --manifest scenarios/manifest.json --out data/india_tax_rows.jsonl
```

Optional overrides (apply to every row):

```bash
python capture_india_tax_dataset.py --manifest scenarios/manifest.json --out data/out.jsonl \
  --no-is-comparision-needed --is-tax-per-slab-needed
```

Console entry point (after install):

```bash
capture-india-tax --manifest scenarios/manifest.json --out data/india_tax_rows.jsonl
```

## JSONL row shape

Each line is one JSON object:

| Field | Description |
|--------|-------------|
| `id` | Scenario id from the manifest |
| `mode` | Always `local` |
| `package_versions` | e.g. `{ "taxcalcindia": "0.1.4" }` |
| `request` | `scenario` (normalized) + `calculate_tax` flags |
| `response` | Full `calculate_tax` dict, or `{ "_error": "..." }` |

Generated `data/*.jsonl` files are gitignored.

## Scenarios

- [`scenarios/manifest.json`](scenarios/manifest.json) lists `id` and `scenario_file`.
- Each scenario JSON may include `tax_settings`, and optional `salary`, `business`, `capital_gains`, `other_income`, `deductions` objects whose keys match `taxcalcindia.models` constructors.
- `employment_type` must be one of: `private`, `government`, `self_employed`.

Per-item manifest overrides: optional `is_comparision_needed` / `is_tax_per_slab_needed` on each manifest entry override `defaults`.

## Disclaimer

This project runs a **third-party simulation library** for dataset generation only. It is **not** tax, legal, or filing advice. Verify results independently before any real-world use.

`taxcalcindia` is MIT-licensed; see [PyPI](https://pypi.org/project/taxcalcindia/).
