---
title: IndiaTaxBench OpenEnv
emoji: 💰
colorFrom: yellow
colorTo: red
sdk: docker
pinned: false
app_port: 8000
tags:
  - openenv
---

# IndiaTaxBench — India income tax (FY 2024–25, old regime)

OpenEnv environment where the agent predicts **old tax regime** liability components from a JSON scenario (oracle from [`taxcalcindia`](https://pypi.org/project/taxcalcindia/) capture data).

**Motivation:** Tax estimation from structured inputs is a real analyst-style workflow (not a game). The environment exposes multi-step actions (submit, revise, optional hints, finalize) with dense feedback so agents can learn from partial numeric match before the episode ends.

## Task source

Tasks are loaded from vendored **[`india_tax_capture/data/india_tax_rows.jsonl`](india_tax_capture/data/india_tax_rows.jsonl)** (one episode per line). See [`india_tax_capture/README.md`](india_tax_capture/README.md) for FY mapping (`financial_year=2025` = FY 2024–25).

Each row defines a deterministic **oracle** (`total`, `initial_tax`, `surcharge`, `cess`) and a curriculum **difficulty** label (scenario complexity).

| Task id | Difficulty | Notes |
|---------|--------------|--------|
| `senior_salary_deductions_fy2425` | easy | Senior slab + common deductions; lower liability band. |
| `salary_non_metro_hra_fy2425` | medium | Salary + HRA (non-metro). |
| `business_only_fy2425` | medium | Self-employed mix (business + property). |
| `salary_metro_80c_fy2425` | hard | Metro HRA, 80C/80D caps, higher income. |
| `salary_capital_gains_fy2425` | hard | Salary plus multiple capital-gains buckets. |

## Typed models (OpenEnv)

- **`IndiaTaxBenchAction`** — `action_type` and optional prediction fields.
- **`IndiaTaxBenchObservation`** — scenario, task metadata, `task_difficulty`, feedback, `submitted_predictions`, `valid_actions`, plus `reward`, `done`, `metadata`.
- **`IndiaTaxBenchReward`** — standalone Pydantic model documenting the scalar reward in **[0.0, 1.0]** (same range as `observation.reward` after clamping).

## Action space

Typed `IndiaTaxBenchAction` (`action_type` + optional fields):

| `action_type` | Fields | Effect |
|---------------|--------|--------|
| `submit_prediction` | `predicted_total`, `predicted_initial_tax`, `predicted_surcharge`, `predicted_cess` (numbers) | Append one candidate prediction; per-step reward from mean field match vs oracle. |
| `revise_prediction` | `item_index` (int), same four `predicted_*` | Replace a prior prediction; reward from score improvement. |
| `request_context` | — | Qualitative hint; small negative reward. |
| `finalize` | — | Episode ends; reward aggregates base match, efficiency, hints. |

## Observation space

`IndiaTaxBenchObservation` includes `scenario_json` (public scenario only), `task_id`, `task_description`, **`task_difficulty`** (`easy` \| `medium` \| `hard`), `feedback`, `submitted_predictions`, `steps_remaining`, `hints_used`, `valid_actions`, plus standard **`reward`** in **[0.0, 1.0]**, `done`, `metadata`.

**Reward shaping:** `submit_prediction` / `revise_prediction` emit partial progress; `request_context` incurs a small penalty; `finalize` combines best submission match, step efficiency, hint usage, and optional auto-finalize penalty.

## Setup

```bash
cd IndiaTaxBench-OpenEnv
uv sync --extra dev
# or: pip install -e ".[dev]"
```

## Run server

```bash
uv run python -m india_tax_bench.server.app
# or: uvicorn india_tax_bench.server.app:app --host 0.0.0.0 --port 8000
```

## API examples

```bash
curl -s -X POST http://localhost:8000/reset -H "Content-Type: application/json" \
  -d '{"task": "salary_metro_80c_fy2425"}' | jq .

curl -s -X POST http://localhost:8000/step -H "Content-Type: application/json" \
  -d '{"action": {"action_type": "submit_prediction", "predicted_total": 242580, "predicted_initial_tax": 233250, "predicted_surcharge": 0, "predicted_cess": 9330}}' | jq .

curl -s -X POST http://localhost:8000/step -H "Content-Type: application/json" \
  -d '{"action": {"action_type": "finalize"}}' | jq .
```

## Baseline inference

```bash
export API_BASE_URL=https://router.huggingface.co/v1   # optional; this is the default
export MODEL_NAME=Qwen/Qwen2.5-7B-Instruct               # optional
# Use either key (OpenAI-compatible hosts accept one bearer-style key):
export OPENAI_API_KEY=sk-...     # preferred name in OpenAI docs
# export HF_TOKEN=hf_...         # alternative (e.g. Hugging Face Inference/Router)
# export API_KEY=...             # generic third option (same usage as the keys above)

# ENV_URL defaults to the live Space; override for local server:
# export ENV_URL=http://127.0.0.1:8000
# optional: export INFERENCE_MAX_TASKS=2
python inference.py
```

Uses the **OpenAI Python client** against `API_BASE_URL` / `MODEL_NAME` with **`temperature=0`** for reproducible trajectories. Parse per-task **`[END] success=... score=...`** lines for baseline numbers.

### Deterministic baseline (oracle policy)

Upper bound when the model submits **exact** oracle components then `finalize` (no hints). Regenerate anytime:

`uv run python scripts/print_grader_calibration.py --markdown`

| task_id | difficulty | oracle_base_reward | oracle_finalize_reward |
|---------|------------|--------------------|-------------------------|
| `business_only_fy2425` | medium | 1.0000 | 1.0000 |
| `salary_capital_gains_fy2425` | hard | 1.0000 | 1.0000 |
| `salary_metro_80c_fy2425` | hard | 1.0000 | 1.0000 |
| `salary_non_metro_hra_fy2425` | medium | 1.0000 | 1.0000 |
| `senior_salary_deductions_fy2425` | easy | 1.0000 | 1.0000 |

**LLM baseline:** depends on provider and weights; run `python inference.py` with your keys and record `[END]` lines per task for submission tables.

## Training notebook

See [`notebooks/train_qwen_india_tax.ipynb`](notebooks/train_qwen_india_tax.ipynb) for a **Qwen2.5** LoRA / SFT stub on the same JSONL.

## Tests and validation

```bash
uv run pytest tests/ -v
uv run openenv validate .
# With a running server (e.g. uv run uvicorn india_tax_bench.server.app:app --port 8000):
uv run openenv validate --url http://127.0.0.1:8000 --timeout 30
```

Full submission-style check (same idea as `meeting_notes_env/validate-submission.sh`): ping your Space’s `/reset`, optionally `docker build`, then `openenv validate .` and `openenv validate --url …`.

```bash
# Fast CI-style (no local Docker image build):
SKIP_DOCKER=1 bash validate-submission.sh https://prateekdebit-india-tax-bench-openenv.hf.space .

# Include Docker build from repo root Dockerfile (~10+ minutes first time):
bash validate-submission.sh https://prateekdebit-india-tax-bench-openenv.hf.space .

# Local openenv only (no runtime --url pass):
SKIP_RUNTIME=1 bash validate-submission.sh https://prateekdebit-india-tax-bench-openenv.hf.space .
```

## Oracle calibration (README tables)

```bash
uv run python scripts/print_grader_calibration.py --markdown
```

## Disclaimer

Simulation and research only — **not** tax, legal, or filing advice. Oracle values come from the `taxcalcindia` library, not government e-filing APIs.

## Docker

```bash
docker build -t india-tax-bench .
docker run -p 8000:8000 india-tax-bench
```

## Upstream

- [OpenEnv](https://github.com/meta-pytorch/OpenEnv)
- [IndiaTaxBench-OpenEnv on GitHub](https://github.com/Prateedk/IndiaTaxBench-OpenEnv)
