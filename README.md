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

## Task source

Tasks are loaded from vendored **[`india_tax_capture/data/india_tax_rows.jsonl`](india_tax_capture/data/india_tax_rows.jsonl)** (one episode per line). See [`india_tax_capture/README.md`](india_tax_capture/README.md) for FY mapping (`financial_year=2025` = FY 2024–25).

## Action space

Typed `IndiaTaxBenchAction` (`action_type` + optional fields):

| `action_type` | Fields | Effect |
|---------------|--------|--------|
| `submit_prediction` | `predicted_total`, `predicted_initial_tax`, `predicted_surcharge`, `predicted_cess` (numbers) | Append one candidate prediction; per-step reward from mean field match vs oracle. |
| `revise_prediction` | `item_index` (int), same four `predicted_*` | Replace a prior prediction; reward from score improvement. |
| `request_context` | — | Qualitative hint; small negative reward. |
| `finalize` | — | Episode ends; reward aggregates base match, efficiency, hints. |

## Observation space

`IndiaTaxBenchObservation` includes `scenario_json` (public scenario only), `task_id`, `task_description`, `feedback`, `submitted_predictions`, `steps_remaining`, `hints_used`, `valid_actions`, plus standard `reward`, `done`, `metadata`.

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
export HF_TOKEN=hf_...
export ENV_URL=http://127.0.0.1:8000
# optional: export INFERENCE_MAX_TASKS=2
python inference.py
```

Uses OpenAI-compatible `API_BASE_URL` / `MODEL_NAME` (default **Qwen/Qwen2.5-7B-Instruct** on Hugging Face router).

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
