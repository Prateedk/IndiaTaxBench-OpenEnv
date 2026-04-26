#!/usr/bin/env python3
"""Baseline multi-step inference for IndiaTaxBench (Qwen / OpenAI-compatible API).

Environment variables:
    API_BASE_URL   — OpenAI-compatible endpoint (default: HF router)
    MODEL_NAME     — model id (default: Qwen/Qwen2.5-1.5B-Instruct; set e.g. Qwen2.5-0.5B for smallest)
    OPENAI_API_KEY — Preferred API key for OpenAI-compatible providers
    HF_TOKEN / API_KEY — Alternative secrets if OPENAI_API_KEY is unset
    ENV_URL        — IndiaTaxBench server base URL (default: live HF Space)
    INFERENCE_MODE — `advisor` (default) or `numeric` (liability four-tuple)
    INFERENCE_HTTP_TIMEOUT — seconds (default: 120)
    INFERENCE_MAX_TASKS    — smoke: only first N tasks
    SUCCESS_SCORE_THRESHOLD — finalize reward above this counts as success (default: 0.35)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import requests
from openai import OpenAI

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import importlib.util

from server.tasks import ALL_TASK_IDS

_NB_SPEC = importlib.util.spec_from_file_location(
    "notebook_env_helpers",
    ROOT / "scripts" / "notebook_env_helpers.py",
)
_nbh = importlib.util.module_from_spec(_NB_SPEC)
assert _NB_SPEC.loader
_NB_SPEC.loader.exec_module(_nbh)

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")


def _resolve_api_key() -> str:
    key = (
        os.getenv("OPENAI_API_KEY")
        or os.getenv("HF_TOKEN")
        or os.getenv("API_KEY")
        or ""
    ).strip()
    if not key:
        raise ValueError(
            "Set OPENAI_API_KEY, HF_TOKEN, or API_KEY for the OpenAI-compatible API."
        )
    return key


API_KEY = _resolve_api_key()

ENV_URL = os.getenv(
    "ENV_URL",
    "https://prateekdebit-india-tax-bench-openenv.hf.space",
)
HTTP_TIMEOUT = float(os.getenv("INFERENCE_HTTP_TIMEOUT", "120"))
_MAX_RAW = os.getenv("INFERENCE_MAX_TASKS", "").strip()
INFERENCE_MAX_TASKS = int(_MAX_RAW) if _MAX_RAW.isdigit() else None
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.35"))
# "numeric" (default): submit_prediction / finalize. "advisor": submit_tax_advice / finalize_advice.
INFERENCE_MODE = os.getenv("INFERENCE_MODE", "advisor").strip().lower()

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

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


def call_llm(messages: List[Dict[str, str]], max_tokens: int = 512) -> str:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()


def clean_json(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        raw = "\n".join(lines).strip()
    return raw


def parse_prediction(raw: str) -> Dict[str, float]:
    cleaned = clean_json(raw)
    data = json.loads(cleaned)
    if not isinstance(data, dict):
        return {}
    out: Dict[str, float] = {}
    for k in ("total", "initial_tax", "surcharge", "cess"):
        v = data.get(k)
        if v is None:
            out[k] = 0.0
        else:
            out[k] = float(v)
    return out


def new_env_session() -> requests.Session:
    """Fresh cookie jar per episode so HF Space sticks reset/step/finalize."""
    return requests.Session()


def env_reset(task_id: str, *, sess: requests.Session, advisor: bool = False) -> dict:
    body: Dict[str, Any] = {"task": task_id}
    if advisor:
        body["advisor"] = True
    resp = sess.post(
        f"{ENV_URL.rstrip('/')}/reset", json=body, timeout=HTTP_TIMEOUT
    )
    resp.raise_for_status()
    return resp.json()


def env_step(action: Dict[str, Any], *, sess: requests.Session) -> dict:
    resp = sess.post(
        f"{ENV_URL.rstrip('/')}/step", json={"action": action}, timeout=HTTP_TIMEOUT
    )
    resp.raise_for_status()
    return resp.json()


REVISE_THRESHOLD = 0.85


def _task_list() -> List[str]:
    ids = list(ALL_TASK_IDS)
    if INFERENCE_MAX_TASKS is not None:
        return ids[: max(0, INFERENCE_MAX_TASKS)]
    return ids


def main() -> None:
    for task_id in _task_list():
        all_rewards: list[float] = []
        success = False
        steps = 0
        scenario = ""

        try:
            print(
                f"[START] task={task_id} env=india-tax-bench model={MODEL_NAME} mode={INFERENCE_MODE}",
                flush=True,
            )

            sess = new_env_session()
            if INFERENCE_MODE == "advisor":
                reset_data = env_reset(task_id, sess=sess, advisor=True)
                obs = reset_data.get("observation", reset_data)
                scenario = obs.get("scenario_json", "")
                tdesc = obs.get("task_description", "")
                msgs = _nbh.build_advisor_messages(
                    scenario, task_id, task_description=tdesc
                )
                raw = call_llm(msgs, max_tokens=700)
                action = {
                    "action_type": "submit_tax_advice",
                    "advice_text": raw,
                }
                step_data = env_step(action, sess=sess)
                step_obs = step_data.get("observation", step_data)
                reward = float(step_data.get("reward", step_obs.get("reward", 0.0)) or 0.0)
                done = step_data.get("done", step_obs.get("done", False))
                error_str = step_obs.get("metadata", {}).get("error", None)
                steps += 1
                all_rewards.append(reward)
                print(
                    f"[STEP]  step={steps} action=submit_tax_advice "
                    f"reward={reward:.2f} done={'true' if done else 'false'} "
                    f"error={error_str or 'null'}",
                    flush=True,
                )
                if not done:
                    sa = step_obs.get("submitted_advice", [])
                    rub = float(sa[-1].get("rubric", 0.0)) if sa else 0.0
                    if rub < 0.45 and sa:
                        feedback = step_obs.get("feedback", "")
                        prev = sa[-1].get("raw", "")[:4000]
                        rmsgs = _nbh.build_revise_advisor_messages(
                            scenario, prev, feedback
                        )
                        rev_raw = call_llm(rmsgs, max_tokens=700)
                        rev_action = {
                            "action_type": "revise_tax_advice",
                            "item_index": 0,
                            "advice_text": rev_raw,
                        }
                        step_data = env_step(rev_action, sess=sess)
                        step_obs = step_data.get("observation", step_data)
                        reward = float(step_data.get("reward", step_obs.get("reward", 0.0)) or 0.0)
                        done = step_data.get("done", step_obs.get("done", False))
                        error_str = step_obs.get("metadata", {}).get("error", None)
                        steps += 1
                        all_rewards.append(reward)
                        print(
                            f"[STEP]  step={steps} action=revise_tax_advice "
                            f"reward={reward:.2f} done={'true' if done else 'false'} "
                            f"error={error_str or 'null'}",
                            flush=True,
                        )
                if not done:
                    fin_action = {"action_type": "finalize_advice"}
                    step_data = env_step(fin_action, sess=sess)
                    step_obs = step_data.get("observation", step_data)
                    reward = float(step_data.get("reward", step_obs.get("reward", 0.0)) or 0.0)
                    error_str = step_obs.get("metadata", {}).get("error", None)
                    steps += 1
                    all_rewards.append(reward)
                    print(
                        f"[STEP]  step={steps} action=finalize_advice "
                        f"reward={reward:.2f} done=true error={error_str or 'null'}",
                        flush=True,
                    )
            else:
                reset_data = env_reset(task_id, sess=sess, advisor=False)
                obs = reset_data.get("observation", reset_data)
                scenario = obs.get("scenario_json", "")

                user_msg = f"Scenario JSON:\n{scenario}\n"
                raw = call_llm(
                    [
                        {"role": "system", "content": PREDICT_PROMPT},
                        {"role": "user", "content": user_msg},
                    ]
                )
                pred = parse_prediction(raw)

                action = {
                    "action_type": "submit_prediction",
                    "predicted_total": pred.get("total", 0.0),
                    "predicted_initial_tax": pred.get("initial_tax", 0.0),
                    "predicted_surcharge": pred.get("surcharge", 0.0),
                    "predicted_cess": pred.get("cess", 0.0),
                }
                step_data = env_step(action, sess=sess)
                step_obs = step_data.get("observation", step_data)
                reward = float(step_data.get("reward", step_obs.get("reward", 0.0)) or 0.0)
                done = step_data.get("done", step_obs.get("done", False))
                error_str = step_obs.get("metadata", {}).get("error", None)
                steps += 1
                all_rewards.append(reward)
                print(
                    f"[STEP]  step={steps} action={json.dumps(action)} "
                    f"reward={reward:.2f} done={'true' if done else 'false'} "
                    f"error={error_str or 'null'}",
                    flush=True,
                )

                if not done:
                    submitted = step_obs.get("submitted_predictions", [])
                    mean_score = 0.0
                    if submitted:
                        mean_score = float(submitted[-1].get("score", 0.0))
                    if mean_score < REVISE_THRESHOLD and submitted:
                        feedback = step_obs.get("feedback", "")
                        prev = json.dumps(submitted[-1])
                        rev_raw = call_llm(
                            [
                                {
                                    "role": "system",
                                    "content": "You are a precise tax numeric estimator.",
                                },
                                {
                                    "role": "user",
                                    "content": REVISE_PROMPT.format(
                                        scenario=scenario,
                                        previous=prev,
                                        feedback=feedback,
                                    ),
                                },
                            ],
                            max_tokens=512,
                        )
                        rp = parse_prediction(rev_raw)
                        rev_action = {
                            "action_type": "revise_prediction",
                            "item_index": 0,
                            "predicted_total": rp.get("total", 0.0),
                            "predicted_initial_tax": rp.get("initial_tax", 0.0),
                            "predicted_surcharge": rp.get("surcharge", 0.0),
                            "predicted_cess": rp.get("cess", 0.0),
                        }
                        step_data = env_step(rev_action, sess=sess)
                        step_obs = step_data.get("observation", step_data)
                        reward = float(
                            step_data.get("reward", step_obs.get("reward", 0.0)) or 0.0
                        )
                        done = step_data.get("done", step_obs.get("done", False))
                        error_str = step_obs.get("metadata", {}).get("error", None)
                        steps += 1
                        all_rewards.append(reward)
                        print(
                            f"[STEP]  step={steps} action={json.dumps(rev_action)} "
                            f"reward={reward:.2f} done={'true' if done else 'false'} "
                            f"error={error_str or 'null'}",
                            flush=True,
                        )

                if not done:
                    fin_action = {"action_type": "finalize"}
                    step_data = env_step(fin_action, sess=sess)
                    step_obs = step_data.get("observation", step_data)
                    reward = float(
                        step_data.get("reward", step_obs.get("reward", 0.0)) or 0.0
                    )
                    error_str = step_obs.get("metadata", {}).get("error", None)
                    steps += 1
                    all_rewards.append(reward)
                    print(
                        f"[STEP]  step={steps} action={{\"action_type\":\"finalize\"}} "
                        f"reward={reward:.2f} done=true error={error_str or 'null'}",
                        flush=True,
                    )

            final_score = all_rewards[-1] if all_rewards else 0.0
            success = final_score > SUCCESS_SCORE_THRESHOLD

        except Exception as exc:
            if not all_rewards:
                all_rewards.append(0.0)
            steps = max(steps, 1)
            print(
                f"[STEP]  step={steps} action=error "
                f"reward=0.00 done=true error={exc}",
                flush=True,
            )
            final_score = 0.0
            success = False

        final_score = min(max(final_score, 0.0), 1.0)
        success_str = "true" if success else "false"
        rewards_str = ",".join(f"{r:.2f}" for r in all_rewards)
        print(
            f"[END]  success={success_str} steps={steps} score={final_score:.2f} rewards={rewards_str}",
            flush=True,
        )


if __name__ == "__main__":
    main()
