# IndiaTaxBench: teaching models to *advise* on tax—not just to quote the tax code

## 1. The problem: why the **advisor** track is the exciting part

Most “AI + tax” demos stop at a quick answer: total tax (maybe a simple breakdown), done. That is useful—but it is not how real filers use help. What people actually need is **actionable, next-year guidance**: how to use 80C room, HRA evidence, NPS, health cover, capital-gains planning, and compliance steps—**grounded in their scenario** and written so a human can act on it.

The **advisor** mode in IndiaTaxBench is built for exactly that. The model must emit **structured JSON** (profile summary, concrete next-year actions, cautions) and the **environment** scores that output with a **rubric** tied to the task—so we are not rewarding memorized boilerplate, we are rewarding *useful* advice that matches the filer’s mix of salary, business, rent, and deductions. When the base model already looks “fine” in prose, the real problem is: **can we still move a scalar reward in a way that reflects real improvement?** That is the bar we care about for research and for product: **advisor quality that can be trained, measured, and improved.**

**Where the scenarios and oracle data come from.** We generate the JSONL-backed tasks by running [`india_tax_capture/capture_india_tax_dataset.py`](india_tax_capture/capture_india_tax_dataset.py) over a **manifest** of scenario files, using the **`taxcalcindia`** Python package as the reference computation engine. In code, each row is built by (1) constructing **`IncomeTaxCalculator(...)`** with `TaxSettings`, `SalaryIncome`, `BusinessIncome`, `CapitalGainsIncome`, `OtherIncome`, and `Deductions` built from the scenario JSON, and (2) calling **`calculate_tax(is_comparision_needed=…, is_tax_per_slab_needed=…, display_result=False)`** to produce the oracle tax outcome (and optional regime comparison or per-slab detail, depending on those two flags) that lands in the dataset’s `response` field.

---

## 2. Our **OpenEnv** environment: tasks + rewards, built for research

IndiaTaxBench is deployed as an **OpenEnv**-style service: you **reset** with a task id, take **steps** (submit or revise advice, then finalize), and get back **observations and rewards** from a real tax-advisor **rubric**—not a hand-wavy LLM-judge in the training loop. Tasks are **scenario-driven** (captured, oracle-checked rows) with clear **task ids**; **rewards** stay in a bounded range and decompose into rubric, efficiency, and penalties so you can debug *why* a run scored what it did.

What makes this setup great for the community: **one HTTP surface** to any trainer or RL stack, **deterministic task definitions** in the repo, and a path from **SFT** to **full RL** without changing the world model—swap the algorithm, keep the same `reset` / `step` contract. That is the kind of **interoperable** env we want for serious LLM + RL work on India income tax.

---

## 3. The training notebook: OpenEnv + reward signal → a trainable policy (today: **SFT**; tomorrow: your favorite RL)

Our notebook [`notebooks/train_qwen_india_tax.ipynb`](notebooks/train_qwen_india_tax.ipynb) shows end-to-end **local** training against the Space: rollouts, **LoRA** + **TRL `SFTTrainer`**, and an **advisor** mode that refreshes data from the env (**best-of-N** style selection per task when configured). We have **not** shipped a full **PPO / policy-gradient** loop in that notebook—**by design**: the point is to prove the **env + reward + task** pipeline first. The same hooks are exactly where a more sophisticated **RL** algorithm would plug in: same tasks, same rewards, richer optimization on top.

**Evidence** is in the final focus plot below: we plot **tasks that were not part of the SFT training set** (held-out and eval-only task ids). You still see **true progress** on those curves because the **shared adapter** generalizes; the graph is a honest check that we are not only “teaching to the test” on a fixed four scenario ids.

**Model for this run (reporting):** **`Qwen/Qwen2.5-3B-Instruct`** with LoRA—enough capacity for structured JSON and tax phrasing, still trainable on a single consumer GPU with the notebook defaults.

### Plot: test-set (non–SFT) focus — held-out + eval-only tasks

![Per-task env reward over training iterations for tasks not used as SFT targets](notebooks/output.png)

*Figure: `notebooks/output.png` — test-set style plot: tasks **not** used as SFT targets (held-out and eval-only relative to the training split). The curve shows whether the **shared** adapter still moves reward out-of-distribution before you scale to heavier RL.*

---

**Bottom line:** IndiaTaxBench’s **advisor** track turns “sounds good” into **measured** improvement. The OpenEnv integration makes that **reproducible**; the notebook shows **SFT + env** working today, with a clear on-ramp to **stronger RL** tomorrow—and a plot that keeps us honest on **out-of-training** tasks.
