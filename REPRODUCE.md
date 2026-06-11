# Reproducing RouteBalance Results

This document provides a step-by-step recipe for reproducing every published result in the RouteBalance paper, end-to-end. Every script referenced here lives under either `reproduce/` (consolidated entry points) or `route_balance/` (the implementation).

> **Required hardware**: a heterogeneous GPU pool with at least four model sizes deployable on disjoint GPU types. The headline numbers used a 13-instance CloudLab reservation: `2× A100-80GB (Qwen2.5-72B)`, `3× V100-32GB (Qwen2.5-14B)`, `5× A30-24GB (Qwen2.5-7B)`, `3× A30-24GB (Qwen2.5-3B)`. A single-host smoke is supported (see Quick start in the README) but does not reproduce the cluster Pareto.

> **Required software**: Python 3.10, vLLM ≥ 0.6, XGBoost ≥ 2.0, sentence-transformers (`all-MiniLM-L6-v2`), DeepEval (off-line scoring only). `pip install -e . && pip install -r requirements.txt` from the repo root.

---

## 0. Bring up the cluster

```bash
# One-time host setup — installs vLLM, model weights, configs
bash route_balance/exp/route_balance/setup.sh

# Deploy all 13 instances + scheduler + predictor in one fan-out
python route_balance/exp/route_balance/deploy_route_balance.py \
    --host-config   route_balance/config/route_balance/host_configs.json \
    --model-config  route_balance/config/route_balance/model_deployment.json \
    --deploy-services model_instance predictor monitor scheduler
```

`host_configs.json` enumerates SSH targets and their GPU type; `model_deployment.json` pins the (model, instance) assignments. Both are versioned in-tree under `route_balance/config/route_balance/`. Health-check via `curl $SCHEDULER:8200/health` after fan-out completes (~6 min).

---

## 1. Headline 3-axis Pareto (Table 3, Figure 2)

The headline numbers come from a 215-cell sweep:

- **Strategy**: `RouteBalance` weight tuples on the simplex with $w_q\!+\!w_l\!+\!w_c\!=\!1$ — Balance (1/3, 1/3, 1/3) plus eight quality-/latency-/cost-emphasized tuples
- **Baselines**: `AvengersPro` (4-model retrained, $p_w\!\in\!\{0.25, 0.4, 0.53, 0.7\}$), `BEST-Route 4-way` (argmax + threshold sweep $\{0.3, 0.5, 0.7\}$), `Passthrough` (round-robin / shortest-queue / random)
- **Loads**: $\lambda \in \{6, 8, 10, 12, 18, 24, 30\}$ Poisson req/s
- **Workload**: $N=3534$ prompts per cell, drawn from `data/best-route-v3-test-3534-eval.jsonl` (zero train leakage; verified against the 14,919-prompt training corpus)

### 1.1. Run the sweep

```bash
SCHEDULER=node5:8200 \
DATA=data/best-route-v3-test-3534-eval.jsonl \
RESULTS_ROOT=results/main_table_full_3534/route_balance_e2e \
  bash reproduce/run_main_table.sh
```

`run_main_table.sh` iterates the strategy × λ matrix, hot-swaps scheduler config between cells via `POST /v1/config` (no scheduler restart), and writes one JSON per cell under `$RESULTS_ROOT/<strategy>/<cell_id>.json`. Total wall: ~13 h on the 13-instance reference cluster.

### 1.2. Aggregate

```bash
python reproduce/aggregate_results.py \
    --results-root results/main_table_full_3534/route_balance_e2e \
    --out-csv      results/AGGREGATE_FULL.csv \
    --out-jsonl    results/AGGREGATE_FULL.jsonl
```

The aggregator joins each per-cell JSON against the pre-computed deepeval lookup (`models/route_balance/deepeval_models/quality_lookup.parquet`), then emits per-cell `mean_e2e`, `p95_ttft`, `p95_tpot`, `dq_quality`, `cost_per_1k_tokens`, plus a summary row per strategy.

### 1.3. Generate Pareto figure + extremes table

```bash
python reproduce/make_main_figures.py \
    --aggregate results/AGGREGATE_FULL.csv \
    --out-dir   nips_draft/figures/
```

Outputs:

- `fig_pareto_qlc.pdf` — three Pareto panels (Q-L, Q-C, L-C convex hulls), shared top legend
- `fig_radar_4axis.pdf` — Quality / Throughput / Cost⁻¹ / Latency⁻¹ radar
- `tab_extremes.tex` — LaTeX table of headline cells (max-quality, max-throughput, min-cost rows)

---

## 2. Bootstrap 95% CIs (paper §5)

```bash
python reproduce/compute_bootstrap_ci.py \
    --aggregate results/AGGREGATE_FULL.csv \
    --lambda    12 \
    --B         1000 \
    --out       results/CI_lambda12.json
```

Reproduces the per-cell 95% confidence intervals reported at $\lambda\!=\!12$ ($N=3534$, $B=1000$ resamples): mean E2E ±43–116 ms, DeepEval quality ±0.009.

---

## 3. Scheduling-overhead breakdown (Table 4, Figure 4)

This study compares three scheduler architectures at $\lambda\!=\!12$ and isolates BEST-Route at $\lambda\!=\!30$:

| Variant | Predictor | Architecture | Overhead at λ=12 |
|---|---|---|---|
| RB (deployed) | KNN + per-instance XGBoost | In-process | 149 ms |
| RB (RoBERTa HTTP sidecar) | RoBERTa-fused | Per-request HTTP | 47 s |
| RB (centralized XGBoost) | KNN + centralized XGBoost | gRPC, single host | 4.2 s |
| BEST-Route (pipeline mode) | DeBERTa | Per-request classifier | 33.1 s @ λ=30 |

### 3.1. Run

```bash
bash reproduce/run_overhead_study.sh   # ~2.5 h cluster
```

This script bounces the scheduler four times — once per predictor configuration in `route_balance/config/route_balance/predictor_*` — each time running the same $\lambda\!=\!12$, $N\!=\!3534$ cell.

### 3.2. Plot

```bash
python reproduce/make_overhead_figure.py \
    --results-root results/overhead_study/ \
    --out-pdf      nips_draft/figures/fig_overhead.pdf \
    --out-tex      nips_draft/tab_sched_combined.tex
```

---

## 4. Budget-control evaluation (paper §5.4)

Demonstrates that RouteBalance respects per-request budgets via dynamic `max_tokens` clamp + streaming early-stop.

### 4.1. Build the calibrated dataset

```bash
python reproduce/build_budget_dataset.py \
    --in-data  data/best-route-v3-test-3534-eval.jsonl \
    --out-dir  data/budget/ \
    --fractions 0.1 0.25 0.5 1.0
```

For each sampled prompt, draws a budget uniformly in $[\,(\text{3B input cost} + \text{3B predicted output cost}),\ (\text{72B}\ldots)\,]$. Emits one dataset per sample fraction.

### 4.2. Run

```bash
SCHEDULER=node5:8200 \
DATA_ROOT=data/budget \
RESULTS_ROOT=results/budget_control \
  bash reproduce/run_budget_control.sh
```

8 cells: 4 sample fractions × {RouteBalance Balance + budget filter ON, BEST-Route 4-way RR}.

### 4.3. Aggregate

```bash
python reproduce/aggregate_budget.py \
    --results-root results/budget_control \
    --out-csv      results/budget_control.csv
```

Reports `budget_violation_rate`, `budget_exhausted_rate`, mean E2E, and quality per cell.

---

## 5. Ablations (paper §5.3, Appendix G)

```bash
SCHEDULER=node5:8200 \
DATA=data/best-route-v3-test-3534-eval.jsonl \
RESULTS_ROOT=results/ablations \
  bash reproduce/run_full_ablation.sh
```

Drives four sub-sweeps via scheduler hot-swap:

1. **LPT off** — `lpt_sort_key=none` × λ ∈ {6, 8, 10, 12, 18, 24, 30}
2. **Adaptive batching off** — `batch_config.adaptive_sizing=false`, `max_batch_size=32` static × same λ grid
3. **Batch-size sweep** — `max_batch_size ∈ {1, 4, 8, 16, 32}` × λ ∈ {6, 12, 24, 30}; `max_batch_size=1` is the no-batching streaming-dispatch baseline
4. **TTFT/TPOT SLO filter on/off** — `filter.type ∈ {route_balance_cdf, none}` with `ttft_slo_ms=500`, `tpot_slo_ms=30` × full λ grid

### 5.1. Plots

```bash
python reproduce/make_ablation_figures.py \
    --results-root results/ablations \
    --out-dir      nips_draft/figures/
```

Emits `fig_ablation_lpt.pdf`, `fig_ablation_batch.pdf`, `fig_ablation_filter.pdf`.

---

## 6. Predictor training (Appendix D)

The deployed predictors come from two pipelines.

### 6.1. Quality + length head (KNN over MiniLM embeddings)

Inputs: the released [`anon/route_balance_model_estimator`](https://huggingface.co/datasets/anon/route_balance_model_estimator) dataset, which already contains `reference_text` and the six quality signals per (prompt, model). The reference-grounded LLM-judge column is `deepeval-llama3.1-8b-it_reference`.

```bash
# 1. (Optional) regenerate the dataset from raw prompts
bash route_balance/predictor/route_balance/offline_training/run_full_eval_pipeline.sh

# 2. Train the deployed KNN head
python route_balance/predictor/route_balance/offline_training/train_knn_estimator.py \
    --train-data hf://anon/route_balance_model_estimator:train \
    --val-data   hf://anon/route_balance_model_estimator:test \
    --embedder   sentence-transformers/all-MiniLM-L6-v2 \
    --k 50 --weighted \
    --quality-col deepeval-llama3.1-8b-it_reference \
    --out-dir    models/route_balance/knn_k50/
```

The off-line judge is **`Llama-3.1-8B-Instruct`**, scored by DeepEval's G-Eval framework against dataset-specific reference text (GSM8K solutions, SQuAD spans, RewardBench-preferred answers, MixInstruct gold, LMSYS chosen turns, CodeUltraFeedback top-rated, refuse-and-explain for harmful prompts). Judge ≠ candidate. Re-running `score_with_deepeval.py` requires a separate vLLM endpoint serving `Llama-3.1-8B-Instruct`.

### 6.2. Per-(model, GPU) latency heads (XGBoost TTFT/TPOT/E2E)

The latency predictors are trained **per (model, GPU) tier**, not from a single cluster-wide dataset. There is no combined dataset to download. The training pipeline runs locally on each tier's head node, producing one XGBoost head per target (TTFT, TPOT, E2E).

**Step 1 — sweep schedule state vs latency on each tier head node.** SSH into one machine of each (model, GPU) tier (e.g., one A100/72B host, one V100/14B host, one A30/7B host, one A30/3B host), then:

```bash
# On each head node, with vLLM serving the tier's target model on :8000
python route_balance/predictor/route_balance/offline_training/generate_latency_benchmark.py \
    --backend-url   http://localhost:8000 \
    --model-name    Qwen2.5-7B \
    --gpu-tag       a30-7b \
    --qps-grid      "1,2,4,6,8,10,14,18,24,30" \
    --num-prompts   2000 \
    --out           data/latency_traces/a30-7b.jsonl
```

The script drives an open-loop Poisson workload at each QPS level and logs the schedule-state features (`pending_tokens`, `kv_pressure`, `running_count`, `service_rate`, ...) the deployed predictor reads at run time, alongside the realized TTFT/TPOT/E2E. Run this once per tier (~30–60 min per tier on the reference cluster).

**Step 2 — convert traces to XGBoost-ready features per tier.**

```bash
python route_balance/predictor/route_balance/offline_training/prepare_xgboost_dataset.py \
    --traces  data/latency_traces/a30-7b.jsonl \
    --out     data/xgboost_features/a30-7b.parquet
```

**Step 3 — train one XGBoost model per (tier, target). Uses `train_xgboost_3model.py`, which trains TTFT, TPOT, and E2E heads in a single invocation per tier:**

```bash
python route_balance/predictor/route_balance/offline_training/train_xgboost_3model.py \
    --features  data/xgboost_features/a30-7b.parquet \
    --tag       a30-7b \
    --out-dir   models/route_balance/xgboost/a30-7b/
```

Repeat per tier (e.g., `a100-72b`, `v100-14b`, `a30-7b`, `a30-3b`). The scheduler loads each tier's three models from `models/route_balance/xgboost/<tag>/{ttft,tpot,e2e}.json` at startup with `CUDA_VISIBLE_DEVICES=""` so they stay CPU-resident — `booster.inplace_predict` over the full $|R_B|\!\times\!|I|$ matrix runs in ~3 ms.

### 6.3. Predictor evaluation (Appendix D MAE table)

```bash
python route_balance/predictor/route_balance/offline_training/evaluate_predictors.py \
    --models-root models/route_balance/ \
    --val-data    data/latency_traces/val/ \
    --out         results/predictor_mae.csv
```

Reports per-(target, model, GPU) MAE in s/tok and end-to-end MAPE.

---

## 7. Cell-level smoke

If you only have one GPU, the smoke recipe in the README's "Local smoke" section is the minimum viable reproduction. For a single-cell smoke against the released cluster code without a cluster:

```bash
SCHEDULER=node5:8200 N=100 LAMBDA=4 STRATEGY=route_balance_balance \
    DATA=data/best-route-v3-test-500-clean.jsonl \
    bash reproduce/smoke_one_cell.sh
```

Runs in ~2 min and emits one `<cell_id>.json` matching the schema consumed by `aggregate_results.py`.

---

## 8. File-to-claim crosswalk

| Paper artifact | Generating script |
|---|---|
| Table 1 (model + GPU + price) | manual; sources cited in caption |
| Table 2 (notation) | manual |
| Table 3 (extremes) | `reproduce/make_main_figures.py` → `tab_extremes.tex` |
| Table 4 (scheduling overhead) | `reproduce/make_overhead_figure.py` → `tab_sched_combined.tex` |
| Figure 1 (architecture) | manual / TikZ |
| Figure 2 (3-axis Pareto) | `reproduce/make_main_figures.py` → `fig_pareto_qlc.pdf` |
| Figure 3 (4-axis radar) | `reproduce/make_main_figures.py` → `fig_radar_4axis.pdf` |
| Figure 4 (overhead breakdown) | `reproduce/make_overhead_figure.py` → `fig_overhead.pdf` |
| §5 bootstrap CIs | `reproduce/compute_bootstrap_ci.py` |
| §5.3 LPT / batch / filter ablations | `reproduce/run_full_ablation.sh` + `make_ablation_figures.py` |
| §5.4 budget control | `reproduce/run_budget_control.sh` + `aggregate_budget.py` |
| Appendix D dataset construction | `route_balance/predictor/route_balance/offline_training/run_full_eval_pipeline.sh` |
| Appendix D predictor MAE | `evaluate_predictors.py` |

---

## 9. Compute budget cheat-sheet

| Stage | Wall (reference cluster) |
|---|---|
| Cluster bring-up | ~6 min |
| Headline 215-cell sweep (§1) | ~13 h |
| Aggregation + figures (§1.2–1.3) | <5 min |
| Scheduling overhead study (§3) | ~2.5 h |
| Budget control (§4) | ~45 min |
| Full ablations (§5) | ~6 h |
| KNN training (§6.1) | ~10 min CPU |
| XGBoost training (§6.2, after traces) | ~30 min CPU |
| Latency-trace regeneration | ~6 h cluster |

Total cluster-time to reproduce all numbers from a fresh reservation: **~28 h**.

---

## 10. Known limitations

- **Single seed.** Headline cells were run with a single per-cell PRNG seed; bootstrap CIs are over the $N=3534$ workload, not over re-runs. Re-seeding at $\lambda \in \{12, 30\}$ is straightforward via the `--seed` flag on `run_main_table.sh`.
- **Reference cluster.** Wall times above assume the 13-instance heterogeneous topology. Smaller clusters scale roughly linearly with the number of model-size tiers; below 4 tiers the Pareto degenerates.
- **DeepEval cost.** Off-line quality scoring with the Llama-3.1-8B judge is the slowest part of dataset regeneration (~5 h on a single A100 for the 18,608-prompt corpus). The released datasets ship the labels so this is optional.

---

If something does not reproduce within 5% of paper values, please open an issue with: (i) which cell, (ii) `git rev-parse HEAD`, (iii) the per-cell JSON, (iv) `pip freeze`. Floating-point non-determinism in vLLM and XGBoost is bounded but non-zero across hardware revisions.

---

## 11. New env-gated experiments (App G fusion-isolation, non-stationary arrivals)

These reviewer-driven ablations are reproduced by env toggles on the same serving stack
(all **default-off**, so the main grid is unaffected). Drivers: `reproduce/route_balance_extras/`.
Gates live in `route_balance/global_scheduler/route_balance/route_balance_serve.py` and
`route_balance/benchmark/route_balance/benchmark_serving.py`. Each cell: `--seed 5`, `N=3534`,
`best-route-v3-test-3534-eval.jsonl`. Quality via `reproduce/aggregate_results.py` (zero-safe).

| Experiment (paper) | Gate / flag | Driver | Expected |
|---|---|---|---|
| App G **arm 3** (decoupled-predictive, Table latablation) | `ROUTE_BALANCE_TIEBREAK=that` | `p1_arm3.sh` | 3.42/3.50/3.75 s @ λ12/24/30, 72B 14%, q0.385 |
| App G **arm 4** (static per-tier latency prior) | `ROUTE_BALANCE_STATIC_LAT="3b=12.3,7b=20.6,14b=18.4,72b=47.4"` (ms/tok) | `arm4_static_lat.sh` | matches arm 1: mix 58/11/31/0, q0.369, E2E 2.41 s (stat) / 3.12 s (40-6 overload) |
| App G **non-stationary square wave** (Table nonstationary) | `ROUTE_BALANCE_SQUARE="lo,hi"` on serve+bench | `p3sq.sh` + `p3sq_avg.sh` | RB 2577 / enh-AvgPro 2733 / enh-BR4 7107 ms (30/6) |
| App G **non-stationary Gamma burst** | `--burstiness 0.3` (bench flag) | `p3_bursty.sh` | RB 2524 / enh-AvgPro 2473 / enh-BR4 4392 ms (b=0.3, λ18) |
| App G **overload probe** (40/6, peak>27.6 sustained max) | `ROUTE_BALANCE_SQUARE=6,40` | `ov_arm1_sq40.sh` | arm 1 E2E 3788 ms; per-phase tier mix stays flat |
| Multi-seed latency stability (App A) | `--seed` variation (no gate) | `uni_reseed.sh` | uniform ±1.4/2.7/2.0% @ λ12/24/30 |

**Native-mode contract (mandatory for `--scheduling route_balance`):** `CUDA_VISIBLE_DEVICES=""`
(else XGBoost probes the co-located GPU → ~600× predict slowdown), `ROUTE_BALANCE_INPROC_PREDICTOR=1`
(in-process predictor; else 200 ms/req sidecar fallback), and do **not** set `OMP_NUM_THREADS=1`
(throttles the MiniLM embedder). Baselines run `--scheduling pipeline`.
