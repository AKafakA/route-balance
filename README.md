# RouteBalance

**A unified router and load balancer for heterogeneous LLM serving under quality, latency, and cost constraints.**

Modern LLM clouds host multiple model sizes across heterogeneous GPU pools, exposing routing decisions that simultaneously affect output quality, end-to-end latency, and per-token cost. Existing model routers select models without instance-level visibility; serving load balancers optimize latency without quality awareness. RouteBalance fuses quality–cost model routing and queue-aware load balancing into a single online assignment over concrete model instances.

## Highlights

- **Three-axis routing on a simplex.** User-supplied weights $w_q\!+\!w_l\!+\!w_c\!=\!1$ pick a named operating point (Balance, Quality, Latency, Cost) without retraining.
- **In-process learned predictors.** A single MiniLM embedding feeds a $k{=}50$ KNN head returning per-model quality, expected output length, and a 16-bucket length distribution. Per-(model, GPU) XGBoost models for TTFT, TPOT, and E2E run in-process on the scheduler over the full $|R_B|\!\times\!|I|$ matrix in $\sim$3 ms.
- **Filter–rank pipeline.** Probabilistic CDF filters enforce per-request budget and TTFT/TPOT SLOs; surviving candidates are ranked by the normalized weighted score.
- **LPT-batched scheduler with sequential local-state updates** to prevent herding under concurrent dispatchers; $O(|R_B||I|)$.
- **Sub-second scheduling overhead** at production load — 149 ms at λ=12 on a 13-instance heterogeneous cluster, 300× below per-request HTTP-sidecar variants we evaluated.

## Architecture

```
route_balance/
├── global_scheduler/route_balance/   # batch scheduler, dispatchers, filters, routers
│   ├── route_balance_serve.py        # main scheduler service (OpenAI-compatible API)
│   ├── route_balance_instance/       # vLLM/Ollama backend wrappers
│   ├── routers/                      # route_balance_native, passthrough, best_route_4way, avengers_pro, ...
│   ├── filters/                      # CDF / budget filters (route_balance_cdf_filter)
│   └── dispatch/                     # round_robin, shortest_queue, random
├── predictor/route_balance/          # learned predictors
│   ├── route_balance_learned_predictor.py   # XGBoost TTFT/TPOT/E2E
│   ├── model_estimator.py            # KNN quality + length-bucket head
│   ├── estimators/                   # KNN, ModernBERT, RoBERTa, PFS, roofline alternatives
│   └── offline_training/             # full training pipeline (datasets → labels → models)
├── benchmark/route_balance/          # async open-loop Poisson benchmark client
│   ├── benchmark_serving.py          # extends vllm/benchmarks/benchmark_serving
│   └── route_balance_end_point_func.py
├── exp/route_balance/                # deployment + sweep orchestrators
│   ├── deploy_route_balance.py       # SSH-fanout deployment to cluster
│   ├── e2e/                          # end-to-end Pareto + budget-control orchestrators
│   ├── ablation/                     # LPT, adaptive batching, filter, w_bal sweeps
│   └── csd3/                         # SLURM scripts for predictor training
├── route_balance_pd/                 # prefill–decode disaggregation extension
├── config/route_balance/             # deployment / predictor / scheduler configs
└── reproduce/                        # scripts to reproduce paper results (see REPRODUCE.md)
```

## Quick start

### Local smoke (single A100 / two A30s)

```bash
# 1. Install deps
pip install -e . && pip install -r requirements.txt

# 2. Bring up two model instances (for routing to make sense, you need ≥2)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-3B --port 8000 &
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B --port 8001 &

# 3. Start the scheduler
python -m route_balance.global_scheduler.route_balance.route_balance_serve \
    --model_config_path route_balance/config/route_balance/model_deployment_smoketest.json \
    --predictor-config route_balance/config/route_balance/predictor_config_smoketest_knn.json \
    --scheduler-config route_balance/config/route_balance/scheduler_config_smoketest.json \
    --scheduling route_balance --port 8200

# 4. Smoke benchmark
python -m route_balance.benchmark.route_balance.benchmark_serving \
    --backend route_balance --host 127.0.0.1 --port 8200 \
    --dataset-path data/best-route-v3-test-500-clean.jsonl \
    --num-prompts 100 --request-rate 4
```

### Multi-node cluster

The cluster orchestrator drives a heterogeneous CloudLab reservation (or any SSH-reachable pool) end-to-end:

```bash
bash route_balance/exp/route_balance/setup.sh                        # one-time host setup
python route_balance/exp/route_balance/deploy_route_balance.py \
    --host-config route_balance/config/route_balance/host_configs.json \
    --model-config route_balance/config/route_balance/model_deployment.json \
    --deploy-services model_instance predictor monitor scheduler
```

`deploy_route_balance.py` brings up vLLM workers, in-process predictor sidecars, and the scheduler service across all nodes; the orchestrator scripts in `route_balance/exp/route_balance/e2e/` and `reproduce/run_main_table.sh` then drive the per-cell sweeps.

## Datasets

**Model-estimator dataset** (released): 18,608 prompts broadcast across the four Qwen2.5 candidates, drawn from seven public datasets (RewardBench, CodeUltraFeedback, BeaverTails, MixInstruct, LMSYS-Chat-1M, GSM8K, SQuAD). Each entry carries a dataset-specific `reference_text` and six per-(prompt, model) quality signals: embedding similarity, reference similarity, blind LLM-judge, **reference-grounded LLM-judge** (`deepeval-llama3.1-8b-it_reference`), output length, and a 16-bucket length label. Distributed as supplementary material; the full prompt → response → labels generation pipeline lives under `route_balance/predictor/route_balance/offline_training/`.

**Latency-prediction data** (regenerated locally, not redistributed). The deployed XGBoost predictors are trained per (model, GPU)-tier from a *per-instance* schedule-state-to-latency sweep run directly on each model-tier head node — there is no cluster-wide combined dataset to download. The included scripts (`route_balance/predictor/route_balance/offline_training/generate_latency_benchmark.py`, `prepare_xgboost_dataset.py`, `train_xgboost_3model.py`) sweep QPS against a live vLLM endpoint, log the schedule-state features the predictor consumes at run time, and train a separate XGBoost head per tier. Re-running the pipeline on a heterogeneous reservation reproduces the deployed predictors end-to-end without any external data dependency.

## Routing strategies

| Strategy | When you'd pick it |
|---|---|
| `route_balance` (Balance) | Default: $w_q\!=\!w_l\!=\!w_c\!=\!\tfrac{1}{3}$ |
| `route_balance` (Quality, $w_q\!\to\!1$) | Maximize quality at iso-throughput |
| `route_balance` (Latency, $w_l\!\to\!1$) | Tail-latency-sensitive workloads |
| `route_balance` (Cost, $w_c\!\to\!1$) | Cost-sensitive batch workloads |
| `passthrough` + dispatcher | Pure load balancer baseline (no quality awareness) |
| `pipeline` + `RouterBase` plug-in | Drop-in router for AvengersPro / BEST-Route / RouteLLM |

All strategies share the same scheduler, batching path, and predictor infrastructure — only the scoring policy differs.

## Reproducing paper results

See **[REPRODUCE.md](REPRODUCE.md)** for a step-by-step walkthrough of every headline result: the 3-axis Pareto frontier, the scheduling-overhead breakdown, the 4-axis CAP-style radar, the budget-control evaluation, and all ablations (LPT, adaptive batching, predictor co-batching, $w_{\mathrm{bal}}$ sensitivity, optional filters).

The `reproduce/` directory contains the consolidated scripts:

```
reproduce/
├── aggregate_results.py       # Build AGGREGATE_FULL.csv from per-cell .json
├── compute_bootstrap_ci.py    # Bootstrap 95% CIs on the headline λ=12 row
├── make_main_figures.py       # Pareto + 4-axis radar + extremes table
├── make_overhead_figure.py    # Scheduling-overhead breakdown
├── build_budget_dataset.py    # Calibrated budget-control dataset
├── run_main_table.sh          # End-to-end Pareto orchestrator
└── run_full_ablation.sh       # LPT + adaptive + filter + w_bal sweeps
```

## Requirements

- Python 3.10+
- vLLM ≥ 0.6 (model serving)
- PyTorch, transformers, sentence-transformers (`all-MiniLM-L6-v2`)
- XGBoost ≥ 2.0, FAISS-CPU, scikit-learn
- FastAPI, aiohttp, uvicorn (scheduler service)
- DeepEval (off-line quality scoring; not required at serving time)

A single A100/H100 host can run the scheduler + KNN predictor on CPU. Per-instance XGBoost predictors live in-process on the scheduler with `CUDA_VISIBLE_DEVICES=""` set. Off-line quality scoring with DeepEval requires a separate vLLM endpoint serving `Llama-3.1-8B-Instruct` (the judge).

## License

Code: Apache-2.0. Datasets: see Hugging Face dataset cards.
