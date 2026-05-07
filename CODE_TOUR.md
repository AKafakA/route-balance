# RouteBalance — Code Tour

A reading guide for paper review. Files marked **★** are the conceptual core
of the contribution; everything else is plumbing or baselines.

## Top-level layout

```
route_balance/
├── global_scheduler/       # Scheduler core: routes prompts → backends
│   ├── route_balance/
│   │   ├── route_balance_serve.py    ★ entry point (HTTP API + main loop)
│   │   ├── filters/                  ★ SLO admission filters
│   │   ├── routers/                  L1 routers (model selection)
│   │   ├── dispatch/                 L2 dispatchers (instance selection)
│   │   ├── route_balance_instance/   vLLM/Ollama backend wrappers
│   │   └── utils.py
│   ├── api_server.py                 OpenAI-compatible adapter
│   └── instance.py                   shared backend abstraction
├── predictor/route_balance/          # Latency + quality predictors
│   ├── route_balance_predictor_api_server.py   XGBoost HTTP sidecar
│   ├── route_balance_learned_predictor.py      client + caching layer
│   ├── route_balance_predictor_config.py       config schema
│   ├── model_estimator.py            ★ quality + length predictor (RoBERTa)
│   ├── estimators/                   per-target sub-models
│   └── offline_training/             reproducibility scripts (one-off)
├── benchmark/route_balance/          # Bench client
│   ├── benchmark_serving.py          OpenAI/RouteBalance bench
│   ├── route_balance_end_point_func.py   request shape per backend
│   └── dataset.py                    HF prompt loader
├── exp/route_balance/                # Cluster deployment + automation
│   ├── deploy_route_balance.py       ★ 18-node cluster orchestrator
│   ├── setup.sh                      CloudLab provisioning (P100 patches noted)
│   ├── e2e/                          end-to-end smoke harness
│   └── csd3/, archive/               training-side scripts (skip for review)
├── config/route_balance/             # JSON configs
│   ├── model_deployment_apr26_phase2.json   18-node hetero topology
│   ├── scheduler_config_smoketest.json      scoring weights, SLO defaults
│   └── route_balance_predictor_config.json
├── route_balance_pd/                 # P-D disaggregation (paper Appendix §A.2)
│   └── route_balance_pd_serve.py
└── route_balance_paper/                       # Paper docs + sweep scripts (kept as-is for provenance)
    ├── codex/                        design-of-record + decisions
    ├── handoffs/                     daily session logs
    └── smoke_test_apr_13/scripts/    sweep automation (full_sweep_overnight.sh,
                                      high_lambda_sweep.sh, phase_2_*, sensitivity_*,
                                      route_balance_orchestrator.sh, aggregate_with_deepeval.py)
```

## Suggested reading order (≈90 min)

### 1. Paper-side novelty (30 min) ★

These are the "what is RouteBalance doing differently" files:

```
route_balance/global_scheduler/route_balance/filters/route_balance_cdf_filter.py
route_balance/global_scheduler/route_balance/filters/base.py
route_balance/global_scheduler/route_balance/filters/factory.py
```

The headline contribution: empirical-CDF SLO admission with tiered relaxation.
Compare against `qlm_filter.py` (Normal-bound point filter), `slos_serve_filter.py`
(point-prediction binary), `polyserve_filter.py` (cumulative deadline),
`timebill_filter.py` (cost-of-time formulation).

### 2. Scheduler entry + scoring (30 min) ★

```
route_balance/global_scheduler/route_balance/route_balance_serve.py
```

The big file (2,350 lines). Module docstring at top explains the modes.
Key sections:
- Lines 1–90: imports, module state, scheduling counters
- ~Line 380–460: per-request route through pipeline / route_balance / fallback
- Search for `def _select_instance` — the per-request strategy selector
  (now raises on unknown strategy, no silent random fallback)
- Search for `def _batch_scheduler_loop` — the route_balance batch path:
  ModelEstimator → CDF filter → multi-objective scoring → LPT assignment
- Search for `def _load_scheduler_config` — POST /v1/config handler
- Bottom of file: argparse + startup validation (lines ~2180+)

### 3. Routers + dispatchers (15 min)

```
route_balance/global_scheduler/route_balance/routers/factory.py
route_balance/global_scheduler/route_balance/routers/route_balance_native.py
route_balance/global_scheduler/route_balance/routers/{routellm,avengers_pro,
                                                       best_route_wrapper,
                                                       vllm_sr}.py
route_balance/global_scheduler/route_balance/dispatch/factory.py
route_balance/global_scheduler/route_balance/dispatch/{round_robin,
                                                       shortest_queue,
                                                       llumnix_minus,
                                                       route_balance_native}.py
```

Routers are baselines (RouteLLM-mf, Avengers-Pro, BEST-Route-wrapper, vLLM-SR)
plus our `route_balance_native` (multi-objective argmax). Dispatchers are L2
instance selection (round_robin / shortest_queue / multi-objective).

### 4. Predictors (15 min)

```
route_balance/predictor/route_balance/model_estimator.py        ★ quality + length
route_balance/predictor/route_balance/route_balance_predictor_api_server.py
route_balance/predictor/route_balance/route_balance_learned_predictor.py
```

- `model_estimator.py` selects best target model + predicts length bucket
  per prompt (RoBERTa fused, runs on CPU per `CUDA_VISIBLE_DEVICES=""`).
- `route_balance_predictor_api_server.py` is the XGBoost TTFT/TPOT/E2E sidecar
  (one HTTP server per vLLM instance, queried by the scheduler).

## What to skip on first pass

- `predictor/route_balance/offline_training/` — 60+ training utility scripts.
  One-off data prep / HP sweep / dataset prep. Not paper-side critical;
  reproducibility evidence only.
- `route_balance_pd/` — P-D disaggregation (paper Appendix §A.2 only).
- `exp/route_balance/archive/` — historical CSD3 jobs (180 stale `route_balance` refs
  intentionally preserved for provenance).
- `exp/route_balance/csd3/` — predictor-training sbatch scripts; auxiliary.

## Reproducibility entry points

For a reviewer to reproduce the headline figure (quality-vs-throughput Pareto
at λ=10):

1. Provision an 18-node CloudLab heterogeneous cluster — see `exp/route_balance/setup.sh`.
2. Deploy 18 vLLM backends + sidecars — `exp/route_balance/deploy_route_balance.py`.
3. Launch the scheduler — `bin/launch_route_balance_serve_apr26.sh route_balance <tag> dummy`
   (CPU-mode wrapper; sets `CUDA_VISIBLE_DEVICES=""` because A100 nodes are full).
4. Run the bench: `route_balance_paper/smoke_test_apr_13/scripts/phase_2_main_route_balance_3634.sh`.
5. Run wrapper baselines: same script but with router restarts to pipeline mode
   (planned; see handoff for status).
6. Aggregate: `route_balance_paper/smoke_test_apr_13/scripts/aggregate_with_deepeval.py`.
