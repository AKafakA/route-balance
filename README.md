# RouteBalance

A serving-layer scheduler for heterogeneous LLM fleets. For each request it picks a (model, instance) pair, weighing
predicted quality, load-aware latency and cost, and it schedules requests in rounds so that the requests of a round
see each other's assignments.

`CODE_TOUR.md` walks the code in the order a request takes through it, with each file's invariants.
`docs/reproduce/` reproduces the paper: its environment, the runs, and every number, table and figure from the
shipped data.

## Layout

| path | contents |
|---|---|
| `routebalance/` | the scheduler: configuration, telemetry, estimators, scheduling rounds, the baselines, the server |
| `bench/` | the open-loop load generator and benchmark client, and the per-run record |
| `analysis/` | per-run metrics, the run manifest, frontiers, the equal-quality comparison, the SLO, the report |
| `gates/` | the gates that decide whether a run counts (`docs/reproduce/gates.md`) |
| `tools/` | the campaign render and drivers, calibration, the learned heads, router training, judging, the engine plan |
| `scripts/` | node setup, engine and scheduler deployment, one run and a row of runs |
| `configs/` | the fleets, the engine arguments per tier, the presets, the campaign scope, the preregistered rules, and the source of every value (`provenance.json`) |
| `vllm-sr-bench/` | the vLLM Semantic Router, deployed as its authors deploy it, as its own stack in front of the same engines, and driven by the same client |
| `tests/` | unit tests, contract tests against a fake vLLM, and the gates' tests |

## Quick start

```bash
uv venv --python 3.10 .venv-uvicorn
uv pip install --python .venv-uvicorn/bin/python -e ".[dev]"
make check
```

`make check` runs offline, on the CPU: lint, the hygiene gate (no host names, private addresses, credentials or
personal paths), the static defect gate (no invented defaults, no experiment parameter from the environment, one
GPU map), the unit and contract tests, and the gates' tests with fixtures that must fail.

## Estimators (`routebalance/estimate/`)

| module | what it predicts |
|---|---|
| `prompt.py` | the user text inside a chat template (the only unwrap) |
| `knn.py` | per-model quality and output length from the prompt's nearest neighbours (`prior="per_model_mean"` is the prompt-blind ablation) |
| `latency.py` | a TPOT source (`constant`, `profiled_curve`, `xgboost`) times an envelope (`own_length`, `queue_aware`, `batch_aware`), plus the prompt's prefill when configured: the latency the objective scores |
| `xgboost_tpot.py` | the learned TPOT heads, over what the scheduler knows at serving time under official vLLM |
| `artifacts.py` | md5-pinned reads of every artifact |

## The campaign (`configs/`, `tools/`, `scripts/`)

| file | what it is |
|---|---|
| `configs/presets.json` | every scheduler configuration of the campaign, as templates (`tools/campaign.py` documents the format) |
| `configs/scope.json` | every row of runs in the order they ran, with its blocks, comparison groups, traces and datasets |
| `configs/prereg_rules.json` | the preregistered analysis rules: the equal-quality tolerance, the bootstrap, the SLO multipliers |
| `tools/campaign.py` | `dry-run` lists every row and block with its number of runs, the estimated hours and every value still to be set, and `render` writes the files the drivers run from |
| `tools/calibrate_driver.py` | closed-loop TPOT at fixed batch sizes on every tier, for `tools/calibrate_tpot_curve.py` |
| `tools/slo_base.py` | each tier's single-request TTFT and TPOT, the bases of the SLO (`analysis/slo.py`) |
| `tools/judge_served.py` | the budget runs' judge loop: the texts they served go to the DeepEval judge once |
| `tools/engines.py` | each engine as it reports itself, read before and after every run: a restart invalidates the run |
| `tools/router_training.py` | the baselines' training inputs, built for both model families: AvengersPro's artifacts and BEST-Route's candidates |
| `scripts/run_cell.sh`, `scripts/run_scope.sh` | one run end to end, and the rows of a render in order, resuming |

A run's quality is the mean DeepEval score of its successful requests, looked up by (prompt, served model) in the
scored corpus (`analysis/quality.py`). The budget runs are scored on the text actually served. A request meets its
SLO when its TTFT and TPOT are within k times its tier's single-request values (after SOLA, with k by model size in
`configs/prereg_rules.json`), and goodput is the largest rate that keeps 90% or 99% of requests within it
(`analysis/slo.py`).
