# RouteBalance

A serving-layer scheduler for heterogeneous LLM fleets. It makes one decision per request over
concrete model instances, weighing predicted quality, load-aware latency and cost.

A minimal, readable codebase: `CODE_TOUR.md` walks it in reading order, with each file's invariants.

## Layout

| path | contents |
|---|---|
| `routebalance/` | the scheduler: config, telemetry, estimators, scheduling, baselines, server |
| `bench/` | load generator and benchmark client |
| `analysis/` | metrics, aggregation, frontier and equal-quality comparisons |
| `tools/` | offline jobs: calibration, index building, replay, microbenchmark; the engine deployment plan |
| `scripts/` | node setup (`setup.sh`, one node; `setup_fleet.sh`, all nodes), engine and scheduler deployment |
| `configs/` | fleets, engine arguments per tier, example hosts and launch files (real host names live in `local/`) |

## Development

```bash
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
```

The test suite and the campaign gates are kept on the maintainers' local branch and are not part of this
snapshot. Every commit passed them there:
- lint;
- a hygiene gate: no internal names, hosts, IPs, credentials or personal paths;
- a static defect gate: no silent defaults, no experiment parameters from the environment, one GPU map;
- unit tests, and contract tests against a fake vLLM;
- an equivalence suite that compares the rebuild with the banked v3 data (bitwise where v3 was
  deterministic);
- the campaign gates (preflight, parity by declared differences, arm, zero-call), each with MUST-FAIL
  fixtures.

## Estimators (`routebalance/estimate/`)

| module | what it predicts |
|---|---|
| `prompt.py` | the user text inside a chat template (the only unwrap) |
| `knn.py` | per-model quality and output length from the prompt's nearest neighbours (v3 port; `prior="per_model_mean"` is the prompt-blind ablation) |
| `latency.py` | TPOT source (`constant`, `profiled_curve`, `xgboost`) x envelope (`own_length`, `queue_aware`, `batch_aware`): the latency the objective scores |
| `xgboost_tpot.py` | the learned TPOT heads, over what the scheduler knows at serving time under official vLLM |
| `artifacts.py` | md5-pinned reads of every artifact |

`tools/calibrate_tpot_curve.py` writes the constant and curve tables; `tools/retrain_xgb_served.py`
retrains the TPOT heads on serving-time features, from the scheduler's own latency sweep
(`tools/tpot_sweep.py`) or from the v3 sweep, where it also scores the v3 heads with their serving-time
zero-fill (`tools/fork_tpot_features.py`, comparisons only).

## Campaign (`configs/`, `tools/`, `scripts/`)

| file | what it is |
|---|---|
| `configs/presets.json` | every scheduler configuration of the lease, as templates (`tools/campaign.py` documents the format) |
| `configs/scope.json` | every lease row (W1-W11) in the plan's order, with its blocks, groups, traces and datasets |
| `configs/prereg_rules.json` | the preregistered analysis rules, including the G1 gate |
| `tools/campaign.py` | `dry-run` lists every cell, the hours and every value still `OWNER-MUST-SET`; `render` writes the scope, the configuration files and the plan the drivers run from, in stages as lease pins and decisions arrive |
| `tools/calibrate_driver.py` | W1: closed-loop TPOT at fixed concurrency levels, for `tools/calibrate_tpot_curve.py` |
| `tools/slo_base.py` | each tier's single-request TTFT and TPOT from W1 at b = 1: the SLO's bases (`analysis/slo.py`) |
| `tools/judge_served.py` | the budget sweep's judge loop: the texts its cells served go to the DeepEval judge once, and its scores into the served-text score store |
| `tools/engines.py` | each engine as it reports itself (start time, KV cache, model, context) with its deploy.sh launch, read before and after every cell: a restart invalidates the cell, and every cell of a group must share the launches |
| `tools/prompt_set.py` | benchmark prompt files from the scored corpora |
| `scripts/run_cell.sh`, `scripts/run_scope.sh` | one cell end to end (drain, configuration, preflight, direct warm-up, the engines before and after, the window of the scheduler's logs, gates, two locations); the rows of a render in order, resuming |

A cell's quality is the mean DeepEval score of its successful requests, looked up by (prompt, served model) in
the scored corpus, as in v2 and v3 (`analysis/quality.py`). The budget sweep's cells are scored on the text actually
served (a reply cut short by its budget is judged as served, an empty one scores 0; `tools/judge_served.py`). A request meets its SLO when its TTFT and TPOT are
within k times its tier's single-request values (after SOLA; k by model size in `configs/prereg_rules.json`);
goodput is the largest rate that keeps 90% or 99% of requests within it (`analysis/slo.py`).
