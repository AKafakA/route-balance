# RouteBalance

A serving-layer scheduler for heterogeneous LLM fleets. For each request it picks a (model, instance) pair, weighing
predicted quality, load-aware latency and cost, and it schedules requests in rounds so that the requests of a round
see each other's assignments.

`docs/reproduce/` reproduces the paper: its environment, the runs, and every number, table and figure from the
shipped data. `paper/` holds the paper data pack: the data the paper's numbers are computed from, the generator and
the plot scripts, and every output the paper uses, with a script that rebuilds and compares them on a CPU.

## Layout

| path | contents |
|---|---|
| `routebalance/` | the scheduler: configuration, telemetry, estimators, scheduling rounds, the baselines, the server |
| `bench/` | the open-loop load generator and benchmark client, and the per-run record |
| `analysis/` | per-run metrics, the run manifest, frontiers, the equal-quality comparison, the SLO, the report |
| `gates/` | the gates that decide whether a run counts (`docs/reproduce/gates.md`) |
| `tools/` | the drivers of a run and of the calibration, the learned heads, router training, judging, the engine plan |
| `scripts/` | node setup, engine and scheduler deployment, one run and a row of runs, building the inputs, the judge pool, BEST-Route's training |
| `configs/` | the fleets, the engine arguments per tier, the scheduler's launch values, the preregistered rules, the rendered campaign the runs used (`render36/`), the CloudLab profile (`cloudlab/`) and BEST-Route's labels files and training environment (`best_route/`) |
| `data/labels/`, `data/prompt_files/` | the paper's judged labels and output lengths per prompt, and the four prompt files the runs sent, without any text (`tools/labels.py`, `tools/prompt_files.py`; `tools/fetch_prompts.py` fetches the prompts and reference answers back from the public datasets) |
| `vllm-sr-bench/` | the vLLM Semantic Router, deployed as its authors deploy it, as its own stack in front of the same engines, and driven by the same client |
| `paper/` | the paper data pack (`routebalance-paper-data-v1.5/`): `data/`, `code/tools/` (the generator of every number and table, the plot scripts of every figure), `expected/`, `reproduce_results.sh` |

## Quick start

```bash
uv venv --python 3.10 .venv-uvicorn
uv pip install --python .venv-uvicorn/bin/python -e .
```

Then follow [`docs/reproduce/`](docs/reproduce/README.md). Its stage 3 rebuilds every number, table and figure of the
paper from the shipped data, on a CPU, with no run needed:

```bash
cd paper
python3.10 -m venv pack-env && . pack-env/bin/activate
python -m pip install -r routebalance-paper-data-v1.5/requirements.txt
bash routebalance-paper-data-v1.5/reproduce_results.sh new-output      # ends in: PASS: 25 reference outputs match
```

## Estimators (`routebalance/estimate/`)

| module | what it predicts |
|---|---|
| `prompt.py` | the user text inside a chat template (the only unwrap) |
| `knn.py` | per-model quality and output length from the prompt's nearest neighbours (`prior="per_model_mean"` is the prompt-blind ablation) |
| `latency.py` | a TPOT source (`constant`, `profiled_curve`, `xgboost`) times an envelope (`own_length`, `queue_aware`, `batch_aware`), plus the prompt's prefill when configured: the latency the objective scores |
| `xgboost_tpot.py` | the learned TPOT heads, over what the scheduler knows at serving time under official vLLM |
| `artifacts.py` | the md5-checked read of every artifact a configuration pins by md5, and the md5 of every file of an artifact directory loaded by path (`md5_tree`, recorded in the execution manifest) |

## The campaign (`configs/`, `tools/`, `scripts/`)

| file | what it is |
|---|---|
| `configs/render36/` | the rendered campaign the runs used: `plan.json`, the rows of runs in the plan's order with their groups, fleets and launch values (`tools/campaign.py` documents it), `scope.json`, every run's configuration, trace and seed with its block and comparison group (`bench/scope.py`), and one complete scheduler document per configuration under `configs/` |
| `configs/prereg_rules.json` | the preregistered analysis rules: the equal-quality tolerance, the bootstrap, the SLO multipliers |
| `tools/calibrate_driver.py` | closed-loop TPOT at fixed batch sizes on every tier, for `tools/calibrate_tpot_curve.py` |
| `tools/slo_base.py` | each tier's single-request TTFT and TPOT, the bases of the SLO (`analysis/slo.py`) |
| `tools/judge_served.py` | the budget runs' judge loop: the texts they served go to the DeepEval judge once |
| `tools/engines.py` | each engine as it reports itself, read before and after every run: a restart invalidates the run |
| `tools/router_training.py` | the baselines' training inputs, built for both model families: AvengersPro's artifacts and BEST-Route's candidates |
| `tools/knn_index.py`, `tools/row_subsets.py` | the estimator's k-NN index, built from the training rows, and the rows behind the ablations' indexes |
| `tools/best_route_upstream.py`, `scripts/train_best_route.sh` | BEST-Route's data in upstream's format, and its ranker trained by upstream's trainer |
| `tools/collect_answers.py` | every model's answer to every prompt of a prompt set, directly on the engines |
| `tools/judge_shards.py`, `scripts/judge_pool.sh`, `tools/geval_score.py`, `scripts/judge_assemble.sh`, `tools/judge_gapfill.py` | the G-Eval judging of the collected answers on the judge fleet (`configs/fleet_judge.json`) |
| `scripts/build_inputs.sh` | every estimator and router input of a family from its judged training corpus |
| `scripts/run_cell.sh`, `scripts/run_scope.sh` | one run end to end, and the rows of a render in order, resuming |

A run's quality is the mean DeepEval score of its successful requests, looked up by (prompt, served model) in the
scored corpus (`analysis/quality.py`). The budget runs are scored on the text actually served. A request meets its
SLO when its TTFT and TPOT are within k times its tier's single-request values (after SOLA, with k by model size in
`configs/prereg_rules.json`), and goodput is the largest rate that keeps 90% or 99% of requests within it
(`analysis/slo.py`).
