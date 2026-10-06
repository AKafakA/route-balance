# Stage 2: running the experiments

This stage collects new runs on a fresh cluster. To check the paper's numbers from the runs already measured, use
[stage 3](03-run-analysis.md). The paper's campaign ran for seven days on the 13 nodes of
[stage 1](01-environment.md).

A run is one configuration (a system and its settings) at one trace (an arrival process and a rate) and one seed. The
campaign is two files: `configs/presets.json` holds every configuration as templates, and `configs/scope.json` every
row of runs in the order they ran, with their comparison groups, traces, datasets and seeds. `tools/campaign.py`
documents both formats.

## 1. Build the inputs each family needs

Every artifact a configuration loads is pinned by md5 in the render (step 2), and the scheduler refuses one whose md5
differs. Each is built by a tool of this repository, and each tool's docstring gives its exact usage:

| input | built by |
|---|---|
| the TPOT table and the prefill table | row W1 of the scope, closed-loop TPOT at fixed batch sizes on every tier (`tools/calibrate_driver.py`), then `tools/calibrate_tpot_curve.py` and `tools/fit_prefill.py` |
| the SLO bases | `tools/slo_base.py`, each tier's single-request TTFT and TPOT from the same calibration |
| the baselines' training data and AvengersPro's artifacts | `tools/router_training.py` (`filter`, `avengers-pro`, `best-route-data`) |
| BEST-Route's checkpoint | upstream best-route-llm's training, at the commit `scripts/setup.sh` pins, on `best-route-data`'s output |
| the learned TPOT heads (the learned variant only) | `tools/retrain_xgb_served.py`, on the latency sweep the scheduler itself logs |
| the scored corpora | each prompt with every model's answer (`tools/collect_answers.py`, directly on the engines) and its judged score |

The quality and length estimator's index is the earlier implementation's KNN index format, built over the same
training rows. This repository loads it and checks its md5 (`routebalance/estimate/knn.py`). The artifacts the
paper's runs used are listed, with their md5, in the rendered campaign's `render.json` (the meta-analysis pack).

## 2. Render the campaign

```bash
python -m tools.campaign dry-run --presets configs/presets.json --scope configs/scope.json
python -m tools.campaign render --presets configs/presets.json --scope configs/scope.json --out RENDER \
  --pins PINS.json --decisions DECISIONS.json --fleet qwen_std=configs/fleet_qwen.json \
  --fleet gemma=configs/fleet_gemma.json --hosts HOSTS.json
```

The dry run lists every row and block with its number of runs, the estimated hours, and every value still to be
set. A render writes
`scope.json`, one scheduler document per configuration under `configs/`, and `plan.json`, the rows with their runs,
groups and fleets, which the drivers run from. A block whose inputs are not given yet is left out and listed as
pending, so a render with pending blocks can never be called complete.

## 3. Deploy

```bash
export PY=.venv-uvicorn/bin/python        # the operator's environment (stage 1)
scripts/deploy.sh engines --fleet configs/fleet_qwen.json --hosts HOSTS.json \
  --engine-args configs/engine_args_qwen.json --record ENGINES.record
scripts/deploy.sh scheduler --fleet configs/fleet_qwen.json --hosts HOSTS.json --node <scheduler node> \
  --config RENDER/configs/<any configuration>.json --launch LAUNCH.json --artifacts <artifact root on the node> \
  --tree <repository on the node> --log-dir <new log dir on the node> --record SCHEDULER.record
```

`LAUNCH.json` holds the scheduler's launch values (`configs/scheduler_launch.example.json` documents them), the
same for every system. A running engine, a busy port or a busy GPU is refused, never killed. The engines' record
names the process each engine runs as, and every run is checked against it.

## 4. Run the rows

On the scheduler's node, with `PY` naming the scheduler's environment there:

```bash
PY=<scheduler venv>/bin/python scripts/run_scope.sh --render RENDER --rows G1,W2 --max-invalid 3 --fresh-scheduler <scheduler run dir> \
  --scheduler http://127.0.0.1:<port> --scheduler-log-dir <its log dir> --fleet configs/fleet_qwen.json \
  --hosts HOSTS.json --engines-record ENGINES.record --data-root <data dir> --cells-root <runs dir> \
  --mirror <second location>
```

Each run (`scripts/run_cell.sh`) drains the engines and the scheduler, applies the run's configuration, checks
the fleet and the scheduler (preflight), reads every engine as it reports itself, with its serving counters, then
warms the engines up directly while emptying every engine's prefix cache and proving it, so the counters cover the
reset and the whole run. It then runs the benchmark client, reads the engines again, cuts the run's window out of
the scheduler's decision log, runs the gates and writes the verdict. With
`--fresh-scheduler`, every run starts on a new scheduler process. A run already published is never run again.
Each run is published under `--cells-root` and copied to `--mirror`, and the copy is checked file by file.

A row runs on one fleet. Switching the model family is a redeploy (step 3), then the other family's rows.

The vLLM Semantic Router runs outside this scheduler, as its authors deploy it, with the same client and the same
records: `vllm-sr-bench/README.md`.

## 5. Check the runs

A run counts only when its verdict is valid and its group's parity gate has passed over all the group's runs
([`gates.md`](gates.md)). `python -m analysis.report` exits 0 only when every expected run is valid or accounted
for, nothing is pending and every group's parity has passed, so its exit status says whether the campaign is
complete. Then go to [stage 3](03-run-analysis.md).
