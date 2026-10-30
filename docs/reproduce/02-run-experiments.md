# Stage 2: running the experiments

This stage collects new runs on a fresh cluster. To check the paper's numbers from the runs already measured, use
[stage 3](03-run-analysis.md). The paper's campaign ran for seven days on the 13 nodes of
[stage 1](01-environment.md).

A run is one configuration (a system and its settings) at one trace (an arrival process and a rate) and one seed. The
campaign is the render the runs used, `configs/render36/`. Its `plan.json` lists the rows of runs in the plan's
order, with their comparison groups and fleets, `scope.json` gives every run's configuration, trace and seed, and
`configs/` holds one complete scheduler document per configuration. `tools/campaign.py` documents the plan and
`bench/scope.py` the scope.

## 1. Pin a copy of the render to your cluster

The drivers run from a render. In `configs/render36/`, each row of `plan.json` names its runs and the comparison
groups they belong to, and each group its fleet: the Qwen fleet (`qwen_std`, `configs/fleet_qwen.json`) or the Gemma
fleet (`gemma`, `configs/fleet_gemma.json`). Every run of a row must run on that fleet, and the driver refuses another.

The render pins the deployment the runs used. Each fleet's digest covers each instance's tier, node and port (the
fleet file) and its addresses (the hosts file), as `fleet_digest` in `fmrlb/config.py` computes it, and
`scope.json` pins each configuration's digest. Every driver checks these pins, the calibration and the answer
collection of step 2 included. So for a new cluster, first copy the render:

```bash
cp -r configs/render36 RENDER
```

Then pin the copy to your hosts file. From the repository's root, with the environment of stage 1, this sets the
fleet and configuration digests in place:

```bash
.venv-uvicorn/bin/python - RENDER HOSTS.json <<'EOF'
import json, sys
from pathlib import Path
from fmrlb.config import config_digest, fleet_digest, load_fleet, load_scheduler
render, hosts = Path(sys.argv[1]), Path(sys.argv[2])
fleets = {"qwen_std": Path("configs/fleet_qwen.json"), "gemma": Path("configs/fleet_gemma.json")}
plan = json.loads((render / "plan.json").read_text())
for name, entry in plan["fleets"].items():
    entry["fleet_digest"] = fleet_digest(load_fleet(fleets[name], hosts))
(render / "plan.json").write_text(json.dumps(plan, indent=1) + "\n")
scope = json.loads((render / "scope.json").read_text())
for cid, entry in scope["configs"].items():
    entry["config_digest"] = config_digest(load_scheduler(render / "configs" / f"{cid}.json"))
(render / "scope.json").write_text(json.dumps(scope, indent=1) + "\n")
EOF
```

Run with the runs' own hosts file and unchanged documents, it writes the shipped render back byte for byte. Pass
`--render RENDER` to every command below that takes a render (`tools/calibrate_driver.py`,
`tools/collect_answers.py`, `tools/judge_served.py`, `scripts/run_scope.sh`, `python -m analysis.report` and the
vLLM Semantic Router's drivers), give `deploy.sh scheduler` a document from `RENDER/configs/`, and keep the new runs
in their own runs folder.

## 2. Build the inputs each family needs

A scheduler document names the estimator's index and the latency tables by path, under the artifact root given at
deploy (step 3), and by md5, and the scheduler refuses a file whose md5 differs. It names the routers' artifacts
(AvengersPro's and BEST-Route's) by path, and the scheduler records the md5 of every file it loads in its execution
manifest, which the parity gate compares across a comparison group. Each input is built by a tool of this repository,
and each tool's docstring gives its exact usage. Some are measured on the cluster. The calibration and the answer
collection send to the engines directly, so deploy the engines first (step 3's first command). The latency sweep runs
as rows W1b and W1c (step 4) on a deployed scheduler. Deploy the scheduler (step 3's second command) once the inputs
its document loads are built and pinned. With `--fresh-scheduler`, every run restarts it from the deployed document,
so after a later re-pin of an artifact that document names, stop it
(`scripts/stop_scheduler.sh --fleet F --hosts H --node N`), deploy it again with a new `--log-dir`, and pass the new
run directory to step 4's `--fresh-scheduler`.

| input | built by |
|---|---|
| the benchmark prompt files the plan pins (`datasets`) | `tools/prompt_set.py`, from a corpus's prompts: the evaluation set, and the training split, which it refuses if it holds an evaluation prompt (`--exclude`); or the four files the runs sent, byte for byte as pinned, from `data/prompt_files/` (`tools/prompt_files.py build`, below) |
| the TPOT table and the prefill table | row W1 of the plan (W8-cal for Gemma), closed-loop TPOT at fixed batch sizes on every tier (`tools/calibrate_driver.py`), then `tools/calibrate_tpot_curve.py` and `tools/fit_prefill.py` |
| the SLO bases | `tools/slo_base.py`, each tier's single-request TTFT and TPOT from the same calibration |
| the training rows, the k-NN indexes (the ablations' too), BEST-Route's data and AvengersPro's artifacts | `scripts/build_inputs.sh` on a family's judged training corpus: `tools/router_training.py` (`filter`, `best-route-data`, `avengers-pro`), `tools/best_route_upstream.py`, `tools/knn_index.py` and `tools/row_subsets.py` |
| BEST-Route's checkpoint | `scripts/train_best_route.sh` on `build_inputs.sh`'s `br_upstream/`: upstream best-route-llm's trainer, at the commit `scripts/setup.sh` pins, with the runs' arguments; the checkpoint folder gets the family's labels file from `configs/best_route/` |
| the learned TPOT heads (the learned variant only) | `tools/retrain_xgb_served.py`, on the latency sweep the scheduler itself logs |
| the scored corpora | every model's answer to each prompt (`tools/collect_answers.py collect`, directly on the engines, then `merge`: the unscored corpus) and the G-Eval judge's score of each answer: the judges deployed as a fleet (`configs/fleet_judge.json`), the corpus cut into shards of 200 rows (`tools/judge_shards.py`), scored by `scripts/judge_pool.sh` (one worker per judge, each shard by `scripts/judge_one.sh` running `tools/geval_score.py`, `--max-concurrent 64 --chunk-size 200`) and joined back in shard order (`scripts/judge_assemble.sh`, checked by `tools/judge_validate.py`); the few pairs the judge leaves unscored are judged again on one judge (`tools/judge_gapfill.py`, on a copy of the joined corpus; it tags each pair it fills by temperature `_judge_temperature`, which `tools/judge_validate.py` counts as a changed field, so validate a copy without the tags) |

The quality and length estimator's index is built by `tools/knn_index.py`, the builder the paper's runs used, over the
training rows, and the scheduler checks its md5 when it loads it (`fmrlb/estimate/knn.py`). The embeddings' last
bits depend on the machine, so a rebuilt index has another md5 and is pinned again (below): an index rebuilt from the
paper's Gemma rows on another machine holds the same labels and lengths, and the same 50 nearest neighbours for all but
one of its 8,086 prompts. The scheduler documents under `configs/render36/configs/` name the artifacts the paper's runs
used.

The training prompt files the plan pins (`train_qwen`, 14,919 rows, and `train_gemma`, 8,086) hold the whole training
split, which the latency sweep sends. `filter` keeps a prompt only if every model's answer to it has a judge score:
14,908 of the Qwen prompts (11 have an answer the judge could not score) and all 8,086 Gemma prompts. Those rows build
the estimator's index and the routers' artifacts, and they are the training counts of the paper's Table A1
(`paper/fmrlb-paper-data-v1.5/data/dataset_rows.json`).

### Reusing the paper's labels

`data/labels/` holds the paper's judged labels without any text (`tools/labels.py extract`): for every retained training
row and every evaluation prompt of each family, its public source (`source`, the row's id in its dataset), the sha256 of
its prompt and of its reference answer, and every model's output length and judged score. `data/prompt_files/` holds
the four prompt files the runs sent the same way: per line its id and its prompt's sha256 and dataset, and the file's
pin. `tools/fetch_prompts.py` fetches the prompts back from their public datasets with the loaders that built the corpus,
each checked against its sha256 (all 18,505 are found), in an environment of its own, and `tools/prompt_files.py build`
writes each prompt file again, refusing any whose sha256 is not the plan's (all four are as pinned):

```bash
python -m tools.fetch_prompts --labels data/labels/*.jsonl data/prompt_files/*.jsonl --out PROMPTS.jsonl \
  --references-out REFS.jsonl
python -m tools.prompt_files build --manifest data/prompt_files/qwen_train_prompts.jsonl --prompts PROMPTS.jsonl \
  --out qwen_train_prompts.jsonl
```

- **Without judging again.** `tools/labels.py join` rebuilds the rows from the labels and the prompts: the training
  labels give the rows `scripts/build_inputs.sh --no-best-route` builds the k-NN indexes and AvengersPro from, and the
  evaluation labels give the scored corpora the analysis reads, each equal to the paper's in every prompt, output
  length and score. BEST-Route's data holds the answers' texts, which are not shipped: it needs the answers collected.

  ```bash
  python -m tools.labels join --labels data/labels/qwen_train.jsonl --prompts PROMPTS.jsonl --out qwen_train_rows.jsonl
  python -m tools.labels join --labels data/labels/qwen_eval.jsonl --prompts PROMPTS.jsonl --out qwen_eval_scored.jsonl
  ```

- **Judging new answers.** The judge compares each answer with the prompt's reference answer, which `--references-out`
  rebuilds as the corpus built it, each checked against its sha256. Joined with `--references`, the labels give the
  corpus `tools/collect_answers.py merge` starts from (`--corpus`); the collected answers merged into it are judged as
  the table above describes.

  ```bash
  python -m tools.labels join --labels data/labels/qwen_train.jsonl --prompts PROMPTS.jsonl --references REFS.jsonl \
    --out qwen_train_corpus.jsonl
  ```

When a rebuilt input replaces one the render pins, set its pin in `RENDER` by hand: an artifact's md5 in the
scheduler documents that name it, a prompt file's sha256 and row count (and a budget table's sha256) in every
`datasets` entry that names it, in both `plan.json` and `scope.json`, and a scored corpus's sha256 in `plan.json`
(`scored_corpora`). Then run the pinning command of step 1 again (not the copy), which recomputes the fleet and
configuration digests.

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
same for every system. They must equal the plan's `server` values (`torch_threads` null, `max_connections` 4096,
`request_timeout_s` 21600), and the driver refuses a scheduler launched with others before every run
(`tools/cell_driver.py check-scheduler`). A running engine, a busy port or a busy GPU is refused, never killed. The engines' record
names the process each engine runs as, and every run is checked against it.

## 4. Run the rows

This step runs on the scheduler's node, from its copy of this repository (step 3's `--tree`). Copy `RENDER`, the
hosts file and `ENGINES.record` there first, and again after any re-pin (`RENDER`) or engine deploy
(`ENGINES.record`): the node setup copies only the repository's tracked files, and `deploy.sh` only the files of its
own run. Then, with `PY` naming the scheduler's environment there:

```bash
PY=<scheduler venv>/bin/python scripts/run_scope.sh --render RENDER --rows W2 --max-invalid 3 --fresh-scheduler <scheduler run dir> \
  --scheduler http://127.0.0.1:<port> --scheduler-log-dir <its log dir> --fleet configs/fleet_qwen.json \
  --hosts HOSTS.json --engines-record ENGINES.record --data-root <data dir> --cells-root <runs dir> \
  --mirror <second location>
```

`--fresh-scheduler` takes the deployed scheduler's run directory, `run_dir` in the first line of `SCHEDULER.record`.

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
