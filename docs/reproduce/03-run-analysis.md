# Stage 3: from the shipped data to the paper's numbers

Every number in the paper reaches it through generated files only: the analysis writes them, and the paper's text
reads them. Nothing is typed in by hand. This stage regenerates them with no GPU and no serving host.

## Route 1: the paper data pack

The pack holds the inputs the paper's numbers are computed from (`data/`), the code that computes them (`code/`),
and every output the paper uses (`expected/`). It needs only CPython 3.10 and the pack's `requirements.txt`. Unpack
it into a path without whitespace, then:

```bash
python3.10 -m venv pack-env && . pack-env/bin/activate
python -m pip install -r requirements.txt
bash reproduce_results.sh /path/to/new-output
```

The script checks every file of the pack against `MANIFEST.sha256`, then runs the generator and the plot scripts
on `data/` into the new folder, and compares every output with `expected/`: the generated TeX (every macro and
table of the paper and its appendix) and the figures. It prints one line per output and `PASS` only when all
match. The pack's `REPRODUCE.md` maps each output to the paper's tables and figures.

## Route 2: the meta-analysis pack

The meta-analysis pack holds the four reports this repository's `analysis/report.py` wrote over the runs (Qwen and
Gemma, the first seed and all seeds), one row per run for all 1,457 runs (configuration, group, trace, seed,
quality, latency percentiles, cost, model mix, overhead stages), the rendered campaign, the fleets, the rules, the
SLO bases and the calibration tables. Its `REPRODUCE.md` rebuilds the paper data pack's `data/` from the reports,
and the check is that the result equals that `data/` byte for byte.

## Route 3: from the run records

The runs' full records (each request's timings and served model, the decision logs, the gates' evidence) are the
input of the reports. The paper's four reports were made from them with this repository's code:

```bash
python -m analysis.report --render configs/render36 --cells RUNS/q1 --data-root INPUTS/qwen/data \
  --fleet qwen_std=configs/fleet_qwen.json --hosts HOSTS.json --slo-base qwen=INPUTS/qwen/slo_base_qwen.json \
  --store INPUTS/qwen/budget-judge/store.s1.jsonl INPUTS/qwen/budget-judge/store.s2-zero.jsonl \
  --rules configs/prereg_rules.json --treatment RouteBalance-static --out report-q1
python -m analysis.report --render configs/render36 --cells RUNS/g1 --data-root INPUTS/gemma/data \
  --fleet gemma=configs/fleet_gemma.json --hosts HOSTS.json --slo-base gemma=INPUTS/gemma/slo_base_gemma.json \
  --rules configs/prereg_rules.json --treatment RouteBalance-static --out report-g1
```

and the same over all seeds (`RUNS/qall`, `RUNS/gall`). The records carry the served texts of the budget runs and
are not part of the reviewer packs. Their parity records are keyed by the source of the gates' folder (`gate_code`
in `runner/store.py`). This tree's `gates/` holds the run gates with the code the runs executed, its comments
edited, and not the repository checks of development, so the key differs and each group's parity reads `not run`
until the parity gate is rerun over the records. Every other value of the reports is the same.

## The boundary

| recomputed in this stage | taken as measured |
|---|---|
| every macro, table and figure of the paper (route 1) | each run's per-request measurements |
| the paper's inputs from the reports (route 2) | the judged score of each (prompt, model) answer |
| the reports from the run records (route 3) | the calibration tables and the SLO bases |

For new measurements, follow [stage 2](02-run-experiments.md) and keep the new runs in their own folders.
