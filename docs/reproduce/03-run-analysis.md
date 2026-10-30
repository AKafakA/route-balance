# Stage 3: from the shipped data to the paper's numbers

Every number in the paper reaches it through generated files only: the analysis writes them, and the paper's text
reads them. Nothing is typed in by hand. This stage regenerates them with no GPU and no serving host.

## Route 1: the paper data pack

The pack holds the inputs the paper's numbers are computed from (`data/`), the code that computes them (`code/`),
and every output the paper uses (`expected/`). It needs only CPython 3.10 and the pack's `requirements.txt`. It is
in this repository as `paper/fmrlb-paper-data-v1.5/`, so from the repository's root, in a path without
whitespace:

```bash
cd paper
python3.10 -m venv pack-env && . pack-env/bin/activate
python -m pip install -r fmrlb-paper-data-v1.5/requirements.txt
bash fmrlb-paper-data-v1.5/reproduce_results.sh new-output
```

The script runs the generator and the plot scripts on `data/` into the new folder, and compares every output with
`expected/`: the generated TeX (every macro and table of the paper and its appendix) and the figures. It prints one
line per output and `PASS` only when all match. The pack's `REPRODUCE.md` maps each output to the paper's tables and figures.

## Route 2: from the run records

The runs' full records (each request's timings and served model, the decision logs, the gates' evidence) are the
input of the reports. The paper's four reports were made from them with this repository's code:

```bash
python -m analysis.report --render configs/render36 --cells RUNS/q1 --data-root INPUTS/qwen/data \
  --fleet qwen_std=configs/fleet_qwen.json --hosts HOSTS.json --slo-base qwen=INPUTS/qwen/slo_base_qwen.json \
  --store INPUTS/qwen/budget-judge/store.s1.jsonl INPUTS/qwen/budget-judge/store.s2-zero.jsonl \
  --rules configs/prereg_rules.json --treatment F-MRLB-static --out report-q1
python -m analysis.report --render configs/render36 --cells RUNS/g1 --data-root INPUTS/gemma/data \
  --fleet gemma=configs/fleet_gemma.json --hosts HOSTS.json --slo-base gemma=INPUTS/gemma/slo_base_gemma.json \
  --rules configs/prereg_rules.json --treatment F-MRLB-static --out report-g1
```

and the same over all seeds (`RUNS/qall`, `RUNS/gall`). The records carry the served texts of the budget runs and
are not shipped here. Their parity records are keyed by the source of the gates' folder (`gate_code`
in `tools/cell_driver.py`). This tree's `gates/` holds the run gates with the code the runs executed, its comments
and messages edited, so the key differs and each group's parity reads `not run` until the parity gate is rerun over
the records. Every other value of the reports is the same; only the two header comment lines of `macros.tex` differ.

## The boundary

| recomputed in this stage | taken as measured |
|---|---|
| every macro, table and figure of the paper (route 1) | each run's per-request measurements |
| the reports from the run records (route 2) | the judged score of each (prompt, model) answer |
| | the calibration tables and the SLO bases |

For new measurements, follow [stage 2](02-run-experiments.md) and keep the new runs in their own folders.
