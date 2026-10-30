# Reproducing the paper's numbers

## The command

Unpack the pack into a path without whitespace. In a new environment with CPython 3.10, made outside the pack:

```bash
cd /path/to/folder-holding-the-pack        # the output folder must lie outside the pack
python3.10 -m venv pack-env && . pack-env/bin/activate
python -m pip install -r fmrlb-paper-data-v1.5/requirements.txt
bash fmrlb-paper-data-v1.5/reproduce_results.sh new-output
```

`PY=<interpreter>` selects another Python. The output folder must not exist and must lie outside the pack: the
script never writes over anything.

## What it does

1. `code/tools/gen_results.py` reads `data/` and writes `generated/results.tex` (every number the text cites, 1,017
   macros), `generated/plot_points.json` (the points the figures draw) and the 16 generated tables. Its guards stop the build when a number the text states in words no
   longer holds, for example a count of beaten settings or the sign of a change.
2. `code/tools/gen_datasets.py table` reads `data/dataset_rows.json` and writes `generated/datasets.tex` (Table A1
   and the four prompt counts the text cites). It stops when a prompt appears twice in a split, or in both the
   training and the evaluation split of a family.
3. The plot scripts draw the six figure files from `plot_points.json`. Each PDF is written without a creation date,
   so the same input gives the same bytes.
4. `verify_results.py compare` compares every output with its reference in `expected/`, byte for byte, and prints
   `PASS: 25 reference outputs match` only when all do.

Tested in a new environment from `requirements.txt` on CPython 3.10.20 (Linux x86-64). The reports' bootstrap
intervals are already in `data/`, so no random draw happens here.

## Where each output appears in the paper

| output | in the paper |
|---|---|
| `generated/results.tex` | every number in the text, through `\res{key}` |
| `tables/gen_cluster.tex` | Table 3, the Qwen cluster |
| `tables/gen_main.tex` | Table 4, the presets against the 41 baseline settings at each rate |
| `tables/gen_overhead.tex` | Table 5, the time a request spends in the scheduler before dispatch |
| `tables/gen_burst.tex` | Table 6, bursty arrivals |
| `tables/gen_budget.tex` | Table 7, per-request budgets |
| `tables/gen_spread.tex` | Table 8, the spread over seeds 1 to 3 |
| `figures/frontier3d.pdf`, `figures/frontier2x6.pdf` | Figure 2, the 3-D frontier above the quality-latency and quality-cost grid |
| `figures/variants.pdf` | Figure 3, F-MRLB against F-decoupled and F-single |
| `figures/frontier_gemma.pdf` | Figure 4, Gemma 3 |
| `generated/datasets.tex` | Table A1, the datasets, and the prompt counts in the text |
| `tables/gen_shipped.tex` | Table A2, the routers as shipped |
| `tables/gen_calib.tex` | Table A3, the TPOT calibration |
| `tables/gen_open.tex` | Table A4, the baseline settings no preset beats |
| `tables/gen_decoupled.tex` | Table A5, the values of Figure 3 |
| `tables/gen_ablation.tex` | Table A6, the ablations |
| `tables/gen_reqnorm.tex` | Table A7, normalization over the round and over each request |
| `tables/gen_roundrange.tex` | Table A8, the round's range at 80/10/10 |
| `tables/gen_gemma.tex` | Table A9, Gemma 3 |
| `tables/gen_vsr.tex` | Table A10, the vLLM Semantic Router |
| `tables/gen_tail.tex` | Table A11, p99 latency |
| `figures/tail_qwen.pdf`, `figures/tail_gemma.pdf` | Figure A1, p99 latency |

Tables 1 and 2 (the comparison with prior work and the notation) and Figure 1 (the architecture) hold no measured
number.

## The boundary

| recomputed here | taken from `data/` as measured |
|---|---|
| every macro, table and figure above | the reports over the runs: per run, its quality, latency percentiles, cost, model mix and overhead, and per baseline setting, its equal-quality comparison with the bootstrap interval |
| the counts and ranges the text states | the seed spreads, the parity record, the vLLM Semantic Router summaries |
| the prompts per dataset and split (Table A1) | each row's source dataset and the sha256 of its prompt text (`data/dataset_rows.json`) |
| | the fleets, the calibration tables and the preregistered rules the runs used |

`data/dataset_rows.json` holds no prompt text: the evaluation prompts come from datasets whose licences restrict
redistribution. Each row is the sha256 of its prompt text and its source dataset, so whoever holds a source dataset
can hash its prompts and find their rows. The training rows are the ones the runs' estimator and routers were built
from: 14,908 Qwen prompts (the 14,919 of the training prompt file the rendered campaign pins, but the 11 with an
answer the judge could not score) and all 8,086 Gemma prompts.

To run the experiments, use the code repository's `docs/reproduce/`.
