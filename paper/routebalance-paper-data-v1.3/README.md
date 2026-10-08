# RouteBalance: the paper data pack

This pack regenerates every number, table and figure of the paper from the data the paper reads, with the code that
reads it, and checks each regenerated file against the one the paper uses, byte for byte. It needs only CPython 3.10
and the pinned packages of `requirements.txt`: no GPU, no serving host, no TeX installation.

```bash
cd /path/to/folder-holding-the-pack        # not inside the pack: its manifest admits no extra file
python3.10 -m venv pack-env && . pack-env/bin/activate
python -m pip install -r routebalance-paper-data-v1.3/requirements.txt
bash routebalance-paper-data-v1.3/reproduce_results.sh new-output
```

The last line printed is `PASS: 24 reference outputs match`. [`REPRODUCE.md`](REPRODUCE.md) explains the steps and
maps every output to the paper.

## Layout

| path | contents |
|---|---|
| `data/` | the inputs of the paper's numbers: the analysis reports over the runs, the seed spreads, the parity record, the vLLM Semantic Router summaries, the fleets, the calibration tables and the rules (`DATA_FORMAT.md`) |
| `code/tools/` | the generator of every number and table (`gen_results.py`) and the plot scripts |
| `code/analysis/`, `code/bench/`, `code/gates/`, `code/routebalance/` | the modules of the RouteBalance code the generator imports (the equal-quality reading and the frontier) |
| `expected/` | the outputs the paper uses: `generated/` (every macro the text cites, and the points the figures draw), `tables/`, `figures/` |
| `reproduce_results.sh`, `verify_results.py` | the entry point and its checks |
| `MANIFEST.sha256` | the sha256 of every file of the pack but itself |
| `INDEX.md`, `CONTENTS.tsv` | every file with its role, and every file but `CONTENTS.tsv`, `INDEX.md` and `MANIFEST.sha256` with its size and sha256 |
| `CODE_INPUTS.sha256` | every code file with the source it was copied from |
| `PROVENANCE.json` | where each part came from: the paper version, the code commit, the runs and the report commands |

## The boundary

The pack recomputes every number, table and figure from `data/`. It takes as measured what `data/` holds: the
reports `analysis/report.py` wrote over the 1,457 runs, and the calibration the runs used. The meta-analysis pack
holds one row per run, rebuilds the seed spreads, the round shares and the round's range of `data/`, and checks 13 of its 16 files by
sha256 against its own. The code repository's `docs/reproduce/` describes the
runs themselves.

`expected/generated/datasets.tex` (the paper's Table A1) is shipped as built: it counts the evaluation prompts and
the estimator's training rows, which carry prompt texts. The prompts come from datasets whose licences restrict
redistribution, so no file of this pack holds a prompt or an answer text from them.
