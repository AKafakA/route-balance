# What this branch removed

The development tree held 390 tracked files. This branch keeps the code that serves, measures and analyses the
runs, the tests and gates that verify it, and the configurations the runs used. It drops 34 files (3,443 lines).

## How the keep-set was derived

A file is kept if any of these holds:

1. it is reachable by Python import from an entry point of stage 2 or stage 3 (the server, the benchmark client,
   the campaign tools and drivers, the gates, the analysis).
2. it is named by something kept: a script, a driver, a configuration, the Makefile.
3. it is a gate or a test that runs offline in `make check`.
4. it is a configuration the runs used.

A search for unused code (vulture, at 60% confidence) found no dead function. Its hits are validators and HTTP
routes, reached through their decorators, helpers the tests call (`TelemetryView.refresh`, `scan_package` in
`gates/static_defects.py`), and the parameters of protocol signatures.

## What was removed, and why

### The equivalence suite against the previous implementation (32 files)

- `tests/equivalence/` (31 files) and `tests/conftest.py`, which only declared that suite's two options.

The suite compares this code with the previous implementation's banked runs and artifacts, bitwise where that
implementation was deterministic: its decisions, routers, dispatchers, estimator, arrival generator and
re-aggregated tables. Its inputs are that implementation's data, which this release does not include, so the suite
cannot run from this tree. Nothing in it executes in a run.

### The scope-freeze gate's log (1 file)

- `configs/after_freeze.json`.

During the campaign, a change to the frozen scope or presets passed only with a logged decision, and this file was
that log. The gate compares the files with a tag of the development repository, which this release does not carry,
so `make check` no longer runs it. The gate itself (`gates/freeze.py`) and its tests stay, because every file under
`gates/` is part of the gates' source fingerprint (below, under Comments). The scope and presets the runs used are
recorded by sha256 in the rendered campaign.

### A probe's output (1 file)

- `vllm-sr-bench/probes/stop_token_probe.qwen05b.out.json`.

It holds the model's answers to 40 benchmark prompts, which are not redistributed. The probe itself
(`probes/stop_token_probe.py`) is kept. Its result: sent with the standard `stop` string, all 40 answers equal the
answers sent with no stop at all, and 13 of the 40 differ from the answers sent with `stop_token_ids`, which is why
the vLLM Semantic Router's requests carry no stop (`vllm-sr-bench/README.md`).

### Makefile targets

`check-heavy` and `equivalence` (the suite above) and `freeze` (the gate above).

## Comments

Comments, docstrings, documents and a few error and log messages cited internal decision ids and dates. Those
citations were removed and the reasoning beside them kept, with these exceptions:

- `gates/` is the development tree's, byte for byte, except the hygiene gate (`gates/hygiene.py`) and its tests. The
  development tree's hygiene gate held its identity terms (an internal codename, development hosts, people, cluster
  paths) in the file itself. Here it reads them from a private `--terms` file that this release does not include,
  and its tests use invented terms.
- `gate_code` in `tools/cell_driver.py` hashes every file under `gates/`, and the analysis reuses a group's parity
  record only when it was made by the same gate code. The runs' parity records were made under the development
  tree's gates, so with this tree the report reads each group's parity as `not run` until the parity gate is rerun
  over the records. Tables, macros and every other value of a report do not depend on it. The raw bundle's own copy
  of the development tree rebuilds the four reports byte for byte.
- Every string the code writes into a report, a data file or a log, and every error message, is unchanged, so a
  few messages still cite an id. Three kinds of output differ: the docstring text some tools print (their `--help`,
  and `tools/venv_parity.py`'s usage message), the vLLM Semantic Router's evaluation `source` URL
  (`vllm-sr-bench/vsrbench/config.py`), which names no repository here, and its test profile's node name and two
  messages, which named the test machine (test-profile output only: no run record or report holds them). A vLLM Semantic Router block compares
  the code of its earlier runs with its own (`vsr/harness.json`), so a block started on the development tree's
  code continues only on that code, and a new block starts on this tree.

The configuration files keep the notes the runs used, except four of them (`decisions.md`).

Nothing in this list executes inside a run, so removing it cannot change a measured number. The check that it
cannot change a reported number is stage 3: the paper data pack regenerates every number, table and figure with
this tree's analysis code, byte for byte, and the reports rebuilt from the run records with it differ from the
shipped ones only in the parity state described above.
