# Gates

A run is not a measurement until its gates pass. The gates check that what executed is what was configured, and
that nothing else ran beside it. None of them judges how a system served: a slow request, a late send or an old
load snapshot is counted and reported, never a reason to drop a run. Each gate is a module under `gates/` with
tests under `tests/gates/`, including fixtures that must fail.

## Before every run

| gate | refuses |
|---|---|
| `preflight.py` | a fleet or a scheduler that is not the one the run expects, or not idle: every engine healthy, on vLLM 0.18.1, serving exactly its tier's model and drained (nothing running or waiting), and the scheduler serving the run's configuration and fleet, by digest |
| fresh scheduler (`tools/cell_driver.py fresh`) | a scheduler process that has already logged a decision |

## Per run

| gate | refuses |
|---|---|
| `arm.py` | a run whose decision log does not show exactly the run: one line per request and no other, every line with the run's configuration digest, each dispatch to an instance of the fleet, no request that waited for a connection, and the values the configuration sets (round sizes, normalization scales, router concurrency) as the scheduler logged them |
| `zero_call.py` | a run whose latency source is the constant or the profiled curve but which called a learned head, or whose loaded latency model is not the configured one |
| isolation (`cell.py`) | a run where any engine's warm-up did not prove its prefix cache empty, where an engine's serving counters changed by anything but the run's own requests, or where a scheduler log writer failed |
| engines (`tools/engines.py`) | a run whose engines are not the processes the deploy record launched, or changed between the readings before and after the run |
| the verdict (`analysis/validity.py`) | a run with a failed request, or whose record does not show the run asked for: the request list is the trace's, the recorded arrivals and prompt order match the schedule regenerated from the run's own trace and seed (by digest), the scheduler's configuration and fleet are unchanged at the end, and no request waited for a pooled connection (the load stayed open loop). Any gate above failing also makes the run invalid |

## Per comparison group

| gate | refuses |
|---|---|
| `parity.py` | a group whose runs did not execute identically except in the differences the group declares: each run's execution manifest (code, library versions, launch values, thread pools, telemetry settings, the md5 of every artifact loaded) the same before and after, the code on disk unchanged while the scheduler ran, the same engine launches for every run |

## The tree

| gate | refuses |
|---|---|
| `hygiene.py` | committed files with host names, private addresses, personal identifiers, credentials, personal absolute paths, or backup and stale copies. The identity terms come from a private `--terms` file (`make check TERMS=<file>`), which this release does not include |
| `static_defects.py` | the defect classes that produce complete, error-free, wrong runs: a falsy `or` used as a value, an invented default, an experiment parameter read from the environment, GPU knowledge outside `routebalance/hw.py`, a hard-coded model list |

## The analysis

`analysis/manifest.py` accounts for every expected run of the render. A run enters the report only when it is
published and its verdict is valid, and a group whose latest parity record failed is left out whole. A missing,
invalid, unexpected or pending run, or a group whose parity has not run over its current runs, makes the report
exit non-zero, with the report still written. The parity records are keyed by the gates' source (`gate_code` in
`tools/cell_driver.py`), so a parity record made by other gate code counts as not run.

## What a failed gate does

A gate that fails before the run (preflight, the fresh scheduler) stops it before the first request: the attempt
stays unpublished under `_attempts/` and the run is retried. A run that ran and failed a gate or the verdict is
published as invalid and kept, never deleted, and the run is repeated. An invalid run never enters the analysis.
