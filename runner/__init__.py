"""The runner: the cells of a render run end to end on a deployed fleet (python -m runner, runner/__main__.py).

A cell is one run: one configuration on one trace and seed, named <config>__<trace>__s<seed>. It runs in an attempt
directory, <cells root>/_attempts/<cell>.<stamp>/<cell>/, and is published to <cells root>/<cell>/ (a rename, never
over an existing directory) once its record exists, valid or not (runner/store.py). The cell directory holds what
the gates read:
  <cell>.cell.json                    the benchmark client's record (bench.runner)
  config.json                         the scheduler document POSTed for the cell, byte for byte the rendered one
  post_config.json                    the scheduler's reply to that POST
  group.json                          {group, reference, intended_differences} for the parity gate
  warmup.json                         the direct warm-up requests
  stats_before.json, stats_after.json, manifest_before.json, manifest_after.json
  decisions.jsonl                     the decision-log lines of the cell's window, all with the cell's digest
  latency.jsonl                       the latency sweep's lines of the window (that configuration only)
  window.json                         the byte offsets of the two logs that bound the window
  engines_before.json, engines_after.json   every engine as it reports itself (start time, KV cache configuration,
                                      model, context, serving counters) with its deploy.sh launch command
  gates/<gate>.log, gates/verdict.json

One cell (runner/cell.py), in order:
  1. drain the scheduler and the engines; refuse a scheduler that has logged a decision (a fresh process per cell);
     POST the cell's configuration, which also resets the dispatchers' state, and check the digests it answers;
     the preflight gate
  2. read every engine as it reports itself, before the warm-up so that the traffic check covers the cache reset;
     the warm-up straight to the engines, which resets every prefix cache and proves it empty; drain again
  3. the scheduler's statistics and manifest once its logs are settled; its code and launch values
  4. the benchmark client, offline and in a clean environment
  5. the statistics and manifest again, and the cell's window cut out of the logs; the engines again
  6. the run gates (arm, zero_call), the verdict, publish; the copy at the second location (runner/mirror.py);
     parity once the cell completes its group
The rows of a render (runner/rows.py) run their cells in the plan's order, resuming, each on a fresh scheduler
process (runner/scheduler.py restarts it from its deploy.sh run directory).
"""
