# Reproducing the paper

RouteBalance schedules each request of a heterogeneous LLM fleet onto a (model, instance) pair, scoring predicted
quality, load-aware latency and cost together, over rounds of requests. This repository holds the scheduler, every
baseline it is compared with (the routers and dispatchers in the same server, on the same telemetry and dispatch
path, and the vLLM Semantic Router as its own stack in front of the same engines), the benchmark client, the
campaign that ran the paper's 1,457 runs, the gates that decide whether a run counts, and the analysis that turns
the runs into the paper's numbers.

## Three stages

Each stage is a separate guide and ends in a check you can run yourself.

| stage | guide | you get |
|---|---|---|
| 1 | [`01-environment.md`](01-environment.md) | the analysis environment (CPU only), and the serving cluster the runs used |
| 2 | [`02-run-experiments.md`](02-run-experiments.md) | new runs, from a fresh cluster to gated run records |
| 3 | [`03-run-analysis.md`](03-run-analysis.md) | every number, table and figure of the paper, from the shipped data, with no GPU |

**To check the paper's numbers, go straight to stage 3.** The runs are measured and their summaries are shipped in
the paper data pack. Stage 2 exists for running them again on a new cluster.

## What is shipped where

| | contents | reproduces |
|---|---|---|
| this repository | the code and scripts, the rendered campaign and the configurations the runs used, and the gates | stage 2, and the analysis of stage 3 |
| the paper data pack | the per-report summaries the paper reads, its generators and plot scripts, the expected outputs | every number, table and figure (stage 3, first route) |
| the meta-analysis pack | one row per run for all 1,457 runs, the four reports, the rendered campaign, the fleets, rules, SLO bases and calibration tables | the paper data pack's inputs from the reports (stage 3, second route) |

Prompt and answer texts are not shipped: the evaluation prompts come from datasets whose licences restrict
redistribution, so the packs carry prompt ids and scores only.

## Also here

- [`gates.md`](gates.md): the gates that decide whether a run counts as a measurement.
- [`decisions.md`](decisions.md): the rulings the code comments refer to, restated.
