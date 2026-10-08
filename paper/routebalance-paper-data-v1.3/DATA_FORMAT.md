# The data files

Every file is JSON. `data/SHA256SUMS` lists each with its sha256, and the generator checks every one before it reads
it. The files are as the analysis wrote them. Their free-text notes (a `fit`, `what` or `about` field) sometimes cite
the development decision log by id, which this release does not include.

A run is one configuration at one trace and one seed. Its id is `<configuration>__<trace>__s<seed>`, for example
`qwen.rb-static.q80l10c10__l30__s1`: RouteBalance at the 80/10/10 weights (quality, latency and cost), at 30
requests per second, seed 1. A trace `lNN` is Poisson arrivals at NN requests per second over the 3,532 Qwen
evaluation prompts, and `glNN` the same over Gemma's 1,979. `gaS-lNN` is gamma arrivals at the same mean rate, of
shape 0.25 (`ga025`), 0.5 (`ga05`) or 2 (`ga2`).

## The reports

`qwen_report.json` (the 659 first-seed Qwen runs) and `gemma_report.json` (the 45 first-seed Gemma runs) are the
output of the code repository's `analysis/report.py`:

| key | contents |
|---|---|
| `coverage` | the render's expected runs, the ones valid in this view, and the ones missing, invalid or pending |
| `parity` | per comparison group, `passed` when its latest parity record covers exactly the runs this report read, and `not run` when no such record exists. Here every group reads `not run`. `parity.json` holds the parity gate over the first-seed runs, which passes for every group |
| `left_out`, `failures`, `gate_failures` | the runs kept out of the comparisons, with the reason, then those of them that make the report incomplete (every one but a latency-sweep run, which has no quality by design), and the runs whose gate verdict is not valid |
| `cells` | per run: its group, configuration, trace and seed, its `summary` (requests, failures, quality, mean and percentile E2E latency in ms, cost per 1,000 requests, SLO attainment, throughput, the time a request spends in the scheduler) and its `model_mix` (the share of requests per model) |
| `configs` | per configuration and trace, the means over its seeds: `quality`, `e2e_mean_ms`, `e2e_p99_ms`, `cost_total_per_1k`, `cost_decode_per_1k`, `slo_miss_tight`, `slo_miss_loose` (the share of requests missing each SLO) and `throughput_rps`, with the configuration's `system`, `knob`, `dispatch`, `group`, `trace_id` and `seeds` |
| `frontiers` | per metric, then per `group|trace|system`, the configurations on that system's quality-metric frontier |
| `equal_quality`, `equal_quality_per_metric` and their `_appendix` | per baseline configuration and trace, the comparison at equal quality: the RouteBalance presets it is read between, and per metric the baseline's and RouteBalance's values, the change and the relative change, per seed, and the request-bootstrap interval |
| `not_compared` | baseline configurations with no comparison, and why |
| `goodput` | per configuration, the largest rate that keeps 90% or 99% of requests within the SLO |
| `rates` | the request rate of each stationary trace |
| `provenance` | the sha256 of the rules and of each family's SLO bases the report used |

## The rest

| file | contents |
|---|---|
| `spread_qwen.json`, `spread_gemma.json` | per configuration and trace run at seeds 1, 2 and 3 (317 for Qwen, 45 for Gemma), each seed's quality, mean E2E latency, cost per 1,000 requests and time off the instances, and for RouteBalance and its variants the wait for a round's decision, with the sha256 of the report they were read from |
| `parity.json` | the parity gate over the first-seed runs, per group: its exit status, its output, and each run with the sha256 of its record |
| `vsr_summary.json` | the vLLM Semantic Router runs at 6 and 9 req/s, and round robin and RouteBalance 80/10/10 at 6 req/s as a check against the report, summarised with the report's own functions |
| `vsr_failure.json` | per vLLM Semantic Router run: where a request's time went (Envoy, the router's signal phase, the rest of its routing) and, where its router stopped, when, how many requests it held and where its threads were |
| `fleet_qwen.json`, `fleet_gemma.json` | the fleets: per tier its model, GPU, input and output price per token and `max_num_seqs`, and per instance its tier and node name |
| `tpot_qwen.json` | the TPOT table RouteBalance uses: the constant per tier, the curve over the running batch, and the calibration's samples |
| `prefill_qwen.json` | the time to first token per tier as a line in the prompt's token count |
| `prereg_rules.json` | the preregistered rules: the judge, the equal-quality tolerance and bootstrap, the SLO multipliers |
| `corpus_stats.json` | per model, the mean judged quality of the two scored evaluation corpora, and how often one model's answer scores at least another's |
| `budget_judge.json` | the record count and sha256 of the judge's two stores for the budget runs' served texts (the texts are not shipped) |
| `round_shares.json` | per reference run of the ablation (RouteBalance 80/10/10, seed 1, at 12, 30 and 42 req/s): the decisions scored after an earlier assignment of the same round, and those that went to an instance the round had already given a request to, with the sha256 of each decision log |
| `round_range.json` | per main-comparison run of RouteBalance 80/10/10 (seeds 1 to 3 at 12, 18, 24, 30, 36 and 42 req/s): the decisions and rounds, the mean round size, the share of rounds at each size limit, the round's quality range averaged over rounds and over requests, the share of requests the 72B model is estimated to answer best (and the share tied), over those requests its mean lead over the next model on the judge's scale and as a share of its round's range and the share of them it received, its share of all requests, the decisions per tier, the model the lead was computed for (`big`), the number of scored instances, and the decision log's sha256 (`code/round_range.py` in the meta-analysis pack, over the decision logs there) |
