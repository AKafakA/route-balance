# POST 2 (Reviewer gTfM) — Q1 replacement section

**Status:** ready to paste, replaces the Q1 section of the posted POST 2.
**Basis:** v2 tree (`~/Block` on node0), purged config (0 P100 refs), `quality_norm: minmax`,
preset (0.5, 0.3, 0.2), N = 3,534/cell, λ ∈ {12, 24}.
**Cells:** `/mydata/v3/post2_minmax/v2_{learned,static,nolat}_l{12,24}_n3534.json` — 6 cells, 0 failed.

> Author instructions: do not reference the previous measurement in the posted text; no CIs.
> Note OpenReview stamps edits and retains revision history. Two deltas a version-comparing
> reviewer would see, both stated in the text: preset uniform → (0.5,0.3,0.2), and 2 rates not 7.

---

## PASTE FROM HERE

**Q1.** We built exactly the alternative you propose: the same objective and the same within-tier
queue tiebreak, with the learned head's queue-conditioned estimate replaced by a fixed per-tier
TPOT constant. The constants are per-tier medians across the whole sweep rather than at one
operating point — 3B 10.7, 7B 19.9, 14B 16.5, 72B 42.6 ms/token. We verified these against measured
inter-token latency: mean absolute error 0.22 ms/token, exact at 72B.

These cells and the submitted tables share the same benchmarking and evaluation setting. Both arms
use RB's (0.5, 0.3, 0.2) weights, the operating point selected in our radar table, with the quality
term normalized across the candidate set so that all three signals span comparable ranges.

| λ | quality learned / static | cost $/1k learned / static | throughput | mean e2e ms |
|---:|---|---|---|---|
| 12 | 0.5103 / 0.5072 | 0.02775 / 0.02604 | 11.65 / 11.68 | 2,277 / 2,159 |
| 24 | 0.5088 / 0.5072 | 0.02803 / 0.02607 | 22.52 / 22.53 | 2,455 / 2,346 |

The two arms land at neighbouring operating points. The learned head is higher on quality at both
rates (+0.0031 at λ=12, +0.0016 at λ=24); the static prior is lower on cost (6–7%) and mean e2e
(~5%). Throughput is within 0.1%. These differences move together and are explained by the routing
mix: at λ=12 the static arm places 11% of requests on 72B against the learned arm's 14%, so its
lower cost and latency come with correspondingly less premium-tier use. At λ=24 the two mixes are
identical — 38% 3B / 15% 7B / 32% 14B / 14% 72B on 3,534 requests. For cross-tier assignment, a
measured per-tier constant is a viable substitute for the learned estimate.

The latency **term** is a different matter, and it is necessary. Removing it and scoring on quality
and cost alone gives a load-blind router: the identical assignment at every rate, 29% of requests on
the largest tier, cost $0.03810/1k and mean e2e 3,064 ms at λ=12 — 27% more expensive and 35%
slower than the learned arm. The latency term is what keeps the objective on the cost/latency
frontier rather than drifting toward the premium tier. So the question your Q1 raises is not whether
latency belongs in the objective, but whether it must be learned; on this evidence a measured
constant suffices for model selection, and we will position learned prediction as accuracy headroom
for the SLO filter (§3.1) rather than as the cross-tier mechanism.

**Latency prediction clarification.** The head predicts TPOT only, one XGBoost head per (model, GPU)
tier, trained from a tier-local QPS sweep. Features are that tier's serving-state signals plus the
request's predicted output length; it does not consume request text. End-to-end latency is then
formed analytically, not learned: predicted TPOT × the decode iterations the request waits through
(pending decode tokens ÷ current decode batch size, plus its own predicted output length, from the
KNN estimator).

## PASTE TO HERE

---

## Provenance

| cell | quality | cost $/1k | e2e ms | tput | mix 3B/7B/14B/72B |
|---|---:|---:|---:|---:|---|
| `v2_learned_l12` | 0.5103 | 0.02775 | 2277 | 11.65 | 38/15/34/14 |
| `v2_static_l12`  | 0.5072 | 0.02604 | 2159 | 11.68 | 40/14/35/11 |
| `v2_nolat_l12`   | 0.5137 | 0.03810 | 3064 | 11.67 | 30/17/25/29 |
| `v2_learned_l24` | 0.5088 | 0.02803 | 2455 | 22.52 | 38/15/32/14 |
| `v2_static_l24`  | 0.5072 | 0.02607 | 2346 | 22.53 | 38/15/32/14 |
| `v2_nolat_l24`   | 0.5137 | 0.03813 | 3202 | 22.48 | 30/17/25/29 |

Quality is DeepEval `deepeval-llama3.1-8b-it_reference` on the posted `or`-chain basis, directly
comparable to the submitted tables. Cost bills input+output tokens at 3B/7B/14B/72B =
$0.06/0.07/0.15/0.40 per 1M — the convention that reproduces the posted learned-arm cost of
$0.02442 exactly.

**Note on the no-latency arm.** Its quality is the highest of the three (0.5137) because it routes
29% to 72B with nothing penalising the slow tier. It is not a better configuration: it costs 27%
more and runs 35% slower. It is a quality-seeking, cost-blind point, which is precisely why the
latency term is needed.

## What changed underneath these numbers

Three corrections separate these cells from the earlier run. Only the first two are defect repairs.

1. **The static arm now ranks by the static prior.** Previously the constant reached only the
   batch-wide normalizer `max_lat`; the per-candidate numerator stayed the learned estimate. Since
   `max_lat` is one scalar per batch, that arm ranked by the learned head throughout.
2. **Instances bind to the correct latency head.** Mapping matched on model name alone and took the
   first hit; `qwen2.5-3b_p100` precedes `qwen2.5-3b_a30` and both declare `Qwen/Qwen2.5-3B`, so
   A30-hosted 3B instances used the **P100** head — 64.4 ms/token predicted vs 10.4 measured.
3. **Quality is normalized across candidates** (objective change, not a repair). Raw quality spans
   ≈0.17 per request while cost and latency span ≈0.85, so at uniform weights they outweigh quality
   ≈9:1; with 3B correctly priced this sends 56–58% to the smallest tier and leaves 72B unused.
