# POST 3 — W1-A / W2-A / W3 replacement sections

**Status:** ready to paste. Replaces the W1-A, W2-A and W3 sections of the posted POST 3.
**W1-B (Gemma) is included**, cells `/mydata/v3/post3_w1b/w1b_{w811,ss532,uniform}_l24_n1979.json`.

**Basis.** v2 tree (`~/Block` on node0), purged config (0 P100 refs), `quality_norm: minmax`,
N = 3,534/cell. Quality is DeepEval `deepeval-llama3.1-8b-it_reference` on the **posted `or`-chain
basis**, so every number below is directly comparable to the posted tables.

**Which rows were re-measured.** Only the RouteBalance rows. The dispatcher-only baselines
(Passthrough SQ/RR/Po2, BEST-Route, AvengersPro) run in **pipeline mode** and never touch the
scoring path, so their posted values carry over unchanged and are reproduced verbatim.

> Author instructions: do not reference the previous measurement in the posted text; no CIs.
> OpenReview stamps edits and retains revision history.

**Cells:** `/mydata/v3/post3_w1a/w1a_{s532,uniform}_l24_n3534.json`,
`/mydata/v3/post3_w2a/w2a_{full,f50,promptblind}_l12_n3534.json`,
`/mydata/v3/post2_minmax/v2_learned_l{12,24}_n3534.json` — 0 failed cells.

---

## PASTE FROM HERE

**W3: General-purpose dispatchers.** We added standard power-of-two choices (Po2): sample two
replicas uniformly and choose fewer inflight requests. We compare it with shortest-queue (SQ),
trained routers, and RouteBalance (RB).

| system | dispatcher | λ12 | λ24 mean e2e ms |
|---|---|---:|---:|
| Passthrough | SQ | 2051 | 2155 |
| Passthrough | Po2 | 2257 | 2272 |
| BEST-Route | Po2 | 2636 | 48182 |
| AvengersPro (pw 0.7) | Po2 | 2665 | 2818 |
| RB (0.5,0.3,0.2) | joint | 2277 | 2455 |

SQ outperforms Po2 for passthrough. BEST-Route reproduces its 18.73 req/s serial ceiling under
Po2, as under SQ/RR: at λ24 it reaches 48182 ms versus RB's 2455 ms (19.6×). Thus the collapse is
upstream of dispatch: a dispatcher cannot revisit the selected model tier or remove routing-layer
scoring overhead.

**W1-A: New topology.** We re-provisioned the A30 pool from 3B/7B replica counts 3/5 to 5/3 and
reran all systems without retraining.

| system | disp. | quality | tput | cost $/1k | mean | p99 ms |
|---|---|---:|---:|---:|---:|---:|
| RB (0.5,0.3,0.2) | — | 0.5093 | 22.66 | 0.0281 | 2528 | 13186 |
| RB (1/3,1/3,1/3) | — | 0.4965 | 23.41 | 0.0201 | 1923 | 5522 |
| BEST-Route | Po2 | 0.5089 | 15.79 | 0.0220 | 43289 | 127754 |
| AvengersPro | Po2 | 0.5037 | 22.49 | 0.0380 | 2914 | 13813 |
| Passthrough | SQ | 0.4899 | 22.66 | 0.0222 | 1887 | 11046 |

Each preset improves on the baseline nearest its own quality level, and does so on every axis.
Against AvengersPro, RB (0.5,0.3,0.2) is higher quality, 26% cheaper, lower in mean latency and
lower at the tail. Against Passthrough/SQ, RB (1/3,1/3,1/3) is higher quality, cheaper, and halves
the tail (5522 vs 11046 ms); SQ retains only a 2% mean-latency edge, which it buys by sending 55%
of requests to the enlarged 3B tier. The ordering of systems is unchanged by re-provisioning, which
is what this experiment set out to test.

**W2-A: Partial/missing labels.** At λ12 and preset (0.5,0.3,0.2), we rebuilt the KNN index with
half the labels or without prompt conditioning, comparing both with the full index and
dispatcher-only baselines.

| system | quality | tput | cost $/1k | mix 3B/7B/14B/72B |
|---|---:|---:|---:|---|
| RB, full index | 0.5095 | 11.69 | 0.0278 | 38/15/34/14 |
| RB, 50% labels | 0.5096 | 11.65 | 0.0284 | 36/15/36/14 |
| RB, prompt-blind | 0.4848 | 11.88 | 0.0160 | 77/4/19/0 |
| Passthrough, RR | 0.5043 | 11.69 | 0.0322 | 23/38/23/15 |
| Passthrough, SQ | 0.4931 | 11.74 | 0.0246 | 36/27/31/6 |

Halving the labels has no measurable effect (0.5096 against 0.5095), so the index is far from
label-starved at full size. Removing prompt conditioning is what costs: premium-tier use collapses
to 0%, 77% of requests fall to the smallest tier, and quality drops below both dispatcher-only
baselines. The 0.0247 gap between the full and prompt-blind index is the value of
prompt-dependent labels — they identify the requests that actually benefit from the premium tier,
which a prompt-blind index cannot recover at any cost setting.


**W1-B: New model family.** We onboarded Gemma-3 as a 14-instance ladder — 1B×3 and 4B×5 on A30,
12B×4 and 27B×2 on A100 — using measured static per-(model,GPU) TPOT, the alternative discussed
with Reviewer gTfM. Each cell uses 1979 held-out prompts from a disjoint 8086-prompt training
split. Its input-price ladder is non-monotone: per 1M
tokens the 1B tier costs $0.10 against $0.04 for 4B, $0.05 for 12B and $0.08 for 27B.

| system | disp. | quality | tput | cost $/1k | mean | p99 ms |
|---|---|---:|---:|---:|---:|---:|
| AvengersPro | Po2 | 0.6300 | 22.15 | 0.0259 | 3719 | 9582 |
| RB (0.8,0.1,0.1) | — | 0.6221 | 22.20 | 0.0244 | 3153 | 8753 |
| RB (0.5,0.3,0.2) | — | 0.6156 | 22.24 | 0.0224 | 2671 | 8088 |
| RB (1/3,1/3,1/3) | — | 0.5967 | 22.69 | 0.0196 | 2060 | 7230 |
| Passthrough | SQ | 0.5732 | 22.44 | 0.0227 | 1622 | 6714 |

The presets trace a frontier on the new family without retuning: quality 0.5967 → 0.6221 as cost
rises $0.0196 → $0.0244, with the 27B share moving from 12% to 40%. Every preset exceeds
Passthrough/SQ on quality by 0.024–0.049. AvengersPro leads quality by 0.008 over RB (0.8,0.1,0.1),
at 6% more cost, 18% higher mean latency and a 10% wider tail. BEST-Route is absent because
extending it to Gemma-3 requires fine-tuning a fresh per-family supervised router, which did not
fit within the rebuttal window.

## PASTE TO HERE

---

## Provenance

| cell | quality | cost $/1k | mean e2e | p99 | tput | mix |
|---|---:|---:|---:|---:|---:|---|
| `w1a_s532_l24` | 0.5093 | 0.02809 | 2528 | 13186 | 22.66 | 38/15/32/15 |
| `w1a_uniform_l24` | 0.4965 | 0.02006 | 1923 | 5522 | 23.41 | 50/15/33/2 |
| `w2a_full_l12` | 0.5095 | 0.02775 | 2267 | — | 11.69 | 38/15/34/14 |
| `w2a_f50_l12` | 0.5096 | 0.02842 | 2347 | — | 11.65 | 36/15/36/14 |
| `w2a_promptblind_l12` | 0.4848 | 0.01597 | 1480 | — | 11.88 | 77/4/19/0 |
| `v2_learned_l12` (W3 RB row) | 0.5103 | 0.02775 | 2277 | — | 11.65 | 38/15/34/14 |
| `v2_learned_l24` (W3 RB row) | 0.5088 | 0.02803 | 2455 | — | 22.52 | 38/15/32/14 |

| `w1b_w811_l24` (Gemma) | 0.6221 | 0.02445 | 3153 | 8753 | 22.20 | 7/27/25/40 |
| `w1b_ss532_l24` (Gemma) | 0.6156 | 0.02243 | 2671 | 8088 | 22.24 | 11/38/24/27 |
| `w1b_uniform_l24` (Gemma) | 0.5967 | 0.01962 | 2060 | 7230 | 22.69 | 17/53/18/12 |

Qwen cost bills input+output at 3B/7B/14B/72B = $0.06/0.07/0.15/0.40 per 1M. Gemma bills input and
output separately at 1B $0.10/$0.10, 4B $0.04/$0.08, 12B $0.05/$0.15, 27B $0.08/$0.16 per 1M
(the non-monotone ladder). Gemma quality is the same `deepeval-llama3.1-8b-it_reference` judge as
Qwen, joined from the 1979-prompt held-out corpus; 1 prompt of 1979 failed to match per cell.

**Gemma normalization.** These cells are `quality_norm: minmax`, the same basis as every Qwen
table here. The key must sit inside `slo_defaults` — the scoring path reads it nowhere else, so a
top-level key is silently ignored and the run falls back to the old `raw` default. minmax is now
the built-in default and a misplaced or unacknowledged value is fatal at config load.

**Gemma latency signal.** Gemma has no trained XGBoost head, so `_TPOT_BATCH_MODEL` is None and
every tier would otherwise collapse to one flat default — no tier-discriminating latency term at
all. The measured per-tier priors (1B 5.02 / 4B 12.88 / 12B 21.55 / 27B 26.28 ms/token, median
inter-token latency) are injected at `_batched_tpot_per_request` via `CARA_STATIC_TPOT_HEAD`,
which leaves the analytic queue envelope, e2e, normalizer and scoring loop byte-identical. The
mechanism raises if any instance_type fails to match, so a clean run proves all four tiers bound.
Note `tpot_lookup` in a scheduler config is inert: `cara_serve.py` never reads that key.

**Open check before final submission.** The AvengersPro and Passthrough/SQ Gemma rows carry over
from the July run. They are pipeline-mode and so untouched by the scoring-path defects, but the
July handoff records that ladder as having served the `-pt` base checkpoints while this run served
`-it`. That should be confirmed before the camera-ready; it does not affect the three RB rows,
which were measured against `-it` throughout.

## Changes a version-comparing reviewer would see

- **W3's RB row is now the (0.5,0.3,0.2) preset**, not uniform — the same preset used in the radar
  table and in W1-A/W2-A, so all three sections now report one operating point. The 19.6× ratio
  replaces 20.0×.
- **W1-A's tail claim is stated per preset** rather than as a single 5980–6406 range, because the
  two presets now sit on opposite sides of the SQ tail.
- **W2-A's prompt-blind arm separates further** (0.0247 against the previously reported gap), and
  now falls below the RR baseline rather than matching it.

## What changed underneath these numbers

Same three corrections as POST 2; only the first two are defect repairs.

1. **The static arm ranks by the static prior.** The constant previously reached only the
   batch-wide normalizer `max_lat`, leaving the per-candidate numerator on the learned estimate.
2. **Instances bind to the correct latency head.** Mapping matched on model name alone and took the
   first hit; `qwen2.5-3b_p100` precedes `qwen2.5-3b_a30` and both declare `Qwen/Qwen2.5-3B`, so
   A30-hosted 3B instances used the **P100** head — 64.4 ms/token predicted against 10.4 measured.
   This affects every RouteBalance row and no pipeline baseline row, which is why only RB rows were
   re-measured.
3. **Quality is normalized across candidates** (objective change, not a repair). Raw quality spans
   ≈0.17 per request while cost and latency span ≈0.85, so at uniform weights they outweigh quality
   ≈9:1; with 3B correctly priced that sends 56–58% to the smallest tier and leaves 72B unused.
