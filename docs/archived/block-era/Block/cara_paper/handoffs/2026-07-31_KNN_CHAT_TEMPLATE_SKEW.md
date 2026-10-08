# KNN train/serve skew: the quality estimator embeds CHAT-WRAPPED prompts, the index was built on RAW ones

**Found 2026-07-31 ~09:40Z, during author-ordered debugging after the sweep was stopped.**
**Status: root cause CONFIRMED to 0.000000 agreement. Fix implemented behind `CARA_UNWRAP_PROMPT`
and ablated (results at the end). Blast radius now VERIFIED across all eras.**

## The defect

`cara_serve` passes `req.prompt_text` to the model estimator. Under `--chat`, that text is the
chat-templated prompt the benchmark client sends:

```
<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<actual prompt>
```

The KNN index was built on **raw** prompts (`build_index` encodes `d["prompt"]` from the training
corpus, and the eval dataset on disk contains no `im_start` markers — the wrapper is added by the
client, not present in the data). So every serving-time query is embedded with ~20 tokens of
boilerplate that is **identical across every request**, which dominates a 384-dim MiniLM embedding
and collapses semantic discrimination. Neighbour retrieval becomes close to arbitrary.

## Proof

Re-predicting on the cells' own prompts (250 sampled from
`p1/wq0.33_wl0.33_wc0.33_lambda12_n3534.json`):

```
mean |recorded - predict(WRAPPED)|   = 0.000000     <-- exact match
mean |recorded - predict(UNWRAPPED)| = 0.100480
```

| tier | recorded (= wrapped) | correct (unwrapped) | actual (test) | corpus |
|---|---:|---:|---:|---:|
| 3B   | 0.3250 | 0.3494 | 0.3716 | 0.3528 |
| 7B   | 0.2857 | 0.3425 | 0.3516 | 0.3389 |
| 14B  | **0.3207** | **0.4051** | 0.3980 | 0.3986 |
| 72B  | 0.3641 | 0.4383 | 0.4604 | 0.4207 |

Unwrapped predictions track the corpus and the test set closely. Wrapped predictions are depressed
by 0.02–0.08, **non-uniformly**, and invert the tier ordering:

* wrapped:   72B 0.3641 > **3B 0.3250 > 14B 0.3207** > 7B 0.2857   ← wrong
* unwrapped: 72B 0.4383 > **14B 0.4051 > 3B 0.3494** > 7B 0.3425   ← matches ground truth

## What this explains

* **RouteBalance under-uses 14B at every weighting** (capped ~23–30% across the whole 8-point
  sweep) while BEST-Route routes 40% there. RB's objective is told 3B is better than 14B.
* **RB reaches past 14B to 72B for quality** — 72B share climbs 1% → 8% → 15% → 28% as `w_quality`
  rises, buying quality at $0.40/1M instead of 14B's $0.15/1M.
* **BEST-Route beats RB on quality AND cost at every rate.** BR's DeBERTa router never consults
  this KNN, so it is unaffected by the skew.
* **Per-request argmax accuracy 29.7%**, against a 25% chance floor for four tiers.

## What was RULED OUT before landing here

* Distribution shift — train and test per-tier means are nearly identical
  (14B 0.3986 vs 0.3979; 3B 0.3528 vs 0.3461).
* Array misalignment — `per_judge_scores`, `output_lengths`, `similarity_scores` are all exactly
  14,919, matching `embeddings (14919, 384)`.
* Build-order bug — `build_index` appends per-model arrays in the same `for d in training_data`
  order as the prompts it encodes.
* Inverted distance weighting — `distances = 1 - top_sims` is applied on BOTH the FAISS and numpy
  paths, so the nearest neighbour correctly receives the largest weight.
* Lookup-path infidelity — API output equals a hand-computed weighted neighbour average to 1e-6.
* Wrong signal — `score_type=deepeval` is live (verified from the running process cmdline), the
  artifact carries the deepeval array, 0 P100 refs, minmax active.

## Blast radius

**Confirmed affected:** every cell whose routing consulted the KNN while the scheduler ran with
`--chat`. That includes the P1 v3 sweep (`lib_serve_v3.sh:51`) and the Gemma W1-B cells
(`gemma_cells_w1b.sh:63`).

**VERIFIED AFFECTED — every era, checked by opening banked cells** (400/400 prompts wrapped in
each): v1/May submitted (`may_baseline_wq033_l12.json`), v1exact tree July
(`gemma_rb_uniform_l24`), v2 June campaign (`cara_wq0.33_wl0.33_wc0.33_l12_s4`), all v3 rebuttal
cells (POST-2, W1-A, W2-A, W1-B) and the v3 P1 sweep. The wrapper is applied client-side by the
benchmark's dataset sampler (`dataset.py:107`, unless `--skip-chat-template`), so it has been
present since the beginning. **This is not a v3 regression — v3 is where it was caught.**
RouteBalance has never been measured with its quality signal working.

**CORRECTION to an earlier claim in this document:** an earlier draft said BEST-Route and
AvengersPro were "not affected". That is WRONG and is retracted. Both consume prompt text —
BEST-Route tokenizes `rreq.prompt` through DeBERTa (`best_route_4way.py:123-124`) and AvengersPro
embeds it with sentence-transformers — and `cara_serve.py:2409` builds
`RouterRequest(prompt=prompt_text)` from the SAME wrapped text. Only the dispatcher-only
passthrough arms (round_robin / shortest_queue / random / po2) are genuinely unaffected, since they
never read the prompt. Consequence: unwrapping ONLY RouteBalance would bias the comparison in its
favour, which is why the fairness patch below puts all three routers under one flag.

## Candidate fixes (none applied — author decision)

1. **Unwrap before embedding.** Strip the chat template in the estimator path so the query matches
   the index. Cheapest, and makes serving consistent with how the index was built.
2. **Send raw prompts from the benchmark.** `cara_serve` already sets `use_chat_endpoint=True`, so
   vLLM applies the template server-side; the client wrapping it first is arguably redundant.
3. **Rebuild the index on wrapped prompts.** Consistent, but bakes the boilerplate into every
   embedding and keeps the discrimination loss.

Option 1 or 2 restores the correct tier ordering. Whichever is chosen, **every KNN-routed cell
needs re-running**, and the v1/v2 scope question above has to be settled first.

---

# RESULTS — 4-arm ablation, λ12, cap 256, RB uniform, N=3534 (2026-07-31 ~10:07Z)

Cells: `/mydata/v3/knnfix/knnfix_{control,fixed,judge,k50}_l12_n3534.json`, 0 errors each.
Metrics on the corrected basis: **zero-safe quality, output-only cost** (the paper's convention,
`neurips_2026.tex:221`). Cap pinned to 256 via `--max-output-len`, matching the submitted paper.

| arm | quality | $out/1k | mean ms | p99 ms | tput | mix 3B/7B/**14B**/72B | Δq vs control |
|---|---:|---:|---:|---:|---:|---|---:|
| control (skewed) | 0.3673 | 0.01053 | 1786 | 5250 | 11.86 | 59/13/**28**/0 | — |
| **fixed** | **0.3921** | 0.01136 | 1856 | 5248 | 11.85 | 40/18/**41**/1 | **+0.0249** / +7.9% cost |
| judge | 0.3826 | 0.01069 | 1809 | 5284 | 11.85 | 47/17/**34**/2 | +0.0154 / +1.6% |
| **k50** | **0.4047** | 0.01267 | 1894 | 5165 | 11.87 | 33/8/**58**/1 | **+0.0375** / +20.3% |

One factor at a time from the `fixed` arm, so nothing is confounded:

* **Skew fix alone (control → fixed): +0.0249 quality for +7.9% cost.** The entire
  RB-vs-BEST-Route gap that motivated this investigation was 0.005 — **this effect is 5× larger**
  and is the biggest single measured effect of the campaign. 14B routing 28% → 41%, which is
  essentially BEST-Route t03's own allocation (40%), reached independently.
* **Signal (fixed vs judge): deepeval beats judge by +0.0095** at 6% more cost. D1 is real but
  second-order next to the skew.
* **k (fixed vs k50): k=50 adds a further +0.0126** over k=10 and pushes 14B to 58%.
  **The k the paper CLAIMS (50) outperforms the k that actually shipped (10)** — so the
  "k=50/14,919" correction item is a case where the paper's stated value would have done better.
  Note k is baked into the pickle (`load()` does `k=state["k"]`), which is why the config's k=50
  was inert; arm D used a re-baked artifact with identical embeddings and score arrays.

## What this does NOT establish

* **No RB-vs-BEST-Route claim.** The BR cells are at cap 1024 while these are at 256, so they are
  not comparable, AND BR's DeBERTa router is still receiving wrapped prompts. Both sides must be
  fixed and re-run at a common cap first.
* **Single cell per arm, one rate, one seed.** The +0.0249 skew effect is large enough to trust.
  The +0.0095 signal delta and +0.0126 k delta are within what one seed can produce and need
  repeats before use.
* **Cost rises with quality in every arm** — the fix moves RB up-and-right on the frontier, it
  does not strictly dominate. Whether that is a net win depends on the chosen operating point.

## Fairness patch — APPLIED LOCALLY, NOT RUN

`RouterRequest(prompt=_estimator_prompt(prompt_text))` puts BEST-Route (which tokenizes
`req.prompt` through DeBERTa, `best_route_4way.py:123-124`) and AvengersPro (which embeds it with
sentence-transformers) under the SAME flag. Without this, unwrapping only RouteBalance would hand
it an advantage the baselines never get. Committed but **not synced to node0 and not executed** —
it needs its own control/fixed pair before any comparative claim.

## Still open

Whether the corpus broadcast applied the chat template. The client wraps at dataset-sampling time
unless `--skip-chat-template` is passed, and that run's flags are not recoverable from the records.
This does not affect the retrieval fix — a raw query matches the raw index either way — but it
determines whether the DeepEval labels describe what serving actually produces. Decisive test:
send one corpus prompt to a model raw and wrapped, and compare both against the corpus's stored
response for that pair.

---

# OBJECTIVE COST DEFINITION — code/paper mismatch is REAL but IMMATERIAL (2026-07-31 ~10:50Z)

**The mismatch.** The paper defines the objective's cost as decode-only — Table 2
(`neurips_2026.tex:154`): `Ĉ_{r,i} = L̂_{r,m(i)} · c_{m(i)}` — and reports realized output tokens
(`:221`). The code billed `num_prompt_tokens*cost_in + expected_length*cost_out`
(`cara_serve.py`, now `_objective_cost()`), so the router optimised a cost the paper neither
defines nor reports.

**Measured effect: none.** `k50_dec` differs from `k50` in exactly one variable
(`CARA_COST_DECODE_ONLY=1`), λ12, cap 256, N=3534, 0 errors:

| arm | quality | PAPER $out | full $in+out | mean ms | mix 3B/7B/14B/72B |
|---|---:|---:|---:|---:|---|
| k50 (input+output objective) | 0.4047 | 0.01267 | 0.02410 | 1894 | 33/8/58/1 |
| k50_dec (decode-only objective) | 0.4047 | 0.01276 | 0.02442 | 1899 | 32/8/59/1 |

**Δquality = +0.0000**, mix 58% → 59% on 14B, cost +0.8%. All within noise.

**Why — predicted by the unit test before the run.** Cost enters the score as
`1 − cost/max_cost`. For the 1:1-priced tiers (3B/7B/14B) dropping the input term shrinks every
candidate's cost *proportionally*, so the normalisation absorbs it — the inter-tier ratio is 2.500
either way (verified in `test_cost_calc.py`). The only surviving channels are per-candidate
variation in predicted output length, and 72B's asymmetric 0.38/0.40 pricing — and 72B carries
1% of traffic here.

**A prediction of mine this refutes.** I claimed decode-only "would likely push RB further toward
14B, since input pricing is what makes 14B look costly next to 3B." That was wrong; the
normalisation argument (surfaced by writing the test) is why.

**Consequence.** Reconcile code and paper for the camera-ready — either change the code to
`L̂·c` or change Table 2 to state input+output — but **no re-runs are needed on its account**.
It also sharpens the earlier result: RB's cost position versus BEST-Route is driven by the
**routing mix**, not by how the objective prices candidates.

---

# WHY INPUT COST FLIPS THE RB-vs-BR VERDICT (subagent analysis, 2026-07-31 ~11:00Z)

**Question:** input and output mean lengths are similar (~103 in, ~115 out), so why does including
input flip RB from 3.0% cheaper to 12.4% dearer? Is it a dataset artifact?

**Answer: it is a covariance effect, and pooled means are the wrong statistic.** Cost is
`E[price × length]`, not `E[price] × E[length]`. Both systems have nearly identical mean input
price (RB 0.11503 vs BR 0.11252 $/Mtok, 2.2% apart), identical mean input length (102.81), and
identical total input tokens (363,319). The whole gap lives in the covariance:

| system | Cov(price, input_len) | corr |
|---|---:|---:|
| RB k50 | −0.00039 | −0.086 |
| BR fixed | **−0.00318** | **−0.327** |
| BR control | −0.00300 | −0.417 |

Per-tier mean input length ÷ pooled 102.8 (1.00 = length-neutral):

| | 3B | 7B | 14B | 72B |
|---|---:|---:|---:|---:|
| RB k50 | 1.07 | 1.12 | 0.95 | 0.32 |
| BR fixed | **1.29** | 0.90 | **0.61** | 0.32 |

**Decomposition (law of total covariance, residual-free).** The naive share-vs-length
counterfactual is degenerate — an 80% interaction term — because tier shares and per-tier lengths
are the *same* routing decision. The exact split:

| component | $/1k | share of gap |
|---|---:|---:|
| mean-price / routing mix | +0.00026 | **8.5%** |
| between-source coupling | +0.00216 | **70.8%** |
| within-source length sorting | +0.00063 | 20.7% |

**The routing mix explains only 8.5%.** ~91% is length-to-price assignment, mostly at corpus
granularity.

**Concentrated in three long-prompt corpora:**

| source | n | mean in | % of gap | RB tier% 3/7/14/72 | BR tier% |
|---|---:|---:|---:|---|---|
| squad | 572 | 203.6 | **59.0%** | 33/6/**61**/0 | **97**/3/0/0 |
| gsm8k | 495 | 109.2 | **25.0%** | 24/4/**72**/0 | 1/**94**/5/0 |
| code_ultra_feedback | 519 | 136.4 | **22.7%** | 38/11/51/0 | **72**/12/15/1 |
| beaver_tails | 583 | 35.2 | −12.8% | 10/15/72/3 | 27/15/16/**42** |

Across the 7 sources, BR's mean input price paid correlates **r = −0.816** with source prompt
length; RB's is −0.217. BR spends 72B on beaver_tails (35-token prompts) and 3B on squad (204).

**Not a dataset artifact — but corpus composition controls the SIGN.** Same prompt set, same
per-request lengths; the gap persists *within* each corpus (squad +90.5%, gsm8k +66.2%), and both
BR arms reproduce it, so it is a stable BEST-Route property. However:

| subset | PAPER | FULL |
|---|---:|---:|
| all | −3.0% | **+12.4%** |
| no squad | −6.7% | +2.1% |
| no squad+gsm8k | −13.6% | **−6.1%** |

Dropping two of seven sources flips the FULL-cost headline. Anything published on the in+out
convention is fragile to corpus composition.

**Two hypotheses measurably ruled out:**
* **Short-input/long-output requests are not the story.** 820 such requests (23.2%) contribute
  **−2.9%** of the gap. The entire effect is on the input axis, in the 100–200 and 200+ buckets
  (together 106% of it).
* **Input and output length are uncorrelated** — Pearson 0.028 (RB), −0.063 (BR), 0.022 on the
  exogenous corpus proxy. Per-tier all within [−0.26, +0.37].

**Corrections to earlier statements in this campaign:**
* BR's higher PAPER cost is **not** a pricing effect — BR generates **6.4% more output tokens**
  (431,206 vs 405,444) and hits the 256 cap more often (936 vs 774). Endogenous, entangled with
  the cap.
* Source labels: there are 7 top-level `dataset` values including **reward_bench**;
  `unified_chip2` and `itwgpt4` are sub-sources inside `mix_instruct`, not top-level groups.

**Unresolved, explicitly not speculated on:** *why* BEST-Route sends long prompts to small models.
Measured strongly and reproducibly in both BR arms, but whether it stems from the t=0.3 threshold,
the shortest-queue tie-break, or their interaction was not tested.

---

# RESOLVED: unwrapping is correct for ALL routers; RB dominates BR's best (2026-07-31 ~11:17Z)

## The training-text question is settled from source, not from scores

Both baselines train on RAW corpus prompts:
* **BEST-Route** — `train_best_route_wrapper.py:54,99-100` reads `rec["prompt"]` directly from the
  scored data.
* **AvengersPro** — `retrain_avengers_pro_qwen.py:76,94` does `r.get("prompt")` →
  `embedder.encode(prompts)`.

The scored corpus stores RAW prompts — proven independently by the fact that every aggregator must
call `unwrap_chat()` on the *cell* prompt to match it back to the corpus key. Same chain that
established the KNN index is raw.

The AvengersPro trainer states the intent at `:13-15`: *"same encoder the runtime adapter uses —
keeps train/test distributions."* Train/serve consistency was the explicit design goal; the
client-side chat wrapping silently broke it for all three routers.

**Consequence:** `br_fixed` is BEST-Route's LEGITIMATE configuration, not a broken one.
`br_control` was BR running with the same skew RB had. This also answers the corpus-generation
question: labels, index and trainers are all on the raw-prompt basis.

## BR threshold sweep — closes the tuning asymmetry

λ12, cap 256, N=3534, zero-safe quality, PAPER (decode-only) cost, 0 errors:

| arm | quality | $out/1k | mean ms | p99 | mix 3B/7B/14B/72B |
|---|---:|---:|---:|---:|---|
| **BR t0.0 fixed** | **0.3996** | 0.01350 | 2686 | 13994 | 50/22/16/12 |
| BR t0.3 fixed | 0.3982 | 0.01305 | 2758 | 14631 | 54/21/13/12 |
| BR t0.5 fixed | 0.3797 | 0.00956 | 2158 | 11178 | 91/0/1/8 |
| BR t0.7 fixed | 0.3682 | 0.00835 | 1982 | 5424 | 95/0/0/5 |
| BR t0.0 wrapped | 0.3969 | 0.01286 | 2694 | 8209 | 40/10/45/6 |
| BR t0.3 wrapped | 0.3889 | 0.01232 | 2952 | 9279 | 45/9/40/5 |
| **RB k50+fix** | **0.4047** | 0.01267 | **1894** | **5165** | 33/8/58/1 |
| RB k10+fix | 0.3921 | 0.01136 | 1856 | 5248 | 40/18/41/1 |

* **t0.0 is BR's best**, as the banked P1 cells predicted. t0.5/t0.7 are degenerate (91%/95% → 3B).
* The fix helps BR at both live thresholds (+0.0093 at t0.3, +0.0027 at t0.0) — less than RB's
  +0.0249, consistent with DeBERTa doing coarser classification than the KNN's per-prompt retrieval.

## Headline: RB's best vs BR's best, both correctly configured

| axis | RB k50+fix vs BR t0.0+fix |
|---|---|
| quality | **+0.0051** |
| cost (paper convention) | **−6.2%** |
| mean latency | **−792 ms** |
| p99 | **−8,829 ms** (5,165 vs 13,994) |

**Dominance on all four axes**, with both systems tuned and both under the same treatment, same
cap, same seed, same prompts. Materially stronger than the posted rebuttal's claim, which compared
RB against an untuned baseline setting.

## What is NOT established

* **One rate (λ12), one seed, one preset pair.** The paper claims a frontier across λ ∈ {6,12,24,30}
  and four systems. λ24/λ30 is where BR collapses and RB's edge should widen — expected, not measured.
* **AvengersPro is untested under the fix.** It reads the wrapped prompt too and was the strongest
  quality baseline in the posted W1-B table, so it is the one baseline that could still challenge RB
  on quality.
* **Every published number remains skewed.** The submitted tables, v2 campaign and posted rebuttal
  all need re-running — the skew changes routing, so re-aggregation cannot fix them.
