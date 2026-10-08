# Corrected rebuttal — measured results

**Basis for every cell below:** unwrap fix ON (`CARA_UNWRAP_PROMPT=1`), **k=50**, cap 256, seed 1,
N=3534, minmax, `quality_norm` in `slo_defaults`. Quality is **zero-safe** DeepEval; cost is the
**paper's convention** (realized output tokens × output price, `neurips_2026.tex:221`).
Cells under `/mydata/v3/rebuttal_fix/`. Phase 1 = 16/16 banked, **0 errors, 0 INVALID**.

Not re-run because their routers never read the prompt (both verified, not assumed): all
Passthrough rows, and W2-A prompt-blind (its index has per-model score std = 0.0000, so retrieval
cannot change its answer).

---

## POST-2 Q1 — learned vs static TPOT

| signal | arm | λ | quality | $out/1k | mean ms | mix 3B/7B/14B/72B |
|---|---|---:|---:|---:|---:|---|
| deepeval | arm 1 learned | 12 | 0.4144 | 0.01732 | 2395 | 25/6/58/11 |
| deepeval | arm 4 static | 12 | 0.4136 | 0.01738 | 2385 | 24/7/57/12 |
| deepeval | arm 1 learned | 24 | 0.4145 | 0.01753 | 2624 | 25/7/56/12 |
| deepeval | arm 4 static | 24 | 0.4142 | 0.01734 | 2586 | 25/7/56/12 |
| judge | arm 1 learned | 12 | 0.4119 | 0.01658 | 2397 | 32/12/41/15 |
| judge | arm 4 static | 12 | 0.4122 | 0.01659 | 2376 | 32/12/41/16 |

**learned − static:** deepeval +0.0008 (λ12) / +0.0003 (λ24); judge −0.0003 / −0.0001.
Cost within ±1.1%, mean within ±90 ms.

⚠ **The posted Q1 comparison does not survive.** Posted: *"the learned head is higher on quality at
both rates (+0.0031 at λ=12, +0.0016 at λ=24); the static prior is lower on cost (6–7%) and mean
e2e (~5%)."* **The 6–7% cost and ~5% latency advantages vanish** and the quality gap shrinks 4–5×.
Posted mix 38/15/32/14 → **25/6/58/11**.

**The reviewer-facing conclusion survives and strengthens.** gTfM asked whether TPOT can be
benchmarked statically instead of learned. The corrected answer — the two estimators are
**equivalent within noise on every axis** — supports that more cleanly than the posted asymmetric
result did.

## Signal ablation (same index, one variable)

deepeval vs judge changes routing materially — **58% vs 41% to 14B** — at near-equal quality
(0.4144 vs 0.4119). Both arrays live in `deploy_v3_k50_14919`, so corpus, k and embeddings are held
fixed; only `score_type` differs. Worth its own CR line: D1 was framed as a defect, and it turns out
to change *where* requests go without changing *how well* they are served.

## POST-3 W2-A — partial labels

| signal | arm | quality | $out/1k | mix |
|---|---|---:|---:|---|
| deepeval | full index | 0.4138 | 0.01712 | 25/7/58/11 |
| deepeval | 50% labels | 0.4094 | 0.01727 | 22/8/60/11 |
| judge | full index | 0.4117 | 0.01665 | 32/12/41/16 |
| judge | 50% labels | 0.4087 | 0.01730 | 32/11/41/16 |

**Halving the labels: −0.0044 (deepeval), −0.0030 (judge).**

⚠ **Posted claim weakens.** Posted: *"Halving the labels has no measurable effect (0.5096 vs
0.5095)."* The corrected effect is small but consistently negative on both signals, and comparable
in size to the whole RB-vs-BEST-Route quality gap.

**Caveat that limits how hard this can be pushed:** the f50 index used here is NOT the posted one.
The original construction was unrecoverable; this is a deterministic 50% subsample of the v3 index
(seed 20260731, 14,919 → 7,459 rows), verified unbiased — per-model means within 0.002 of full on
both judges. A single subsample draw can move ±0.004, so the honest statement is *"halving the
labels costs a small amount"* rather than a specific figure. Pinning a number needs 3–5 draws.

## POST-3 W3 — baselines under the fix

| system | quality | $out/1k | mean ms | p99 | mix 3B/7B/14B/72B |
|---|---:|---:|---:|---:|---|
| RB uniform (1/3,1/3,1/3) | 0.4047 | **0.01267** | **1894** | **5165** | 33/8/58/1 |
| BEST-Route t0.3 / Po2 | 0.3982 | 0.01306 | 2727 | 13969 | 54/21/13/12 |
| RB (0.5,0.3,0.2) | 0.4144 | 0.01732 | 2395 | 12518 | 25/6/58/11 |
| AvengersPro / Po2 | **0.4276** | 0.03595 | 4362 | 13287 | 14/0/34/**52** |

* **RB uniform DOMINATES BEST-Route** — higher quality, lower cost, 833 ms faster, 2.7× better tail.
* **AvengersPro leads on quality (+0.0132 over RB 532)** and buys it with **52% of traffic on 72B**:
  2.1× the cost and 1,967 ms slower.
* At λ24 the BR collapse reproduces: RB is **34.8 s** faster on mean, **112 s** better at p99.
* **RB traces the frontier; the baselines are single points** — RB spans 0.4047→0.4144 across
  $0.0127–0.0173, sitting between BR (cheap/weak) and AP (dear/strong).

⚠ **Puts the posted W1-A claim at risk.** Posted: *"Against AvengersPro, RB (0.5,0.3,0.2) improves
every reported axis"* (RB 0.5093 vs AP 0.5037). Post-fix on the standard topology the quality
ordering **reverses** (AP 0.4276 vs RB 0.4144). W1-A runs on the 5/3 topology so it is not decided,
but the reversal is likely to carry.

**Open at time of writing:** RB's quality-max preset (0.8,0.1,0.1) post-fix — the direct test of
whether RB reaches AP's quality at lower cost. Running.
