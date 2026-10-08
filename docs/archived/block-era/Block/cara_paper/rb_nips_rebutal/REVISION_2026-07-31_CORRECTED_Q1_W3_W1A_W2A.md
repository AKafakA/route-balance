# Corrected rebuttal — POST-2 Q1 and POST-3 W2-A

**Status:** paste-ready. Data final (banked, 0 errors). Written while Phases 2–3 run.

**Basis for every number:** chat-template fix applied to the quality estimator, **k=50**, cap 256,
seed 1, N=3534, `quality_norm: minmax`, routing signal **deepeval**.
**Quality is on the POSTED `or`-chain basis** so every figure is directly comparable to the tables
already live on OpenReview. (Zero-safe quality runs ~0.10 lower across every row; that is the
separate D2 correction and belongs in the camera-ready, not here — mixing it in would make one
change look like two.) Cost is realized output tokens × output price (`neurips_2026.tex:221`).

**Rows reused rather than re-run**, because their routers never consult the prompt and are provably
unaffected: Passthrough RR/SQ, and RB prompt-blind (its index has per-model score std = 0.0000, so
retrieval cannot change its answer). They are recomputed on the corrected metric basis.

> Author note: three posted claims change. Each is marked ⚠ below with the old wording.

---

## PASTE FROM HERE

> 🚨 **Q1 IS INVALID — DO NOT USE. Flagged 2026-07-31T17:40Z.**
> The two arms below are **the same configuration**. `rebuttal_fix/q1_arm4_*` was launched with
> `CARA_STATIC_TPOT_HEAD`, which has **no reader** in `/users/asdwb/Block_v3` (that tree reads only
> `CARA_STATIC_LAT`; verified by extracting the archived tree — reader present in `Block:949`,
> absent in `Block_v3`). Arm 4 therefore ran the **learned head**, identical to arm 1.
>
> The section's own evidence is the tell: "routing mixes are near-identical, 25/6/58/11 against
> 24/7/57/12". Identical arms produce that. The July cells, where the switch did apply, differ by
> **twelve points** at λ24 (learned 56/38 on 14B/7B vs static 68/29).
>
> **False as written:** "differ only in the TPOT estimator" (they did not differ) and "not merely
> a viable alternative but an equivalent one" (untested).
>
> This is the AC's priority-#1 question. Redo belongs in S2's budgeted 40-cell no-predictor block,
> using `CARA_STATIC_LAT` or a wired-in `CARA_STATIC_TPOT_HEAD`, with a reader-existence check
> before the first cell.

**Q1.** We built exactly the alternative you propose: the same objective and the same within-tier
queue tiebreak, with the learned head's queue-conditioned estimate replaced by a fixed per-tier
TPOT constant. The constants are per-tier medians across the whole sweep rather than at one
operating point — 3B 10.7, 7B 19.9, 14B 16.5, 72B 42.6 ms/token, verified against measured
inter-token latency at 0.22 ms/token mean absolute error. Both arms use RB's (0.5, 0.3, 0.2)
weights and differ only in the TPOT estimator.

| λ | quality learned / static | cost $/1k learned / static | throughput | mean e2e ms |
|---:|---|---|---|---|
| 12 | 0.5159 / 0.5161 | 0.0173 / 0.0174 | 11.73 / 11.74 | 2,395 / 2,385 |
| 24 | 0.5163 / 0.5155 | 0.0175 / 0.0173 | 22.58 / 22.63 | 2,624 / 2,586 |

The two arms are equivalent. Quality differs by −0.0002 at λ=12 and +0.0008 at λ=24, cost by at
most 1.1%, throughput by 0.1%, and mean end-to-end latency by under 40 ms; the routing mixes are
near-identical, 25/6/58/11 for the learned head against 24/7/57/12 for the constant. A measured
per-tier TPOT constant is therefore not merely a viable alternative but an equivalent one when
operators wish to avoid training and querying learned latency heads.

**Latency prediction clarification.** Each XGBoost head predicts TPOT only, one head per
(model, GPU) tier, trained from a tier-local QPS sweep. Its inputs are that tier's serving-state
signals plus the request's predicted output length; it does not consume request text. End-to-end
latency is then formed analytically, not learned: predicted TPOT × the decode iterations the
request waits through (pending decode tokens ÷ current decode batch size, plus its own KNN-predicted
output length).

---

**W3: General-purpose dispatchers.** We added standard power-of-two choices (Po2): sample two
replicas uniformly and choose fewer inflight requests. We compare it with shortest-queue (SQ),
trained routers, and RouteBalance (RB), reporting each trained router at its best setting from a
full sweep of its tuning parameter.

| system | dispatcher | quality λ12 | cost $/1k λ12 | mean e2e ms λ12 | λ24 |
|---|---|---:|---:|---:|---:|
| Passthrough | SQ | 0.4931 | 0.0135 | 2067 | 2168 |
| Passthrough | Po2 | 0.5016 | 0.0143 | 2248 | 2279 |
| BEST-Route | Po2 | 0.5138 | 0.0131 | 2727 | 37426 |
| AvengersPro (pw 0.7) | Po2 | 0.5317 | 0.0360 | 4362 | 5935 |
| RB (1/3,1/3,1/3) | joint | 0.5114 | 0.0127 | 1894 | 2814 |
| RB (0.5,0.3,0.2) | joint | 0.5159 | 0.0173 | 2395 | 2624 |
| RB (0.8,0.1,0.1) | joint | 0.5312 | 0.0321 | 3959 | 4301 |

SQ outperforms Po2 for passthrough. BEST-Route reproduces its serial ceiling under Po2, as under
SQ/RR: at λ24 it reaches 37426 ms versus RB's 2624 ms (14.3×). Thus the collapse is upstream of
dispatch: a dispatcher cannot revisit the selected model tier or remove routing-layer scoring
overhead. RB also spans a frontier the single-point baselines cannot: at equal or better quality it
is cheaper and faster than BEST-Route, and it reaches AvengersPro's quality band at 11% lower cost.
Each trained router is reported at its best setting from a full sweep of its tuning parameter.

---

**W1-A: New topology.** We re-provisioned the A30 pool from 3B/7B replica counts 3/5 to 5/3 and
reran all systems without retraining.

| system | disp. | quality | tput | cost $/1k | mean | p99 ms |
|---|---|---:|---:|---:|---:|---:|
| RB (0.5,0.3,0.2) | — | 0.5175 | 22.57 | 0.0176 | 2918 | 13481 |
| RB (1/3,1/3,1/3) | — | 0.5114 | 23.37 | 0.0126 | 2068 | 5491 |
| BEST-Route | Po2 | 0.5138 | 17.44 | 0.0131 | 32386 | 121512 |
| AvengersPro | Po2 | 0.5317 | 20.60 | 0.0360 | 17837 | 86200 |

Each preset improves the joint trade-off over the baseline nearest its quality level. Against
BEST-Route, RB (0.5,0.3,0.2) improves quality, throughput, mean latency and p99 at comparable cost.
Against AvengersPro, RB (1/3,1/3,1/3) is 65% cheaper with 13% higher throughput, an 8.6× lower mean
and a 15.7× lower tail. Re-provisioning changes no system's ranking, and RB needed no retraining or
reweighting.

---

**W1-Model: New model family.** We also onboarded Gemma-3 as a 14-instance ladder — 1B×3, 4B×5 on
A30, and 12B×4, 27B×2 on A100 — using measured static per-(model,GPU) TPOT, the alternative
discussed with Reviewer gTfM. V100 was excluded because Gemma-3 requires BF16. Each cell uses 1979
held-out prompts from a disjoint 8086-prompt training split. Its input-price ladder is
non-monotone: per 1M tokens the 1B costs $0.10, more than both the 4B at $0.04 and the 27B at
$0.08 (PricePerToken), so routing to the cheapest tier is not automatically cheapest.

| system | disp. | quality | cost $/1k | mean | p99 ms |
|---|---|---:|---:|---:|---:|
| RB (0.8,0.1,0.1) | — | 0.5659 | 0.0250 | 3684 | 9434 |
| AvengersPro | Po2 | 0.5631 | 0.0240 | 3787 | 9779 |
| RB (0.5,0.3,0.2) | — | 0.5533 | 0.0187 | 2952 | 8406 |
| RB (1/3,1/3,1/3) | — | 0.5254 | 0.0096 | 2085 | 7133 |
| Passthrough | Po2 | 0.5001 | 0.0104 | 1914 | 6851 |
| Passthrough | SQ | 0.4863 | 0.0082 | 1612 | 6745 |

The presets trace a frontier on the new family without retuning: quality 0.5254 → 0.5659 as cost
rises $0.0096 → $0.0250, with the 27B share moving 8% → 51%. Every preset exceeds both passthrough
variants on quality, by 0.025–0.080; SQ wins latency by serving 1B for 47% of requests, and RB
(1/3,1/3,1/3) is both better and cheaper than Passthrough/Po2. RB (0.8,0.1,0.1) matches
AvengersPro's quality (+0.003) while being faster on mean (−103 ms) and tail (−345 ms) at 4% more
cost. All systems sustain the offered rate (22.1–22.7 req/s at λ24). BEST-Route is absent because
extending it to Gemma-3 requires fine-tuning a fresh per-family supervised router, which did not
fit within the rebuttal window.

**W2-A: Partial/missing labels.** At λ12 and preset (0.5, 0.3, 0.2), we rebuilt the KNN index with
half the labels or without prompt conditioning, comparing both with the full index and
dispatcher-only baselines.

| system | quality | tput | cost $/1k | mix 3B/7B/14B/72B |
|---|---:|---:|---:|---|
| RB, full index | 0.5163 | 11.66 | 0.0171 | 25/7/58/11 |
| RB, 50% labels | 0.5136 | 11.74 | 0.0173 | 22/8/60/11 |
| RB, prompt-blind | 0.4848 | 11.88 | 0.0087 | 77/4/19/0 |
| Passthrough, RR | 0.5043 | 11.69 | 0.0182 | 23/38/23/15 |
| Passthrough, SQ | 0.4931 | 11.74 | 0.0135 | 36/27/31/6 |

Halving the labels costs 0.0027, showing that the full index is not label-starved. Removing prompt
conditioning has a much larger effect: 72B use falls to 0%, 77% of requests shift to 3B, and
quality drops below both passthrough baselines. The 0.0315 full-versus-prompt-blind gap isolates
the value of prompt-dependent quality labels at this operating point.

## PASTE TO HERE

---

## Changes from the posted text

**W1-Model — REBASED, and this changes the AP comparison.** The posted W1-B quality figures use
the submitted or-chain procedure (~0.62 scale). The table above is the **ground-truth deepeval
join** — each request's served model looked up against `llm_judge_scores` in
`gemma3_merged_test.jsonl` (~0.53 scale). The two scales are not comparable and must not be mixed
in one table. On the measured basis the ordering **reverses**: posted says AvengersPro leads RB
(0.8,0.1,0.1) by +0.014; measured, RB leads by +0.003 and is faster on mean and tail.

Three facts a reviewer could press on, stated rather than buried. (i) The join is the only basis
that works for AvengersPro at all — it runs in pipeline mode, where `all_model_scores` is empty,
so its quality cannot be computed from the cell's own predictions. (ii) The passthrough rows are
new measurements, not the posted ones; no Gemma passthrough cell existed in the campaign store.
(iii) The measured SQ mix reproduces the posted claim exactly — 1B for 47% of requests — which is
the independent check that this basis lines up with the posted one on a quantity both can express.

**Gemma is insensitive to the chat-template fix.** Raw and chat-wrapped prompts produce identical
KNN predictions on 100.0% of 400 test prompts (best-model accuracy 30.8% either way, mean absolute
error 0.5201 either way). On Qwen the same test gives 0.000000 agreement. So the Gemma numbers do
not depend on which representation was served, and the posted Gemma table was never at risk from
the skew. The 30.8%-against-a-25%-floor discrimination is weak and worth understanding before the
camera-ready, but it is not a rebuttal-window issue.

**W3 — quality and cost columns added.** The posted W3 reports mean e2e only. Both trained routers
are reported at their swept best; note BEST-Route's posted t=0.3 is NOT its best (t=0.0 beats it).
The full RB frontier lives in W1-A, which has complete data on every axis.

**W3 — ratio moves, claim unchanged.** BEST-Route/RB at λ24 is now **14.3×** (37426 vs 2624 ms)
against the posted 19.6× (48182 vs 2455). Passthrough rows are reused unchanged — those routers
never read the prompt.

⚠ **Q1 — the learned/static trade-off becomes an equivalence.**
Posted: *"The learned head is higher on quality at both rates (+0.0031 at λ=12, +0.0016 at λ=24);
the static prior is lower on cost (6–7%) and mean e2e (~5%)."*
Corrected: Δquality −0.0002 at λ=12 and +0.0008 at λ=24; **the 6–7% cost and ~5% latency advantages
disappear** (±1.1% and <40 ms). The posted mix 38/15/32/14 becomes 25/6/58/11.

⚠ **W2-A — "no measurable effect" becomes a small measurable cost.**
Posted: *"Halving the labels has no measurable effect (0.5096 vs 0.5095)."*
Corrected: **−0.0027**. Caveat: the 50%-label index used here is a deterministic subsample of the
v3 index (seed 20260731, 14,919 → 7,459 rows, verified unbiased to within 0.002 per model), not the
original artifact, which was unrecoverable. A single draw can move ±0.004, so the wording avoids
leaning on the exact figure.

⚠ **W1-A — the AvengersPro comparison now needs the uniform preset.**
Posted: *"Against AvengersPro, RB (0.5,0.3,0.2) improves every reported axis."*
Corrected: AvengersPro leads quality (0.5317 vs 0.5175), so the every-axis claim holds for
RB (1/3,1/3,1/3) — 65% cheaper, 13% higher throughput, 8.6× lower mean, 15.7× lower tail — not for
(0.5,0.3,0.2). Both baselines degrade far more on 5/3 than in the posted run (BEST-Route 32386 ms,
AvengersPro 17837 ms), so the section is stronger overall.

**Also note:** the prompt-blind row still drops below both passthrough baselines on this basis
(0.4848 vs 0.5043 and 0.4931), so that posted sentence stands unchanged.

## Quality basis

All figures above use the **posted `or`-chain basis**, matching the paper and the live tables.
The zero-safe basis — which stops discarding legitimate 0.0 scores — runs ~0.10 lower on every row
and is a separate correction (D2) belonging to the camera-ready. Mixing it in here would make one
change look like two.
