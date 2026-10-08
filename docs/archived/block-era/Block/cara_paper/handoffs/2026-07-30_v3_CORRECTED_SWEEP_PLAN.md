# v3 CORRECTED SWEEP — audit, fix, full rerun → `v3-exp` → clean v3 paper

Author ask (2026-07-30): **full audit → fix all issues → rerun all → `v3-exp` pack → a clean v3
paper built from the v1 content + the rebuttal content.** Key references = NeurIPS submitted
version + reviewer-requested experiments + our rebuttal cells. EuroSys/v2-only extensions
(e.g. enhanced baselines) = optional reference, not required.

## Cluster budget (author, 2026-07-30) — 57 h gross, NOT 57 h usable

| slot | window | gross | setup | usable |
|---|---|---:|---:|---:|
| **S1** | now → Jul 31 18:00 | 24 h | none — cluster live | **~24 h** |
| *gap* | Jul 31 18:00 → Aug 03 01:00 | 55 h | — | **0 h cluster / 55 h offline** |
| **S2** | Aug 03 01:00 → Aug 04 12:00 | 33 h | **re-setup required** | **9–29 h** |

No further reservation after S2.

**The setup risk on S2 is the largest single unknown.** `ENVIRONMENT/README.md:5` records:
*"Day one of the Jul-25 lease was lost entirely to environment setup."* The recipe is now
author-approved and battle-tested (`setup_jul25.sh`), so a repeat should be far faster — but I
cannot promise that, and R2 forbids improvising around a setup failure. Budget 4 h optimistic,
24 h worst case. Usable cluster across both slots is therefore **~33–53 h**, not 57.

S1 state verified 2026-07-30: **13/13 hosts up, full Qwen v1 ladder deployed** (72B×1, 14B×3 on
c4130, 7B×5 + 3B×3 on d7525, node0 = d8545-10s10505). No model redeploy needed in S1.

---

## 0. Does "rerun all" fit?

Measured, not estimated. Cell duration is **arrival-bound** — `num_prompts / λ` dominates, so the
cap 256→1024 change does *not* inflate typical cells:

| λ (n=3534) | measured v1 raw | 3534/λ |
|---|---:|---:|
| 6 | 9.9 min | 9.8 min |
| 12 | 5.0 min | 4.9 min |
| 24 | 2.5 min | 2.5 min |

286 banked v1 cells = **20.1 h raw**, mean 4.2 min. The exception is saturating cells — under cap
1024 the quality-priority and BEST-Route-collapse cells blow out (drift audit: wq0.8 λ24 mean e2e
2.8 s → 14.5 s; v2 log: *"br4 λ18/24 collapse cells slow"*, 50 s mean e2e at λ24). Reserve
15–50 min each for ~30 such cells.

**Full v3 grid:**

| block | cells |
|---|---:|
| paper grid, v2-equivalent (`AGGREGATE.csv` = 287 cells) | 287 |
| rebuttal delta — absent from v2 entirely (see §1) | 72 |
| Gemma family on v3 basis | ~24 + pipeline |
| **total** | **~380** |

≈ **30 h raw**. At a well-oiled chain's 1.5× overhead → **45 h**; at the July repair phases' 2.5×
→ **75 h**.

**Verdict: 45 h needed against 33–53 h usable. A complete 380-cell v3 grid does not reliably
fit.** It fits only if S2 setup is fast *and* the chain runs clean and unattended. I am not going
to plan as if both hold. §4 sets a priority order so that whatever completes is a coherent paper
rather than a truncated grid — but the ordering is an author call, not mine (R7).

---

## 1. What the audit and rerun are actually against

**The paper side is already corrected once.** The v2 campaign (June, audit-validated) fixed every
submitted-run defect and re-ran the paper's grid; `rb-exp/v2-exp` (15 GB, 287 cells) holds
corrected Fig 2 / extremes / bootstrap / multiseed / dominance / tail / appendix.

**The July reviewer-requested work exists on the v1 basis only.** Verified 2026-07-30:
`v2-exp/results/` contains **zero** Po2/dispatcher, topology, Gemma, or label-ladder cells. PATH
v4 (author, 2026-07-25) ran every rebuttal cell on the submitted v1 basis deliberately, so the
rebuttal stayed basis-coherent with the paper.

| requested item | corrected-basis status | v3 action |
|---|---|---|
| Fig 2 / Pareto hull | v2 (`AGGREGATE.csv` → `make_fig2_fig3_final.py`) | rerun on v3 |
| Table 3 headline extremes | v2 (`tab_extremes` ← `consolidate_v2.py`) | rerun on v3 |
| bootstrap CI | v2 (`tab:bootstrap` + `tab_multiseed` ← `multiseed_v2/`) | rerun on v3 |
| iso-quality conclusion | **not yet located** — audit task A4 | derive + rerun |
| appendix (predictor, tail, k-sens, third-judge, safety, cost-sens, tier-loss) | v2 `_aggregate/` | rerun on v3 |
| static-vs-learned, 7 rates | **v1 only** (v2 has λ18 + sq40 only) | rerun on v3 |
| dispatchers Po2 / SQ / RR | **v1 only** | rerun on v3 |
| topology re-provisioning | **v1 only** (`topology_flip_v1x`) | rerun on v3 |
| partial-label ladder | **v1 only**; v2 has R4 prompt-blind only | rerun on v3 |
| Gemma family | **v1 only**, ladder torn down, 12 systems × **2 rates** | rerun on v3 |

The Jul-23 audit named this gap itself (§6.5) and sized the minimum version of it — *"bank
v2-recipe duplicates of {Po2, Gemma sample, arm extensions} in the Jul 29–31 bonus slot (~30–40
cells, ~0.5 d, cluster already live)"*. That slot is now; the rebuttal consumed it. The author's
current ask is the maximal version: rerun everything on one clean basis.

---

## 2. What v3 is

**Author rule, 2026-07-30: v3 = v1 (plus the rebuttal extensions) with defects fixed. The config
and experiment setup are KEPT unless a fix is actually needed** (author's example: remove the
P100 entries). **v3 is therefore NOT v2.** v2 changed several things that were improvements
rather than defect repairs; those do not carry over automatically.

| v1 setting | defect? | v3 action |
|---|---|---|
| serving cap **256** | **No** — it is vLLM's own custom-dataset default (`datasets.py:1380`) and the submitted paper's setting | **KEEP** |
| corpus generation cap **1024** | **Yes — as a mismatch.** Corpus built at 1024, server caps at 256 | **FIX the mismatch** — direction is an author call (§2a) |
| routing on `score_type=judge` while **reporting deepeval** | **Yes** — routes on a signal correlating r=0.22 with the reported metric | **FIX → deepeval** |
| KNN **k=50** | **No** — evidenced-minor; R5 band ~0.013 over k∈{5,10,20,50}; it is what the paper describes | **KEEP 50** (v2's k=10 does not carry over) |
| `quality_norm` **raw** | **No** — it is what the paper's printed Eq. 1 states | **KEEP raw** unless the audit shows a bug |
| A30 latency tier first-match-**P100** | **Yes** | **FIX** |
| **P100 entries** in instance capacities | **Yes** — stale; no P100 in the leased topology (c4130=V100, d7525=A30, d8545=A100) | **REMOVE** |
| latency XGB trained pre-1024 | contingent on §2a | contingent |
| aggregator `or`-chain drops 0.0 scores (= the paper's 72–78% "coverage") | **Yes** | **FIX → zero-safe** |

Sources: `2026-07-23_neurips_rebuttal_audit.md` §2/§6.4, `2026-07-25_v1_v2_drift_audit.md` §1–3.

The Jul-25 audit anticipated exactly this configuration (§6.4): *"~5–10 cells on the v1 stack
with ONLY `score_type=deepeval` flipped (k=50 kept) → isolates the fix on the submitted config
for a minimal-delta CR story."* v3 is that, taken to the full grid.

**Consequence: v2's corrected headline does not transfer.** RB 0.4184 vs BR 0.4056, CI
[+0.005,+0.022] was measured at k=10 / minmax / cap 1024. v3 keeps k=50 and raw norm, so it must
establish its own numbers — which is the point of the rerun.

### 2a. The cap mismatch — NEW finding, not on the CR list

Measured 2026-07-30 directly from the artifacts:

| | cap | at-cap rate |
|---|---:|---|
| offline corpus (`gemma3_merged_train.jsonl`, 32,344 pairs) | **1024** | max observed 1023; 0 truncated (filter removed them) |
| online serving, Qwen `v1x_ap_po2_l12_n3534` | **256** | **841 / 3,534 = 23.8%** |
| online serving, Gemma `gemma_rb_uniform_l12_n1979` | **256** | **610 / 1,979 = 30.8%** |

The KNN length estimator is fitted on a corpus whose lengths run to 1024 with zero truncated
examples, then deployed against a server that stops at 256. It therefore over-predicts length for
roughly a quarter of requests, and over-predicts *most* for the verbose large tiers. Predicted
length feeds **both** the analytic e2e latency term and the cost term.

Cross-system comparisons are not invalidated — every arm runs the same prompts under the same
256 cap, so the cap is a shared constant. But RouteBalance is the only arm that *consumes*
predicted length, so the bias is not symmetric across arms.

**Two directions to fix it; author's call:**
- **(a) raise serving 256 → 1024.** What v2 did. Costs ~20% more cell wall (v2 median 5.0 min vs
  v1 4.2 min). Keeps both corpora as built. Changes latency/throughput absolutes substantially
  (drift audit: wq0.8 λ24 mean e2e 2.8 s → 14.5 s), so v1↔v3 latency stops being comparable.
- **(b) rebuild the corpus at 256.** Preserves v1 latency comparability, but requires regenerating
  **and re-judging** both families, and would shrink the corpora hard — ~24–31% of responses
  exceed 256, so the "drop any prompt where any model hit the cap" rule would remove far more than
  the 45.6% it removes at 1024.

Recommendation: **(a)**. (b) costs more cluster time than the whole rerun budget.

### Why the rerun has to be on the corrected *system*, not just re-aggregated

From `V1_REAGG_BOTH.csv` (175 cells, Jul-23):

- Buggy retrieval **reproduces the submitted numbers exactly** (λ12 RB wq0.8 0.5141 @ cov .782;
  all four extremes peaks; appendix argmax cell 0.5089) — the audit chain is validated.
- Corrected zero-safe **on the v1 config**: RB peak 0.4042 vs BR t0.5 0.4056 → the submitted
  **"+0.006 peak-quality" edge over BEST-Route inverts**.
- Corrected **on the v2 stack**: RB 0.4184 ± 0.0003 (4-seed) vs 0.4056, paired CI
  **[+0.005, +0.022]** — restored and separated.

Fixing the accounting alone breaks the headline; fixing the system restores it. That is the
argument the v3 paper carries.

Quality is deterministic per (prompt, model) → must reproduce exactly on rerun. Latency is not,
and v1↔v3 latency absolutes are non-comparable (cap-driven). Every table carries its basis label.

---

## 3. The audit — split by whether a finding can invalidate a *cell*

The user's order is audit → fix → rerun, and it holds where it matters. But the layers differ in
consequence, and idling a live cluster for a layer that cannot invalidate a cell wastes S1.

### 3a. GATING — must finish before any v3 cell runs (offline, S1 h0–6)

| # | layer | check | evidence required |
|---|---|---|---|
| A3 | index / predictor artifacts | KNN label column, k, corpus size, train/test leakage; XGBoost heads per (model,GPU) | rebuild, diff md5 vs deployed |
| A4 | aggregation | every `.get(x) or .get(y)`, every mean over a possibly-empty set, cost convention (CR-2), **locate the iso-quality derivation** | re-aggregate all 286 v1 + 287 v2 cells both ways |
| A5 | serving / routing | hardcoded family/tier/GPU assumptions beyond the known Port-#4 Qwen ladder | `/proc/PID/cmdline` on a live cell — **needs S1 cluster** |
| A6 | harness | pinned vLLM fork `2894a9872` — sampling params, seed handling, `max_tokens` plumbing | compare live invocation to recorded recipe — **needs S1 cluster** |

A5/A6 are the reason S1 is not wasted during the audit: they can only be done against a live
server, and the server is live *now*.

### 3b. NON-GATING — a finding invalidates a table, not a cell (the 55 h gap)

| # | layer | check |
|---|---|---|
| A1 | corpus prep | `prepare_benchmark_data.py` filter/truncation/join; silent field defaults |
| A2 | labelling | `score_with_deepeval.py`, `llm_judge_scorer.py` — known silent `is_harmful → [False]*n`; any other silent rubric switch |
| A7 | table / figure generation | `consolidate_v2.py`, `make_fig2_fig3_final.py` — do plotted values equal aggregate values |

Rule throughout (memory `feedback_no_fabricated_explanations`): **no finding is recorded without
the file and line that proves it.** "Looks fine" is not a result; "checked X:NNN, clean" is.
Verify by execution, never by grep (`feedback_r2_no_env_improvisation`).

A4 also settles the question that must be answered while posting is still open: **does re-aggregating
everything zero-safe move a number we already posted?**

---

## 4. Rerun priority order

Ordered so that stopping at any point leaves a coherent paper. **This ordering needs author
sign-off — it is the scope decision, not a technical one.**

Prerequisite: swap node0 from the v1exact config to the v3 config (`score_type=deepeval`, k=10,
`quality_norm=minmax`, cap 1024, v2 latency heads). Models unchanged, no redeploy.

**Gates before any batch:**
- Reference cell (uniform ⅓, λ12) against the v2 pack's own `cara_wq0.33_wl0.33_wc0.33_l12`.
  Quality is deterministic → must match. Latency will not, and is not gated.
- **Free anchor:** BEST-Route quality is stack-invariant (0.4056 → 0.4056 exactly across v1→v2,
  deterministic per-prompt router). Any v3 BR cell not landing on 0.4056 means the basis swap is
  wrong. Run one first.

| P | block | cells | ~raw | why this rank |
|---|---|---:|---:|---|
| **P0** | basis gate: uniform λ12 + BR t0.5 λ12 | 2 | 10 m | nothing is valid until this passes |
| **P1** | main grid core — Fig 2 Pareto hull + Table 3 extremes | ~90 | 8 h | the paper has no headline without it |
| **P2** | rebuttal delta: static-vs-learned ×7λ, time-matched triples | 32 | 2 h | gTfM Q1 / AC priority #1; exists on no corrected basis |
| **P3** | rebuttal delta: dispatchers Po2/SQ/RR | 18 | 1 h | faAU / AC priority #2; same |
| **P4** | bootstrap CI + multiseed | ~45 | 4 h | the CI claim is load-bearing in the rebuttal |
| **P5** | rebuttal delta: topology, partial-label ladder, AP per-model prices (#11) | 20 | 1.5 h | faAU generality + cold-start |
| **P6** | appendix set (predictor, tail, k-sens, third-judge, safety, cost-sens, tier-loss, budget) | ~110 | 9 h | v2 covers these; last to lose |
| **P7** | Gemma family on v3 basis | ~24 + pipeline | 6 h+ | needs a full ladder swap — S2 only, if P0–P6 are done |

P0–P5 ≈ **17 h raw** → fits S1. P6–P7 → S2.

**Gemma is the realistic casualty.** It needs Qwen down, Gemma up, generation, judging, index
rebuild — and the family currently has 2 rates against Qwen's 7. If P0–P6 run long, P7 does not
happen, and the v3 paper carries the Gemma family on the v1 basis with an explicit basis label,
or not at all. Flagging now rather than discovering it at Aug 04 11:00.

---

## 5. Timeline

| when | cluster | offline |
|---|---|---|
| S1 h0–6 | A5, A6 (live-server audit) | A3, A4 — the gating audit |
| S1 h6–8 | fixes applied; v3 stack assembled; P0 gate | A4 verdict: does anything posted move? |
| S1 h8–24 | **P1 → P5** | fix-verification; two-location sync per batch |
| gap (55 h) | *no cluster* | A1, A2, A7; all fixes landed + committed; `v3-exp` pack skeleton; v3 paper drafted from v1 + rebuttal content; S2 runbook + `setup_jul25.sh` dry-read |
| S2 h0–4 (–24 worst case) | **re-setup via `setup_jul25.sh` only** | — |
| S2 → Aug 04 10:00 | **P6 → P7** | rolling aggregation |
| Aug 04 10:00–12:00 | teardown | `closegate.sh v3-exp` → verify → offsite |

**Hard stop Aug 04 12:00.** Nothing may be mid-flight at 11:00 — R2/R4 and
`feedback_keep_deployed_models` (preserve exact deployed models to ≥2 locations + manifest
*before* release).

The 55 h gap is the most valuable block in this plan: it is free, uncontended time for the audit
tail, every fix, and the entire v3 paper draft. The cluster slots should carry nothing that could
have been done in the gap.

---

## 6. Decisions needed before go (R3 — not mine to make)

1. **v3 = v2 + audit findings.** Confirm, or name a different definition.
2. **Priority order in §4**, given that P0–P7 does not reliably fit 33–53 usable hours. Approve
   the ordering, or reorder it.
3. **Gemma (P7)**: accept it may not happen, or promote it above P6 (costs the appendix set).
4. **Basis swap in S1** ends the ability to run a *new* v1-basis cell before posting closes
   Aug 03. Only the 337 banked v1 cells remain answerable-from. Accept?
5. **If the audit finds a bug touching a posted number** — does silent-in-rebuttal /
   truthful-if-asked / corrected-at-CR still hold for a bug found *after* posting?

## 7. Standing constraints

- Internal codename never leaves the repo; zero occurrences in any artifact.
- No latency comparison across bases or sessions; every table carries its basis label.
- 3-way invalidation: v1 / v2 / v3 numbers never mixed in one table.
- Never re-assert the two basis-dependent claims (Jul disclosure audit, rail 1).
- Screen every near-capacity latency cell for non-stationarity; re-run failures at the same seed.
- Fix the script, not the symptom; one launcher file per unit of work.
- No bulk transfers while cells are live; incremental result syncs remain required.
- S2 environment: `setup_jul25.sh` only. First post-setup failure = stop and report.
