# Codex evaluator-audit question set — answers (technical evidence only; 2026-07-25)

Scope note: "old/submitted" = the May v1 pipeline behind the reviewed NeurIPS PDF (243 configs);
"v2" = the June rerun stack. Full drift inventory: `2026-07-25_v1_v2_drift_audit.md`.

## A. What exactly differs between the old and fixed evaluators?

**A1. Complete definition of the submitted evaluator.**
Producers: `cara_paper/smoke_test_apr_13/scripts/aggregate_full_215.py` (cell means → AGGREGATE_FULL)
and `scripts/bootstrap_ci.py` (App-A CIs) — identical retrieval logic (bootstrap_ci.py:74-78).
- Inputs: per-cell serving trace (`response_details[]`: chat-formatted `prompt`, `model`, `error`, …)
  + offline lookup `data/cara/scored/test_scored_filtered.jsonl` (3,634 prompts × 4 models, key
  `deepeval-llama3.1-8b-it_reference`).
- Lookup build (`aggregate_full_215.py:44-60`): stores per-prompt dict with keys `f"Qwen/{model_name}"`
  and `model_name`. The scored file's model names ALREADY carry the `Qwen/` prefix, so the stored keys
  are `"Qwen/Qwen/…"` (junk) and `"Qwen/…"` (correct); the bare stripped form (`"Qwen2.5-7B"`) is
  NEVER stored.
- Retrieval (L178-190): unwrap chat template → `per_model = lookup.get(prompt)` (miss ⇒ silently
  excluded) → `score = per_model.get(m) or per_model.get(m.replace("Qwen/",""))` → include iff
  `score is not None`.
- Missing-pair handling: prompt-miss → excluded. Pair PRESENT with score **0.0** → `0.0` is falsy →
  `or` falls through to the never-stored stripped key → `None` → **excluded**. Non-zero → included.
- Coverage (L190): `matched / total_non-error_responses`.
- Mean: per cell over that cell's own included subset ⇒ each system is averaged over
  (its served prompts) ∩ (lookup hit) ∩ (score ≠ 0.0). Effective statistic:
  **E[deepeval | served, matched, score>0]** — a zero-truncated conditional mean, with the truncation
  fraction system-dependent (λ12 coverage: RB 78.2%, BR 78.0%, AP 77.4%, PT 72.2%).

**A2. What the fix changed.**
- Class: **missing-value / data-join handling bug** (Python `or` on falsy 0.0 combined with asymmetric
  key storage). NOT a metric-definition change.
- Fix artifacts: public repo `route-balance` commit **`4aeb956e752c`** (2026-06-11),
  `reproduce/aggregate_results.py`, message verbatim: "zero-safe DeepEval match (explicit None, not
  'or') — was dropping 855 valid 0.0 scores, coverage 0.758→1.0, quality 0.489→0.371". Block-side:
  `aggregate_v2_grid.py` (zero-safe), bundled in checkpoint **`2074f4a`** (2026-06-07; handoff
  `2026-06-07_v2_rerun_handoff.md:10` "aggregate_v2_grid.py (zero-score fix)"). Old producers remain
  on disk unfixed: `aggregate_full_215.py:185`, `aggregate_overnight_grid.py:129`, `bootstrap_ci.py:78`.
- Pseudocode. OLD: `s = pm.get(m) or pm.get(strip(m)); if s is not None: keep(s)`.
  NEW: `s = pm.get(m); if s is None: s = pm.get(strip(m)); if s is not None: keep(s)`.
- Consistent with the paper's description? **No.** Submitted App-A defines coverage as "the fraction
  of the cell's served prompts that map to a scored (prompt, model) pair in the offline lookup"
  (submitted tex L318). The dropped pairs ARE in the lookup, with value 0.0. Implementation ≠ stated
  definition.

**A3. Deterministic / consistent / ordering effects.**
- Deterministic + reproducible: **yes** — pure function of (trace, lookup). Verified bit-exact:
  buggy re-run reproduces every submitted number (extremes 0.5181/0.5119/0.5101/0.5093; λ12
  0.5141/0.5089/0.5101/0.5037; App-A means exact; CI edges differ only via 1k-vs-10k resamples:
  paper RB [0.5052,0.5234] vs recomputed [0.5051,0.5231], seed 42).
- Internally consistent across systems: **no** — each system is scored on a different, endogenous
  subset (its own non-zero-scored served pairs; exclusion rate depends on the routing under test).
- Effects: absolute scores −0.109…−0.140 at λ12; **ordering CAN change** — corrected-v1 peak: RB
  0.4042 vs BR-t0.5 0.4056 (the submitted +0.006 RB edge inverts to −0.0014); RB-vs-AP gap WIDENS
  under correction (λ12 0.0040→0.0071; peaks 0.0092). Full table:
  `rb-exp/v1-exp/results/_reagg_corrected/V1_REAGG_BOTH.csv`.

## B. Offline reporting only, or online too?

**B4.** Post-hoc aggregation **only**: cell means + App-A bootstrap. Non-involvement evidence:
- KNN training labels: `block/predictor/cara/offline_training/*` read `llm_judge_scores` via direct
  dict access per record (e.g. `evaluate_predictors.py:49`, `compute_reference_score.py:143`,
  `analyze_judge_scores.py:310`) — no cross-key `or`-join exists in the training path; 0.0 labels are
  preserved in training data.
- Online prediction/routing: serving-time quality = KNN output (`cara_serve.py:1511` raw predicted
  quality); the deepeval lookup is never loaded by the scheduler.
- Budget filtering: runtime fields per request (`budget_cost`, `budget_exhausted`,
  `affordable_output_tokens` — verified in `budget_calibrated_demo/*.json` response_details);
  Table-6 rates count the runtime flag (`aggregate_full_215.py:172-173`). Evaluator-independent.
- Operating-point selection: **affected** — the highlighted "peak" cells were chosen by ranking
  BUGGY means. Under correction, RB's peak family is unchanged (wq0.8; peak cell λ18), but the
  cross-system peak comparison vs BR flips (A3).

**B5.** With runtime/logs/k=50/prompts/seeds held fixed, changing only the evaluator:
**everything quality-related is offline-recomputable and HAS BEEN** (V1_REAGG_BOTH.csv, 175+28 cells;
dual-basis bootstrap). Latency, throughput, tier mix, overhead, budget-exhaustion are measured fields —
unchanged by construction. **No serving reruns are required for any re-scoring.**

**B6.** KNN training data identical in both pipelines: `data/cara/train_scored_filtered.jsonl`
(14,919 prompts × 4 models; Apr-5). v1 estimator `deploy_knn` metadata: k=50, num_examples=14,919;
v2 `deploy_knn_k10`: k=10, 14,919. Label values unchanged (same records; trainer never uses the
or-join). What changed between eras is the routing-signal **config** (`score_type: judge→deepeval`)
and k — those DO change routing decisions, but they are stack changes (C7), not evaluator-coverage
effects, and not label edits.

## C. Confounding between the evaluator fix and k

**C7. Complete submitted→v2 change inventory** (all verified; details in the drift audit):
zero-safe evaluator; `score_type` judge→deepeval; k 50→10; `quality_norm` raw→minmax; generation cap
256→1024 (empirical: v1 max output_len=256 with 738/3,534=20.9% AT cap in uniform-λ12; v2 max 1024,
≈1%); 16-bucket length + fused-length heads ADDED (absent from v1 config); default scoring weights
(0.3/0.2/0.3/0.2-balance → ⅓/⅓/⅓/0); a30 latency-tier first-match-p100 fix; batched-FAISS KNN;
`client_ttft` field added; AvengersPro shim fix; arm gates (`CARA_TIEBREAK`/`CARA_STATIC_LAT`) added
(0 occurrences in v1-era code, rb mirror `e2d9f4d`); APC off (May, verified) → default-on (v2);
param grids normalized ("silently ≠ May → fixed", handoff:41). 48/66 config keys identical
(incl. batch_config {8,100 ms,adaptive}, budget_tokens 256, capacities, kv, prices). Latency heads
UNCHANGED across eras (same Mar-27 `deploy_{ttft,tpot,e2e}`; v1 per-tier configs and v2 md5 manifest
point at the same files).

**C8. Fixed evaluator + submitted k=50: EXISTS** (offline re-score of the as-run k50 system).
Headline (no-budget grid):
| | buggy (submitted) | fixed, same runs |
|---|---|---|
| RB peak | 0.5181 @17.25 req/s | 0.4042 @17.25 |
| BR peak (t0.5; λ-invariant — deterministic router, an internal validation) | 0.5119 | 0.4056 |
| AP peak | 0.5101 | 0.3950 |
| PT peak | 0.5093 | 0.3691 |
λ12 four-cell means + 10k CIs: RB 0.4021 [0.3922,0.4120] · BR-argmax 0.3969 [0.3870,0.4070] ·
AP 0.3950 [0.3850,0.4048] · PT 0.3642 [0.3542,0.3745] (coverage 0.999–1.000 for all).
What does NOT exist: SERVING runs of k=50 + deepeval-routing (a different, config-level question;
~hours of cluster time if wanted). For the evaluator isolation itself, no further audit is needed.

**C9. k ∈ {5,10,20,50} under one fixed pipeline** (`_aggregate/R5_k_sensitivity.json`; deployed
KNNEstimator, judge-signal predictions, deepeval ground truth; decision-level, serving-free):
- bestacc: 0.3182 / 0.3479 / 0.3675 / **0.4041** (k=50 highest)
- quality-greedy routed quality: 0.4117 / 0.4210 / 0.4164 / **0.4245** (band 0.0128; k=50 highest;
  oracle 0.5822)
- Tier distribution / latency / cost per k at SERVING level: **not measured** — only k=50 (v1) and
  k=10 (v2) were ever served, and those differ by more than k (confounded with C7).
- Ordering-between-systems vs k: not directly measured. Note: the corrected-v1 RB-vs-BR peak margin
  (−0.0014) is far inside the k band (0.0128) — at that margin, k choice, seed, and evaluator basis
  are all material; the peak-quality comparison is fragile under any of these knobs.

## D. Claim-by-claim table
Footnote [†]: the "Fixed (k=10)" column is the v2 campaign — k=10 PLUS deepeval-routing, cap 1024,
minmax, zero-safe scoring. It is NOT an isolated k change; no isolated-k serving data exists (C9).

| # | Claim | Submitted | Repro w/ old code? | Fixed (k=50) | Fixed (k=10)[†] | Ordering preserved? | Rerun? |
|---|---|---:|---|---:|---:|---|---|
| 10 | RB peak quality | 0.5181 | YES (bit-exact) | 0.4042 | 0.4187±0.0003 (4-seed) | see #11 | no (offline) |
| 11 | BR peak quality | 0.5119 | YES | 0.4056 | 0.4056 | RB>BR: buggy YES (+0.006) → fixed-k50 **NO** (−0.0014) → v2 YES (+0.013, CI [+0.005,+0.022]) | no |
| 12 | AP peak quality | 0.5101 | YES | 0.3950 | 0.376 (v2 enhanced-AP best) | RB>AP preserved, gap WIDENS under fix | no |
| 13 | CIs & coverage | RB [0.5052,0.5234], cov 78.2% (1k, seed 42) | YES ([0.5051,0.5231] @10k) | 0.4021 [0.3922,0.4120], cov 1.000 | paired RB−BR [+0.005,+0.022] | CI method sound; basis changes | no |
| 14 | Max throughput | RB 28.85 / BR 18.73 | YES (measured) | unchanged | 27.6 / 21.8 (cap-driven stack Δ) | preserved | no |
| 15 | Mean E2E (unif λ12) | 2,219 ms | YES | unchanged | 2,311 ms (cap) | preserved | no |
| 16 | Sched overhead | 141→220 ms (149@12) | YES | unchanged | 130→231 ms | preserved | no |
| 17 | BR collapse ratio | 33.1 s vs 220 ms @λ30 ("≈150×"; the "23×" figure is v2-era: 64 s vs 2.8 s) | YES | unchanged | reproduced (64 s @λ30) | preserved; real in both | no |
| 18 | bs=1 45.4/74.6 s | 45,438/74,579 ms (traces; sched_oh 43.4/72.5 s; tput plateau 11.21/12.37) | YES | unchanged (evaluator-indep.) | v2 CODE: 2,412/4,042 ms (`bbs1_l16/l24`) — magnitude is implementation-specific (E21) | qualitative penalty preserved (v2 sched_oh 1,693 vs 228 ms @λ24); 19–31× magnitude does NOT transfer | only if a v2-code figure is wanted |
| 19 | Budget exhaustion 32.4/19.2/12.6% | runtime flags | YES | unchanged (evaluator-indep.) | v2 budget rerun exists (18 cells, stack Δ) | qualitative finding (filter dominates) preserved | no |
| 20 | Four-arm XGBoost-vs-static | N/A — arms absent from v1 code | — | would need serving (planned window ext.) | arm1 2373/2597/2780 vs arm2 3329/3525/3887 vs arm3 3422/3499/3751 ms (mix 57/10/32/1 vs 38/19/30/14; within-tier wash +2.8/−0.7/−3.5%; fusion 26–31% faster arm3→arm1, equivalently arm3 up to 44% slower; 72B 14→1%); arm4 vs arm1: 2408 vs 2615 ms @λ18, q 0.36872 vs 0.370, overload 3123 vs 3788 | internal (single-config isolation) | v1-stack version requires runs |

## E. Independence of the two corrections

**E21. bs=1 collapse root cause: independent of the evaluator; a real property of the v1 code path.**
- Mechanism (arithmetic-consistent with traces): at bs=1 every request pays the full per-batch fixed
  cost serially (per-request MiniLM embed + FAISS k=50 query + XGBoost round + scoring). Observed
  throughput plateaus of 11.21 (λ16) and 12.37 (λ24) req/s ⇒ per-request scheduler service time
  ≈ 85–90 ms ⇒ service ceiling ≈ 11–12 req/s. Offered λ16/λ24 > ceiling ⇒ unbounded scheduler queue;
  `mean_scheduling_overhead_ms` = 43,364/72,486 (≈ the whole E2E) is queueing, not compute. bs=1@λ8
  (below ceiling) is healthy: 2,172 ms.
- Not a wrong path or config error: it is the deliberate bs=1 ablation of the as-built v1 estimator
  path; the numbers are reproducible from the archived traces and re-derivable (cost × λ > 1).
- Evaluator involvement: none (pure latency measurement).
- Why v2 code gives ≈2.4/4.0 s: the per-request path got faster (batched-FAISS KNN + related
  optimizations in checkpoint `2074f4a`), lifting the serial ceiling above 24 req/s — no divergence;
  the qualitative bs=1 penalty persists (sched_oh 1,693 ms vs 228 ms at bs=16, E2E +52% @λ24).
  Both figures are true measurements of their respective implementations.

**E22. The baseline-collapse result is real in both eras.** Submitted: BR pipeline-mode overhead
33.1 s @λ30 vs RB 220 ms (~150×). v2: br4 E2E 64 s @λ30 vs uniform 2.8 s (23×). The number evolved
with the stack (cap-1024 lengthens generations; enhanced baselines added); the phenomenon did not
change. The submitted paper ALREADY attributed the mechanism and separated the factors: "BEST-Route
runs one DeBERTa scoring call per request serially… once the arrival rate exceeds the classifier's
serving rate the scoring queue saturates" and "runs every baseline inside RouteBalance's pipeline
mode so the routing-policy gap is measured separately from this architectural one" (submitted tex
L263-265). v2 additionally quantified the split with concurrent-scoring ("enhanced") baselines
(2.6–4.5× residual gap).

## F. Final classification: **Category 2 — localized correctable defect**

Justification (evidence above): the defect is a single retrieval-join error in one function,
replicated in three post-hoc aggregation scripts; it touches NO training labels, NO online
prediction, NO routing decision, NO budget mechanics (B4); all runtime results (latency, throughput,
mix, overhead, collapse behavior, budget-exhaustion) are independently valid; and all quality
absolutes are offline-recomputable from preserved traces (done, C8). It is NOT Category 1: the
implementation contradicts the paper's own printed definition of coverage (A2), the project's own
public history labels it a bug (`4aeb956`), and it conditions each system on a system-dependent
subset — not a defensible protocol choice. It is NOT Category 3 in full: the comparison is not
globally invalid — the frontier's measured axes and most quality orderings survive correction
(RB>AP widens; RB>PT holds) — with one explicit Category-3-adjacent exception that must be stated:
**the specific peak-quality RB-vs-BR margin (+0.006) lies inside the defect's distortion and inverts
under correction on the submitted configuration** (restored, with CI separation, only after the
separate routing-signal change).

**Final explicit question — old-code reruns: reproducing the submitted protocol, or continuing a
known-incorrect implementation?** Technically **both, simultaneously; the two descriptions are not
mutually exclusive.** New cells scored with the old retrieval reproduce the submitted protocol AS
EXECUTED (bit-exact semantics; absolutes directly comparable to the printed numbers) — and that
protocol is a known-defective implementation (publicly documented as a bug in `4aeb956`) whose output
is a zero-truncated, system-dependent conditional mean. Consequences of using it for new results:
(+) comparability with the reviewed paper's numbers; (−) inherited inflation, system-dependent
conditioning, and unreliability of small (≲0.01) cross-system quality margins. Both properties hold
at once; which to weight is outside this audit's scope.

---

# PART 2 — CR impact assessment: evaluator-fix-only, submitted runtime + k=50 kept (2026-07-25)

Recomputation evidence (fresh, this audit): `2026-07-25_evaluator_audit_evidence/V1_REAGG_FULL_costs.csv`
— **all 203 no-budget routing cells** re-scored dual-basis with per-cell cost_per_req (token-priced,
`PRICE` table = paper's), p99, and tier mixes; plus paired per-prompt bootstraps (10k, seed 42).

## A. Runtime/serving measurements (items 1–15): ALL **Class 1 — Unaffected**
The aggregation path consults DeepEval ONLY for `deepeval_quality_mean`/`quality_coverage`
(`aggregate_full_215.py:178-190`). Every A-item is a direct trace field or runtime-derived:
1–2 `request_throughput`, `mean_e2el_ms` (trace top-level); 3–4 `mean_ttft_ms`, `mean_tpot_ms`;
5 `p99_e2el_ms` (also p50); 6 `mean_scheduling_overhead_ms` + per-request `scheduling_overhead`;
7–9 per-request `scheduling_overhead_breakdown` (batch-wait / predictor-compute / stats-fetch spans);
10 `completed`/`failed`; 11 per-request `model` + `instance_id` (routing distributions are trace-direct);
12 runtime telemetry consumed live, snapshots in breakdown/serve logs — never touches the lookup;
13 cost = (input_len+output_len)×per-token price (`aggregate_full_215.py:163-176` — no DeepEval);
14 max sustained = completed/duration; 15 stability/saturation = trace behavior (e.g. bs=1 45.4/74.6 s,
vLLM-SR failures). **Verified numerically unchanged under offline re-scoring** (re-aggregation
reproduces all latency/throughput/cost/mix values bit-identically while quality columns change).

## B. Architectural findings (16–26)
- **16 valid exactly** (141→220 ms sub-linear; traces).
- **17 valid exactly** — BR t0.5 E2E 2.5 s (λ12) → 5.5 s (λ18) → 27.7 s (λ24) → 41.9 s (λ30) with
  throughput ceiling 18.5–18.6 req/s (V1_REAGG_FULL_costs.csv); argmax overhead 1.1 s→33.1 s.
- **18 valid exactly** (AP 53 ms → 1.31 s @λ30).
- **19 valid, phrase narrowly**: the submitted text already attributes serial scoring and isolates it
  via pipeline mode (submitted tex L263-265); the QUANTIFIED policy-vs-architecture split
  (enhanced concurrent-scoring baselines, 2.6–4.5×) is v2 — Class 4, disclose separately.
- **20 valid; Class 2 for the decomposition**: bs-ablation + sub-linearity support amortization as
  submitted; the explicit per-request compute-decrease decomposition (44.7→23.9 ms) exists in trace
  breakdown fields and is recomputable, but was NOT printed in the reviewed PDF (post-submission tex).
- **21 phrase narrowly / requires additional experiment if quantified**: dead-reckoning-reduces-herding
  is design rationale; the nearest ablation (LPT-off, ±0.5%) is not a direct state-update A/B; no
  herding ablation exists in the v1 grid.
- **22 valid exactly** (code structure). **23 valid exactly** (traces + ~85–90 ms/req serial-ceiling
  arithmetic). **24 valid exactly** (latency measurement; evaluator uninvolved). **25 valid exactly**
  (runtime flags `budget_cost`/`budget_exhausted`/`affordable_output_tokens`). **26 valid exactly**
  (filter 32.4/19.2/12.6 vs no-filter 42.7/26.1/16.9, same runtime caps, runtime flags).

## C. Quality quantities (27–40): ALL **Class 3 — recompute** (DONE)
Recomputed in V1_REAGG_FULL_costs.csv / Part-1 C8. Confirmations: latency, throughput, and cost
COORDINATES of every cell are unchanged; only the quality coordinate (and hence pairings, hulls,
peak selection, dominance/iso-quality statements, radar quality axis, CIs, coverage) moves.
Corrected peak re-selection: RB's peak stays in the wq0.8 family (peak cell λ18, 0.4042) — the
highlighted preset family survives; cross-system peak identity changes (see D/E).

## D. Statements under fixed evaluator + submitted k=50 traces (41–48)
- **41 VERIFIED**: paired per-prompt RB(wq0.8)−BR(t0.5) at λ12: Δ=−0.0033, CI **[−0.0112, +0.0045]**,
  n=3,532 → statistically indistinguishable.
- **42 VERIFIED**: "parity/tie" is the accurate description (CI spans zero — NOT a BR win either).
- **43 QUALIFIED**: peaks 0.4042 vs 0.3950 (nominal +0.009), but paired λ12 RB−AP = +0.0074,
  CI **[−0.0021, +0.0171]** — includes zero. Accurate wording: "nominally higher; not statistically
  separated at matched cells."
- **44 VERIFIED**: RB−PT paired +0.0383, CI **[+0.0287, +0.0479]** — separated.
- **45 VERIFIED** (mechanism + 855-zeros commit evidence + magnitudes −0.109…−0.140).
- **46 VERIFIED**: the +0.006 peak claim must be removed (inverts to a tie).
- **47 VERIFIED**: the central systems case rests on measured axes (A-items) + E-results below.
- **48 proposed summary — ACCURATE WITH ONE QUALIFIER**: "matches the strongest router at peak
  quality" ✔; "better throughput" ✔ (27.84 vs 18.6 sustained); "better cost" ✔ at iso-quality
  (3.10–3.17e-5 vs 3.29e-5 $/req ≈ 4–6% cheaper, every λ); "better latency" — **only under load**
  (λ≥18: 2.7–3.1 s vs 5.5–41.9 s; at λ≤12 BR t0.5 is 100–440 ms FASTER); "robustness" ✔.

## E. Iso-quality & dominance, recomputed (49–55)
- **49**: the corrected iso-quality pairing is RB wq0.8 (0.4020–0.4042 across λ) ↔ BR t0.5
  (0.4056, λ-invariant; mix 0/0/98/2). Δ ≤ 0.0036, inside the paired CI.
- **50** (per λ; e2e ms RB vs BR; cost $/req 3.10–3.17e-5 vs 3.29e-5 at all λ):
  λ6 2502 vs 2093 · λ8 2543 vs 2222 · λ10 2594 vs 2372 · λ12 2648 vs 2535 · λ18 **2748 vs 5510** ·
  λ24 **2844 vs 27,734** · λ30 **3117 vs 41,908**; tput ceiling RB 27.84 vs BR 18.6.
- **51 faster at iso-quality?** λ≤12: **no** (BR faster by 0.1–0.4 s). λ≥18: **yes, 2×–13×**, and RB
  sustains 1.5× BR's max throughput.
- **52 cheaper at iso-quality?** **Yes** — ~4–6% at every λ (RB's 16/25/41/18 mix averages below
  BR's 98%-14B route despite the 72B share).
- **53 dominate on all non-quality axes at parity?** Under load (λ≥18): yes (latency, throughput,
  cost, robustness). At light load: no — RB trades 0.1–0.4 s latency for cost + headroom.
- **54 corrected q-tput upper envelope**: below ~18.6 req/s BR t0.5 sits nominally on top
  (statistically tied); **above 18.6 req/s RB solely owns the envelope** (0.402–0.404 vs AP 0.3950,
  PT 0.3691). The submitted "upper envelope at every overlapping throughput level" fails nominally.
- **55 weaken/remove**: the +0.006 peak win (remove); everywhere-upper-envelope (→ "envelope beyond
  BR's saturation; statistically tied with it below"); blanket iso-quality latency win (→ "under
  load"); AP separation (→ nominal). **Keep**: iso-quality cost win; throughput/robustness dominance;
  PT separation; frontier extension beyond 18.6 req/s.

## F. NOT evaluator fixes (56–70): ALL **Class 4 — independent v2 evolution, disclose separately**
56 k 50→10 · 57 judge→deepeval · 58 raw→minmax · 59 cap 256→1024 · 60 fused/length heads ·
61 default weights · 62 a30 tier fix · 63 batched-FAISS · 64 APC · 65 baseline shim/config fixes ·
66 four-arm experiments (gates absent from v1 code) · 67 enhanced baselines · 68 second judge ·
69 multi-seed v2 · 70 Gemma. **Explicit confirmation: `0.4187 vs 0.4056` CANNOT be described as
"the submitted result after fixing the evaluator."** The evaluator-only corrected submitted result is
**0.4042 vs 0.4056 (tie)**; 0.4187 additionally requires the serving-level routing-signal change
(judge→deepeval) plus cap/norm/k — new runs, not re-scoring.

## G. Proposed CR wording, clause by clause
1. "corrected a post-hoc quality aggregation bug that treated valid zero-valued scores as missing" —
   **accurate**. 2. "does not affect training, online prediction, routing decisions, or runtime
   measurements" — **accurate** (B4/Part-1). 3. "changes absolute quality values and removes the
   previously reported small peak-quality advantage over BEST-Route" — **accurate**. 4. "statistically
   tied with BEST-Route at peak quality" — **accurate** (CI [−0.0112,+0.0045]). 5. "remains ahead of
   Avengers-Pro and dispatcher-only baselines" — **accurate with qualification**: dispatcher-only =
   CI-separated ✔; Avengers-Pro = nominal only (paired CI includes zero) → suggest "ahead of
   dispatcher-only baselines and nominally ahead of Avengers-Pro". 6. "central systems result — joint
   model-and-replica assignment provides a better serving trade-off **under load** — remains
   unchanged" — **accurate**; the "under load" qualifier is exactly what E51 requires.

## H. Final outputs
**Classification table (condensed):**
| Points | Classification | Evidence | Values unchanged? | Recompute? |
|---|---|---|---|---|
| A1–15 | 1 Unaffected | direct trace fields; agg code path | YES | no |
| B16–18, 22–26 | 1 Unaffected | traces + code | YES | no |
| B19 | 2 (narrow phrasing) | submitted text vs v2 quantification | YES | no |
| B20 | 2 (decomposition recomputable) | breakdown fields | YES | optional |
| B21 | needs experiment if quantified | no direct ablation | — | yes (if claimed) |
| C27–40 | 3 Affected | V1_REAGG_FULL_costs.csv | coords YES / quality NO | DONE |
| D41–48, E49–55 | 3 (recomputed) | paired bootstraps + iso tables above | — | DONE |
| F56–70 | 4 Independent evolution | drift audit | n/a | n/a |

**Safe to carry unchanged into CR:** all A-items; overhead sub-linearity; BR/AP serial-saturation
results and mechanism; pipeline-mode isolation; bs=1 collapse (as a v1-implementation property);
budget-exhaustion table + filter-dominates finding; centralized/fixed-deployment scoping; iso-quality
COST advantage; throughput ceiling 28.85 vs 18.6; all routing mixes.

**Must be recomputed or rewritten:** every quality mean/peak/CI/coverage (done — corrected values
available); peak-quality claim → tie; upper-envelope → beyond-saturation phrasing; iso-quality latency
→ under-load phrasing; AP comparison → nominal; radar quality axis; operating-point highlights.

**v2 changes to disclose separately from the evaluator fix:** items 56–70 (list above); in particular
the CI-separated +0.013 quality result belongs to the signal-aligned system, not to the evaluator fix.

**Final verdict on the statement:** **TRUE with the above scope.** A CR that retains the submitted
k=50 runtime and only fixes the post-hoc evaluator is the same reviewed system: every runtime
measurement and architectural finding carries unchanged (Classes 1–2); what changes is the quality
interpretation — absolutes drop uniformly, the peak-quality edge becomes a statistical tie, the
envelope claim narrows to beyond-BR's-saturation, iso-quality latency needs the under-load qualifier,
and the AP margin becomes nominal. No training, routing, or measurement artifact differs from what
reviewers evaluated.
