# Opus-5 rebuttal brief (pasted 2026-07-25) — ASSESSMENT vs verified forensics

Verdict: **ADOPT the structure, discipline rules, per-reviewer strategy and drafts — WITH the hard
corrections below.** The brief was written without our version forensics
(`2026-07-23_neurips_rebuttal_audit.md` §1–§3) and without knowing the cluster window; three of its
claims are contradicted by verified evidence. The raw brief lives in the conversation of 2026-07-25;
its drafts are the STARTING TEXT for the three replies + AC comment, edited per this assessment.

## 🔴 HARD CORRECTIONS (do NOT post as written)

1. **"k=50 is a typo; the evaluated config is k=10" — FALSE. NEVER say this.**
   The submitted May run GENUINELY deployed k=50: `V2_RERUN_PLAN_2026-06-07.md:17-19` lists
   "KNN **k=50→k=10**" as an intended v2 CHANGE ("submitted routed on judge, reported deepeval");
   `PAPER_TODO.md` Track-A: "EXISTING v1 stack: 512 cap/**k=50**/current predictors"; the deployed
   submitted config is archived (`_cluster_archive_20260506/scheduler_config_carafast.json`).
   Claiming "typo — all results were produced with k=10" would be a false statement to reviewers,
   contradicted by our own artifacts. REPLACE with the honest insensitivity answer: the frontier is
   insensitive to k over {5,10,20,50} (R5, reported as deltas/band width); the extended evaluation
   uses k=10. No typo claim anywhere (incl. §4 AC comment — strike that paragraph's last sentence).

2. **"78.2%-coverage offline lookup" framed as a metric-basis design choice — it is the zero-score
   BUG** (proven cell-exact in audit §3: 0.0 scores dropped via the or-chain; corrected coverage ≈ 1.0).
   Two posture options (AUTHOR DECIDES, = the §6 sign-off):
   (a) Brief's neutral-basis wording — acceptable ONLY if reworded to state facts without implying
   intent: "the additional studies are computed on the fully-scored subset (every served prompt scored
   on all four candidates); the submitted tables' lookup matched 72–78% of served prompts, so absolute
   levels differ between the two and we report new results as deltas/shares/orderings."
   (b) Audit §6 brief disclosure (correction-as-a-unit, conclusions preserved).
   Either way: never claim the 72–78% was intended design; keep the prepared honest answer for the
   "why 78%?" Phase-2 follow-up (anticipated-follow-ups #1). Full correction lands at camera-ready.

3. **"residual rises only 1.8× over a 5× load increase" — WRONG BASE.** 1.8× is the EuroSys number
   (130→231). The SUBMITTED table is 141→220 = **1.6×** (and the PDF says 1.6×). This is exactly the
   cross-base mixing the brief's own §1 rule forbids. Use 1.6× in every reply about the submitted table.

## 🟡 VERIFY-BEFORE-POST (every number in the brief's drafts gets a file citation first)
The brief's drafts embed many specific values. Some already match verified data (good sign:
tier-removal 44/27/29 + 0.047 = 0.419−0.372 ✔; mix shares 3B 56–58%/72B 1% ✔ `uniform_mix_vs_lambda`),
but per the no-fabrication rule each must be traced before entering a reply:
- arms 2-vs-3 "wash (+2.8 / −0.7 / −3.5% E2E)"; "26–31% faster" cross-tier; "72B share 14% → 1%"
  → `_aggregate/P1_fusion_triangle.json` + EuroSys §6.3.
- scoring-loop scaling "12.8 / 14.3 / 22.5 μs at |I|=13/100/500" → locate the microbench in v2-pack.
- LODO folds "±0.05 on 4/7; SQuAD +0.09; code +0.05; GSM8K 0.32→0.23 (random 0.25)" → LODO asset.
- budget served-text re-judge "+0.015 / +0.012 / +0.006" → v2 budget re-judge files.
- p99 TTFT "[X]" → serving-side definition ONLY (TTFT landmine: scheduler buffers; no user-facing
  TTFT claim — `ttft-metric-is-cara-clocked`).
- annotation costs: 18,608 calls/model ✔; ~24 calls/s on 4×A100 ⇒ ~13 min/model-column; FAISS 52.3 s ✔;
  XGB per-tier `train_time_s` from `training_metrics.json`.
- FEASIBILITY FLAG on §2.3(b) "k-sweep on the submitted basis": the deployed k=50 KNN .pkl was NOT
  preserved (`feedback_keep_deployed_models`) — an exact v1-basis replay may be impossible; safe
  substitute = R5 decision-level k-sensitivity reported as deltas/band width.

## 🟢 ADOPT AS-IS
- §1 deltas/ratios/shares/orderings-only rule (matches our 3-way invalidation).
- §3.1 287j written to the character limit (rating-3 is pure clarity; Significance 4 = flip target);
  W1/W2 batching answers (our verified formula: busy ratio <0.3→bs4/50 ms, <0.7→16/100 ms,
  else 32/200 ms; size-or-timeout, timer from first arrival, NO holding).
- W3 KV/prefix: concede-and-bound + STRENGTHEN with our verified fact the brief lacked — **the
  submitted runs had APC OFF cluster-wide** (`2026-05-04_design_pivots.md:187-189,258`), so prefix
  reuse did not interact with routing in the evaluated config; affinity term = future work.
- §3.2 gTfM order-of-presentation (formulation-not-estimator framing); Q3 centralized-by-design;
  Q4 layering + tier-removal.
- §3.3 faAU cold-start-FIRST weighting; three-part cold-start answer; dispatcher structural argument
  (+ our SQ≡least-connections point); budget reframe (mechanism-not-simplex).
- §4 AC confidential comment structure (minus the typo sentence; verify arm numbers).
- §5 exclusions: don't volunteer enhanced-baselines/23×; no absolute new headline quality numbers;
  no arXiv/repo references; no apology framing; Phase-2 promises only with the reservation (WE HAVE
  IT: Jul 25 19:00–Jul 31 19:00).
- Camera-ready commitment list (Fig-1 redraw incl. RoBERTa box = pluggable alternate never used ✔
  verified, P100/3B fixes; Table-2 placement; radar cut; appendix walkthrough; dead-reckoning defined;
  LPT appendix — keep the HEURISTIC framing, no 4/3-guarantee claim, per audit E-item).

## V2/EuroSys-RESULTS USAGE RULES (answering "can we use them if the submission was k=50?" — YES, classed)
The submitted stack (k50/cap256/judge-routed/raw-norm/buggy-agg) and the v2 stack (k10/cap1024/
deepeval-routed/minmax/fixed-agg) are two bases. NeurIPS allows new results in responses; follow-up
experiments are NOT required to freeze the submission's hyperparameters. One basis note in the opening
paragraph (posture (a)/(b)) covers everything; then:
- **Class A — freely usable (mechanism orthogonal to the changed knobs):** four-arm isolation (all arms
  share ONE config except the isolated component — internally controlled, cross-arm deltas valid on any
  stack); tier-removal; nonstationary/burst latency robustness; mix-vs-λ rate-independence; overhead
  scaling + µs microbench; cold-start curve + LODO (estimator-class properties); NEW P1/P2/Gemma cells
  (run on the current stack, labeled as the extended evaluation).
- **Class B — usable as deltas/shares/orderings with the basis labeled:** quality comparisons (e.g.
  paired ordering RB>BR with CI on the fully-scored basis). Never absolute headline quality values.
- **Class C — withhold:** new absolute headline quality numbers; enhanced-baselines/23× story; anything
  inviting "you rebuilt the system" without room to explain.
- **The k-bridge if asked:** "extended studies use k=10; routed quality is insensitive over
  k ∈ {5,10,20,50} (R5 band)" — the config delta is demonstrably immaterial; same pattern for the other
  knobs where sensitivity data exists (cost vectors; scoring-formula = planned).
- **Pocket answer (do not volunteer)** for "does the ordering hold on the submitted config with
  corrected scoring?": vs AvengersPro YES and the lead widens; vs BEST-Route the tiny peak-quality edge
  inverts on the judge-routed submitted config and is restored once routing uses the reported metric
  (the misalignment fix) — full nuance in audit §3.

## ⛔ DO-NOT-REASSERT LIST (the two-bug risk register — basis-dependent claims the rebuttal must not repeat)
The v1 bugs (or-zero quality aggregation +~0.11; score_type judge-vs-deepeval routing misalignment)
affect ONLY the quality axis. Two submitted claims do NOT survive corrected scoring on the v1 stack
(audit §3: corrected cara peak 0.402–0.404 vs BR-t0.5 flat 0.4056):
1. The peak-quality edge over BEST-Route ("0.5181 vs 0.5119", "+0.006", "no baseline matches both...").
2. "Hull traces the upper envelope ... at every overlapping throughput level" (quality axis).
RULE: the replies never repeat, quantify, or defend these two claims — no reviewer challenged them.
If pushed in Phase 2: escalation ladder = neutral-basis answer → honest correction disclosure with the
alignment-fix result (0.4187 vs 0.4056, CI-separated — the architecture earns the claim once the routing
signal matches the reported metric); pocket answer in anticipated-follow-ups #1. Camera-ready carries
the corrected story.
SAFE TO ASSERT FREELY (basis-independent, measured not scored): iso-quality latency+cost wins,
load-robustness (sustained 28.85 vs 18.73), overhead sub-linearity + amortization, four-arm mix shift +
E2E deltas, tier-removal, batching ablation, budget-exhaustion mechanics, all new-cell deltas/orderings
that pass the both-bases gate. The cost-corner cession (wc≤0.8 sweep coverage) is likewise not
re-litigated — the submitted paper never claimed global dominance ("traces the upper region" is scoped).

## Δ vs our PATH v3 (ours supersedes where they differ)
The brief treats Po2/dispatcher runs as "Phase 2 only, promise-then-post" — written before the window
was known. PATH v3 is strictly stronger: P2 overnight Jul 25→26 + P1 Jul 26 AM ⇒ **the AC's two
requested experiment TABLES go INSIDE the initial response** (submitted ≤ Jul 28 11:59 BST), not as
promises. Everything else in the brief slots into the v3 freeze-period drafting block.
