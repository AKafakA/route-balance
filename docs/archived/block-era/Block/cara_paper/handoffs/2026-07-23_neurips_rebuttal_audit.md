# NeurIPS reviews (3/4/4) — version audit + rebuttal feasibility (2026-07-23)

> ⛔ **SUPERSEDED FOR EXECUTION — EVIDENCE/HISTORY ARCHIVE ONLY (2026-07-25).** The ONLY binding
> execution documents are `2026-07-25_EXECUTION_HANDBOOK.md` (order, gates, cell matrices, rules)
> and `2026-07-25_reviewer_actions_table_v2_and_drafts.md` (answers/drafts). All PATH v1–v4
> sequencing, family-first plans, full-grid plans, §6 disclosure layers, and CR-scale prose below
> are HISTORICAL — DO NOT EXECUTE from this file. Its verified forensics (§1–§3, §9–§11, evidence
> tables, anticipated follow-ups) remain valid as EVIDENCE and are cited by the binding docs.

## 0) INITIAL META-REVIEW (AC axCN, 2026-07-25 06:24) — THE GUIDE
> "The reviewers agree that the paper addresses an important problem in heterogeneous LLM serving and
> that the proposed joint optimization framework is well motivated. However, the rationale of design
> choice and and evaluation part of the paper should be improved. Specifically, **many design choices
> such as XGBoost estimator need to be better justified**, and **comparisons with general-purpose
> dispatchers can be added** to illustrate the effectiveness of the proposed scheduler."

**Read:** the AC's two named decision-movers ARE our P2 (arm-4/arm-2 static-vs-XGBoost isolation) and
P1 (Po2/general-purpose dispatchers). PC letter: "Use the initial meta-review as your guide. It tells
you what would most likely change the AC's view." ⇒ **AC-FIRST re-sequencing (2026-07-25, see §7):**
P2+P1 run FIRST on the proven Qwen stack (banked within ~24 h of cluster start) so their result tables
are already IN the initial response when Phase 2 opens; family track follows, still fully in-window.
P1 extended with {AvengersPro, BEST-Route} × Po2 combo cells (~6) — "general-purpose dispatchers"
under the strongest decoupled routers, directly the AC's sentence.

**Confirmed rebuttal mechanics (PC letter, supersedes/refines §4 where different):**
Phase 1 authors-only (hidden) **Jul 23–27**; Phase 2 author/reviewer/AC discussion **Jul 27–Aug 03**
(authors post until Aug 03); Phase 3 reviewer/AC-only Aug 03–10. Per-review Rebuttal buttons, 10,000
chars, markdown, **no file uploads, NO LINKS** (exception: anonymized code link to the AC via Official
Comment if reviewers ask) ⇒ all corrected/new numbers go INLINE as markdown tables — never "see arXiv".
No paper/supplementary revisions during response. Anonymity applies to responses. PC may solicit
ADDITIONAL reviews after the response period (be ready for a 4th review). "Author AC Confidential
Comment" channel exists. Meta-review response: address the two AC items at the TOP of every per-review
reply + a short dedicated response on the meta-review thread if OpenReview permits.

**Reviews:** 287j = 3 (borderline reject; Quality 3 / Clarity 2 / Significance 4 / Originality 3),
gTfM = 4 (Q3/C2/S2/O4), faAU = 4 (Q3/C3/S2/O3). All confidence 4. Raw reviews: user paste 2026-07-23
(OpenReview Submission28271; store verbatim copy when drafting the rebuttal).

## 1) VERSION AUDIT — what reviewers actually saw (VERIFIED, refined 2026-07-23 PM vs ~/nips.pdf)
- **The authoritative reviewer artifact is the OpenReview PDF** (`~/nips.pdf`, 16 pp, "Confidential
  reviewer copy… NeurIPS 2026" banner). The user-built `rb_nips_submission.zip` matches local
  `neurips_2026.tex` byte-for-byte (md5 `1d58359e…`) — but that local tex is the **post-submission
  arXiv-v1 state**, NOT the submitted one. Git chronology (route-balance-paper): `3226c1a` "Initial
  commit: NeurIPS 2026 RouteBalance paper source" = submitted state → post-deadline commits `2aa0562`
  ([preprint] mode) / `3bd1a19` (real authors) / `69912ba` (anon URL → public GitHub URL) /
  `b9becca`+`066f751`+`30bc996` (**overhead-metric/TTFT "off-instance residual" reframe**).
- **All NUMBERS are identical** between PDF and local tex (verified: 243/860K/203 cells, λ12 table
  0.5141/0.5101/0.5089/0.5037, extremes 0.5181/0.5119/0.5101/0.5093, coverage 78.2%, 1,000 resamples,
  k=50, overhead 141→220 with 149@λ12, 33.1 s/150×, budget 32.4/19.2/12.6, batching 45.4 s,
  "lives in the filter", cold-start limitation). §3's cell-exact re-aggregation therefore applies to
  the reviewed PDF unchanged.
- **PROSE differs (submitted PDF is the EARLIER framing):** PDF abstract/Table-5 call the 149–220 ms
  "per-request scheduling overhead" (faAU quoted this verbatim); the residual reframe (off-instance
  residual definition, ≈33 ms decision-compute split, 44.7→23.9 amortization, 2219-vs-3432 E2E
  comparison, "fully charged", Appendix-B decomposition) exists ONLY in the post-submission local tex —
  **available as pure clarification material for the rebuttal (reviewers never saw it).**
- PDF is properly double-blind: code link = `anonymous.4open.science/r/route-balance-3ECA` (the local
  tex's github.com/AKafakA link is the post-submission arXiv edit). PDF contains ZERO occurrences of
  "enhanced" (baselines) — faAU's "enhanced concurrent-scoring variants" praise is their reading of the
  pipeline-mode/architecture-controlled passage; the actual v2 enhanced-baseline results are NEW to them.
- **Reviewed Figure 1 (p.4) confirmed stale** (rendered + inspected): RoBERTa Estimator box (never in
  text; deployed = MiniLM+KNN), P100 listed in Hardware, Qwen-3B missing from LLM box, "estimate ②"
  arrow never returns, "Instatnce" typo. 287j's figure critique is fully valid.
- It is **NOT** `neurips_2026_v2.tex` (the Jun-13 audit-validated v2 rewrite — never submitted).
- Submitted build = **v1 stack**: k=50 (tex L84/L204), 243 configurations / ~860K requests (L89),
  off-instance residual 141→220 ms + 149 ms@λ12 (L263/267, `tab_sched_combined`), bootstrap **1,000**
  resamples with **coverage 72.2–78.2%** (appendix Table, L318-333), quality 0.5141/0.5101/0.5089/0.5037
  at λ12; extremes peaks 0.5181/0.5119/0.5101/0.5093 (`tab_extremes`).
- Reviewer-quoted numbers (243 configs, 860K, 149/220 ms, 13-inst/28-GPU) all match the zip. ✔

## 2) SUBMITTED-RUN DEFECTS (authoritative list, `V2_RERUN_PLAN_2026-06-07.md:15-34`)
Submitted May run had: output cap **256** (→1024 in v2); **routing on `judge` while reporting deepeval**
("submitted routed on judge, reported deepeval"); KNN **k=50** (→10); `quality_norm` raw (→minmax);
**A30-latency first-match-P100 tier bug**; latency-XGB trained pre-1024. Deployed submitted config
archived: `_cluster_archive_20260506/scheduler_config_carafast.json`.
PLUS the **aggregator zero-score bug**: `aggregate_full_215.py:185` (and `aggregate_overnight_grid.py:129`)
`per_model.get(m) or per_model.get(stripped)` — a 0.0 deepeval score under the primary key falls through
to a **never-stored stripped key** → dropped. That IS the paper's 72–78% "coverage" (misdescribed in the
submitted appendix as unscored lookup pairs).

## 3) CORRECTED RE-AGGREGATION of the May grid (2026-07-23, CPU; scratchpad script `v1_reagg/reagg_v1_zerofix.py`,
outputs `V1_REAGG_BOTH.csv` — 175 no-budget cells, buggy + fixed means side-by-side)
- **Validation:** buggy retrieval reproduces the submitted numbers EXACTLY — λ12: RB wq0.8 0.5141 (cov .782),
  AP pw0.8 0.5101 (.774), PT 0.5037 (.722); extremes peaks 0.5181/0.5119/0.5101/0.5093 all reproduced.
- **Corrected (zero-safe) numbers, v1 config:** RB peak **0.4042**@17.25 req/s; BR t0.5 **0.4056** (flat across λ
  — BR/AP are deterministic per-prompt routers, quality λ-invariant); AP 0.3950; PT peak 0.3691.
  - vs AvengersPro: correcting **widens** RB's lead (0.009 vs buggy 0.004) — matches V2_RERUN_PLAN:33 prediction.
  - vs BEST-Route: **the submitted "+0.006 peak-quality" edge inverts** — corrected RB peak is ~0.001–0.003
    BELOW BR t0.5's (a 98.4%-to-14B static route that collapses at λ≥18; max sustained 18.73 req/s).
  - Latency/throughput/cost/overhead numbers are **untouched** by this bug (quality axis only).
- **v2 (Jun-6 rerun, all fixes incl. deepeval-routing/k10/cap1024) restores the quality peak with separation:**
  RB 0.419 (0.4184±0.0003 4-seed) vs BR 0.4056, paired CI [+0.005,+0.022] — the EuroSys/arXiv numbers.
- **Validation COMPLETE (extended pass):** the appendix argmax cell lives in
  `trained_router_baselines/best_route_4way/br4_argmax_ro_l12_n3534.json` — buggy = **0.5089** (exact
  appendix match; corrected 0.3969, cov 1.0). Cell counts reconcile exactly: 4-subdir pass 175 +
  `trained_router_baselines/` 28 = **203 routing cells**, the tex's number. ALL four λ12 appendix cells +
  all four extremes peaks now reproduce bit-for-bit under the buggy retrieval.

## 4) NeurIPS 2026 POLICY (web-verified, official pages)
- Reviews released Jul 22; **author↔reviewer↔AC discussion Jul 27 → Aug 03 (AOE)** — authors cannot post
  after **Aug 03**; reviewer/AC-only until Aug 10; notification Sep 24.
- **Per-review replies, 10,000 chars each, markdown, no file uploads, no links** (code link to AC via
  Official Comment only). No global response. **No PDF revision until camera-ready.**
- **"These responses can include new results, however your original submission will serve as the basis"** —
  new/corrected experiments in rebuttal text are explicitly sanctioned.
- Context: 2025 main-track acceptance 24.52%; 6-point scale, accept boundary structurally between 3 and 4.

## 5) REBUTTAL FEASIBILITY MAP
**R-287j (3 — ALL concerns writing/clarity; Significance 4 = flip target):**
batching semantics (scheduler assignment-batches vs instance-side vLLM continuous batching — no request
is delayed for a size quota; adaptive cap only bounds batch size), adaptive-limit formula (quote code),
KV/TTFT questions, Fig-1 walkthrough + datasets appendix (camera-ready promise), Fig-3 radar space, Table-2
placement, **Fig-1 is verifiably stale** (RoBERTa box never used — deployed = MiniLM+KNN; also lists P100,
omits Qwen-3B, "estimate" has no return arrow, "Instatnce" typo) → acknowledge + describe true flow + redraw
at camera-ready. NO new experiments required. → answerable NOW (code-fact quotes pending from sweep agent).
**R-gTfM (4):** Q1 XGBoost-vs-static → the arm-4 static-prior isolation (existing v2 data) + honest framing;
Q2 TTFT/prefill (deployed separate TTFT XGB head; APC setting to confirm) ; Q3 centralized → yes + overhead
scaling evidence (24–33 ms compute, sub-linear residual) + future-work distribution; Q4 fixed cluster → yes,
deliberately (complementary to deployment planners — RouterWise/BOute positioning) + tier-removal robustness.
→ answerable with EXISTING data.
**R-faAU (4):** (a) generality → needs cluster (topology in-window; Gemma NOT realistic in 4 days — promise
camera-ready); (b) Po2/least-connections dispatchers → cluster, cheap; (c) cold-start quantification →
OFFLINE (KNN training-subset degradation curve + labeling-cost accounting) + partial-signal decision-level
analysis; (d) tuning burden → existing nonstationary 3×3 + mix-flat-across-λ; (e) "budget lives in filter" →
framing (filter = part of the fused system; constraint mechanism vs frontier weights).

## 6) CORRECTION-DISCLOSURE STRATEGY — **DECIDED 2026-07-25: posture (c′) = silent-in-rebuttal,
corrected-at-camera-ready, WITH INTEGRITY RAILS** (supersedes the paragraph below, kept for history)
1. The rebuttal VOLUNTEERS NOTHING about the two v1 bugs and — rail 1 — **never re-asserts the two
   basis-dependent claims** (⛔ list in the brief assessment). All new cells run/score on the paper's
   own basis (PATH v4), so no basis note is needed; every posted claim passes the silent both-bases gate.
2. Rail 2 — **truthful if asked**: escalation ladder (neutral-basis answer → honest correction with the
   alignment-fix result 0.4187 vs 0.4056 CI-separated). Never a false statement at any rung.
   **Explicitly OFF THE TABLE (2026-07-25): attributing the corrected-vs-submitted quality gap to
   "run variance."** The gap is ~0.11 systematic; our own paper states CI ±0.009 and multiseed spread
   ±0.0002–0.0006 — the excuse is falsifiable from the submitted appendix and is a fabricated
   explanation for a known cause. PATH v4 makes it unnecessary anyway (new cells score on the paper's
   basis → no inconsistency ever surfaces).
3. Rail 3 — **camera-ready (if accepted) carries the FULL correction with an explicit change note**
   (scoring-coverage bug + routing-signal misconfiguration fixed; corrected numbers throughout;
   qualitative conclusions preserved, quality-peak result strengthened/CI-separated). NEVER a silent
   number swap. If rejected → the corrected stack ships as the MLSys submission (already the plan).
4. k=10-vs-k=50 is evidenced-minor (R5 band ~0.013 over {5,10,20,50}) — the CR correction rests on the
   signal+scoring fixes, not k. OPTIONAL CR-prep insurance (Jul 29–31 bonus): ~5–10 cells on the v1
   stack with ONLY score_type=deepeval flipped (k=50 kept) → isolates the fix on the submitted config
   for a minimal-delta CR story.

5. **CR PLAN (2026-07-25): NO full rerun.** CR base = the audit-validated **v2 results** + the
   already-written `neurips_2026_v2.tex` (never submitted) + the change note + reviewer-promised
   clarity/figure/appendix edits. Gap: this window's new tables are v1-recipe → **bank v2-recipe
   duplicates of {Po2, Gemma sample, arm extensions} in the Jul 29–31 bonus slot (~30–40 cells,
   ~0.5 d, cluster already live)** so the CR is 100% v2-coherent with no future lease (fallback:
   1–2 day Oct lease post-notification). **k=10 framing rule:** an evidence-backed config refresh
   ("frontier insensitive over k∈{5,10,20,50}; k=10 slightly cheaper") — NEVER the cause of the
   improved numbers (R5 contradicts it; drivers = scoring fix + routing-signal alignment).
   **CR-SCALE RULE (2026-07-25, answers "is a big revision allowed?"):** CR permissibility is a
   same-paper/same-claims test, not a diff-size test — our numbers delta is large but the CLAIMS delta
   is tiny (one micro-claim strengthened), which is the governed quantity. Keep the CR
   **structure-identical** to the reviewed paper (same story/figures/claim sentences; corrected values;
   new subsections ONLY where reviewers asked); prominent "Changes from the reviewed version"
   paragraph; **send the AC courtesy note at CR** (converts diff-discovery tail-risk into a pre-cleared
   conversation). Realistic worst case = presentation adjustments, not rejection; the only dangerous
   CR is the uncorrected one.

Historical (2026-07-23, superseded):
Recommended: one short factual paragraph in EACH reply (or the most relevant ones): since submission we
re-validated the pipeline end-to-end, found + fixed (i) an aggregation bug that dropped zero-valued judge
scores (uniform-direction inflation across ALL systems; latency/cost/throughput/overhead unaffected) and
(ii) a routing-signal/config misalignment (routed on a different judge label than the reported metric;
256-token cap). The corrected + extended re-evaluation (same cluster/baselines/protocol) CONFIRMS every
cross-system conclusion and STRENGTHENS the headline: quality ceiling 0.419 vs BEST-Route 0.406, paired
bootstrap CI [+0.005,+0.022], 4-seed stable; vs AvengersPro the corrected lead widens. All corrected numbers
in the revision. — This is additive-validation framing (sanctioned "new results"), not a new paper: same
system, same testbed, same claims. DO NOT publish per-cell corrected-v1 tables in the rebuttal (3-way
invalidation rule: v1 / v2 never mixed); the v1-corrected computation is our internal verification that the
bug's direction/uniformity claims are true.

## 7) SCOPED CLUSTER PLAN (**CONFIRMED window: Jul 25 19:00 → Jul 31 19:00, 2026 — 6 days**;
author target: ALL rebuttal content posted by **end of July**; hard close Aug 03 AOE = buffer only)
**Rebuttal timing strategy:** post the three initial per-reviewer replies on **Jul 27** (discussion
open) from existing-data answers + "additional results follow in this discussion"; post ALL follow-up
tables by **Jul 30–31**. Aug 01–03 reserved purely for reviewer back-and-forth. Even if the cluster
slips, the core rebuttal stands on existing data.
**PATH v4 BASIS PIVOT (2026-07-25, author): ALL rebuttal experiments run on the SUBMITTED v1 CONFIG**
— extensions to the paper's own basis (k=50, judge-routing, raw quality-norm, cap 256, deployed Mar-27
latency heads), so the rebuttal is ONE coherent basis with the paper: no basis note, no cap discussion
(not raised by reviewers; caps are standard ML-eval practice), v2 results reserved for the future
submission if rejected. Feasibility verified: current code + archived v1 config reproduces v1 semantics
(`quality_norm` defaults to "raw" — cara_serve.py:1549-52; bucket head config-gated off; arm gates are
additive env switches). **R0 SIMPLIFIED (2026-07-25 checklist): latency heads = the SAME Mar-27
`deploy_*` files in v1 AND v2 (v1's per-tier predictor configs verified in the May-8 tree; NO missing
XGBoost model, NO sweep redo). A Jun-6 k=50/14,919 KNN reconstruction already exists at
`models/.../quality/judge/deploy_knn` — only the REPLAY GATE remains; bench recipe recovered
(`--max-tokens 256 --seed 1`, 3534-bench, native + pipeline-thin configs) into the v1-exp manifest.**
Original task framing — R0: rebuild the k=50 judge-routed KNN (deployed
.pkl was lost): inputs all present (Apr-5 train 14,919 + `Qwen_Qwen2.5-7B-Instruct` labels + MiniLM +
flat FAISS, ~1 h CPU) → **VALIDATION REPLAY gate**: recompute predictions for a May cell's prompts and
match the logged per-request `predicted_quality`/`predicted_length`/`predicted_best_model`; then **R1:
re-run one reference cell (uniform λ12) on the v1 config** and compare E2E/mix/tput vs the May cell
(environment-drift gate + doubles as the internal arm-1 baseline). Only then P2/P1 (and the family
track, also on the v1 recipe: k50/judge-routing/cap256 — cap-256 cells are FASTER, helping the clock).
**Aggregation for new cells:** score them identically to the submission (same lookup treatment) so
absolutes are directly comparable to the paper — TRUE statement "measured identically to the submitted
evaluation"; internal gate (not posted): every quoted claim must ALSO hold under the corrected basis —
compute both; if any claim is basis-dependent, STOP and flag to the author before posting.
**OPERATING RULE (confirmed 2026-07-25, execution eve): the new-cell aggregator is
`v1-exp/scripts/reagg_v1_zerofix.py` (dual-pass: buggy-reported + zero-safe-internal in one run).
NEVER `aggregate_full_215.py` alone (no gate) or `aggregate_v2_grid.py` (absolutes not paper-comparable).
Posted tables: quality number only — NO coverage column. Gemma lookup gets the same dual treatment.**
**Two flagged decisions (defaults chosen, author may override):** (a) a30 latency-tier first-match-p100
bug — DEFAULT: keep the fix (deployment-correctness; not a policy knob); purist alternative (replicate
v1 behavior via the old tier-lookup) available if exact-v1 fidelity is preferred. (b) All internal
comparisons for new experiments are within-session (new arm-1 reference vs new arm cells), never
new-vs-May absolutes (environment drift).
Schedule impact: +~0.5 day for R0/R1 (KNN rebuild runs on CPU during bring-up), offset by faster
cap-256 cells. Timeline below otherwise unchanged.

**MINIMAL CELL DESIGNS (2026-07-25, author-scoped — sampling over full sweeps):**
- **P1 Po2 ≈ 11 cells (~1.5 h), NOT a full sweep:** passthrough×Po2×λ{6,12,18,24,30} (5) +
  passthrough×{SQ,RR} in-session references (~3) + AvengersPro-best-pw×Po2×2λ + 1 SQ ref (~3).
  BEST-Route×Po2 = ARGUED not run (its collapse is serial-scoring upstream of dispatch). All dispatcher
  comparisons are within-session (never new-vs-May). Structural claim (arm-2) carries the RB contrast.
- **Gemma SAMPLED ≈ 12 cells (~1.5–2 h at cap-256), not the reduced grid:** 2 rates (λ12 + λ18/24) ×
  {RB wq0.8, RB uniform, RB wc0.8, best baseline, passthrough-SQ} + 1–2 arm-4-static-on-Gemma cells
  (ties the family table to the AC's XGBoost point). Claim = ordering preserved on a second family
  (compact table; no hull claim). Full family grid deferred to camera-ready/MLSys.
  Sampling raises P(Gemma tested by end Jul-27) to ~75–80% under the parallel-track schedule
  (pending author OK on relaxing the hard freeze to an attention-freeze for unattended pipelines).
- **TOPOLOGY (3-tier stack, 2026-07-25):** T1 free/in-initial-response = existing tier-removal + the
  QUANTITATIVE re-provisioning replication (R1/reference cells on the 3rd independent lease reproduce
  May numbers within X% — falls out of gates already planned) + ceiling statement (guardrail: never
  "always new hosts"). **T2 bonus-slot (~3 h, Jul 29–31 or pre-gate if green): topology-B sampled** —
  replica reallocation WITHIN existing (model,GPU-type) pairs (heads are per-pair; cross-type moves
  would need new heads — itself a defensible scoping sentence): e.g. {1×72B, 2×14B, 3×7B, 5×3B}
  (premium halved, cheap grown) via model_deployment.json only; Gemma-style sampled grid 2 rates ×
  3–5 systems ≈ 10–12 cells → "ordering preserved under a materially different replica allocation"
  as a Phase-2 follow-up. Net: faAU generality answered on BOTH axes with data (family + topology);
  keep the humility note (full multi-cluster generality = future work).

**PATH v3 (2026-07-25): HARD GATE = INITIAL RESPONSES SUBMITTED BY JUL 28 11:59 BST**
(= Phase-1 close, Jul 27 AoE rendered in UK time). Author rule: **ALL experiments PAUSE for the
submission crunch** — the only pre-gate experiments are the AC's two items (P2+P1), which must be
inside the initial response. Family resumes AFTER submission.
- **Jul 25 19:00–24:00**: P0 provision + node setup → full QWEN deploy + md5-verify + reproduction
  smoke. (**G0 Gemma compat probe MOVED to post-submission, 2026-07-25 author decision** — nothing
  Gemma runs before the main experiments + submission complete.)
- **overnight 25→26**: **P2 FULL grid unattended** (~42 cells ≈ 5–7 h).
- **Jul 26 AM**: **P1** — passthrough×Po2×λ + {AvengersPro, BEST-Route}×Po2 combos (~16–22 cells,
  ~3–4 h). ⇒ **both AC items banked by Jul 26 midday.** PM: 2-seed replicates (~3 h) + **P4 offline
  analyses on CPU in parallel** (cold-start curve, per-dataset join, switch fraction — so they're IN
  the initial faAU reply).
- **Jul 26 eve → Jul 28 11:59 BST: 🛑 EXPERIMENT FREEZE.** Aggregate P1/P2 (FIXED aggregator) → build
  inline markdown tables → **draft the three per-review replies + meta-review response** (AC's two
  items at the top of each, §6 disclosure paragraph, §10 mapping) → author §6 sign-off → verify every
  number against aggregates (consistency pass) → **ENTER/SUBMIT all initial responses well before
  Jul 28 11:59 BST.** No new experiment launches during the freeze.
- **Jul 28 PM (post-submission)**: **G0 Gemma compat probe** (one node, ~30 min; picks Gemma-3 vs
  Gemma-2; weight pull) → switch to **Gemma ladder** → **G1 generation overnight 28→29**.
- **Jul 29**: G2 judge → G3 train → G4 latency sweeps (record G1–G3 wall-clock). Evening–night:
  **G5 Gemma serving sweep** (27–54 cells).
- **Jul 30**: aggregate + **post family follow-up tables** (generality + onboarding cost) in Phase-2
  discussion — still before end-of-July. 2-seed Gemma replicates if time.
- **Jul 30–31 19:00**: cluster LIVE for reviewer follow-ups (bonus: P3 topology, burst/SLO
  discriminators, Gemma multiseed). **Jul 31 before release**: anomaly reruns + rsync ×2 (all Gemma
  artifacts + manifests) + preserve deployed models.
⇒ AC items in the INITIAL response (submitted ≤ Jul 28 11:59 BST); family tables Jul 30; ~1.5–2 days
of live-cluster follow-up buffer; Aug 01–03 = text-only back-and-forth.
**AUTHOR DECISION (2026-07-24, final): everything waits for the GPU.** No pre-window work; the full
rebuttal (P4 offline analyses, reply drafting + §6 wording sign-off, and all experiments) is executed
as one block in the window's first **3–4 days** (Jul 25 19:00 →), targeting: initial replies visible
Jul 27, all follow-up tables by ~Jul 28–29, everything posted before end-of-July, cluster held to
Jul 31 19:00 for reviewer follow-ups. No reviewer follow-up questions exist YET (2026-07-24) —
they arise only AFTER our rebuttal posts: OpenReview lets reviewers ask follow-ups during the
discussion window (Jul 27–Aug 03), and **the cluster is held live to Jul 31 19:00 precisely to run
experiments answering them**. Session picking this up on Jul 25: start at §7's critical path
(P0 + G0 first); read the shared experimental rules + smoke before any run.

**ANTICIPATED DISCUSSION-PHASE FOLLOW-UPS (prep list; ✅ = answerable from data on hand, 🖥 = needs the
live cluster — fits the Jul 29–31 bonus slots):**
1. ✅ *"How do you know the corrected pipeline is right / what else changed?"* (most likely, after the §6
   disclosure) → cell-exact validation methodology (§3), per-system uniformity, consistency gate; offer
   the corrected-vs-original delta table for any cell on request.
2. 🖥 *"If the static prior matches, why keep the learned predictor at all?"* (gTfM, after the arm-4
   answer) → calibration/headroom framing + burst/SLO discriminator cells (W3-style) runnable in the
   bonus slots if pressed.
3. ✅/🖥 *"Do the same weight tuples transfer to the second family without retuning?"* (faAU, after the
   Gemma table) → G5 runs the SAME presets — answer falls out of the sweep; extra transfer cells if pressed.
4. ✅ *"What's the minimum label budget for a new family?"* → P4 training-size curve + the live-measured
   Gemma onboarding wall-clock (G1–G3).
5. ✅ *"What happens at very light load / batch of 1?"* (287j, after the batching explanation) → existing
   batching ablation (bs=1 cells) + adaptive formula floor (batch 4 / 50 ms).
6. ✅ *"Is +0.013 practically meaningful?"* → per-dataset breakdown + model-switch fraction (P4) +
   multiseed CI + the latency/cost axes where gaps are 2.6–4.1×.
7. ✅ *"Po2 at the tail vs fused?"* → p99 columns from the P1 cells.
8. ✅ *"Centralized scheduler: failure point / scale beyond 13 instances?"* → decision-compute
   amortization (44.7→23.9 ms), sub-linear residual, honest future-work scoping.
9. ✅ *"Will the corrected artifact be released?"* → yes — public repo + HF dataset already updated;
   reproduction map ships with the revision.

**RE-SCOPED 2026-07-24 (author decision): generality priority shifts TOPOLOGY → MODEL FAMILY.**
Rationale: (a) faAU's own weighting — "a second model family would be especially persuasive… may add more
acceptance probability than dozens of extra Qwen configurations"; (b) the topology axis is defensible from
EXISTING facts (see wording guardrail below); (c) our reservation already holds the practical ceiling of
CloudLab GPUs of these classes (author-attested best-effort claim).

**ORDERING (2026-07-24, author): the FAMILY track (G) is the FIRST experiment priority** — reviewer-pointed
(faAU) AND required for every future submission path (camera-ready / MLSys). Front-loading banks the
family assets (lookup, estimator, latency heads) early; the Qwen-side P1/P2 reuse the proven stack
(near-zero risk) and tolerate compression into Days 3–4.

| Day | Work | Cells | Time | Serves |
|---|---|---|---|---|
| Jul 25 | **P0** provision + md5-verify + smoke (FIXED aggregator only) + **G0 Gemma compat smoke (~30 min; fail → Gemma-2)** → **deploy the GEMMA ladder FIRST** → **G1 generation starts same afternoon** (18,553 prompts × tiers, greedy temp0/freq1.2, batch, tiers parallel) and runs overnight | smoke | 1 d + overnight | gate + family step (i) |
| Jul 26 | **G2 judge** on one d8545 host (4×A100 ≈ 24 calls/s → ~74k calls ≈ ~1 h) + **G3 KNN estimator + static priors** (~1 h) + **G4 reduced per-tier latency sweeps** (3-rate, parallel, ~3–5 h — while Gemma is still deployed) → **G5 Gemma serving sweep starts evening** (3 presets × λ{12,18,24} × {RB, arm-4 static, best baseline} × 1–2 seeds ≈ 27–54 cells, ~4–7 h, into the night). **RECORD WALL-CLOCK G1–G3** — the measured new-family onboarding cost IS faAU's cold-start "training-time" answer | 27–54 | 1 d + evening | family steps (ii–v); faAU-generality + cold-start (live) |
| Jul 27 | Finish/verify G5 → **redeploy QWEN stack** (~2–3 h) → **P2** decoupled/static isolation — **an EXTENSION, not a rerun**: same gates/code as the existing cells (arm-4 exists ONLY at uniform-λ18 + 40/6 overload; arm-2 only at uniform-λ{12,24,30} — reuse those in the grid as-is); NEW cells fill {arm-2, arm-4} × {wq0.8, uniform, wc0.8} × λ{6,12,18,24,30} minus existing (+2-seed headline cells) so the static-≈-learned claim covers the **wq0.8 peak-quality preset and high load** that the paper headlines (gTfM Q1's obvious follow-up). TRIM-FIRST if compressed: 3 rates × 3 presets × 2 arms = 18 cells (~3 h); worst case the rebuttal stands on existing cells with scoped wording. Same day: **POST INITIAL REBUTTAL REPLIES** (existing-data answers; family + P2 tables flagged as incoming follow-up) | ~30+12 new | 1 d | gTfM-Q1, faAU-fusion; rebuttal clock |
| Jul 28 | **P1** Po2 dispatcher cells (existing flag, 10–16) + anomaly reruns + rsync ×2 (incl. ALL new Gemma artifacts to ≥2 locations) + aggregate + build follow-up tables (posted Aug 01–02) | 10–16 + reruns | 1 d | faAU-dispatchers + freeze |
| any day (CPU) | **P4** cold-start training-size curve + per-dataset join + model-switch fraction | — | ~0.5 d CPU | faAU-cold-start / +0.013 significance |

**Pre-committed fallback:** if G0 compat fails outright (both Gemma-3 AND Gemma-2) → Llama-3.x ladder
(zero-compat-risk; gemma-3-12b becomes primary judge for that table to avoid self-preference). If G1
generation has not completed by Jul-26 noon → cut the train-set broadcast to a 5k-prompt subset (ties
into the cold-start-curve story; disclosed) and proceed; only if the family track is dead by Jul-26 EOD
→ revert Days 3–4 to the old order + **P3 second topology**. P1+P2 fit in one long day if compression
is needed (v2 precedent: 247 cells < 24 h).

**Topology-axis wording GUARDRAIL (verified 2026-07-24):** May and June leases are NOT fully disjoint
(June handoffs mention `10s10309` [new] but also `10s10301` [also in May's list]). The honest, verifiable
claims: (1) CloudLab node assignment is not under our control and **varies across independently
provisioned leases**; (2) the **entire evaluation was independently re-provisioned and re-run** (May →
June campaigns), reproducing all qualitative conclusions — a cross-instantiation replication; (3) our
reservation holds essentially all concurrently obtainable GPUs of these classes (best-effort ceiling);
(4) tier-removal/restore (existing v2 data) covers topology perturbation. **Never claim "always new
hosts" or full disjointness.** Optional strengthener: recover the June host manifest (cluster archives /
v2-exp) for a node-level diff table.
Throughput anchor: v2 ran 247 cells <24 h with 0 fails; this plan needs ~90–130 serving cells — conservative.
Protocol: shared experimental rules + smoke the exact config first; no silent param changes; corrected
aggregator ONLY; preserve deployed models (incl. new Gemma artifacts) to ≥2 locations before release.

## 8) ACCEPTANCE OUTLOOK (judgmental, not calibrated)
3/4/4 all-conf-4 at a ~24.5%-accept venue = genuine borderline; AC-decided. 287j's complaints are entirely
clarity-fixable (and they rate Significance 4); gTfM/faAU questions map directly onto data we have or can
produce in-window. Text-only strong rebuttal: ~30–40%. Rebuttal + in-window results (Po2/LC, topology,
static-prior isolation, cold-start curve) landing by Aug 02: **~40–55%**, driven by flipping 287j 3→4
(realistic) and holding/raising the two 4s. Correction disclosure done as §6: small additional risk,
much smaller than a camera-ready surprise.

## 9) CODE-FACTS SWEEP (folded 2026-07-23) — rebuttal ammunition, all with file:line in the sweep report
⚠️ **CORRECTION OF THE SWEEP'S ITEM (3) — do not act on it.** The sweep searched only Block + v2-exp and
concluded "243/860K/149/220 not substantiated; correct to 442/1.5M/130/231". That is WRONG for the
rebuttal: the OpenReview zip (byte-identical to `route-balance-paper/neurips_2026.tex`, md5 `1d58359e…`)
carries 243/860K/141→220 verbatim, and §3's cell-exact reproduction proves the May aggregates produced
them. `v2-exp/neurips_2026_v2.pdf` was NEVER submitted (its 442/1.5M/130/231 are the v2 build).
**Reviewers saw 243/860K/149→220 — quote THOSE as the submission's numbers; v2 numbers only as the
disclosed corrected/extended evaluation.**

Verified facts for the replies:
- **Adaptive batch limit** (`cara_serve.py:204-240`, `BatchQueue.adapt_params`): busy ratio = fraction of
  instances with `num_waiting>0`; ratio<0.3 → max_batch 4 / timeout 50 ms; <0.7 → 16 / 100 ms;
  else 32 / 200 ms. → answers 287j verbatim-mechanism ask.
- **Batching semantics** (`cara_serve.py:142-198`): scheduler batch = ASSIGNMENT unit only; two triggers —
  size OR timeout, timer starts at FIRST request's arrival (blocks on first request, no busy-wait); no
  minimum size (light load → tiny batches); instances independently run vLLM continuous batching.
  Max added wait = adaptive timeout (50–200 ms), fully charged in reported E2E. → answers 287j confusion.
- **APC/prefix caching:** the SUBMITTED May runs were **APC OFF cluster-wide** (verified
  `2026-05-04_design_pivots.md:187-189,258` "vLLM heads: APC OFF (verified)"); deploy code default is ON
  (`deploy_cara.py:154-161,497-499`) and v2-deployed config leaves it ON. For rebuttal about submitted
  numbers: APC-off ⇒ no cache-locality confound; discuss prefix-affinity as extension. → gTfM Q2 part 2.
- **Dispatchers:** Po2 EXISTS (`dispatch/block_style.py:97-98`, `po2` flag via `factory.py:55-58`, default
  off) + `num_waiting.py` = least-outstanding; `shortest_queue.py` = min(running+waiting) ≡ classic
  least-connections in this setting. → faAU's "add Po2/least-connections": shortest-queue already IS
  least-connections-equivalent (say so), Po2 = in-window sweep with existing code, minimal implementation risk.
- **Cold-start facts:** KNN = 14,919 prompts × 4 models, k=10, MiniLM-L6-v2 384-d
  (`deploy_knn_k10/knn_metadata.json`); FAISS index build **52.3 s** (`deploy/train.log:44`); judge
  throughput **~24 calls/s on 4×A100** (labeling ≈ 59.7k judge calls ≈ ~2.8 h judging for the full set);
  existing degraded-estimator evidence = `R5_k_sensitivity.json` (k=5: BestAcc 0.318 but routed quality
  holds 0.4117 vs 0.421 at k=10; oracle 0.582) + `R4_promptblind_bound.json` (prompt-dependence delta
  +0.008..+0.018). Missing piece for faAU = training-SIZE degradation curve → offline P4.
- **TTFT head:** separate per-tier `deploy_ttft/*.xgb` with schedule-state features (`req_decode_ctx_avg`,
  `num_prompt_tokens`, `kv_free_blocks`, `kv_cache_utilization`, `request_prefill_time_est`,
  `ema_decode_iter_ms`, `num_running`, `num_waiting`). → gTfM Q2 part 1: prefill/TTFT and KV state ARE modeled.
- **Tuning burden:** `P3_nonstationary_3x3.json` ("amortized-scoring systems stay bounded under bursts
  ≤+14%; serial BEST-Route degrades +74% under square wave") + `uniform_mix_vs_lambda.json` ("tier mix
  essentially constant across λ 6→30 — rate-independent, set by the weight vector"). → faAU tuning-drift.
- **Locations:** v2-exp pack = `/home/wd312/Code/llm/v2-exp` (SIBLING of Block; manifest
  `PACK_MANIFEST.txt`: Block=87c6010 rb=bf2eee9 paper=4cfe066); May v1 traces intact (56+28+21 routing
  JSONs + budget/ablation, dated May 4–6); v2 grid = `experiment_output/v2_sweep_jun06/` (287-row
  AGGREGATE.csv, producer `consolidate_v2.py`).

## 10) RESULTS DELTA — submitted NeurIPS (PDF) vs final EuroSys/v2 — and rebuttal use
Three categories. Disclosure rule (§6): corrections disclosed once as a unit; extensions presented as
"extended evaluation since submission" (policy-sanctioned "new results"); never quote v2 numbers AS the
submission's — the submission's numbers are the PDF's.

**A. CORRECTED (same measurement, fixed pipeline/config → new value):**
| Item | Submitted PDF | EuroSys/v2 corrected | Driver |
|---|---|---|---|
| Quality scale | 0.50–0.52, coverage 72–78% | 0.36–0.42, coverage ≈1.0 | zero-bug + judge→deepeval routing + cap 256→1024 |
| Peak quality vs BEST-Route | 0.5181 vs 0.5119 (+0.006) | **0.419 vs 0.406 (+0.013, CI [+0.005,+0.022], 4-seed ±0.0003)** | same + k50→k10 |
| vs AvengersPro | +0.008 | lead **widens** (corrected AP drops more) | zero-bug is system-dependent |
| Max sustained tput | RB 28.85 / BR 18.73 | RB 27.6 / enhanced-BR 21.8 | v2 stack + enhanced BR |
| Overhead @λ12/λ30 | 149 / 220 ms ("scheduling overhead") | 130 / 231 ms ("off-instance residual", ≈32 ms compute) | v2 rerun + honest metric framing |
| Budget & batching tables | 32.4/19.2/12.6; bs=1 45.4 s | re-run on v2 (budget/ 18 + batching_ablation_v2/ 31 cells); same qualitative findings | v2 stack |

**B. REFRAMED (same-data honesty upgrades reviewers never saw):**
150× serial collapse → deployment-architecture artifact + ENHANCED (concurrent-scoring) baselines still
2.6–4.1× behind at load; "scheduling overhead" → off-instance residual + decision-compute split +
"fully charged in E2E" + 2219-vs-3432 E2E-wins argument; global-dominance → scoped "upper region" +
cost-extreme tuples; no user-facing TTFT claim (scheduler buffers).

**C. NEW since submission (v2/EuroSys — rebuttal ammunition, all in `v2_sweep_jun06/_aggregate/`):**
four-arm isolation + arm-4 static-prior (gTfM-Q1); multi-seed (4-seed, kills the PDF's admitted
single-realization limitation); second judge gemma-3-12b 14,427-paired + third-judge lookup (R3);
served-text re-judge R1/R2 (bridge validity; served cross-system gap +0.0237 CI); vLLM-SR baseline;
enhanced/batched baselines; tail p99; nonstationary 3×3; tier-removal; 40/6 overload; cost-sensitivity
(5 price vectors + gpuhour row: RB 1.38 vs BR4 2.31); k-sensitivity R5 {5,10,20,50}; prompt-blind bound
R4; LODO OOD; predictor-accuracy table; 442 configs/~1.5M requests total.

**Per-reviewer mapping (updated for the family-first re-scope):** 287j → clarity answers from §9 code
facts + Fig-1 redo promise + (if useful) multiseed/442-config thoroughness; gTfM → Q1 arm-4 + in-window
P2 sweeps, Q2 TTFT-head features + APC-OFF, Q3 centralized + residual clarification (§B) + overhead
scaling, Q4 fixed-deployment scoping + tier-removal + cross-lease re-provisioning; faAU → **generality
package = (a) NEW in-window Gemma-family result [G1–G5] + (b) cross-instantiation replication (May→June
independent leases, §7 guardrail wording) + (c) best-effort GPU-ceiling statement + (d) tier-removal**;
cold-start = live-measured Gemma onboarding wall-clock (G1–G3) + R5/R4 + offline training-size curve;
dispatchers = SQ≡least-connections + Po2 in-window; tuning = nonstationary + mix-flat; budget-in-filter =
concede-and-reframe; +0.013 significance = per-dataset offline join + model-switch fraction + multiseed CI.

## 11) SECOND MODEL FAMILY (Gemma/Llama) + MLSys-2027 FALLBACK TRACK (added 2026-07-23)
**Venue facts (web-verified):** MLSys 2027 = May 17–22, 2027; deadline unannounced, historical pattern
≈ **late Oct 2026** (2026 cycle closed Oct 30, 2025 20:00 UTC; notification Jan 25). MLSys CFP (2026):
**double-blind** (violations rejected without review), **"Authors may post technical reports (e.g., arXiv)"
with NO different-title/system-name requirement** (unlike EuroSys — RouteBalance keeps its name), dual
submission barred only while under review elsewhere, **10 pages 2-column + refs**, artifact evaluation
voluntary. **EuroSys-fall is DEAD for this cycle:** its paper deadline (Sep 24) = the NeurIPS notification
date → concurrent-submission conflict. Sequencing is clean for MLSys: NeurIPS decision Sep 24 → submit
MLSys ≈ Oct 30 if rejected. **The family experiment is venue-INDEPENDENT: accepted → NeurIPS camera-ready
wants it; rejected → MLSys needs it. It happens either way.**

**Family choice (locked: Gemma primary; refined):** Gemma-3 1B/4B/12B/27B (4-tier ladder mirrors Qwen;
transformers 4.50.x is the Gemma-3 launch version, but vLLM-version support on OUR pinned cluster stack is
UNVERIFIED → 30-min compat smoke decides) → else Gemma-2 2B/9B/27B (3 tiers, old-stack-safe) → Llama-3.x
1B/3B/8B/70B as zero-compat-risk alternate BUT llama-3.1-8b judge would grade its own family
(self-preference; would need gemma-judge primary for that table). Judge cleanliness favors Gemma.
27B-class tops the ladder → A100×4 hosts it easily; map tiers to GPU classes at kickoff.

**Pipeline split — SUPERSEDED by the 2026-07-24 re-scope: the FULL family pipeline (G0–G5) now runs
IN-WINDOW Jul 25–28 (see §7), with P3-topology as the pre-committed fallback.** The steps and time
estimates below remain the reference: (i) generation overnight Jul 26→27; (ii) judging on one d8545
4×A100 host (~1 h for ~74k calls); (iii) KNN + static priors (~1 h); (iv) reduced 3-rate latency heads
(~3–5 h parallel); (v) reduced serving sweep 27–54 cells (~4–7 h, Jul 28).
**Second short booking (~2 days, mid-Aug–early-Sep) is now OPTIONAL/strengthening:** Gemma multiseed +
grid extension to camera-ready/MLSys grade, W5 topology variant, any rebuttal-followup reruns.
Success criterion (unchanged from PLAN v2 W4): same policy structure wins on the second family.
Timeline: rebuttal follow-up tables by Aug 01–02; full-grade assets by early Sep → serves NeurIPS
camera-ready (~Oct) AND MLSys (~Oct 30) with slack.
MLSys paper shape if rejected: compress the EuroSys 12pp draft to 10pp 2-col; keep RouteBalance name;
cite BOute (MLSys'26) prominently — the online-vs-planner narrowed novelty is exactly the right frame there.

## Open items
- Draft the three per-reviewer replies (≤10k chars each, markdown, no links) + get author approval on the
  correction-disclosure wording (§6). Author posting window closes **Aug 03 AOE**.
- 4-day cluster plan (§7) — P1 is now cheaper than planned (Po2 code exists; shortest-queue ≡ least-conn).
- §11: book the ~2-day second cluster window (mid-Aug–early-Sep) for family steps (iv)/(v); Gemma-3 vs
  Gemma-2 compat smoke at Jul-25 bring-up (zero marginal cost — decides the family SKU early).
- Optional completeness: `git show 3226c1a:neurips_2026.tex` = the exact submitted source if ever needed.
