# Rebuttal-window /goal + /loop prompts (2026-07-25) — copy-paste source

Usage: set **GOAL A** now; swap to **GOAL B** when drafting starts (Jul 27ish); swap to **GOAL C**
after submission. Start the **/loop** (dynamic, no interval) once execution begins — it can drive P0
itself. The loop appends to `cara_paper/handoffs/2026-07-25_window_execution_log.md` (creates it if
absent). Cluster manifest: `cl_manifest_main.xml` (new lease).

---

## GOAL A — master window goal (set at start)

GOAL: Execute the NeurIPS Submission28271 rebuttal window. Cluster live until Jul 31 19:00
(manifest: cl_manifest_main.xml). HARD GATE: initial per-reviewer replies + AC comment SUBMITTED
on OpenReview by Jul 28 11:59 BST.

PLAYBOOK (follow, do not re-derive): Block/cara_paper/handoffs/2026-07-25_EXECUTION_HANDBOOK.md
(BINDING: order, gates, cell matrices, rules) + 2026-07-25_reviewer_actions_table_v2_and_drafts.md
(answers, drafts, guardrails) + rb-exp/v1-exp/V1_PACK_MANIFEST.txt (V1 RERUN RECIPE).
2026-07-23_neurips_rebuttal_audit.md = EVIDENCE ARCHIVE ONLY — never execute from it.

ORDER (authoritative sequence = 2026-07-25_EXECUTION_HANDBOOK.md; summary):
P0 manifest-verify (cl_manifest_main.xml) → EXACT v1 deploy (carafast · deploy_knn k=50 ·
score_type=judge · --max-tokens 256 --seed 1 · Mar-27 heads) + hash verification → KNN REPLAY GATE →
pre-Stage-A checks (Arm-4 purity inspection; arrival-determinism check) → STAGE A0 (arm-1/uniform/λ12
= R1, BLOCKING thresholds) → STAGE A1 (12 core cells: arms 1–4 × uniform × λ{12,24,30}; +2 overload
optional; +3 arm-4b if purity check requires) → dual-basis aggregate → pre-registered comparison
table → insert into gTfM/AC drafts → EXP-PO2 (Po2-Queue mandatory, ~11 cells + refs) → FREEZE: verify
every number, enter replies → SUBMIT ≤ Jul 28 11:59 BST → ONLY THEN G0 Gemma compat probe → family
track (sampled 12 cells) + P4 offline + follow-ups Jul 28–30 → CONDITIONAL Stage B + repeated seeds →
bonus slots → Jul 31: rsync ×2, preserve models+manifests, release.

HARD RULES: smoke the exact config before any long run; NO silent parameter changes — on
OOM/crash/anomaly STOP and report; v1 config is immutable; aggregate new cells ONLY with the
dual-pass reagg_v1_zerofix.py (report paper-basis, gate on fixed basis silently); every posted
number traced to a file; comparisons within-session only; ⛔ never state: the +0.006 peak-vs-BR
win, upper-envelope-at-every-throughput, blanket iso-quality latency win, statistical RB>AP
separation. Replies: ≤10k chars, markdown, NO links, inline tables, no coverage column.
Internal codename never appears in anything posted. Verify before reporting success.

---

## GOAL B — freeze-phase swap-in (Jul 27 → submission)

GOAL: SUBMISSION CRUNCH — replies entered before Jul 28 11:59 BST. No new experiment launches
(unattended family pipeline may continue only if author-approved). Fill EXP-ARMS/EXP-PO2 tables
into the §5/§6 drafts in 2026-07-25_reviewer_actions_table_v2_and_drafts.md; verify every number
against aggregates (dual-pass, both-bases gate); enforce ⛔ list + no-links + 10k limits; author
reviews wording before entry; submit all four responses (3 reviewers + AC), confirm on OpenReview.

---

## GOAL C — follow-up-phase swap-in (post-submission → Jul 31 19:00)

GOAL: Phase-2 follow-ups with the cluster held live. Priority: G0 Gemma compat probe (SKU pick + weight pull) → family track G1–G5 (sampled 12
cells, v1 recipe, record onboarding wall-clock) → post family + P4 tables in discussion by
Jul 30 → answer reviewer follow-ups (anticipated-answers list in the audit) with experiments if
needed → bonus: topology-B, CR-prep v2-recipe duplicates of new tables → Jul 31 before 19:00:
anomaly reruns, rsync everything ×2 locations, preserve deployed+new models with manifests,
release. Same hard rules as the window goal.

---

## /loop prompt (dynamic mode — no interval; self-paced)

NeurIPS rebuttal window autopilot. Each iteration:
(1) READ the tail of cara_paper/handoffs/2026-07-25_window_execution_log.md (create if absent) +
live state: cluster reachability, running sweeps (pgrep -f / screen / log tails), newest result
cells, failure counts, disk on VPS and cluster nodes, current phase vs the EXECUTION ORDER in
2026-07-25_EXECUTION_HANDBOOK.md.
(2) VERIFY: every new cell passes sanity (completed==3534, failed==0, response_details fields
present); aggregate incrementally with the dual-pass reagg (paper-basis reported, fixed-basis
silent gate); if any planned claim becomes basis-dependent → STOP-item, surface to author.
(3) ACT: when the current step completes, launch the next step per the playbook order. Auto-fix
ONLY infra-class issues (dirs, PYTHONPATH, stale process kill, transient network retry, permissions
— per CLAUDE.md); NEVER change experiment parameters — on OOM/crash/anomalous numbers: STOP the
affected track, log, surface. rsync any new results to the VPS (two-location rule) in the same
iteration they were produced.
(4) DOCUMENT: append a timestamped entry to the execution log — phase, cells done/total, failures,
notable numbers, action taken, next action, ETA vs the Jul-28 11:59 BST gate — and keep entries
one-per-iteration, terse, handoff-grade.
(5) PACE: schedule the next wakeup to match what is running (long unattended chains: 30–45 min;
short steps or gates: ~15 min; blocked on author input: say so and lengthen). Once the freeze
begins, stop launching experiments (except an author-approved unattended family pipeline) and
shift iterations to draft-verification support.

---

Notes: the loop may drive P0 as its first iteration. Swap GOAL B in before the crunch so the loop's
freeze behavior matches the goal. If the loop is interrupted, the execution log is the single
source of truth for resuming — start each new session by reading its tail.

---

## SUPERSEDED — HISTORICAL ONLY — DO NOT EXECUTE: execution refinements v1 (counts/staging replaced by v2 below + the HANDBOOK)

1. **Two response gates:** initial complete replies + AC comment by **Jul 28 11:59 BST**;
   evidence-only follow-ups through the discussion deadline. **Initial replies must stand without
   unfinished experiments** — announcing forthcoming evidence is allowed only where no claim depends
   on it (family/topology "underway" mentions).
2. **EXP-ARMS is STAGED.** Stage A (first, ~13 cells): four arms × uniform × λ{12,24,30}, arm-1 λ12
   doubling as R1; optional 40/6 overload pair. ANALYZE + INSERT into drafts BEFORE Stage B.
   Stage B (overnight, ~34 cells): arms {1,2,4} × {wq0.8, wc0.8} × λ{6,12,18,24,30} + uniform
   λ{6,18} fill-ins; 2-seed replicates Day-2. Initial reply stands on Stage A if B is incomplete.
3. **Quality guard (sharpened):** dual-basis aggregation for EVERY new cell; if the fixed evaluator
   changes any qualitative conclusion → claim PROHIBITED + escalate to author; **never post a small
   quality margin (≲0.01) supported only by the paper-basis evaluator** — it must hold with CI on
   the fixed basis.
4. **Po2:** standard queue-based Po2 (random-2 → fewer outstanding requests) is MANDATORY; the
   predictive block_style variant is an optional stronger arm only.
5. **R1 pass thresholds (gate is quantitative):** config/hash exact · zero unexplained failures ·
   tier mix within 2 pp of the May cell · throughput/E2E within 5% or variance documented and
   author-acknowledged before proceeding.
6. **Per-run manifest:** every sweep emits a run_manifest (code commit, config hashes, model +
   tokenizer + predictor md5s, node mapping from cl_manifest_main.xml, env flags, seed, exact
   command line) stored next to its results and rsynced with them.
7. **Text freeze:** all numbers file-traced; cross-reply consistency check (same number, same value
   everywhere); no placeholders; no promises a claim depends on.

---

## EXECUTION REFINEMENTS v2 (2026-07-25 final, BINDING — supersedes v1 counts/rules where different)

1. **Counts:** Stage A = **12 core cells** (4 arms × uniform × λ{12,24,30}; arm-1/λ12 doubles as R1,
   not an extra cell) **+2 optional** 40/6 overload cells (+3 if arm-4b added) — use consistently.
2. **Stage A0 = BLOCKING gate:** run ONLY arm-1/uniform/λ12 first → aggregate immediately → R1
   thresholds (hash-exact · 0 unexplained fails · mix ±2pp vs May · tput/E2E ±5% or documented
   variance · no unexplained metric inversion). FAIL → stop and diagnose. PASS → Stage A1 (remaining
   11 core cells).
3. **Identical arrival traces across arms:** verify seeded-generator determinism (code inspection +
   double-smoke same-seed arrival-schedule diff) BEFORE Stage A; if non-deterministic, build
   immutable per-λ trace-file replay. Same prompt order, inter-arrival times, seeds, session,
   deployment, and generation settings across all arms.
4. **Pre-registered comparisons (decided BEFORE results):** 1v2 = value of latency in cross-tier
   selection · 2v3 = predictive vs reactive within-tier (no cross-tier term) · 1v4 = learned/live vs
   static prior. Report each: mean E2E, tput, fixed-basis quality, tier mix, p99 E2E, abs+rel deltas.
5. **Arm-4 causal purity — VERIFY AT DEPLOY, BEFORE STAGE A:** check what the within-tier tiebreak
   read in the arm-4 path (dead-reckoned queue state live or not). If arm-4 = static prior + NO live
   within-tier state → rename accurately ("static/no-telemetry control") AND add **arm-4b = static
   cross-tier prior + live reactive within-tier placement** (env combination STATIC_LAT +
   TIEBREAK=queue; likely zero new code; +3 cells uniform×3λ) — arm-4b is gTfM's literal ask. Never
   attribute a 1v4 delta solely to XGBoost if telemetry also changed.
6. **Refined margin rule (replaces mechanical ≤0.01 ban):** no quality margin ≲0.01 may be presented
   as a WIN without a paired CI on the fixed basis; small deltas MAY be reported as ties
   ("Δq=−0.003, paired CI spans zero; statistically unchanged").
7. **Stage B is CONDITIONAL on Stage A:** 2v3 wash → drop arm-3 from presets. 1≈4 → expand arms
   {1,4}(+2) across presets/rates. 1≠4 materially → repeated seeds + diagnosis BEFORE broad sweep.
   Any anomaly → stop, investigate, never expand an anomaly across 34 cells. Mechanism unsupported →
   revise rebuttal framing to match observation.
8. **Initial AC response stands on Stage A alone** (cross-tier value; within-tier predictive vs
   reactive; static-prior share; learned = essential vs calibration/headroom). Stage B = supporting.
9. **Interpretation language follows FRESH evidence** (draft sentences are hypotheses until Stage A):
   Δ<~5% → "recovers most of the benefit"; 5–15% → "retains the dominant share"; larger → "both the
   formulation and learned/live estimation contribute". v2 results never substitute.
10. **Sequence:** manifest-verify → v1 deploy → hash-verify (model/tokenizer/predictor/heads/config/
    env) → KNN replay gate → A0 → R1 thresholds → A1 (11 cells) → dual-basis aggregate →
    pre-registered table → insert into gTfM/AC drafts → freeze defensible initial answer →
    conditional Stage B → repeated seeds → Po2 → (post-submission) generality.
