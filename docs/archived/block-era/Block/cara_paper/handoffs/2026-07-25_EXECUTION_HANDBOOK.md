# EXECUTION HANDBOOK — NeurIPS rebuttal window (2026-07-25)
**THE binding execution document.** Companion (binding for answers/drafts):
`2026-07-25_reviewer_actions_table_v2_and_drafts.md`. Everything else in `handoffs/` — including
`2026-07-23_neurips_rebuttal_audit.md` (PATH v3/v4 layers) — is EVIDENCE/HISTORY ONLY, not
executable instruction. **NO RUN HAS STARTED YET. P0 begins only after author review of this file.**

# BINDING STATE — 2026-07-25
The following override all conflicting text anywhere else:
0. **RULE 0 — PER-CELL EXACTNESS AUDIT (FIRST PRIORITY, author-mandated 2026-07-25).** No cell
   launches without a PASSED pre-flight audit, and no next cell launches before the previous
   cell's post-audit. Pre-flight (hard gate, scripted — `cell_audit.sh`): scheduler code-tree
   identity (v1-exact manifest hash for A0 / git SHA for arms) · config bytes md5 272e9903 ·
   KNN pkl md5 b9c5c606 · latency-head md5s · scheduler process env matches the arm spec
   (CARA_INPROC_PREDICTOR/CUDA_VISIBLE_DEVICES/STATIC_LAT/TIEBREAK) · vLLM tree = cara_v_11
   @ SHA 2894a9872 on every node (CORRECTED 2026-07-25 late: "route_balance_v_11" was the
   scrubbed doc name and silently fell to main; NO VLLM_USE_V1 override — the v1 deploy script
   sets none and the branch implements AsyncLLM traces) · APC OFF on every backend cmdline ·
   dataset md5 15ed0e67 ·
   /v1/config POST payload echoed and recorded · seed 1. Post-cell: completed==3534 · failed==0 ·
   response_details fields present · mix computed · INDEX + run_manifest appended · same-iteration
   rsync to VPS. ANY mismatch → cell INVALID (never aggregated), STOP the track, log, surface.
   Purpose: never again discover post-hoc that cells ran mis-configured (v1's or-zero aggregator
   + dead-config-k lessons). **A0 decision (author 2026-07-25): A0/R1 runs on the reconstructed
   v1-exact May-8 scheduler tree; A1 arms on current code (arms are v2-native); arm-1/λ12 exists
   in both trees so the code-drift delta is measured explicitly.**
1. Reproduce the exact submitted v1 stack first.
2. KNN replay and R1 are BLOCKING gates.
3. AC-critical four-arm Stage A runs before any optional generality work.
4. Standard queue-based Po2 (Po2-Queue) is mandatory; Po2-Predictive is optional supporting evidence.
5. Initial reviewer replies + AC response must stand ENTIRELY on completed evidence.
6. Gemma, topology variants, and other strengthening begin only AFTER the initial-response gate.
7. Every new quality-dependent claim must pass the corrected zero-safe evaluator internally.
8. No small quality margin (≲0.01) may be presented as a WIN without a paired CI on the fixed basis
   (ties MAY be reported: "Δq=−0.003, paired CI spans zero; statistically unchanged").
9. TTFT is profiled but is NOT an explicit term in the submitted online routing objective.
10. Shortest queue is a reactive ANALOGUE of least-connections (fewest outstanding =
    num_running+num_waiting), not strictly equivalent.
11. The camera-ready evaluator correction and the broader v2 system evolution are SEPARATE categories.
12. Any later/other section conflicting with this state is historical and non-binding.

## EXECUTION ORDER (the only binding order)
1. Manifest/config verification (`cl_manifest_main.xml`).
2. Exact-v1 deployment (scheduler_config_carafast · **Mar-15 `deploy/` pkl (baked k=10, 14,966
   examples) placed at the config's `deploy_knn` path** · score_type=judge · --max-tokens 256
   --seed 1 · Mar-27 latency heads) + hash verification (git commit, config hashes,
   model/tokenizer/KNN/head md5s, env flags). ⚠ RESOLVED 2026-07-25: config `quality_config.k=50`
   is DEAD CONFIG for type="default" (k comes from the pickle, `knn_estimator.py:430`); the Jun-6
   `deploy_knn` k=50/14,919 reconstruction NEVER SERVED — do not deploy it. The submitted PDF's
   "k=50 / 14,919" description is a CR correction item; rebuttal never repeats estimator internals.
3. KNN replay gate — **PASS 2026-07-25** (`rb-exp/replay_gate.py` v2): Mar-15 pkl reproduces
   May-logged quality 99.0–99.3% <1e-6 (median 9.1e-08) + length 100% <1e-3 tok over 300 reqs ×
   2 cells; Jun-6 rebuild median err 0.13 (0/300) — definitively not the served index.
4. **Pre-Stage-A checks:** (a) Arm-4 purity inspection (does the static path retain live/dead-reckoned
   within-tier state + reactive tiebreak + replica differentiation?); (b) arrival-determinism check
   (two same-seed smokes → identical prompt order, request IDs, max-token inputs, inter-arrival
   times, schedule hash; else build immutable per-λ trace files and replay).
5. Stage A0: arm-1 / uniform / λ12 → aggregate immediately → **R1 thresholds**: config/hash exact ·
   0 unexplained failures · mix ±2 pp vs May · tput/E2E ±5% or documented variance · no unexplained
   inversion. FAIL → stop, diagnose. 
6. Stage A1: remaining core cells.
7. Dual-basis aggregation → pre-registered comparison table → insert into gTfM/AC drafts.
8. Po2 dispatcher experiment.
9. Initial replies + AC response SUBMITTED (gate: Jul 28 11:59 BST).
10. **Generality track COMPRESSED (author 2026-07-25: DONE by end of Jul-27, 1.5 days)** —
    Jul-26 AM after phase-1: Gemma weight prefetch (fresh token ✔ gated repos) + G0 compat probe
    (fail → Gemma-2 → Llama → 5k-subset per the pre-committed fallback chain); Jul-26 PM: ladder
    deploy + G1 generation (18,553 prompts × tiers, greedy temp0/freq1.2) overnight; Jul-27 AM:
    G2 judge (d8545, ~24 calls/s) + G3 KNN + static priors; Jul-27 PM: sampled family grid
    (12 cells) + tables. Clean results land IN the initial replies (submitted ≤ Jul-28 11:59 BST);
    G1-G3 wall-clock RECORDED (= faAU cold-start answer). No weight prefetch while latency cells
    run (I/O perturbation).
11. Conditional Stage B + repeated seeds as time permits (post-gate follow-ups).

## STAGE A / PO2 CELL MATRIX (binding counts)
- **Stage A0**: 1 cell (arm-1/uniform/λ12; included in the 12 below — the R1 gate, not extra).
- **Stage A1**: **12 core cells** = arms 1–4 × uniform × λ{12,24,30}.
- **+2 optional** 40/6 overload cells (arm-1, arm-4) after the core table completes.
- **+3 conditional** Arm-4b cells (uniform × λ{12,24,30}) if the purity check requires (below).
- **Po2**: passthrough × Po2-Queue × λ{6,12,18,24,30} (5) + passthrough × {SQ, RR} in-session
  references (~3) + best-AvengersPro × Po2-Queue × 2λ + 1 SQ reference (~3) ≈ **11 cells**.
  Po2-Queue = sample 2 replicas uniformly WITHOUT replacement (seeded) → fewer outstanding
  (num_running+num_waiting); <2 replicas → full comparison. Po2-Predictive (existing block_style,
  sidecar-predicted delay) = OPTIONAL bonus only. Standard Po2 needs a ~15-line dispatcher + smoke.
- Superseded and NOT to be executed: any "launch the full ~42–60 four-arm grid first" plan.

## ARM DEFINITIONS + PRE-REGISTERED COMPARISONS (decided before results)
Arm 1 = full objective (learned/live latency). Arm 2 = w_lat=0 + reactive SQ tiebreak.
Arm 3 = w_lat=0 + predictive T̂ tiebreak. Arm 4 = static cross-tier prior (TPOT_nominal × L̂) as
gated (`CARA_STATIC_LAT`) — **PURITY VERDICT (2026-07-25, code-inspected): PURE CONTROL — Arm-4b NOT
required.** `cara_serve.py:1476-1500`: the gate replaces only the latency VALUE in the cross-tier
score ("Identical normalization + queue tiebreak; default path unchanged"); the within-tier tiebreak
(sort key `(-score, q, lru)`) continues to read LIVE/dead-reckoned queue state. Arm-4 = static
cross-tier prior + live reactive within-tier placement = exactly gTfM's asked-for control ("per-tier
average TPOT and live queue state"). "Zero telemetry" in older notes = the latency VALUE computation
only. Stage A stays 12 core (+2 optional overload). Historical pending-note below superseded:
if it retains live within-tier state → document as the pure static-prior control; if it removes live
telemetry → rename "static cross-tier prior + no-live-telemetry broader control" AND add
**Arm 4b = static cross-tier prior + live reactive within-tier queue state** (env composition
STATIC_LAT + TIEBREAK=queue; expected zero new code).
Comparisons: **1v2** value of cross-tier latency in model selection · **2v3** predictive vs reactive
within-tier (no cross-tier term) · **1v4b** learned/live cross-tier vs static prior WITH live queue
state (gTfM's literal question) · **4v4b** effect of removing live telemetry. Report each: mean E2E,
throughput, fixed-basis quality, tier mix, p99 E2E, absolute + relative deltas. Never attribute 1v4
solely to XGBoost if telemetry also changed.
**Conditional Stage B:** 2v3 wash → drop arm-3 from presets; 1≈4/4b → expand arms {1,4(4b)} across
{wq0.8, wc0.8} × λ{6..30}; 1 vs 4/4b materially different → repeated seeds + diagnosis FIRST; any
anomaly → stop, never expand it; mechanism unsupported → revise rebuttal framing to observations.
**Interpretation thresholds:** Δ<~5% "recovers most of the benefit"; 5–15% "retains the dominant
share"; larger → "both the formulation and learned/live estimation contribute". Fresh submitted-stack
evidence overrides all v2-derived draft sentences.

## CORRECTED FACTS (consistent everywhere; supersede older phrasings)
- **TTFT:** "The online assignment objective is decode-dominated: learned per-(model,GPU) TPOT ×
  queue-aware decode iterations (pending_decode/running + own predicted output). A separate TTFT head
  exists for profiling/analysis; TTFT is not an explicit objective term in the submitted scheduler."
  APC/prefix caching was DISABLED in all submitted runs → prefix-hit effects are not a confound;
  model choice affects KV pressure via model size and sequence lengths; explicit prefix affinity =
  future work.
- **SQ/LC:** "The submitted shortest-queue dispatcher selects the instance with the fewest
  outstanding requests (num_running + num_waiting) — the closest reactive LLM-serving analogue of
  least-connections; it does not account for heterogeneous remaining decode work."
- **Quality-basis rule (replaces 'gate silently'):** every new quality-dependent result is aggregated
  on BOTH the submitted paper basis and the corrected zero-safe basis. Runtime claims may be reported
  from the submitted stack. A quality claim is permitted only if its qualitative interpretation also
  holds on the corrected basis; if the bases disagree → claim PROHIBITED, escalate to author. Small
  differences may be reported as ties when the paired CI spans zero.
- **No mixed tables:** no rebuttal table mixes submitted-stack and v2 runtime cells, old/new caps,
  different routing signals, or different evaluators without explicit internal labeling.
- **Corrected-v1 claim guardrails (⛔ as WINS):** RB-vs-BR peak = statistical tie (paired
  Δ −0.0033, CI [−0.0112,+0.0045]); upper-envelope holds only beyond BR's ~18.6 req/s saturation;
  iso-quality latency advantage requires "under load" (λ≥18); RB-vs-AP nominal (+0.0074, CI spans 0);
  iso-quality COST advantage (~4–6%) and PT separation (+0.038 CI[+0.029,+0.048]) are safe.
  **`0.4187 vs 0.4056` is a v2/signal-aligned result and can NEVER be described as "the submitted
  result after fixing the evaluator"** (that result is 0.4042 vs 0.4056, a tie).

## CAMERA-READY (corrected framing; supersedes "claims delta is tiny"/"worst case cannot be rejection")
The system and central fusion contribution remain the same, but several quality-dependent claims
require correction or narrowing (tie at peak; envelope beyond saturation; under-load latency
qualifier; AP nominal). A camera-ready must clearly distinguish the post-hoc evaluator correction
from the broader v2 system evolution. Diff size alone does not guarantee permissibility; the change
note + AC courtesy note are mandatory; the corrected-v1 vs v2 provenance labels are never mixed.

## COLD-START ACCOUNTING (single source; measured vs derived labeled)
| Quantity | Value | Status |
|---|---|---|
| Prompts per added model (full corpus) | 18,608 (train-only minimum 14,919) | measured corpus size |
| Judge calls per model column | = prompts per model | definitional |
| Sustained judge throughput | **~4.2 pairs/s aggregate, Llama-3.1-8B (metric of record) on 4×A100 sharded bf16** | MEASURED 2026-07-27 23:40Z |
| ~~24 calls/s~~ | ⚠ CORRECTED 2026-07-28: that rate was the **gemma-3-12b-it judge-ALT** (2026-06-07_v2_rerun_handoff.md:125), NOT Llama-3.1-8B. Do not use for onboarding cost. | mis-attribution |
| Per-model judging time | **18,608 / 4.2 ≈ 74 min per model column** at the metric-of-record rate — judging ONLY; excludes generation, setup, I/O, retries | DERIVED from the MEASURED 4×A100 Llama rate (supersedes the ~13 min figure, which used the judge-alt rate) |
| "~2.8 h" figure | judging all four Qwen columns (~74.4k calls) at the same rate | DERIVED estimate |
| Candidate-generation wall-clock | to be MEASURED (family onboarding G-track) | pending |
| FAISS index build | 52.3 s (14,966-example build log) | MEASURED |
| XGBoost per-tier training time | `train_time_s` fields in `deploy_*/training_metrics.json` (pull exact values at draft time) | MEASURED (per file) |
| Judging GPU-hours per model column | ≈ 4 GPU × 13 min ≈ 0.9 GPU-h | DERIVED |
Rebuttal wording: "Measured steady-state judge throughput implies ≈13 minutes per model column,
excluding candidate generation, setup, I/O, and retries; we report those stages separately and will
provide measured end-to-end onboarding wall-clock when available."

## EXECUTION CONSISTENCY GATE (before any number enters a response)
Per new cell, record: git commit · config hash · model revision · tokenizer revision · KNN/index
hash (Mar-15 `deploy/` pkl md5) · latency-head hashes · score_type · effective k=10 (pkl-baked;
config k=50 dead) · quality normalization · max_tokens=256 · APC state ·
eager/async/chunked-prefill flags · node mapping · request-trace hash · prompt order · arrival
timestamps · seed · exact command line · output artifact path. Stored as run_manifest next to
results; rsynced with them (two-location).

## RESPONSE GATES
Initial complete replies + AC comment by **Jul 28 11:59 BST**, standing entirely on completed
evidence (forthcoming-evidence mentions allowed only where no claim depends on them). Evidence-only
follow-ups through the discussion deadline (authors post until Aug 03). Text freeze: all numbers
file-traced; cross-reply consistent; no placeholders.

## SUPERSEDED — HISTORICAL ONLY — DO NOT EXECUTE (in the audit doc and older prompt layers)
Family-first scheduling · full-grid-before-response four-arm plans (~42–60 cells first) · proactive
correction-disclosure strategy drafts (§6 historical layers; the decided posture = silent-in-rebuttal
+ corrected-at-CR with rails) · old TTFT framing ("TTFT modeled in the objective") · SQ≡LC framing ·
old acceptance-probability prose · any CR text calling all v2 changes "a simple evaluator fix" ·
PATH v1–v4 sequencing layers in `2026-07-23_neurips_rebuttal_audit.md` (that file = evidence archive:
version forensics, corrected re-aggregation, review verdicts, anticipated follow-ups).

## ═══ BINDING RULINGS 2026-07-27 (author-locked; supersede conflicting text above) ═══
1. RB PRESET IN FINAL TABLES: exactly ONE setting shown per table — DEFAULT uniform (arm1's
   preset); ss532 only if it preserves a key trend uniform breaks; the selection is named in
   each table's verification line; unshown preset's cells stay banked (truthful-if-asked);
   no cross-preset claims where presets diverge.
2. V1EXACT CELLS = the paper-facing basis (tree 553d76c3→v4 lineage, A0 re-gate certified);
   BR4/AP latencies = REAL numbers used NORMALLY vs v1 cells and each other; appendix note:
   serial routing collapses under load w/ high run-to-run variance (~2x compute component
   under co-tenancy; ~4% e2e at low rate; amplifies near saturation); routing decisions
   BIT-EXACT vs May (0.00pp, both routers); dispatcher-independent; largely lost to RB whose
   stability under high rate = amortized batch scoring (thesis-reinforcing).
3. GATES: routing correctness (mix ≤2pp) hard-gates; latency recorded/reported, judged only
   by claim-materiality; gate bands derive from the measurand's VARIANCE CLASS (calm-regime
   anchors at low λ; never uniform bands on near-saturation quantities).
4. SCHEDULE: pw-sweep x28 + EXP2-ss532 x8 = post-Gemma overnight batch (qwen_overnight.sh,
   staged); "updated Table 4 (+Po2)" framing holds — pw rows land ~07:00Z Jul-28 pre-wire.
5. TABLE BASIS MAP: V1/V2 = v1exact; V3/V4 = current-tree within-window (labeled); no mixed
   tables; per-table verification lines name basis + preset + gates.
6. NO PUBLISHED-CONFIG DUPLICATES (author 2026-07-27): no rebuttal table displays a new run of
   a published-config cell on a different basis (two-numbers-one-setting inconsistency); V4 =
   ladder-only (unpublished configs; f75 internal reference); anchor cells for published
   configs are NOT rerun.
7. RELATIVES-ONLY FOR CURRENT-TREE TABLES (author concern 2026-07-27): V3/V4 (and any
   current-tree surface) report ONLY relative quantities (deltas, ratios, % change, normalized
   curves) — never absolute e2e/quality that could be cross-referenced against published rows.
   Absolutes appear only in v1exact tables (paper-consistent) and Gemma (no paper counterpart).
   Internal verification files retain all absolutes.

## BINDING RULING 2026-07-27 ~14:35Z — REVIEW-FIRST; FLIP+GEMMA = GOOD-TO-HAVE
Author order: FULL REVIEW + REVISE against all existing banked results BEFORE anything
redeployment-gated. Topology flip (V3 argmax column) and Gemma model family are GOOD-TO-HAVE
— not required by reviewer 3 — and run only after the review completes, if at all. No
unconditional model-family promises in any reply. Arm2 ×4 v1exact banked 14:32Z (3-arm table
complete: arm1/arm2/arm4 × λ{6,12,24,30}).
CORRECTION (author, ~14:40Z): flip + Gemma are NOT optional — they REMAIN REQUIRED, targeted
at raising reviewer 3 from weak-accept to accept. The 14:35Z ruling is sequencing only: full
review/revision pass FIRST (close every gap), then the redeployment-gated families. Gemma
forthcoming-results sentence stays in the replies per the approved plan.
RULING (author, ~14:55Z): AP pw-sweep x28 = GOOD-TO-HAVE, not rebuttal-required (V2's required
AP row = banked pw0.7). HOLD all experiment launches until the full-review conclusion; the
review's gap list (missing key cells, if any) runs FIRST, then topology flip, then Gemma.

## BINDING RAIL #8 (author, 2026-07-27 ~16:1xZ) — HIDE V2 TOTALLY
This is a V1 REBUTTAL. Reviewer-facing surfaces NEVER reference: the v2 paper draft (any tex),
v2-era (June) results, or any version/build/tree language. V3 labeling = "same system, both
topologies measured in this discussion window" — no version distinction surfaced. Scrub tokens
for reply verification: "v2", "updated draft", "revised paper", tex filenames. V4 = relatives
only (rail #7 intact), internal source = v1exact 6-cell set; no absolutes displayed.

## BINDING RULING 2026-07-28 ~00:0xZ — BEST-Route t=0.5 REMOVED EVERYWHERE
Author: at t=0.5 the confidence threshold is high enough that nearly all requests fall through
to the default model — earlier debugging established this and the flip cells confirm it
(98.4% of requests to a single tier at both λ12 and λ24 = an always-one-model baseline, not a
router). REMOVE all BR4/BEST-Route t=0.5 rows and references from the rebuttal. The trained
router is represented by ARGMAX mode only (the submitted Table-4 mode) — in the topology block,
in any dispatcher table, and in the Gemma family table (G5 matrix must be argmax, not t0.5).
