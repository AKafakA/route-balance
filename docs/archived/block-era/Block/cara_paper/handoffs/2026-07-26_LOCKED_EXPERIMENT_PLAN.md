# 🔒 LOCKED EXPERIMENT PLAN — NeurIPS rebuttal window (authored 2026-07-26, author-approved)

**This document is the authoritative experiment plan for the remainder of the window.
Read it FIRST every iteration. No reordering, no scope change, no parameter change without
the author's explicit word in chat. On conflict with any older doc, THIS doc + the
execution log's later author decisions win.**

## The queue (node0, single process: `~/run_final_queue.sh` v3, running since 11:45Z Jul-26)

Order is author-set (Po2 first). Every cell: RULE-0 pre-audit + post-audit via
`~/cell_audit.sh`; skip-existing logic; results under `~/Block/experiment_output/jul25_stage_a/`.

1. **`po2_router_ext_full.sh`** — base-matrix fill: only `br4_po2_l6` missing (rest skip).
2. **`po2_variants_full.sh`** — the 49-cell Po2 variant space (PC key #2 completion):
   BR4 confidence t{0.3, 0.5, 0.7} × po2_queue × λ{6,8,10,12,18,24,30} = 21 cells
   + AvengersPro pw{0.25, 0.39, 0.53, 0.8} × po2_queue × 7λ = 28 cells.
   Pipeline mode; artifacts verified present (all 5 AP pw dirs). ETA ~17:30Z.
3. **`burst_table.sh`** — four-arm burst (PC key #1): arms 1–4 × {sq40_6, sq30_3, **gamma025**}
   (arm1/arm4 sq40_6 skip — exist in `overload/`) + sweet-spot ss532 (0.5/0.3/0.2) ×
   arms{1,4} × 3 patterns. **AUTHOR CHANGE Jul-26: gamma burstiness 0.3 → 0.25** (cells named
   `gamma025`; script edited + verified on node0 before the burst phase starts). 0.5 and 0.75
   = later ablation ladder ONLY if 0.25 works — not queued. Banked pre-swap: arm1_sq30_3
   (valid, kept) + arm1_gamma03 (b=0.3 — SUPERSEDED data, never the rebuttal value; arm1 reruns
   at 0.25). Decision pair: `arm1_gamma025` vs `arm4_gamma025` (~19:00Z; 15 burst cells remain).
4. **`stage_b_expansion.sh`** — ⛔ DROPPED by author Jul-26 eve (5/42 cells banked, kept; REVIVE-ONLY if all other tracks finish early). Was: arms{1,4} × 3 presets × 7λ.
   ETA all Qwen cells ~23:00Z.

**PERMANENTLY EXCLUDED:** `v1exact_fill.sh` (corrupted; its work is done),
`v1exact_weightgrid.sh` (author-CANCELLED weight grid — never relaunch; its 10 grid jsons +
16 orphan bench logs in `cara_weight_sweep/` are dead work, excluded from all tables).

## Banked cell inventory (all audited, synced node0 + VPS)

arms 28 · overload 2 (square40_6 arm1/arm4) · po2 17 (passthrough po2/sq/rr × 7λ + refs) ·
po2_routers 15+1 filling (br4_po2×7, ap_po2×7, br4_sq_l12, ap_sq_l12) · burst 2 ·
cara_weight_sweep 17 (7 uniform v1exact count toward reproduction; 10 = cancelled grid, dead) ·
stage_b 0 (pending). Aggregator `reagg_jul25_dualpass.py` parses ALL current + pending
name patterns (unit-tested Jul-26): burst {sq40_6,sq30_3,gamma025 (+gamma03 legacy, gamma05/075 future)} ± ss532, stage_b ss532,
po2_routers t/pw variants. Dual basis: paper-basis reported, fixed-basis silent gate.

## Binding drafting rules (author, Jul-26)

- **Burst evidence in the rebuttal = GAMMA CELLS ONLY** (native vllm-bench `--burstiness 0.25`,
  standard + verifiable). Square-wave (our env-gated CARA_SQUARE) = backup ONLY if gamma shows
  no learned-predictor win, and then only with full mechanism disclosure (reviewer will judge).
  Never mix mechanisms in one table. NO burst-superiority wording before arm1_gamma025 + arm4_gamma025 both audit-PASS (author set b=0.25 Jul-26; 0.5/0.75 later ablation if it works).
  If gamma shows no win → STOP-item, author chooses (square-with-disclosure vs no burst claim).
- Po2 = APPEND a po2 column/rows to Table-4's existing structure; v1 rows byte-unchanged;
  comparability via the measured bridges. Never re-assert the 2 basis-dependent claims; no k=50
  statements; margin + ⛔-claim rails per EXECUTION_HANDBOOK.
- Chat status in plain words (no bare step symbols). Codename never public.

## After the queue (author-approved sequence)

1. **FAMILY track FINAL (Jul-26 ~16:00Z): GEMMA-4 DROPPED by author (spike: arch needs modern
   vLLM; serve died at CUDA-graph capture on Turing CC7.5 = V100-tier preview — kills the story).
   Family = GEMMA-3 DENSE on the frozen fork: 1b-pt→3×A30, 4b-pt→5×A30, 12b-pt→3×V100 (TP4),
   27b-pt→2×A100 nodes × 2 instances TP2 = 4 replicas (V2-instances config). fp16 (v1 parity).
   Judge = deepeval-llama3.1-8b (v1 judge). NO branch decision tonight: queue → flip (armed
   watcher) → Gemma-3 auto-starts. Cells early afternoon Jul-27. Fallback Llama-3 → 5k subset.**
2. **Offline learned-vs-static prediction-accuracy analysis** (VPS-side, owed; run during sweeps).
   Design locked Jul-26 (code-verified): static prior = per-tier nominal TPOT ms/tok ×
   predicted_length, zero telemetry (`cara_serve.py:1476-1488`); learned = per-tier XGBoost on
   23-feature live-telemetry vector (`cara_serve.py:920-956`). broadcast_results is EMPTY in
   result jsons → learned predictions NOT offline-recomputable; honest scope = (a) static-prior
   TPOT MAE vs actuals sliced by λ (error growth with load), (b) learned heads' held-out
   training-time MAE from Mar-27 artifacts, clearly labeled.
2b. **P4 COLD-START CURVES (author-approved Jul-26 eve — fills faAU-W2's "quantify degradation
   under partial or missing quality signals"):** offline replay of the DEPLOYED Mar-15 KNN
   estimator with its exact serving recipe (gate-verified <1e-6): P4a = corpus subsampled at
   {100,50,25,10,5,2}% (seeds ×3 at ≤25%) → BestAcc/Top2/delivered-quality vs label fraction;
   P4b = per-model full-label removal (new-unlabeled-model scenario; ties to tier-removal E7).
   Validation gate: 100% replay reported against deploy/evaluation_results.json. VPS CPU only,
   isolated venv ~/venvs/p4_jul26; artifacts: window_setup_work/p4_coldstart_curve.py →
   _aggregate/p4_coldstart_jul26.json (two locations). RUNNING since ~15:45Z Jul-26.
3. **Topology answer (AUTHOR DECISION Jul-26, per ChatGPT-web suggestion + pre-planned audit T1):
   NO new experiments needed for the initial rebuttal.** T1 package from existing data:
   (a) June E7 tier-removal case study (v2_sweep_jun06/tier_removal — ⚠ JUNE/v2-era config:
   behavior-level claims ONLY — 0 fails, KNN re-normalizes, uniform 0.371 unchanged, mix 44/27/29,
   e2e 2.9s, honest wq0.8 0.419→0.372 14B-bound — NEVER numerically cross-referenced to the
   submitted Table 4, never presented as the submitted build); (b) THIS lease's v1-exact
   re-provisioning replication (3rd independent provisioning, R1 gate PASS: mix Δ≤0.6pp, tput
   match, e2e +0.6%); (c) guardrail wording: independent re-provisioning/replication, NEVER
   "always new hosts".
4. **T2 TOPOLOGY-B FLIP SWEEP — AUTHOR-PROMOTED TO CRITICAL PATH (Jul-26 evening decision):**
   within-pair count swap on the A30 fleet ONLY: 5×7B + 3×3B → **3×7B + 5×3B** (72B/14B
   unchanged; author's stated plan — NOT the audit's premium-halving example; NO new
   (model,GPU) pairs). **FULL LEARNED system runs (all heads exist) — no static shortcut.**
   Cells mirror the Gemma sampling: ss532 preset × λ{12,18,24} × {RB full-learned,
   passthrough-SQ, Po2} = 9 cells, v1 config, audits as everywhere. Original-topology anchors
   come free from today's queue (stage_b arm1_ss532 λ12/18/24 + po2 passthru sq/po2 same rates).
   Prep TODAY (config with field-parity diff, staged); 3B weights → 2 flipped A30 nodes
   prefetched during the Jul-27 judge phase (not during latency cells); deploy+smoke after
   Gemma sampled cells (~16:00Z) → 9 cells → table by ~19:30Z Jul-27.
   The June-E7 case study (item 3a) demotes to BACKUP if the flip lands — era-clean preferred.
4b. **P4-ONLINE COLD-START SWEEP (author-locked Jul-26 eve, CONDITIONAL):** triggers only if
   the Gemma experiments finish by end of Jul-27 BST; else the OFFLINE analysis (2b) = backup
   evidence. Spec: KNN index at {0, 25, 50, 75}% corpus × 3 rates × RB at the Table-4
   BOTH presets (uniform + ss532, author pairing rule) = 24 cells + baseline refs banked + 1-2
   leave-one-source-out ablations (index built without one source; evaluated ONLINE on that
   source's prompts; 'source' field verified in corpus). Rates (author-locked): λ{6,12,30} = the low/moderate/high standard; "0%" = degenerate uniform-prior pkl (identical scores → quality
   term washes out; same code path/config, artifact-only change); held-out source = largest
   group. Index builds offline (Mar-15 recipe, staged Jul-27 AM); ~1.5-2h cluster post-Gemma.
5. Tables → the four reply drafts (gTfM/AC/287j/faAU) → number re-verification vs files →
   k-wording sweep. **AUTHOR GATE (revised Jul-26 eve): ALL first-rebuttal DATA in hand by
   end of Jul-27; midnight + morning of Jul-28 reserved for TEXT EDITING; submit ≤ Jul-28
   13:00.** Freeze at submission.
6. Post-gate: full family grid, preservation close-out (checksummed manifests ×2 locations)
   before lease end Jul-31 19:00.

## Open flags (recorded, not resolved)

- λ30 cara-path e2e +34.7% vs May (SQ bridge clean at λ30 ⇒ system-path-specific) — flagged.
- ap_sq_l12 tier mix = 100% 72B (AP pw0.7 policy; recorded, not judged).
- node0 sshd drops sessions: ALL kills/launches via uploaded scripts + setsid + marker polls;
  bracket-pattern pkill (no cmdline self-match); verify launches by process+log+first-cell.
- Offered to author, NOT approved (do not run unless approved): extra gamma mean rates (18, 30).

## 🔒 AUTHOR PAIRING RULE (Jul-26 eve, binding for all remaining experiments)
RB runs at BOTH uniform and sweet-spot 0.5/0.3/0.2 in every remaining experiment (flip: 12
cells; Gemma G5: 12 cells; P4-online: doubled RB arms) — prevents cross-table preset
inconsistency (uniform-only earlier vs sweet-spot-only later). Baselines (SQ/Po2/RR) are
preset-free. GAMMA-HIGH addendum runs uniform-only (in-flight when rule landed) — ss532
gamma-high pair = candidate for the revive list, not queued.

## 🔒 RATE STANDARD FINAL (author, Jul-26 late): λ6=LOW, λ12=MODERATE, λ24=HIGH, λ30=PRESSURE
(four-point standard, all grid-anchored incl. v1 Table-4; λ30 "pressure" = past saturation
onset, carries the standing +34.7% cara-path flag as footnote). Burst dimension: gamma at
23/27/30. Flip = {6,12,24,30} per system (main {12,18,24} banked + supplement {6,30}; 18 =
bonus). Gemma G5 = {6,12,24,30} × 8 systems = 32 cells. P4-online = 4 rates.

## 📤 POST-REBUTTAL TASK (author, Jul-26): publish the Gemma family dataset to HF
After local campaign packing + organization completes: upload the Gemma-3 generations +
judge scores (the G1/G2 corpus) to https://huggingface.co/datasets/asdwb/route_balance_model_estimator
— for paper rewriting, future reruns, open-sourcing. RAILS (per the HF-publish memory):
(1) KEEP the full judged dataset permanently (part of the preservation KEEP list from tonight —
gemma_family/ outputs auto-sync into the working tree + v1-pack); (2) upload = MANUAL-APPROVAL
flow only (auto-mode OFF — agent local→public uploads need the author's explicit approval step);
(3) ZERO codename in any field/filename/README before upload (same scrub standard as the
existing dataset); (4) format consistent with the existing train/test layout on that repo.
Sequencing: experiments → draft → backup/packing → THEN this.

## 🔒 AUTHOR REORDER + JUDGING-ENV RULE (Jul-26 ~19:50Z)
1. **P4-ONLINE BEFORE GEMMA** (all Qwen cells finish first; Gemma owns the whole day):
   night order = cal-test + gamma-low (~00:15Z, Qwen stack already serving) → P4-ONLINE sweep
   (~00:45-04:45: {0,25,75,50}% pkl-row-subsampled indexes [lineage-consistent with serving] +
   LOO rebuilt from the source-mapped flat corpus with its own 100%-rebuild reference [pkl rows
   carry no source field — disclosed]; per-index audit labels assert each index's md5) → Gemma:
   prefetch/deploy ~04:45 → generation ~05:45-09:00 → parallel judging → gate/probes/KNN/BR4 →
   G5 32 cells → ALL done ~17:00Z (tight; Gemma-fallback trim = G5 rates/subset).
   Gemma watcher retargeted to the _p4_online.done marker (verified rearmed).
2. **JUDGING IN ISOLATED uv ENVIRONMENTS** (author): judge vLLM servers + scorer deps in fresh
   uv venvs (user-level uv, zero system/fork-env touch) on both A100 nodes + the dev-gpu shard
   worker.

## HF-upload schema addendum (author Q&A Jul-26 late)
- ⚠ DOCUMENTED-VS-DEPLOYED FLAG (k-item class): the published dataset README calls
  deepeval-llama3.1-8b-it_reference "the routing quality of record" — but the DEPLOYED routing
  basis was the plain 1-10 rubric judge (pkl fractions = (s-1)/9; rubric scores ABSENT from the
  published schema). Correct README wording at the upload pass (author decides phrasing;
  coordinate with CR corrections; deepeval = evaluation-of-record stays true).
- Gemma upload = FULL published schema: generation fields direct; similarity_score anchored to
  the 27B response (top-tier mirror of "cosine to 72B"); both judge keys (deepeval + rubric —
  new rubric key optionally backfilled for Qwen rows if the Mar rubric-scored file survives —
  check during packing); protectai safety pass; reference_similarity/reference_score via the
  era scripts (all present in offline_training/). Tagging = CPU-light VM post-processing.

# ═══ PLAN REVISION v5 (author order, Jul-27 ~00:50Z — supersedes conflicting items above) ═══
FALLBACK AFFAIR: all router fallback paths REMOVED from code (routes-or-raises, verified zero
remain). RERUN SETS: AP×36 + BR4-ARGMAX×8 (proof-of-clean rerun) + flip trained-router column
REPLACED by argmax×3disp×4rates=12 (flipped-topo redeploy). BR4 t>0 (30) = DEAD/withdrawn
(config raises now). G5 REVISED: Gemma-BR4-ARGMAX column (never t0.5). Unaffected-verified:
PT (no router path), 4-arm/RB/P4 (cara path — no reachable model-fallback; filter:none makes
the relaxation branch structurally unreachable; estimator-missing = loud 503). LADDER on
release: P4-remaining → BR4-argmax-rerun → Gemma → cut-line 15:30Z → AP → flip column.
CLUSTER FROZEN until the author's explicit release order.

# ═══ Jul-27 FINAL RULINGS: see EXECUTION_HANDBOOK "BINDING RULINGS 2026-07-27" block ═══
# (RB one-preset rule · v1exact basis map · normal latency treatment + appendix framing ·
#  variance-class gates · pw+exp2ss532 overnight schedule). Handbook remains authoritative.
