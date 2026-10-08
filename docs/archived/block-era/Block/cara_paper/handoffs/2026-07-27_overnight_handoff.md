# Overnight handoff — Jul 26 → Jul 27 (consolidated; details in the execution log + LOCKED_EXPERIMENT_PLAN)

**Gate**: all first-rebuttal DATA by EOD Jul-27 BST; draft text midnight+morning Jul-28; author
submits ≤ Jul-28 13:00. Binding docs: `2026-07-26_LOCKED_EXPERIMENT_PLAN.md` (+ later author
chat decisions), actions-table for wording, execution log for the audit trail.

## Both PC keys: EVIDENCE COMPLETE (Qwen side)
- **Po2/dispatcher**: base matrix 16/16 + 49-cell variant space, 0 fails. Table-4 design locked
  (append Po2 rows; v1 rows untouched; e2e+mix columns carry the argument). ⛔ RB-vs-BR-t05
  quality ordering = basis-dependent → prohibited both directions. Key finding: BR-t0.5 = 98%
  single-tier concentration; RB ties its quality at −27% e2e @λ24 + far cheaper mix.
  **AP QUARANTINED** (mix inversion vs May full grid: 100% 72B now vs 61/37/2 then; 35 cells
  excluded from tables; root-cause = post-draft investigation).
- **XGBoost/four-arm**: stationary 4 arms × 7λ (parity ≤λ24; λ30 = quality-trade row, carries
  the +34.7% flag); GAMMA LADDER {6,12 (running tonight), 23, 27, 30}: parity below capacity,
  **learned −9.5%/−10.3% e2e at 27/30** (the rails-clean burst win); offline analysis (static
  error 4→15.8 ms/tok; calibration robustness — no single point escapes; discovery-first
  framing, "moderate λ12, balanced" — never "light"); paired quality: 13/13 positive, CI
  excludes 0; cal-test tonight (cal@24/cal@30 statics under gamma-high) closes "any static
  always loses under bursty pressure" online.

## Tonight's armed chain (all unattended, marker-gated)
flip main (18/24 @19:08Z, 0 INVALID; overlap-window cells CLEARED by author's acceptance test)
→ flip {6,30} supplement (16 cells; completes the 4-point rate standard: 6=low/12=moderate/
24=high/30=pressure) → redeploy ORIGINAL topology (health-gated) → cal-test (4) + gamma-low (8)
→ **Gemma pipeline**: prefetch → ladder deploy (15 backends; 27b-pt = TP2×2/node via
V2-instances config) → 15/15 health gate → broadcast generation (18,549 prompts × 4 tiers,
temp0/freq1.2 defaults, max-out 1024, fp16; T0 stamped for cold-start wall-clock).

## Tomorrow (Jul-27) plan
1. Judge (llama-3.1-8b, v1 judge of record) → G2.5 ladder-separation gate → static-prior
   probes (λ12-equivalent per-instance load) + KNN build + **train Gemma BR4-t0.5** (era
   wrapper; wall-clock → cold-start answer).
2. **G5 family matrix = 32 cells** (same 8-system shape as the flip × λ{6,12,24,30}).
3. **P4-online cold-start sweep** ({0,25,50,75}% index × 2 presets × 4 rates + 1-2
   leave-one-source-out; conditional on Gemma timing; offline P4 = backup, parked on dev-gpu).
4. Tables → 4 reply drafts (strategy: technical reviewer → 5 via suggested-and-applied;
   clarity 3-scorer → 4 via commitments incl. the continuous-batching disambiguation).
5. THEN (author ordering): backup everything → VPS clean (v1/v2 packs only, approvals + Drive
   upload + devgpu verify first).

## Standing rules added today (memory-backed)
No bulk transfers during live phases (node0 NEVER a relay; incremental result syncs = required
protocol); no-overwrite gates on all movement (Astrolabe lesson; sync_and_pack has mechanical
gates + pack inode-verified clean); stage_b DROPPED (revive-only); pairing rule (RB both presets
everywhere); rate standard 6/12/24/30; APC-May = verify-before-answering; gamma-only burst
evidence (square = internal).

## Open items
AP root-cause (post-draft) · λ30 cara-path flag (footnote) · APC-May archive check ·
Gemma-4 spike artifacts on dev-gpu (documentation) · preservation archives (staged, post-draft)
· node0 /mydata leftovers (die with lease; noted for close-out).

**Post-rebuttal addition (Jul-26 late)**: Gemma judged corpus → HF dataset
(asdwb/route_balance_model_estimator) after packing; manual-approval upload flow, codename-zero
scrub; corpus = permanent KEEP.
