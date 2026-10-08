# V1EXACT REPAIR PLAN — full cell audit vs the revision plan (2026-07-27, author-ordered)

STANDARD (author): (1) every number a reviewer can compare with the paper must be produced by
the paper's own code (Block_v1exact tree, manifest-locked); (2) no planned cell dropped/reduced.
RULE: no mixed tables — a table is either all-v1exact (paper-facing) or all-current (internal
within-window comparison); every table's verification line names its tree basis.

## Prerequisite ports into Block_v1exact (disclosed, inert-by-default; then A0 RE-GATE ≤1pp)
- CARA_STATIC_LAT hunk (cara_serve ~1476-1499; env-gated — env unset ⇒ byte-identical): for arm4
- dispatch/po2_queue.py + 2-line factory registration (unreachable unless payload selects it)
- regenerate TREE_MANIFEST.md5 → update cell_audit.sh EXP value (logged as v1exact-v2)
- A0 gate cell re-run: must reproduce May ≤1pp mix / ≤5% tput+e2e before ANY repair cell counts

## FULL AUDIT TABLE — every planned family

| # | family | planned | banked (basis) | paper-facing rows? | v1exact REPAIR cells | disposition |
|---|--------|---------|----------------|--------------------|----------------------|-------------|
| 1 | A0/R1 + λ24/30 retests | 4 | 4 (v1exact) ✅ | bridge certification | 0 (+1 re-gate post-port) | COMPLIANT |
| 2 | Weight grid (Table-4 suppl) | 17 | 17 (v1exact) ✅ | YES | 0 | COMPLIANT |
| 3 | Four-arm stationary 28 | 28 | 28 (current) | arm1/arm4 rows land beside paper uniform rows | **14** = arm1+arm4 × uniform × λ{6,8,10,12,18,24,30} | banked 28 → INTERNAL 4-arm table (current, complete); NEW v1exact bridge table |
| 4 | Overload 40/6 | 2 | 2 (current) | no paper counterpart | **2** (arm1+arm4, for a fully v1exact bridge table) | optional-but-included per no-reduction |
| 5 | Gamma/burst 37 | 37 | 37 (current) | ladder bridge rows | **10** = arm1+arm4 × γ0.25 × r{6,12,23,27,30} | banked 37 → internal table; v1exact ladder bridge |
| 6 | PT dispatcher rows | 17 | 17 (current) | YES (paper has dispatcher-only rows) | **17** = sq×7λ, rr×3λ, po2×7λ | current 17 → corroboration |
| 7 | BR4 extension | 8 | 8 (current) | YES (Table-4/Fig-2 extension) | **8** = br4_po2×7λ + br4_sq anchor | current 8 → corroboration |
| 8 | AP rerun | 36 | 15/36 (current, RUNNING) | YES (Fig-2 AP extension) | **36** = pw0.7 po2×7 + sq + pw{025,039,053,08}×po2×7 | current run completes → corrected-code corroboration set (author call: complete vs halt) |
| 9 | P4-online | 32 | 7 + 25 pending (current) | no — self-contained ablation | 0 | proceed current, unchanged |
| 10 | EXP2 holdout | 8 | 0 (armed, current) | no — self-contained | 0 | proceed current, unchanged |
| 11 | Flip + argmax column | 40 + 12 | 40 (current) | no — flip vs original, both window | 0 | column proceeds current, unchanged |
| 12 | Gemma family | pipeline | 0 | no — own family | 0 | proceed per pipeline plan |

## REPAIR TOTALS
- v1exact repair cells: 14 + 2 + 10 + 17 + 8 + 36 = **87 cells** (+1 A0 re-gate)
- ports: 2 hunks + manifest regen + audit-gate update
- gates per cell: RULE-0 pre-audit (v1exact mode, new manifest) · mix post-audit ·
  first-cell twins per router family (AP vs May 50/28/20/1; BR4 vs 45/40/10/6; PT vs era refs)
- AP-v1exact safety (era code carries fallback machinery BY DESIGN — faithful): CVD="" enforced,
  loud import probe pre-launch, per-cell degenerate-mix halt, first-cell twin gate

## WALL-CLOCK (measured pace ~7.5 min/cell avg incl. restarts)
- ports + A0 re-gate: ~1.0h
- 87 repair cells: ~11.0h
- v1exact program TOTAL: **~12h** (start T → T+12h)
- remaining current-tree planned work after it: P4 25 (~2.7h) + EXP2 8 (~0.9h) +
  flip column 12 (~3h incl. redeploys) = ~6.6h
- Gemma pipeline: ~11h

## THE ARITHMETIC (honest, no recommendation hidden in it)
12h (v1exact) + 6.6h (current remaining) + 11h (Gemma) = **~29.6h serial**
Time to submission gate (Jul-28 11:59 BST) from 05:00Z Jul-27: **~30h**
⇒ zero slack; any hiccup pushes Gemma tables past the gate. Fallbacks that preserve
"no cell dropped": Gemma tables posted in the discussion window (≤Aug-03, allowed) and/or
part of the v1exact PT/pw-sweep breadth runs in the held-live window Jul-28-31 (cells still
produced, only later). Sequencing choice + any timing fallback = AUTHOR DECISION.
