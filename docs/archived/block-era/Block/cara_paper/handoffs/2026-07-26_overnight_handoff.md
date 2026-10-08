# Overnight handoff — Jul 25→26 (rebuttal window, night 1)
Rolling detail: `2026-07-25_window_execution_log.md` (40+ timestamped entries). This doc = consolidation.

## Where things stand (as of ~02:55Z)
- **R1 REPRODUCTION GATE: PASSED near-exactly** — arm-1/uniform/λ12 on the v1-exact May tree:
  mix Δ≤0.6pp/tier (14B 58.7 vs 59.3, 7B 36.3/36.2, 3B 4.3/3.9, 72B 0.7/0.6), tput 11.85=11.85, e2e +0.6%.
- Sampled reproduction 3/3: λ24 (0.9pp, 0.0%, 1.2%) · λ30 (1.0pp, 1.8%, **e2e +34.7% — FLAGGED**:
  saturation-adjacent capacity drift, new units; routing reproduces; within-session tables unaffected).
- **ARM SWEEP RUNNING since 02:50Z** (current tree): arms 1-4 × λ{6,8,10,12,18,24,30} = 28 cells,
  then 2 overload (square-wave 40/6), then Po2 family (Po2×7λ + SQ×7λ + RR×3λ = 17). ETA ~10:00Z.
  Every cell: pre-audit (tree hash, config 272e9903, KNN pkl b9c5c606, dataset 15ed0e67, head-md5
  aggregate 6cfd94d5, PREDICTOR LIVENESS log marker, env-vs-arm) + post-audit (3534/0, fields, mix).

## The night's incidents (all root-caused; details in the rolling log)
1. R2 violation #4 (env improvisation) → author-ordered reload; setup rebuilt as reviewed script.
2. Wrong vLLM branch all evening ("route_balance_v_11" = scrubbed doc name; real = cara_v_11@2894a9872).
3. May env reconstructed by PyPI-date archaeology (numpy 2.2.6 etc.); env snapshots now captured per node.
4. Node7 experiment-NIC dead — SAME machine/failure as May-06 close; May's rule "use public ip always"
   re-violated by my generated config → fixed (public IPs = May-faithful); portal power-cycle done.
5. **Headless-predictor incident (the big one): first A0 ran with latency heads NEVER LOADED**
   (mirror-scrubbed /users/anon path in the mapped tree + xgboost never installed) → 3B flipped to 44%;
   gate caught it; cell quarantined; both causes fixed as restorations; May xgboost 3.2.0 pinned.
   Lesson institutionalized: byte-sync ≠ function-sync → predictor-liveness gate now per-cell.
6. Author charges accepted + filed in permanent memory: May preservation promise unkept (KNN pkl lost,
   heads only June-anchored) = misrepresentation-class; search-first + generated-config parity +
   function-level verification now binding.

## Morning plan (Jul-26)
1. Verify sweep completion + per-cell audits; dual-basis aggregation (reagg_v1_zerofix) over all cells.
2. Build: pre-registered arm comparisons (1v2/2v3/1v4), Po2 table, reproduction curve (v1exact + arm-1
   × 7λ vs May twins; code-drift delta via arm-1 λ12 both trees).
3. Insert numbers into gTfM/AC/287j/faAU drafts; k-wording sweep; freeze prep.
4. Afternoon: topology split-flip (A30 5/3→3/5; 12 cells). Evening: Gemma handover (G0 probe → ladder →
   overnight generation). Preservation close-out: checksummed artifact manifest ×2 locations.

## Open flags for the author
- λ30 cross-era e2e drift (+34.7%) — caveat wording needed if any cross-era latency claim at high λ.
- Fork branches on public GitHub are codename-named (cara_v_11 …) — your call, no action taken.
- setup_env.sh still hardcodes the expired HF token (nodes neutralized; repo file untouched).

## SWEEP COMPLETE (07:35:24Z) — 50/50 cells, 0 invalid
**Po2 family (fixed basis, q/e2e-ms/tput):** Po2-Queue sits exactly where theory puts it — beats
round-robin on latency by ~16% at equal quality (λ12: 2248 vs 2668ms), loses to full-scan
shortest-queue by ~8-9% (2067ms) — the classic O(1)-sample vs O(n)-scan tradeoff, now measured
in-session on the identical pipeline path as May's baselines.
**The AC answer in one line:** the full system (arm-1: 0.3699/1511ms at λ12) beats EVERY
general-purpose dispatcher — including the requested Po2 — on BOTH axes simultaneously
(+0.006..+0.010 quality AND 27-33% lower latency): content/quality-aware model selection with
latency fusion is not replicable by dispatch policy alone.
**Four-arm recap:** 1v2 clean separation (formulation load-bearing); 2v3 wash except near
saturation; 1v4 static prior recovers most benefit on stationary load, live signal wins under
burst — topology flip (today PM) is the designed static-prior stress test.
**Reproduction (the night's foundation):** 3/3 sampled May cells: mix ≤1pp, tput ≤1.8%,
quality Δ≤0.0013 (corrected basis), e2e ≤1.2% below saturation (λ30 +34.7% flagged, env-drift).
## Next: morning tables→drafts (now), topology flip (early PM), Gemma handover (evening).
