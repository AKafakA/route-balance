# BR collapse cells — rerun plan + POST3 update decision (author-ordered 2026-08-03 ~19:50Z)

## The comparison already visible (today's fail-loud grid, valid rr survivors)

BR-rr banks at every rate; BR-sq/po2 go INVALID at λ≥18 (the storm kills the scheduler). But
rr and sq/po2 measure the SAME collapse — the era proved all three complete at ~equal means
when telemetry is absorbed rather than raised. Today's banked rr numbers:

| λ | BR-rr mean (ms) | BR-rr p99 (ms) | vs POST3's posted BR-Po2 |
|---|---|---|---|
| 18 | 14,207–18,109 | 62k–116k | — (POST3 reports λ12/λ24) |
| 24 | 35,994–42,183 | 128k–134k | POST3 BR-Po2 λ24 = **48,182 ms** (Table 1) / mean 43,289 (Table 2) |
| 30 | 47,377–56,032 | 112k–122k | consistent collapse magnitude |

**The posted POST3 number (48,182 ms at λ24) sits INSIDE today's fail-loud rr band
(35,994–42,183 for the four thresholds, closest tuple ~42k).** Same phenomenon, same order of
magnitude. The collapse finding is not in question; only whether the exact posted digits move.

## Rerun plan — with the FIXED (paper-basis) semantics

**Purpose**: fill the BR sq/po2 tuples at λ≥18 on the same measurement basis as POST3/the
paper, so the grid's BR row is complete and internally consistent, and so POST3's Po2 digit
can be checked against a fresh Po2 cell.

**Blocked on the unruled decision** (per the no-self-decided-fallback principle): the rerun
needs a dispatcher-side blind policy, which the author has NOT yet set. Options to rule:
  (A) era-parity: counter fallback (the running code) — reproduces POST3/paper basis exactly.
  (B) random-on-blind — the corrected design, a NEW basis (slightly better, no hotspot).
**Recommendation: (A) for the rerun** (series-consistency is the whole point); ship (B) as the
permanent fix AFTER the window with a CR note. Author sets this before the rerun runs.

**Mechanics** (ready once the policy is ruled): `stats_blind_absorb=1` + the ruled dispatcher
policy, on the 24 BR sq/po2 cells (λ18/24/30), each stamped `parity=1` in provenance; ~50 min;
endgame head slot; RB path and fail-loud untouched elsewhere.

## POST3 update decision — DEFERRED to the data

Rule: after the rerun, compare fresh BR-Po2 λ24 mean vs POST3's 48,182 / 43,289.
  * within ≤5% → **keep POST3 as posted** (author's ≤5% rule, already set for W3).
  * >5% → replace POST3 Table 1 (and note Table 2 if it moves); the collapse conclusion is
    unchanged either way — only the digit.

**Table 2 (W1-topology)** needs a redeployment (5/3 replicas) → OUT unless time miraculously
allows, per the author.

## Status
* No serving-code change reaches the fleet until the endgame sync; rates 18/12/6 finish on the
  running (reverted) code.
* Both UNALLOWABLE bugs + the two basic principles documented (GATES.md, memory).


## FINALIZED endgame execution (author "move forward" 2026-08-03 ~20:30Z)

Apparatus complete + unit-tested (default=counter=running behavior; static gate + selftest
clean). Fleet sync and launch gated to the post-rate-6 boundary.

Sequence at the boundary:
1. Grid finishes fail-loud (rate 6 last). 09:30Z sync is HARD.
2. Sync the reviewed apparatus to the fleet (P0_TREE). Default counter/absorb-off = grid
   behavior unchanged; only the rerun cells select random/absorb via config.
3. `br_fill_rerun.sh` (CUTOFF 09:10Z): **CANARY FIRST** — one B cell (t0.3/po2/λ24). The open
   question is whether absorb prevents the scheduler's Bad-fd death; if the canary does not
   COMPLETE or the scheduler dies, the script STOPS and reports — no workaround. If OK: the
   remaining 23 B cells (t0.3 first) + 1 A cell.
4. `br_fill_compare.py`: A vs B on t0.3/po2/λ24 (same seed → same arrivals). POST3's 37426 is
   an older non-reproducible draw; the fresh digit will differ and that is expected.

Cells land in `s2_p1_l<λ>_BRFILL/` with parity=A_counter|B_random stamped — never mixed into
the fail-loud grid dirs. RB retuning dropped.

**Hard stops honored:** scheduler death on the canary, absorb-not-applied, cutoff reached,
09:30 sync — each halts and reports rather than improvising.
