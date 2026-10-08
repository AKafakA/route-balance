# REBUTTAL CLOSEGATE — S2 production window → NeurIPS author-posting close

> **🏁 STATUS 2026-08-04 12:42Z: CAMPAIGN FULLY CLOSED.** Lease released ~10:12Z; OpenReview
> posting closed; deliverable done + audited; results backed up (`v3_pack_2026-08-04.tar.zst`
> 3.8G on Drive; v1 basis on Drive); repo restructured (Block standalone, v1-exp deleted).
> **Nothing live to operate — the hourly loop now only re-confirms a static state; consider
> retiring it (CronDelete).** Cold-resume doc: `2026-08-04_V3_CLOSEOUT_HANDOFF.md`. Open items
> are author-side only: CR "evaluator+cap" note + the aggregator-path check.

**Created 2026-08-02 23:45Z. This is the single operating document for closing the rebuttal.**
The hourly loop reads it; the goal references it; every decision it defers is listed in §7.

| event | UTC | BST |
|---|---|---|
| S2 window opens (13-host lease asdwb-312083) | Aug 03 00:00 | 01:00 |
| **HARD: full sync + verification BEGINS** (author 2026-08-03 17:30Z — moved earlier) | **Aug 04 09:30** | **10:30** |
| no new cells after (a λ6-length cell must finish by 09:30) | Aug 04 ~09:15 | ~10:15 |
| lease RELEASED by CloudLab | Aug 04 10:12 | 11:12 |
| **author posting closes (HARD)** | **Aug 04 11:59** | **12:59** |

**BR sq/po2 rerun + POST3 decision**: see `2026-08-03_BR_RERUN_AND_POST3_PLAN.md` — blocked
on the author ruling the dispatcher blind policy (A era-parity vs B random); ≤5%-vs-POST3
rule decides the POST3 edit. 

**Post-grid buffer order (author 2026-08-03 17:30Z): (1) the sq/po2 revisit FIRST, highest
priority — the pipeline-parity rerun block (~35 min) is PREPPED and applies on the author's
word; (2) fill cells for any NOT-DOMINATED rate; then (3) 09:30Z sync, sharp.**

**The 09:30Z sync start is absolute**: cells not synced by release are gone. The loop plans
backwards from it — no cell may START if it cannot finish AND sync before 10:00Z; the final
sync + verification (§5) begins at 10:00Z sharp, not after the last cell.

Posting rails (standing): inline tables only, **no links**; never re-assert the two
basis-dependent claims; state the measurement and stop (no volunteered caveats); Cara is an
internal codename, never public.

---

## 1. Status at doc creation

* **Machinery: no untested piece.** Driver 11 phases, fleet-subset boosters, Phase E′
  verdict+refusal, Q1 pilot refusal, burst runner both presets (refusal AND sweep paths),
  per-(pattern,preset) baselines, burst analysis, monitor clone-collapse, per-cell provenance.
* Gate self-test **153 cases / 84 MUST-FAIL / 0 wrong**; static gate 0 new findings.
* **0 INVALID cells** across all rehearsals; all artifacts two-location (VPS + git).
* Rebuttal already posted 2026-07-29 (`rebuttal-submitted-2026-07-29`); this window's output
  feeds the **final author post** (Q1 4-tier numbers, burst rows, grid confirmation) and the
  CR list (`2026-08-02_CR_CORRECTION_LIST.md`) is maintained separately.
* Monitor running (fixed classifier), publishes to head `~/S2_CONTAM_STATUS`; self-heals onto
  the new lease at P0_REACH because the freshness probe targets the *new* head and a stale
  status forces a restart with the new `hosts_all`.

## 1a. CURRENT status (2026-08-04 08:42Z) — live, supersedes §1 creation snapshot

* **Grid deliverable DONE + two-location:** RB **strictly dominates AP at λ18/24/30** (3 rates);
  **λ6/λ12 near-parity** (RB (0.6,0.2,0.2): quality within ~1.5% of AP-pw0.70, wins cost/mean/p99)
  — verified 2026-08-04 decode-basis; matches posted frontier framing; corrected from the earlier
  "4 rates dominated" shorthand (λ12 is near-parity, not dominated). Grid on VPS `s2_cluster/cells/`.
* **Q1 4-tier POSTED** by author (both presets, sched-overhead column, posted-series 0.51
  quality basis; scoring-basis exposure deferred to paper/CR per author).
* **BR fill COMPLETE** (see §3b): 13 B + 1 era-parity A banked, two-location (λ24 full 8/8,
  λ18 partial 5/8; cutoff before λ30). **A/B on POST3 t0.3/po2/λ24: dispatch fix +3.1% (<5%,
  immaterial); fresh +10.5% = collapse-draw variance. POST3 37426 STANDS** (author decision
  queued, non-blocking). Doc `2026-08-04_BRFILL_AB_RESULT.md`.
* **λ6 eff-0.70 retune RUNNING** (author-picked exploratory, after-sync): cara mode, weights
  applied; dominance-vs-AP verdict imminent. λ<12 pre-registered off → disclosure only if it
  enters the paper.
* **🟡 disk >90%:** VPS 95%/6.4G, head root 92%/5.1G (cells on /mydata 807G). Deliverable is
  two-location, so closegate pack runs POST-release from the VPS copy — not blocking the lease.
* **INVALID = 24**, ALL the BR sq/po2 fail-loud collapse class (λ18/24/30) — accounted; the
  `_BRFILL` dirs carry the absorb-completed versions. No new INVALID.
* Gate self-test **162 / 89 MUST-FAIL / 0 wrong** (was 153/84 at creation; +9 cases, no
  MUST-FAIL drop). Monitor fresh, 0 findings.
* **Disk WATCH:** VPS root 95% (6.8G free), head root 92% (5.1G free); cells on /mydata (807G).
  Verify headroom before the §5 pack.
* **Rate-6 retune** (author-requested `0.66/0.17/0.17` + `0.7/0.3/0.3`): REGISTERED for NEXT
  lease — λ6 fill is pre-registered off and cannot finish before sync. Note: `0.7/0.3/0.3` is
  argmax-scale-invariant → effective w_q 0.538 (lower bracket), NOT above 0.66.

## 2. 4-tier safety recheck — every 2-tier-era change, production effect, evidence

**Verdict: no 2-tier change alters what a production 4-tier cell measures.** Two changes cost
schedule (flagged → §7); none touch semantics.

| change (this session) | production 4-tier effect | evidence |
|---|---|---|
| `TPOT_SRC` overridable; stage-copy writes `deploy_tpot/` explicitly | none — default source unchanged; copy **byte-identical** on the real 4-tier set (tree md5 `cc97ef44…`) | differential copy test; P2B+G-TIERS 4-tier simulation PASS |
| `TPOT_SUBSET=1` subset generator | inert — only `s2_smoke.sh` exports it; driver **dies** on it outside rehearsal | `grep` containment; MUST-FAIL self-test |
| `dominance.py` sidecar exclusion | **required** for production — production cells carry sidecars too; only `.provenance.json` excluded, stray files still fail the rate | 20/20 parse on the failing dir; 2 self-test cases |
| `dominance.py` `MetricUnavailable` + all-co-maximizers | fail-closed offline analysis, fleet-agnostic | self-test MUST-FAIL cases |
| monitor ppid + clone-collapse + argv findings | fleet-size-agnostic; collapse also absorbs python-multiprocessing bench children; **without it ~every production sweep refuses the next cell** | head-node fork repro; 4 self-test cases; live sweep counted `runner:1, 0 findings` |
| burst runner baselines per (pattern, preset) + SKIP-PRESET | correct on 4-tier: if 532 unexpectedly refuses there, uniform still runs and the refusal is loud in `PRESET_SKIPS.txt` — that refusal would itself be a finding for the author, not a script decision | mini-smoke exercised both paths live |
| `burst_analysis.py` preset regex + sidecar exclusion | old preset-less names still parse (`preset=None`); new files carry `_p` | regex test all three shapes |
| `s2_q1_burst.sh` PAY_* fix | the block never ran before; now it runs — **adds cells** (see §7.2) | mini-smoke end-to-end |
| gate_selftest +14 cases | test-only | 153/84/0 |

## 3. Launch recipes (the loop executes these; commands are exact)

**PINNED (2026-08-03 00:30Z): the production rspec is `rspec_asdwb-312083.xml` BY NAME** —
"newest file" selection is retired, because the AUX lease's file is newer and a glob would
mis-pick a 4-node fleet for production.

**AUX FLEET (lease asdwb-312082): 4× d7525 A30** — `s2_cluster/hosts_aux`
(10s10315/317/325/339, re-carved from the old smoke lease). Purpose per author: offline
analysis / extra computing / spare pool. **Never in `hosts_all`, never part of the experiment
fleet, never receives cells.** Separate lease and NICs, so using it does not violate the
no-side-traffic rule even during live cells. Unprovisioned until first use (R2: setup script
only, on demand). Any promotion of an aux node into the experiment fleet is an author
decision, never automatic.

**Precondition — TWO conditions, both required (author 2026-08-02 23:55Z):** (1) the
author's **explicit GO in chat** — the rspec landing alone does NOT authorize a launch; the
loop may dry-run against a new rspec but must not launch production without the word. (2) the
author saves the new lease rspec as
`window_setup_work/s2_cluster/rspec_*.xml`. **An rspec counts as NEW only if its mtime is
later than this document's creation (2026-08-02T23:45Z)** — `rspec_asdwb-311959.xml` already
present is the OLD lease and must never drive a production launch. `make_s2_configs.py` reads
the **public** ipv4 from it — never LAN 10.10.1.x.

```bash
cd ~/Code/llm/Block/cara_paper/handoffs/window_setup_work

# (a) dry-run first, always
RSPEC=s2_cluster/rspec_<NEW>.xml ALLOWED_HW=a30,v100,a100 S2_MODE=production \
  ./v3_scripts/s2_driver.sh --dry-run

# (b) production driver: P0..P4 gates, then P5_Q1 PILOT-SCALE, then P6_GRID (5 rates + Phase E')
RSPEC=s2_cluster/rspec_<NEW>.xml ALLOWED_HW=a30,v100,a100 S2_MODE=production SEEDS=1 \
  ./v3_scripts/s2_driver.sh
#   SEEDS=1 scopes ONLY the driver's P5 block (4 pilot cells, ~20 min): the full Q1 lives in
#   the burst runner below, per the runbook's 144-cell budget — SEEDS=1 implements that
#   budget rather than changing it. Confirm in §7.1.

# (c) full Q1 (144 cells, both presets, ~9.5 h + baselines) — standalone, after P4 gates green:
#     launcher-file pattern, run on the head node (loop generates _run_qb_prod.sh the same way
#     as _run_qb_mini.sh, with: )
BURST_FLEET=4tier ALLOWED_HW=a30,v100,a100 SEEDS='1 2 3 4 5 6' \
  BURST_LAMBDAS='30 24 12' PRESETS='532 uniform' N_PROMPTS=3534 bash s2_q1_burst.sh

# (d) after the grid: RB retuning fill points via dominance.py --suggest, each stamped fill=1
```

Sequencing default (loop follows unless author reorders): **driver (b) through P4 → Q1 (c) →
grid resumes/continues (b keeps state; P6 runs per rate with Phase E′) → RB retuning (d)**.
532 preset runs before uniform so a truncated night still yields the posted-config rows.

## 3b. ENDGAME OUTCOME (CORRECTED 2026-08-04 08:42Z) — B fill ROOT-CAUSED + RUNNING

**SUPERSEDES the 07:31Z "infeasible" note (a MISDIAGNOSIS — I blamed aiohttp fd exhaustion).**
The canary's 3174/3534 EMPTY-model failures were the campaign FAIL-LOUD stats gate
(`_get_instance_stats` raises `InstanceStateUnavailable`): sq/po2 fetch per-request stats and
hit it; rr fetches none, so it "worked". Two fixes, both INERT for grid/Q1:
  1. `_get_instance_stats` MASKS under `stats_blind_absorb` (returns {}) = era/POST3 behavior
     for the fill only; grid/Q1 keep fail-loud (commit 9af80f2; self-test green 162/89/0).
  2. head root `/` hit 100% and truncated the write → fill output redirected to `/mydata`
     (807G) after a verified q1burst clear (300/300 on VPS first).
Canary then completed **3534/3534**. Fill running, grid-poolable names
(`br4_t<t>_<sq|po2>_<A|B>_l<lam>`): B sweep = 4 thresholds × 2 dispatchers × λ24/18/30, **λ24
first** (the comparison rate). A/B for POST3's t0.3/po2/λ24 (posted p99 **37426**): B=random
banked; A=counter fired one-shot (`a_cell.sh`) after the fill stops (scheduler stays up), so
both draws come from identical conditions → the ≤5%-vs-POST3 rule is judged on a clean pair,
not fresh-B-vs-stale-post. **Deliverable UNAFFECTED regardless: grid dominance (AP strictly
dominated λ18/24/30; λ6/λ12 near-parity) + Q1 posted.** `2026-08-04_BRFILL_CANARY_FINDING.md` is likewise SUPERSEDED (header-marked).

## 4. What feeds the final post

1. **Q1 4-tier table** — 532 rows (against the posted text) + uniform (robustness); paired
   burst degradation ratios with t(5) CIs from 6 seeds.
2. **Grid + Phase E′** — dominance verdicts per rate on the pre-registered baselines' own best
   knobs; fill points only stamped `fill=1`.
3. Numbers enter the post only from cells that passed every gate; **a SKIP/refusal is reported
   as such, never papered over.**

## 5. Data closegate (before lease release — non-negotiable)

1. Incremental pulls run all window (the driver already pulls per rate; the loop re-syncs
   `experiment_output/` including Q1/burst + logs + decision slices every pass).
2. End of window: full `rsync` of `~/Block/experiment_output/` from the head; verify counts
   vs `expected_cells.py`; **0 INVALID or each INVALID accounted**.
3. Commit all cells + analysis to the Block repo (LOCAL git), then pack:
   `rb-exp/closegate.sh <pack> <campaign>` — one tar.zst, `zstd -t` verify, then Drive.
4. Only then may the lease be released. **Deployed-model preservation rule applies** (≥2
   locations + manifest) for anything not already mirrored.

## 6. Issue playbook (what the loop may do alone vs what STOPS)

**Auto-proceed (documented paths only):** resume a crashed/ssh-dropped driver with
`--from <phase>` when the halt was infrastructure-transient AND the phase's gates passed;
restart the monitor; kill orphans via the driver's own pre-launch clear; re-run a pull/analysis.

**STOP + report at the top of the pass (no workaround):** any gate FAIL / INVALID cell /
SKIP-PRESET on 4-tier / SHORT rate / dominance refusal / anything contradicting a posted
claim / any fix that would change an experiment parameter (R3/R7) / OOM (never retried with
changed params). A stopped phase stays stopped until the author speaks.

## 7. Decisions deferred to the author

**RESOLVED 2026-08-03 02:10Z — zero-call static arm.** The author directed mid-window that
static cells make zero XGBoost calls (apples-to-apples E2E). Implemented (`9bbd04a`), gated
per cell (ASSERT-5), pilot re-run and passed on the new arm: static 0.044 ms/batch, calls
delta 0, constants exact per tier, routing identical to the old implementation, arms still
diverge. Old-impl pilot cells retired+preserved; the Q1 table comes from the burst (all
new-impl). §7.1 SEEDS=1 is thereby also settled (the driver ran pilot-scale).

1. **Confirm `SEEDS=1` on the driver's P5 block** (pilot-scale; full Q1 in the burst runner).
   Default without it: P5 runs 24 Poisson cells (~2.1 h) that duplicate burst-532-pois rows.
2. **Baseline cost:** per-(pattern,preset) baselines are 12 cells @ λ30/n3534 ≈ **26 min**
   (was ~13 min before the rebuild) against the window's ~36 min slack. Accept (correct,
   recommended) or trim to per-pattern (repeats the imported-floor risk).
3. **If 532 refuses divergence on the 4-tier pilot** (would contradict expectations): uniform
   becomes the headline Q1 basis and the 532 refusal is itself reported. Pre-authorize or hold?
4. RB retuning time-box after the grid (~1.3 h in plan).

## 8. Goal prompts (short forms to paste)

* `finish the S2 production window per cara_paper/handoffs/2026-08-03_REBUTTAL_CLOSEGATE.md and close the NeurIPS rebuttal before Aug 04 11:59Z; gates decide, no experiment-parameter changes without me`
* `run the rebuttal closegate: cara_paper/handoffs/2026-08-03_REBUTTAL_CLOSEGATE.md §5 data preservation then lease release`
* `draft the final author post from the S2 results per CLOSEGATE §4 rails`
