# CHECKPOINT — v1 rebuttal submitted, window closed (2026-07-29)

Covers Jul 25 → Jul 29. Supersedes the rolling logs for status; those remain the detail record.

**State: initial rebuttal COMPLETE — all four comments posted 29 Jul. Cluster idle, data packed.**
**Next action: none until reviewers reply. Author posting closes Aug 03 AOE.**

---

## 1. What went out

| # | to | posted UTC | thrust |
|---|---|---|---|
| 1 | 287j | 29 Jul 03:58 | two-layer scheduling, adaptive batching formula, KV/TTFT scope, revision commitments |
| 2 | gTfM | 29 Jul 05:30 | calibrated-static comparator; learned head buys **cost only** |
| 3 | faAU | 29 Jul 08:26 | Po2 dispatchers, topology, Gemma-3 family, label/prompt-blind ladder, onboarding cost |
| 4 | AC | 29 Jul | summary against the meta-review's two priorities |

Verbatim text and known defects: `rb_nips_rebutal/POSTED_2026-07-29.md`.
Reviews were 3 / 4 / 4. Meta-review asked for (a) design-choice justification, notably the
XGBoost estimator, and (b) general-purpose dispatcher comparisons.

## 2. The claim that changed, and why

The packet went through eight versions. The substantive movement was on the estimator.

- **v1–v3** claimed the learned head was ~8% *faster* than a calibrated static prior at λ24/28.
- The **time-matched control** (`v1exact_timematch`, 18 cells, both arms back-to-back) showed
  that margin was a cross-session artefact. Measured paired: λ12 +0.8%, λ24 −1.1%, λ28 +2.1%.
- **v4 onward** claims cost only: **8.1–9.8% cheaper at indistinguishable quality**, and states
  the scope is narrower than the submission implies.

A third arm removing the latency term entirely is what keeps the component justified: routing
goes load-blind, 69.1% to the smallest tier, quality 0.4899 against 0.504–0.510.

**Dropped and not restored:** the burst axis (same cross-session confound, never re-measured),
λ=30 (above sustained capacity), and the 25%/75% label rungs (both score *above* the full
index, unexplained).

## 3. Findings that constrain the camera-ready

Recorded in `rb_nips_rebutal/POST_REBUTTAL_CR_NOTES.md`. Summary:

- **CR-1** evaluator drops 0.0 scores (`or`-chain) — coverage ~76%, quality 0.507 vs 0.385
  zero-safe. `aggregate_overnight_grid.py` still carries the bug; `aggregate_v2_grid.py` is fixed.
- **CR-2** rebuttal cost prices input+output; the submission prices output only.
- **CR-3** KNN served k=10 over 14,966 rows; the paper says k=50 / 14,919.
- **CR-4** manuscript wording implies TTFT is modelled; it is not.
- **CR-5** Vidur citation promised to gTfM — verify the characterisation before writing it.

**Not written down anywhere, by author decision:** routing optimises the Qwen2.5-7B-Instruct
judge while every reported quality figure is deepeval; the two correlate at r = 0.22 over
59,636 paired scores. This explains why the label ladder is flat and why the 25% rung looks
anomalous on deepeval but not on the routing judge (+0.001). Kept out of the CR notes at the
author's instruction; it lives only in this session's transcript.

## 4. Data at the checkpoint

Everything moved out of `Block/` into the packs. **Block now holds code, documents, logs and
aggregates only** — 126 MB under `experiment_output/`, down from 7.8 GB.

| location | contents |
|---|---|
| `rb-exp/v1-exp/rebuttal_jul25/jul25_stage_a/` | 337 active cells, 1.8 GB gzipped |
| `rb-exp/v1-exp/rebuttal_jul25/archive/` | 210 uncited cells, 1.2 GB gzipped |
| `rb-exp/v2-exp/experiment_output/` | v2-era smoke and sweep, moved out of Block |

Docs in the pack: `CELL_MAP.md` (what each directory backs + re-run script), `INDEX_v1exp.md`
(per-cell facts), `CLEANUP_LIST.md`, `CELL_MAP.json`.

Deleted: 20 `FALLBACK_INVALID` AvengersPro originals (degenerate 100%-72B Jul-26 runs), each
verified to have its corrected Jul-27 rerun in place first. Nothing else deleted; quarantined
cells with `INVALID` / `SCREENED_OUT` markers are kept as evidence of real bugs.

Disk: `/` went 95% → 91%, 5.9 GB → 12 GB free.

## 5. Scripts

All 107 node0 scripts are now in `window_setup_work/node0_scripts/`, including the RULE-0
machinery (`cell_audit.sh`, `deep_cell_audit.py`). Analysis scripts added this window:
`build_v1exp_index.py`, `build_cell_map.py`, `ladder_1sess_analysis.py`,
`p4_offline_bestacc.py` (**does not work** — see below).

## 6. Open items — dispositioned by the author at close

1. **AC comment — DONE.** Submitted 29 Jul; all four comments are on OpenReview.
2. **POST 3 Gemma price sentence — FIXED** on OpenReview by the author. The two remaining
   defects in that post are cosmetic and left as posted.
3. **AvengersPro pw sweep hole — accepted.** pw039 has no λ24/λ30 and pw053/pw08 were deleted
   as degenerate. Not a gap for the rebuttal, which reports AvengersPro at pw 0.7 and uses
   rates 24/28 only.
4. **`p4_offline_bestacc.py` — closed, will not be fixed.** The online results answer the same
   question; the offline replay is unnecessary. Script stays in the repo labelled non-working.
5. **Waiting on reviewer replies.** Author posting closes **Aug 03 AOE**. No action until then.
6. **The 25%-beats-full-index result stays unexplained** and stays out of the packet.
7. Gemma family follow-up was planned for the held-open window and was not run.

## 7. Process lessons from this window

- **Latency is not comparable across sessions; quality is.** The A/B control showed a 5.9%
  swing at λ28 from run timing alone. Every latency comparison must be paired in one session.
  Quality reproduced to ±0.0002 across a two-day gap.
- **Bracket long runs.** Running the reference first *and* last turned "is 0.006 an effect?"
  into a measured 0.0001 drift.
- **A launcher file, not an inline ssh command.** Twice a `pkill -f <pattern>` matched the
  invoking shell because the pattern was in its own argv, and the relaunch silently did nothing.
- **State it, don't defend it.** Recorded in memory as `feedback_rebuttal_state_dont_defend`.
