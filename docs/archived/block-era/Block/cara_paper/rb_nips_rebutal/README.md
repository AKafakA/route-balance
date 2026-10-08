# RouteBalance — NeurIPS 2026 rebuttal (Submission 28271)

Posting limit is **5,000 characters per comment**. Each post therefore exists twice: the full
text (what we would say with no limit) and a compressed version that fits. Compression never
introduces a number absent from the full text — that is checked mechanically, not by eye.

```
master_versions/     every version of the whole packet, v1 -> v8, plus the version log
Post/
  POST0_general/     general response + meta-review
  ReviewA_287j/      writing, clarity, batching            [SUBMITTED]
  ReviewB_gTfM/      estimator justification, deployment   <- meta-review key #1
  ReviewC_faAU/      dispatchers, generality, tuning       <- meta-review key #2 (3 comments)
  POST4_AC/          Area Chair
```

## Status

| post | full | compressed | state |
|---|---|---|---|
| POST 0 general | 1,648 | not needed | fits as-is |
| POST 1 — 287j | 6,984 | — | **already submitted** |
| POST 2 — gTfM | 7,848 | **4,852** | ready |
| POST 3a — faAU dispatchers | 6,034 | pending | needs compression |
| POST 3b — faAU topology | 5,624 | pending | needs compression |
| POST 3c — faAU family/tuning | 7,214 | pending | needs compression |
| POST 4 — AC | 2,915 | not needed | fits as-is |

Posting order: POST 0 first (it carries the two scoping notes the others rely on), then 2 and
3a (the meta-review's two keys), then 3b, 3c, and 4 last.

## The single claim

Against a static prior calibrated to measured per-tier decode medians, the learned head is
**8.1–9.8% cheaper per 1,000 requests** at equal quality (all seven paired CIs span zero),
equal latency (three time-matched rates, within 2.1%, sign-changing) and equal throughput.
We claim no latency or quality advantage for the learned head — that concession is deliberate
and is what the measurements support.

## How POST 2 was compressed, 7,848 -> 4,852

Nothing load-bearing was cut. Savings came from:

- not restating the reviewer's own questions (just Q1..Q4)
- the per-tier TPOT constants table -> the four constants inline, keeping the sweep-median
  construction that shows the comparator is not a straw man
- the per-rate CI column -> "all seven span zero, widest [-0.0063, +0.0014]" — equally
  checkable, ~180 characters cheaper
- the throughput column -> the sentence that already stated it
- removing a duplicated disclaimer: "we make no latency claim" had been said twice

Kept in full: the seven-rate cost table, the 8.1-9.8% claim, the three-rate latency tie, the
third arm establishing the latency *term* is necessary (69.1% smallest-tier at 0.4899), the
explicit concession, the cost-basis warning, and complete answers to Q2-Q4.

## Figures verified against cell data, not against earlier drafts

| figure | source |
|---|---|
| tier mix 56%/38% vs 68%/29% at λ=24 | recomputed from the 6 time-matched λ=24 cells: 55.6/38.6 vs 67.5/29.3 |
| latency 2,243 / 2,224 · 2,458 / 2,487 · 3,143 / 3,079 | 18 time-matched cells, 3 seeds each, all pass the stationarity screen |
| 8.1–9.8% | recomputed from the cost columns |
| widest CI [-0.0063, +0.0014] | λ=24 row; all seven verified to span zero, max abs(Δq) = 0.0037 |
| constants 10.7 / 19.9 / 16.5 / 42.6 | medians of their own per-rate rows |
| 69.1% smallest-tier, 0.4899 | v1exact arm2 cells — mix identical at λ 6/12/24/30, the load-blind signature |
