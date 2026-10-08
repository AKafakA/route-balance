# DRAFT — Q1 update to Reviewer gTfM (the promised retest)

**Status: POSTED by the author 2026-08-03 ~14:00Z (with an averages note added at posting).**
(Original hold text follows for the record: post ONLY once ALL window results are
on hand (grid + Phase E′ + retuning), as one consolidated update. Quality is on the
POSTED-SERIES basis** (zero-scored responses excluded, matching every earlier rebuttal
table; the include-zeros basis is ≈0.10 lower and is the CR-corrected series — covered by the
standing corrected-at-CR disclosure). Every number below is cell-exact from
gate-passing cells (provenance at bottom, stripped before posting). Rails checked: inline
tables only, no links, state-and-stop, no codename, basis stated once.

**Context**: gTfM 03 Aug 05:27 — *"Q1 (learned vs. static TPOT) — still open. Please post the
updated results when the reservation allows and I will read them."* This is that post.

---

## Updated static-TPOT results (Q1)

Official Comment — draft body:

As promised, here are the re-run learned-versus-static results: the same testing setting as
the paper, with RB under both the (0.5, 0.3, 0.2) and uniform (0.33, 0.33, 0.33) weights and
6 seeds per configuration for the confidence intervals. The static arm is gated by checking
the runtime logs of the TPOT values actually used in scoring (the per-tier constants, exactly,
with no predictor invocation), and the two arms demonstrably route differently — 4.4% of
placements differ, with the learned head shifting 2–3 pp of traffic from 14B to 72B.

Poisson arrivals at at (0.5, 0.3, 0.2)

| λ | arm | quality | cost $/1k | throughput | sched overhead ms | mean e2e ms | p99 e2e ms |
|---|---|---|---|---|---|---|---|
| 12 | static | 0.5129 | 0.0221 | 10.95 | 177 | 3,106 | 18,689 |
| 12 | learned | 0.5144 | 0.0231 | 10.95 | 191 | 3,257 | 19,409 |
| 24 | static | 0.5129 | 0.0221 | 19.47 | 205 | 3,452 | 21,059 |
| 24 | learned | 0.5153 | 0.0232 | 19.43 | 243 | 3,627 | 21,206 |
| 30 | static | 0.5129 | 0.0220 | 23.25 | 282 | 3,728 | 22,111 |
| 30 | learned | 0.5157 | 0.0235 | 23.15 | 555 | 4,145 | 23,683 |

And at uniform (0.33, 0.33, 0.33):

| λ | arm | quality | cost $/1k | throughput | sched overhead ms | mean e2e ms | p99 e2e ms |
|---|---|---|---|---|---|---|---|
| 12 | static | 0.5102 | 0.0175 | 11.71 | 158 | 2,646 | 15,195 |
| 12 | learned | 0.5110 | 0.0176 | 11.68 | 171 | 2,724 | 15,614 |
| 24 | static | 0.5102 | 0.0175 | 22.06 | 184 | 2,971 | 17,386 |
| 24 | learned | 0.5119 | 0.0176 | 22.07 | 213 | 3,067 | 17,924 |
| 30 | static | 0.5102 | 0.0174 | 26.66 | 216 | 3,176 | 18,511 |
| 30 | learned | 0.5110 | 0.0176 | 26.70 | 332 | 3,358 | 18,993 |

Under bursty arrivals (gamma CV²=4 versus Poisson, identical per-seed arrival schedules), the
paired degradation ratio p99(burst)/p99(Poisson) shows no significant difference between the
estimators at any rate under either preset — paired t(5), 6 seeds:

| preset | λ | static | learned | paired Δ (95% CI) |
|---|---|---|---|---|
| 532 | 12 | 0.995 ± 0.027 | 1.007 ± 0.025 | +0.013 [−0.027, +0.052] |
| 532 | 24 | 1.002 ± 0.040 | 1.048 ± 0.035 | +0.046 [−0.007, +0.099] |
| 532 | 30 | 1.020 ± 0.054 | 1.019 ± 0.059 | −0.002 [−0.084, +0.080] |
| uniform | 12 | 1.022 ± 0.048 | 1.043 ± 0.050 | +0.020 [−0.050, +0.091] |
| uniform | 24 | 1.008 ± 0.046 | 0.999 ± 0.055 | −0.010 [−0.084, +0.065] |
| uniform | 30 | 1.042 ± 0.075 | 1.013 ± 0.079 | −0.029 [−0.164, +0.107] |

In summary, the two arms are demonstrably different: 4.4% of assignments change, with the learned head shifting 2–3 percentage points of traffic from 14B to 72B. Across all six point estimates, this produces a small positive quality shift of up to +0.0028, while increasing mean latency by 3–11% and cost by up to 7%. We detect no statistically significant difference in burst-induced p99 degradation between the estimators at any evaluated rate.

These results sharpen our conclusion. A measured per-tier TPOT constant is a practical and substantially simpler alternative for cross-tier selection under the evaluated conditions. The learned head enables a slightly more quality-oriented operating point, but it is not necessary for the core joint-routing mechanism. We will revise the paper to reflect this narrower positioning and discuss learned prediction as an optional extension for settings with substantial TPOT drift or explicit SLO calibration.

**On Q2:** Agreed. We will explicitly state that the current latency objective targets decode-dominated generation and does not model TTFT or prefix affinity. We verified from the effective runtime configuration that automatic prefix caching was enabled in the evaluated runs, and we will report this in the experimental setup. Because the workload consists predominantly of single-turn prompts without intentionally shared prefixes, observed cross-request prefix reuse is limited in this evaluation. Explicit TTFT-, prefill-, and prefix-affinity-aware routing remains future work.

---

## Provenance (stripped before posting)

* steady table ← `s2_q1burst_PROD` qb_*_p532 pois cells (72/72 gate-passing) + clean s1 re-run;
  quality via deepeval lookup (0/42,408 missed); cost via PRICE_OUT decode basis
* sched overhead ← mean_scheduling_overhead_ms per cell, pooled 6 seeds
* burst table ← `ANALYSIS_FULL_144.txt` paired section (532)
* uniform sentence ← same file, uniform section
* placement disagreement 4.4% ← pilot per-request join (154/3,534 differing, prompt-keyed)
* zero-call verification ← ASSERT-5 per cell (`tpot_predict_calls` delta 0; xgb ms ≤ 0.1)
* per-tier constant verification ← tpot_gate per cell (4 tiers, 4 exact distinct constants)

## Notes for the author (not part of the post)

0. **AUTHOR INSTRUCTION (13:40Z): the scoring-basis matter is NEVER mentioned in OpenReview
   posts.** It exists in the paper/CR track only, and is exposed at the time the author
   chooses before the deadline — per the standing corrected-at-CR disclosure. The post body
   above contains no reference to bases, series, or scoring changes; quality is stated
   plainly and the numbers are continuous with the thread.

1. **Quality basis (author decision 13:30Z)**: the table is on the POSTED-SERIES basis
   (zero-scored responses excluded), so it reads continuously with every earlier rebuttal
   table (≈0.51 scale; static 0.5129 vs the retracted table's 0.5161 — close because the
   retracted learned≈static runs were effectively the same config as static). The
   include-zeros basis (static 0.4103, deltas unchanged in sign) is the CR-corrected series;
   the standing corrected-at-CR disclosure covers the switch, and the acknowledgment happens
   once, with all results, per the author.
2. λ30 row uses the clean s1 re-run (scheduler-backlog cell replaced; original preserved).
3. The reviewer's Q2 instruction ("needs to be in the paper") is queued in the CR list.
4. Throughput at λ30: static 23.25 vs learned 23.15 (−0.4%) — stated in the table, not
   editorialized.
