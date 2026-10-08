# FINAL AUTHOR POST — draft skeleton (S2 results slot in here)

**Status: SKELETON — no numbers yet.** Every `⟨…⟩` is a slot filled only from cells that
passed every gate. Drafted before the window per CLOSEGATE §4 so post-assembly is filling,
not writing.

**Rails checked at every edit (CLOSEGATE §4 + standing):**
- inline tables only, **no links**
- **never re-assert the two basis-dependent claims**
- state the measurement and stop — no volunteered caveats, no explaining artefacts
- degradation ratios paired by seed, t(5) CIs from 6 seeds — never z on n=6
- a SKIP/refusal is reported as such if it lands in scope; nothing papered over
- codename check before posting: the word "Cara" must not appear — RouteBalance only
- numbers cell-exact against the analysis CSVs, no rounding beyond the table format

---

## Post body (draft)

Following up on our Q1 reply with the completed static-vs-learned comparison on the full
four-tier fleet, and the bursty-arrival results the earlier reply committed to.

**Static prior vs learned head, four tiers, both at RB's (0.5, 0.3, 0.2).** As posted, both
arms share the objective, the within-tier queue tiebreak and the weights, and differ only in
the TPOT estimator (fixed per-tier constants vs the queue-conditioned head).

*(532 rows FILLED 2026-08-03 from the completed 6-seed half — DRAFT, author approval owed;
quality = deepeval lookup, cost = decode USD/1k req, pooled over 6 seeds, Poisson)*

| λ | arm | quality | cost | mean E2E (ms) | p99 E2E (ms) |
|---|---|---|---|---|---|
| 12 | static | 0.4103 | 0.0221 | 3106 | 18689 |
| 12 | learned | 0.4125 | 0.0231 | 3257 | 19409 |
| 24 | static | 0.4103 | 0.0221 | 3452 | 21059 |
| 24 | learned | 0.4137 | 0.0232 | 3627 | 21206 |
| 30 | static | 0.4103 | 0.0220 | 3728 | 22111 |
| 30 | learned | 0.4146 | 0.0235 | 4145 | 23683 |

⟨One sentence stating the steady-state outcome, direction only, from the table. If the arms
are statistically indistinguishable at some rates, say exactly that.⟩

**Under bursty arrivals** (gamma CV²=4 interarrivals vs Poisson, identical arrival schedules
per seed, 6 seeds), the paired degradation ratio p99(burst)/p99(Poisson):

*(532 FILLED — DRAFT; paired by seed, t(5); "ns" = CI includes 0)*

| λ | static | learned | paired Δ (t(5) CI) |
|---|---|---|---|
| 12 | 0.995 ± 0.027 | 1.007 ± 0.025 | +0.0125 [−0.0268, +0.0519] ns |
| 24 | 1.002 ± 0.040 | 1.048 ± 0.035 | +0.0460 [−0.0065, +0.0986] ns |
| 30 | 1.020 ± 0.054 | 1.019 ± 0.059 | −0.0017 [−0.0837, +0.0803] ns |

⟨One sentence: which arm degrades less, stated from the paired test.⟩

**Robustness to the weight preset.** The same comparison at uniform weights
(0.33, 0.33, 0.33):

*(uniform FILLED 2026-08-03 — DRAFT; same 6 seeds; quality/cost/mean-E2E pooled, Poisson)*

| λ | arm | quality | cost | mean E2E (ms) | burst ratio (paired Δ, t(5)) |
|---|---|---|---|---|---|
| 12 | static | 0.4025 | 0.0175 | 2646 | 1.022 ± 0.048 — Δ+0.0203 ns |
| 12 | learned | 0.4046 | 0.0176 | 2724 | 1.043 ± 0.050 |
| 24 | static | 0.4025 | 0.0175 | 2971 | 1.008 ± 0.046 — Δ−0.0095 ns |
| 24 | learned | 0.4054 | 0.0176 | 3067 | 0.999 ± 0.055 |
| 30 | static | 0.4025 | 0.0174 | 3176 | 1.042 ± 0.075 — Δ−0.0287 ns |
| 30 | learned | 0.4042 | 0.0176 | 3358 | 1.013 ± 0.079 |

⟨Draft sentence for approval: The pattern is preset-independent: under both weightings the
learned head yields higher quality (via increased 72B routing) at slightly higher latency and
cost, and burst degradation shows no significant difference between estimators at any rate.⟩

**Full grid confirmation.** Re-running the complete comparison grid (RouteBalance and all
baselines at their own best knobs, rates λ∈{6,12,18,24,30}) on this fleet:
⟨one sentence per Phase E′ verdict at the rates the earlier tables cited — strict on all four
axes or not, as measured⟩.

---

## Assembly checklist (run before posting)

1. Fill slots ONLY from `DOMINANCE_l*.csv` + `burst_analysis` output + Q1 cells; cite the
   producing file for every number in the margin notes below (stripped before posting).
2. Delete any table row whose cells did not all pass gates; say what is absent and why in one
   clause, not a paragraph.
3. Rails re-check (top of this file), then codename grep, then post.
4. Margin notes (provenance, stripped at post time):
   - Q1 532 table ← s2_cluster/cells/s2_q1burst_PROD qb_*_p532 (72/72) + ANALYSIS_532_6seeds.txt
   - burst 532 rows ← burst_analysis.py --cells s2_q1burst_PROD (6 seeds, paired)
   - uniform block ← qb_*_puniform (72/72) + ANALYSIS_FULL_144.txt
   - 532 λ30 rows ← clean s1 re-run (qb_learned_pois_s1_l30_p532; orig preserved .BACKLOG_ORIG)
   - grid sentence ← ⟨DOMINANCE csv paths — fills per rate⟩
