# INITIAL REBUTTAL DRAFT — assembly frame (no Gemma), 2026-07-27
# Status: SCAFFOLD — number slots fill as chain batches pass audits; NOTHING here is final
# until the author signs off per table (presentation gate). Binding wording source:
# 2026-07-25_reviewer_actions_table_v2_and_drafts.md (survivors only — VOID register purged).
# RB-PRESET SELECTION (author rule ~11:00Z): final tables show ONE RB setting — DEFAULT
# uniform (arm1's preset); switch to ss532 only if it preserves a key trend uniform breaks;
# selection named in each table's verification line; the unshown preset's cells stay banked
# (truthful-if-asked); never claim "across presets" where they diverge.
# RAILS (binding): no cross-era latency outside v1exact cells · no mixed-basis tables ·
# no k=50 statements · never re-assert BEST-Route-frontier characterization or any voided
# t05 framing · Pareto-shaped claims only vs single-model references · NO LINKS (inline only).

## Table V1 — Learned vs static estimator on the SUBMITTED system (AC key #1)
Basis: v1exact-v2 (paper's binary; A0 re-gate PASS 03:06Z: ≤0.8pp mix, +0.7% e2e vs May).
Source: v1exact_bridge/v1x_arm{1,4}_uniform_l{6,8,10,12,18,24,30} + v1x_arm{1,4}_g025_r{6,12,23,27,30} + v1x_arm2_uniform_l{6,12,24,30} (+ sq40_6 INTERNAL ONLY).
| condition | arm1 (learned) e2e | arm4 (static prior) e2e | Δ e2e | quality Δ (paper basis) |
|---|---|---|---|---|
| λ=6 | 2080 | 2065 | +0.8% | ≤0.003 (tie) |
| λ=12 | 2230 | 2212 | +0.8% | −0.0005 (tie) |
| λ=18 | 2298 | 2299 | −0.0% | tie |
| λ=24 | 2408 | 2413 | −0.2% | −0.0031 (tie) |
| λ=30 | 3284 | 3077 | +6.7% | −0.0019 (tie) |
| γ0.25 r=23 | 2585 | 2520 | +2.6% | −0.0016 (tie) |
| γ0.25 r=27 | 2705 | 2718 | −0.5% | tie |
| **γ0.25 r=30 (mean of 6 paired runs)** | **3156** | **3391** | **−6.7% (per-run mean)** | tie |
| overload 40/6 [INTERNAL — WITHDRAWN from posting, R1] | 4492 | 6492 | −30.8% | −0.0016 (tie) |
CLAIM SET (rails-checked, rev. per author 14:0xZ): stationary parity below capacity (±0.8%,
λ6–24) · γ30 = AUTHOR-FINAL n=6 form: "consistently directionally favorable... mean 6.7%
(up to 15.3% in individual runs), though 95% CI [−14.8%,+1.4%]" (single-run −15.3% claim
RETIRED; mean row 3156 vs 3391) at
statistically indistinguishable quality · SQUARE-WAVE 40/6 ROW WITHDRAWN from reviewer-facing
surfaces (custom pattern, not standard bench; cells retained internally) · V1 presents the
3-ARM TABLE arm1/arm2/arm4 × λ{6,12,24,30} (arm2 ×4 BANKED 14:32Z; mix-shift sentence
RESTORED in assembled replies) · k-sensitivity permanently dropped (not reviewer-asked; faAU-W2 = the
degradation ladder + cold-start cost, both covered)
(all Δ≤0.004, paired-bootstrap CIs span zero in every condition — VERIFIED 17:5xZ on dev-gpu; NO quality margin claimed anywhere) · γ27 reported as parity
(its current-tree win is basis-dependent → NOT claimed) · λ30-stationary reported honestly
(static nominal holds at steady saturation; the learned edge is under burst/mismatch — which IS
the claim) · [INTERNAL ONLY — rests on withdrawn sq40/6 cells, NEVER reviewer-facing:
overload direction sign-stable on both trees (−3.7% current / −30.8% v1exact)].
Verification line: "All cells: submitted v1 configuration and code (tree 553d76c3 = May-8 +
2 inert env-gated additions, re-certified against the May campaign at ≤0.8pp mix / ≤1% e2e),
Mar-15 estimator artifacts, seed 1, n=3534/cell, 0 failures, per-cell exactness audits."
Claim frame (author-locked existence-proof narrative): TPOT is load-dependent → deployment
cannot know the future rate → measured mismatch counter-example → learned removes the bet.
DUAL-BASIS RULE: every drafted Δ sign-checked on both bases before insertion; gamma-quality
wording = FIXED-basis scope (paper-basis mixed-sign, CI-covered — forensics caveat).

## Table V2 — Dispatcher comparison incl. Po2 on the SUBMITTED system (AC key #2)
PRESENTATION (author framing, verbatim intent): "Updated Table 4 after adding the Po2
dispatcher following your suggestion" — the paper's Table 4 rows + their Po2 counterparts,
AP pw0.7 rows FILLED from banked core; pw-preset extension rows = good-to-have (~14:55Z ruling).
Basis: v1exact-v2. Source: v1x_{pt_sq,pt_rr,pt_po2}_l* + v1x_br4_{sq,po2}_l* + v1x_ap_{sq,po2}_l*.
| system | dispatcher | λ6 | λ8 | λ10 | λ12 | λ18 | λ24 | λ30 (e2e ms) |
|---|---|---|---|---|---|---|---|---|
| passthrough | SQ | 1947 | 1966 | 2017 | 2051 | 2105 | 2155 | 2182 |
| passthrough | Po2 | 2254 | 2238 | 2261 | 2257 | 2258 | 2272 | 2315 |
| passthrough | RR | — | — | — | 2670 | — | 2715 | 2734 |
| BR4 (argmax) | Po2 | 1911 | 1991 | 2175 | 2636 | 18273 | 48182 | 54112 |
| BR4 (argmax) | SQ (λ12) | — | — | — | 2626/2971 (2 runs) | — | — | — |
| AP pw0.7 | Po2 | 2553 | 2592 | 2624 | 2665 | 2737 | 2818 | 2971 |
| AP pw0.7 | SQ (λ12) | — | — | — | 2675 | — | — | — |
<!-- REVIEW-CORRECTED Jul-27 ~16:00Z: BR4 λ18/24/30 were 18216/21558/11392 (stale in-chat
aggregate, matched NO banked file); replaced with banked audited values, self-recomputed.
AP row filled from banked ap core (not pw-sweep-gated). -->
ROUTER FACTS (verified): BR4 + AP routing BIT-EXACT vs May (0.00pp both) · AP calm-regime e2e
within 1.3% of both May λ6 references · BR4 collapse knee λ12→18 (2636→18273) = the paper's documented 12-18 req/s
serial ceiling, reproduced · collapse-regime cells high-variance (banked means λ18/24/30 =
18.3/48.2/54.1 s; run-to-run spread evidenced by the two br4_sq λ12 runs 2626/2971) —
appendix note · RB stable 2230→3284 across the same sweep (20.0x lower at λ24, 16.5x at λ30).
PT finding (v1exact basis): SQ < Po2 < RR at every rate — Po2 sits between the strong and weak
simple dispatchers on latency, uniformly. Table = ONE unified v1exact table, all systems, all
three axes, normal comparison (author ruling 09:1xZ; two-block fallback design OBSOLETE).
AUTHOR RULING (extended, 09:2xZ): all v1exact cells incl. BR4/AP compare against the
SUBMITTED PAPER's numbers NORMALLY — real measurements, best-effort reproduction; no cross-era
hedging beyond the standard verification line (both br4_sq runs recorded) and the paper's own
serial-scoring sensitivity note. Gates: mix-twins ≤2pp (routing correctness) + degenerate halts;
e2e recorded. Serial high-λ climb reproduction reported as continuity evidence.
+ AP pw-sweep rows (pw ∈ {0.25,0.39,0.53,0.8} × Po2) — GOOD-TO-HAVE (author ruling Jul-27 ~14:55Z: NOT rebuttal-required; runs only if capacity remains after review-identified gaps + flip + Gemma). V2's required AP row = pw0.7 (banked: ap core ×8). No launches until the full-review conclusion names any missing key cells.

## Table V3 — Topology flip (re-provisioning adaptation) — REQUIRED (author correction Jul-27 ~14:40Z: part of the reviewer-3 weak-accept→accept push; SEQUENCED AFTER the full review/revision pass; argmax column needs topoB redeploy)
RAIL #7: RELATIVES ONLY (% change per system flip-vs-original + ordering statements; no absolutes).
Basis: current-tree WITHIN-WINDOW ONLY (both topologies measured in-window, same binary —
explicitly labeled; no May-adjacent numbers in this table).
Source: topology_flip/ banked RB+PT (25) + flip_br4_argmax col (12, lands ~21:00Z).
Claim: RB re-provisions with zero retraining; ordering preserved; NEVER "always new hosts".

## Table V4 — Robustness to partial/missing quality signal (reviewer comment)
RAIL #7: RELATIVES ONLY (deltas vs f75 + within-cell domain gaps; no absolute quality/latency).
FORM DECIDED (author, Jul-27 ~16:1xZ): INTERNAL SOURCE = the v1exact λ12/ss532 6-cell set
(f0/f25/f50/f75 + hold_gsm8k + hold_code_uf, banked+audited); DISPLAY = relatives only, rail #7
FULLY INTACT — no absolutes shown (no override/adjacency risk vs any paper number; all ladder
configs are unpublished anyway). Current-tree EXP2 remainder + task-#6 re-base: NOT needed.
Basis: current-tree WITHIN-WINDOW (labeled). LADDER-ONLY per author rail (no published-config
duplicates displayed): f75→f50→f25→f0 + both domain holdouts (v1exact 6-cell source) — ALL unpublished configs; internal
reference = f75. Claim (RECOMPUTED on v1exact source ~16:4xZ): prompt-blind bound = −0.013
global (fixed-basis −0.019, internal check only — NEVER reviewer-facing); ladder flat f25–f75
(±0.003); holdout globals: gsm8k tie; code_uf ≤0.006 (paper-basis CI excludes zero — no tie claim); excluded-domain slices unchanged (+0.002 gsm8k / +0.001
code_uf, paper basis; fixed-basis agrees, signs stable);
EXP2 adds the structured-missingness slices (excluded-domain prompts; ss532 preset shown per
selection rule). NO full-index anchor row, NO BR4 row (that comparison = V2's updated Table 4).
RAIL (binding, author 2026-07-27): no rebuttal table displays a NEW run of a published-config
cell on a different basis — prevents two-numbers-one-setting inconsistency.

## Supporting blocks (all banked/ready)
- APPENDIX NOTE (author-ruled): serial-routing collapse under load + CPU co-tenancy => high
  latency with high run-to-run variance; ROUTING-side property, independent of the dispatcher;
  does not affect key conclusions/thesis (high-load gaps 20x the variance).
- Single-model references (fixed basis, frontier framing only — never "beats always-72B").
- Cold-start accounting: ~405k profiled requests / ~4h derived (+ P4/EXP2 as empirical arm).
- Clarity commitments block (reviewer 1 writing items; continuous-batching clarification).
- Anticipated follow-ups #1-11 incl. Gemma-4 engine-consistency rail, TTFT/KV APC-silence rail.

## Reply skeletons (prose pulled from actions-table v2 SURVIVOR sections at assembly)
- R-gTfM (writing/clarity): commitments + V1 pointer. — R-287j: V1+V2+V4 + cold-start.
- R-faAU: V2 + V3 + single-model frame + follow-ups. — AC comment: 2-para, V1+V2 headline
  (AC-first ordering), family-results-forthcoming sentence (Gemma slot).
## Gemma slot — REQUIRED (author correction Jul-27 ~14:40Z: part of the reviewer-3
weak-accept→accept push; redeployment-gated; runs AFTER the full review/revision pass closes
all gaps). Per plan, one sentence in each relevant reply:
"We are additionally validating on a second model family (Gemma-3 dense ladder) in this
discussion window and will post those results as a follow-up comment."
