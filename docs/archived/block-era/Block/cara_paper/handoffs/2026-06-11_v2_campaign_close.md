# v2 CAMPAIGN CLOSE-OUT — 2026-06-06 → 2026-06-11 (cluster released 2026-06-11 ~09:55Z)

The complete record of the 5-day v2 experimental campaign: what ran, what it concluded, what broke,
what we fixed, and what the next slot inherits. Companion to the rolling log in
`2026-06-07_v2_rerun_handoff.md` (operational detail) — this doc is the synthesis.

## 0. Final state (the one-screen version)

| Artifact | State |
|---|---|
| **Paper** `route-balance-paper/neurips_2026_v2.tex` | converged **7/7/7 across all 3 reviewer tracks, 0 blocking** — commit `4cfe066`, PDF md5 `addf518c`, 23pp (body ≤9, refs top of p10), 0 "cara" hits, consistency gate PASS 7/7 |
| **Block** (internal, source of truth) | `1e0d013` — all code, env-gated ablations, orchestration scripts, docs |
| **route-balance** (public mirror, 0 cara) | `bf2eee9` — recovered + regenerated from Block, aggregator bug fixed, full-paper drivers + REPRODUCE.md §11 |
| **v2-exp pack** | results 11G (16 `_aggregate` JSONs) · models 2.4G · data 144M · paper · configs · scripts · `rb_recovered/` — **22/22 paper elements sourced, 37/37 instruction scripts present** |
| Cluster | RELEASED. Final sweep verified 0 unsynced cells; deployed predictors in ≥2 locations; as-run configs in pack |
| All commits | **LOCAL only, never pushed** (standing rule) |

## 1. Campaign timeline (process)

- **Jun 6–7 — v2 stack + main grid.** Committed v2 stack (`2074f4a`): cap 1024, route-on-deepeval,
  k=10 KNN, minmax quality norm, fixed param grids, zero-safe aggregator. Main grid ran Jun 7
  09:00→23:00Z: **baselines 168 cells + cara tuples 56 + batching 23, 0 failures** (~247 cells,
  N=3,534 each). The br4 collapse curve arrived live: 27–30s @λ18 → 49–51s @λ24 → 64s @λ30 while
  cara uniform stayed flat 2.62/2.68/2.78s → the **23× headline reproduced on v2**.
- **Jun 8 — downstream tracks.** Multi-seed (45 cells, 0 fail; quality seed-stable to ±0.0006);
  budget (pipeline-mode filter was a NO-OP → root-caused → native redo: filter 32.8/19.2/12.6%
  ≈ paper ✓); batching ablation redone after the `batch_config` POST bug (top-level vs nested →
  toggles never applied; corrected run matches paper: LPT-off −1.6%, adaptive-off +3.5%; NEW
  finding: bs=1 no longer cliffs — batched-KNN makes per-request scheduling cheap); cost-extreme
  (cara cheap corner beats AvgPro cheap corner at high λ); vLLM-SR brought up (2 infra bugs fixed,
  content-aware complexity-gate baseline built per user).
- **Jun 9 — vLLM-SR grid + second judge + consolidation.** vLLM-SR all-7λ (user option A): quality
  flat ~0.37 but **service collapses** — 0 fail ≤λ10, 7.7% fail @λ12, 89–97% fail @λ18–30 (router
  CPU embedding saturates <10 req/s). Gemma-3-12B second judge on 4×A100 (14,427 paired): r=0.555
  per-pair, **system ordering preserved**. Fresh consolidation `AGGREGATE.csv` (287 cells) +
  adversarial code review (no number-invalidating bugs; v1 zero-drop impact +0.105 confirmed fixed).
  rb-mirror reproducibility demos + AvengersPro flood root-caused (shim auto-CUDA → pinned CPU).
- **Jun 10 — enhanced baselines + native-repro crisis + external review round.** Discovered grid-era
  BEST-Route ran its DeBERTa forward **inline on the scheduler event loop** → built the **enhanced**
  (concurrent-scoring) variants: br4 survives at 6.9–11.4s — still 2.6–4.5× behind, quality
  byte-identical; AvengersPro same flaw class patched. Framing locked: serial = shipped deployment
  pattern, "enhanced" never "fix"; two-tier presentation (main = serial, App B ladder). Native-mode
  reproduction crisis root-caused: **CUDA_VISIBLE_DEVICES="" mandatory** (XGBoost probes co-located
  GPU → 600× predict slowdown) + my own OMP_NUM_THREADS=1 throttled MiniLM (the 10% gap the user
  caught). E7 tier-removal (graceful degrade), oracle/cost-sensitivity/tails/safety/scaling offline
  analyses, BestAcc eval bug found (0.436 was garbage; canonical = 0.348). ChatGPT/Claude web round-2
  reviews ingested.
- **Jun 11 — reviewer-driven ablations + paper convergence + rb incident + close.** P1 3-arm
  fusion-isolation; P3 non-stationary completed to a fair 3×3 (enhanced-AvgPro added per review);
  n=3 latency reseeds (λ12/24/30); 40/6 overload probe; **arm-4 static-prior ablation** (the
  decisive one — see §2.6); paper loop converged 7/7/7. Then the **rb rm-rf incident** (see §4),
  full recovery + provenance audit, rb smoke test (caught a real public-repo bug), final
  preservation sweep, cluster released.

## 2. Experiments & conclusions (every number traced; sources in PAPER_REPRODUCTION_MAP.md)

### 2.1 Main grid (the paper's spine)
287 aggregated cells. **One deployed stack sweeping a weight vector spans cheapest-to-highest-quality
while every baseline is one fixed point**: quality ceiling 0.419 wq0.8 (+0.013 vs BEST-Route CI
[+0.005,+0.022]; +0.043 vs AvengersPro), cost corner 1.67e-5 USD/req, uniform 2.3s/0.371.
Load-robustness: cara bounded 2.2–2.8s all λ; br4 serial collapses to 64s (23×); enhanced br4 still
2.6–4.5× behind (instance-blind policy → 98% onto the 14B tier); vLLM-SR independently corroborates
the serial-router collapse class. Quality-vs-sustained-throughput: RB dominates at every level
(287-cell hull check); reaches 27.6 req/s vs br4's 21.8.

### 2.2 Budget control (§5.4)
Native-mode only (filter lives in the batch scheduler, NOT the pipeline router path — the no-op
artifact cost one redo). Filter cuts exhaustion −6.3/−4.3/−2.9pp across 75/45/30% mixes;
cross-system vs br4-argmax is a wash at medium/loose — contribution honestly scoped to the
within-system filter mechanism. OFFLINE-A forensics: exhaustion is genuine infeasibility
(cap=1 in 99%), not a fencepost.

### 2.3 Multi-seed + bootstrap (App A)
Quality/cost: cara wq0.8 0.4180±0.0002, uniform 0.3697±0.0005, deterministic routers exactly-zero
variance. Paired bootstrap CIs exclude zero vs both baselines. Latency: multiseed_v2 latency was
contention-contaminated (documented, excluded); replaced by **n=3 independent reseeds on the healthy
cluster: uniform ±1.4/±2.7/±2.0% at λ12/24/30** — load-robustness magnitudes are not single-run
artifacts.

### 2.4 Second judge (App D)
gemma-3-12b-it, same GEval criteria, 14,427 paired: per-pair r=0.555 (judges disagree per response),
**system-level ordering preserved**; headline t=0.5 re-judge across two seeds: Δ=+0.020 [+0.008,+0.031]
and +0.012 [+0.001,+0.023] — RB>BR judge-robust at the exact operating points.

### 2.5 Non-stationary arrivals (App G, Table 15 — completed Jun 11 to fair 3×3)
Gamma-burst (b=0.3) + square-wave (30/6) at matched mean λ18, all three systems:
RB 2615→2524/2577 · enh-AvgPro 2400→2473/2733 · enh-BR4 4079→4392/**7107 (+74%)**.
**Conclusion: burst-latency robustness tracks scoring architecture (amortized vs serial), not routing
policy**; RB's edge over the latency-competitive AvgPro is the quality/cost axes it holds simultaneously.

### 2.6 Fusion-isolation triangle + ARM-4 (App G — the campaign's biggest scientific correction)
- Arm2 vs arm3 (reactive queue vs predictive T̂ tiebreak, identical weights/mix): **wash** (±3%).
- Arm1 vs arm3: latency term in the *model* score = 26–31% faster, steers 72B 14%→1%.
- Tier mix is **rate-independent** (Table mixvslambda: 72B pinned 1%, λ6→30) and **phase-flat even
  under a 40/6 overload** whose 40 req/s peak exceeds the 27.6 sustained max.
- **Arm-4 (static per-tier TPOT_nominal × L̂, zero telemetry, no XGBoost): reproduces the full
  learned objective on every axis** — mix 58/11/31/0 vs 57/10/32/1, quality 0.369 vs 0.370, E2E
  equal-or-faster (2.41 vs 2.62s stationary; 3.12 vs 3.79s overload).
- **Paper reframed accordingly**: contribution = latency-aware model scoring + queue-aware dispatch;
  the learned predictors are accuracy headroom (SLO filter), not the load-bearing mechanism;
  abstract's "no decoupled combination achieves" → "decoupled baselines do not reach" (the
  impossibility claim was falsifiable and is gone). All three reviewers scored this *up*, not down —
  honest demotion of our own component bought credibility.

### 2.7 Other ablations
Batching (LPT-off ≈ default; adaptive-off +3.5%; **bs=1 no cliff** — flag vs v1 §5.5); tier-removal
(72B loss = quality-ceiling event, not availability event; uniform unchanged, 0 fails);
cost-model sensitivity (orderings invariant under 5 price vectors); safety (no preset dumps
is_harmful prompts); scaling microbench (estimator amortizes); predictor accuracy (BestAcc fixed
0.348 vs 0.25 random; oracle 0.582 / random 0.376 / always-72B 0.428 — served 0.419 closes ~1/5 of
the random→oracle gap); LODO OOD (±0.05 on 4/7 folds, gsm8k −0.09 = honest cold-start caveat).

## 3. Reviewer convergence (process + outcome)

Three tracks: ChatGPT-web-seeded persona, Claude-web-seeded persona, unseeded local-general.
- Jun 10 round: 6 / 5 / 6 with blocking items (fusion mechanism un-isolated, Table-10 quality-scale
  bug, mechanism contradiction between new appendices).
- Jun 11: P1+P3+n3+mix-table integrated → ChatGPT 7, Claude 7; local-general initially 6 with the
  "decorative predictor / static prior would suffice" blocker → **arm-4 ran the exact experiment**
  → local-general 7. **Final: 7/7/7, zero blocking.** Consensus gap-to-8 = inherent (thin ML novelty,
  one family/topology, single-judge quality) = next-slot empirical work, not fixable defects.
- Standing author-declined items (do NOT re-raise): arch-figure style, task-native EM, second
  family/topology this cycle, venue-fit editorializing, quality-headline hedging
  (feedback_no_quality_claim_hedging).

## 4. Incidents, root causes, lessons (the expensive part — read before next slot)

1. **rb repo rm-rf deletion (Jun 11, severity: highest).** Ran `transform_block_to_route_balance.sh`
   with `OUT=` inside the live repo; script line 5 is `rm -rf "$(dirname "$OUT")"` → deleted
   `route-balance/` incl. `.git`. Local-only commits May 8→Jun 9 + `v2-reproduce/` mirror lost.
   Recovered from 3 sources (GitHub May-8 remote, cluster `/mydata/rb-public` Jun-10 validated pkg,
   regeneration from intact Block). **Lessons: (a) NEVER run a script containing rm-rf with a custom
   path without reading it first; (b) the repo was un-pushed for a month against the two-location
   rule — that's what made it a near-disaster; (c) direction-of-diff audit afterwards proved Block
   was a strict superset (0 rb-only code lines) — the mirror discipline (Block = source of truth)
   is what bounded the damage.**
2. **Native-mode 600× reproduction scare (Jun 10).** XGBoost ≥2 probes the co-located GPU on every
   inplace_predict (0.17ms→930ms). Fix: `CUDA_VISIBLE_DEVICES=""` + code-level `device=cpu`.
   Compounded by my own `OMP_NUM_THREADS=1` throttling MiniLM (estimator 22→51ms → λ24 collapse,
   the 10% gap the user caught). **Lesson: the serving env contract is part of the experiment;
   document it or the results are unreproducible. Both contracts now in code + docs + pack.**
3. **Zero-drop quality bug, third and fourth occurrences.** `score or fallback` drops valid 0.0
   DeepEval scores (+~0.12 inflation): v1 submitted numbers (May), my Table-10 inline script (Jun 10,
   caught by Claude-web), and the **public rb `reproduce/aggregate_results.py`** (Jun 11, caught by
   the smoke test: 0.489/cov 0.758 → fixed → 0.3708/cov 1.000). **Lesson: this bug class regenerates
   whenever quality code is rewritten — `check_quality_consistency.py` gate now blocks paper builds,
   and any new quality script must use the canonical qcompute join.**
4. **Config-POST silent no-ops.** Batching `batch_config` nested under `slo_defaults` → toggles never
   applied (invalidated 38 cells, redone); budget filter inert in pipeline mode (redone native).
   **Lesson: POST-verify every config change against live state (`/v1/batch_stats`, smoke-gate with
   an asserted observable) — a 200 response proves nothing.**
5. **Multiseed latency contamination.** Back-to-back cells under transient contention inflated
   latency (uniform λ24 23.5s vs healthy 2.7s); quality/cost unaffected (deterministic). Excluded
   latency from that table, later replaced with clean n=3 reseeds. **Lesson: latency claims need a
   health-verified window; quality claims don't.**
6. **Observability failures in the final hour.** rb re-serve "hang" was first blamed on the harness;
   actual measurement showed a 7B backend already torn down by lease reclaim. **Lesson (the
   session's recurring one): measure before explaining — name no cause without the log/curl that
   proves it.** (Same discipline as feedback_no_fabricated_explanations.)

## 5. Process improvements that should persist

- **Env-gated ablations, default-off** (`CARA_TIEBREAK` / `CARA_STATIC_LAT` / `CARA_SQUARE`,
  rb: `ROUTE_BALANCE_*`): reviewer-grade ablations with zero risk to main results; documented with
  expected numbers in REPRODUCE docs §"Env-gated extra experiments" / rb REPRODUCE.md §11.
- **Pre-build consistency gate** (`check_quality_consistency.py`): recomputes every quoted quality
  number from released response_details; PASS required before any pdflatex.
- **Smoke test the public repo like an experiment**: the rb smoke (py_compile 165 files, 0-cara scan,
  functional aggregate against a known cell) caught a would-have-shipped wrong-numbers bug.
- **Persona-reviewer convergence loop** with verbatim web-review seeds + an unseeded fresh track;
  author-declined list to prevent re-litigating decided framing.
- **Provenance audits**: cluster-vs-local code diff (0 unique lines), 119×-Block/8×-rb run trace,
  37/37 instruction-script existence check, 22/22 paper-element→pack source map.
- Established infra playbook items held: bracketed pkill, detached tmux + verify-produces, bounded
  SSH, staggered vLLM starts, A30-no-/mydata.

## 6. Open items → next slot (no cluster needed except where noted)

1. **Push rb + off-machine pack backup** — the root-cause fix for the single-copy failure. Highest
   priority before any other work. (Pack currently complete but openclaw-only.)
2. **Rebuild `v2-reproduce/`** self-contained mirror from Block (the deletion's one casualty).
3. **Framing decisions (author's)**: quality-headline strength (declined to hedge — standing);
   venue narrative (systems contribution vs ML novelty) for the real reviews.
4. **Arm-4 follow-through (cluster)**: multi-seed + longer-horizon confirmation; if the learned-T̂
   framing is ever revived, the discriminating test is documented in `Q3_static_vs_live_probe.json`.
5. **Deferred generality (cluster, all three reviewers' gap-to-8)**: second model family (gemma),
   second topology, trace-driven workload, human/task-native eval.
6. E2 judge_t05 re-run never produced output this lease (0 files) — App D already stands on the
   Jun-9 gemma data; rerun only if a reviewer demands seed-matched re-judging.
7. Known minor flags: bs=1 no-cliff difference vs v1 §5.5 (flagged in paper); gsm8k OOD drop to
   chance (disclosed); greedy-vs-Hungarian gap (future work).
