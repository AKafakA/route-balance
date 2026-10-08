# Handoff for Claude: CARA Baseline Framing, Fixes, and Experiment Plan (revision 3, 2026-04-15 PM)

## Revision 3 deltas

1. **Arrival-rate λ is the primary Phase 2 load axis** (replacing fixed λ=10 + optional QPS sweep).
2. **Sweep-scope layering** — λ-axis curves use default knob per router (~35 runs), Pareto sweeps use knob × 2 anchor rates (~50 runs, net ~40 new), ablations at 3 anchor rates (~30 runs), supporting at knee only (~12 runs). Total ~127 runs.
3. **Phase 1 is preflight-only** — paper numbers strictly from Phase 2 18-node.
4. **BEST-Route has a soft deadline of Apr 24 morning**; cite-only fallback if it doesn't train in time.
5. **Go/no-go checklist** in design doc §7.C governs Apr 26 cluster go-ahead.
6. **Train/val hygiene rule** — retrained baselines never touch the 500-prompt clean bench.

Revision 2 deltas preserved below.

---

## Status

**Revision 3** of this handoff. The methodology, baselines, and figure plan are locked as of 2026-04-15 PM. The design-of-record is `apr15_baseline_comparison_design.md` (revision 3, same directory). This file is the short brief — read the design doc for detail. Revision-3 deltas summarized in the previous section; revision-2 material preserved below for audit.

Revision 1 of this handoff remains in git history; the headline changes in revision 2 (now subsumed by revision 3) were:

1. **BEST-Route is now a real baseline**, trained on Qwen with a stripped-down pipeline (no Best-of-N). The prior `best_route` argmax-quality implementation was a misattribution; renamed to `quality_greedy_ablation`.
2. **Avengers-Pro is promoted to an explicit main-table baseline** with `performance_weight` knob sweep.
3. **RouteLLM switches from the retrained BERT variant (val_acc=0.389) to the paper-recommended MF variant** with pretrained `routellm/mf_gpt4_augmented` checkpoint + published threshold α=0.11593.
4. **Bench data-integrity fix** — `best-route-v3-test-500.jsonl` overlapped 289/500 prompts with the predictor's training split. Resampled to a clean 500-prompt bench from `test_scored_filtered.jsonl` (the HF-canonical clean held-out scored set).
5. **Methodology grounded in published precedents** — RouterBench (Hu 2024) for the Pareto + AIQ framework, MoE-CAP (Jiang 2025) for the CAP radar, RouterEval (Huang 2025) for scaling-law context, and LLMRouter / BentoML Explorer for extensibility and interactive-tooling references.
6. **Cost model is per-token vendor pricing** (MaaS framing), not hardware TCO.
7. **Balance is scheduler-specific** — kept minimal in the main paper (one comparison figure) and moved to appendix for full w_balance sensitivity sweep.
8. **Figure plan finalized** — 2D cost-quality Pareto (headline) + 2D cost-latency Pareto + CAP radar per router + goodput bar (constraint) + throughput bar (raw work done) + memory-variance time-series (§5.X) + routing-overhead stack + Table 1.

## Agreed Core Framing

### 1. What `Cara-full` is

Integrated online batch scheduler with:

- LPT batch ordering,
- bucket-based probabilistic feasibility filtering (TTFT/TPOT in `CaraCDFFilter`; budget + quality in legacy inline path — unified four-constraint bucket filter planned),
- multi-objective scoring over **four** continuous optimization axes (`w_latency`, `w_quality`, `w_cost`, `w_balance`),
- score-based joint (model, instance) assignment,
- **online, per-request, heterogeneous**.

Present as the system itself, not as a pluggable-tuple row.

### 2. What the pluggable matrix is for

Router × Dispatcher × Filter decomposition remains useful for **supporting studies only**:

- signal-swap analysis (Supporting Study 1),
- filter comparison (Supporting Study 2),
- engineering validation before integration.

It does not define the paper's main-table row.

### 3. What external baselines are

Practical wrapper / peer systems with appropriate disclaimers:

- **vllm_sr-wrapper** — external peer scheduler.
- **RouteLLM-mf-wrapper** — pretrained `mf` + published threshold 0.11593 + domain-shift footnote.
- **Avengers-Pro-wrapper** — upstream `SimpleClusterRouter` with our Qwen-trained K-means + rankings.
- **BEST-Route-wrapper** — DeBERTa-v3-small multi-head router trained on our Qwen pool via armoRM pair-wise preference labels (stripped Best-of-N).

Each wrapper paired with both round-robin and shortest-queue dispatchers for fair evaluation; llumnix_minus dispatcher optional.

`BOute` (Jiang MLSys'26) stays in related work — it is an **offline MOBO joint-optimization** framework, not an online router, and implementing it honestly requires re-building the MOBO loop (multi-week).

### 4. Quality evaluation

Pre-computed deepeval lookup per (prompt, chosen_model) from `asdwb/cara_model_estimator` (HF) via the 3,634 clean held-out prompts in `test_scored_filtered.jsonl`. No live judging needed. RouterBench-style methodology.

### 5. Failure handling / ops

No failure-recovery pytest gating for this paper phase. Minimal pre-run sanity checks only.

## Decisions Locked (with defaults if missing)

| # | Decision | Value |
|---|---|---|
| 1 | BEST-Route | Trained in — stripped Best-of-N, ~3-4 h pipeline (§5.15 of design doc) |
| 2 | Quality bench | Resample 500 from `test_scored_filtered.jsonl` (clean, 100% deepeval coverage) |
| 3 | Latency metric on Pareto / radar | Mean E2E latency |
| 4 | Wrapper dispatcher pairings | Round-robin + shortest-queue; `llumnix_minus` optional |
| 5 | Balance scope | Minimal main (single comparison figure) + full sensitivity in appendix |
| 6 | Apr 26 18-node run | Included as §7.B Phase 2 in the design doc |
| 7 | Knob grid | 5 values per router (§3.2 of design doc) |
| 8 | monitor.py sampling | 5 s all nodes |
| 9 | Hypervolume normalization | Dropped HV from the main figures; radar uses per-axis native units via the MoE-CAP-Dashboard `normalize`/`normalize_cost` helpers (Apache 2.0) |
| 10 | LLMRouter library adapter | Deferred post-Apr 17; cite as extensibility precedent |
| 11 | Cost model | Per-token vendor pricing (MaaS), not MoE-CAP hardware TCO |
| 12 | Primary paper quality metric | Pre-computed deepeval lookup |

## Items Fixed Since Revision 1

### A. BEST-Route is a real baseline (was: dropped / cited only)

Stripped-down pipeline laid out in §5.15. ~3-4 h wall-clock on the 4-A30 cluster. Replaces the prior `best_route.py` argmax-quality placeholder, which is renamed to `quality_greedy_ablation`.

### B. RouteLLM swap BERT→MF (was: using retrained BERT with val_acc=0.389)

Use pretrained `routellm/mf_gpt4_augmented` checkpoint + threshold α=0.11593 (paper's 50%-strong calibration point for Chatbot Arena). Footnote the domain-shift caveat (trained on GPT-4 vs Mixtral; applied to Qwen pool).

### C. Avengers-Pro promoted to main-table baseline

Already wrapped upstream `SimpleClusterRouter`, trained artifact at `models/cara/avengers_pro_qwen/`. Knob: `performance_weight ∈ {0.25, 0.39, 0.53, 0.7, 0.8}`.

### D. Bench data integrity (CRITICAL)

`best-route-v3-test-500.jsonl` contaminated (289/500 in predictor train; 61/500 in predictor test; 150/500 in neither). Resampled to `best-route-v3-test-500-clean.jsonl` from `test_scored_filtered.jsonl` — zero overlap with predictor training, 100% deepeval coverage for Qwen-3B/7B. All Apr 15 morning sweep data is superseded; re-run on clean bench before quoting numbers.

### E. Methodology grounded in published precedents

Section §2 of the design doc tabulates what we inherit from RouterBench, MoE-CAP, RouterEval, LLMRouter library, and BentoML Performance Explorer.

### F. Filter naming consistency

Paper uses "CaraCDF"; runnable code uses `cara_tiered` — kept distinct in the methods section with explicit mapping. See §5.11 of design doc for the honest scope (TTFT/TPOT only today; full four-constraint bucket filter planned).

### G. Balance minimal-main + appendix-sensitivity

§4.5 of design doc. Main paper has one comparison figure (`w_balance=0` vs `=0.2`); appendix has full sweep.

### H. Failure-recovery testing NOT in scope

Confirmed not a blocker for this paper phase.

## Recommended Paper Experiment Split (with updates)

### 1. Main table — integrated system vs wrapper baselines

| # | Row | Knob |
|---|---|---|
| 1 | Cara-full | `w_quality` ∈ {0.1, 0.3, 0.5, 0.7, 0.9} |
| 2 | vllm_sr-wrapper | single point |
| 3 | RouteLLM-mf-wrapper | `α` ∈ {0.05, 0.11593, 0.2, 0.3, 0.5} |
| 4 | Avengers-Pro-wrapper | `performance_weight` ∈ {0.25, 0.39, 0.53, 0.7, 0.8} |
| 5 | BEST-Route-wrapper | `t` ∈ {0.3, 0.5, 0.6, 0.7, 0.8} |

Wrappers 3/4/5 paired with RR **and** shortest-queue dispatch (optional `llumnix_minus`).

### 2. Main ablation — integrated Cara ingredients

| # | Row |
|---|---|
| 1 | Cara-full |
| 2 | Cara-no-LPT (FIFO) |
| 3 | Cara-no-filter |
| 4 | Cara-point-filter (planned, contingent on §5.11) |
| 5 | Cara-no-quality-term |
| 6 | Cara-no-latency-term |
| 7 | Cara-no-cost-term |
| 7b | Cara-no-balance-term |
| 8a | Cara-simple-shortestQ |
| 8b | Cara-simple-llumnix |
| 9 | quality_greedy_ablation (renamed from revision-1 "best_route") |

### 3. Supporting study 1 — routing signal swap (inside fixed Cara shell)

Swap routing signal only: `cara_score` / `best_route_signal` / `vllm_sr` / `routellm_mf` / `avengers_pro` / `null_fixed`.

### 4. Supporting study 2 — filter comparison (inside fixed Cara shell)

Swap filter only: `none` / `cara_tiered` / `qlm` / `slos_serve` / `polyserve` / `timebill`.

### 5. Figures

(Full detail in §4.3 of design doc)

| Figure | Role |
|---|---|
| F1 | 2D Pareto cost vs quality (headline) |
| F2 | 2D Pareto cost vs mean E2E latency (partner) |
| F3 | CAP radar per router |
| F4 | Goodput bar chart (constraint) |
| F5 | Throughput bar chart |
| F6 | Memory-variance time-series (balance) |
| F7 | w_balance sensitivity (appendix) |
| F8 | Routing overhead decomposition |
| Table 1 | Full knob-sweep dump (appendix) |

## Execution Order (unchanged intent)

1. Clean bench resample (task #56) — ~5 min.
2. RouteLLM BERT→MF swap — ~1 h.
3. BEST-Route training pipeline — ~3-4 h (parallelized with doc work).
4. Monitor.py sweep wrapper — ~30 min.
5. Main + ablation + supporting-study sweeps on clean bench — ~60-90 min cluster time.
6. Aggregation + figure generation — ~1-2 h.
7. Codex review round 2 on this design doc.

## Recommended Workload / Evaluation Arrangement

### Workload

**One** workload in Phase 1 — the clean 500-prompt resample. Preserves the 7-dataset distribution (gsm8k, squad, beaver_tails, code_ultra_feedback, lmsys, reward_bench, mix_instruct). "Hard-skewed" subset optional in Phase 2 if space allows.

### SLO tiers

Balanced + strict in main paper; lenient for appendix sanity. Exact thresholds in §5.6 of design doc.

### Load points (revision 3)

Phase 1 (preflight only): single fixed λ = 10 req/s, clean bench, monitor.py deployed, scripts exercised.
Phase 2 (paper-facing): **arrival-rate λ sweep is the primary load axis.** Pilot → frozen grid (typically {4, 8, 12, 16, 20, 24, 32}). Anchor rates (low/knee/stress) predeclared per design §4.6. Sweep-scope layering caps total at ~127 runs.

## What Claude Should Review/Change

Handled in this revision:

1. ✅ `Cara-full` presented as the integrated system.
2. ✅ Main table = integrated-system comparison against wrappers.
3. ✅ Ablation = integrated-objective ingredient study.
4. ✅ Signal-swap + filter-swap as supporting studies.
5. ✅ BOute moved to related work only (not a target baseline).
6. ✅ Deepeval as main quality metric (via pre-computed lookup).
7. ✅ No failure-recovery testing required.
8. ✅ Bench data integrity fixed (resample from clean test split).
9. ✅ BEST-Route treated as real baseline with stripped-BoN training pipeline.
10. ✅ RouteLLM switched to MF + pretrained + published threshold.
11. ✅ Avengers-Pro promoted to explicit main-table row.
12. ✅ Figure plan locked with MoE-CAP-Dashboard radar helpers.

## Short Version

1. `Cara-full` is the system.
2. Wrapper baselines: RouteLLM-mf, Avengers-Pro, vllm-SR, BEST-Route.
3. Bench is resampled clean; predictor is valid.
4. Methodology inherits RouterBench Pareto + MoE-CAP radar; cost model is per-token MaaS.
5. Balance is minimal in main, full sensitivity in appendix.
6. Figures: 2D Pareto × 2 + CAP radar + goodput + throughput + memory-time-series + sensitivity + overhead + table.
