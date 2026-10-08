# Codex Review Package — 2026-04-15 PM (Revision 3 of Apr 15 design)

## Revision 3 delta (since rev-2 codex review)

1. **Arrival-rate λ is the primary Phase 2 load axis** (§4.6 + §5.8 of design doc). Sweep grid pilot-justified; anchor-rate rule (low / knee / stress) predeclared before data is seen.
2. **Sweep-scope layering added** (§4.6) to avoid knob × rate explosion:
   - λ-axis curves (F1-F3, F8): 5 rows × default knob × full λ = ~35 runs
   - Pareto at knee + stress (F4, F5): 5 rows × full knob × 2 anchors = ~50 runs (net ~40 additional)
   - CAP radar at knee (F3r): reuses F4 data, 0 additional
   - Ablations: 10 rows × 3 anchor rates = ~30 runs
   - Supporting studies: knee only, ~12 runs
   - **Total ~127 runs, 2.5-4 h cluster time**
3. **Default knob per router** predeclared (§4.6): Cara `w_quality=0.5`, RouteLLM-mf `α=0.11593`, Avengers-Pro `performance_weight=0.7`, BEST-Route `t` per-training-calibration (placeholder 0.6), vllm-SR single config.
4. **Phase 1 is preflight-only** (§7.A) — metric + monitor + script validation. Paper numbers come exclusively from Phase 2 (§7.B).
5. **Go/no-go checklist** (§7.C) added: hard blockers before Apr 26 cluster window opens, including a **BEST-Route soft deadline of Apr 24 morning** (48 h before Apr 26); if training fails by then, cite-only fallback in related work, row 5 removed from main table. No delay of the Apr 26 main sweep.
6. **Train/val hygiene rule** (§5.18) — retrained baselines (Avengers-Pro, BEST-Route, any RouteLLM recalibration) use `train_scored_filtered` for training + held-out slice of `test_scored_filtered` EXCLUDING the clean 500-bench for validation. Bench is inference-time evaluation only.

Revision 2 changelog preserved below.

---

## Revision 2 delta (historical, for context)

## Read order (in this directory)

1. `apr15_baseline_comparison_design.md` (revision 2) — design of record
2. `apr15_baseline_comparison_claude_handoff.md` (revision 2) — short brief
3. This file — review package, changelog, asks

Secondary docs sync'd with the new methodology:

- `../ROADMAP_TO_SUBMISSION.md` — top-level roadmap
- `../active/PAPER_UPDATES_TODO.md` — paper section list
- `../active/SUBMISSION_PLAN.md` — submission plan
- `../active/CARA_PROJECT_STATUS.md` — project status
- `../active/EXPERIMENT_PLAN_CLOUDLAB.md` — cloudlab experiment plan
- `../active/APR26_PREFLIGHT.md` — Apr 26 Phase 2 preflight
- `../smoke_test_apr_13/BASELINE_IMPL_TRACKER.md` — implementation tracker

## Changelog since revision 1 (Apr 15 AM) — preserved below, superseded where overlapping

### Methodology
*(Revision-2 delta list — preserved for audit; superseded where revision 3 overrides above. Phase 1/2 framing and load axis are both superseded by revision 3.)*

1. **Axis roles re-framed** per user guidance: Cost + Quality + Mean E2E latency are continuous regression targets that the scorer optimizes; TTFT / TPOT / Budget SLOs are constraints enforced by the filter and reported aggregated as goodput. Balance is scheduler-specific and split between minimal-main + appendix-sensitivity.
2. **Methodology grounded in published precedents** — RouterBench (Hu 2024) for knob-sweep Pareto + AIQ, MoE-CAP (Jiang 2025) for CAP radar, RouterEval (Huang 2025) for scaling context. LLMRouter and BentoML Explorer cited as extensibility / dashboard precedents.
3. **Radar helpers adopted directly from MoE-CAP-Dashboard** (Apache 2.0) — `normalize` / `normalize_cost` for per-axis native-unit scaling to [20, 100] with inverted axes for lower-is-better metrics.
4. **Hypervolume dropped from main figures** — 2D Pareto + radar + goodput + overhead suffice. HV kept as an optional appendix scalar if reviewers ask.
5. **Cost model is per-token vendor pricing** (MaaS framing), not MoE-CAP's hardware-TCO formula.
6. **Evaluation quality via pre-computed deepeval lookup** from `test_scored_filtered.jsonl` (HF `asdwb/cara_model_estimator`). RouterBench-style. No live judge calls for the main metric. **Code status: methodology locked but code patch pending** — current `aggregate_e2e_results.py` still emits `predicted_quality_mean` as placeholder with deferred-deepeval note. Task #54 re-scoped to land the lookup join (~30-min patch, gated on the re-run with clean bench).

### Baselines
7. **BEST-Route is now a real baseline**, trained on Qwen via a stripped-down pipeline (no Best-of-N sampling, no proxy reward model — direct armoRM pair-wise labels for Qwen-3B vs 7B, DeBERTa-v3-small backbone). Replaces the revision-1 `best_route.py` argmax-quality implementation, which was a misattribution and is renamed to `quality_greedy_ablation`.
8. **RouteLLM switched BERT→MF** — paper-recommended `mf` variant + pretrained HF checkpoint + published threshold α=0.11593. Drops the retrained-BERT version with val_acc=0.389.
9. **Avengers-Pro promoted** to explicit main-table row with `performance_weight` knob sweep (paper's published α values).
10. **Wrappers paired with both RR and shortest-queue dispatch** — revision 1 only tested RR, which was weak-baseline. Steelman with load-aware dispatch required for defensible comparison.

### Data integrity (CRITICAL)
11. **Bench file was contaminated** — `best-route-v3-test-500.jsonl` overlapped 289/500 prompts with the predictor's training split. Resampled to `best-route-v3-test-500-clean.jsonl` from `test_scored_filtered.jsonl` (3,634 clean held-out prompts). **All Apr 15 morning sweep data is superseded.**
12. **Predictor itself is valid** — clean 80/20 train/test split at the prompt level (confirmed via `test_scored ∩ train_scored = 0`). Only the bench file was bad.

### Execution + observability
13. **monitor.py to be deployed on all 4 nodes during the upcoming knob-sweep re-run** — pre-existing tool at `block/exp/cara/monitor.py`. Provides per-instance KV-util time-series for the memory-variance figure and process-level CPU/RSS for honest routing-overhead breakdown. **Note:** the Apr 15 morning sweeps ran *without* monitor.py — that's the data gap motivating the rerun. Sweep wrapper script is task #61 (queued).
14. **Phase 2 (Apr 26) 18-node** is now an explicit §7.B in the design doc. *(Revision 3 update: Phase 2 is the **paper-facing** run; Phase 1 is preflight only. See revision-3 deltas above.)*

## Full decisions-locked list

| # | Decision | Value |
|---|---|---|
| 1 | BEST-Route | Trained in (stripped Best-of-N) |
| 2 | Quality bench source | Resample 500 from `test_scored_filtered.jsonl` |
| 3 | Latency metric | Mean E2E |
| 4 | Wrapper dispatcher | RR + shortest-queue (`llumnix_minus` optional) |
| 5 | Balance scope | Minimal main + sensitivity appendix |
| 6 | Apr 26 | Included as §7.B Phase 2 |
| 7 | Knob grid | 5 values per router |
| 8 | monitor.py | 5 s sampling, all nodes |
| 9 | HV normalization | Dropped HV; radar uses MoE-CAP-Dashboard helpers |
| 10 | LLMRouter adapter | Deferred post-Apr 17 |
| 11 | Cost model | Per-token vendor pricing (MaaS) |
| 12 | Quality metric | Pre-computed deepeval lookup (code patch pending, task #54) |
| 13 | Primary Phase 2 load axis | Arrival-rate sweep λ (pilot → {4, 8, 12, 16, 20, 24, 32}) |
| 14 | Sweep-scope layering | §4.6: λ-axis curves at default knob + Pareto at 2 anchor rates + ablations at 3 anchor rates + supporting at knee |
| 15 | Phase 1 scope | Preflight only (metric/monitor/script validation); no paper numbers |
| 16 | BEST-Route deadline | Apr 24 morning; cite-only fallback if training fails |
| 17 | Train/val hygiene | Retrained baselines trained on `train_scored_filtered`, validated on held-out slice EXCLUDING the 500-prompt clean bench |

## Open questions

None currently blocking. Potential codex flags:

- Whether BEST-Route without Best-of-N is substantially different from RouteLLM-MF (both become binary threshold routers on our 2-model pool). We call this out explicitly in §5.15 of the design doc.
- Whether the 4-A30 cluster is sufficient to show balance benefit (probably not — balance is mainly a Phase 2 story; Phase 1 shows the methodology works).

## Specific review asks

1. **Citation set sufficiency for related work.** Is the set (RouterBench + RouterEval + MoE-CAP + LLMRouter lib + BentoML + Hybrid-LLM + BEST-Route + RouteLLM + Avengers-Pro + BOute + vllm-SR) complete and well-positioned, or missing obvious neighbors?

2. **Fair-comparison methodology.** Is the knob-sweep convex-hull Pareto with goodput as a constraint defensible? Does pairing wrappers with both RR and shortest-queue adequately steelman them? Any reviewer objection I'm missing?

3. **Data-integrity narrative clarity.** Is §5.14 of the design doc clear that the predictor is valid and only the bench file was bad? Is the resample procedure ("stratified sample 500 from `test_scored_filtered`") sufficient, or do we need more detail (per-dataset stratification, random seed, reproducibility)?

4. **BEST-Route stripped-BoN plan.** Is the adapted pipeline (direct armoRM pair-wise labels + DeBERTa-v3-small + threshold knob) an acceptable "best-effort implementation of BEST-Route on Qwen" given no pretrained weights exist, or should we cite-only?

5. **Balance scope split.** Is minimal-main + sensitivity-appendix the right call, or should balance be fully elevated to the main Pareto? Tradeoff: elevating means 4 axes on Pareto (awkward visually), not elevating means one of Cara's ingredient contributions only gets appendix space.

6. **Figure stack.** 8 figures (F1-F8) + Table 1 — reasonable for a 9-page NeurIPS paper? If too many, what's the cut list? (My suggestion: F6+F7 could merge into one compound figure for balance.)

7. **Phase 1 vs Phase 2 split.** ~~Is using Phase 1 (4-A30) for primary paper numbers plus Phase 2 (18-node) for validation the right risk stratification~~ **Resolved in revision 3**: Phase 1 is preflight-only; all paper numbers come from Phase 2 18-node. Codex round-2 review agreed. Remaining reviewer-worthy question is whether the ~127-run Phase 2 matrix + ~3-4h cluster time is the right risk budget inside the Apr 26 - May 6 window.

## Verification artifacts (local to VPS, for reviewer spot-checks)

- `data/cara/scored/test_scored_filtered.jsonl` — 3,634 clean held-out prompts with 3B+7B deepeval (22 MB, mirrored from HF `asdwb/cara_model_estimator`)
- `data/cara/train_scored_filtered.jsonl` — 14,919 predictor training prompts
- `data/cara/best-route-v3-test-500.jsonl` — **contaminated** bench (legacy, kept for audit)
- **`data/cara/best-route-v3-test-500-clean.jsonl`** — **clean resample, written 2026-04-15 19:30 UTC** by `cara_paper/smoke_test_apr_13/scripts/resample_clean_bench.py` (500 unique prompts, 0 predictor-train leak, 100% 3B+7B deepeval coverage, stratified across source datasets). Task #56 complete.

Quick verification commands:
```bash
python3 -c "
import json
scored = {json.loads(l)['prompt'] for l in open('data/cara/scored/test_scored_filtered.jsonl')}
train = {json.loads(l)['prompt'] for l in open('data/cara/train_scored_filtered.jsonl')}
print('test ∩ train =', len(scored & train))  # expect 0
"
```

## File paths relevant to review

| Concern | File(s) |
|---|---|
| Scorer implementation | `block/global_scheduler/cara/cara_serve.py:763-1134` |
| Four-term scoring weights | `cara_serve.py:780-783, :1117-1122` |
| Filter implementation | `block/global_scheduler/cara/filters/cara_cdf_filter.py` + legacy inline at `cara_serve.py:906-960` |
| Router implementations | `block/global_scheduler/cara/routers/{best_route,routellm,avengers_pro,vllm_sr,...}.py` |
| Monitor.py | `block/exp/cara/monitor.py` |
| Bench | `block/benchmark/cara/benchmark_serving.py` |
| Resample script (landed 2026-04-15 19:30 UTC) | `cara_paper/smoke_test_apr_13/scripts/resample_clean_bench.py` (clean bench file generated + verified) |
| Aggregation | `cara_paper/smoke_test_apr_13/scripts/aggregate_e2e_results.py` |
| Sweep | `cara_paper/smoke_test_apr_13/scripts/run_cara_full_ablation.sh` + `run_main_table.sh` |
