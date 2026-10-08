# Action Plan — March 27, 2026 (End of Session)

## Current Resource Status

| Resource | Status | Remaining Time | Current Task |
|----------|--------|---------------|--------------|
| **CloudLab 2×A30** | Ready, smoke test done | ~1.5 days | Idle — can rerun with fixes |
| **Vast.ai RTX 8000** | Running | ~1.5 days | Qwen LoRA fused training |
| **CSD3 A100** | Available | INTR anytime, Q3 budget remaining | Fused models done |
| **GPU VM L40S** | Available | Permanent | KNN eval done |
| **VPS** | Available | Permanent | Code hub |

---

## Code Implementation Tasks

### P0 — Correctness Fixes (do before any evaluation)

| # | Task | Where to run | Effort | Details |
|---|------|-------------|--------|---------|
| C1 | **Batch-wide quality/cost normalization + accumulated latency/balance** | VPS (code), CloudLab (test) | Medium | Two-phase: batch-wide for quality+cost (pre-loop from ModelEstimator), stats-initialized accumulated for latency+balance. ~50 lines in `cara_serve.py` |
| C2 | **Projected memory in balance score** | VPS (code), CloudLab (test) | Medium | Add `model_kv_config` to config (3B:55KB, 7B:115KB, 14B:393KB, 72B:328KB). Compute `projected_util = current + (prompt+E[output]) × kv_per_token / capacity`. |
| C3 | **Global SLO enforcement** | VPS (code) | Low | Add `global_slo` config. Filter checks `min(request.slo, global.slo)`. Track counters: overall_goodput, rso_goodput, slo_goodput. |
| C4 | **Save batch traces for Pareto analysis** | VPS (code) | Medium | Each batch saves: requests (prompt_tokens, output_tokens per model, quality per model), instance states, greedy assignment, all candidate scores. JSON per batch. |

### P1 — Evaluation Infrastructure

| # | Task | Where to run | Effort | Details |
|---|------|-------------|--------|---------|
| C5 | **RSO sampling in benchmark** | VPS (code) | Medium | Each field sampled independently at configurable rate (default 10%). Fields: budget_cost, ttft_slo_ms, tpot_slo_ms, quality_min. |
| C6 | **KNN-only model estimator** | VPS (code) | Medium | Single KNN call → point estimate + bucket distribution + quality. Config: k, index_type. |
| C7 | **Per-request detail saving in benchmark** | VPS (code) | Low | Save model chosen, all scores, filter reasons per request. |
| C8 | **Estimator sub-breakdown timing** | VPS (code) | Low | KNN embedding, FAISS search, ModernBERT forward — separate timers. |
| C9 | **Monitor process** | VPS (code), CloudLab (run) | Done | `monitor.py` — scrapes vLLM /metrics + system stats → CSV. |

### P2 — Model Training & Latency Predictor Fix

| # | Task | Where to run | Effort | Details |
|---|------|-------------|--------|---------|
| C10 | **Train TTFT XGBoost** | CSD3 INTR | 5 min | `--target ttft`. Feature `num_predicted_output_tokens` should have ~0 importance (auto-dropped, always 1024 in training). |
| C11 | **Train TPOT XGBoost** | CSD3 INTR | 5 min | `--target tpot`. Same — output token feature irrelevant for TPOT. |
| C12 | **Train E2E XGBoost with actual output** | CSD3 INTR | 5 min | `--target e2el --use-actual-output-len`. Uses `actual_output_tokens` (real values) as `num_predicted_output_tokens` feature. At serving time, receives `E[length]` from bucket classifier. |
| C13 | **Fix sidecar predictor** | VPS code | Medium | `cara_learned_predictor.py`: load separate TTFT + TPOT + E2E XGBoost models. `predict_ttft()` returns TTFT (not E2E). `predict_tpot()` returns TPOT XGBoost (not static lookup). Config: `xgboost_ttft.model_dir`, `xgboost_tpot.model_dir`, `xgboost_e2e.model_dir`. |
| C14 | **Fix sidecar API response** | VPS code | Low | Return `{"ttft": TTFT_model, "tpot": TPOT_model, "e2e": E2E_model}`. Scheduler uses TTFT/TPOT for filtering, E2E for scoring. |
| C15 | **Wire LSTM + roofline in sidecar** | VPS code | Medium | Support `predictor_type: "lstm"/"roofline"/"xgboost"` in sidecar config. For latency predictor ablation. |
| C16 | **Qwen LoRA training** | Vast (running) | In progress | Fused + per_model × 4 targets. ~6 hrs remaining. |
| C17 | **Long training (20-50ep) for winning models** | Vast or CSD3 queue | Medium | After short-run results confirm fused > per-model, retrain best targets with more epochs + early stopping. |

**CRITICAL BUG FOUND**: Current `predict_ttft()` returns E2E prediction from E2E XGBoost (mislabeled). Then scheduler computes `e2e = "ttft" + E[length] × tpot` = double-counting. Latency scoring is fundamentally broken until C13-C14 are fixed.

**Training feature field mapping**:
- Training: `actual_output_tokens` (real, 3-2200) replaces `num_predicted_output_tokens` (always 1024) via `--use-actual-output-len`
- Serving: sidecar receives `num_predicted_output_tokens = E[length]` from scheduler's bucket classifier
- Same feature name, different values. XGBoost handles this correctly because TTFT/TPOT auto-drop the feature (~0 importance).

**Paper framing (two-tier, NOT TimeBill)**:
- TimeBill approach: `E2E = predicted_length × TPOT_rate` (point estimate, length-dependent)
- Our approach: TTFT/TPOT for SLO **filtering** (no length dependency), E2E for soft **scoring** (acceptable noise)
- E2E XGBoost is a blackbox predictor (learns non-linear interactions), not simple `TTFT + length × TPOT`
- Ablation: compare E2E-direct vs TTFT+TPOT×length vs TimeBill-style

### P3 — Analysis Tools

| # | Task | Where to run | Effort | Details |
|---|------|-------------|--------|---------|
| C13 | **Brute-force Pareto evaluator** | VPS (code), any GPU (run) | Done | `eval_optimal_assignment.py` — enumerate all assignments for batch=2-6 × 18 instances. Compute Pareto frontier + greedy position. |
| C14 | **Normalization ablation variants** | VPS (code) | Low | Add `normalization_mode` param: `per_request`, `accumulated`, `topsis`. ~20 lines each. |
| C15 | **Baseline assignment simulators** | VPS (code) | Low | For Pareto comparison: `random_assign`, `round_robin`, `shortest_queue`, `quality_greedy` — simulate on saved batch traces. |

---

## Paper Changes

| # | Section | Change | Priority |
|---|---------|--------|----------|
| P1 | §3.2 Eq.1 | Remove memory capacity hard constraint. Note vLLM handles via preemption. | P0 |
| P2 | §3.1 Listing 1 | Update RSO API: ttft_slo_ms, tpot_slo_ms, budget_cost, quality_min | P0 |
| P3 | §3.3 | Added: load-dependent NP-hardness, 1-swap local optimality, brute-force for small batches | Done ✅ |
| P4 | §3.3 | Added: batch bubble overhead model (Eq. 5) | Done ✅ |
| P5 | §3.3 | Added: SBS differentiation (intra vs inter-instance batching) | Done ✅ |
| P6 | §4.2 | Added: fused multi-head architecture, PFS comparison, baseline table | Done ✅ |
| P7 | §4.3 | Clarify: E2E is optional soft filter, TTFT/TPOT are hard filters | P1 |
| P8 | §4.3 | Added: XGBoost latency prediction table | Done ✅ |
| P9 | §5.4 | Added: fused judge comparison table | Done ✅ |
| P10 | §3.3 | Tiered filtering discussion + ablation design | P1 |
| P11 | §3.3 | Normalization discussion: two-phase accumulated, cite TOPSIS/SAW/MCDM | P1 |
| P12 | §5 | KV bytes per model table (explains why balance matters) | P1 |
| P13 | Appendix | Pareto efficiency analysis: brute-force batch=2-6, 2D projections. Linked from §3.3. | P2 (after full eval) |
| P14 | §4.3 | 2D trade-off plots: each scheduler as one dot on (latency, quality) and (cost, quality) planes | P2 (after full eval) |
| P15 | §6 Conclusion | Update with E2E results | P2 (after full eval) |
| P16 | §4.3 | Fix latency predictor framing: E2E XGBoost (blackbox, uses output tokens) for scoring, TTFT/TPOT XGBoost (no output tokens) for SLO filtering. NOT TimeBill-style. Two-tier = different models for different purposes. | P0 |
| P17 | §4.3 | Add latency predictor ablation: E2E-direct vs TTFT+TPOT×length vs TimeBill vs LSTM vs roofline | P1 |

---

## Ablation Studies

| # | Ablation | Variables | Where to run | Priority |
|---|---------|-----------|-------------|----------|
| A1 | **Normalization** | per_request vs accumulated | CloudLab full eval | P1 |
| A2 | **KNN k** | k=5, 10, 20, 50 | GPU VM (offline) | P1 |
| A3 | **Tiered filtering** | STRICT vs TIERED vs RELAXED | CloudLab full eval | P1 |
| A4 | **RSO density** | 0%, 10%, 30%, 50% sampling rate | CloudLab full eval | P1 |
| A5 | **Scoring weights** | 9 configs (quality-first, latency-first, etc.) | CloudLab full eval | P1 |
| A6 | **Batch size** | 1, 4, 8, 16, 32 | CloudLab full eval | P1 |
| A7 | **Budget threshold θ** | 0, 0.3, 0.5, 0.7, 0.9 | CloudLab full eval | P2 |
| A8 | **LPT sort key** | max, min, mean | CloudLab full eval | P2 |
| A9 | **Estimator type** | Fused, PFS, KNN-only, per-model | CloudLab full eval | P1 |
| A10 | **Pareto efficiency** | Brute-force batch=2-6, 10 batches each | Post-hoc on traces | P2 |
| A11 | **KNN index** | numpy, faiss_flat, faiss_ivfpq | GPU VM (offline) | P3 |
| A12 | **Latency predictor** | E2E-direct, TTFT+TPOT×length, LSTM, roofline, static-TPOT | CloudLab full eval | P1 |
| A13 | **E2E vs decomposed for SLO** | E2E-filter vs TTFT/TPOT-filter: SLO compliance rate | CloudLab full eval | P1 |
| A14 | **Quality threshold** | quality_min=0.0, 0.3, 0.5, 0.7, 0.9 | CloudLab full eval | P1 |
| A15 | **Filter relax order** | ttft_first, budget_first, quality_first, tpot_first (tight SLOs) | CloudLab full eval | P1 |
| A16 | **Per-filter impact** | Disable one filter at a time vs all active vs RELAXED | CloudLab full eval | P1 |
| A17 | **RSO density** | 0%, 10%, 30%, 50%, 100% with per-field sampling | CloudLab full eval | P1 |

---

## Evaluation Figures Needed

| # | Figure | Type | Data Source |
|---|--------|------|-------------|
| F1 | **Throughput vs QPS** (all baselines) | Line plot | Main E2E eval |
| F2 | **Latency percentiles** (TTFT/TPOT/E2E) | CDF or bar | Main E2E eval |
| F3 | **Quality score distribution** per scheduler | Box plot | Main E2E eval |
| F4 | **2D trade-off plots** (lat vs qual, cost vs qual) | Scatter, each scheduler = 1 dot | Main E2E eval |
| F5 | **SLO goodput** (% meeting all SLOs) per scheduler | Bar chart | Main E2E eval |
| F6 | **Weight sweep Pareto** | Multi-dot per weight config | Weight sweep eval |
| F7 | **Batch size impact** | Line plot: overhead, throughput, quality vs batch size | Batch sweep eval |
| F8 | **Pareto frontier** (brute-force) | 2D projection, frontier + scheduler dots | Post-hoc analysis |
| F9 | **Filter rejection breakdown** | Stacked bar per scheduler | From scheduling_stats |
| F10 | **Resource utilization** (GPU memory, CPU) over time | Time series | monitor.py CSV |
| F11 | **Per-filter impact** | Grouped bar: goodput/quality/latency with each filter disabled | Per-filter ablation (A16) |
| F12 | **Filter order sensitivity** | Grouped bar: relaxation counts by order variant | Filter order ablation (A15) |
| F13 | **Quality threshold trade-off** | Line: quality vs rejection rate vs latency | Quality threshold sweep (A14) |

---

## Execution Plan

### Phase 1: Code Fixes — ✅ COMPLETE (March 27-28)
All code tasks done. Smoke test 45/45 PASS. Config refactor done. E2E pipeline 14/14 PASS.
- ✅ C1-C4: Two-phase normalization, projected memory, global SLO, batch trace saving
- ✅ C5-C8: RSO sampling, KNN estimator, per-request detail, normalization variants
- ✅ C15: LSTM + Roofline sidecars
- ✅ Paper: P10, P11, P12, P17
- ✅ Config refactor: scheduler_config + model_deployment_v2
- ✅ Experiment pipeline scripts: e2e, sensitivity, ablation, collect

### Phase 2: Remaining Before Full Eval (Now → April 25)
1. Finish remaining experiment scripts (budget_sweep, qps_scaling, rso_density, latency_predictor, lpt_sort, filter_order, quality_threshold)
2. Build analysis module (plot_e2e.py, plot_sensitivity.py, tables.py)
3. setup.sh: clone from GitHub instead of manual rsync
4. Upload final trained models to HF
5. Update configs with HF model paths
6. (Optional) TimeBill baseline

### Phase 3: Training Completion (parallel)
**Where**: Vast.ai + CSD3

1. Finish Qwen LoRA (Vast: similarity ~67%, judge + ref_score + per_model queued)
2. Final long retraining based on short training results
3. Upload to HF (single source of truth)
4. A2: KNN k ablation (GPU VM, offline)

### Phase 4: Full Evaluation (April 26 — May 4)
**Where**: CloudLab 18 nodes (8×A30 + 3×V100 + 5×P100 + 2×A100)

Day 1: `setup.sh` + `deploy_cara.py --deploy-services model_instance predictor monitor scheduler`
Day 2: `e2e/run.sh` — 8 baselines × 5 QPS × 2000 req
Day 3: `sensitivity/weight_sweep.sh` + `sensitivity/batch_sweep.sh`
Day 4: `sensitivity/budget_sweep.sh` + `ablation/filter_impact.sh` + `ablation/filter_order.sh` + `ablation/quality_threshold.sh`
Day 5: `ablation/normalization.sh` + `ablation/lpt_sort.sh` + `sensitivity/rso_density.sh`
Day 6: `ablation/estimator.sh` + `ablation/latency_predictor.sh`
Day 7: `collect/collect_results.sh` + `analysis/aggregate_monitor.py` + buffer
Day 8: Buffer / rerun failures
Day 9: P-D disaggregation on A100 NVLink (validated on A30 TCP March 28, need NVLink numbers)

### Phase 5: Paper Finalization (May 4-6)
1. `analysis/plot_*.py` + `analysis/tables.py` → paper figures/tables
2. Write evaluation section
3. Update conclusion with results
4. Submit NeurIPS May 6 (or polish for EuroSys May 14)

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|-----------|
| CloudLab reservation fails April 26 | Cannot run full eval | Fallback: subset eval on 2-4 nodes + simulation |
| Vast training OOMs on LoRA | Missing LoRA baseline | LoRA is ablation only, not critical |
| Normalization change degrades results | Need to debug | A/B test on smoke test first |
| Balance score fix changes routing | Smoke test results change | Expected — new results are more correct |
| NeurIPS May 6 too tight | Incomplete eval | Submit EuroSys May 14 instead |
