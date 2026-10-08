# Cara CloudLab Experiment Plan — April 26 to May 6, 2026

## Overview
- **Cluster**: 18 instances across 4 GPU types (2×A100/72B, 3×V100/14B, 5×A30/7B+3B, 5×P100/3B, 3×A30 coordinator)
- **Deadline**: NeurIPS May 6 submission
- **Total experiment time**: ~55 hours across 10 days
- **Methodology**: **frozen per revision 2 (Apr 15 PM)** — see `cara_paper/codex/apr15_baseline_comparison_design.md` §7.B (Phase 2 plan)

---

## ⚠ Revision-2 deltas (Apr 15 PM) relevant to this doc

- **Baselines now include BEST-Route** (custom-trained Qwen+armoRM), RouteLLM-mf (pretrained, not retrained BERT), Avengers-Pro (with α sweep), vllm-SR. See design §3.2 for knob ranges.
- **Wrapper dispatchers**: each wrapper paired with RR + shortest-queue. `llumnix_minus` optional.
- **Bench** resampled to `best-route-v3-test-500-clean.jsonl`. Supersedes contaminated legacy bench.
- **Phase 2 (paper-facing) primary load axis**: **arrival-rate λ sweep** (pilot-justified grid ≈ {4, 8, 12, 16, 20, 24, 32}); anchor rates (low/knee/stress) predeclared per design §4.6. Sweep-scope layering caps total at ~127 bench runs. Phase 1 (4-node, Apr 15-17) is **preflight only** at fixed λ=10 — no paper numbers. Balance CV time-series + CAP radar on per-GPU-type cost also produced in Phase 2.
- **monitor.py** deployed on every node during every sweep (5 s sampling).

---

## Baselines — corrected structure (Apr 15 PM revision 2)

> See `cara_paper/codex/apr15_baseline_comparison_design.md` for the authoritative, codex-reviewed design. The live task queue for the toggles needed to execute the ablations is #51–#55.

**Main table (Cara-full vs practical wrapper baselines):**

| # | Row | Role | Knob |
|---|---|---|---|
| 1 | **Cara-full** (`--scheduling cara`) | integrated online batch scheduler — LPT sort + bucket feasibility filter + four-term `(model, instance)` scoring | `w_quality` sweep |
| 2 | `vllm_sr-wrapper` | external peer scheduler (Envoy + extproc) | single point |
| 3 | `RouteLLM-wrapper (mf)` | pretrained `mf` variant + published α=0.11593; paired with RR + shortest_queue | α sweep |
| 4 | `Avengers-Pro-wrapper` | upstream `SimpleClusterRouter`, Qwen-trained; paired with RR + shortest_queue | `performance_weight` sweep |
| 5 | `BEST-Route-wrapper` | DeBERTa-v3-small on Qwen + armoRM pair-wise labels (no Best-of-N); paired with RR + shortest_queue | t sweep |

Retired: `best_route-wrapper` quality-argmax (renamed to `quality_greedy_ablation`, moved to ablations). "Strongest composed baseline" superseded by wrapper + load-aware-dispatch pairings above.

**Primary ablation table (Cara-full ingredients):** `Cara-no-LPT`, `Cara-no-filter`, `Cara-point-filter` (planned — requires the unified bucket filter; see design §5.11), `Cara-no-quality-term`, `Cara-no-latency-term`, `Cara-no-cost-term`, **`Cara-no-balance-term`** (7b: `w_balance = 0`), `Cara-simple-shortestQ` (8a), `Cara-simple-llumnix` (8b).

**Supporting studies:** (a) routing-signal swap inside a fixed pluggable shell (`best_route`, `vllm_sr`, `routellm`, optional `avengers_pro`, plus `cara_score` as reference); (b) filter swap inside a fixed Cara shell (`none`, `cara_tiered`, `qlm`, `slos_serve`, `polyserve`, `timebill`).

**Sanity-check table:** standalone SR (Envoy end-to-end) — latency + throughput + quality only; goodput blank with footnote (SR is SLO-blind).

**Appendix only:** vllm-native latency floor + SR internal overhead via Jaeger (task #49).

### Retired (do NOT run)

- ~~Flat 8-baseline list (Random / Round-Robin / Shortest-Queue / Quality-Greedy / Cost-Greedy / Length-Aware (Block) / TimeBill-style / Cara-full)~~ — collapses into the structured main + supporting-study tables above. Random / RR / Shortest-Queue / Quality-Greedy / Cost-Greedy re-appear as ingredient ablations or pipeline-dispatcher variants, not as top-level rows.
- **TimeBill-style** remains available as a filter-swap option (supporting study b). The point-estimate-vs-bucket filter comparison moves into `Cara-point-filter` (ablation row 4) once the unified bucket-feasibility filter lands.

---

## Metrics

| Category | Metric | Description |
|----------|--------|-------------|
| **Latency** | TTFT P50/P95/P99 | Time to first token distribution |
| | TPOT P50/P95/P99 | Per-token decode latency |
| | E2E P50/P95/P99 | End-to-end per request |
| **Goodput** | SLO goodput | Requests meeting ALL SLOs per second |
| | Budget goodput | % requests meeting token budget |
| | Latency goodput | % requests meeting TTFT+TPOT SLOs |
| **Quality** | Avg quality score | Mean quality across served requests |
| | Best-model hit rate | % routed to actual best model |
| **Cost** | Total tokens consumed | Token cost across all requests |
| | Goodput per cost | Cost efficiency |
| **Balance** | Memory utilization CV | Coefficient of variation across instances |
| **Overhead** | Scheduling decision time | Per-request scheduling overhead |

---

## Experiment Schedule

### Day 1: Deployment + Smoke Test (~8h)
- Deploy 18 vLLM instances (setup.sh)
- Deploy predictor sidecars with trained models
- Load learned predictors into cara_serve.py
- Smoke test: 100 requests, verify all metrics collected
- **Verify `actual_output_tokens` in sidecar output (CRITICAL — check immediately)**

### Day 2: Main Comparison (~8h) — structured per Apr 15 design revision 3
- Main table rows (5): Cara-full + vllm_sr-wrapper + RouteLLM-wrapper(mf) + Avengers-Pro-wrapper + BEST-Route-wrapper. Primary ablation rows (10) + supporting studies. Gate on #54 deepeval-lookup + #60 monitor wrapper + #61 p95 fix + #63 sweep-script rev-3 landing (see design §7.C).
- **Primary load axis = arrival-rate λ** (not a single representative QPS). Pilot on Day 1 → freeze grid ≈ {4, 8, 12, 16, 20, 24, 32}. Anchor rates (low/knee/stress) picked by predeclared rule (§4.6).
- Sweep-scope layering (caps runs at ~127 total): λ-axis curves at default knob (5 rows × 7 λ ≈ 35 runs), Pareto at knee + stress (5 rows × 5 knobs × 2 anchors ≈ 50 runs), ablations at 3 anchors (10 × 3 = 30), supporting at knee (~12).
- Collect per row: goodput @ SLO (balanced + strict), deepeval quality via pre-computed lookup (#54), throughput, cost per successful request, TTFT/TPOT/E2E distributions, Cara-full `scheduling_overhead_breakdown`, monitor.py-sourced process CPU/RSS for wrappers.

### Day 3: Per-term ablations + supporting studies (~10h)

Superseded: ~~9-combo weight sweep.~~ Replaced by **per-term zero-out** ablations (rows 5/6/7/7b of the primary ablation table) which cleanly isolate each scorer term without crossing them:
- `Cara-no-quality-term` (`w_quality = 0`)
- `Cara-no-latency-term` (`w_latency = 0`)
- `Cara-no-cost-term` (`w_cost = 0`)
- `Cara-no-balance-term` (`w_balance = 0`)
- Plus `Paper default` / `Cara-full` reference row (all four weights at their tuned values).

Supporting filter-swap study (fixed Cara shell, swap filter only): `none`, `cara_tiered`, `qlm`, `slos_serve`, `polyserve`, `timebill`.

Supporting routing-signal swap study (fixed pluggable shell, swap router only): `best_route`, `vllm_sr`, `routellm`, optional `avengers_pro`, plus `cara_score` as internal reference.

**Batch size sweep** removed from the main plan — Cara dispatches LPT-sorted requests one-by-one, so "batch size" is no longer a tuning axis. If needed for an LPT-related ablation, define it clearly as "queue-snapshot size" and document the intent.

### Day 4: Budget + Quality + Filter Strategy (~10h)
**Budget threshold sweep**: θ = 0 (off), 0.3, 0.5, 0.7, 0.9 × 3 QPS
**Budget size sweep**: budget_tokens = 64, 128, 256, 512, 1024 × 3 QPS
**Quality threshold sweep**: quality_min = 0.0, 0.3, 0.5, 0.7, 0.9 × 3 QPS
**RSO density sweep**: rate = 0%, 10%, 30%, 50%, 100% × 3 QPS

**Filter order ablation** (tight SLOs to ensure all filters active):
- `ttft_first`: ["ttft", "tpot", "quality", "budget"] (default)
- `budget_first`: ["budget", "ttft", "tpot", "quality"]
- `quality_first`: ["quality", "budget", "ttft", "tpot"]
- `tpot_first`: ["tpot", "ttft", "quality", "budget"]

**Per-filter impact** (disable one filter at a time, moderate SLOs):
- All filters active (baseline)
- No budget filter (threshold=0)
- No TTFT filter (SLO=∞)
- No TPOT filter (SLO=∞)
- No quality filter (min=0)
- No filters (RELAXED mode)

**Burstiness test**: 4 burst levels × 3 main-table rows — `Cara-full`, `vllm_sr-wrapper`, and one simple-assignment ablation (`Cara-simple-shortestQ`). Retired: ~~Shortest-Queue and Quality-Greedy as top-level baselines~~ (folded into dispatcher variants / wrappers).

### Day 5: Homogeneous Ablation (~6h)
**Setup**: 8× A30 nodes with single model (Qwen2.5-7B), like Block's evaluation
**Experiments** (per corrected Apr 15 design):
- `Cara-homogeneous-full` — integrated scheduler on single-model cluster
- `Cara-homogeneous-no-LPT` — FIFO ordering, everything else held (#51 toggle)
- `Cara-homogeneous-no-balance-term` — `w_balance = 0` (#53 per-term ablation)
- `Cara-homogeneous-simple-shortestQ` — no scoring, LPT + shortest-queue (#52 toggle)
- Compare against Block's simulation-based single-model system as a separate reference row.

Retired: ~~"Without batch scheduling"~~ (Cara has no batch-level joint optimization to ablate — LPT dispatches one-by-one). ~~"batch_size=8" sweep~~ (no batch-size concept). ~~"`w_u`" term labeling~~ — the cluster-memory-balance weight is named `w_balance` in the code and the `Cara-no-balance-term` ablation pins it to 0.

**LPT sort key ablation** (heterogeneous only — on full cluster, not homogeneous):
- `lpt_sort_key=max` (default): sort by max predicted length across all candidate models
- `lpt_sort_key=min`: sort by min predicted length (optimistic — assumes best-case model)
- `lpt_sort_key=mean`: sort by mean predicted length across models
- In homogeneous setting (single model), all three are identical — this ablation only matters for heterogeneous
- Config: set `slo_defaults.lpt_sort_key` to "max", "min", or "mean"
- Run at QPS 10,15,20 × 500 requests each, compare makespan + P95 latency + load balance

### Day 6: Noise Injection + Workload Mix (~8h)
**Predictor noise injection**:
- Add Gaussian noise to quality predictions (σ = 0.05, 0.1, 0.2)
- Add noise to latency predictions (σ = 10%, 25%, 50%)
- Measure goodput degradation → shows predictor accuracy requirements

**Workload mix**:
- Chat-heavy (80% LMSYS)
- Code-heavy (80% code_ultra_feedback)
- Safety-heavy (80% beaver_tails)
- Mixed (even distribution, default)

### Day 7: Buffer / Re-runs
- Re-run any failed experiments
- Additional experiments if time permits

### Day 8: Appendix Experiments (~5h)
**TimeBill comparison** (appendix):
- Same setup as Cara but use point-estimate E[length] × TPOT for budget check instead of P(tokens ≤ B)
- Compare budget goodput: probabilistic vs point-estimate

**Empirical competitive ratio**:
- For each batch, solve MILP offline (Gurobi/scipy) with ground-truth metrics
- Report Cara_score / Optimal_score across all batches

### Day 9: P-D Disaggregation A100 NVLink Test (~4h)

**Status**: A30 TCP feasibility **already validated** (March 28). Need A100 NVLink numbers.

**What's done** (March 28, CloudLab 2×A30):
- `cara_pd_serve.py` — full Cara P-D scheduler with ModelEstimator integration
- vLLM's built-in NixlConnector — **no lmcache needed**, no cherry-picking needed
- Cross-node results: Prefill=28ms, Decode=764ms, Transfer=1.2ms (analytical), Sched=20ms
- Bandwidth: 1.67 GB/s (10GbE TCP)
- Paper appendix §A.2 updated with formulation + results table

**What's needed on A100** (Day 9):
- **Setup**: 2× A100 nodes (NVLink), Qwen-72B or 14B with 1P1D
- **Software**: Same `cara_pd_serve.py` + `pd_config.json` with A100 hosts
- **Expected**: Transfer time drops from 1.2ms to <0.01ms (NVLink ~300 GB/s)
- **Test**: 500 requests, 3 QPS, compare P-D vs aggregated TTFT
- **Fix needed**: Benchmark client metrics mapping (TTFT/TPOT from P-D response)

**No prep needed**: Code is ready. Just change `pd_config.json` with A100 hosts and `bandwidth_key: "a100_nvlink"`.

Remaining buffer time for re-runs or additional experiments.

### Day 10: Final Data Collection (~4h)
- Sync all results
- Verify completeness
- Backup to VPS + CSD3

---

## Pre-CloudLab Preparation (March 26 — April 25)

### March 26-28: Finish Model Training
- [ ] CSD3 batch job completes (log-transform 100ep, bucket 100ep)
- [ ] Judge_class retries complete
- [ ] Collect all results, run unified evaluation
- [ ] Select winning models for each predictor

### March 29-31: Fix Critical Bugs
From code_review_march24.md (9 CRITICAL):
- [ ] #4: Instance.py output_tokens (merge fix to cara-small-a100 branch)
- [ ] #7: host_configs.json (add all 18 hosts for new CloudLab reservation)
- [ ] #1: train_llm_predictor.py indentation (already fixed)
- [ ] #18: run_e2e_benchmark.py QPS control (broken asyncio)
- [ ] #8: deploy_cara.py argparse bool
- [ ] #9: cara_serve.py sampling param override
- [ ] #2: train_bert_predictor.py data leakage (only when no --test-input)
- [ ] #3: judge_class mapping (already handled by safety-aware scoring)
- [ ] #5: generate_latency_benchmark.py dead code (already fixed)

### April 1-5: Code Changes — retired / landed
Historical entries from the pre-Apr 15 runbook:
- ~~Add `w_u · ΔBalance` to scoring in `_schedule_cara`~~ — landed (scorer emits all four weighted terms; see `cara_serve.py:780,1117`).
- ~~Add LPT sort before per-request scheduling~~ — landed as the LPT step in `_schedule_cara_batch`.
- ~~Track mean cluster utilization during batch scheduling~~ — landed via `_compute_mean_utilization`.
- ~~Integrate winning models into `cara_learned_predictor.py`~~ — landed.
- ~~Implement load-aware adaptive batch sizing~~ — obsolete (no batch-level joint optimization anymore; LPT-sorted queue dispatches one-by-one).
- ~~Implement TimeBill-style baseline~~ — landed as a filter-swap option in Supporting Study 2.
- ~~Write all experiment automation scripts in `block/exp/cara/`~~ — pluggable-pipeline scaffolding landed under `block/global_scheduler/cara/`; Cara-full-specific toggles are live tasks #51 / #52.

Live code changes still needed (tasks #51–#55 in the tracker):
- [ ] #51 LPT on/off toggle in `_schedule_cara_batch` (for `Cara-no-LPT` ablation).
- [ ] #52 `assignment_strategy={scoring, shortest_queue, llumnix_minus}` toggle feeding all instances across models (for `Cara-simple-shortestQ` / `Cara-simple-llumnix` ablations).
- [ ] #53 verify `/v1/config` accepts per-term zero weights including `w_balance`.
- [ ] #54 deepeval post-hoc scoring pass.
- [ ] #55 main + ablation sweep script.

### April 6-10: CSD3 Dry Run
- [ ] Deploy 2 A100s on CSD3 via batch job
- [ ] Run Qwen-3B + Qwen-7B (2 models, 2 instances)
- [ ] Test full pipeline: deploy → predict → schedule → collect
- [ ] Verify experiment scripts work end-to-end
- [ ] Fix any issues

### April 11-12: Fix Dry Run Issues

### April 13-16: Block Experiments (separate CloudLab)
- SoCC rebuttal experiments on A30 + A100 nodes

### April 17-20: Paper Writing
- Methodology section (training pipeline, objective function)
- EuroSys format (ACM template, 12 pages)

### April 21-25: Final Prep
- [ ] Update host_configs.json for new CloudLab nodes
- [ ] P-D disaggregation prep: cherry-pick vLLM endpoints onto upstream, install lmcache+nixl, test on CSD3 A100 if possible
- [ ] Pre-download all model weights on coordinator
- [ ] Experiment runbook (exact commands per experiment)
- [ ] Pre-flight checklist
- [ ] Dry run on P100 (if CloudLab on-demand available)

---

## Experiment Automation Scripts

### Implemented March 28 — NOTE: weight_sweep.sh and batch_sweep.sh RETIRED in Apr 15 PM revision 2
```
block/exp/cara/
├── e2e/run.sh                        ✅ 7 baselines × N QPS (tested 14/14 PASS) — superseded by run_main_table.sh + run_cara_full_ablation.sh under cara_paper/smoke_test_apr_13/scripts/
├── sensitivity/
│   ├── weight_sweep.sh               ⚠ RETIRED — replaced by per-term zero-out ablation rows 5/6/7/7b in design §3.3
│   └── batch_sweep.sh                ⚠ RETIRED — adaptive batch_queue obsoletes static batch sizes (single-request dispatch from LPT-sorted queue)
├── ablation/
│   ├── normalization.sh              ✅ 4 normalization modes
│   ├── estimator.sh                  ✅ default/knn/pfs (with restart)
│   └── filter_impact.sh              ✅ per-filter disable (6 configs)
├── collect/
│   └── collect_results.sh            ✅ Gather results + monitor CSVs
└── analysis/
    └── aggregate_monitor.py          ✅ Cluster-wide resource metrics
```

### Remaining (build before April 26) — NOTE: qps_scaling.sh retired in rev 3; use sweep_with_monitor.sh (#60) for arrival-rate sweeps
```
├── sensitivity/
│   ├── budget_sweep.sh               — budget_tokens + threshold θ (optional, appendix)
│   ├── qps_scaling.sh                ⚠ RETIRED — arrival-rate λ sweep is the primary Phase 2 axis; handled by scripts/sweep_with_monitor.sh (#60) + #63 updated run_main_table.sh
│   └── rso_density.sh                — RSO rate 0%-100% (optional, appendix)
├── ablation/
│   ├── latency_predictor.sh          — xgboost/roofline/static_tpot (restart)
│   ├── lpt_sort.sh                   — max/min/mean sort keys
│   ├── filter_order.sh               — 4 relax_order variants
│   └── quality_threshold.sh          — quality_min 0.0-0.9
└── analysis/
    ├── plot_e2e.py                   — F1-F5 paper figures
    ├── plot_sensitivity.py           — F6-F7 paper figures
    ├── plot_ablation.py              — ablation comparison
    ├── plot_filter.py                — F9, F11-F13 filter analysis
    ├── plot_resources.py             — F10 GPU/CPU time series
    └── tables.py                     — LaTeX tables
```

### Config Structure (March 28 refactor)
```
block/config/cara/
├── model_deployment_v2.json          — deploy_cara generates (instances with embedded host info)
├── scheduler_config_smoketest.json   — scheduler: weights, SLOs, estimator
├── scheduler_config_{default,knn,pfs}.json — estimator variants
├── predictor_config_smoketest_v2.json — sidecar-only: predictor_type, model paths
└── predictor_config_smoketest_{knn,roofline,static_tpot}.json — predictor variants
```
Scripts use env vars: `MODEL_DEPLOY`, `SCHEDULER_CONFIG`, `PREDICTOR_CONFIG`

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Deployment fails (day 1) | setup.sh tested on March run, 9 CRITICAL bugs fixed |
| Experiment script bugs | CSD3 dry run catches them before CloudLab |
| CloudLab node failures | 3 A30 buffer nodes, can redistribute |
| Results don't show clear wins | Knob sweep per router + arrival-rate sweep shows Pareto frontier (revision-3 design §4.6); some operating-point + knob-value combinations will win |
| Not enough time | Day 7+9 buffer; cut noise injection + workload mix if needed |
| EuroSys format not ready | Start conversion April 17, 12 pages gives more room |
| P-D disaggregation fails on A100 | A30 results already in appendix; A100 numbers are bonus |

## Priority Order (revision 3 — if time runs short)
1. **Must have**: main-table 5-row × default-knob λ-axis curves (F1-F3, F8), Pareto at knee + stress (F4, F5), ablations at 3 anchor rates — per design §4.6.
2. **Should have**: CAP radar at knee (F3r, reuses F4 data), goodput bar (F4v2), throughput bar (F5v2), memory-variance time-series (F6), overhead stack (F8).
3. **Nice to have**: w_balance sensitivity sweep (F7), filter-swap supporting study, signal-swap supporting study.
4. **Appendix bonus**: burstiness, noise injection, workload mix (hard-skewed subset), P-D A100 NVLink, TimeBill filter, competitive ratio.

Retired: "weight sweep (day 3)" — superseded by per-term zero-out ablation rows 5/6/7/7b + knob sweep per router at anchor rates.

## Post-CloudLab (May 7-14)
- Paper rewrite in EuroSys format (ACM template, 12 pages)
- Fill evaluation results
- Submit May 14
