# Cara — Roadmap to Submission

**Target**: NeurIPS 2026 (May 6) primary / EuroSys 2027 (May 14) backup
**Last updated**: April 15, 2026 PM (**revision 3** — arrival-rate sweep primary, Phase 1 preflight-only, Phase 2 paper-facing, BEST-Route Apr 24 soft deadline, go/no-go checklist landed)

---

## Evaluation design — pinned (Apr 15 PM, revision 2)

**Source of truth**: `cara_paper/codex/apr15_baseline_comparison_design.md` (revision 2) + `cara_paper/codex/CODEX_REVIEW_PACKAGE.md` (changelog). Brief summary below.

**Cara-full (paper system)** = integrated online batch scheduler:
1. LPT-sort pending queue by predicted output length.
2. Dispatch one request at a time in sorted order.
3. Per-request: bucket-based probabilistic feasibility filter (`CaraCDFFilter` today covers TTFT/TPOT; four-constraint filter planned) + multi-objective joint scoring over `(model, instance)` pairs on **four continuous optimization axes**: quality / latency / cost / cluster-memory balance (`w_quality, w_latency, w_cost, w_balance`).
4. No batch-level joint optimization (Hungarian-style matching skipped).

Invoked via `--scheduling cara` (code path: `_schedule_cara_batch`).

**Main eval table (revised)** — Cara-full vs wrapper baselines with knob-sweep Pareto:
- `Cara-full` (5 points via `w_quality` sweep)
- `vllm_sr-wrapper` (single point)
- `RouteLLM-mf-wrapper` (5 points via α, pretrained `mf` + published threshold 0.11593) — paired with RR + shortest-queue
- `Avengers-Pro-wrapper` (5 points via `performance_weight`) — paired with RR + shortest-queue
- `BEST-Route-wrapper` (5 points via threshold t; stripped Best-of-N, DeBERTa-v3-small trained on Qwen + armoRM labels) — paired with RR + shortest-queue

**Primary ablation table** (rows 1-9): Cara-full, no-LPT, no-filter, point-filter (planned), no-{quality,latency,cost,balance}-term, Cara-simple-shortestQ, Cara-simple-llumnix, quality_greedy_ablation (renamed from the revision-1 "best_route" which was misattributed).

**Supporting studies** (not headline): routing-signal swap inside a fixed Cara shell; filter swap inside a fixed Cara shell.

**Primary quality metric**: pre-computed **deepeval** lookup (RouterBench-style) from the clean held-out `test_scored_filtered.jsonl` split (HF `asdwb/cara_model_estimator`). No live judge calls. Bench file resampled to `best-route-v3-test-500-clean.jsonl` (500 clean prompts, zero predictor-train leakage) — **supersedes the contaminated `best-route-v3-test-500.jsonl`**.

**Figures**: revision-3 sweep-scope layering (design §4.6) — **Phase 2 primary load axis = arrival rate λ (pilot → ~{4, 8, 12, 16, 20, 24, 32})**. Figures: λ-axis curves (goodput/throughput/latency vs λ, 5 rows × default knob) + 2D Pareto at knee + 2D Pareto at stress (5 rows × 5 knobs × 2 anchor rates) + CAP radar at knee + goodput bar + throughput bar + memory-variance time-series + w_balance sensitivity appendix + overhead stack + Table 1. Anchor rates predeclared: low/knee/stress picked by rule, not by eyeballing results. ~127 bench runs, ~2.5-4h cluster time.

**Phase 1 (4-node, Apr 15-17) is preflight-only** — metric/monitor/script validation. Paper numbers come exclusively from Phase 2 18-node.

**BEST-Route deadline: Apr 24 morning** — if training fails, cite-only fallback in related work; row 5 removed from main table. No delay of Apr 26 cluster.

Methodology inherits: RouterBench (AIQ Pareto), MoE-CAP (radar), RouterEval (scaling context). Go/no-go checklist for Apr 26 in design §7.C.

**Cost model**: per-token vendor pricing (MaaS framing); published Qwen-2.5 API rates in `scheduler_config_smoketest.json`. Not MoE-CAP's hardware-TCO formula.

**Balance analysis**: minimal in main paper (`w_balance=0` vs `=0.2`, one figure), full sensitivity in appendix.

**Phase A pluggable Router × Dispatcher × Filter infrastructure** (built in #23-#33) is engineering scaffolding for the wrapper baselines and supporting studies — **NOT the paper's main-table framing of Cara-full**.

---

## Phase 1: Model Exploration — CLOSED (April 11)

All offline evaluation complete and verified. Winning models identified.

- [x] Quality predictor comparison: 6 targets × 5 architectures
- [x] Latency predictor comparison: XGBoost vs baselines
- [x] SLO filter comparison: SLOs-Serve vs QLM vs CaraCDF
- [x] Multi-target tradeoff analysis
- [x] HF datasets uploaded and datacards updated
- [x] Route-balance repo synced with all code + docs

**Winner**: RoBERTa fused encoder, targets = length_bucket (CDF filtering) + deepeval (quality routing)

## Phase 1b: Online Preflight — COMPLETE (April 16, with caveats)

4-node A30 preflight (3 runs, 10 bugs fixed, all §7.A pass criteria green). NOT paper numbers.

- [x] All 7 §7.C hard blockers closed (#54, #60, #61, #62, #63, #64, Avengers-Pro α-sweep)
- [x] BEST-Route DeBERTa-v3-small trained (val_acc=0.58)
- [x] 5 Avengers-Pro artifacts (pw=0.25/0.39/0.53/0.7/0.8)
- [x] Clean bench deepeval coverage = 1.0
- [x] Warmup step, regex fix, dataset fix all landed

**Caveats for Phase 2:**
1. Pipeline per-request stats fetch bottleneck (~200ms/req) — need lower λ for pipeline baselines or batched-stats optimization.
2. SQ dispatcher pathological in pipeline mode (stale-state oscillation) — RR is the fair dispatch baseline.
3. RouteLLM-mf domain shift (all-to-3B on Qwen) — needs footnote + possible threshold recalibration.

---

## Phase 2: Deployment Model Training — IN PROGRESS

- [ ] **CSD3 long training** (job 27444574 queued): RoBERTa fused × 200 epochs × 2 targets (length_bucket + deepeval), patience=10
  - Venv: `.venv_roberta` (transformers==4.50.3)
  - Check: `ssh csd3 'squeue -u wd312 && tail -20 /rds/user/wd312/hpc-work/llm/Block/slurm-long200-27444574.out'`
- [ ] Compare 200ep vs 5ep results — expect marginal improvement with early stopping
- [ ] If significant improvement: use 200ep model for deployment. If not: 5ep is fine.

---

## Phase 3: Pre-CloudLab Preparation (before April 26)

### Deployment Configs
- [ ] Select deployment configs D1-D4 with winning models
  - D1: RoBERTa bucket + RoBERTa quality (3 encoder calls)
  - D2: RoBERTa bucket + RoBERTa quality (2 encoder calls)
  - D3: RoBERTa bucket + KNN quality (1 encoder + KNN)
  - D4: Multi-target B (1 encoder call, bucket + deepeval heads)
- [ ] Create scheduler_config JSONs for each config
- [ ] Integrate winning models into `cara_learned_predictor.py`

### Performance Optimization
- [ ] ONNX export for encoder inference (target: <15ms CPU per prompt)
- [ ] Benchmark overhead: D1-D4 configs on CPU

### Experiment Preparation
- See **Evaluation design — pinned (Apr 15)** at the top of this file and `cara_paper/codex/apr15_baseline_comparison_design.md` for the live plan.
- [ ] Main table (revision 2): `Cara-full` + `vllm_sr-wrapper` + `RouteLLM-wrapper (mf)` + `Avengers-Pro-wrapper` + `BEST-Route-wrapper`. Wrappers paired with RR + shortest_queue dispatch. **Not** the stale 8-baseline checklist. `best_route-wrapper` and "strongest composed baseline" rows are retired; see `cara_paper/codex/apr15_baseline_comparison_design.md` §3.2.
- [ ] Primary ablation table: Cara-full ingredient toggles — LPT, filter, bucket-vs-point, `w_quality` / `w_latency` / `w_cost` / `w_balance` individually, and `Cara-simple-shortestQ` / `Cara-simple-llumnix`.
- [ ] Supporting study 1 — router-signal swap inside a fixed pluggable shell.
- [ ] Supporting study 2 — filter swap inside a fixed Cara shell.
- [ ] Arrival-rate λ sweep is the primary Phase 2 load axis (§4.6 + §5.8 of design doc). Pilot → frozen grid ≈ {4, 8, 12, 16, 20, 24, 32}. Anchor-rate rule (low/knee/stress) predeclared. Sweep-scope layering (λ-axis curves at default knob; Pareto at 2 anchors × 5-knob sweep; ablations at 3 anchors; supporting at knee) caps total at ~127 runs.
- [ ] Tasks #51-#55 (toggles + first sweep script) — already complete on the legacy matrix. Rev-3 matrix update tracked via #63 (sweep-script rev-3), #60 (monitor wrapper), #61 (p95 fix), #54 (deepeval lookup patch), #62 (BEST-Route train).

**Retired ideas (superseded by the Apr 15 design):**
- ~~8 baselines flat list (Random / RR / Shortest-Queue / Quality-Greedy / Cost-Greedy / Length-Aware / SR+RR / Cara)~~ — collapses down to the structured Cara-full-vs-wrappers main table + ingredient ablations above.
- ~~9-combo (w_lat, w_cost, w_qual, w_bal) weight sweep~~ — replaced by single-term zero-out ablations (rows 5, 6, 7, 7b) that isolate each term cleanly.

### Code Readiness
- [ ] Route-balance full sync review (see `project_route_balance_sync.md` — mostly done, verify remaining)
- [ ] Verify `setup.sh` works on fresh CloudLab nodes (NEVER manual setup)
- [ ] Smoke test deployment on 2-node CloudLab reservation if possible
- [ ] Verify `deploy_cara.py` automated deployment works end-to-end

### Paper Draft
- [ ] Polish paper with offline results (§5.1-5.3 have final numbers)
- [ ] Fill in result tables and figures from `eval_results/20260410/`
- [ ] Leave placeholders for CloudLab e2e results (§5.4)

---

## Phase 4: CloudLab Evaluation (April 26 - May 6)

### Day 1: Deployment + Latency Data Collection (~18h)
- [ ] Deploy on 18 nodes using `setup.sh` + `deploy_cara.py`
- [ ] **VERIFY IMMEDIATELY** after first QPS level:
  - `actual_output_tokens` field present in sidecar output
  - `pending_decode_tokens > 0` in `/instance_stats`
  - `predicted_decode_tokens` in `/schedule_trace` per-request entries
- [ ] Run latency sweep at QPS 8, 12, 16, 20, 24, 32, 40, 48 (include overloaded levels)
- [ ] Download all latency data before moving on

### Day 2-3: Main Comparison (~16h) — structure per pinned Apr 15 design
Superseded: ~~8 baselines × 5 QPS × 2000 requests flat matrix.~~ Replaced by the Apr 15 design (see pinned section at top of this file + `cara_paper/codex/apr15_baseline_comparison_design.md`).
- [ ] Main table (revision 2): `Cara-full` + `vllm_sr-wrapper` + `RouteLLM-wrapper (mf)` (pretrained, threshold 0.11593) + `Avengers-Pro-wrapper` (α sweep) + `BEST-Route-wrapper` (Qwen-armoRM trained, t sweep). Wrappers paired with RR + shortest_queue dispatch. `--scheduling cara` for Cara-full; `--scheduling pipeline` for wrapper rows. See design §3.2.
- [ ] Primary ablation table (rows 2–8b): `Cara-no-LPT`, `Cara-no-filter`, `Cara-point-filter`, `Cara-no-quality-term`, `Cara-no-latency-term`, `Cara-no-cost-term`, **`Cara-no-balance-term`** (7b — `w_balance` is a real scorer term), `Cara-simple-shortestQ` (8a), `Cara-simple-llumnix` (8b).
- [ ] Supporting study 1 (router-signal swap) and supporting study 2 (filter swap) — pluggable-pipeline harness.
- [ ] **Arrival-rate λ is the primary Phase 2 load axis** (not a QPS tier choice). Sweep-scope layering per design §4.6. Anchor rates (low/knee/stress) predeclared.
- [ ] Collect per row: goodput @ SLO (balanced + strict tiers), deepeval quality (via pre-computed lookup, #54), throughput, cost per successful request, latency distribution, per-stage scheduling overhead (`batch_wait_ms, estimator_ms, stats_fetch_ms, sidecar_ms, scoring_ms, total_scheduling_ms` for Cara-full; monitor.py process-level CPU/RSS for wrappers).
- [ ] Hard blockers landed per design §7.C: #54 deepeval lookup, #60 monitor wrapper, #61 p95→p99 fix, #63 sweep-script rev-3 matrix, RouteLLM-mf swap, Avengers-Pro α-sweep artifact. #51/#52/#53/#55/#56 already complete.

### Day 4-5: Sensitivity Analysis
Superseded: ~~Weight sweep (9 combos).~~ Replaced by per-term zero-out ablations (rows 5/6/7/7b above) that isolate each scorer term cleanly.
- [ ] Batch size sweep (if relevant given single-request dispatch from LPT-sorted queue).
- [ ] Budget-threshold sweep (stress CaraCDF + the legacy inline feasibility path).
- [ ] SLO filter ablation (supporting study 2: `none` vs `cara_tiered` vs QLM vs SLOs-Serve vs PolyServe vs TimeBill — single router held fixed).

### Day 6-7: Ablation Studies
- [ ] Homogeneous ablation (8× A30 like Block) — compares Cara-full against the same integrated scheduler on a homogeneous cluster.
- [ ] LPT sort key ablation (max/min/mean) — separate from #51's LPT on/off; explores the sort-key choice once on.
- [ ] Quality signal ablation (deepeval vs ref_score vs judge vs none) — complements supporting study 1.

### Day 8-9: Buffer + Extensions
- [ ] Rerun any failed experiments
- [ ] P-D disaggregation smoke test (if time permits)
- [ ] Noise injection / burstiness experiments

### Day 10: Cleanup
- [ ] Download ALL data from CloudLab before release
- [ ] Verify data integrity (record counts, field completeness)
- [ ] Upload fresh latency data to HF

### Post-CloudLab: Model Retraining
- [ ] Retrain XGBoost on fresh latency data (with real `pending_decode_tokens`)
- [ ] Retrain LSTM v2 on high-contention data (should show improvement over XGBoost)
- [ ] Update latency eval results

---

## Phase 5: Paper Finalization (May 1-6 / May 1-14)

### Results
- [ ] Plug CloudLab e2e results into §5.4
- [ ] Generate all figures (latency CDFs, QPS scaling, weight Pareto, SLO attainment)
- [ ] Empirical competitive ratio vs offline optimal

### Writing
- [ ] Abstract: update with final numbers
- [ ] Introduction: sharpen contributions with results
- [ ] §5 Evaluation: complete with CloudLab results
- [ ] §6 Related Work: final pass
- [ ] Appendix: P-D disaggregation results (if available)

### Final
- [ ] Internal review pass
- [ ] Format check (NeurIPS: 9 pages + references / EuroSys: 12 pages ACM)
- [ ] Submit by May 6 (NeurIPS) or May 14 (EuroSys)

---

## Key Resources

| Resource | Location |
|----------|----------|
| Offline eval results | `eval_results/20260410/` (VPS + NFS) |
| Winning models | NFS `models/cara/roberta_fused_5ep/` |
| Long training | CSD3 job 27444574 → `models/cara/roberta_fused_200ep/` |
| Paper draft | `cara_paper/cara_new.tex` |
| Experiment plan | `cara_paper/active/EXPERIMENT_PLAN_CLOUDLAB.md` |
| Latest handoff | `cara_paper/handoffs/2026-04-11_handoff.md` |
| Route-balance repo | `github.com/AKafakA/route-balance` (master) |
| vLLM fork | `github.com/AKafakA/vllm` (cara_v_11 + cara_v_11_p100) |
| HF quality data | `huggingface.co/datasets/asdwb/cara_model_estimator` |
| HF latency data | `huggingface.co/datasets/asdwb/cara_latency_prediction` |
| CloudLab reservation | April 26 - May 6, 18 nodes (8×A30 + 3×V100 + 5×P100 + 2×A100) |

---

## 2026-05-05 11:05 UTC — Status Update: Overnight Grid COMPLETE

**Headline numbers locked.** 105/105 cells, 0 errors, 371,070 successful requests at N=3534 prompts/cell. Bench data, CSV, and plots all in `cara_paper/smoke_test_apr_13/results/main_table_full_3534/cara_final_e2e/_aggregate/`.

**Main table data ready** (8 cara weight points + 4 trained-router-baseline cells + 3 dispatcher-only baselines × 7 λ rates × N=3534):
- λ=12 headline (cara uniform): 2219ms mean E2E, 5408ms p99, q=0.508 deepeval, $0.024 / 1k req
- λ=12 vs avengers_pro (-21% E2E, -37% cost, +0.4pp q) — **3-axis Pareto domination**
- λ=30 saturation: cara 2.5s vs BEST-Route 47s — **massive scaling gap**
- weight knob: cost-priority cuts cost 33% at -2.5% quality

**Code state**: KNN k=50 weighted predictor deployed; in-process per-tier XGBoost (~3ms TPOT via `inplace_predict`); passthrough router landed for dispatcher-only baselines; BEST-Route 4-way wrapper landed; pipeline-mode unfair `stats_fetch` overhead patched (requires_stats gating).

**Synced to**: VPS (`/home/wd312/Code/llm/Block/cara_paper/...`) and route-balance repo (code edits earlier today).

### Remaining work (24h cluster budget + paper rewrite)

| Slot | Task | Hours | Status |
|---|---|---|---|
| Cluster 0-3h | BEST-Route confidence_threshold sweep (t∈{0.5,0.7} × {RR,SQ} × 7 λ = 28 cells) | ~3 | PENDING user go |
| Cluster 3-4.5h | Variance reseed @ λ=12 (4 systems × 3 seeds = 12 cells) | ~1.5 | PENDING |
| Cluster 4.5-5.5h | vLLM-SR feasibility (3 cells @ λ=12) | ~0.7 | PENDING |
| Cluster 5.5-6h | APC=ON ablation @ λ=12 (4 cells) | ~0.5 | OPTIONAL |
| Cluster 6-24h | RELEASE / safety buffer | — | — |
| Paper Day 1 | Abstract numbers + Main table writeup + 2 figs | 12 | PENDING |
| Paper Day 2 | §3 method tightening + §4 architecture + §5 ablations | 14 | PENDING |
| Paper Day 3 | Related work, intro, polish, checklist | 10 | PENDING |

### Cuts (deferred to camera-ready)
- 2nd quality judge re-scoring (too risky in 24h)
- Multi-topology robustness study
- AvengersPro pw knob sweep (already 3-axis dominated, low marginal value)

### Known caveats reviewers will flag
1. Single quality judge (deepeval-llama3.1-8b-it_reference)
2. n=1 per cell (variance reseed addresses headline λ=12 only)
3. BEST-Route argmax-only (threshold sweep addresses)
4. No vLLM-SR in main table (feasibility cell addresses minimally)
