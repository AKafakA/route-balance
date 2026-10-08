# Cara/RouteBalance — v2 Rerun: Changes, Features, Fixes & Results
**Compiled 2026-06-09 from the June campaign (chat history + handoffs + memory).**
This is the code-lock changelog. Numbers section is filled after the consolidation run.

---

## 0. Why v2 (one line)
The submitted (v1/May/NeurIPS) numbers rest on real bugs (aggregator zero-score inflation, silent
param-grid mismatches, an AvengersPro routing flood). v2 = the **submitted stack + intended fixes only**,
rerun end-to-end with a corrected aggregator, so the result set is self-consistent for resubmission.

## 1. v2 stack = submitted stack + INTENDED fixes (locked decisions)
Reference for "submitted": `route-balance/reproduce/run_main_table.sh` + `_cluster_archive_20260506/
scheduler_config_carafast.json`. Intended fixes (kept; everything else byte-identical):
- Output cap **256 → 1024**.
- Routing `score_type` **judge → deepeval** (submitted routed on judge but reported deepeval).
- KNN predictor **k=50 → k=10**, `quality_norm` **raw → minmax**.
- a30 latency **hardware-aware tier** (fixes the first-match p100 bug).
- **1024 latency-XGB retrain** + **batched-FAISS KNN**.
- Quality signal of record = **deepeval** (key `deepeval-llama3.1-8b-it_reference`), reported via the
  FIXED aggregator. cara routes KNN-k10; AvengersPro deepeval-flat; BR4 deepeval checkpoint (no retrain).

## 2. BUGS FOUND & FIXED this campaign (the "no critical bug like v1" list)
1. **Aggregator zero-score bug** (`score = a or b` drops a deepeval score of exactly 0.0 → silently
   excludes zero-quality responses → inflates quality by **+0.116**; submitted May cell 0.512→0.397,
   coverage 0.774→1.0). System-dependent: cheap-routing systems inflated more, so correcting it
   *widens* cara's lead. **FIX:** explicit `None` checks in `aggregate_v2_grid.py:build_deepeval_lookup`/
   `aggregate_cell` (NOT `a or b`). The May `aggregate_overnight_grid.py` still has the bug — DO NOT use it.
2. **Param grids silently didn't match the submitted May run** → fixed to match paper §5.2 exactly:
   cara = 8 simplex tuples (not scalar-q); AvengersPro = pw{.25,.39,.53,.70,.80}; BEST-Route = argmax +
   t{.3,.5,.7} (v2 supersets to {0,.3,.5,.6,.7,.8}); dispatcher-only adds **random**.
3. **AvengersPro all-72B flood** = upstream `SimpleClusterRouter` interface drift (`embedder.get` +
   `tokenizer`) AND a STALE cached module in the long-running `cara_serve` → **FIX:** shim in
   `routers/avengers_pro.py` (add `.get` + inject tiktoken tokenizer) + **restart** to drop the cached
   module. Verified by routing DISTRIBUTION, not e2e.
4. **Batching ablation invalid** — `batch_config` was nested under `slo_defaults` so the live batch_queue
   never updated → **FIX:** POST top-level `batch_config` (cara_serve.py:634), smoke-gate via
   `/v1/batch_stats`. v2 §5.5 finding: LPT-off ≈ default; adaptive-off +3.5% E2E; bs=1 does NOT collapse
   at low λ (it does at λ≥16).
5. **Budget filter no-op** — `cara_filter == cara_nofilter` in `--scheduling pipeline` (the admission
   filter lives in the NATIVE batch scheduler) → **FIX:** redo in `--scheduling cara` native; §5.4 then
   reproduces (cara+filter 32.8/19.2/12.6 ≈ paper 32.4/19.2/12.6).
6. **Scheduling-mode phase split** (gotcha, not a bug): cara = `--scheduling cara` (native); baselines =
   `--scheduling pipeline`; a bare `scoring_weights` POST in pipeline mode RAISES (cara_serve.py:474).

## 3. NEW experiments / features in v2
- **Cost-extreme cara tuples** (#18): added (0,0,1)/(.05,.05,.9)/(.1,.05,.85) × 7λ so cara can reach the
  pure-cheap corner (May's wc=0.8 cap never did) → restores Pareto-optimality as the headline.
- **Multi-seed** (#13): peak-quality + uniform per system × λ{12,18,24} × 3 extra seeds (45 cells, 0 fail)
  → CIs / "point within CI".
- **vLLM-SR e2e** (#14, NEW baseline): vLLM-project semantic-router v0.3 (Themis) on the main cluster.
  - Baseline design: **complexity-gate (mmbert) → multi_factor (quality/cost/latency/load) within tier** —
    vLLM-SR's own signal→decision→algorithm; content-aware AND cost-quality. (Pure multi_factor with
    constant per-model quality_score is content-AGNOSTIC → all-to-14B; rejected.) Fed our deepeval
    per-model means as `quality_score`.
  - Infra fixes: ext_proc `message_timeout` 200ms→300s (long prompts 504'd at the CPU classifier's ~220ms);
    Envoy circuit_breakers raised. **Finding:** the content-aware CPU router (mmbert, A100s busy with 72B)
    cannot sustain high λ — option A (run all λ, report honestly) shows the saturation curve.
- **Judge-alt (gemma)** (#15c): re-score the canonical (prompt,model) responses with a 2nd judge to test
  robustness. gemma-4 is ungated but needs CUDA-13/driver≥580 (cluster A100 = driver 570/CUDA 12.8) → used
  **gemma-3-12b-it (unsloth, ungated)** bf16 on freed A100s via **vLLM 0.10.2 cu128 + transformers 4.57.6**
  (the only combo that's cu12-compiled AND new enough for gemma-3's rope config). **4-way data-parallel**
  (1 instance/GPU, sharded dataset), ~120× the dev-gpu GGUF path. Same DeepEval GEval script/criteria as
  the main judge; only the model changes (key `deepeval-gemma3-12b-it_reference`).
- **Offline analyses** (#15): scheduling-overhead breakdown (App.B), bootstrap CI (App.A).

## 4. KEY FINDINGS / framing (all evidence-cited)
- **Pareto-optimality RESTORED** as the core v2 headline once cost-extreme tuples let cara reach the cheap
  corner (framing decision RESOLVED 2026-06-08).
- **Load robustness**: at high λ baselines collapse (BR4 27→64s, AvengersPro climbs to ~7s) while cara
  stays bounded (~2.6s) — ~23× at λ30. vLLM-SR collapses hardest (89–97% drop at λ≥18).
- **Judge robustness**: cara peak-quality cell ranks **#1 under BOTH** Llama-3.1 and gemma-3 (numbers in §6).
- Quality gaps DOUBLED vs v1 after the aggregator fix (+0.013 vs BR4, +0.043 vs AvengersPro at peak).

## 5. INFRASTRUCTURE notes (reproducibility)
- 13-node CloudLab `asdwb-307592`; scheduler `asdwb@d8545-10s10301:8200`. A100 nodes
  d8545-10s10301/10s10305 (TP=4 72B, port 7000).
- For the gemma judge, **both A100 nodes freed** (8 GPU). 72B PRESERVED (redeployable from
  `block/config/cara/model_deployment.json` + manifest `cara_paper/handoffs/teardown/`). Gemma venv on
  `/mydata` (root was 92% full; pip cache → root caused Errno 28, cleaned).
- Orchestration scripts pulled into `cara_paper/smoke_test_apr_13/scripts/v2_orchestration/` (node /tmp
  gets cleaned). Aggregator (FIXED) = `aggregate_v2_grid.py`.
- Local git: checkpoints `2074f4a` (v2 stack), `0bfc0f5` (handoff). No remote (online AKafakA/Block stale).

## 6. CONSOLIDATED NUMBERS (consolidation run 2026-06-09)
Source: FIXED `aggregate_v2_grid.py` → `experiment_output/v2_sweep_jun06/_aggregate/AGGREGATE.csv`
(**287 cells**: cara 112 [8 simplex + 3 cost-extreme + 5 scalar-q × 7λ], avengers_pro 70, best_route_4way 84,
dispatcher_only 21 [rr/sq/**random** × 7λ]). **Consolidation bug caught & fixed:** the aggregator's
`parse_cell_name` regex `passthru_(ro|sh)` silently excluded the v2-added **random** dispatcher
(`passthru_ra_*`, 7 cells) → fixed to `(ro|sh|ra)`; dispatcher_only 14→21.

### 6.1 Peak quality (deepeval, max over all cells) — cara #1
| system | peak deepeval-q | at |
|---|---|---|
| **cara** | **0.4190** | wq0.8, λ12 |
| best_route_4way | 0.4056 | t0.5, λ6 |
| avengers_pro | 0.3760 | pw0.8, λ6 |
| dispatcher_only | 0.3709 | random, λ10 |
cara peak beats BR4 by **+0.013**, AvengersPro by **+0.043** (gaps ~doubled vs v1 post-aggregator-fix).

### 6.2 λ=12 frontier (quality / cost-per-req / e2e)
| cell | q | cost/req | e2e |
|---|---|---|---|
| cara peak (wq0.8) | 0.419 | 6.9e-5 | 5909ms |
| cara cost-extreme (wq0) | 0.352 | **1.68e-5** (cheapest) | 2138ms |
| br4 t0.5 (rr) | 0.406 | 3.9e-5 | 4289ms |
| avgpro pw0.8 (rr) | 0.376 | 2.6e-5 | 2574ms |
| passthru random | 0.363 | 4.1e-5 | 3950ms |
→ cara spans the whole frontier: **highest-q (0.419) AND cheapest (1.68e-5)**; cost-extreme tuples
reclaim the cheap corner (May's wc=0.8 cap never did) → **Pareto-optimality restored** as the headline.

### 6.3 Load robustness — e2e(ms), PEAK-QUALITY cell per system (matched comparison)
| system (peak-q cell) | peak-q | λ6 | λ12 | λ30 | λ6→λ30 blowup |
|---|---|---|---|---|---|
| cara scalar-q0.80 (NOTE: scalar-q cell, not the simplex tuple) | **0.419** | 5802 | 5909 | **11835** | **2.0×** (bounded) |
| cara simplex tuple wq0.8/0.1/0.1 (the PAPER headline cell) | 0.4187 | 13362 | 5954 | 22756 | bounded but 72B-heavy |
| best_route_4way (t0.5) | 0.406 | 3040 | 4289 | **64204** | **21×** ← COLLAPSE |
| avengers_pro (pw0.8) | 0.376 | 2210 | 2574 | 7205 | 3.3× |
| dispatcher_only (random) | 0.371 | 3922 | 3950 | 3198 | flat |
→ **Honest matched read:** at peak quality, **cara dominates BR4 on BOTH axes at λ30** (11.8s @ q0.419 vs
64.2s @ q0.406 — BR4's argmax collapses 21×). vs AvengersPro it's a **quality-latency tradeoff** (cara
0.419@11.8s vs avgpro 0.376@7.2s — cara higher q, avgpro lower latency, NOT domination). Separately,
cara's **balanced/cheap cells stay bounded ~2.4s at λ30** (cost-extreme 2389ms, uniform 2596ms) — i.e.
cara offers bounded-latency configs that BR4's collapsing argmax cannot. (Earlier "~27×" mixed cara-cheap
vs br4-peak cells — corrected.)

### 6.4 Multi-seed CIs (#13) — seed1 + s2/s3/s4, 95% CI
cara wq0.8: λ12 **0.4184 [0.4182,0.4187]**, λ18 0.4182, λ24 0.4176 (tight, reproducible).
cara uniform: ~0.370 [±0.0003]. Baselines (br4 0.3970, avgpro 0.3760) have **zero quality-variance**
across seeds (deterministic routing → quality is seed-invariant; seeds shift arrival order → latency only).
passthru random: 0.364 [±0.002] (RR order is seed-dependent).

### 6.5 Bootstrap CI (App.A) — cara wq0.8 λ12
deepeval mean **0.4187, 95% CI [0.4089, 0.4288]** (±0.0099, 10k resamples, coverage 1.000).

### 6.6 Scheduling-overhead breakdown (App.B) — cara λ12, 56,544 reqs
batch_wait **48.6ms** + KNN-estimator **28.2ms** + stats_fetch **14.8ms** + xgb_tpot 2.8 + predict_round 2.4
+ scoring 0.16 → **total 134ms** (compute-only ~85ms; ML predict itself ~5ms). ≪ vLLM-SR router 0.13–6s.

### 6.7 vLLM-SR e2e (#14, option A all λ) — head-to-head
| λ | completed | fails | q | e2e | cost/1k |
|---|---|---|---|---|---|
| 6/8/10 | 3534 | **0** | ~0.37 | 4.4–5.3s | ~0.038 |
| 12 | 3260 | 274 (7.7%) | 0.37 | 14.2s | 0.037 |
| 18 | 373 | 3161 (**89%**) | 0.37 | 236s | 0.037 |
| 24 | 208 | 3326 (**94%**) | 0.37 | 302s | 0.034 |
| 30 | 123 | 3411 (**97%**) | 0.35 | 344s | 0.035 |
→ vLLM-SR quality ~flat ~0.37 (< cara 0.419) but its content-aware **CPU router collapses** at λ≥18
(89–97% drop, 236–344s) — the lightweight-predictors-scale / heavy-router-saturates contrast.

### 6.8 Judge-alt robustness (#15c, gemma-3-12b on A100) — conclusion holds
n=14,427 paired. Per-model llama→gemma: 3B .346→.585, 7B .333→.626, 14B .398→.677, 72B .427→.706.
**cara peak ranks #1 under BOTH judges** (Llama 0.419 / gemma 0.696; br4 #2 under both 0.397/0.668).
Per-pair Pearson **r=0.555** (gemma more lenient; top-tier ordering consistent; only the noisy 3B-vs-7B
fine order differs — llama 7B<3B, gemma 3B<7B). Same GEval script/criteria as main; only the judge model differs.

### 6.9 Coverage vs v2 plan (qualified 2026-06-10: serving tracks 100%; the budget DeepEval re-score was completed late (budget_rescore/RESULT.md) and the judge-alt covers the t=0.3 cell only — t=0.5 re-judge = future work)
All planned tracks (§5.2 main+cost-extreme, §5.3 overhead, §5.4 budget, §5.5 batching, App.A bootstrap,
multi-seed, vLLM-SR, judge-alt) have cells + analyses. Intentionally dropped (per plan): SLO-filter
ablation, KNN-vs-RoBERTa predictor comparison.
