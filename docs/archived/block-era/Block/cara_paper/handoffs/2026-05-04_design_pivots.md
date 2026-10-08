# 2026-05-04 — Design + Implementation Pivots (NeurIPS abstract day)

This document captures all design and implementation pivots made today
(2026-05-04) on the path to the NeurIPS abstract submission. It supplements
the running cron handoff log at `2026-04-28_handoff.md`. Updated 21:20 UTC.

---

## TL;DR

Today's main thread: get a fast, fair, all-CPU, in-process Cara scheduler with
defensible numbers vs trained baselines for the abstract main table.

End-of-day deployed configuration:
- **cara_serve** running `--scheduling cara`, `CUDA_VISIBLE_DEVICES=""`,
  `CARA_INPROC_PREDICTOR=1`, scheduler-config `/tmp/scheduler_config_carafast.json`
  (k=50 KNN), pipeline-mode wrappers also patched and validated.
- **Predictors**: in-process per-tier XGBoost tpot (5 boosters loaded once at
  startup, called via `booster.inplace_predict` with no DMatrix); KNN k=50
  inverse-distance weighted for quality + length + bucket distribution; e2e
  computed analytically via Little's-Law from the predicted tpot.
- **Baselines**: AvengersPro 4-model pw=0.7 (May 2 retrain) and BEST-Route
  4-way (May 2, num_labels=4) both validated through pipeline mode with the
  new lightweight stats-skip path.

---

## Design pivots (largest-impact first)

### 1. Predictor stack: HTTP-sidecar XGBoost → in-process per-tier batched XGBoost

**Before today**: each cara_serve request fanned out to per-instance
`/predict_latency` sidecars (one HTTP round-trip per instance per request × 13
instances per batch). HTTP serialization, JSON parsing, and connection setup
dominated; throughput capped at ~10 rps.

**Pivot**: load a per-tier XGBoost tpot booster (`XGBoostLatencyPredictor`,
loaded once at module init from `models/cara/latency/deploy_tpot/`) and run all
predictions in-process via `asyncio.run_in_executor`. Each request fires 4
parallel `_predict_tier` thread-pool calls (one per active tier) via
`_batched_tpot_per_request` (`cara_serve.py:826`). Returns
`{instance_id: tpot_seconds}`. ttft is approximated via constant 0.05 s; e2e is
computed analytically from tpot via Little's-Law:

```
slot_open = (running_n < max_seqs) and (waiting_n == 0)
e2e = own_out * tpot                 if slot_open
    = (queue_iters + own_out) * tpot otherwise   # queue_iters = pending_decode/running
```

Rationale: the analytical Little's-Law form captures cross-batch queue effects
without an extra ML call, while keeping per-tier learnt tpot for hardware
heterogeneity.

**Throughput chain** (single-host scheduler, λ=24 N=500):
- HTTP sidecar baseline: ~9.5 rps
- Option A pipelining (1 prepper + 1 causal worker): 12 rps
- carafast (KNN-only estimator + inproc xgb hosted): 17.8 rps
- + `CARA_INPROC_PREDICTOR=1` (real per-instance tpot lookup): 24.86 rps
- + per-tier batched inplace_predict: same throughput; cheaper estimator

### 2. XGBoost call: `booster.predict(DMatrix(X))` → `booster.inplace_predict(X)`

**Bug discovered**: per-tier batched call cost was 9646 ms in initial
deployment. Root cause was the same bug as 2026-04-29 19:31 UTC — the
pip-installed `xgboost==3.2.0` wheel ships with `USE_CUDA=True`, so every
predict probes the CUDA context. On the 72B host whose A100s are saturated by
vLLM, the probe stalls.

**Fixes (chained)**:
1. `CUDA_VISIBLE_DEVICES=""` on cara_serve startup → 9646 ms → 23 ms (~360× speedup)
2. `xgb.DMatrix(X, feature_names=...)` adds 32 ms per call. Replaced with
   `booster.inplace_predict(X)` which skips DMatrix construction. 23 → 3 ms.

**Net**: per-request batched-tpot wall is now **~3 ms** (mean), 6 ms (p99) at
λ=24 in steady-state.

### 3. KNN k: 10 → 50 (weighted)

**Trigger**: user pointed at literature (RouterBench k=40, arXiv 2505.12601
"kNN beats complex routers" k=100) and observed that one search result feeds
THREE signals (quality, length, bucket distribution).

**Offline eval** (N=3,634 test set, dev-gpu-wd312):

| Predictor | deepeval ρ | reference_score ρ | length ρ | length_bucket acc |
|---|---|---|---|---|
| KNN k=10 weighted | 0.379 | 0.507 | 0.576 | 0.407 |
| **KNN k=50 weighted** | **0.405** | 0.496 | **0.597** | **0.421** |
| KNN k=100 weighted | 0.398 | 0.485 | 0.588 | 0.419 |
| KNN k=50 unweighted | 0.398 | 0.490 | 0.590 | 0.419 |
| KNN k=100 unweighted | 0.390 | 0.479 | 0.581 | 0.415 |
| **RoBERTa-fused-deepeval (warmup0.1, Apr 19)** | **0.456** | — | — | — |
| RoBERTa-fused-length (5ep) | — | — | **0.679** | — |
| RoBERTa-fused-length_bucket (cosine, Apr 20) | — | — | — | **0.424** (acc) |

**Decision**: deploy `k=50 weighted` for the abstract row (cara-knn-k50). It
wins on 3 of 4 KNN axes and is essentially free at serving time
(`booster.inplace_predict` is microseconds; KNN overhead unchanged).

**Edits**:
- `block/predictor/cara/estimators/knn_estimator.py`: added `weighted: bool = True`
  param to all `predict_*` methods + new `predict_bucket(prompt, model,
  bucket_size, num_buckets, weighted)` returning bucket prob distribution. The
  `_distance_weights(distances, weighted)` returns uniform `1/k` when
  `weighted=False`.
- `block/predictor/cara/offline_training/evaluation/adapters/knn_adapter.py`:
  plumbs `k`, `weighted`, `bucket_size`, `num_buckets` through.
- `/tmp/scheduler_config_carafast.json` deployed with `"k": 50`.

### 4. Quality predictor — RoBERTa rejected for tonight's abstract

**Found**: RoBERTa-deepeval (Apr 19 vast sweep, `warmup0.1` variant) wins
deepeval Spearman by +5.1 pp over KNN k=50 (0.456 vs 0.405). Also found
RoBERTa-length_bucket (cosine variant, Apr 20) accuracy 0.424 — comparable to
KNN k=50 (0.421) but with mae_tokens ~102.

**However**, deploying 2 separate RoBERTa heads (length_bucket + deepeval) on
CPU exceeds throughput budget: each forward ≈ 80 ms on B=4-8; running 2 in
"parallel" via `run_in_executor + gather` is ≈ 1.7× single-fwd due to OMP
contention → ~150 ms estimator wall. Critical-path math at B=4 caps the
scheduler at ~17 rps vs cara-knn 31 rps.

**Decision for abstract**: keep KNN k=50 deployed. Document RoBERTa's
prediction-accuracy edge and defer its deployment to Day 2 (requires a single
two-head fused RoBERTa with shared encoder backbone, ~30 min training-script
edit + 1.5 h fine-tune).

### 5. Pipeline mode unfair-overhead audit + fix

**Found**: pipeline-mode (used for AvengersPro/BEST-Route wrappers) was
fetching `/instance_stats` from candidate vLLM nodes on every request, adding
~87-255 ms mean (p99 0.6-1.5 s) of scheduler overhead per request — pure waste
when dispatcher is RR (which doesn't read stats) and filter is `none`. Cara
mode pays only ~9 ms because it batches the stats fetch.

**Fix** (this session):
- `block/global_scheduler/cara/dispatch/base.py`: added
  `requires_stats: bool = True` to `DispatchBase` (default for safety).
- `block/global_scheduler/cara/dispatch/round_robin.py`: overrides
  `requires_stats = False` (pure rotate, no stats use).
- `block/global_scheduler/cara/filters/factory.py`: `_NoFilter` overrides
  `requires_stats = False`.
- `block/global_scheduler/cara/cara_serve.py:2125`: `_select_via_pipeline`
  now skips `_fetch_all_instance_stats` when both dispatcher and filter
  declare `requires_stats=False`.

**Result** (probe v7, λ=12, N=300):
- AvengersPro total_sched: 155 ms → **40 ms** (4× faster, p99 849→72 ms).
- BEST-Route 4-way total_sched: 251 ms → **57 ms** (p99 801→135 ms).
- mean_e2el AvengersPro: 3654 → 2726 ms (−25%); BEST-Route: 2226 → 2122 ms.

### 6. BEST-Route 4-way wrapper (new code)

The trained May 2 4-class artifact at `models/cara/best_route_4way_qwen/`
(num_labels=4, val_acc=0.392) had no integration in cara_serve. Existing
`best_route_wrapper.py` was binary strong/weak.

**New code**: `block/global_scheduler/cara/routers/best_route_4way.py` —
loads HF `AutoModelForSequenceClassification` with num_labels=4, reads
`label_to_model` from `training_results.json`, routes by argmax (with
optional `confidence_threshold` for fallback). Registered in factory.py at
`rtype == "best_route_4way"`.

Probe v7 confirmed all 4 sizes routed (3B 46% / 14B 34% / 7B 15% / 72B 5%),
errors=0.

### 7. AvengersPro: binary 2-model → 4-model (artifact path correction)

**Discovered**: the `models/cara/avengers_pro_qwen_pw0.7/` path I initially
used was the Apr 15 binary (3B+7B only) artifact. The 4-model retrain (task
#53, May 2) lives at `models/cara/avengers_4model/avengers_pro_qwen_pw0.7/`
(`models: [3B,7B,14B,72B]`, n_clusters=32, judge_key=deepeval).

**Wrapper kwarg correction**: `AvengersProRouter` accepts `artifact_dir`
(NOT `checkpoint_dir`); the binary artifact uses default fallback so it
appeared to work in earlier runs. Probe v3+ uses the correct kwarg.

### 8. Filter for baselines: `cara_tiered` → `none`

For apples-to-apples main-table baselines, baselines must run with no cara
filter (router → dispatcher → vLLM only). Probe v3-v4 used cara_tiered which
was a no-op in our config (slo_defaults `filter_enabled:false`,
`constraint_mode:NONE`) but explicit `filter=none` is paper-correct. Probe v5+
uses `filter=none`.

### 9. APC for tonight's abstract: OFF (cluster-wide)

Cluster vLLM heads run with `--no-enable-prefix-caching` (verified on
d7525-10s10317). All routers see cold cache → no cross-router cache leak.
Day 2 paper main table will redo with APC ON + vLLM redeploy between routers
for fairness (rev-3 plan).

---

## Implementation surface area touched today

| File | Change |
|---|---|
| `block/global_scheduler/cara/cara_serve.py` | + `_TPOT_BATCH_MODEL` global; + `_batched_tpot_per_request` helper; CARA_INPROC_PREDICTOR branch in `_predict_one_for_req`; integration in both `_causal_worker_loop` and legacy `_batch_scheduler_loop`; `inplace_predict` swap; `xgb_batched_tpot_ms` in breakdown; `BatchRequest.__slots__` extension; `_select_via_pipeline` requires-stats gate |
| `block/global_scheduler/cara/dispatch/base.py` | `+ requires_stats: bool = True` (DispatchBase) |
| `block/global_scheduler/cara/dispatch/round_robin.py` | `+ requires_stats = False` (override) |
| `block/global_scheduler/cara/filters/factory.py` | `_NoFilter.requires_stats = False` |
| `block/global_scheduler/cara/routers/best_route_4way.py` | NEW — 4-class DeBERTa wrapper |
| `block/global_scheduler/cara/routers/factory.py` | `+ "best_route_4way"` registration |
| `block/predictor/cara/estimators/knn_estimator.py` | `+ weighted` param on all `predict_*`; new `predict_bucket`; `_distance_weights(weighted=False)` returns uniform 1/k |
| `block/predictor/cara/offline_training/evaluation/adapters/knn_adapter.py` | plumb `k`, `weighted`, `bucket_size`, `num_buckets` |
| `block/predictor/cara/cara_learned_predictor.py` | `_load_xgboost` uses `raw_config`; new-schema loader for `xgboost_3model.model_paths_by_instance`; analytical `predict_e2e` |
| `block/predictor/cara/model_estimator.py` | `_neighbor_lengths` + `_neighbor_weights` saved in `estimate_batch`; bucket distribution + p_under_budget assembled when `_has_fused_bucket=False` |
| `cara_paper/smoke_test_apr_13/scripts/eval_knn_k_sweep.py` | NEW — KNN k×weighted×{deepeval, length, bucket, reference_score} eval |
| `/tmp/scheduler_config_carafast.json` | `"k": 10 → 50`; KNN-only model_estimator (no `bucket_config`, no `length_config`) |
| `/tmp/audit_errors.py` | Re-created on cluster (was missing from prior cron cycles) |

---

## Today's headline numbers (cara-knn k=50, N=3,534, APC OFF)

| λ | rps | mean_e2el | p99_e2el | ttft | tpot | xgb_batched_tpot | total_sched_work | dispatch (14B/7B/3B/72B) |
|---|---|---|---|---|---|---|---|---|
| 12 | 11.85 | 2121 ms | 5342 ms | 44 | 21.6 | 3.0 ms | ~110 ms | 45/33/19/2 |
| **24** | **23.27** | **2304 ms** | **5633 ms** | 47 | 23.8 | 3.2 ms | ~110 ms | **41/33/24/3** |
| 36 | 31.75 | 6441 ms | 13556 ms | 49 | 146 | 3.2 ms | ~118 ms | 39/33/25/3 |
| 48 | 31.81 | 18431 ms | 36154 ms | 48 | 501 | 3.1 ms | ~112 ms | 38/33/26/3 |

Saved at `cara_paper/smoke_test_apr_13/results/main_table_full_3534/cara-knn-full/k50/wq0.5_lambda{12,24,36,48}_n3534.json`. All errors=0.

Cluster saturates at ~31 rps (vLLM ceiling). Headline cell λ=24 is pre-saturation.

---

## Baseline saturation probe (probe v7, post-stats-skip fix, N=300, λ=12, RR, filter=none)

| Router | rps | mean_e2el | p99 | total_sched | Distribution (4 sizes) |
|---|---|---|---|---|---|
| Cara-knn k=50 | 11.85 | 2121 ms | 5342 ms | ~110 ms | 45/33/19/2 |
| BEST-Route 4-way | 10.39 | 2122 ms | 5826 ms | 57 ms | 46/34/15/5 (3B/14B/7B/72B) |
| AvengersPro 4-model pw=0.7 | 8.66 | 2726 ms | 12006 ms | 40 ms | 46/31/22/1 (3B/14B/72B/7B) |

All three routers use 4 distinct Qwen sizes and pass the 5-criterion smoke
gate. AvengersPro p99 is high because it routes 4× more to 72B (which has
TP=2 A100, slow tpot) for harder/longer prompts.

---

## What's deferred to Day 2

1. **2-head fused RoBERTa** (length_bucket + deepeval, shared encoder) — replaces the deployed KNN k=50 row in the main table.
2. **APC ON main-table cells** — rerun all 5 main-table rows × 5 λ with APC ON + vLLM redeploy between routers.
3. **SQ dispatcher rows** — rev-3 plan calls for both RR and SQ for wrapper baselines.
4. **vLLM-SR baseline** — Docker reinstall + ModernBERT semantic-router smoke.
5. **Per-instance stats TTL cache** for SQ paths (sidesteps the unfair stats-fetch tax that the requires_stats gate currently dodges only for RR + filter=none).

---

## Cluster state at 21:20 UTC

- cara_serve PID 1534097, `--scheduling cara`, k=50, CUDA hidden, `/health=200`.
- vLLM heads: APC OFF (verified).
- /tmp scratch cleaned of all today's dev XGBoost/probe/sweep artefacts.
- Disk 51/63 GB (85%, no growth).
- Audit corpus: 18 May-4-04:37 errored cells (pre-existing, retained per
  user instruction). Today's 8 cells (4 × _dev_xgb_inproc_runs/ + 4 × k50/)
  all errors=0.
- `/tmp/audit_errors.py` restored (cluster's own depth-1-glob version).
- `/tmp/scheduler_config_carafast.json` k=50 active.

Ready to launch the abstract grid: 3 routers × 2 λ ∈ {12, 24} × N=3534, RR,
filter=none, APC OFF. ~30-40 min wall.
