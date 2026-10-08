# Code-Paper Audit & Resolution Plan — March 27, 2026

## Audit Results

Systematic comparison of every technical claim in `cara_new.tex` vs code.
Total: 62 items checked. 3 mismatches, 4 partial, 55 matches.

---

## Resolved Issues (Fixed Today)

| Issue | Resolution |
|-------|-----------|
| "Fused" models were per-model | True multi-head architecture implemented |
| Budget filter token-based | Cost-based per-model filtering added |
| Balance score used queue depth | Now uses kv_cache_utilization |
| Judge metrics reported mae_tokens | Fixed: mae_class for judge, mae_tokens only for bucket |

---

## Open Issues & Decisions

### 1. Batch-Wide Normalization (PARTIAL fix needed)

**Paper says**: "per-signal min-max normalization within the batch"
**Code does**: Per-request normalization across that request's candidates

**Decision**: Two-phase scoring
- Phase 1 (batch-wide): Normalize quality + cost across all (request, model) pairs — already available from model estimator
- Phase 2 (per-request): Normalize latency + balance per-request across candidates — sidecar calls are per-request anyway

**Rationale**: Batch-wide normalization for all signals would require 32×18=576 sidecar calls upfront (impossible latency). Quality and cost are prompt-dependent (model estimator computes for all models), so batch-wide normalization is free. Latency is instance-state-dependent and per-request normalization is unavoidable. Similar to GDPO's batch normalization of rewards before ranking.

**Action**: Fix `_schedule_cara_batch` to receive batch-wide quality/cost stats. Update paper to clarify "batch-wide for prompt-dependent signals, per-candidate for state-dependent signals."

### 2. Memory Capacity Constraint (Paper fix)

**Paper says**: `sum(M_ri) ≤ Cap_i` as hard constraint in MILP
**Code**: No memory capacity filter

**Decision**: Remove from MILP. Single requests don't exhaust KV capacity. Keep projected memory in balance score (soft penalty).

**Action**: Update MILP formulation in paper. Add note that memory capacity is enforced by the vLLM engine itself (preemption/eviction when full).

### 3. TPOT XGBoost Model (Config fix + retrain)

**Paper says**: "per-instance XGBoost for both TTFT and TPOT"
**Code**: E2E XGBoost in deploy, TPOT static lookup as fallback

**Decision**: The XGBoost models were trained for E2E. Need to retrain with `--target ttft` and `--target tpot` separately, or clarify that E2E model is used and sidecar decomposes.

**Also**: The `actual_output_tokens` feature in training data — currently `num_predicted_output_tokens` is always 1024 (the max, not actual). This confuses the model. For E2E prediction, actual output tokens matter. For TTFT/TPOT, it should be irrelevant (auto-dropped by XGBoost feature importance).

**Action**:
1. Retrain XGBoost with separate TTFT and TPOT targets
2. Verify `num_predicted_output_tokens` feature weight is ~0 for TTFT/TPOT
3. For E2E, use `actual_output_tokens` in training (already recovered in latency data)
4. Update sidecar to load target-specific models

### 4. Balance Score: Projected Memory Gain (Code fix)

**Current**: Uses kv_cache_utilization snapshot
**Needed**: Project memory impact of assignment

**Formula**:
```
projected_kv_util(i, r) = current_kv_util(i) +
    (prompt_tokens + E[output_tokens]) × kv_bytes_per_token(model_i) / kv_capacity(i)
```

**Per-model KV bytes/token (fp16)**:
| Model | Layers | KV Heads | Head Dim | KV/token |
|-------|--------|----------|----------|----------|
| 3B | 36 | 2 | 192 | 55 KB |
| 7B | 28 | 4 | 256 | 115 KB |
| 14B | 48 | 8 | 256 | 393 KB |
| 72B | 80 | 8 | 128 | 328 KB |

**Action**: Add `model_kv_config` to predictor config. Compute projected util in balance scoring. Need `kv_capacity` from instance_stats or config.

### 5. Global SLO Tracking (Code fix)

**Paper says**: `min(RSO_lat, SLO_lat)` — check both per-request and system-wide
**Code**: Only checks per-request RSO thresholds

**Decision**: Add system-wide SLO enforcement + tracking.

**Action**:
1. Add `global_slo` section to config (ttft_slo_ms, tpot_slo_ms)
2. Filter checks: `min(request.ttft_slo_ms, global_slo.ttft_slo_ms)`
3. Track counters: overall_goodput, rso_goodput, slo_goodput separately
4. Report in `/v1/scheduling_stats`

### 6. RSO API Schema (Paper fix)

**Paper listing** shows: `latency_ms` (single field)
**Code**: `ttft_slo_ms` + `tpot_slo_ms` + `budget_cost`

**Action**: Update paper Listing 1 to show current API:
```json
{
  "rso": {
    "ttft_slo_ms": 2000,
    "tpot_slo_ms": 100,
    "budget_cost": 0.001,
    "quality_min": 0.5,
    "constraint_mode": "TIERED",
    "relax_order": ["ttft", "tpot", "quality", "budget"]
  }
}
```

### 7. E2E Latency as Optional Filter (Paper + code)

**Issue**: E2E latency depends on predicted output length (inaccurate). Should not be a hard filter.
**Decision**: E2E is subscore in ranking only. TTFT/TPOT are hard filter criteria.

**Action**: Update paper to clarify:
- TTFT SLO: hard filter (accurate, doesn't depend on output length)
- TPOT SLO: hard filter (accurate, doesn't depend on output length)
- E2E latency: soft subscore in w_lat ranking (uses predicted length, acceptable noise for ranking)
- Budget (cost): hard filter via P(cost ≤ B) from bucket distribution
- Note: E2E could optionally be a filter but not recommended due to length prediction MAPE

### 8. Makespan w_m (Paper framing — already OK)

**Paper lines 388-389**: Already explain LPT ordering as proxy for makespan minimization with Graham's 4/3 approximation. The MILP includes w_m for completeness but the greedy algorithm approximates it.

**Decision**: No change needed. The framing is correct — MILP is the ideal formulation, greedy is the practical implementation.

---

## Implementation Priority & Full Action List

### Code Changes

| # | Item | File(s) | Effort | Details |
|---|------|---------|--------|---------|
| C1 | **Multi-objective normalization** | `cara_serve.py` | Medium | Implement 4 normalization variants for ablation study. See Normalization Design section below. |
| C2 | **Projected memory in balance score** | `cara_serve.py` + config | Medium | Add `model_kv_config` with kv_bytes_per_token per model (3B:55KB, 7B:115KB, 14B:393KB, 72B:328KB). Compute `projected_util = current_kv_util + (prompt + E[output]) × kv_per_token / capacity`. Need kv_capacity from instance_stats or config. |
| C3 | **Global SLO enforcement + tracking** | `cara_serve.py` | Low | Add `global_slo` config section. Filter checks `min(request.slo, global_slo)`. Track counters: overall_goodput (RSO+SLO both met), rso_goodput (per-filter), slo_goodput (system-wide). Expose in `/v1/scheduling_stats`. |
| C4 | **RSO sampling in benchmark** | `benchmark_serving.py` | Medium | Each field sampled independently at configurable rate (default 10%). Fields: budget_cost ∈ [0.0001, 0.0005, 0.001, 0.005], ttft_slo_ms ∈ [500, 1000, 2000], tpot_slo_ms ∈ [30, 50, 100], quality_min ∈ [0.3, 0.5, 0.7]. Ablation: sweep rate 0%, 10%, 30%, 50%. |
| C5 | **KNN-only model estimator** | `model_estimator.py` + config | Medium | Single KNN call (k neighbors) → point estimate (weighted mean) + bucket distribution (neighbor histogram) + quality scores. Config: k, index_type. One call serves all needs. |
| C6 | **Estimator sub-breakdown timing** | `model_estimator.py` | Low | Add timing for: KNN embedding, FAISS search, ModernBERT forward, total. Return in overhead dict. |
| C7 | **Per-request detail saving in benchmark** | `benchmark_serving.py` | Low | Save per-request: model chosen, all candidate scores, filter reasons, quality predicted vs actual, scheduling overhead breakdown. |
| C8 | **TPOT/TTFT XGBoost retrain** | Training scripts + config | Medium | Retrain with `--target ttft` and `--target tpot` separately. Verify `num_predicted_output_tokens` feature has ~0 importance for TTFT/TPOT. For E2E, use `actual_output_tokens` (already in tagged data). Update sidecar config to load target-specific models. |
| C9 | **LSTM wiring in scheduler** | `cara_serve.py` + config | Low | Wire existing `lstm_predictor.py` into config loading. Add `predictor_type: "lstm"` option alongside xgboost. |
| C10 | **Resource monitor integration** | `monitor.py` + benchmark | Low | Start monitor as subprocess during benchmark. Scrape vLLM `/metrics` (KV cache, running/waiting, GPU util) + system CPU/memory. Write CSV for post-analysis. |

### Paper Changes

| # | Item | Section | Details |
|---|------|---------|---------|
| P1 | **Remove memory capacity hard constraint from MILP** | §3.2, Eq.1 line 296 | Remove `sum(M_ri) ≤ Cap_i`. Add note: memory capacity enforced by vLLM engine (preemption/eviction when full). Balance score provides soft penalty. |
| P2 | **Update RSO API schema** | §3.1, Listing 1 | Show current API: `ttft_slo_ms`, `tpot_slo_ms`, `budget_cost`, `quality_min`, `constraint_mode`, `relax_order`. Remove stale `latency_ms` field. |
| P3 | **E2E latency framing** | §4.3, §3.3 | Clarify: TTFT/TPOT are hard filter criteria (accurate, no output-length dependency). E2E is subscore in ranking only (uses predicted length, acceptable noise for relative ordering). E2E could optionally be a filter but not recommended due to length prediction MAPE. |
| P4 | **Tiered filtering discussion** | §3.3 new paragraph | Explain STRICT/TIERED/RELAXED modes. Show trade-off: STRICT may reject too many (high rejection rate under tight SLOs), RELAXED may route suboptimally. TIERED progressively relaxes in priority order. Ablation study planned. |
| P5 | **Normalization discussion** | §3.3 benefit (1) | Update to describe two-phase approach: batch-wide for prompt-dependent signals (quality, cost), accumulated for state-dependent signals (latency, balance). Reference GDPO analogy. |
| P6 | **KV bytes per model table** | §4.3 or §5 | Add table showing per-model KV cache cost: 3B=55KB/tok, 7B=115KB, 14B=393KB, 72B=328KB. Explains why balance score matters — assigning a request to 14B consumes 7× more memory than 3B. |
| P7 | **XGBoost feature analysis** | §4.3 or §5 | Show `num_predicted_output_tokens` has zero importance for TTFT/TPOT (always 1024 in training data, irrelevant). Validates the decomposition claim. |

### Ablation Studies Needed

| # | Ablation | Variables | Metrics | Purpose |
|---|---------|-----------|---------|---------|
| A1 | **Normalization scope** | per-request vs batch-wide vs accumulated | Routing quality, quality score, load balance CV | Does batch-wide actually help? |
| A2 | **KNN k** | k=5, 10, 20, 50 | Bucket accuracy, point MAE, Spearman, latency | Optimal k for bucket vs point |
| A3 | **KNN index** | numpy, faiss_flat, faiss_ivfpq | Accuracy, latency | Speed vs accuracy trade-off |
| A4 | **Tiered filtering** | STRICT, TIERED, RELAXED | SLO goodput, rejection rate, routing quality | Constraint handling strategy |
| A5 | **RSO density** | 0%, 10%, 30%, 50% per-field rate | Goodput, rejection rate, scheduling overhead | Impact of user constraints |
| A6 | **Scoring weights** | 9 weight configs (quality-first, latency-first, etc.) | Quality, latency, cost, balance CV | Pareto frontier |
| A7 | **Batch size** | 1, 4, 8, 16, 32 | Scheduling overhead, latency, throughput, quality | Batch vs online dispatch |
| A8 | **Budget threshold θ** | 0, 0.3, 0.5, 0.7, 0.9 | Budget goodput, rejection rate | Probabilistic filter sensitivity |
| A9 | **LPT sort key** | max, min, mean across models | Makespan, P95 latency, balance CV | Sort key for heterogeneous models |
| A10 | **Fused vs per-model vs KNN vs PFS** | 4 estimator types | All metrics | Model estimator comparison |

### Experiment Infrastructure Needed

| # | Item | Details |
|---|------|---------|
| E1 | **Benchmark sends per-request RSO** | Currently all requests use server defaults |
| E2 | **Benchmark saves per-request details** | Model chosen, scores, filter reasons |
| E3 | **Monitor process** | GPU memory, CPU, KV cache utilization over time |
| E4 | **Scheduling stats endpoint** | Filter/relaxation counters (done) |
| E5 | **Multiple estimator configs** | Fused, PFS, KNN-only, per-model configs ready to swap |
| E6 | **Weight sweep script** | Modify scoring_weights via `/v1/config` between runs |

---

---

## Normalization Design (C1 Detail)

### Problem Constraint
Standard MCDM (TOPSIS, SAW, GDPO) assumes all alternatives are evaluated upfront → full decision matrix available. In Cara, latency predictions require per-instance sidecar calls. We can't afford `|batch| × |instances|` calls (e.g. 32×18=576 at ~5ms each = 3s).

### Signal Classification
| Signal | Depends On | Pre-computable? | Normalization |
|--------|-----------|-----------------|---------------|
| Quality | Prompt + model | Yes (ModelEstimator batch call) | Batch-wide (exact) |
| Cost | Prompt + model + price | Yes (E[length] × cost_per_token) | Batch-wide (exact) |
| Latency (E2E) | Instance state + prompt + model | No (sidecar per request) | Accumulated |
| Balance | Instance state + assignment | No (changes after each assignment) | Accumulated |

### Normalization Variants to Implement & Compare

| Variant | Quality/Cost | Latency/Balance | Extra Cost | Description |
|---------|-------------|-----------------|------------|-------------|
| **per_request** (current) | Per-request candidates | Per-request candidates | 0 | Simple baseline |
| **topsis** | Per-request ideal/anti-ideal distance | Per-request ideal/anti-ideal | 0 | Classical MCDM baseline |
| **two_phase_accumulated** (proposed) | Batch-wide (exact) | Stats-initialized + accumulated running min/max | 0 | Our design |
| **z_score_accumulated** | Batch-wide mean/std | Running mean/std from accumulated sidecar results | 0 | Statistical variant |

### Stats-Initialized Accumulated Approach
Before the scheduling loop, use instance stats (already fetched) to estimate initial latency range:
```
For each instance i:
  load_i = num_running + num_waiting
  est_latency_i = base_rate[model_i] × (1 + load_i × queue_factor)
→ running_lat_min, running_lat_max initialized from estimates
```
During scheduling, actual sidecar predictions update the running range (expand only, never shrink).
LPT ordering helps: first request (most expensive) sees widest range → good initialization.

### TOPSIS Implementation
```
ideal = (min_lat, min_cost, max_qual, min_balance)
anti_ideal = (max_lat, max_cost, min_qual, max_balance)
d_ideal = sqrt(Σ w_i × (x_i - ideal_i)²)
d_anti = sqrt(Σ w_i × (x_i - anti_i)²)
score = d_anti / (d_ideal + d_anti)  # higher = better
```
Per-request scope (same constraint as our approach).

### Citations
- TOPSIS: Hwang & Yoon (1981) — original MCDM method
- SAW: simple additive weighting — what we essentially do with min-max
- MONWS [Applied Sciences 2023] — multi-objective normalization for cloud workflow scheduling
- TOPSIS vs SAW comparison [Annals of Operations Research 2023]
- GDPO [NeurIPS 2024?] — batch normalization of rewards in RL post-training

### Ablation Design (A1)
- Run full evaluation with each normalization variant
- Same workload, same weights, same baselines
- Measure: routing quality (best-model hit rate), latency P50/P95, load balance CV, throughput
- Hypothesis: two_phase_accumulated ≥ per_request, with largest gap under high load (divergent instance states)

---

## Verified Correct (No Action Needed)

- Bucket classification (16×64, Eq 3-4): MATCH
- LPT ordering (max across models sort key): MATCH
- Tiered constraint relaxation: MATCH
- Adaptive batch sizing (load-aware): MATCH
- Sidecar architecture (latency at instance, quality at scheduler): MATCH
- Thundering herd avoidance (sequential assignment + local update): MATCH
- Broadcasting data collection: MATCH
- Safety-aware quality scoring (protectai vs Qwen judge): MATCH
- Algorithm 1 all 12 steps: 11/12 MATCH (makespan proxied via LPT)
