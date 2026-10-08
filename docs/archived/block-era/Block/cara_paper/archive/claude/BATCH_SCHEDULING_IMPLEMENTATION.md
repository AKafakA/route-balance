# Batch Scheduling Implementation — March 26-27, 2026

Session documentation for context continuity. All 7 paper-vs-code gaps closed March 26. March 27 additions: fused model estimator, PFS baseline, cost-based budget filtering, GPU memory balance score, scheduling counters.

---

## March 27 Updates

### New Code Gaps Found & Fixed
1. **Balance score**: was queue-depth, now kv_cache_utilization (GPU memory). TODO: projected memory gain per assignment.
2. **Budget filtering**: was token-based (model-independent), now cost-based (per-model token budget via cost_per_token). Added `budget_cost` RSO field.
3. **Fused model estimator**: true multi-head ModernBERT (1 encoder, N heads). `fused_model_dir` config option with automatic fallback to per-model.
4. **PFS baseline**: `PFSModelEstimator` as drop-in replacement. Sliding window + input-length bins.
5. **Scheduling counters**: per-reason filter/relaxation tracking, exposed via `/v1/scheduling_stats`.
6. **Overhead model**: formalized batch bubble (Eq. 5 in paper): `effective_wait = max(0, t_batch - t_instance_queue) + t_scheduling + t_instance_queue`

### Still Open (from March 27 review)
- Balance score needs projected memory gain (kv_bytes_per_token per model)
- RSO sampling in benchmark (independent 10% per field)
- KNN-only model estimator (single call → point + bucket)
- Estimator sub-breakdown timing
- Tiered filtering paper discussion + ablation

---

## 1. Implementation Summary

### What was done
Closed all 7 gaps between the CARA paper (`cara_new.tex`) and the scheduling code (`cara_serve.py`). Before this session, the code was a per-request dispatcher with 3-term scoring and hardcoded normalization. After this session, it matches the paper's batch-oriented LPT scheduler with 4-term scoring, per-request min-max normalization, balance tracking, tiered constraints, and sidecar latency prediction.

### Phases implemented (in order)

**Phase 0: ModelEstimator** (new file)
- Abstract `ModelEstimator` interface + `DefaultModelEstimator` combining ModernBERT bucket classifiers (length) and KNN (quality)
- Runs ONCE per prompt at scheduler side, before instance-level scheduling
- `/v1/estimate` REST endpoint for debugging and external access
- Score type configurable: `judge` / `similarity` / `combined`
- KNN extensions: `predict_similarity()` and `predict_judge()` methods added to `KNNEstimator`

**Phase 1: Batch Scheduling Core** (Gap #1 + #7)
- `BatchQueue` class: async queue with configurable timeout and max batch size
- `BatchRequest` dataclass: wraps HTTP request + asyncio.Future for response delivery
- LPT sort: requests sorted by predicted output tokens (descending) before scheduling
- LPT sort key configurable: `max` / `min` / `mean` across models via `slo_defaults.lpt_sort_key`
- `_batch_scheduler_loop()`: background task collecting batches and scheduling
- Local state updates: `_update_local_stats()` propagates assignments within a batch so later requests see load from earlier ones
- Adaptive batch sizing: `_adapt_batch_params()` adjusts max_batch_size and timeout based on cluster load ratio (busy instances / total)
- Old `_schedule_cara()` unified to use `_schedule_cara_batch()` with batch_size=1

**Phase 2: Balance Term + Per-Request Normalization** (Gap #3 + #4)
- `w_balance` (w_u) weight added to scoring: `score = w_lat * lat + w_cost * cost - w_qual * qual + w_u * balance`
- Balance delta: `max(0, (util_i - mean_util) / (mean_util + 1e-6))` where util = (running + waiting) / capacity
- Mean utilization recomputed after each assignment in batch via `_compute_mean_utilization()`
- Per-request min-max normalization replaces hardcoded `ttft/10.0` and `cost/0.05`
- Two-pass scoring: Pass 1 collects all candidate predictions, computes min/max; Pass 2 normalizes and scores

**Phase 3: E2E Soft Scoring** (Gap #5)
- Two-tier latency as described in paper:
  - Tier 1 (hard filter): TTFT + TPOT checked against SLOs (no length dependency)
  - Tier 2 (soft score): E2E latency = TTFT + expected_length * TPOT used for scoring
- No separate E2E XGBoost model needed; computed from existing TTFT + TPOT predictions

**Phase 4: Tiered Constraint Handling** (Gap #6)
- Three modes: `STRICT` (current behavior), `RELAXED` (skip all filters), `TIERED` (relax one at a time)
- `relax_order` configurable: default `["ttft", "tpot", "quality", "budget"]`
- When TIERED and all candidates filtered, relaxes constraints in order until candidates found
- Falls back to full RELAXED if all individual relaxations fail

**Phase 5: Sidecar /predict_latency Endpoint** (Gap #2)
- `/predict_latency` added to `cara_predictor_api_server.py`
- Sidecar fetches local instance_stats (no network hop), runs XGBoost TTFT prediction
- Scheduler calls sidecar first via `_call_sidecar_predict()`, falls back to local prediction
- Matches paper's scalability argument (local prediction, no network overhead for instance state)

### Key unification
- Baseline strategies (random, round_robin, shortest_queue, quality_greedy, cost_greedy, length_aware) keep their per-request path unchanged
- Only `scheduling == "cara"` activates batch scheduling
- Clean A/B comparison preserved for experiments

---

## 2. Files Modified/Created

### New files
| File | Description |
|------|-------------|
| `block/predictor/cara/model_estimator.py` | `ModelEstimator` ABC + `DefaultModelEstimator` (275 lines) |
| `tests/global_scheduler/cara/test_batch_scheduling.py` | 26 unit tests (636 lines) |
| `block/config/cara/predictor_config_smoketest.json` | Smoke test predictor config (2 models, d7525) |
| `block/config/cara/model_config_smoketest.json` | Smoke test model config (7B + 3B on d7525) |
| `block/config/cara/model_deployment_smoketest.json` | Deployment config with actual CloudLab hostnames |
| `block/exp/cara/start_predictor.sh` | Predictor sidecar start script |
| `block/exp/cara/start_scheduler.sh` | Scheduler start script |

### Modified files
| File | Changes |
|------|---------|
| `block/global_scheduler/cara/cara_serve.py` | `BatchQueue` (line 99), `BatchRequest` (line 60), `_schedule_cara_batch()` (line 568), `_batch_scheduler_loop()` (line 780), 4-term scoring with normalization, tiered constraints, sidecar latency calls, `/v1/estimate` endpoint, `/v1/batch_stats` endpoint, adaptive batch sizing |
| `block/predictor/cara/estimators/knn_estimator.py` | Added `predict_similarity()` (line 181), `predict_judge()` (line 192) |
| `block/predictor/cara/cara_predictor_api_server.py` | Added `/predict_latency` endpoint (line 127) |
| `cara_paper/plans/paper_vs_code_gaps.md` | Updated status: all 7 gaps marked IMPLEMENTED |

---

## 3. Design Decisions

### LPT sort key: max across models (conservative)
- `req.predicted_tokens = max(est.length_expected for est in req.model_estimates.values())`
- Preserves LPT invariant: longest potential output scheduled first regardless of which model ultimately serves it
- Configurable via `slo_defaults.lpt_sort_key`: `max` (default), `min`, `mean`

### ModelEstimator runs once per prompt, not per instance
- Quality and length depend only on (prompt, model), not instance state
- Eliminates duplicate ModernBERT/KNN loading across 18 instances
- XGBoost TTFT/TPOT stays per-instance (needs instance state)

### Score type configurable
- `model_estimator.score_type`: `judge` (LLM judge scores), `similarity` (embedding similarity), `combined` (weighted average from KNN)
- Default: `judge` (matches paper)

### Fused bucket classifiers
- Single ModernBERT checkpoint per target model works for all target LLMs
- No need for per-instance classifiers

### Sidecar-first latency
- Scheduler calls sidecar `/predict_latency`, falls back to local XGBoost
- Sidecar has zero-network-cost access to instance_stats
- Config: `predictor_architecture: "scheduler"` or `"sidecar"`

### Baseline strategies unaffected
- Only `cara` strategy uses batching
- All 6 baselines (random, round_robin, shortest_queue, quality_greedy, cost_greedy, length_aware) keep per-request path

---

## 4. Config Reference

### predictor_config_smoketest.json structure

```json
{
  "predictor_type": "learned",
  "schedule_trace_timeout": 5,
  "enable_data_collection": false,

  "model_estimator": {
    "type": "default",
    "score_type": "judge",
    "bucket_config": {
      "model_dir": "/users/asdwb/Block/models/cara/bucket_classifier",
      "model_map": {
        "Qwen/Qwen2.5-3B": "Qwen_Qwen2.5-3B_length_bucket",
        "Qwen/Qwen2.5-7B": "Qwen_Qwen2.5-7B_length_bucket"
      },
      "bucket_size": 64,
      "max_buckets": 16,
      "max_length": 1024
    },
    "quality_config": {
      "type": "knn",
      "model_dir": "/users/asdwb/Block/models/cara/knn",
      "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
      "k": 10
    }
  },

  "scoring_weights": {
    "w_latency": 0.3,
    "w_cost": 0.2,
    "w_quality": 0.3,
    "w_balance": 0.2
  },

  "slo_defaults": {
    "ttft_slo_ms": 5000,
    "tpot_slo_ms": 200,
    "budget_tokens": 256,
    "quality_min": 0.3,
    "budget_confidence_threshold": 0.5,
    "constraint_mode": "STRICT",
    "relax_order": ["ttft", "tpot", "quality", "budget"],
    "lpt_sort_key": "max",
    "batch_config": {
      "max_batch_size": 8,
      "batch_timeout_ms": 100,
      "adaptive_sizing": true
    },
    "instance_capacity": {
      "qwen2.5-3b_d7525": 10,
      "qwen2.5-7b_d7525": 6
    }
  },

  "xgboost_ttft": {
    "model_dir": "/users/asdwb/Block/models/cara/xgboost_e2e/xgboost_e2e_actual"
  },

  "tpot_lookup": {
    "qwen2.5-3b_d7525": 0.025,
    "qwen2.5-7b_d7525": 0.035
  },

  "instance_metadata": {
    "qwen2.5-3b_d7525": {
      "model_name": "Qwen/Qwen2.5-3B",
      "gpu_type": "a30",
      "cost_per_token": 0.002
    },
    "qwen2.5-7b_d7525": {
      "model_name": "Qwen/Qwen2.5-7B",
      "gpu_type": "a30",
      "cost_per_token": 0.003
    }
  }
}
```

### New config fields explained

| Field | Location | Description |
|-------|----------|-------------|
| `model_estimator` | top-level | ModelEstimator configuration (type, score_type, bucket_config, quality_config) |
| `model_estimator.score_type` | model_estimator | Quality score source: "judge", "similarity", or "combined" |
| `scoring_weights.w_balance` | scoring_weights | Weight for cluster balance term (default 0.2) |
| `slo_defaults.constraint_mode` | slo_defaults | "STRICT", "RELAXED", or "TIERED" (default "STRICT") |
| `slo_defaults.relax_order` | slo_defaults | Order to relax constraints in TIERED mode |
| `slo_defaults.lpt_sort_key` | slo_defaults | LPT sort key across models: "max", "min", "mean" (default "max") |
| `slo_defaults.batch_config` | slo_defaults | Batch scheduling params: max_batch_size, batch_timeout_ms, adaptive_sizing |
| `slo_defaults.instance_capacity` | slo_defaults | Max concurrent requests per instance type (for balance computation) |

---

## 5. Unit Tests

26 tests in `tests/global_scheduler/cara/test_batch_scheduling.py`, all passing.

| Category | Count | Tests |
|----------|-------|-------|
| BatchQueue | 7 | enqueue/dequeue, timeout, max batch size, min batch size, concurrent enqueue, empty queue, ordering |
| LPT Sort | 2 | descending sort, configurable sort key (max/min/mean) |
| Local State Updates | 2 | stats propagation within batch, mean util update |
| Mean Utilization | 2 | computation, edge cases (empty cluster) |
| Normalization | 1 | per-request min-max normalization produces [0,1] values |
| Tiered Constraints | 3 | STRICT (filter all), TIERED (relax one), RELAXED (skip all) |
| Balance Term | 1 | loaded instance gets higher penalty than idle |
| E2E Soft Scoring | 1 | Tier 2 uses TTFT + length * TPOT, not just TTFT |
| ModelEstimator | 3 | interface contract, DefaultModelEstimator with KNN fallback, score_type switching |
| KNN Extensions | 4 | predict_similarity, predict_judge, predict_all_models, distance weighting |

Run tests:
```bash
cd /home/wd312/Code/llm/Block
python -m pytest tests/global_scheduler/cara/test_batch_scheduling.py -v
```

---

## 6. CloudLab Smoke Test (March 26, 2026)

### Infrastructure
- 2x A30 (d7525) nodes reserved for 6 hours on Wisconsin CloudLab
- Node 0 (`d7525-10s10337.wisc.cloudlab.us`): Qwen2.5-7B vLLM + sidecar predictor + CARA scheduler
- Node 1 (`d7525-10s10333.wisc.cloudlab.us`): Qwen2.5-3B vLLM + sidecar predictor

### Setup steps
1. `setup.sh` ran successfully (CUDA 12.8 + nvidia-open + vLLM cara_v_11)
2. Models uploaded to `/users/asdwb/Block/models/cara/`:
   - KNN: 23MB (knn directory with embeddings + indices)
   - XGBoost: 3MB + d7525 symlinks (see known issues)
   - Bucket classifiers: 1.1GB (ModernBERT checkpoints, tokenizer issue)

### Deployment
- `deploy_cara.py` with `--fresh-deploy` succeeded
- vLLM instances started on both nodes
- Predictor sidecars started via `start_predictor.sh`
- Scheduler started via `start_scheduler.sh`

### Verification
- `/predict_latency` sidecar endpoint: working (XGBoost TTFT + TPOT lookup)
- `/v1/estimate` scheduler endpoint: working (KNN fallback for length since bucket classifiers failed to load)
- First E2E request: batch=1, 7B instance filtered (TTFT > SLO at current load), 3B selected, correct response returned
- Adaptive sizing observed: max_batch 8 -> 4, timeout 100 -> 50ms (low load regime)

---

## 7. Feasibility Test Results (March 26 evening session)

### Estimator optimization progression
| Version | Per-prompt latency | Speedup |
|---------|-------------------|---------|
| Original (sequential, redundant KNN) | 623ms | 1x |
| Fix redundant embedding calls | 200ms | 3.1x |
| + FAISS Flat index | 131ms | 4.7x |
| + Batched inference (batch=8) | **27ms** | **23x** |

### Component latency breakdown (warm, on A30 CPU)
| Component | Latency |
|-----------|---------|
| Sentence-transformer embed (MiniLM-L6) | 6.5ms |
| Numpy search (15k vectors) | 0.25ms |
| FAISS Flat search | 1.3ms |
| FAISS IVF-PQ search | 0.06ms |
| KNN score aggregation (4 models) | 0.13ms |
| ModernBERT tokenize | 0.16ms |
| ModernBERT forward pass | 54.8ms |
| XGBoost predict | 0.43ms |

### Key optimizations applied
1. **Single KNN embedding** — was calling `_find_neighbors()` per model (5 calls), now once
2. **FAISS index** — flat (exact) and IVF-PQ (approximate) support added to KNNEstimator
3. **Batched `estimate_batch()`** — sentence-transformer and ModernBERT process N prompts in one forward pass
4. **Parallel ModernBERT** — ThreadPoolExecutor runs bucket classifiers for different models concurrently

### Architecture decision: single batch boundary
The estimator is embedded in the CARA scheduler, NOT a separate service. This avoids nested batch queues (CARA batch → estimator batch) which would cause "batch bubbles" — double waiting at low QPS. Instead, the CARA batch collects N requests, calls `estimate_batch(N prompts)` directly. The batch amortizes prediction overhead: 212ms total / 8 prompts = 27ms/prompt.

### Feasibility test: 7 schedulers comparison
| Scheduler | QPS=2 Thpt | QPS=5 Thpt | Mean TTFT | Mean E2E | Model Split |
|-----------|-----------|-----------|-----------|----------|-------------|
| shortest_queue | 1.77 | **3.98** | 25ms | **1653ms** | 3B:32/7B:18 |
| cost_greedy | 1.82 | 3.97 | 21ms | **1063ms** | 3B:50 |
| random | 1.70 | 3.69 | 27ms | 1784ms | 3B:29/7B:21 |
| round_robin | 1.77 | 3.37 | 27ms | 2178ms | 3B:25/7B:25 |
| quality_greedy | 1.69 | 3.33 | 41ms | 3034ms | 7B:50 |
| cara (optimized) | 1.72 | 2.72 | 23ms | 5908ms | 3B:49/7B:1 |
| length_aware | 1.61 | 1.83 | 349ms | 2933ms | 3B:42/7B:8 |

### /v1/config runtime update tested
Weight sweeps and batch config sweeps completed without scheduler restart.

---

## 7b. Known Issues (Updated)

### Bucket classifiers: tokenizer incompatibility — FIXED
- Error: `Tokenizer class TokenizersBackend does not exist`
- **Fix applied**: Changed `tokenizer_class` from `TokenizersBackend` to `PreTrainedTokenizerFast` in saved checkpoint `tokenizer_config.json`
- Bucket classifiers now load: 2/2 working, 16-bucket probability distributions

### XGBoost instance type mismatch
- XGBoost models trained with instance types `qwen2.5-7b_a30`, `qwen2.5-3b_a30`
- CloudLab d7525 nodes report as `d7525` not `a30`
- Fix applied: symlinks (`qwen2.5-7b_d7525.xgb -> qwen2.5-7b_a30.xgb`) + metadata JSON update
- Proper fix: add d7525 instance types to XGBoost training pipeline

### Missing packages on fresh CloudLab
- `setup.sh` does not install: xgboost, sentence-transformers, tokenizers
- These are needed by the predictor sidecar and scheduler
- Fix: add to `setup.sh` pip install list for future deployments

---

## 7c. Quality Evaluation Framing (for paper)

**Why no ground-truth best-model accuracy**: Quality labels are LLM-judge predictions, not ground truth. Different judges rank models differently. Under load, the "best" model changes with SLO context (a 72B model with 5x worse TTFT may not be "best" for a latency-sensitive request). There is no context-free ground truth for quality-aware scheduling.

**Metrics we report instead:**
1. **Average predicted quality** — measures if scheduler successfully selects higher-quality models when possible
2. **Quality-weighted goodput** — requests meeting ALL SLOs whose selected model's predicted quality >= threshold
3. **Predicted best-model hit rate** — did scheduler pick the model that KNN predicts is best? (Note: biased toward larger models)
4. **All-model score distribution** — per-request scores for all models, enables post-hoc analysis

**Paper text (draft for evaluation section):**
> Unlike classification tasks where ground-truth labels exist, quality-aware scheduling operates over *predicted* quality scores that are themselves model-dependent (LLM judge, embedding similarity, or human preference). We evaluate quality through two complementary metrics: (1) the average predicted quality of selected instances, which measures whether the scheduler successfully trades off quality against latency and cost constraints, and (2) quality-weighted goodput, which counts only SLO-compliant requests whose selected model's quality exceeds a minimum threshold. We do not report best-model oracle accuracy because the "best" model is itself a function of the quality predictor and SLO context.

**Response fields added (March 26):**
- `predicted_quality`: quality score of the selected model (from KNN/judge)
- `predicted_length`: predicted output length for selected model
- `all_model_scores`: {model_name: quality_score} for all candidates
- `predicted_best_model`: model with highest predicted quality
- `predicted_best_hit`: whether scheduler picked that model

---

## 8. Next Steps

### Immediate (before full CloudLab reservation April 26)
1. Fix tokenizers version for bucket classifiers (either upgrade on CloudLab or re-save checkpoints)
2. Add xgboost, sentence-transformers, tokenizers to `setup.sh`
3. Add d7525 instance types to XGBoost training data (or keep symlink approach)

### Full evaluation (April 26 - May 6, 18-node heterogeneous cluster)
1. Run full E2E benchmark: 7 schedulers x QPS 5,10,15,20 x 500 requests
2. LPT sort key ablation: max vs min vs mean (heterogeneous cluster only)
3. Scoring weight sweep: vary w_balance (0.0, 0.1, 0.2, 0.3) with others fixed
4. Constraint mode comparison: STRICT vs TIERED
5. Sidecar vs scheduler-side latency prediction overhead comparison
6. **Overhead breakdown chart** for paper: batch_wait + estimator + stats_fetch + scoring
7. **ONNX export** of ModernBERT for further latency reduction (55ms → ~15ms projected)
8. Consider **Triton Inference Server** for production deployment of estimator models

### Paper updates (done March 26-27)
- Algorithm 1 pseudocode: LPT sort key = max across models (done)
- Section 4: configurable sort key, batch amortization (done)
- Section 4: 6 batch benefits paragraph — normalization, amortization, thundering herd, adaptive sizing (done)

### Paper updates (still needed)
- Overhead breakdown figure showing per-component latency
- Discussion of single-batch-boundary design vs nested batching
- FAISS index as system optimization
- Reference_score quality evaluation methodology (new)
- Model study section: add reference_score target results

---

## 10. March 27 Evening Session — Reference Similarity

### What was done
- **Reference tracing verified**: source IDs in `cara_v3_all_training_fixed.json` correctly map to HF dataset indices (100% match on all 5 verified datasets)
- **`add_reference_similarity.py` created**: loads references from all 7 HF source datasets, computes cosine similarity per model response
- **`collect_data.py` updated**: GSM8K, SQuAD, LMSYS loaders now extract `reference` field for future collections
- **KNN + ModelEstimator updated**: `reference_similarity_scores` array, `predict_reference_similarity()`, `score_type="reference"`
- **Full run on GPU VM**: 19,727/19,727 references loaded, 6/7 datasets have reference_similarity (all except beaver_tails)
- **Results**: train 12,645 with ref / test 3,041 with ref

### Key findings
- **GSM8K**: embedding similarity (0.5-0.6) is poor for factual answers — need exact-match accuracy instead
- **code_ultra_feedback**: high ref_sim (0.90-0.95) — references are highest-rated code from 4 model responses
- **reward_bench**: 7B notably lower ref_sim (0.77) — meaningful quality signal
- **Single-batch-boundary**: confirmed best design — no nested batching (avoids "batch bubble")
- **Estimator optimized to 27ms/prompt** via batched BERT + FAISS (was 623ms)

### Naming decision
Rename `reference_similarity` → **`reference_score`** — unified field where:
- GSM8K: exact-match accuracy (0 or 1)
- SQuAD: F1 score against answer span
- beaver_tails: protectai safety score (refusal=1, compliance=0)
- Others: embedding similarity to reference

### Data locations
- `data/cara/training_data_with_ref/train_fixed.jsonl` (local + GPU VM) — 12,645 entries with ref
- `data/cara/training_data_with_ref/test_fixed.jsonl` (local + GPU VM) — 3,041 entries with ref
- Comprehensive experiment results: `experiment_output/comprehensive_results/` (32 files)
- Feasibility experiment results: `experiment_output/feasibility_results/` (27 files)

### Next session execution plan
See `memory/reference_similarity_plan.md` for detailed 11-step plan:
1. Fix reference_score (GSM8K exact-match, SQuAD F1, beaver_tails safety)
2. HF upload with updated datacard
3. Update training scripts
4. KNN rebuild
5. GPU VM debugging (RoBERTa, length exploration)
6-8. CSD3 INTR × 3 windows (ModernBERT fused+per-model, Qwen-LoRA with resume)
9. Update paper model study
10. Submit long sbatch for winning config
11. Offline CloudLab analysis

---

## 9. Quick Start Commands

### Generate configs (on VPS/local, before deploying)
```bash
python3 block/exp/generate_config.py \
  --manifest_path block/cl_manifest_smoketest.xml \
  --num_predictors 1
```

### Setup CloudLab node (from repo root on each node)
```bash
bash block/exp/cara/setup.sh
```

### Deploy all services
```bash
python3 block/exp/cara/deploy_cara.py \
  --config block/config/cara/model_config_smoketest.json \
  --output block/config/cara/model_deployment_smoketest.json \
  --hosts block/config/hosts \
  --host-config block/config/host_configs.json \
  --predictor-type learned \
  --learned-predictor-config block/config/cara/predictor_config_smoketest.json \
  --fresh-deploy
```

### Start predictor sidecar (on each node)
```bash
bash ~/Block/block/exp/cara/start_predictor.sh <hostname> <instance_type>
# Example:
bash ~/Block/block/exp/cara/start_predictor.sh d7525-10s10337.wisc.cloudlab.us qwen2.5-7b_d7525
```

### Start scheduler (on scheduler node)
```bash
bash ~/Block/block/exp/cara/start_scheduler.sh cara
```

### Test endpoints
```bash
# ModelEstimator
curl -X POST http://localhost:8200/v1/estimate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Hello, how are you?","budget_tokens":256}'

# Batch stats
curl http://localhost:8200/v1/batch_stats

# Sidecar latency prediction
curl -X POST http://<node>:8300/predict_latency \
  -H "Content-Type: application/json" \
  -d '{"num_prompt_tokens":50,"num_predicted_output_tokens":128}'

# Full completion (triggers batch scheduling)
curl -X POST http://localhost:8200/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Explain quantum computing","max_tokens":256,"model":"Qwen/Qwen2.5-3B"}'
```

### Run unit tests
```bash
cd /home/wd312/Code/llm/Block
python -m pytest tests/global_scheduler/cara/test_batch_scheduling.py -v
```

---

## 10. Architecture Diagram

```
Request --> /v1/completions
               |
               v
          BatchQueue
          (collect batch, timeout/max_size)
               |
               v
        ModelEstimator.estimate()        <-- ONCE per prompt
        (KNN quality + ModernBERT length)
               |
               v
         LPT Sort (descending by max predicted length)
               |
               v
     For each request in batch:
        |
        +-- For each instance candidate:
        |     |
        |     +-- Sidecar /predict_latency  <-- per-instance (XGBoost TTFT + TPOT)
        |     |   (fallback: scheduler-side prediction)
        |     |
        |     +-- Tier 1: SLO filter (TTFT < ttft_slo, TPOT < tpot_slo)
        |     +-- Tier 1: Quality filter (score >= quality_min)
        |     +-- Tier 1: Budget filter (p_under_budget >= threshold)
        |     |
        |     +-- Tier 2: E2E soft score
        |           lat_norm = normalize(TTFT + length * TPOT)
        |           cost_norm = normalize(length * cost_per_token)
        |           qual_norm = normalize(quality_score)
        |           bal_norm = max(0, (util_i - mean_util) / mean_util)
        |           score = w_lat*lat - w_qual*qual + w_cost*cost + w_bal*bal
        |
        +-- Select instance with LOWEST score
        +-- Update local stats (propagate within batch)
        +-- Dispatch request to selected instance
        +-- Resolve future (return response to caller)
```

---

## 11. Implementation Plan Reference

The full implementation plan is at: `/home/wd312/.claude/plans/atomic-prancing-tiger.md`

The gap analysis (now marked complete) is at: `/home/wd312/Code/llm/Block/cara_paper/plans/paper_vs_code_gaps.md`
