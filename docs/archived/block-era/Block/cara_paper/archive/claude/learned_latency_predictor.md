# Learned Latency Prediction for Heterogeneous LLM Serving

## Core Argument

### The Observation
LLM serving latency is a **deterministic function** of observable instance state and request characteristics. Given the same:
- Queue state (num_running, num_waiting, pending tokens)
- Hardware state (KV cache utilization, free blocks)
- Request properties (prompt tokens, max output tokens)
- Service rates (EMA decode tok/s, prefill tok/s)

...the hardware produces the same end-to-end latency, regardless of how that state was reached (Poisson arrivals, bursty gamma arrivals, or any other pattern).

### The Analogy: Learned Index Structures
This observation parallels the insight behind learned index structures (Kraska et al., SIGMOD 2018): traditional B-trees implement a hand-crafted function mapping keys to positions, but this function can be approximated by a learned model that captures the data distribution directly.

Similarly, traditional LLM serving latency prediction relies on:
- **Analytical models**: Roofline models computing prefill_time + decode_time + overhead
- **Simulation-based models**: Digital twins (e.g., Vidur in Block) that replay the scheduler/memory allocator behavior
- **Profiling tables**: Pre-measured latency lookup tables indexed by (batch_size, seq_len)

We observe that all these approaches attempt to approximate the same underlying function:

```
f(instance_state, request_shape) → e2e_latency
```

A learned model can approximate this function directly from observed (state, latency) pairs, without modeling the internal mechanics of the serving engine.

## Why Learned Prediction Works

### 1. Deterministic Hardware Behavior
Given identical GPU state and request characteristics, the computation is deterministic:
- Prefill time ≈ linear in prompt tokens (compute-bound)
- Decode time ≈ linear in output tokens × per-token latency (memory-bandwidth-bound)
- Queuing delay ≈ determined by current queue depth and service rate

The only source of noise is OS-level scheduling jitter and memory allocation variance, which are small relative to the signal.

### 2. Observable State is Sufficient
The instance state captured by the sidecar predictor contains all information needed for latency prediction:
- **Queue features**: num_running, num_waiting, pending_prefill_tokens, pending_decode_tokens
- **Capacity features**: kv_cache_utilization, kv_free_blocks, max_num_seqs
- **Rate features**: ema_decode_tok_per_s, ema_prefill_tok_per_s, ema_decode_iter_ms
- **Request features**: num_prompt_tokens, num_predicted_output_tokens

These features form a complete description of the system state relevant to latency prediction.

### 3. Generalization Across Arrival Patterns
A key property: the model learns the mapping from **state to latency**, not from **arrival pattern to latency**. This means:
- Train on Poisson arrivals at 10 QPS levels → covers idle to saturated states
- Deploy under any arrival pattern (gamma bursts, time-varying rates)
- As long as the resulting instance state falls within the training distribution, predictions are accurate

This is analogous to how a learned index trained on one key distribution can serve queries for any access pattern — the model captures the data structure, not the query workload.

## Advantages Over Prior Approaches

### vs. Analytical Models (Roofline)
| Aspect | Roofline | Learned |
|--------|----------|---------|
| Accuracy | Approximate (ignores queuing, batching effects) | Captures real behavior |
| Hardware-specific | Requires per-GPU calibration | Learns from data automatically |
| Deployment config | Must model enforce-eager, TP, chunked prefill | Captured implicitly |
| Development cost | Manual modeling per serving engine | Train once, works for any engine |

### vs. Simulation (Block/Vidur)
| Aspect | Simulation | Learned |
|--------|------------|---------|
| Accuracy | High but requires exact simulator | Comparable, from real data |
| Overhead | Heavy (runs full simulator per request) | Lightweight (single model inference) |
| Maintenance | Must update simulator for each vLLM version | Retrain on new data |
| Heterogeneous | Needs separate simulator per GPU type | Learns per-instance automatically |

### vs. Profiling Tables
| Aspect | Profiling | Learned |
|--------|-----------|---------|
| Granularity | Fixed (batch_size, seq_len) buckets | Continuous (any state) |
| Queue effects | Not captured | Captured via queue state features |
| Interpolation | Linear between buckets | Non-linear learned interpolation |
| Scalability | Exponential in features | Constant model size |

## Model Design

### Feature Engineering (23 features for XGBoost)

**Instance state features (17):**
1. num_running — current running requests
2. num_waiting — queued requests
3. num_active_decode_seqs — actively decoding sequences
4. decode_ctx_p50/p95/max — decode context length distribution
5. pending_prefill_tokens — total prefill tokens in queue
6. pending_decode_tokens — total decode tokens in queue
7. kv_cache_utilization — fraction of KV cache in use
8. kv_free_blocks — available KV cache blocks
9. token_budget_per_iter — scheduler token budget
10. prefill_chunk_size — chunked prefill size
11. max_num_seqs — max concurrent sequences
12. num_preempted — preempted sequence count
13. ema_decode_tok_per_s — exponential moving average decode throughput
14. ema_prefill_tok_per_s — EMA prefill throughput
15. ema_decode_iter_ms — EMA per-decode-iteration latency
16. kv_evictions_per_s — KV cache eviction rate

**Request features (2):**
17. num_prompt_tokens — input prompt length
18. num_predicted_output_tokens — max output tokens

**Derived features (4):**
19. total_pending_tokens = pending_prefill + pending_decode
20. queue_depth = num_running + num_waiting
21. estimated_prefill_time = prompt_tokens / ema_prefill_tok_per_s
22. estimated_decode_time = output_tokens / ema_decode_tok_per_s

### Model Choices

**XGBoost (Primary)**:
- Per (model_size, GPU_type) ensemble — captures hardware-specific behavior
- Gradient boosted trees handle non-linear feature interactions
- Fast inference (<1ms per prediction)
- Feature importance reveals which state variables matter most
- Robust to outliers, no normalization needed

**LSTM (Sequence)**:
- Sliding window over recent schedule states per instance
- Captures temporal patterns (queue building up, batch effects)
- Higher overhead but potentially more accurate under load transitions
- Complementary to XGBoost's point-in-time prediction

**Linear Regression (Baseline)**:
- 3 features: prompt_tokens, output_tokens, num_waiting
- Establishes lower bound for comparison
- Fast, interpretable, but ignores interaction effects

## Training Data

### Collection Method
- Real prompts from training data (not dummy tokens)
- Random scheduling across 18 heterogeneous instances
- 10 QPS levels (9-36) covering idle to saturated
- 2 sweeps × 10 QPS × 20,000 requests = 400,000 records
- Sidecar on each instance captures (state → latency) pairs

### Data Properties
- **Diverse states**: QPS=9 (low load, short queues) to QPS=36 (high load, long queues)
- **Diverse hardware**: A100 (72B), V100 (14B), A30 (7B/3B), P100 (3B)
- **Natural responses**: Real prompts produce natural output lengths (98% natural stopping)
- **Rich features**: 20 schedule_state fields + request properties + actual latency

### Generalization Argument
The training data covers the full spectrum of instance states:
- **Idle state**: QPS=9, num_running ≈ 0-1, kv_utilization ≈ 0
- **Moderate load**: QPS=18-24, num_running ≈ 2-5, some queuing
- **Saturated**: QPS=33-36, num_running ≈ max, significant queuing

Any deployment scenario that produces instance states within this range will be predicted accurately, regardless of the arrival pattern that produced those states.

## Ablation Studies (Proposed)

### A1: Arrival Pattern Generalization
- **Train**: Poisson arrivals (current data)
- **Test**: Gamma arrivals (bursty), deterministic arrivals (fixed interval)
- **Measure**: MAPE change — should be minimal if the learned function argument holds
- **Expected result**: <5% MAPE increase, confirming state-based prediction generalizes

### A2: Predictor Ablation
- Remove each feature group, measure accuracy impact:
  - Without queue features → how much does queuing state matter?
  - Without EMA rates → are historical rates important?
  - Without KV cache features → does memory state matter?
  - Without derived features → does feature engineering help?

### A3: Model Comparison
- XGBoost vs LSTM vs Linear vs Roofline vs Simulation
- Accuracy (MAE, MAPE) vs inference overhead (ms per prediction)
- Shows learned models achieve better accuracy/overhead tradeoff

### A4: Training Data Size
- Train on 50k, 100k, 200k, 400k records
- At what point does more data stop helping?
- Shows diminishing returns and practical data collection requirements

### A5: Cross-QPS Generalization
- Train on subset of QPS levels (e.g., 9, 18, 27, 36)
- Test on held-out levels (12, 15, 21, 24, 30, 33)
- Measures interpolation accuracy between observed load levels

## Connection to Quality-Aware Scheduling

The latency predictor is one component of Cara's multi-objective scheduler:

```
score(request, instance) = α · latency_term + β · cost_term + γ · quality_term
```

Where:
- `latency_term`: Uses the learned latency predictor (this section)
- `quality_term`: Uses the learned quality predictor (ModernBERT/Qwen LoRA)
- `cost_term`: Based on GPU-hours consumed

The scheduler queries all candidate instances, predicts latency and quality for each, and routes to the instance that optimizes the scalarized objective while satisfying SLO constraints.

The learned latency predictor enables this by providing **fast, accurate, per-instance latency estimates** — something analytical models struggle with in heterogeneous clusters where each instance has different hardware, model size, and current load.

## Key Findings (to be updated with final results)

- XGBoost achieves [X] MAE, [Y] MAPE for latency prediction (vs roofline: [A] MAE)
- Model generalizes across arrival patterns: [Z]% MAPE under gamma arrivals
- Feature importance: queue depth and EMA service rates are top predictors
- Per-instance models outperform single global model by [W]%
- Training data beyond [N]k records shows diminishing returns
- Prediction overhead: <1ms per request (negligible vs actual latency)

## References

- Kraska et al., "The Case for Learned Index Structures", SIGMOD 2018
- Zhong et al., "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving", OSDI 2024
- Block paper (arXiv:2508.03611) — simulation-based prediction baseline
- Sequence Scheduling (NeurIPS) — LLM-based length prediction
