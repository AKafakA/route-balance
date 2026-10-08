# CARA Latency Predictor Design: Extensible Scheduler-Side Architecture

**Document Version**: 2.0
**Created**: January 23, 2026
**Updated**: January 25, 2026
**Status**: Implementation Phase

> **📖 vLLM API Specification**: See [INSTANCE_STATS_API.md](INSTANCE_STATS_API.md) for the finalized `/instance_stats` endpoint documentation.

---

## Executive Summary

This document describes the design for CARA's latency predictor, which predicts inference latency for routing decisions. The design pivots from the original LSTM-based sidecar approach to a **scheduler-side predictor** with **per-batch state queries**. The architecture supports **multiple predictor implementations** (XGBoost, LSTM, Roofline, Linear) for ablation studies.

### Key Design Decisions

| Aspect | Previous Design | New Design |
|--------|-----------------|------------|
| **Model** | LSTM (sequence model) | Extensible (XGBoost, LSTM, Roofline, Linear) |
| **Location** | Sidecar (instance-side) | Scheduler-side |
| **State Query** | Per-request | Per-batch |
| **Features** | Per-request details | Aggregated instance state |
| **Prediction** | Remote API call | Local inference |
| **Data Collection** | Coupled with predictor | Decoupled (separate service) |
| **Config** | Single monolithic file | Separate config per predictor type |

---

## 0. Extensible Predictor Architecture (NEW)

### 0.1 Design Principles

The predictor architecture is designed for **ablation studies** - comparing different prediction approaches while keeping the rest of the system unchanged.

**Key Principles**:
1. **Extensibility**: Support multiple predictor implementations (XGBoost, LSTM, Roofline, Linear)
2. **Decoupling**: Separate data collection from actual prediction
3. **Single-writer**: Sequential scheduling within batch means no predictor replicas needed
4. **Factory pattern**: Use `predictor_type` field to dispatch to correct implementation

### 0.2 Supported Predictor Types

| Type | Description | State API | Use Case |
|------|-------------|-----------|----------|
| `xgboost` | Tree-based ensemble | `INSTANCE_STATS` | Primary CARA predictor |
| `lstm` | Sequence model | `SCHEDULE_TRACE` | Research baseline |
| `roofline` | Analytical model | `INSTANCE_STATS` | Ablation baseline |
| `linear` | Simple linear regression | `INSTANCE_STATS` | Ablation baseline |
| `dummy` | Heuristic-based | `SCHEDULE_TRACE` | Data collection only |

### 0.3 State API Types

Each predictor declares which state API it requires:

```python
class StateApiType(Enum):
    SCHEDULE_TRACE = "schedule_trace"   # Detailed per-request data
    INSTANCE_STATS = "instance_stats"   # Aggregated stats (preferred)
```

**Mapping**:
- `SCHEDULE_TRACE`: Provides `running_requests[]`, `waiting_requests[]` with per-request details
- `INSTANCE_STATS`: Provides `num_running`, `num_waiting`, `decode_ctx_p95`, `pending_tokens`, etc.

### 0.4 Configuration Factory Pattern

Each predictor type has its own config file parsed by a factory:

```python
# block/predictor/cara/cara_predictor_config.py

PREDICTOR_CONFIG_REGISTRY: Dict[str, Type[CARABasePredictorConfig]] = {
    "dummy": DummyPredictorConfig,
    "xgboost": XGBoostPredictorConfig,
    "roofline": RooflinePredictorConfig,
    "lstm": LSTMPredictorConfig,
    "linear": LinearPredictorConfig,
}

def load_predictor_config(config_path: str) -> CARABasePredictorConfig:
    """Factory function to load predictor config."""
    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    predictor_type = config_dict.get("predictor_type")
    if predictor_type not in PREDICTOR_CONFIG_REGISTRY:
        raise ValueError(f"Unknown predictor_type: {predictor_type}")

    config_class = PREDICTOR_CONFIG_REGISTRY[predictor_type]
    # Filter to only fields defined in the config class
    valid_fields = {f.name for f in dataclasses.fields(config_class)}
    filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}
    return config_class(**filtered_dict)
```

### 0.5 Example Config Files

**XGBoost Predictor** (`predictor_config_xgboost.json`):
```json
{
  "predictor_type": "xgboost",
  "model_dir": "./models/cara/xgboost",
  "enable_data_collection": true,
  "data_collection_sample_rate": 1.0,
  "data_output_dir": "./training_data/cara"
}
```

**Roofline Predictor** (`predictor_config_roofline.json`):
```json
{
  "predictor_type": "roofline",
  "prefill_toks_per_sec": 10000,
  "decode_toks_per_sec": 500,
  "enable_data_collection": false
}
```

**LSTM Predictor** (`predictor_config_lstm.json`):
```json
{
  "predictor_type": "lstm",
  "model_path": "./models/cara/lstm/latency_predictor.pt",
  "hidden_size": 128,
  "num_layers": 2,
  "enable_data_collection": true
}
```

### 0.6 Instance Type Derivation

Instance type is derived at **deployment time** (not hardcoded in config):

```python
def derive_instance_type(model_name: str, backend: str, node_type: str) -> str:
    """Derive instance type identifier.

    Example: "qwen-2-5-3b_vllm_c240g5"
    """
    # Normalize model name
    model_part = model_name.lower().replace("/", "-").replace(".", "-")
    return f"{model_part}_{backend}_{node_type}"
```

**Usage in deployment**:
```python
# deploy_cara.py
for model, model_config in model_dict.items():
    for node_type, node_config in model_config["nodes"].items():
        instance_type = derive_instance_type(
            model_config["hf_model_name"],
            model_config["backend"],
            node_type
        )
        # Pass instance_type to data collection server
```

### 0.7 Decoupled Data Collection

Data collection is **decoupled** from scheduling policy. The instance-side service
is a **Data Collection Service**, NOT a predictor. It only records training data.

**Key Distinction**:
| Component | Location | Purpose |
|-----------|----------|---------|
| Data Collection Service | Instance-side | Record (state, request) → actual metrics |
| Latency Predictor | Scheduler-side | Predict latency for routing decisions |

**Current State**: `DummyCARAPredictor` performs data collection but uses predictor interface.

**Planned Refactoring** (Task #14): Rename to `DataCollectionService` with simplified interface:

```python
# block/predictor/cara/data_collection_service.py
class DataCollectionService:
    """Instance-side service for collecting training data only.

    NOT a predictor - just records (state, request) → actual metrics pairs.
    No state parameter needed since we're not making predictions.
    """

    async def record_request(self, request_id: str, num_prompt_tokens: int,
                             num_predicted_output_tokens: int):
        """Record incoming request and current backend state."""
        state = await self.schedule_client.fetch_schedule_trace()
        # or: state = await self.schedule_client.fetch_instance_stats()
        await self.data_collector.log_prediction(...)

    async def record_completion(self, request_id: str, e2e_latency: float,
                                ttft: float, tpot: float, actual_output_length: int):
        """Record actual metrics after completion."""
        await self.data_collector.log_actual_result(...)
```

**Architecture**:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Data Collection Service                       │
│  (block/predictor/cara/data_collection_service.py)              │
│                                                                  │
│  Endpoints:                                                      │
│  - POST /record_request  → Log request + query backend state    │
│  - POST /record_completion → Match with actual metrics          │
│  - POST /flush           → Write buffered data to disk          │
│                                                                  │
│  Output: training_data/{instance_type}_*.jsonl                  │
└─────────────────────────────────────────────────────────────────┘
                         ↑
                         │ (runs alongside vLLM backend)
                         │
┌─────────────────────────────────────────────────────────────────┐
│                    vLLM Backend Instance                         │
│                                                                  │
│  Endpoints:                                                      │
│  - GET /schedule_trace   → Per-request state (for LSTM training)│
│  - GET /instance_stats   → Aggregated state (for XGBoost training)│
└─────────────────────────────────────────────────────────────────┘
```

### 0.8 Port Configuration (Updated)

**Before** (in `host_configs.json`):
```json
{
  "hostname": {
    "ip_address": "192.168.1.1",
    "predictor_ports": [8100, 8101, 8102, 8103],  // Multiple ports, unclear naming
    "backend_port": 8000
  }
}
```

**After** (in `model_config_template.json`):
```json
{
  "model_name": {
    "hf_model_name": "Qwen/Qwen2.5-3B",
    "backend": "vllm",
    "nodes": {
      "c240g5": {
        "count": 4,
        "backend_port": 8000,
        "data_collection_port": 8100  // Single port, clear naming
      }
    }
  }
}
```

**Key Changes**:
- Renamed `predictor_ports` → `data_collection_port` (single port per instance)
- Moved from `host_configs.json` to `model_config_template.json`
- Clarifies that this is for data collection, not prediction

---

## 1. Architecture Overview

### 1.1 High-Level Architecture

```
┌───────────────────────────────────────────────────────────────────────┐
│                    CARA Global Scheduler                              │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │              Scheduler-Side Predictor Service                   │  │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐            │  │
│  │  │XGBoost Model │ │XGBoost Model │ │XGBoost Model │            │  │
│  │  │Instance A    │ │Instance B    │ │Instance C    │            │  │
│  │  │(3B/P100)     │ │(7B/A30)      │ │(72B/A100)    │            │  │
│  │  │              │ │              │ │              │            │  │
│  │  │Local State:  │ │Local State:  │ │Local State:  │            │  │
│  │  │ num_running  │ │ num_running  │ │ num_running  │            │  │
│  │  │ num_waiting  │ │ num_waiting  │ │ num_waiting  │            │  │
│  │  │ kv_util      │ │ kv_util      │ │ kv_util      │            │  │
│  │  └──────▲───────┘ └──────▲───────┘ └──────▲───────┘            │  │
│  └─────────┼────────────────┼────────────────┼─────────────────────┘  │
│            │                │                │                        │
│            │    Per-Batch Query (parallel)   │                        │
└────────────┼────────────────┼────────────────┼────────────────────────┘
             │                │                │
    ┌────────┴────┐   ┌───────┴─────┐   ┌──────┴──────┐
    │vLLM Instance│   │vLLM Instance│   │vLLM Instance│
    │     A       │   │     B       │   │     C       │
    │             │   │             │   │             │
    │/simple_     │   │/simple_     │   │/simple_     │
    │scheduler_   │   │scheduler_   │   │scheduler_   │
    │trace        │   │trace        │   │trace        │
    │             │   │             │   │             │
    │┌───────────┐│   │┌───────────┐│   │┌───────────┐│
    ││Data       ││   ││Data       ││   ││Data       ││
    ││Collector  ││   ││Collector  ││   ││Collector  ││
    │└───────────┘│   │└───────────┘│   │└───────────┘│
    └─────────────┘   └─────────────┘   └─────────────┘
```

### 1.2 Data Flow: Per-Batch Query

```
Batch of N requests arrives at scheduler
         │
         ▼
┌─────────────────────────────────────────────────────┐
│ 1. Query all instances in PARALLEL                  │
│    GET /instance_stats (one RTT, ~5-20ms)   │
└─────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│ 2. Update local state cache for all instances       │
│    (in-memory, ~μs)                                 │
└─────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│ 3. For each request in batch:                       │
│    - Compute features: state + request attrs        │
│    - XGBoost prediction (~μs per request)           │
│    - Get latency estimates for all instances        │
└─────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│ 4. Route each request to best instance              │
│    (min predicted latency or multi-objective)       │
└─────────────────────────────────────────────────────┘
         │
         ▼
Execute requests on assigned instances
```

---

## 2. Why This Design?

### 2.1 Why Tree-Based (XGBoost) Over Alternatives?

#### XGBoost vs LSTM (Sequence Model)

| Factor | XGBoost | LSTM |
|--------|---------|------|
| **Input shape** | Fixed-size feature vector | Variable-length (batch size varies) |
| **Training data needs** | Lower | Higher |
| **Inference speed** | ~μs | ~ms |
| **Interpretability** | High (feature importance) | Low |
| **Deployment complexity** | Simple (single file) | Complex (model + weights) |

#### XGBoost vs Simple Linear Regression

| Factor | XGBoost | Linear Regression |
|--------|---------|-------------------|
| **Non-linear effects** | ✅ Handles naturally | ❌ Cannot model |
| **Feature interactions** | ✅ Automatic (tree splits) | ❌ Requires manual terms |
| **Context length effects** | ✅ O(n²) attention captured | ❌ Only linear relationship |
| **Chunked prefill constraints** | ✅ Token budget interactions | ❌ Too simplistic |
| **Interpretability** | ✅ Feature importance | ✅ Coefficients |

**Why not simple linear regression** (as in the paper's "gray box" affine model)?

The paper proposes: `P_lat = α·N_in + β·L_kv + γ·(L_kv × I_prefill) + C`

This is too simplistic because:
1. **Attention is O(n²)** - context length effects are non-linear
2. **Chunked prefill constraints** - `token_budget_per_iter` creates complex interactions
3. **Batch composition matters** - decode_ctx_p95 vs mean have different effects
4. **Memory pressure is non-linear** - KV utilization near 1.0 causes preemption spikes

**XGBoost is "gray box" with more expressive power**:
- Features are still interpretable (we engineer them from `/instance_stats`)
- Model can learn non-linear relationships and feature interactions
- Unlike LSTM, we can inspect feature importance for debugging

**Key Insight**: With **chunked prefill enabled** across all instances:
- Latency is more **uniform and predictable**
- Per-request details matter less (prefill is chunked and interleaved)
- **Aggregate statistics** capture the key drivers

### 2.2 Why Per-Batch Query Over Alternatives?

Three options were considered:

| Option | Pros | Cons |
|--------|------|------|
| **Per-request query** | Most accurate | O(requests) network overhead |
| **Periodic broadcast + calibration** | Low overhead | Complex calibration model, state drift |
| **Per-batch query** ✅ | Fresh state when needed, simple | O(batches) overhead |

**Winner: Per-batch query** because:
1. Natural fit for CARA batch scheduling (requests routed concurrently)
2. No calibration complexity
3. Amortized cost: 1 query serves N requests in batch
4. State is accurate at decision time

### 2.3 Why Scheduler-Side Over Sidecar?

| Aspect | Sidecar (Instance-Side) | Scheduler-Side |
|--------|-------------------------|----------------|
| **Per-request latency** | +RTT per prediction | ~0 (local inference) |
| **Batch efficiency** | Query each instance for each request | Query once per batch |
| **State freshness** | Always fresh | Fresh at batch boundary |
| **Deployment** | N predictors (one per instance) | 1 predictor service |

---

## 3. Feature Engineering

> **Design Note**: This feature set is inspired by DuetServe's (q, c) batch modeling, adapted for
> routing-time prediction where we don't know the exact future batch composition.

### 3.1 Instance State Features (from `/instance_stats`)

```python
@dataclass
class SimpleScheduleState:
    """Aggregated state for latency prediction.

    Captures service rate, load distribution, and memory pressure.
    """

    # === SERVICE RATE SIGNALS (Critical!) ===
    # These collapse "hardware + saturation + kernel path" into usable numbers
    ema_decode_tok_per_s: float       # EMA of decode throughput (tokens/sec)
    ema_prefill_tok_per_s: float      # EMA of prefill throughput (tokens/sec)
    ema_decode_iter_ms: float         # EMA of decode iteration time (ms)

    # === BATCH STATE (counts) ===
    num_running: int                  # Running requests
    num_waiting: int                  # Pending requests
    num_active_decode_seqs: int       # Sequences in decode phase

    # === CONTEXT LENGTH DISTRIBUTION (attention cost drivers) ===
    # Attention is O(n²) in context length - utilization alone is insufficient
    decode_ctx_p50: int               # Median context length of decode seqs
    decode_ctx_p95: int               # 95th percentile context length
    decode_ctx_max: int               # Max context length in batch

    # === BACKLOG IN TOKENS (not just counts!) ===
    pending_prefill_tokens: int       # Total prefill tokens in queue
    pending_decode_tokens: int        # Total remaining decode tokens in queue
    # pending_requests_p95_prompt_len: int  # Optional: prompt length distribution

    # === SCHEDULER KNOBS (define processing envelope) ===
    token_budget_per_iter: int        # Max tokens per iteration
    prefill_chunk_size: int           # Chunk size for chunked prefill
    max_num_seqs: int                 # Max sequences in batch

    # === MEMORY PRESSURE ===
    kv_cache_utilization: float       # 0.0 - 1.0
    kv_free_blocks: int               # Free KV cache blocks
    kv_evictions_per_s: float         # Eviction rate (memory pressure indicator)
    # swap_bytes_per_s: float         # Optional: CPU offload activity

    # === METADATA ===
    timestamp: float
```

### 3.2 Request Features

```python
@dataclass
class RequestFeatures:
    """Features from the incoming request."""

    # Raw request attributes
    num_prompt_tokens: int            # L_prompt
    num_predicted_output_tokens: int  # L_out

    # Derived: approximate (q, c) evolution over request lifetime
    req_decode_ctx_avg: float         # L_prompt + (L_out-1)/2 (avg attention cost)
    req_decode_ctx_max: float         # L_prompt + L_out - 1 (final attention cost)
```

### 3.3 Derived Features (Computed at Prediction Time)

```python
def compute_derived_features(state: SimpleScheduleState,
                             request: RequestFeatures) -> dict:
    """Compute powerful derived features."""

    # Backlog in seconds (wait time estimate without simulation)
    backlog_seconds = (
        state.pending_prefill_tokens / max(state.ema_prefill_tok_per_s, 1) +
        state.pending_decode_tokens / max(state.ema_decode_tok_per_s, 1)
    )

    # Request's expected processing time (rough)
    request_prefill_time = request.num_prompt_tokens / max(state.ema_prefill_tok_per_s, 1)
    request_decode_time = request.num_predicted_output_tokens / max(state.ema_decode_tok_per_s, 1)

    # How much will this request's context affect batch attention?
    ctx_relative_to_p95 = request.req_decode_ctx_max / max(state.decode_ctx_p95, 1)

    return {
        'backlog_seconds': backlog_seconds,
        'request_prefill_time_est': request_prefill_time,
        'request_decode_time_est': request_decode_time,
        'ctx_relative_to_p95': ctx_relative_to_p95,
    }
```

### 3.4 Combined Feature Vector

```python
def build_feature_vector(state: SimpleScheduleState,
                         request: RequestFeatures) -> np.ndarray:
    """Build feature vector for XGBoost prediction."""

    derived = compute_derived_features(state, request)

    return np.array([
        # Service rate signals (3) - CRITICAL
        state.ema_decode_tok_per_s,
        state.ema_prefill_tok_per_s,
        state.ema_decode_iter_ms,

        # Context length distribution (3) - attention cost
        state.decode_ctx_p50,
        state.decode_ctx_p95,
        state.decode_ctx_max,

        # Batch state (3)
        state.num_running,
        state.num_active_decode_seqs,
        state.num_waiting,

        # Backlog in tokens (2)
        state.pending_prefill_tokens,
        state.pending_decode_tokens,

        # Scheduler knobs (3)
        state.token_budget_per_iter,
        state.prefill_chunk_size,
        state.max_num_seqs,

        # Memory pressure (3)
        state.kv_cache_utilization,
        state.kv_free_blocks,
        state.kv_evictions_per_s,

        # Request features (4)
        request.num_prompt_tokens,
        request.num_predicted_output_tokens,
        request.req_decode_ctx_avg,
        request.req_decode_ctx_max,

        # Derived features (4)
        derived['backlog_seconds'],
        derived['request_prefill_time_est'],
        derived['request_decode_time_est'],
        derived['ctx_relative_to_p95'],
    ])
```

**Total features**: 25

### 3.5 Feature Importance (Expected)

Based on DuetServe-like analysis, expected high-importance features:

| Priority | Feature | Rationale |
|----------|---------|-----------|
| 1 | `ema_decode_tok_per_s` | Service rate = capacity |
| 2 | `backlog_seconds` | Derived wait time |
| 3 | `decode_ctx_p95` | Attention cost tail |
| 4 | `num_prompt_tokens` | Request prefill cost |
| 5 | `pending_prefill_tokens` | Queue depth in work |
| 6 | `kv_cache_utilization` | Memory pressure |

### 3.6 Minimal Feature Set (If Constrained)

If vLLM changes are limited, use this minimal set:

```python
minimal_features = [
    # Instance (must have)
    'ema_decode_tok_per_s',      # Service rate
    'ema_prefill_tok_per_s',
    'decode_ctx_p95',            # Context distribution
    'num_active_decode_seqs',
    'pending_prefill_tokens',    # Backlog
    'pending_decode_tokens',
    'kv_cache_utilization',

    # Request (always available)
    'num_prompt_tokens',
    'num_predicted_output_tokens',
    'req_decode_ctx_avg',
    'req_decode_ctx_max',
]
```

**Minimal features**: 11

---

## 4. Model Architecture

### 4.1 Per-Instance-Type Models

Each instance type (model + GPU combination) has its own XGBoost model:

```
models/
├── qwen2.5-3b_p100.xgb      # 3B on P100
├── qwen2.5-7b_a30.xgb       # 7B on A30
├── qwen2.5-14b_v100.xgb     # 14B on V100x4
└── qwen2.5-72b_a100.xgb     # 72B on A100x4
```

**Rationale**: Different model sizes and GPU types have fundamentally different performance characteristics. Separate models avoid conflating these differences.

### 4.2 XGBoost Configuration

```python
xgb_params = {
    'objective': 'reg:squarederror',  # Regression for latency
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 1,
    'reg_alpha': 0.1,  # L1 regularization
    'reg_lambda': 1.0,  # L2 regularization
}
```

### 4.3 Multi-Output Prediction

Train separate models or multi-output for:
- **E2E latency** (primary target)
- **TTFT** (time-to-first-token)
- **TPOT** (time-per-output-token)

```python
class XGBoostLatencyPredictor:
    def __init__(self, model_dir: str, instance_type: str):
        self.e2e_model = xgb.Booster()
        self.e2e_model.load_model(f"{model_dir}/{instance_type}_e2e.xgb")

        self.ttft_model = xgb.Booster()
        self.ttft_model.load_model(f"{model_dir}/{instance_type}_ttft.xgb")

    def predict(self, features: np.ndarray) -> dict:
        dmatrix = xgb.DMatrix(features.reshape(1, -1))
        return {
            'e2e': self.e2e_model.predict(dmatrix)[0],
            'ttft': self.ttft_model.predict(dmatrix)[0],
        }
```

---

## 5. Implementation Plan

### 5.1 Phase 1: vLLM Changes (Separate Repo) ✅ COMPLETED

**Goal**: Add `/instance_stats` endpoint to vLLM

**Files modified**:
- `vllm/entrypoints/openai/api_server.py` - Added endpoint
- `vllm/v1/core/sched/scheduler.py` - Added `get_aggregated_stats()`
- `vllm/v1/request.py` - Added `predicted_decode_tokens` field with auto-extension
- `vllm/entrypoints/openai/protocol.py` - Added `predicted_decode_tokens` to request protocol

**See**: [INSTANCE_STATS_API.md](INSTANCE_STATS_API.md) for the finalized API specification

### 5.2 Phase 2: Block Changes

#### 5.2.1 New Files to Create

| File | Purpose |
|------|---------|
| `block/predictor/cara/simple_schedule_state.py` | Lightweight aggregated state dataclass |
| `block/predictor/cara/xgboost_predictor.py` | XGBoost model wrapper (implements CARABasePredictor) |
| `block/predictor/cara/roofline_predictor.py` | Analytical roofline model (ablation baseline) |
| `block/predictor/cara/linear_predictor.py` | Simple linear regression (ablation baseline) |
| `block/predictor/cara/lstm_predictor.py` | LSTM sequence model (research baseline) |
| `block/predictor/cara/state_manager.py` | Centralized state cache with refresh_all() |
| `block/predictor/cara/predictor_registry.py` | Maps instance_type to predictor |
| `block/predictor/cara/scheduling_policy.py` | BOOTSTRAP vs CARA mode selection |
| `block/predictor/cara/offline_training/train_xgboost.py` | XGBoost training script |
| `block/predictor/cara/offline_training/evaluate_predictor.py` | Ablation evaluation framework |

#### 5.2.2 Files to Modify

| File | Changes |
|------|---------|
| `block/predictor/cara/data_structures.py` | Add `InstanceStats`, `instance_type`, `actual_output_length` |
| `block/predictor/cara/base_predictor.py` | Add `STATE_API_TYPE` class attribute |
| `block/predictor/cara/cara_predictor_config.py` | Add factory pattern with registry |
| `block/predictor/cara/schedule_trace_client.py` | Add `fetch_instance_stats()` method |
| `block/predictor/cara/training_data_collector.py` | Add `state_api` field for format tracking |
| `block/global_scheduler/cara/cara_serve.py` | Add `/v1/batch_completions`, integrate StateManager |
| `block/global_scheduler/cara/cara_instance/Instance.py` | Single `data_collection_port`, `on_completion_callback` |
| `block/exp/cara/deploy_cara.py` | Derive `instance_type` at deployment time |
| `block/exp/generate_config.py` | Remove `predictor_ports` generation |

#### 5.2.3 Config Files to Update

| File | Changes |
|------|---------|
| `block/config/cara/predictor_config_xgboost.json` | New XGBoost config (see Section 0.5) |
| `block/config/cara/predictor_config_roofline.json` | New Roofline config (see Section 0.5) |
| `block/config/cara/predictor_config_lstm.json` | New LSTM config (see Section 0.5) |
| `block/config/cara/model_config_template.json` | Add `data_collection_port` per node_type |

#### 5.2.4 Implementation Task List

| # | Task | Description | Dependencies |
|---|------|-------------|--------------|
| 1 | Port cleanup | Rename `predictor_ports` → `data_collection_port` | None |
| 2 | Extend predictor interface | Add `STATE_API_TYPE`, config factory pattern | None |
| 3 | StateManager | Implement `refresh_all()`, `apply_assignment()` | #2 |
| 4 | PredictorRegistry | Map instance_type to predictor | #2 |
| 5 | SchedulingPolicy | BOOTSTRAP vs CARA mode | #3, #4 |
| 6 | XGBoostPredictor | Implement predictor class | #2 |
| 7 | XGBoost training | Training pipeline script | #6 |
| 8 | Integration | `/v1/batch_completions`, completion callback | #3, #4, #5 |
| 9 | EMA tracking | Service rate tracking at scheduler | #3 |
| 10 | Unit tests | Test all new components | #1-#9 |
| 11 | RooflinePredictor | Ablation baseline implementation | #2 |
| 12 | Evaluation framework | Compare predictor types | #6, #11 |
| 13 | Documentation | Update all docs | #1-#12 |

### 5.3 Phase 3: Training Pipeline

1. **Collect training data** using existing infrastructure (dummy predictor at instance side)
2. **Extract features** from collected JSONL files
3. **Train XGBoost models** per instance type
4. **Evaluate** on held-out test set
5. **Deploy** models to scheduler

---

## 5.4 New Endpoint Design: `/v1/batch_completions`

**Key Decision**: Keep `/v1/completions` unchanged, add new endpoint for CARA batch scheduling.

### Why a New Endpoint?

1. **Backward compatibility**: Existing clients using `/v1/completions` continue working
2. **Clear semantics**: Batch endpoint explicitly accepts array of requests
3. **Scheduling mode**: New endpoint enables CARA scheduling vs random/round-robin

### Endpoint Specification

```python
@app.post("/v1/batch_completions")
async def batch_completion(request: Request) -> Response:
    """CARA batch completion endpoint.

    Request body:
    {
        "requests": [
            {"prompt": "...", "max_tokens": 256, "prompt_len": 128, ...},
            {"prompt": "...", "max_tokens": 256, "prompt_len": 96, ...},
            ...
        ],
        "scheduling_mode": "cara"  # or "random", "round_robin"
    }

    Response:
    {
        "responses": [
            {"request_id": "...", "text": "...", "model": "...", ...},
            ...
        ],
        "scheduling_stats": {
            "mode": "cara",
            "assignments": {"instance_0": 3, "instance_1": 5, ...}
        }
    }
    """
```

### Scheduling Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `BOOTSTRAP` | Random/round-robin | Data collection (cold start) |
| `CARA` | XGBoost-based prediction | Production routing |

### Mode Selection Logic

```python
class SchedulingMode(Enum):
    BOOTSTRAP = "bootstrap"
    CARA = "cara"

def select_scheduling_mode(predictor_available: bool, min_training_samples: int = 1000) -> SchedulingMode:
    """Select scheduling mode based on predictor availability."""
    if not predictor_available:
        return SchedulingMode.BOOTSTRAP

    # Check if we have enough training data
    training_samples = count_training_samples()
    if training_samples < min_training_samples:
        return SchedulingMode.BOOTSTRAP

    return SchedulingMode.CARA
```

### 5.5 Completion Callback Mechanism

Instances notify the scheduler when requests complete via a **constructor-injected callback**.

```python
# block/global_scheduler/cara/cara_instance/Instance.py

class Instance(ABC):
    def __init__(self,
                 instance_id: str,
                 hostname: str,
                 ip_address: str,
                 model_name: str,
                 backend_port: int,
                 data_collection_port: int,  # Single port (renamed from predictor_ports)
                 on_completion_callback: Optional[Callable] = None,  # NEW
                 **kwargs):
        self._instance_id = instance_id
        self._hostname = hostname
        self._ip_address = ip_address
        self._model_name = model_name
        self._backend_port = backend_port
        self._data_collection_port = data_collection_port
        self._on_completion_callback = on_completion_callback  # Store callback

    async def query_instance(self, payload: dict, predicted_num_decode_tokens: int):
        # ... execute request ...
        response_dict = await self.query_backend(payload)

        # Notify scheduler of completion
        if self._on_completion_callback:
            await self._on_completion_callback(
                request_id=payload.get("request_id"),
                instance_id=self._instance_id,
                response=response_dict
            )

        return response_dict
```

### 5.6 Scheduler-Side Components

```
┌─────────────────────────────────────────────────────────────────┐
│                    CARA Global Scheduler                         │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                   StateManager                            │   │
│  │  - refresh_all(): Query /instance_stats from all backends │   │
│  │  - apply_assignment(): Update local state after assignment│   │
│  │  - get_state(instance_id): Get cached state               │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                 PredictorRegistry                         │   │
│  │  - register(instance_type, predictor): Add predictor      │   │
│  │  - predict(instance_type, state, request): Get prediction │   │
│  │  - get_predictor(instance_type): Get predictor instance   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                 SchedulingPolicy                          │   │
│  │  - schedule_batch(requests): Assign requests to instances │   │
│  │  - select_mode(): BOOTSTRAP or CARA                       │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 5.7 StateManager Design

```python
class StateManager:
    """Centralized state management for scheduler-side predictor."""

    def __init__(self, instances: List[Instance], timeout_ms: float = 10.0):
        self.instances = {inst._instance_id: inst for inst in instances}
        self.states: Dict[str, SimpleScheduleState] = {}
        self.timeout = aiohttp.ClientTimeout(total=timeout_ms / 1000)
        self._consecutive_failures: Dict[str, int] = defaultdict(int)
        self._max_failures = 3

    async def refresh_all(self) -> Dict[str, SimpleScheduleState]:
        """Query all instances in parallel at batch start."""
        tasks = {
            inst_id: self._fetch_state(inst_id)
            for inst_id in self.instances
        }
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)

        for inst_id, result in zip(tasks.keys(), results):
            if isinstance(result, Exception):
                self._consecutive_failures[inst_id] += 1
                if self._consecutive_failures[inst_id] >= self._max_failures:
                    logger.warning(f"Instance {inst_id} marked down after {self._max_failures} failures")
                # Use stale state on timeout
                continue
            else:
                self._consecutive_failures[inst_id] = 0
                self.states[inst_id] = result

        return self.states

    def apply_assignment(self, instance_id: str, request: PredictRequest):
        """Update local state after assigning a request (within-batch only)."""
        state = self.states.get(instance_id)
        if state is None:
            return

        # Update pending tokens
        state.pending_prefill_tokens += request.num_prompt_tokens
        state.pending_decode_tokens += request.num_predicted_output_tokens

        # Update queue counts
        if state.num_running < state.max_num_seqs:
            state.num_running += 1
        else:
            state.num_waiting += 1
```

---

## 6. Detailed Component Specifications

### 6.1 SimpleScheduleState

```python
# block/predictor/cara/simple_schedule_state.py

from dataclasses import dataclass, field
from typing import Optional
import time

@dataclass
class SimpleScheduleState:
    """Aggregated state for latency prediction.

    Captures service rate, context distribution, backlog, and memory pressure.
    Parsed from /instance_stats endpoint response.
    """
    instance_id: str

    # === SERVICE RATE SIGNALS ===
    ema_decode_tok_per_s: float = 0.0
    ema_prefill_tok_per_s: float = 0.0
    ema_decode_iter_ms: float = 0.0

    # === BATCH STATE ===
    num_running: int = 0
    num_waiting: int = 0
    num_active_decode_seqs: int = 0

    # === CONTEXT LENGTH DISTRIBUTION ===
    decode_ctx_p50: int = 0
    decode_ctx_p95: int = 0
    decode_ctx_max: int = 0

    # === BACKLOG IN TOKENS ===
    pending_prefill_tokens: int = 0
    pending_decode_tokens: int = 0

    # === SCHEDULER KNOBS ===
    token_budget_per_iter: int = 2048
    prefill_chunk_size: int = 512
    max_num_seqs: int = 256

    # === MEMORY PRESSURE ===
    kv_cache_utilization: float = 0.0
    kv_free_blocks: int = 0
    kv_evictions_per_s: float = 0.0

    # === METADATA ===
    num_preempted: int = 0
    timestamp: float = field(default_factory=time.time)

    def is_stale(self, max_age_ms: float = 200) -> bool:
        """Check if state is too old."""
        age_ms = (time.time() - self.timestamp) * 1000
        return age_ms > max_age_ms

    def compute_backlog_seconds(self) -> float:
        """Estimate queue wait time in seconds."""
        prefill_time = (self.pending_prefill_tokens /
                        max(self.ema_prefill_tok_per_s, 1.0))
        decode_time = (self.pending_decode_tokens /
                       max(self.ema_decode_tok_per_s, 1.0))
        return prefill_time + decode_time

    @classmethod
    def from_response(cls, instance_id: str, response: dict) -> 'SimpleScheduleState':
        """Parse from /instance_stats response."""
        return cls(
            instance_id=instance_id,
            # Service rates
            ema_decode_tok_per_s=response.get('ema_decode_tok_per_s', 0.0),
            ema_prefill_tok_per_s=response.get('ema_prefill_tok_per_s', 0.0),
            ema_decode_iter_ms=response.get('ema_decode_iter_ms', 0.0),
            # Batch state
            num_running=response.get('num_running', 0),
            num_waiting=response.get('num_waiting', 0),
            num_active_decode_seqs=response.get('num_active_decode_seqs', 0),
            # Context distribution
            decode_ctx_p50=response.get('decode_ctx_p50', 0),
            decode_ctx_p95=response.get('decode_ctx_p95', 0),
            decode_ctx_max=response.get('decode_ctx_max', 0),
            # Backlog
            pending_prefill_tokens=response.get('pending_prefill_tokens', 0),
            pending_decode_tokens=response.get('pending_decode_tokens', 0),
            # Scheduler knobs
            token_budget_per_iter=response.get('token_budget_per_iter', 2048),
            prefill_chunk_size=response.get('prefill_chunk_size', 512),
            max_num_seqs=response.get('max_num_seqs', 256),
            # Memory
            kv_cache_utilization=response.get('kv_cache_utilization', 0.0),
            kv_free_blocks=response.get('kv_free_blocks', 0),
            kv_evictions_per_s=response.get('kv_evictions_per_s', 0.0),
            # Metadata
            num_preempted=response.get('num_preempted', 0),
            timestamp=response.get('timestamp', time.time()),
        )
```

### 6.2 Base Predictor Interface (Updated)

```python
# block/predictor/cara/base_predictor.py

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Optional
from dataclasses import dataclass

class StateApiType(Enum):
    """State API types supported by predictors."""
    SCHEDULE_TRACE = "schedule_trace"   # Detailed per-request data
    INSTANCE_STATS = "instance_stats"   # Aggregated stats

@dataclass
class CARABasePredictorConfig:
    """Base config for all predictor types."""
    predictor_type: str
    enable_data_collection: bool = True
    data_collection_sample_rate: float = 1.0
    data_output_dir: str = "./training_data/cara"

class CARABasePredictor(ABC):
    """Base class for CARA latency predictors.

    Subclasses must declare:
    - STATE_API_TYPE: Which state API they require
    - REQUIRES_MODEL: Whether they need a trained model file
    """

    # Subclasses MUST override this
    STATE_API_TYPE: StateApiType = None
    REQUIRES_MODEL: bool = False

    @abstractmethod
    async def predict(self, state, request) -> Dict:
        """Predict latency for a request given instance state.

        Args:
            state: Either ScheduleState or SimpleScheduleState depending on STATE_API_TYPE
            request: PredictRequest with request attributes

        Returns:
            Dict with 'latency_prediction' and optional 'confidence'
        """
        pass

    @abstractmethod
    async def predict_batch(self, state, requests) -> list:
        """Predict latencies for multiple requests (same state)."""
        pass

    def get_state_api_type(self) -> StateApiType:
        """Return required state API type."""
        if self.STATE_API_TYPE is None:
            raise NotImplementedError("Subclass must set STATE_API_TYPE")
        return self.STATE_API_TYPE
```

### 6.3 XGBoost Predictor

```python
# block/predictor/cara/xgboost_predictor.py

import numpy as np
import xgboost as xgb
from pathlib import Path
from typing import Dict, List, Optional
import logging

from block.predictor.cara.base_predictor import CARABasePredictor, StateApiType
from block.predictor.cara.simple_schedule_state import SimpleScheduleState
from block.predictor.cara.data_structures import PredictRequest

logger = logging.getLogger(__name__)


class XGBoostLatencyPredictor(CARABasePredictor):
    """XGBoost-based latency predictor for a single instance type.

    Each instance type (model + GPU combination) has its own predictor.
    E.g., qwen2.5-3b_vllm_p100, qwen2.5-7b_vllm_a30, qwen2.5-72b_vllm_a100
    """

    # Declare required state API and model dependency
    STATE_API_TYPE = StateApiType.INSTANCE_STATS
    REQUIRES_MODEL = True

    # Feature names for interpretability and validation
    FEATURE_NAMES = [
        # Service rate signals (3)
        'ema_decode_tok_per_s', 'ema_prefill_tok_per_s', 'ema_decode_iter_ms',
        # Context distribution (3)
        'decode_ctx_p50', 'decode_ctx_p95', 'decode_ctx_max',
        # Batch state (3)
        'num_running', 'num_active_decode_seqs', 'num_waiting',
        # Backlog (2)
        'pending_prefill_tokens', 'pending_decode_tokens',
        # Scheduler knobs (3)
        'token_budget_per_iter', 'prefill_chunk_size', 'max_num_seqs',
        # Memory pressure (3)
        'kv_cache_utilization', 'kv_free_blocks', 'kv_evictions_per_s',
        # Request features (4)
        'num_prompt_tokens', 'num_predicted_output_tokens',
        'req_decode_ctx_avg', 'req_decode_ctx_max',
        # Derived features (4)
        'backlog_seconds', 'request_prefill_time_est',
        'request_decode_time_est', 'ctx_relative_to_p95',
    ]

    def __init__(self, model_path: str, instance_type: str):
        """
        Args:
            model_path: Path to .xgb model file
            instance_type: Identifier (e.g., "qwen2.5-3b_p100")
        """
        self.instance_type = instance_type
        self.model = xgb.Booster()
        self.model.load_model(model_path)
        logger.info(f"Loaded XGBoost model for {instance_type} from {model_path}")

    def predict(self, state: SimpleScheduleState,
                request: PredictRequest) -> float:
        """Predict E2E latency for a request given instance state.

        Args:
            state: Current instance state
            request: Request to predict latency for

        Returns:
            Predicted E2E latency in seconds
        """
        features = self._build_features(state, request)
        dmatrix = xgb.DMatrix(features.reshape(1, -1),
                              feature_names=self.FEATURE_NAMES)
        prediction = self.model.predict(dmatrix)[0]
        return float(prediction)

    def _build_features(self, state: SimpleScheduleState,
                        request: PredictRequest) -> np.ndarray:
        """Build feature vector from state and request."""

        # Request-derived features
        L_prompt = request.num_prompt_tokens
        L_out = request.num_predicted_output_tokens
        req_decode_ctx_avg = L_prompt + (L_out - 1) / 2
        req_decode_ctx_max = L_prompt + L_out - 1

        # Derived features
        backlog_seconds = state.compute_backlog_seconds()
        request_prefill_time_est = L_prompt / max(state.ema_prefill_tok_per_s, 1.0)
        request_decode_time_est = L_out / max(state.ema_decode_tok_per_s, 1.0)
        ctx_relative_to_p95 = req_decode_ctx_max / max(state.decode_ctx_p95, 1)

        return np.array([
            # Service rate signals (3)
            state.ema_decode_tok_per_s,
            state.ema_prefill_tok_per_s,
            state.ema_decode_iter_ms,
            # Context distribution (3)
            state.decode_ctx_p50,
            state.decode_ctx_p95,
            state.decode_ctx_max,
            # Batch state (3)
            state.num_running,
            state.num_active_decode_seqs,
            state.num_waiting,
            # Backlog (2)
            state.pending_prefill_tokens,
            state.pending_decode_tokens,
            # Scheduler knobs (3)
            state.token_budget_per_iter,
            state.prefill_chunk_size,
            state.max_num_seqs,
            # Memory pressure (3)
            state.kv_cache_utilization,
            state.kv_free_blocks,
            state.kv_evictions_per_s,
            # Request features (4)
            L_prompt,
            L_out,
            req_decode_ctx_avg,
            req_decode_ctx_max,
            # Derived features (4)
            backlog_seconds,
            request_prefill_time_est,
            request_decode_time_est,
            ctx_relative_to_p95,
        ], dtype=np.float32)

    def predict_batch(self, state: SimpleScheduleState,
                      requests: List[PredictRequest]) -> List[float]:
        """Predict latencies for multiple requests (same state).

        Efficient batch prediction for routing decisions.

        Args:
            state: Current instance state (shared for all requests)
            requests: List of PredictRequest objects

        Returns:
            List of predicted latencies
        """
        if not requests:
            return []

        # Build feature matrix
        features = np.vstack([
            self._build_features(state, req) for req in requests
        ])

        # Batch prediction
        dmatrix = xgb.DMatrix(features, feature_names=self.FEATURE_NAMES)
        predictions = self.model.predict(dmatrix)

        return predictions.tolist()

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance for interpretability."""
        importance = self.model.get_score(importance_type='weight')
        return {
            self.FEATURE_NAMES[int(k[1:])]: v
            for k, v in importance.items()
        }
```

### 6.4 Roofline Predictor (Ablation Baseline)

The roofline predictor uses an analytical model based on peak throughput rates.
Peak rates can be:
1. **Hardcoded** (default values - simple but inflexible)
2. **Loaded from calibration file** (per-instance-type rates)
3. **Auto-calibrated** from training data (infer as 99th percentile of observed rates)

```python
# block/predictor/cara/roofline_predictor.py

from dataclasses import dataclass
from typing import Dict, List, Optional
import json
import logging

from block.predictor.cara.base_predictor import CARABasePredictor, StateApiType
from block.predictor.cara.data_structures import SimpleScheduleState, PredictRequest

logger = logging.getLogger(__name__)


class RooflinePredictor(CARABasePredictor):
    """Analytical roofline model for ablation baseline.

    Uses simple throughput-based estimation:
    latency = prefill_tokens / prefill_rate + decode_tokens / decode_rate + queue_delay

    Peak rates can be calibrated from training data for better accuracy.
    """

    STATE_API_TYPE = StateApiType.INSTANCE_STATS
    REQUIRES_MODEL = False  # No trained model needed, but can use calibration

    def __init__(self, config, instance_type: str = None):
        super().__init__(config, config.backend_port)
        self.instance_type = instance_type

        # Load peak rates (priority: calibration_file > auto_calibrate > defaults)
        self.prefill_rate, self.decode_rate = self._load_peak_rates(config)
        logger.info(f"Roofline predictor: prefill={self.prefill_rate} tok/s, decode={self.decode_rate} tok/s")

    def _load_peak_rates(self, config) -> tuple:
        """Load peak rates from calibration or use defaults."""
        # Option 1: Load from calibration file
        if config.calibration_file:
            with open(config.calibration_file, 'r') as f:
                calibration = json.load(f)
            if self.instance_type and self.instance_type in calibration:
                rates = calibration[self.instance_type]
                return rates['prefill_toks_per_sec'], rates['decode_toks_per_sec']

        # Option 2: Auto-calibrate from training data
        if config.auto_calibrate and config.calibration_data_path:
            return self._calibrate_from_data(config.calibration_data_path)

        # Option 3: Use hardcoded defaults
        return config.prefill_toks_per_sec, config.decode_toks_per_sec

    def _calibrate_from_data(self, data_path: str) -> tuple:
        """Infer peak rates from training data (99th percentile of observed rates)."""
        import numpy as np
        prefill_rates, decode_rates = [], []

        with open(data_path, 'r') as f:
            for line in f:
                example = json.loads(line)
                ttft = example.get('actual_ttft', 0)
                e2e = example.get('actual_e2e_latency', 0)
                prompt_tokens = example.get('num_prompt_tokens', 0)
                output_tokens = example.get('actual_output_length', 0)

                if ttft > 0 and prompt_tokens > 0:
                    prefill_rates.append(prompt_tokens / ttft)
                decode_time = e2e - ttft
                if decode_time > 0 and output_tokens > 0:
                    decode_rates.append(output_tokens / decode_time)

        # Use 99th percentile as "peak" (robust to outliers)
        prefill_peak = np.percentile(prefill_rates, 99) if prefill_rates else 10000.0
        decode_peak = np.percentile(decode_rates, 99) if decode_rates else 500.0
        logger.info(f"Calibrated peak rates: prefill={prefill_peak:.0f}, decode={decode_peak:.0f}")
        return prefill_peak, decode_peak

    async def predict(self, target_request: PredictRequest, state=None) -> Dict:
        """Predict latency using roofline model."""
        if state is None:
            # Fetch state if not provided
            state = SimpleScheduleState()  # Empty state as fallback

        # Prefill time
        prefill_time = target_request.num_prompt_tokens / self.prefill_rate

        # Decode time
        decode_time = target_request.num_predicted_output_tokens / self.decode_rate

        # Queue delay estimate (linear in pending tokens)
        queue_delay = (
            state.pending_prefill_tokens / self.prefill_rate +
            state.pending_decode_tokens / self.decode_rate
        )

        latency = prefill_time + decode_time + queue_delay
        return {
            "target_metric": latency,
            "latency_prediction": latency,
            "predictor_type": "roofline"
        }
```

**Calibration File Format** (`roofline_calibration.json`):
```json
{
  "qwen-2-5-3b_vllm_c240g5": {"prefill_toks_per_sec": 12000, "decode_toks_per_sec": 600},
  "qwen-2-5-7b_vllm_c240g5": {"prefill_toks_per_sec": 8000, "decode_toks_per_sec": 400},
  "qwen-2-5-72b_vllm_a100": {"prefill_toks_per_sec": 15000, "decode_toks_per_sec": 200}
}
```

### 6.5 Linear Predictor (Ablation Baseline)

The linear predictor uses a simple affine model with **trainable coefficients**.
All parameters (alpha_*, bias) can be learned via ordinary least squares (OLS).

**Training**: `python -m block.predictor.cara.offline_training.train_linear --input data.jsonl --output weights.json`

```python
# block/predictor/cara/linear_predictor.py

import json
from typing import Dict, List
import logging

from block.predictor.cara.base_predictor import CARABasePredictor, StateApiType
from block.predictor.cara.data_structures import SimpleScheduleState, PredictRequest

logger = logging.getLogger(__name__)


class LinearPredictor(CARABasePredictor):
    """Simple linear model for ablation baseline.

    Model: P_lat = α_prefill·N_prefill + α_decode·N_decode + α_waiting·N_waiting + bias

    All coefficients are TRAINABLE via linear regression:
    - Train: sklearn.linear_model.LinearRegression on training data
    - Deploy: Load weights from JSON file

    Based on paper's "gray box" affine model, simplified for clarity.
    """

    STATE_API_TYPE = StateApiType.INSTANCE_STATS
    REQUIRES_MODEL = False  # Coefficients can be configured or trained

    def __init__(self, config):
        super().__init__(config, config.backend_port)

        # Load weights from file if provided, otherwise use config defaults
        if config.weights_path:
            self._load_weights(config.weights_path)
        else:
            self.alpha_prefill = config.alpha_prefill
            self.alpha_decode = config.alpha_decode
            self.alpha_waiting = config.alpha_waiting
            self.bias = config.bias

        logger.info(f"Linear predictor: α_p={self.alpha_prefill}, α_d={self.alpha_decode}, "
                    f"α_w={self.alpha_waiting}, bias={self.bias}")

    def _load_weights(self, weights_path: str):
        """Load trained weights from JSON file."""
        with open(weights_path, 'r') as f:
            weights = json.load(f)
        self.alpha_prefill = weights['alpha_prefill']
        self.alpha_decode = weights['alpha_decode']
        self.alpha_waiting = weights['alpha_waiting']
        self.bias = weights['bias']
        logger.info(f"Loaded trained weights from {weights_path}")

    async def predict(self, target_request: PredictRequest, state=None) -> Dict:
        """Predict latency using linear model."""
        if state is None:
            state = SimpleScheduleState()  # Empty state as fallback

        latency = (
            self.alpha_prefill * target_request.num_prompt_tokens +
            self.alpha_decode * target_request.num_predicted_output_tokens +
            self.alpha_waiting * state.num_waiting +
            self.bias
        )
        return {
            "target_metric": latency,
            "latency_prediction": latency,
            "predictor_type": "linear"
        }
```

**Training Script** (`train_linear.py`):
```python
from sklearn.linear_model import LinearRegression
import json
import numpy as np

def train_linear_predictor(training_data_path: str, output_path: str):
    """Train linear predictor coefficients using OLS."""
    X, y = [], []

    with open(training_data_path, 'r') as f:
        for line in f:
            ex = json.loads(line)
            if ex.get('actual_e2e_latency') is None:
                continue
            X.append([
                ex['num_prompt_tokens'],
                ex['num_predicted_output_tokens'],
                ex['schedule_state'].get('num_waiting', 0)
            ])
            y.append(ex['actual_e2e_latency'])

    model = LinearRegression()
    model.fit(X, y)

    weights = {
        "alpha_prefill": float(model.coef_[0]),
        "alpha_decode": float(model.coef_[1]),
        "alpha_waiting": float(model.coef_[2]),
        "bias": float(model.intercept_)
    }

    with open(output_path, 'w') as f:
        json.dump(weights, f, indent=2)

    print(f"Trained weights: {weights}")
    print(f"R² score: {model.score(X, y):.4f}")
```

**Trained Weights File Format** (`linear_weights.json`):
```json
{
  "alpha_prefill": 0.00012,
  "alpha_decode": 0.0018,
  "alpha_waiting": 0.15,
  "bias": 0.08
}
```

### 6.6 Scheduler Predictor Service

```python
# block/predictor/cara/scheduler_predictor_service.py

import asyncio
import aiohttp
import logging
from typing import Dict, List, Optional
from pathlib import Path

from block.predictor.cara.simple_schedule_state import SimpleScheduleState
from block.predictor.cara.xgboost_predictor import XGBoostLatencyPredictor
from block.predictor.cara.data_structures import PredictRequest

logger = logging.getLogger(__name__)


class SchedulerPredictorService:
    """Manages scheduler-side predictors for all instances.

    Maintains local state cache and provides fast local predictions.
    """

    def __init__(self, instances: List, model_dir: str):
        """
        Args:
            instances: List of Instance objects from scheduler
            model_dir: Directory containing XGBoost model files
        """
        self.instances = {inst._instance_id: inst for inst in instances}
        self.model_dir = Path(model_dir)

        # State cache: instance_id -> SimpleScheduleState
        self.states: Dict[str, SimpleScheduleState] = {}

        # Predictors: instance_type -> XGBoostLatencyPredictor
        self.predictors: Dict[str, XGBoostLatencyPredictor] = {}

        # Instance to instance_type mapping
        self.instance_types: Dict[str, str] = {}

        self._init_predictors(instances)

    def _init_predictors(self, instances: List):
        """Initialize predictors for each instance type."""
        for inst in instances:
            instance_id = inst._instance_id
            # Derive instance type from model name and GPU (customize as needed)
            instance_type = self._get_instance_type(inst)
            self.instance_types[instance_id] = instance_type

            # Initialize empty state
            self.states[instance_id] = SimpleScheduleState(instance_id=instance_id)

            # Load predictor if not already loaded for this type
            if instance_type not in self.predictors:
                model_path = self.model_dir / f"{instance_type}.xgb"
                if model_path.exists():
                    self.predictors[instance_type] = XGBoostLatencyPredictor(
                        str(model_path), instance_type
                    )
                else:
                    logger.warning(f"No model found for {instance_type} at {model_path}")

    def _get_instance_type(self, instance) -> str:
        """Derive instance type identifier from instance."""
        # Example: "qwen2.5-3b_p100" from model name and GPU type
        model_name = instance._model_name.lower().replace("/", "-").replace(".", "")
        # GPU type could be stored in instance or derived from hostname
        gpu_type = getattr(instance, '_gpu_type', 'unknown')
        return f"{model_name}_{gpu_type}"

    async def refresh_all_states(self) -> Dict[str, SimpleScheduleState]:
        """Query all instances in parallel and update state cache.

        Called once per batch before making predictions.

        Returns:
            Updated state cache
        """
        async with aiohttp.ClientSession() as session:
            tasks = {
                inst_id: self._fetch_state(session, inst_id)
                for inst_id in self.instances
            }
            results = await asyncio.gather(*tasks.values(), return_exceptions=True)

            for inst_id, result in zip(tasks.keys(), results):
                if isinstance(result, Exception):
                    logger.warning(f"Failed to fetch state from {inst_id}: {result}")
                else:
                    self.states[inst_id] = result

            return self.states

    async def _fetch_state(self, session: aiohttp.ClientSession,
                           instance_id: str) -> SimpleScheduleState:
        """Fetch state from a single instance."""
        inst = self.instances[instance_id]
        url = f"http://{inst._ip_address}:{inst._backend_port}/instance_stats"

        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=0.1)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return SimpleScheduleState.from_response(instance_id, data)
                else:
                    logger.warning(f"Bad response from {instance_id}: {resp.status}")
                    return self.states.get(instance_id,
                                           SimpleScheduleState(instance_id=instance_id))
        except Exception as e:
            logger.warning(f"Error fetching state from {instance_id}: {e}")
            return self.states.get(instance_id,
                                   SimpleScheduleState(instance_id=instance_id))

    def predict_latency(self, instance_id: str, request: PredictRequest) -> float:
        """Predict latency for a request on a specific instance.

        Uses cached state - call refresh_all_states() first!

        Args:
            instance_id: Target instance
            request: Request to predict

        Returns:
            Predicted E2E latency in seconds
        """
        state = self.states.get(instance_id)
        if state is None:
            logger.warning(f"No state for {instance_id}, returning default")
            return float('inf')

        instance_type = self.instance_types.get(instance_id)
        predictor = self.predictors.get(instance_type)

        if predictor is None:
            logger.warning(f"No predictor for {instance_type}, using heuristic")
            # Fallback heuristic: prefer less loaded instances
            return float(state.num_running + state.num_waiting)

        return predictor.predict(state, request)

    def predict_all(self, request: PredictRequest) -> Dict[str, float]:
        """Predict latency for a request across all instances.

        Args:
            request: Request to predict

        Returns:
            Dict mapping instance_id -> predicted latency
        """
        return {
            inst_id: self.predict_latency(inst_id, request)
            for inst_id in self.instances
        }

    def predict_batch(self, requests: List[PredictRequest]) -> List[Dict[str, float]]:
        """Predict latencies for multiple requests across all instances.

        Args:
            requests: List of requests to predict

        Returns:
            List of dicts, each mapping instance_id -> predicted latency
        """
        return [self.predict_all(req) for req in requests]

    def get_best_instance(self, request: PredictRequest) -> str:
        """Get instance with lowest predicted latency for a request.

        Args:
            request: Request to route

        Returns:
            instance_id of best instance
        """
        predictions = self.predict_all(request)
        return min(predictions, key=predictions.get)
```

---

## 7. Training Pipeline

### 7.1 Training Data Collection

Training data is collected by the **instance-side dummy predictor** (unchanged from current design):

```
Request arrives at scheduler
    │
    ├─→ Scheduler queries instance /instance_stats (new)
    │
    ├─→ Instance predictor logs: (schedule_state, request, timestamp)
    │
    ├─→ Request executes
    │
    └─→ Instance sends feedback: (request_id, e2e_latency, ttft, tpot)
            │
            ▼
        Training data JSONL file (instance-side)
```

### 7.2 Training Data Format

```jsonl
{
  "request_id": "batch1-req-42",
  "num_prompt_tokens": 128,
  "num_predicted_output_tokens": 256,
  "schedule_state": {
    "num_running": 5,
    "num_waiting": 2,
    "total_prefill_tokens": 450,
    "total_decode_tokens": 1200,
    "total_unprocessed_tokens": 650,
    "kv_cache_utilization": 0.65,
    "free_gpu_blocks": 3500,
    "num_preempted": 0
  },
  "instance_id": "qwen-3b-0",
  "instance_type": "qwen2.5-3b_p100",
  "actual_e2e_latency": 1.234,
  "actual_ttft": 0.089,
  "actual_tpot": 0.012
}
```

### 7.3 Training Script

```python
# block/predictor/cara/offline_training/train_xgboost.py

"""
Train XGBoost latency predictor from collected training data.

Usage:
    python -m block.predictor.cara.offline_training.train_xgboost \
        --input-dir training_data/cara \
        --output-dir models/cara \
        --instance-type qwen2.5-3b_p100
"""

import argparse
import json
import numpy as np
import xgboost as xgb
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_training_data(input_dir: Path, instance_type: str):
    """Load and filter training data for a specific instance type."""
    features = []
    labels = []

    for jsonl_file in input_dir.glob("*.jsonl"):
        with open(jsonl_file) as f:
            for line in f:
                example = json.loads(line)

                # Filter by instance type
                if example.get('instance_type') != instance_type:
                    continue

                # Skip incomplete examples
                if example.get('actual_e2e_latency') is None:
                    continue

                # Extract state features
                state = example['schedule_state']

                # Request features
                L_prompt = example['num_prompt_tokens']
                L_out = example['num_predicted_output_tokens']
                req_decode_ctx_avg = L_prompt + (L_out - 1) / 2
                req_decode_ctx_max = L_prompt + L_out - 1

                # Derived features
                ema_decode = state.get('ema_decode_tok_per_s', 1000)  # default fallback
                ema_prefill = state.get('ema_prefill_tok_per_s', 2000)
                pending_prefill = state.get('pending_prefill_tokens', 0)
                pending_decode = state.get('pending_decode_tokens', 0)

                backlog_seconds = (pending_prefill / max(ema_prefill, 1) +
                                   pending_decode / max(ema_decode, 1))
                request_prefill_time_est = L_prompt / max(ema_prefill, 1)
                request_decode_time_est = L_out / max(ema_decode, 1)
                decode_ctx_p95 = state.get('decode_ctx_p95', 1)
                ctx_relative_to_p95 = req_decode_ctx_max / max(decode_ctx_p95, 1)

                feature_vec = [
                    # Service rate signals (3)
                    ema_decode,
                    ema_prefill,
                    state.get('ema_decode_iter_ms', 10),
                    # Context distribution (3)
                    state.get('decode_ctx_p50', 0),
                    decode_ctx_p95,
                    state.get('decode_ctx_max', 0),
                    # Batch state (3)
                    state.get('num_running', 0),
                    state.get('num_active_decode_seqs', 0),
                    state.get('num_waiting', 0),
                    # Backlog (2)
                    pending_prefill,
                    pending_decode,
                    # Scheduler knobs (3)
                    state.get('token_budget_per_iter', 2048),
                    state.get('prefill_chunk_size', 512),
                    state.get('max_num_seqs', 256),
                    # Memory pressure (3)
                    state.get('kv_cache_utilization', 0),
                    state.get('kv_free_blocks', 0),
                    state.get('kv_evictions_per_s', 0),
                    # Request features (4)
                    L_prompt,
                    L_out,
                    req_decode_ctx_avg,
                    req_decode_ctx_max,
                    # Derived features (4)
                    backlog_seconds,
                    request_prefill_time_est,
                    request_decode_time_est,
                    ctx_relative_to_p95,
                ]
                features.append(feature_vec)
                labels.append(example['actual_e2e_latency'])

    return np.array(features), np.array(labels)


def train_model(X_train, y_train, X_val, y_val, params=None):
    """Train XGBoost model with validation."""
    if params is None:
        params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
        }

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    evallist = [(dtrain, 'train'), (dval, 'val')]

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=200,
        evals=evallist,
        early_stopping_rounds=20,
        verbose_eval=10
    )

    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model on test set."""
    dtest = xgb.DMatrix(X_test)
    predictions = model.predict(dtest)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    # Percentile errors
    errors = np.abs(predictions - y_test)
    p50_error = np.percentile(errors, 50)
    p90_error = np.percentile(errors, 90)
    p99_error = np.percentile(errors, 99)

    return {
        'mae': mae,
        'rmse': rmse,
        'p50_error': p50_error,
        'p90_error': p90_error,
        'p99_error': p99_error,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--instance-type', type=str, required=True)
    parser.add_argument('--test-size', type=float, default=0.2)
    args = parser.parse_args()

    # Load data
    logger.info(f"Loading training data for {args.instance_type}...")
    X, y = load_training_data(args.input_dir, args.instance_type)
    logger.info(f"Loaded {len(y)} examples")

    if len(y) < 100:
        logger.error("Not enough training data (need at least 100 examples)")
        return

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42
    )

    logger.info(f"Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")

    # Train model
    logger.info("Training XGBoost model...")
    model = train_model(X_train, y_train, X_val, y_val)

    # Evaluate
    logger.info("Evaluating on test set...")
    metrics = evaluate_model(model, X_test, y_test)

    logger.info(f"Test MAE: {metrics['mae']:.3f}s")
    logger.info(f"Test RMSE: {metrics['rmse']:.3f}s")
    logger.info(f"P50 Error: {metrics['p50_error']:.3f}s")
    logger.info(f"P90 Error: {metrics['p90_error']:.3f}s")
    logger.info(f"P99 Error: {metrics['p99_error']:.3f}s")

    # Save model
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / f"{args.instance_type}.xgb"
    model.save_model(str(model_path))
    logger.info(f"Saved model to {model_path}")

    # Save metrics
    metrics_path = args.output_dir / f"{args.instance_type}_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")


if __name__ == '__main__':
    main()
```

---

## 8. vLLM Implementation Prompts

The following prompts can be used in a separate Claude Code session for the vLLM repository at `/auto/homes/wd312/Code/llm/llm-infer/vllm`:

### Prompt 1: Add `/instance_stats` Endpoint (Full Feature Set)

```
I need to add a new endpoint `/instance_stats` to vLLM that returns aggregated scheduler
state for latency prediction. This needs to capture service rate, context length distribution,
and backlog information - not just counts.

Current `/schedule_trace` location:
- API endpoint: vllm/entrypoints/openai/api_server.py (lines 379-407)
- Scheduler method: vllm/v1/core/sched/scheduler.py get_scheduler_trace() (lines 1272-1277)

Please implement `get_simple_scheduler_trace()` in `vllm/v1/core/sched/scheduler.py` returning:

```python
{
    # === SERVICE RATE SIGNALS (Critical for prediction!) ===
    # These need EMA tracking - add state variables to scheduler
    "ema_decode_tok_per_s": float,      # EMA of decode throughput
    "ema_prefill_tok_per_s": float,     # EMA of prefill throughput
    "ema_decode_iter_ms": float,        # EMA of decode iteration time

    # === BATCH STATE ===
    "num_running": int,                 # len(self.running)
    "num_waiting": int,                 # len(self.waiting)
    "num_active_decode_seqs": int,      # requests past prefill phase

    # === CONTEXT LENGTH DISTRIBUTION ===
    # For running decode sequences, compute percentiles of num_computed_tokens
    "decode_ctx_p50": int,              # median context length
    "decode_ctx_p95": int,              # 95th percentile
    "decode_ctx_max": int,              # max context length

    # === BACKLOG IN TOKENS ===
    "pending_prefill_tokens": int,      # sum of prompt tokens in waiting queue
    "pending_decode_tokens": int,       # sum of remaining output tokens in waiting

    # === SCHEDULER KNOBS ===
    "token_budget_per_iter": int,       # self.max_num_scheduled_tokens
    "prefill_chunk_size": int,          # chunked prefill chunk size
    "max_num_seqs": int,                # self.max_num_running_reqs

    # === MEMORY PRESSURE ===
    "kv_cache_utilization": float,      # 1.0 - (free_blocks / total_blocks)
    "kv_free_blocks": int,              # free GPU blocks
    "kv_evictions_per_s": float,        # rate of KV evictions (need counter)

    # === METADATA ===
    "num_preempted": int,               # cumulative preemptions
    "timestamp": float,                 # time.time()
}
```

Implementation notes:

1. **EMA Service Rates**: Add these state variables to Scheduler.__init__():
   ```python
   self._ema_decode_tokens = 0.0
   self._ema_prefill_tokens = 0.0
   self._ema_decode_iter_ms = 0.0
   self._last_ema_update = time.time()
   self._ema_alpha = 0.1  # smoothing factor
   ```
   Update in `update_from_output()` after each iteration.

2. **Context Length Percentiles**: For decode sequences (num_computed_tokens >= num_prompt_tokens):
   ```python
   decode_ctx_lengths = [r.num_computed_tokens for r in self.running
                         if r.num_computed_tokens >= r.num_prompt_tokens]
   if decode_ctx_lengths:
       decode_ctx_p50 = int(np.percentile(decode_ctx_lengths, 50))
       decode_ctx_p95 = int(np.percentile(decode_ctx_lengths, 95))
       decode_ctx_max = max(decode_ctx_lengths)
   ```

3. **Pending Tokens**: For waiting queue:
   ```python
   pending_prefill = sum(r.num_prompt_tokens for r in self.waiting)
   pending_decode = sum(r.sampling_params.max_tokens for r in self.waiting)
   ```

4. **KV Evictions**: Add counter `self._kv_evictions_total` incremented on eviction,
   compute rate from delta since last call.

Add the endpoint to `vllm/entrypoints/openai/api_server.py`:
```python
@router.get("/instance_stats")
async def get_simple_scheduler_trace() -> Response:
    trace = await engine_client.get_simple_scheduler_trace()
    return Response(content=orjson.dumps(trace))
```

Update `vllm/engine/protocol.py` to include the new method.
```

### Prompt 2: Minimal Implementation (If Full Is Too Complex)

```
If the full feature set is too complex to implement quickly, here's a minimal version
that still provides useful signal for latency prediction:

```python
{
    # Minimal but useful
    "num_running": int,
    "num_waiting": int,
    "num_active_decode_seqs": int,      # count of seqs past prefill

    "decode_ctx_p95": int,              # 95th percentile context length
    "decode_ctx_max": int,              # max context length

    "pending_prefill_tokens": int,      # total prefill tokens in queue
    "pending_decode_tokens": int,       # total decode tokens in queue

    "kv_cache_utilization": float,
    "kv_free_blocks": int,

    "token_budget_per_iter": int,       # scheduler config
    "max_num_seqs": int,

    "timestamp": float,
}
```

This skips EMA service rates (hardest to add) but keeps context distribution and
token-level backlog which are high-value features.

The Block predictor can compute `backlog_seconds` using historical throughput data
collected from actual request completions if vLLM doesn't provide EMAs directly.
```

### Prompt 3: Verify and Test

```
Please verify the `/instance_stats` endpoint implementation:

1. Check that the endpoint is properly registered and accessible
2. Verify the response format matches the specification
3. Test edge cases:
   - Empty scheduler (no running/waiting requests)
   - Heavy load (many requests)
   - After preemption
   - Mix of prefill and decode sequences

4. Verify context length percentiles are computed correctly:
   - Only include decode sequences (num_computed_tokens >= num_prompt_tokens)
   - Handle edge case of no decode sequences (return 0)

5. If EMA tracking was added, verify:
   - EMAs update correctly after each iteration
   - Values are reasonable (not NaN, not negative)
   - Alpha smoothing factor produces stable estimates

Also update the legacy API server at `vllm/entrypoints/api_server.py` for consistency.
```

---

## 9. Code Changes Summary

### 9.1 New Files to Create

| File | Description |
|------|-------------|
| **Predictor Implementations** | |
| `block/predictor/cara/simple_schedule_state.py` | Lightweight aggregated state dataclass |
| `block/predictor/cara/xgboost_predictor.py` | XGBoost model wrapper (implements CARABasePredictor) |
| `block/predictor/cara/roofline_predictor.py` | Analytical roofline model (ablation baseline) |
| `block/predictor/cara/linear_predictor.py` | Simple linear regression (ablation baseline) |
| `block/predictor/cara/lstm_predictor.py` | LSTM sequence model (research baseline) |
| **Scheduler Components** | |
| `block/predictor/cara/state_manager.py` | Centralized state cache with `refresh_all()` |
| `block/predictor/cara/predictor_registry.py` | Maps instance_type to predictor instance |
| `block/predictor/cara/scheduling_policy.py` | BOOTSTRAP vs CARA mode selection |
| **Training & Evaluation** | |
| `block/predictor/cara/offline_training/train_xgboost.py` | XGBoost training script |
| `block/predictor/cara/offline_training/evaluate_predictor.py` | Ablation evaluation framework |
| **Config Files** | |
| `block/config/cara/predictor_config_xgboost.json` | XGBoost predictor config |
| `block/config/cara/predictor_config_roofline.json` | Roofline predictor config |
| `block/config/cara/predictor_config_lstm.json` | LSTM predictor config |

### 9.2 Files to Modify

| File | Changes |
|------|---------|
| **Core Framework** | |
| `block/predictor/cara/base_predictor.py` | Add `STATE_API_TYPE`, `REQUIRES_MODEL` class attributes |
| `block/predictor/cara/cara_predictor_config.py` | Add factory pattern with `PREDICTOR_CONFIG_REGISTRY` |
| `block/predictor/cara/data_structures.py` | Add `InstanceStats`, `instance_type`, `actual_output_length` |
| **Scheduler Integration** | |
| `block/global_scheduler/cara/cara_serve.py` | Add `/v1/batch_completions`, integrate StateManager |
| `block/global_scheduler/cara/cara_instance/Instance.py` | Single `data_collection_port`, `on_completion_callback` |
| **Data Collection** | |
| `block/predictor/cara/schedule_trace_client.py` | Add `fetch_instance_stats()` method |
| `block/predictor/cara/training_data_collector.py` | Add `state_api` field for format tracking |
| **Configuration** | |
| `block/config/cara/model_config_template.json` | Add `data_collection_port` per node_type |
| `block/exp/cara/deploy_cara.py` | Derive `instance_type` at deployment time |
| `block/exp/generate_config.py` | Remove `predictor_ports` generation |

### 9.3 Configuration Changes

**Port Configuration (IMPORTANT)**:

| Aspect | Before | After |
|--------|--------|-------|
| **Field name** | `predictor_ports` | `data_collection_port` |
| **Cardinality** | List (multiple ports) | Single port |
| **Location** | `host_configs.json` | `model_config_template.json` |
| **Purpose** | Unclear | Explicitly for data collection |

### 9.4 Documentation Updates

| Document | Changes |
|----------|---------|
| `cara_paper/claude/LATENCY_PREDICTOR_DESIGN.md` | This document (v2.0) |
| `cara_paper/claude/CARA_PROJECT_STATUS.md` | Update Phase 2 status, add implementation tasks |
| `cara_paper/claude/README.md` | Add extensible predictor section, update architecture |
| `block/predictor/cara/README.md` | Add predictor type documentation |
| `block/predictor/cara/offline_training/README.md` | Add evaluation framework docs |

---

## 10. Success Metrics

### 10.1 Prediction Accuracy

| Metric | Target |
|--------|--------|
| E2E MAE | < 200ms |
| E2E RMSE | < 300ms |
| P90 Error | < 400ms |
| TTFT MAE | < 50ms |

### 10.2 System Performance

| Metric | Target |
|--------|--------|
| Per-batch query latency | < 50ms |
| Per-request prediction | < 100μs |
| State staleness | < 100ms |
| Memory overhead | < 100MB |

### 10.3 Scheduling Quality

| Metric | Baseline (Random) | Target (XGBoost) |
|--------|-------------------|------------------|
| P99 latency | baseline | -20% |
| Mean latency | baseline | -15% |
| Load imbalance | baseline | -30% |

---

## 11. Rollout Plan

### Phase 1: Shadow Mode (Week 1-2)
- Deploy XGBoost predictor alongside existing system
- Log predictions but don't use for routing
- Compare predictions vs actual latencies
- Tune models based on production data

### Phase 2: Gradual Rollout (Week 3-4)
- Route 10% of traffic using predictions
- Monitor latency metrics
- A/B test vs random/round-robin baseline
- Increase to 50% if metrics improve

### Phase 3: Full Deployment (Week 5+)
- Route 100% of traffic using predictions
- Continuous monitoring and retraining
- Collect data for model improvements

---

## 12. Handling Missing EMA Service Rates

If vLLM doesn't provide EMA service rates directly, compute them at the scheduler side:

### 12.1 Scheduler-Side EMA Tracking

```python
# block/predictor/cara/service_rate_tracker.py

from collections import defaultdict
from dataclasses import dataclass, field
import time

@dataclass
class ServiceRateTracker:
    """Track service rates per instance from completed requests."""

    alpha: float = 0.1  # EMA smoothing factor

    # Per-instance EMAs
    ema_decode_tok_per_s: dict = field(default_factory=lambda: defaultdict(lambda: 1000.0))
    ema_prefill_tok_per_s: dict = field(default_factory=lambda: defaultdict(lambda: 2000.0))
    last_update: dict = field(default_factory=lambda: defaultdict(float))

    def update(self, instance_id: str, request_result: dict):
        """Update EMAs from completed request."""
        ttft = request_result.get('ttft', 0)
        e2e = request_result.get('e2e_latency', 0)
        prompt_tokens = request_result.get('num_prompt_tokens', 0)
        output_tokens = request_result.get('output_tokens', 0)

        if ttft > 0 and prompt_tokens > 0:
            prefill_rate = prompt_tokens / ttft
            self.ema_prefill_tok_per_s[instance_id] = (
                self.alpha * prefill_rate +
                (1 - self.alpha) * self.ema_prefill_tok_per_s[instance_id]
            )

        decode_time = e2e - ttft
        if decode_time > 0 and output_tokens > 0:
            decode_rate = output_tokens / decode_time
            self.ema_decode_tok_per_s[instance_id] = (
                self.alpha * decode_rate +
                (1 - self.alpha) * self.ema_decode_tok_per_s[instance_id]
            )

        self.last_update[instance_id] = time.time()

    def get_rates(self, instance_id: str) -> tuple:
        """Get current rate estimates for an instance."""
        return (
            self.ema_decode_tok_per_s[instance_id],
            self.ema_prefill_tok_per_s[instance_id],
        )
```

### 12.2 Augment State with Computed Rates

```python
# In SchedulerPredictorService.refresh_all_states()

async def refresh_all_states(self) -> Dict[str, SimpleScheduleState]:
    # Fetch base state from instances
    states = await self._fetch_all_states()

    # Augment with scheduler-side EMAs if not provided by vLLM
    for inst_id, state in states.items():
        if state.ema_decode_tok_per_s == 0:
            decode_rate, prefill_rate = self.rate_tracker.get_rates(inst_id)
            state.ema_decode_tok_per_s = decode_rate
            state.ema_prefill_tok_per_s = prefill_rate

    return states
```

---

## 13. Batch Scheduling Algorithm Integration

The latency predictor integrates with CARA's batch scheduler using **LPT (Longest Processing Time First)** heuristic and **Active Pull** state management.

### 13.1 Algorithm Overview

Unlike the original mini-batch permutation design (O(|I|^|C|) per mini-batch), the new algorithm uses a simple sorted linear scan:

```
Complexity: O(|R_batch| × |I|)
```

**Key Components**:
1. **Active Pull**: Fresh `/instance_stats` query before each batch (not Dead Reckoning)
2. **LPT Sort**: Sort requests by predicted tokens descending ("Big Rocks First")
3. **Greedy Assignment**: For each request, pick best instance based on multi-objective score
4. **Local Update**: Update state copy after each assignment (within-batch only)

### 13.2 Batch Scheduling Cycle

```
┌─────────────────────────────────────────────────────────────────┐
│                    BATCH SCHEDULING CYCLE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. PULL PHASE (before batch)                                   │
│     ┌─────────────────────────────────────────────┐             │
│     │  states = await pull_all_instance_stats()   │  ← 1 RTT    │
│     └─────────────────────────────────────────────┘             │
│                                                                  │
│  2. SORT PHASE (LPT heuristic)                                  │
│     ┌─────────────────────────────────────────────┐             │
│     │  sorted = sort_by_tokens_desc(batch)        │  ← O(N log N) │
│     └─────────────────────────────────────────────┘             │
│                                                                  │
│  3. ASSIGN PHASE (greedy linear scan)                           │
│     for r in sorted_batch:                                      │
│         scores = [score(r, i, states[i]) for i in instances]    │
│         best = argmin(scores)                                   │
│         assign(r, best)                                         │
│         ┌──────────────────────────────────────┐                │
│         │  if num_running < max_num_seqs:      │  ← Local only  │
│         │      states[best].num_running += 1   │                │
│         │  else:                               │                │
│         │      states[best].num_waiting += 1   │                │
│         │  states[best].pending_tokens += ...  │                │
│         └──────────────────────────────────────┘                │
│                                                                  │
│  4. DISPATCH PHASE                                              │
│     dispatch_all(assignments)                                    │
│                                                                  │
│  5. NEXT BATCH → Go to step 1 (fresh pull)                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 13.3 Why LPT (Longest Processing Time First)?

Graham (1969) showed LPT is a 4/3-approximation for makespan minimization on identical machines.

**Intuition**: Placing large jobs first leaves flexibility to fill gaps with smaller jobs:

```
Without LPT (FCFS):          With LPT:
┌────┬─┬──────┐              ┌──────┬────┬─┐
│ A  │B│  C   │              │  C   │ A  │B│
├────┴─┴──────┤              ├──────┴────┴─┤
│     D       │              │     D       │
└─────────────┘              └─────────────┘
  Makespan: 10                 Makespan: 8

Small jobs at end create gaps  Big jobs first, small fill gaps
```

### 13.4 Dynamic Batch Size (Load-Aware Adaptive Batching)

Unlike fixed batch size + timeout (e.g., Ray Serve), CARA uses **load-aware adaptive batching**:

```python
class AdaptiveBatcher:
    def __init__(self, min_batch=4, max_batch=64, min_timeout_ms=10, max_timeout_ms=100):
        self.min_batch = min_batch
        self.max_batch = max_batch
        self.min_timeout_ms = min_timeout_ms
        self.max_timeout_ms = max_timeout_ms

    def get_batch_params(self, num_busy_instances: int, total_instances: int):
        """Adjust batch size based on cluster load."""
        load_ratio = num_busy_instances / total_instances

        if load_ratio < 0.3:  # Low load
            # Small batch, short timeout → minimize latency
            return self.min_batch, self.min_timeout_ms
        elif load_ratio < 0.7:  # Medium load
            # Interpolate
            batch = int(self.min_batch + (self.max_batch - self.min_batch) * load_ratio)
            timeout = int(self.min_timeout_ms + (self.max_timeout_ms - self.min_timeout_ms) * load_ratio)
            return batch, timeout
        else:  # High load
            # Large batch, long timeout → improve throughput
            # (requests would queue at instance anyway)
            return self.max_batch, self.max_timeout_ms
```

**Key insight**: When instances are busy, batching delay at scheduler is "free" - requests would queue at the instance regardless. Larger batches improve global optimization.

### 13.5 Multi-Objective Scoring with Tiered Constraint Handling

Each (request, instance) pair is scored with a **tiered constraint relaxation** strategy:

```python
def score_with_constraints(request, instance, state, config):
    """Score with tiered constraint handling.

    Returns: (tier, score) where lower tier = better (more constraints satisfied)
    """
    # Get predictions
    latency = predictor.predict(state, request)
    quality = quality_estimator.predict(request, instance.model)
    cost = cost_model.predict(request, instance)

    # Check constraints and count violations
    violations = []

    if latency > min(request.rso_latency, config.slo_latency):
        violations.append(('latency', latency - request.rso_latency))
    if quality < request.rso_quality_min:
        violations.append(('quality', request.rso_quality_min - quality))
    if cost > request.rso_budget:
        violations.append(('cost', cost - request.rso_budget))

    # Tier = number of violations (0 = all satisfied)
    tier = len(violations)

    # Base score (weighted sum of satisfied metrics)
    base_score = (
        config.w_lat * normalize(latency) +
        config.w_cost * normalize(cost) -
        config.w_qual * normalize(quality) +
        config.w_util * state.kv_cache_utilization
    )

    # Violation penalty (for sorting within tier)
    violation_penalty = sum(
        config.violation_weights.get(v[0], 1.0) * v[1]
        for v in violations
    )

    return (tier, base_score + violation_penalty)


def select_best_instance(request, instances, states, config):
    """Select best instance with tiered relaxation.

    Strategy (configurable):
    1. STRICT: Only consider tier 0 (all constraints satisfied), reject if empty
    2. RELAXED: Sort by tier, then by score within tier
    3. PENALTY: Add penalty to score (original approach)
    """
    scored = []
    for inst in instances:
        tier, score = score_with_constraints(request, inst, states[inst.id], config)
        scored.append((tier, score, inst))

    # Sort by tier first, then by score within tier
    scored.sort(key=lambda x: (x[0], x[1]))

    if config.constraint_mode == 'STRICT':
        # Only accept tier 0 candidates
        tier0 = [s for s in scored if s[0] == 0]
        if not tier0:
            return None  # Reject request
        return tier0[0][2]

    elif config.constraint_mode == 'RELAXED':
        # Accept best available (lowest tier, then lowest score)
        if not scored:
            return None
        return scored[0][2]

    elif config.constraint_mode == 'TIERED':
        # Try each tier in order, reject only if all tiers empty
        for target_tier in range(max(s[0] for s in scored) + 1):
            tier_candidates = [s for s in scored if s[0] == target_tier]
            if tier_candidates:
                return tier_candidates[0][2]
        return None
```

**Tiered Selection Example**:

```
Request with RSO: latency < 500ms, quality > 0.8

Instance A: latency=400ms, quality=0.85  → tier=0 (all satisfied)
Instance B: latency=600ms, quality=0.90  → tier=1 (latency violated)
Instance C: latency=450ms, quality=0.75  → tier=1 (quality violated)
Instance D: latency=700ms, quality=0.70  → tier=2 (both violated)

Sorted order:
  Tier 0: [A]                    ← Pick from here first
  Tier 1: [B, C] (sorted by score)  ← Only if tier 0 empty
  Tier 2: [D]                    ← Last resort
```

**Configuration Options**:

| Mode | Behavior |
|------|----------|
| `STRICT` | Reject if no tier-0 candidate |
| `RELAXED` | Always pick best available |
| `TIERED` | Try tier-0, then tier-1, etc. Reject only if all empty |

### 13.6 Within-Batch State Updates

After assigning a request to an instance, update the local state copy:

```python
def update_local_state(state: InstanceStats, request: Request):
    """Update state estimate for next request in batch."""

    # Check if instance has capacity for immediate execution
    if state.num_running < state.max_num_seqs:
        # Request goes to running queue (will start immediately)
        state.num_running += 1
    else:
        # Instance at capacity, request goes to waiting queue
        state.num_waiting += 1

    # Update pending tokens (backlog)
    state.pending_prefill_tokens += request.num_prompt_tokens
    state.pending_decode_tokens += request.predicted_output_tokens

    # Estimate new KV utilization
    tokens_to_add = request.num_prompt_tokens + request.predicted_output_tokens
    blocks_needed = tokens_to_add // state.block_size
    state.kv_free_blocks = max(0, state.kv_free_blocks - blocks_needed)
    state.kv_cache_utilization = 1.0 - (state.kv_free_blocks / state.total_blocks)
```

**Important**:
- These updates are only used within the current batch. The next batch starts with a fresh pull from `/instance_stats`.
- The `num_running < max_num_seqs` check determines if the request can start immediately or must wait.

### 13.7 Comparison: Active Pull vs Dead Reckoning

| Aspect | Dead Reckoning (Paper) | Active Pull (Implementation) |
|--------|------------------------|------------------------------|
| State freshness | Estimated via decay | Fresh each batch |
| Network overhead | Event-driven telemetry | 1 RTT per batch |
| Complexity | Higher (decay model) | Lower (just pull) |
| Accuracy | Depends on decay model | Ground truth from vLLM |
| State drift | Possible | None |

Active Pull is chosen for implementation simplicity and accuracy. The 1 RTT overhead per batch is acceptable for batch sizes > 8.

---

## 14. Open Questions (Updated)

1. **Feature engineering**: Are there additional features that would improve predictions?
   - Request priority
   - Time since last request
   - Model-specific features (context window utilization)

2. **Multi-objective**: How to combine latency prediction with quality estimation?
   - Weighted sum: `score = α * latency + β * (1 - quality)`
   - Pareto optimization

3. **Online learning**: Should we update models online or batch retrain?
   - Online: More adaptive but complex
   - Batch: Simpler but may lag behind workload changes

4. **Staleness handling**: What to do when state is stale?
   - Use stale state with penalty
   - Fallback to heuristic
   - Force refresh (adds latency)

5. **EMA implementation**: vLLM-side or scheduler-side?
   - vLLM-side: More accurate (sees every iteration)
   - Scheduler-side: Easier to implement, works with existing vLLM

---

## Appendix A: Feature Importance Analysis

After training, analyze feature importance to validate design:

```python
import matplotlib.pyplot as plt

# Get feature importance
importance = model.get_score(importance_type='weight')

# Plot
feature_names = [
    'num_running', 'num_waiting', 'total_prefill_tokens',
    'total_decode_tokens', 'kv_cache_utilization', 'free_gpu_blocks',
    'num_preempted', 'num_prompt_tokens', 'num_predicted_output_tokens'
]

plt.figure(figsize=(10, 6))
plt.barh(feature_names, [importance.get(f'f{i}', 0) for i in range(len(feature_names))])
plt.xlabel('Feature Importance (weight)')
plt.title('XGBoost Latency Predictor Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')
```

Expected high-importance features:
1. `num_prompt_tokens` (request size)
2. `num_running` (current load)
3. `total_prefill_tokens` (pending work)
4. `kv_cache_utilization` (memory pressure)

---

**Document End**
