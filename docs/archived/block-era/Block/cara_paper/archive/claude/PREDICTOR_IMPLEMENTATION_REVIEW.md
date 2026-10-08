# CARA Predictor Implementation Review Document

**Last Updated**: January 26, 2026

## Status Update

> **📖 Task 1.4 Multi-Objective Scoring**: ✅ COMPLETE (January 26, 2026)
> - RSO/SLO data structures and tiered constraint handling
> - MultiObjectiveScorer with ScoringConfig integration
> - Full integration with CARABatchPolicy
> - 166 passing tests
>
> **📖 Implementation Tasks V2**: See [IMPLEMENTATION_TASKS_V2_DESIGN.md](IMPLEMENTATION_TASKS_V2_DESIGN.md) for detailed designs addressing review feedback (12 pending tasks).

## Overview

This document provides a comprehensive review of the scheduler-side latency predictor implementation for the CARA (Context-Aware Request Allocation) system. The implementation enables the global scheduler to make intelligent routing decisions based on predicted request latencies.

**Total Lines of Code Added:**
- Core Implementation: ~4,500+ lines (updated with Task 1.4)
- Test Code: ~2,500+ lines (166 tests for CARA predictor)
- Modified Files: ~1,000+ lines of changes

## Architecture Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CARA Global Scheduler                                 │
│                        (cara_serve.py)                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────┐  ┌──────────────────┐  ┌─────────────────────┐       │
│   │  StateManager   │  │ PredictorRegistry│  │  SchedulingPolicy   │       │
│   │                 │  │                  │  │                     │       │
│   │ - refresh_all() │  │ - register()     │  │ - route_batch()     │       │
│   │ - get_state()   │  │ - predict()      │  │ - BOOTSTRAP/CARA    │       │
│   │ - apply_assign()│  │ - from_config()  │  │   mode selection    │       │
│   └────────┬────────┘  └────────┬─────────┘  └──────────┬──────────┘       │
│            │                    │                       │                   │
│            │                    ▼                       │                   │
│            │   ┌────────────────────────────────────┐  │                   │
│            │   │        Predictor Types             │  │                   │
│            │   │  ┌──────────┬─────────┬─────────┐ │  │                   │
│            │   │  │ XGBoost  │ Linear  │Roofline │ │  │                   │
│            │   │  │(25 feat) │(4 coef) │(analyt.)│ │  │                   │
│            │   │  └──────────┴─────────┴─────────┘ │  │                   │
│            │   └────────────────────────────────────┘  │                   │
│            │                                           │                   │
│            ▼                                           ▼                   │
│   ┌─────────────────┐                     ┌─────────────────────┐         │
│   │ServiceRateTracker│◄────────────────────│ Completed Requests  │         │
│   │                 │  (feedback loop)     │ (ttft, e2e_latency) │         │
│   │ - update()      │                      └─────────────────────┘         │
│   │ - augment_state │                                                       │
│   └─────────────────┘                                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## File Changes Summary

### New Files Created

| File | Lines | Purpose |
|------|-------|---------|
| `state_manager.py` | 317 | Centralized state caching with parallel refresh |
| `predictor_registry.py` | 444 | Factory pattern for predictor instantiation |
| `scheduling_policy.py` | 407 | BOOTSTRAP vs CARA mode selection and routing |
| `service_rate_tracker.py` | 322 | EMA service rate computation from feedback |
| `roofline_predictor.py` | 265 | Analytical baseline predictor |
| `linear_predictor.py` | 199 | Linear regression predictor |
| `xgboost_predictor.py` | 340 | Primary XGBoost predictor (25 features) |
| `train_xgboost.py` | 562 | XGBoost training pipeline |
| `train_linear.py` | 255 | Linear model training pipeline |
| `evaluate_predictor.py` | 710 | Evaluation framework for ablation studies |

### Modified Files

| File | Changes | Purpose |
|------|---------|---------|
| `cara_serve.py` | +360 lines | Integration of all predictor components |
| `base_predictor.py` | +85 lines | Abstract base class with StateApiType enum |
| `cara_predictor_config.py` | +196 lines | Extended configuration for predictor types |
| `data_structures.py` | +165 lines | SimpleScheduleState for INSTANCE_STATS API |
| `schedule_trace_client.py` | +69 lines | Added fetch_instance_stats() method |
| `dummy_cara_predictor.py` | +21 lines | Updated to use new base class |

### Test Files Created

| File | Lines | Tests |
|------|-------|-------|
| `test_data_structures.py` | 216 | 13 tests |
| `test_predictors.py` | 371 | 23 tests (9 skipped async) |
| `test_state_manager.py` | 248 | 17 tests (4 skipped async) |
| `test_predictor_registry.py` | 195 | 16 tests (4 skipped async) |
| `test_scheduling_policy.py` | 297 | 19 tests (6 skipped async) |
| `test_service_rate_tracker.py` | 313 | 21 tests |
| `test_evaluate_predictor.py` | 358 | 18 tests |

**Total: 108 tests passed, 23 skipped (async tests need pytest-asyncio)**

---

## Component Details

### 1. StateManager (`state_manager.py`)

**Purpose:** Centralized state management for scheduler-side predictors. Queries all instances in parallel at batch boundaries and maintains cached state.

**Key Design Decisions:**
- Uses `asyncio.gather()` for parallel state refresh across all instances
- Implements health tracking with consecutive failure counts
- Marks instances as "down" after configurable max failures
- Preserves stale state on failure (doesn't clear cache)

**Key Classes:**
```python
@dataclass
class InstanceHealth:
    consecutive_failures: int = 0
    last_success: float = field(default_factory=time.time)
    is_down: bool = False

class StateManager:
    def __init__(self, instances, timeout_ms=50.0, max_consecutive_failures=3, stale_threshold_ms=200.0)
    async def refresh_all(self) -> Dict[str, SimpleScheduleState]
    def get_state(self, instance_id) -> Optional[SimpleScheduleState]
    def apply_assignment(self, instance_id, request)  # Within-batch update
    def get_healthy_instances(self) -> List[str]
```

**Integration Points:**
- Called by `cara_serve.py` at batch start via `refresh_all()`
- Used by `SchedulingPolicy` to get instance states for routing decisions

**Review Notes:**
- Line 80-82: Instance mapping uses `instance_id` as key
- Line 116-117: Skips down instances in refresh
- Line 243-257: `apply_assignment()` updates local state copy after routing decision

---

### 2. PredictorRegistry (`predictor_registry.py`)

**Purpose:** Factory pattern for managing multiple predictor types. Maps instance types to specific predictor instances.

**Key Design Decisions:**
- Supports per-instance-type models (e.g., `qwen-2-5-3b_vllm_c240g5.xgb`)
- Validates STATE_API_TYPE consistency across all predictors
- Supports fallback predictor for unknown instance types

**Key Functions:**
```python
def derive_instance_type(model_name: str, backend: str, node_type: str) -> str:
    """Creates normalized instance type string.
    Example: "Qwen/Qwen2.5-3B", "vllm", "c240g5" -> "qwen-qwen2-5-3b_vllm_c240g5"
    """

class PredictorRegistry:
    def register(self, instance_type: str, predictor: CARABasePredictor)
    def get_predictor(self, instance_type: str) -> Optional[CARABasePredictor]
    async def predict(self, instance_type, state, request) -> Dict
    @classmethod
    def from_config(cls, config, instance_types, model_dir=None) -> 'PredictorRegistry'
```

**Factory Method (`from_config`):**
- Lines 250-310: Creates appropriate predictor based on config type
- Supports: `xgboost`, `linear`, `roofline`, `dummy`
- Automatically loads models from `model_dir/{instance_type}.xgb`

**Review Notes:**
- Line 75-92: Validates that all predictors use same STATE_API_TYPE
- Line 157-170: `predict()` delegates to registered predictor or fallback

---

### 3. SchedulingPolicy (`scheduling_policy.py`)

**Purpose:** Manages scheduling mode (BOOTSTRAP vs CARA) and routes requests to instances.

**Key Design Decisions:**
- BOOTSTRAP mode: Uses simple strategies (random, round_robin, least_loaded) for data collection
- CARA mode: Uses XGBoost predictions for greedy min-latency routing
- Automatic mode transition after collecting enough training samples

**Key Classes:**
```python
class SchedulingMode(Enum):
    BOOTSTRAP = "bootstrap"  # Data collection phase
    CARA = "cara"            # Prediction-based routing

class BootstrapStrategy(Enum):
    RANDOM = "random"
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"

class SchedulingPolicy:
    def __init__(self, state_manager, predictor_registry=None,
                 bootstrap_strategy=ROUND_ROBIN, min_training_samples=1000)
    async def route_batch(self, requests, instance_ids) -> Dict[str, List[PredictRequest]]
    async def route_single(self, request, instance_ids) -> str
    def record_training_sample(self)  # Called on request completion
```

**Mode Transition Logic:**
- Lines 85-105: Starts in BOOTSTRAP if no predictor_registry
- Line 123-130: Transitions to CARA after `min_training_samples`

**CARA Routing Algorithm (Lines 210-270):**
```
For each request in batch:
  1. Get all instance states from StateManager
  2. Predict latency for each instance using PredictorRegistry
  3. Select instance with minimum predicted latency
  4. Apply assignment to update local state copy
```

**Review Notes:**
- Line 157-175: `_route_bootstrap()` implements simple strategies
- Line 178-208: `_route_cara()` implements greedy min-latency
- Line 235-252: `_route_least_loaded()` uses num_running + num_waiting

---

### 4. ServiceRateTracker (`service_rate_tracker.py`)

**Purpose:** Computes EMA service rates from completed request feedback when vLLM doesn't provide them directly.

**Key Design Decisions:**
- Per-instance EMA tracking for decode and prefill rates
- Thread-safe with lock protection
- Augments `SimpleScheduleState` when EMA fields are zero
- Configurable alpha (smoothing factor)

**Key Classes:**
```python
@dataclass
class InstanceRateStats:
    ema_decode_tok_per_s: float = 1000.0
    ema_prefill_tok_per_s: float = 2000.0
    ema_decode_iter_ms: float = 10.0
    last_update: float = 0.0
    update_count: int = 0

class ServiceRateTracker:
    def update(self, instance_id: str, request_result: dict)
    def get_rates(self, instance_id) -> Tuple[float, float]
    def augment_state(self, state: SimpleScheduleState)
    def augment_states(self, states: Dict[str, SimpleScheduleState])
```

**EMA Update Formula (Lines 114-137):**
```python
# Prefill rate from TTFT
prefill_rate = prompt_tokens / ttft
ema_prefill = alpha * prefill_rate + (1 - alpha) * ema_prefill

# Decode rate from (e2e - ttft)
decode_rate = output_tokens / decode_time
ema_decode = alpha * decode_rate + (1 - alpha) * ema_decode
```

**Review Notes:**
- Line 251-261: `_is_stale_unlocked()` prevents deadlock in `get_stats()`
- Line 213-231: `augment_state()` only fills zero fields (non-destructive)

---

### 5. Predictor Implementations

#### 5.1 XGBoostLatencyPredictor (`xgboost_predictor.py`)

**Purpose:** Primary CARA predictor using XGBoost with 25 features.

**Feature List (25 total):**
```python
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
    'num_prompt_tokens', 'num_predicted_output_tokens', 'req_decode_ctx_avg', 'req_decode_ctx_max',
    # Derived features (4)
    'backlog_seconds', 'request_prefill_time_est', 'request_decode_time_est', 'ctx_relative_to_p95',
]
```

**Key Methods:**
```python
class XGBoostLatencyPredictor(CARABasePredictor):
    STATE_API_TYPE = StateApiType.INSTANCE_STATS
    REQUIRES_MODEL = True

    def _build_features(self, request, state) -> np.ndarray  # Returns 25-dim vector
    async def predict(self, request, state) -> Dict
    async def predict_batch(self, requests, state) -> List[Dict]
```

**Review Notes:**
- Lines 78-150: `_build_features()` computes all 25 features including derived ones
- Line 164-175: `predict()` returns `{'target_metric': latency, 'latency_prediction': latency}`

#### 5.2 LinearLatencyPredictor (`linear_predictor.py`)

**Purpose:** Simple linear baseline with 4 coefficients.

**Formula:**
```
latency = α_prefill * num_prompt_tokens
        + α_decode * num_predicted_output_tokens
        + α_waiting * num_waiting
        + bias
```

**Key Methods:**
```python
class LinearLatencyPredictor(CARABasePredictor):
    def set_weights(self, alpha_prefill, alpha_decode, alpha_waiting, bias)
    def load_weights(self, weights_path)
    async def predict(self, request, state) -> Dict
```

#### 5.3 RooflineLatencyPredictor (`roofline_predictor.py`)

**Purpose:** Analytical baseline using roofline model.

**Formula:**
```
latency = backlog_time + prefill_time + decode_time + iteration_overhead

where:
  backlog_time = pending_prefill/prefill_rate + pending_decode/decode_rate
  prefill_time = prompt_tokens / prefill_rate
  decode_time = output_tokens / decode_rate
  iteration_overhead = output_tokens * decode_iter_ms / 1000
```

---

### 6. Integration with cara_serve.py

**Changes Overview:**
- Added 360 lines of new code
- New endpoints: `/v1/batch_completions`, `/v1/scheduler_stats`
- New CLI arguments for predictor configuration

**New Global Variables (Lines 51-55):**
```python
state_manager: Optional[StateManager] = None
predictor_registry: Optional[PredictorRegistry] = None
scheduling_policy: Optional[SchedulingPolicy] = None
service_rate_tracker: Optional[ServiceRateTracker] = None
instance_type_map: Dict[str, str] = {}
```

**New CLI Arguments:**
```
--predictor-config     Path to predictor config JSON
--scheduler-config     Path to scheduler config JSON
--model-dir            Directory containing trained models
--state-query-timeout-ms  Timeout for /instance_stats queries (default: 50.0)
--ema-alpha            EMA smoothing factor (default: 0.1)
--default-decode-rate  Default decode rate (default: 1000.0)
--default-prefill-rate Default prefill rate (default: 2000.0)
```

**Initialization Flow (`_init_scheduler_side_components`):**
```
1. Build instance_type_map from model config
2. Initialize StateManager with instances
3. Initialize ServiceRateTracker
4. Initialize PredictorRegistry from config (if provided)
5. Initialize SchedulingPolicy with all components
```

**Batch Completions Endpoint Flow (`/v1/batch_completions`):**
```
1. Refresh all instance states via StateManager
2. Augment states with EMA rates from ServiceRateTracker
3. Convert requests to PredictRequest format
4. Route batch using SchedulingPolicy
5. Execute requests on assigned instances (parallel)
6. Update ServiceRateTracker with completed request metrics
7. Return responses with scheduling stats
```

**Review Notes:**
- Lines 139-147: State refresh + EMA augmentation
- Lines 186-203: Request execution with instance tracking
- Lines 203-217: ServiceRateTracker update on completion

---

### 7. Evaluation Framework (`evaluate_predictor.py`)

**Purpose:** Compare predictor types using standard metrics.

**Metrics Computed:**
```python
@dataclass
class EvaluationMetrics:
    mae: float           # Mean Absolute Error
    rmse: float          # Root Mean Squared Error
    mape: float          # Mean Absolute Percentage Error (%)
    r2: float            # R² coefficient of determination
    p50_error: float     # Median absolute error
    p75_error: float
    p90_error: float
    p95_error: float
    p99_error: float
    mean_error: float    # Bias (positive = overprediction)
    under_prediction_rate: float  # % underpredictions
    avg_prediction_time_us: float
```

**Evaluation Functions:**
```python
async def evaluate_xgboost(examples, model_path, instance_type) -> EvaluationMetrics
async def evaluate_linear(examples, weights_path, instance_type) -> EvaluationMetrics
async def evaluate_roofline(examples, calibration_path, instance_type) -> EvaluationMetrics
def evaluate_baseline_random(examples, instance_type) -> EvaluationMetrics
```

**Usage Example:**
```bash
python -m block.predictor.cara.offline_training.evaluate_predictor \
    --test-data test_data/*.jsonl \
    --model-type all \
    --xgboost-model models/xgboost/instance_type.xgb \
    --linear-weights models/linear/instance_type.json \
    --output-json report.json --detailed
```

---

### 8. Training Pipelines

#### 8.1 XGBoost Training (`train_xgboost.py`)

**Features:**
- Loads training data from JSONL files
- Groups data by instance type
- Trains with early stopping and cross-validation
- Outputs model files and metrics JSON

**Usage:**
```bash
python -m block.predictor.cara.offline_training.train_xgboost \
    --input training_data/*.jsonl \
    --output-dir models/xgboost \
    --max-depth 6 --n-estimators 100
```

#### 8.2 Linear Training (`train_linear.py`)

**Features:**
- Uses sklearn LinearRegression (OLS)
- Extracts 4 features: prompt_tokens, output_tokens, num_waiting, bias
- Outputs JSON weights file

---

## Data Structures

### SimpleScheduleState (Extended)

```python
@dataclass
class SimpleScheduleState:
    instance_id: str = ""

    # === SERVICE RATE SIGNALS ===
    ema_decode_tok_per_s: float = 0.0
    ema_prefill_tok_per_s: float = 0.0
    ema_decode_iter_ms: float = 0.0

    # === BATCH STATE ===
    num_running: int = 0
    num_waiting: int = 0
    num_active_decode_seqs: int = 0

    # === CONTEXT LENGTH DISTRIBUTION ===
    decode_ctx_mean: int = 0
    decode_ctx_p50: int = 0
    decode_ctx_p95: int = 0
    decode_ctx_max: int = 0
    decode_ctx_total: int = 0

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

    def compute_backlog_seconds(self) -> float
    def is_stale(self, max_age_ms=200) -> bool
    @classmethod
    def from_response(cls, instance_id, response) -> 'SimpleScheduleState'
```

### StateApiType Enum

```python
class StateApiType(Enum):
    SCHEDULE_TRACE = "schedule_trace"    # Detailed per-request data
    INSTANCE_STATS = "instance_stats"    # Aggregated statistics (used by XGBoost)
```

---

## Configuration Examples

### Predictor Config (XGBoost)
```json
{
    "predictor_type": "xgboost",
    "model_dir": "models/cara/xgboost",
    "state_api_type": "instance_stats"
}
```

### Predictor Config (Linear)
```json
{
    "predictor_type": "linear",
    "weights_path": "models/cara/linear/instance_type.json",
    "state_api_type": "instance_stats"
}
```

### Scheduler Config
```json
{
    "mode": "cara",
    "bootstrap_strategy": "round_robin",
    "min_training_samples": 1000
}
```

---

## Known Limitations and Future Work

1. **Async Tests Skipped**: 23 async tests require `pytest-asyncio` plugin
2. **EMA Cold Start**: ServiceRateTracker uses default rates until first request completes
3. **Single-Instance-Type Models**: Each predictor is trained per instance type
4. **No Online Learning**: Models must be retrained offline

---

## Test Coverage Summary

| Component | Tests | Passed | Skipped |
|-----------|-------|--------|---------|
| Data Structures | 13 | 13 | 0 |
| Predictors | 23 | 14 | 9 |
| StateManager | 17 | 13 | 4 |
| PredictorRegistry | 16 | 12 | 4 |
| SchedulingPolicy | 19 | 13 | 6 |
| ServiceRateTracker | 21 | 21 | 0 |
| Evaluation Framework | 18 | 18 | 0 |
| **Total** | **108** | **108** | **23** |

---

## Files to Review (Priority Order)

1. **`cara_serve.py`** - Integration point, most critical
2. **`scheduling_policy.py`** - Core routing logic
3. **`state_manager.py`** - State caching and health tracking
4. **`service_rate_tracker.py`** - EMA computation
5. **`predictor_registry.py`** - Factory pattern
6. **`xgboost_predictor.py`** - Primary predictor
7. **`data_structures.py`** - SimpleScheduleState definition

---

## Quick Validation Commands

```bash
# Run all tests
python -m pytest tests/predictor/cara/ -v --tb=short

# Run specific component tests
python -m pytest tests/predictor/cara/test_state_manager.py -v
python -m pytest tests/predictor/cara/test_service_rate_tracker.py -v
python -m pytest tests/predictor/cara/test_scheduling_policy.py -v

# Check imports work
python -c "from block.predictor.cara.state_manager import StateManager; print('OK')"
python -c "from block.predictor.cara.scheduling_policy import SchedulingPolicy; print('OK')"
python -c "from block.predictor.cara.service_rate_tracker import ServiceRateTracker; print('OK')"
```
