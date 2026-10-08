# CARA Implementation Tasks V2 - Detailed Design

**Date**: January 26, 2026
**Status**: Design Phase
**Reference**: `review_comments_for_implementation_predictor_v1.md`

---

## Overview

This document provides detailed design specifications for 12 pending tasks identified from the review comments. Each task includes:
- Problem statement and motivation
- Detailed design with code snippets
- File changes required
- Testing strategy
- Dependencies on other tasks

---

## Task #1: Fix cara_serve batch_completions to Support Streaming Response

### Problem Statement

The current `batch_completions` endpoint waits for ALL requests to complete before returning ANY response. This introduces unnecessary delay for faster requests that complete early.

**Current Behavior** (cara_serve.py:225-253):
```python
# Waits sequentially for each future
for request_id, inst_id, prompt_tokens, future in response_futures:
    response = await future  # BLOCKING: waits even if other requests finished
    responses.append(response)
```

**Design Requirement**: Return responses immediately as they complete, not wait for the slowest request.

### Detailed Design

#### Option A: Streaming Response with Server-Sent Events (SSE)

```python
from fastapi.responses import StreamingResponse
import asyncio

@app.post("/v1/batch_completions")
async def batch_completions(request: Request) -> Response:
    request_json = await request.json()
    batch_requests = request_json.get("requests", [])
    wait_all = request_json.get("wait_all", False)  # Default: streaming

    if wait_all:
        # Legacy behavior: wait for all
        return await _batch_completions_wait_all(batch_requests)
    else:
        # New streaming behavior
        return StreamingResponse(
            _batch_completions_streaming(batch_requests),
            media_type="text/event-stream"
        )

async def _batch_completions_streaming(batch_requests: List[dict]):
    """Generator that yields responses as they complete."""
    # ... dispatch requests ...

    # Use asyncio.as_completed to yield results as they finish
    for coro in asyncio.as_completed(response_futures):
        try:
            request_id, response = await coro
            yield f"data: {json.dumps({'request_id': request_id, 'response': response})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'request_id': request_id, 'error': str(e)})}\n\n"

    # Final message with stats
    yield f"data: {json.dumps({'type': 'complete', 'stats': scheduling_stats})}\n\n"
```

#### Option B: Callback-based with Future Resolution (Recommended)

This option maintains JSON response format while enabling streaming internally.

```python
class StreamingBatchScheduler:
    """Background scheduler that dispatches requests and resolves futures immediately."""

    def __init__(self):
        self.pending_responses: Dict[str, asyncio.Future] = {}

    async def dispatch_request(self, inst_id: str, request: dict) -> asyncio.Future:
        """Dispatch request and return future that resolves when complete."""
        request_id = request["request_id"]
        response_future = asyncio.Future()
        self.pending_responses[request_id] = response_future

        # Fire-and-forget: dispatch with callback
        asyncio.create_task(self._execute_and_resolve(inst_id, request, response_future))
        return response_future

    async def _execute_and_resolve(self, inst_id: str, request: dict, future: asyncio.Future):
        """Execute request and resolve future immediately when done."""
        try:
            response = await instance_map[inst_id].query_instance(request)
            future.set_result(response)
        except Exception as e:
            future.set_exception(e)


@app.post("/v1/batch_completions")
async def batch_completions(request: Request) -> Response:
    request_json = await request.json()
    batch_requests = request_json.get("requests", [])
    wait_all = request_json.get("wait_all", False)

    # Dispatch all requests
    futures_with_ids = []
    for req in batch_requests:
        inst_id = await scheduling_policy.route_single(req, instance_ids)
        future = await streaming_scheduler.dispatch_request(inst_id, req)
        futures_with_ids.append((req["request_id"], future))

    if wait_all:
        # Wait for all (legacy behavior)
        responses = []
        for request_id, future in futures_with_ids:
            response = await future
            responses.append(response)
        return JSONResponse(content={"responses": responses, "scheduling_stats": stats})
    else:
        # Return responses as they complete (using as_completed)
        responses = []
        for coro in asyncio.as_completed([f for _, f in futures_with_ids]):
            response = await coro
            responses.append(response)
        return JSONResponse(content={"responses": responses, "scheduling_stats": stats})
```

### API Changes

**Request**:
```json
{
    "requests": [...],
    "scheduling_mode": "cara",
    "wait_all": false  // NEW: default false for streaming
}
```

**Response** (SSE mode):
```
data: {"request_id": "req-1", "response": {...}}
data: {"request_id": "req-3", "response": {...}}  // req-3 finished before req-2
data: {"request_id": "req-2", "response": {...}}
data: {"type": "complete", "stats": {...}}
```

### Files to Modify

1. `block/global_scheduler/cara/cara_serve.py`
   - Add `wait_all` parameter handling
   - Implement streaming response generator
   - Use `asyncio.as_completed()` for response ordering

### Testing Strategy

1. Unit test: Verify responses arrive out-of-order when using wait_all=false
2. Integration test: Mock slow/fast instances, verify fast responses return first
3. Backward compatibility: Verify wait_all=true maintains current behavior

---

## Task #2: Fix XGBoost Predictor Target Field Naming and Config

### Problem Statement

Current XGBoost predictor returns both `target_metric` and `latency_prediction`, which is:
1. Redundant
2. Incorrect if predicting TTFT or TPOT instead of E2E latency

**Current** (xgboost_predictor.py:171-176):
```python
return {
    "target_metric": float(prediction),
    "latency_prediction": float(prediction),  # Wrong if predicting TTFT
    ...
}
```

### Detailed Design

#### Config Changes

```python
# In cara_predictor_config.py

@dataclass
class XGBoostPredictorConfig(CARABasePredictorConfig):
    predictor_type: str = field(default="xgboost")

    # NEW: Target metric type being predicted
    target_metric_type: str = field(
        default="e2e_latency",
        metadata={"help": "Target metric: 'e2e_latency', 'ttft', 'tpot'"}
    )

    # ... other fields ...
```

#### Predictor Changes

```python
# In xgboost_predictor.py

class XGBoostLatencyPredictor(CARABasePredictor):

    def __init__(self, config: XGBoostPredictorConfig, ...):
        super().__init__(config, backend_port)
        self.target_metric_type = config.target_metric_type
        # ...

    async def predict(self, target_request: PredictRequest, state=None) -> Dict:
        # ... prediction logic ...

        return {
            "prediction": float(prediction),
            "target_metric_type": self.target_metric_type,
            "predictor_type": "xgboost",
            "instance_type": self.instance_type,
            "confidence": self._compute_confidence(features) if self.config.compute_confidence else None,
        }
```

#### Base Predictor Interface Update

```python
# In base_predictor.py

class CARABasePredictor(ABC):
    """Base class for all CARA predictors."""

    @abstractmethod
    async def predict(self, target_request: PredictRequest, state=None) -> Dict:
        """Make prediction.

        Returns:
            Dict with keys:
            - prediction: float - The predicted value
            - target_metric_type: str - What metric is predicted ('e2e_latency', 'ttft', 'tpot')
            - predictor_type: str - Predictor implementation type
            - instance_type: str - Instance type identifier
        """
        pass
```

### Files to Modify

1. `block/predictor/cara/cara_predictor_config.py`
   - Add `target_metric_type` field to all predictor configs

2. `block/predictor/cara/xgboost_predictor.py`
   - Update return format
   - Remove redundant `latency_prediction` field

3. `block/predictor/cara/base_predictor.py`
   - Document standard return format

4. `block/predictor/cara/linear_predictor.py`, `roofline_predictor.py`
   - Update return format consistently

5. `block/predictor/cara/scheduling_policy.py`
   - Update code that reads `latency_prediction` to use `prediction`

### Testing Strategy

1. Unit test: Verify return format matches specification
2. Integration test: Verify scheduling policy correctly uses new field
3. Backward compatibility: Consider deprecation warning for old field access

---

## Task #3: Add Option for Real vs Predicted Decode Length

### Problem Statement

Currently, only `num_predicted_output_tokens` from the request is used for predictions. For evaluation and training refinement, we need the option to use actual output length from callbacks.

### Detailed Design

#### Data Structure Changes

```python
# In data_structures.py

@dataclass
class PredictRequest:
    """Request information for CARA predictor prediction."""
    request_id: str
    num_prompt_tokens: int
    num_predicted_output_tokens: int
    prompt: Optional[str] = None
    rso: Optional['RequestSpecificObjective'] = None

    # NEW: Actual output length (filled after completion)
    actual_output_length: Optional[int] = None

    def get_output_tokens(self, use_actual: bool = False) -> int:
        """Get output token count.

        Args:
            use_actual: If True and actual_output_length is available, use it.
                       Otherwise use predicted.

        Returns:
            Output token count
        """
        if use_actual and self.actual_output_length is not None:
            return self.actual_output_length
        return self.num_predicted_output_tokens


@dataclass
class TrainingExample:
    """Training data point for CARA predictor."""
    # ... existing fields ...

    # NEW: Store both for analysis
    num_predicted_output_tokens: int  # From request
    actual_output_length: Optional[int] = None  # From callback

    # NEW: Flag indicating which was used for prediction
    used_actual_length: bool = False
```

#### Config Changes

```python
# In cara_predictor_config.py

@dataclass
class XGBoostPredictorConfig(CARABasePredictorConfig):
    # ... existing fields ...

    # NEW: Output length source for predictions
    output_length_source: str = field(
        default="predicted",
        metadata={
            "help": "Source for output length: 'predicted' (from request), "
                   "'actual' (from callback, if available), 'both' (log both for analysis)"
        }
    )
```

#### Predictor Changes

```python
# In xgboost_predictor.py

def _build_features(self, state: SimpleScheduleState, request: PredictRequest) -> np.ndarray:
    # ...

    # Use configured output length source
    if self.config.output_length_source == "actual":
        output_tokens = request.get_output_tokens(use_actual=True)
    else:
        output_tokens = request.num_predicted_output_tokens

    # Build features with output_tokens
    # ...
```

#### Training Data Collection Changes

```python
# In training_data_collector.py

async def log_actual_result(
    self,
    request_id: str,
    e2e_latency: float,
    ttft: float = None,
    tpot: float = None,
    actual_output_length: int = None  # NEW parameter
):
    """Log actual metrics for a completed request."""
    if request_id not in self.pending_predictions:
        return

    example = self.pending_predictions[request_id]
    example.actual_e2e_latency = e2e_latency
    example.actual_ttft = ttft
    example.actual_tpot = tpot
    example.actual_output_length = actual_output_length  # NEW
    # ...
```

### Files to Modify

1. `block/predictor/cara/data_structures.py`
   - Add `actual_output_length` to PredictRequest and TrainingExample
   - Add `get_output_tokens()` helper method

2. `block/predictor/cara/cara_predictor_config.py`
   - Add `output_length_source` config option

3. `block/predictor/cara/xgboost_predictor.py`
   - Use configured output length source

4. `block/predictor/cara/training_data_collector.py`
   - Accept and store actual_output_length

5. `block/global_scheduler/cara/cara_serve.py`
   - Pass actual_output_length from response to callbacks

### Testing Strategy

1. Unit test: Verify `get_output_tokens()` returns correct value based on flag
2. Unit test: Verify training data includes both predicted and actual lengths
3. Integration test: Verify predictor uses configured source

---

## Task #4: Rename DummyCARAPredictor to DataCollectionService

### Problem Statement

`DummyCARAPredictor` is misnamed - it performs data collection, not prediction. The name and interface are confusing.

### Detailed Design

#### New Class Structure

```python
# NEW FILE: block/predictor/cara/data_collection_service.py

"""
Data collection service for CARA training data.

Collects (state, request) -> actual_metrics pairs for offline training.
This is NOT a predictor - it records data without making predictions.
"""
import logging
from typing import Dict, Optional

from block.predictor.cara.schedule_trace_client import ScheduleTraceClient
from block.predictor.cara.training_data_collector import TrainingDataCollector
from block.predictor.cara.data_structures import ScheduleState

logger = logging.getLogger(__name__)


class DataCollectionService:
    """Instance-side service for collecting CARA training data.

    Deployed alongside vLLM instances to capture:
    1. Backend state at request arrival time (via /schedule_trace)
    2. Request characteristics (prompt length, predicted output)
    3. Actual metrics after completion (E2E latency, TTFT, TPOT)

    This is NOT a predictor. For baseline scheduling during data collection,
    use StreamingBaselinePolicy (random/round_robin/least_loaded).

    Usage:
        # In instance deployment
        service = DataCollectionService(
            config=DataCollectionConfig(...),
            backend_port=8000,
            hostname="node1.cluster"
        )

        # On request arrival
        await service.record_request(
            request_id="req-123",
            num_prompt_tokens=1024,
            num_predicted_output_tokens=256
        )

        # On request completion (from callback)
        await service.record_completion(
            request_id="req-123",
            e2e_latency=2.5,
            ttft=0.3,
            tpot=0.01,
            actual_output_length=245
        )
    """

    def __init__(
        self,
        config: 'DataCollectionConfig',
        backend_port: int,
        hostname: str = "localhost"
    ):
        """Initialize data collection service.

        Args:
            config: Data collection configuration
            backend_port: vLLM backend port for state fetching
            hostname: Hostname for instance identification
        """
        self.config = config
        self.backend_port = backend_port
        self.hostname = hostname

        # State fetcher
        self.schedule_client = ScheduleTraceClient(
            backend_host=hostname,
            backend_port=backend_port,
            timeout=config.schedule_trace_timeout
        )

        # Data storage
        self.data_collector = TrainingDataCollector(
            output_dir=config.data_output_dir,
            hostname=hostname,
            predictor_port=config.service_port,
            sample_rate=config.sample_rate,
            save_batch_size=config.save_batch_size
        )

        # Derive instance type
        self.instance_type = self._derive_instance_type()

        logger.info(
            f"DataCollectionService initialized: "
            f"hostname={hostname}, backend_port={backend_port}, "
            f"instance_type={self.instance_type}"
        )

    def _derive_instance_type(self) -> str:
        """Derive instance type from hostname and config."""
        # Format: model_backend_nodetype
        # e.g., qwen-2-5-3b_vllm_c240g5
        return f"{self.config.model_name}_{self.config.backend}_{self._extract_node_type()}"

    def _extract_node_type(self) -> str:
        """Extract node type from hostname."""
        import re
        match = re.search(r'([a-zA-Z0-9]+)-', self.hostname)
        return match.group(1) if match else "unknown"

    async def record_request(
        self,
        request_id: str,
        num_prompt_tokens: int,
        num_predicted_output_tokens: int,
        prompt: Optional[str] = None
    ):
        """Record incoming request and current backend state.

        Args:
            request_id: Unique request identifier
            num_prompt_tokens: Number of prompt tokens
            num_predicted_output_tokens: Predicted output tokens
            prompt: Optional prompt text for quality estimation
        """
        # Fetch current backend state
        state = await self.schedule_client.fetch_schedule_trace()

        if state is None:
            logger.warning(f"Failed to fetch state for request {request_id}")
            state = ScheduleState()  # Empty state

        # Log to collector
        await self.data_collector.log_prediction(
            request_id=request_id,
            num_prompt_tokens=num_prompt_tokens,
            num_predicted_output_tokens=num_predicted_output_tokens,
            schedule_state=state,
            instance_type=self.instance_type
        )

    async def record_completion(
        self,
        request_id: str,
        e2e_latency: float,
        ttft: Optional[float] = None,
        tpot: Optional[float] = None,
        actual_output_length: Optional[int] = None
    ):
        """Record actual metrics after request completion.

        Args:
            request_id: Request identifier
            e2e_latency: End-to-end latency in seconds
            ttft: Time to first token in seconds
            tpot: Time per output token in seconds
            actual_output_length: Actual number of output tokens generated
        """
        await self.data_collector.log_actual_result(
            request_id=request_id,
            e2e_latency=e2e_latency,
            ttft=ttft,
            tpot=tpot,
            actual_output_length=actual_output_length
        )

    async def flush(self):
        """Flush pending data to disk."""
        await self.data_collector.flush()

    def get_stats(self) -> Dict:
        """Get collection statistics."""
        return {
            "service_type": "data_collection",
            "instance_type": self.instance_type,
            "hostname": self.hostname,
            **self.data_collector.get_stats()
        }


@dataclass
class DataCollectionConfig:
    """Configuration for DataCollectionService."""

    # Service identification
    service_port: int = 8001
    model_name: str = "unknown"
    backend: str = "vllm"

    # Data collection settings
    data_output_dir: str = "training_data/cara"
    sample_rate: float = 1.0
    save_batch_size: int = 100

    # State fetching
    schedule_trace_timeout: float = 1.0
    state_api: str = "schedule_trace"  # or "instance_stats"
```

#### Backward Compatibility

```python
# In dummy_cara_predictor.py - Add deprecation warning

import warnings

class DummyCARAPredictor(CARABasePredictor):
    """DEPRECATED: Use DataCollectionService instead.

    This class is maintained for backward compatibility only.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "DummyCARAPredictor is deprecated. Use DataCollectionService instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(*args, **kwargs)
```

### Files to Modify

1. **NEW**: `block/predictor/cara/data_collection_service.py`
   - New DataCollectionService class
   - DataCollectionConfig dataclass

2. `block/predictor/cara/dummy_cara_predictor.py`
   - Add deprecation warning
   - Keep for backward compatibility

3. `block/predictor/cara/__init__.py`
   - Export DataCollectionService

4. `block/exp/cara/deploy_cara.py`
   - Update to deploy DataCollectionService

5. Tests: Create `tests/predictor/cara/test_data_collection_service.py`

### Testing Strategy

1. Unit test: Verify record_request captures state correctly
2. Unit test: Verify record_completion matches with pending requests
3. Integration test: Full data collection flow
4. Deprecation test: Verify DummyCARAPredictor shows warning

---

## Task #5: Decouple ServiceRateTracker Raw Rates from EMA

### Problem Statement

The `alpha` parameter in ServiceRateTracker affects EMA computation, which then affects training data. If alpha is set incorrectly during data collection, it's hard to distinguish whether poor model performance is due to:
1. Bad alpha setting (data quality issue)
2. Insufficient features (model issue)

### Detailed Design

#### Solution: Store Both Raw and EMA Rates

```python
# In service_rate_tracker.py

from dataclasses import dataclass, field
from typing import List, Tuple
import time

@dataclass
class RateObservation:
    """Single rate observation from a completed request."""
    timestamp: float
    prefill_rate: Optional[float]  # tokens/sec
    decode_rate: Optional[float]   # tokens/sec
    decode_iter_ms: Optional[float]

    # Request context
    num_prompt_tokens: int
    output_tokens: int
    ttft: float
    e2e_latency: float


@dataclass
class InstanceRateStats:
    """Rate statistics for a single instance."""
    # EMA values (for real-time scheduling)
    ema_decode_tok_per_s: float = 1000.0
    ema_prefill_tok_per_s: float = 2000.0
    ema_decode_iter_ms: float = 10.0

    # Raw observations (for training data)
    recent_observations: List[RateObservation] = field(default_factory=list)
    max_observations: int = 100  # Keep last N observations

    last_update: float = 0.0
    update_count: int = 0

    def add_observation(self, obs: RateObservation):
        """Add new observation, maintaining window size."""
        self.recent_observations.append(obs)
        if len(self.recent_observations) > self.max_observations:
            self.recent_observations.pop(0)

    def get_raw_rates(self, window_s: float = 60.0) -> Tuple[List[float], List[float]]:
        """Get raw rates from recent window.

        Returns:
            Tuple of (prefill_rates, decode_rates) lists
        """
        cutoff = time.time() - window_s
        recent = [o for o in self.recent_observations if o.timestamp >= cutoff]

        prefill_rates = [o.prefill_rate for o in recent if o.prefill_rate is not None]
        decode_rates = [o.decode_rate for o in recent if o.decode_rate is not None]

        return prefill_rates, decode_rates

    def compute_statistics(self, window_s: float = 60.0) -> Dict:
        """Compute statistics over raw rates for training features."""
        prefill_rates, decode_rates = self.get_raw_rates(window_s)

        import numpy as np

        return {
            # EMA values
            "ema_prefill_tok_per_s": self.ema_prefill_tok_per_s,
            "ema_decode_tok_per_s": self.ema_decode_tok_per_s,
            "ema_decode_iter_ms": self.ema_decode_iter_ms,

            # Raw rate statistics (for training)
            "raw_prefill_mean": np.mean(prefill_rates) if prefill_rates else None,
            "raw_prefill_std": np.std(prefill_rates) if len(prefill_rates) > 1 else None,
            "raw_prefill_min": np.min(prefill_rates) if prefill_rates else None,
            "raw_prefill_max": np.max(prefill_rates) if prefill_rates else None,

            "raw_decode_mean": np.mean(decode_rates) if decode_rates else None,
            "raw_decode_std": np.std(decode_rates) if len(decode_rates) > 1 else None,
            "raw_decode_min": np.min(decode_rates) if decode_rates else None,
            "raw_decode_max": np.max(decode_rates) if decode_rates else None,

            "observation_count": len(self.recent_observations),
        }


class ServiceRateTracker:
    """Track service rates with both EMA (for scheduling) and raw (for training)."""

    def __init__(
        self,
        alpha: float = 0.1,
        observation_window_s: float = 60.0,
        max_observations_per_instance: int = 100,
        # ... other params ...
    ):
        self.alpha = alpha
        self.observation_window_s = observation_window_s
        self.max_observations = max_observations_per_instance
        # ...

    def update(self, instance_id: str, request_result: dict):
        """Update rates from completed request."""
        stats = self._stats[instance_id]

        # Extract metrics
        ttft = request_result.get('ttft', 0)
        e2e = request_result.get('e2e_latency', 0)
        prompt_tokens = request_result.get('num_prompt_tokens', 0)
        output_tokens = request_result.get('output_tokens', 0)

        # Compute raw rates
        prefill_rate = prompt_tokens / ttft if ttft > 0 else None
        decode_time = e2e - ttft if e2e > ttft else None
        decode_rate = output_tokens / decode_time if decode_time and decode_time > 0 else None

        # Store raw observation
        obs = RateObservation(
            timestamp=time.time(),
            prefill_rate=prefill_rate,
            decode_rate=decode_rate,
            decode_iter_ms=decode_time * 1000 / output_tokens if decode_time and output_tokens > 0 else None,
            num_prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            ttft=ttft,
            e2e_latency=e2e,
        )
        stats.add_observation(obs)

        # Update EMA (for scheduling)
        if prefill_rate is not None:
            stats.ema_prefill_tok_per_s = (
                self.alpha * prefill_rate +
                (1 - self.alpha) * stats.ema_prefill_tok_per_s
            )

        if decode_rate is not None:
            stats.ema_decode_tok_per_s = (
                self.alpha * decode_rate +
                (1 - self.alpha) * stats.ema_decode_tok_per_s
            )

        stats.update_count += 1
        stats.last_update = time.time()

    def get_training_features(self, instance_id: str) -> Dict:
        """Get both EMA and raw rate features for training data.

        This allows the model to learn appropriate smoothing.
        """
        stats = self._stats.get(instance_id)
        if stats is None:
            return {}
        return stats.compute_statistics(self.observation_window_s)
```

#### Training Data Changes

```python
# In training_data_collector.py

async def log_prediction(self, ..., service_rate_stats: Dict = None):
    """Log prediction with extended rate features."""

    example = TrainingExample(
        # ... existing fields ...
    )

    # Store both EMA and raw rate statistics
    if service_rate_stats:
        example.rate_features = service_rate_stats  # Dict with EMA + raw stats
```

### Files to Modify

1. `block/predictor/cara/service_rate_tracker.py`
   - Add `RateObservation` dataclass
   - Update `InstanceRateStats` with observation storage
   - Add `get_training_features()` method

2. `block/predictor/cara/data_structures.py`
   - Add `rate_features: Optional[Dict]` to TrainingExample

3. `block/predictor/cara/training_data_collector.py`
   - Store extended rate features

4. `block/predictor/cara/offline_training/train_xgboost.py`
   - Option to use raw rate features vs EMA

### Testing Strategy

1. Unit test: Verify observations are stored correctly
2. Unit test: Verify statistics computed correctly
3. Analysis: Compare model performance with EMA vs raw features

---

## Task #6: Update deploy_cara.py for New Architecture

### Problem Statement

`deploy_cara.py` hasn't been updated for the new architecture with:
- DataCollectionService (not predictor)
- Scheduler-side prediction
- New config options (ScoringConfig, SLO)

### Detailed Design

#### New Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Scheduler Node                              │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  cara_serve.py                                            │  │
│  │  - StateManager                                           │  │
│  │  - PredictorRegistry (XGBoost models)                     │  │
│  │  - SchedulingPolicy (CARA or Bootstrap)                   │  │
│  │  - ServiceRateTracker                                     │  │
│  │  - MultiObjectiveScorer                                   │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  Instance 0     │ │  Instance 1     │ │  Instance 2     │
│  ┌───────────┐  │ │  ┌───────────┐  │ │  ┌───────────┐  │
│  │  vLLM     │  │ │  │  vLLM     │  │ │  │  vLLM     │  │
│  │  :8000    │  │ │  │  :8000    │  │ │  │  :8000    │  │
│  └───────────┘  │ │  └───────────┘  │ │  └───────────┘  │
│  ┌───────────┐  │ │  ┌───────────┐  │ │  ┌───────────┐  │
│  │DataCollect│  │ │  │DataCollect│  │ │  │DataCollect│  │
│  │  :8001    │  │ │  │  :8001    │  │ │  │  :8001    │  │
│  └───────────┘  │ │  └───────────┘  │ │  └───────────┘  │
└─────────────────┘ └─────────────────┘ └─────────────────┘
```

#### Updated CLI Interface

```python
# deploy_cara.py - New command line interface

def parse_args():
    parser = argparse.ArgumentParser(description="Deploy CARA system")

    # Deployment mode
    parser.add_argument("--mode", choices=["bootstrap", "cara"], default="bootstrap",
                       help="Deployment mode: bootstrap (data collection) or cara (production)")

    # Model configuration
    parser.add_argument("--model-config", type=str, required=True,
                       help="Path to model_deployment.json")
    parser.add_argument("--model-name", type=str,
                       help="Model to deploy (if not all)")

    # Instance configuration
    parser.add_argument("--host-file", type=str,
                       help="File with host list")
    parser.add_argument("--instance-port", type=int, default=8000,
                       help="vLLM instance port")
    parser.add_argument("--data-collection-port", type=int, default=8001,
                       help="Data collection service port (bootstrap mode)")

    # Scheduler configuration
    parser.add_argument("--scheduler-host", type=str,
                       help="Host for cara_serve scheduler")
    parser.add_argument("--scheduler-port", type=int, default=8080,
                       help="Port for cara_serve")

    # CARA mode configuration
    parser.add_argument("--predictor-model-dir", type=str, default="models/cara/xgboost",
                       help="Directory with trained predictor models")
    parser.add_argument("--scoring-config", type=str,
                       help="Path to scoring configuration JSON")

    # Bootstrap mode configuration
    parser.add_argument("--data-output-dir", type=str, default="training_data/cara",
                       help="Directory for training data output")
    parser.add_argument("--sample-rate", type=float, default=1.0,
                       help="Data collection sample rate")

    # Deployment options
    parser.add_argument("--fresh-deploy", action="store_true",
                       help="Fresh deployment (download models)")
    parser.add_argument("--parallel", action="store_true",
                       help="Deploy instances in parallel")

    return parser.parse_args()


def generate_instance_type(model_name: str, backend: str, hostname: str) -> str:
    """Derive instance type from deployment info.

    Format: {model}_{backend}_{node_type}
    Example: qwen-2-5-3b_vllm_c240g5
    """
    # Normalize model name
    model_normalized = model_name.lower().replace("/", "-").replace(".", "-")

    # Extract node type from hostname
    import re
    match = re.search(r'([a-zA-Z0-9]+)-', hostname)
    node_type = match.group(1) if match else "unknown"

    return f"{model_normalized}_{backend}_{node_type}"


def generate_scheduler_config(args, instance_configs: List[Dict]) -> Dict:
    """Generate scheduler configuration."""

    # Build instance type map
    instance_type_map = {}
    instance_model_map = {}

    for config in instance_configs:
        instance_id = f"{config['host']}:{config['port']}"
        instance_type_map[instance_id] = config['instance_type']
        instance_model_map[instance_id] = config['model_name']

    # Load or create scoring config
    if args.scoring_config:
        with open(args.scoring_config) as f:
            scoring_config = json.load(f)
    else:
        scoring_config = {
            "w_latency": 1.0,
            "w_cost": 0.1,
            "w_quality": 0.0,
            "w_memory": 0.1,
            "normalization": "minmax",
            "constraint_mode": "relaxed"
        }

    return {
        "mode": args.mode,
        "bootstrap_strategy": "round_robin" if args.mode == "bootstrap" else None,
        "scoring": scoring_config,
        "enable_lpt_sort": True,
        "lpt_sort_key": "total_tokens",
        "instance_type_map": instance_type_map,
        "instance_model_map": instance_model_map,
        "predictor_model_dir": args.predictor_model_dir if args.mode == "cara" else None,
    }


def deploy_instance(host: str, config: Dict, args) -> bool:
    """Deploy a single instance (vLLM + DataCollectionService if bootstrap)."""

    commands = []

    # 1. Cleanup existing processes
    commands.extend(get_cleanup_commands(config['backend']))

    # 2. Deploy vLLM
    commands.extend(get_vllm_commands(
        model_path=config['hf_model_name'],
        hf_token=os.environ.get('HF_TOKEN'),
        precision=config.get('precision', 'fp16'),
        vllm_params=config.get('vllm_params', {}),
        fresh_deploy=args.fresh_deploy
    ))

    # 3. Deploy DataCollectionService (bootstrap mode only)
    if args.mode == "bootstrap":
        commands.extend(get_data_collection_commands(
            backend_port=args.instance_port,
            service_port=args.data_collection_port,
            model_name=config['model_name'],
            backend=config['backend'],
            data_output_dir=args.data_output_dir,
            sample_rate=args.sample_rate
        ))

    # Execute
    run_ssh_cmd(host, commands, f"Deploy {config['model_name']} on {host}")
    return True


def get_data_collection_commands(
    backend_port: int,
    service_port: int,
    model_name: str,
    backend: str,
    data_output_dir: str,
    sample_rate: float
) -> List[str]:
    """Generate commands to start DataCollectionService."""

    return [
        f"echo 'Starting DataCollectionService on port {service_port}...'",
        f"cd ~/Block && nohup python -m block.predictor.cara.data_collection_serve "
        f"--port {service_port} "
        f"--backend-port {backend_port} "
        f"--model-name {model_name} "
        f"--backend {backend} "
        f"--data-output-dir {data_output_dir} "
        f"--sample-rate {sample_rate} "
        f"> logs/data_collection.log 2>&1 &",
        "sleep 2",
        f"echo 'DataCollectionService started'"
    ]
```

### Files to Modify

1. `block/exp/cara/deploy_cara.py`
   - Complete rewrite with new architecture
   - Add instance_type derivation
   - Add scheduler config generation

2. **NEW**: `block/predictor/cara/data_collection_serve.py`
   - FastAPI server wrapping DataCollectionService

3. **NEW**: `block/config/cara/scheduler_config.json`
   - Example scheduler configuration

### Testing Strategy

1. Test: Deploy single instance in bootstrap mode
2. Test: Deploy cluster in CARA mode with predictor models
3. Test: Verify instance_type derived correctly

---

## Task #7: Add Predictor Overhead Tracking (Optional)

### Problem Statement

Need to measure the overhead of different predictors as part of system evaluation.

### Detailed Design

```python
# In cara_predictor_config.py

@dataclass
class CARABasePredictorConfig(ABC):
    # ... existing fields ...

    # NEW: Overhead tracking
    track_prediction_overhead: bool = field(
        default=False,
        metadata={"help": "Track prediction timing for overhead analysis"}
    )


# In base_predictor.py

class CARABasePredictor(ABC):

    def __init__(self, config: CARABasePredictorConfig, backend_port: int):
        # ... existing ...

        # Overhead tracking
        self._track_overhead = config.track_prediction_overhead
        self._prediction_times_ms: List[float] = []
        self._max_tracked = 10000  # Keep last N measurements

    async def predict_with_timing(self, target_request: PredictRequest, state=None) -> Dict:
        """Predict with optional timing measurement."""
        if self._track_overhead:
            import time
            start = time.perf_counter()
            result = await self.predict(target_request, state)
            elapsed_ms = (time.perf_counter() - start) * 1000

            self._prediction_times_ms.append(elapsed_ms)
            if len(self._prediction_times_ms) > self._max_tracked:
                self._prediction_times_ms.pop(0)

            result['prediction_overhead_ms'] = elapsed_ms
            return result
        else:
            return await self.predict(target_request, state)

    def get_overhead_stats(self) -> Dict:
        """Get prediction overhead statistics."""
        if not self._prediction_times_ms:
            return {}

        import numpy as np
        times = np.array(self._prediction_times_ms)

        return {
            "prediction_overhead": {
                "count": len(times),
                "mean_ms": float(np.mean(times)),
                "std_ms": float(np.std(times)),
                "p50_ms": float(np.percentile(times, 50)),
                "p95_ms": float(np.percentile(times, 95)),
                "p99_ms": float(np.percentile(times, 99)),
                "min_ms": float(np.min(times)),
                "max_ms": float(np.max(times)),
            }
        }
```

### Files to Modify

1. `block/predictor/cara/cara_predictor_config.py`
   - Add `track_prediction_overhead` field

2. `block/predictor/cara/base_predictor.py`
   - Add timing tracking
   - Add `get_overhead_stats()` method

3. `block/predictor/cara/scheduling_policy.py`
   - Use `predict_with_timing()` when overhead tracking enabled

### Testing Strategy

1. Unit test: Verify timing captured correctly
2. Unit test: Verify stats computed correctly
3. Integration test: Verify overhead included in scheduling stats

---

## Task #8: Add State Sync Time Tracking to StateManager

### Problem Statement

Track time spent fetching instance states for performance analysis.

### Detailed Design

```python
# In state_manager.py

class StateManager:

    def __init__(self, ...):
        # ... existing ...

        # Sync timing
        self._last_sync_time_ms: float = 0.0
        self._sync_times_ms: List[float] = []
        self._per_instance_latencies: Dict[str, float] = {}
        self._max_tracked = 1000

    async def refresh_all(self) -> Dict[str, SimpleScheduleState]:
        """Refresh all instance states with timing."""
        import time
        start = time.perf_counter()

        # Track per-instance latencies
        instance_latencies = {}

        async def fetch_with_timing(instance_id: str) -> Tuple[str, SimpleScheduleState, float]:
            inst_start = time.perf_counter()
            state = await self._fetch_instance_state(instance_id)
            inst_elapsed = (time.perf_counter() - inst_start) * 1000
            return instance_id, state, inst_elapsed

        # Fetch all in parallel
        tasks = [fetch_with_timing(inst_id) for inst_id in self._instance_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"State fetch failed: {result}")
                continue
            instance_id, state, latency_ms = result
            if state is not None:
                self.states[instance_id] = state
                instance_latencies[instance_id] = latency_ms

        # Record timing
        total_elapsed = (time.perf_counter() - start) * 1000
        self._last_sync_time_ms = total_elapsed
        self._per_instance_latencies = instance_latencies

        self._sync_times_ms.append(total_elapsed)
        if len(self._sync_times_ms) > self._max_tracked:
            self._sync_times_ms.pop(0)

        # Warn if slow
        if total_elapsed > 100:  # > 100ms
            logger.warning(f"Slow state refresh: {total_elapsed:.1f}ms")

        return self.states

    def get_sync_stats(self) -> Dict:
        """Get state synchronization statistics."""
        if not self._sync_times_ms:
            return {}

        import numpy as np
        times = np.array(self._sync_times_ms)

        return {
            "state_sync": {
                "last_sync_ms": self._last_sync_time_ms,
                "count": len(times),
                "mean_ms": float(np.mean(times)),
                "p50_ms": float(np.percentile(times, 50)),
                "p95_ms": float(np.percentile(times, 95)),
                "max_ms": float(np.max(times)),
                "per_instance_latencies": self._per_instance_latencies,
            }
        }

    def get_stats(self) -> Dict:
        """Get manager statistics including sync timing."""
        return {
            # ... existing stats ...
            **self.get_sync_stats()
        }
```

### Files to Modify

1. `block/predictor/cara/state_manager.py`
   - Add sync timing tracking
   - Add `get_sync_stats()` method

### Testing Strategy

1. Unit test: Verify timing captured
2. Unit test: Verify stats computed correctly
3. Integration test: Verify included in cara_serve stats

---

## Task #9: Create E2E Training Pipeline Shell Script

### Problem Statement

Need automated pipeline for training data collection, processing, and model training.

### Detailed Design

```bash
#!/bin/bash
# scripts/cara_training_pipeline.sh
#
# End-to-end CARA training pipeline:
# 1. Deploy instances with data collection
# 2. Run benchmark with bootstrap mode
# 3. Collect and process training data
# 4. Train models for each instance type
# 5. Evaluate and generate reports

set -e

# =============================================================================
# Configuration
# =============================================================================

MODEL_CONFIG="${MODEL_CONFIG:-block/config/cara/model_deployment.json}"
HOST_FILE="${HOST_FILE:-hosts.txt}"
DATA_DIR="${DATA_DIR:-training_data/cara}"
MODEL_DIR="${MODEL_DIR:-models/cara/xgboost}"
REPORT_DIR="${REPORT_DIR:-reports/cara}"

# Training parameters
NUM_TRAINING_SAMPLES="${NUM_TRAINING_SAMPLES:-10000}"
QPS="${QPS:-4.0}"
TRACE_FILE="${TRACE_FILE:-data/traces/default_trace.json}"
SAMPLE_RATE="${SAMPLE_RATE:-1.0}"

# Evaluation split
TRAIN_RATIO="${TRAIN_RATIO:-0.8}"
VAL_RATIO="${VAL_RATIO:-0.1}"
TEST_RATIO="${TEST_RATIO:-0.1}"

# =============================================================================
# Functions
# =============================================================================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

check_prerequisites() {
    log "Checking prerequisites..."

    # Check config files exist
    if [[ ! -f "$MODEL_CONFIG" ]]; then
        echo "ERROR: Model config not found: $MODEL_CONFIG"
        exit 1
    fi

    if [[ ! -f "$HOST_FILE" ]]; then
        echo "ERROR: Host file not found: $HOST_FILE"
        exit 1
    fi

    # Check Python packages
    python -c "import xgboost" || { echo "ERROR: xgboost not installed"; exit 1; }
    python -c "import numpy" || { echo "ERROR: numpy not installed"; exit 1; }

    log "Prerequisites OK"
}

deploy_instances() {
    log "Deploying instances in bootstrap mode..."

    python -m block.exp.cara.deploy_cara \
        --mode bootstrap \
        --model-config "$MODEL_CONFIG" \
        --host-file "$HOST_FILE" \
        --data-output-dir "$DATA_DIR" \
        --sample-rate "$SAMPLE_RATE" \
        --parallel

    log "Waiting for instances to be ready..."
    sleep 60

    log "Instances deployed"
}

run_data_collection() {
    log "Starting data collection benchmark..."

    # Start cara_serve in bootstrap mode
    python -m block.global_scheduler.cara.cara_serve \
        --scheduling-mode bootstrap \
        --bootstrap-strategy round_robin \
        --model-config "$MODEL_CONFIG" \
        --host-file "$HOST_FILE" \
        --port 8080 &
    SCHEDULER_PID=$!

    sleep 10  # Wait for scheduler to start

    # Run benchmark
    python -m block.benchmark_serving \
        --endpoint http://localhost:8080/v1/batch_completions \
        --mode bootstrap \
        --trace-file "$TRACE_FILE" \
        --qps "$QPS" \
        --num-samples "$NUM_TRAINING_SAMPLES" \
        --use-dummy-prompts

    # Stop scheduler
    kill $SCHEDULER_PID 2>/dev/null || true

    log "Data collection complete"
}

collect_training_data() {
    log "Collecting training data from instances..."

    mkdir -p "$DATA_DIR/collected"

    # Copy data from each instance
    while read -r host; do
        [[ -z "$host" ]] && continue
        log "Collecting from $host..."
        scp -r "$host:~/Block/$DATA_DIR/*.jsonl" "$DATA_DIR/collected/" || true
    done < "$HOST_FILE"

    # Merge all data
    cat "$DATA_DIR/collected"/*.jsonl > "$DATA_DIR/all_training_data.jsonl"

    log "Collected $(wc -l < "$DATA_DIR/all_training_data.jsonl") training examples"
}

split_dataset() {
    log "Splitting dataset..."

    python -c "
import json
import random
from pathlib import Path

# Load all data
data = []
with open('$DATA_DIR/all_training_data.jsonl') as f:
    for line in f:
        data.append(json.loads(line))

# Shuffle
random.seed(42)
random.shuffle(data)

# Split
n = len(data)
train_end = int(n * $TRAIN_RATIO)
val_end = train_end + int(n * $VAL_RATIO)

train_data = data[:train_end]
val_data = data[train_end:val_end]
test_data = data[val_end:]

# Save
for split_name, split_data in [('train', train_data), ('val', val_data), ('test', test_data)]:
    path = Path('$DATA_DIR') / f'{split_name}.jsonl'
    with open(path, 'w') as f:
        for item in split_data:
            f.write(json.dumps(item) + '\n')
    print(f'{split_name}: {len(split_data)} examples')
"

    log "Dataset split complete"
}

train_models() {
    log "Training models..."

    mkdir -p "$MODEL_DIR"

    # Train XGBoost models for each instance type
    python -m block.predictor.cara.offline_training.train_xgboost \
        --input "$DATA_DIR/train.jsonl" \
        --val-input "$DATA_DIR/val.jsonl" \
        --output-dir "$MODEL_DIR" \
        --early-stopping 50 \
        --save-learning-curve

    log "Models trained"
}

evaluate_models() {
    log "Evaluating models..."

    mkdir -p "$REPORT_DIR"

    # Evaluate on test set
    python -m block.predictor.cara.offline_training.evaluate_predictor \
        --test-data "$DATA_DIR/test.jsonl" \
        --model-dir "$MODEL_DIR" \
        --output-dir "$REPORT_DIR" \
        --generate-plots

    log "Evaluation complete. Reports in $REPORT_DIR"
}

cleanup_instances() {
    log "Cleaning up instances..."

    python -m block.exp.cara.deploy_cara \
        --mode cleanup \
        --host-file "$HOST_FILE"

    log "Cleanup complete"
}

# =============================================================================
# Main
# =============================================================================

main() {
    log "Starting CARA training pipeline"

    check_prerequisites

    case "${1:-all}" in
        deploy)
            deploy_instances
            ;;
        collect)
            run_data_collection
            collect_training_data
            ;;
        train)
            split_dataset
            train_models
            ;;
        evaluate)
            evaluate_models
            ;;
        cleanup)
            cleanup_instances
            ;;
        all)
            deploy_instances
            run_data_collection
            collect_training_data
            split_dataset
            train_models
            evaluate_models
            cleanup_instances
            ;;
        *)
            echo "Usage: $0 {deploy|collect|train|evaluate|cleanup|all}"
            exit 1
            ;;
    esac

    log "Pipeline complete"
}

main "$@"
```

### Files to Create

1. **NEW**: `scripts/cara_training_pipeline.sh`
   - Main pipeline script

2. **NEW**: `scripts/cara_config.sh`
   - Configuration template

### Testing Strategy

1. Test each stage independently
2. Test full pipeline on small scale
3. Document expected runtime

---

## Task #10: Add CDF Error Distribution Plots to Evaluation

### Problem Statement

Need CDF plots of prediction errors for evaluation analysis.

### Detailed Design

```python
# In offline_training/evaluate_predictor.py

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_error_cdf(
    errors: np.ndarray,
    output_path: Path,
    title: str = "Prediction Error CDF",
    xlabel: str = "Absolute Error (ms)"
):
    """Plot CDF of prediction errors."""
    sorted_errors = np.sort(np.abs(errors))
    cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(sorted_errors, cdf, linewidth=2)

    # Add percentile markers
    percentiles = [50, 90, 95, 99]
    for p in percentiles:
        val = np.percentile(sorted_errors, p)
        ax.axhline(y=p/100, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=val, color='gray', linestyle='--', alpha=0.5)
        ax.annotate(f'P{p}: {val:.1f}', xy=(val, p/100),
                   xytext=(val + sorted_errors.max()*0.05, p/100))

    ax.set_xlabel(xlabel)
    ax.set_ylabel("CDF")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_relative_error_cdf(
    predictions: np.ndarray,
    actuals: np.ndarray,
    output_path: Path,
    title: str = "Relative Error CDF"
):
    """Plot CDF of relative prediction errors."""
    relative_errors = np.abs(predictions - actuals) / np.maximum(actuals, 1e-6) * 100

    sorted_errors = np.sort(relative_errors)
    cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(sorted_errors, cdf, linewidth=2)

    ax.set_xlabel("Relative Error (%)")
    ax.set_ylabel("CDF")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, min(sorted_errors.max(), 200))  # Cap at 200%
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_prediction_scatter(
    predictions: np.ndarray,
    actuals: np.ndarray,
    output_path: Path,
    title: str = "Predicted vs Actual"
):
    """Plot scatter of predicted vs actual values."""
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(actuals, predictions, alpha=0.3, s=10)

    # Perfect prediction line
    max_val = max(actuals.max(), predictions.max())
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect')

    ax.set_xlabel("Actual (ms)")
    ax.set_ylabel("Predicted (ms)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_error_histogram(
    errors: np.ndarray,
    output_path: Path,
    title: str = "Error Distribution"
):
    """Plot histogram of prediction errors."""
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.hist(errors, bins=50, edgecolor='black', alpha=0.7)

    # Add mean and std annotations
    mean_err = np.mean(errors)
    std_err = np.std(errors)
    ax.axvline(x=mean_err, color='red', linestyle='--', label=f'Mean: {mean_err:.1f}')
    ax.axvline(x=mean_err - std_err, color='orange', linestyle=':', alpha=0.7)
    ax.axvline(x=mean_err + std_err, color='orange', linestyle=':', alpha=0.7)

    ax.set_xlabel("Error (Predicted - Actual) (ms)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def generate_evaluation_plots(
    predictions: np.ndarray,
    actuals: np.ndarray,
    output_dir: Path,
    instance_type: str,
    target_metric: str = "e2e_latency"
):
    """Generate all evaluation plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    errors = predictions - actuals
    abs_errors = np.abs(errors)

    # 1. Absolute error CDF
    plot_error_cdf(
        errors=abs_errors,
        output_path=output_dir / f"{instance_type}_error_cdf.png",
        title=f"Absolute Error CDF - {instance_type}",
        xlabel=f"Absolute Error ({target_metric}) (ms)"
    )

    # 2. Relative error CDF
    plot_relative_error_cdf(
        predictions=predictions,
        actuals=actuals,
        output_path=output_dir / f"{instance_type}_relative_error_cdf.png",
        title=f"Relative Error CDF - {instance_type}"
    )

    # 3. Scatter plot
    plot_prediction_scatter(
        predictions=predictions,
        actuals=actuals,
        output_path=output_dir / f"{instance_type}_scatter.png",
        title=f"Predicted vs Actual - {instance_type}"
    )

    # 4. Error histogram
    plot_error_histogram(
        errors=errors,
        output_path=output_dir / f"{instance_type}_error_hist.png",
        title=f"Error Distribution - {instance_type}"
    )

    print(f"Plots saved to {output_dir}")
```

### Files to Modify

1. `block/predictor/cara/offline_training/evaluate_predictor.py`
   - Add plotting functions
   - Add `--generate-plots` flag
   - Add `--output-dir` for plots

### Testing Strategy

1. Unit test: Verify plots generated correctly
2. Visual inspection: Check plot quality

---

## Task #11: Add Training Learning Curve Reporting

### Problem Statement

Need learning curves to identify overfitting/underfitting during training.

### Detailed Design

```python
# In offline_training/train_xgboost.py

def train_with_learning_curve(
    dtrain: xgb.DMatrix,
    dval: xgb.DMatrix,
    params: Dict,
    output_dir: Path,
    instance_type: str,
    num_boost_round: int = 1000,
    early_stopping_rounds: int = 50
) -> Tuple[xgb.Booster, Dict]:
    """Train XGBoost with learning curve tracking."""

    evals_result = {}

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=early_stopping_rounds,
        evals_result=evals_result,
        verbose_eval=100
    )

    # Plot learning curve
    plot_learning_curve(evals_result, output_dir, instance_type)

    # Save learning curve data
    curve_path = output_dir / f"{instance_type}_learning_curve.json"
    with open(curve_path, 'w') as f:
        json.dump(evals_result, f, indent=2)

    return model, evals_result


def plot_learning_curve(
    evals_result: Dict,
    output_dir: Path,
    instance_type: str
):
    """Plot training and validation loss curves."""
    import matplotlib.pyplot as plt

    train_loss = evals_result['train']['rmse']
    val_loss = evals_result['val']['rmse']
    iterations = range(1, len(train_loss) + 1)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(iterations, train_loss, label='Training Loss', linewidth=2)
    ax.plot(iterations, val_loss, label='Validation Loss', linewidth=2)

    # Mark best iteration
    best_iter = np.argmin(val_loss)
    ax.axvline(x=best_iter + 1, color='green', linestyle='--',
               label=f'Best iteration: {best_iter + 1}')

    ax.set_xlabel('Boosting Iterations')
    ax.set_ylabel('RMSE')
    ax.set_title(f'Learning Curve - {instance_type}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Log scale if range is large
    if max(train_loss) / min(train_loss) > 10:
        ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(output_dir / f"{instance_type}_learning_curve.png", dpi=150)
    plt.close()

    # Also save numeric summary
    summary = {
        "best_iteration": int(best_iter + 1),
        "best_val_rmse": float(val_loss[best_iter]),
        "final_train_rmse": float(train_loss[-1]),
        "final_val_rmse": float(val_loss[-1]),
        "total_iterations": len(train_loss),
        "overfitting_ratio": float(val_loss[-1] / train_loss[-1]) if train_loss[-1] > 0 else None
    }

    return summary
```

### Files to Modify

1. `block/predictor/cara/offline_training/train_xgboost.py`
   - Add `train_with_learning_curve()`
   - Add `plot_learning_curve()`
   - Add `--save-learning-curve` flag

### Testing Strategy

1. Unit test: Verify learning curve data captured
2. Visual inspection: Check curve plots

---

## Task #12: Update benchmark_serving for Bootstrap Mode with Dummy Prompts

### Problem Statement

Need benchmark tool support for bootstrap mode data collection with:
- Length-only trace input (no real prompts)
- Dummy prompt generation
- Sample count stopping condition

### Detailed Design

```python
# In benchmark_serving.py or new file

class BootstrapBenchmarkRunner:
    """Benchmark runner for CARA bootstrap data collection.

    Generates dummy requests based on trace length distributions.
    """

    def __init__(
        self,
        endpoint: str,
        trace_file: str,
        qps: float,
        num_samples: int,
        use_dummy_prompts: bool = True,
        dummy_token: str = "a"
    ):
        self.endpoint = endpoint
        self.qps = qps
        self.num_samples = num_samples
        self.use_dummy_prompts = use_dummy_prompts
        self.dummy_token = dummy_token

        # Load trace
        self.trace_data = self._load_trace(trace_file)
        self.trace_index = 0

        # Statistics
        self.samples_collected = 0
        self.requests_sent = 0

    def _load_trace(self, trace_file: str) -> List[Dict]:
        """Load trace file with length distributions."""
        with open(trace_file) as f:
            data = json.load(f)

        # Shuffle for randomization
        import random
        random.shuffle(data)

        return data

    def _get_next_request(self) -> Dict:
        """Get next request from trace (with shuffling)."""
        if self.trace_index >= len(self.trace_data):
            # Reshuffle and restart
            import random
            random.shuffle(self.trace_data)
            self.trace_index = 0

        trace_entry = self.trace_data[self.trace_index]
        self.trace_index += 1

        prompt_len = trace_entry.get('prompt_len', trace_entry.get('num_prompt_tokens', 100))
        max_tokens = trace_entry.get('max_tokens', trace_entry.get('num_output_tokens', 256))

        if self.use_dummy_prompts:
            # Generate dummy prompt of correct length
            prompt = self._generate_dummy_prompt(prompt_len)
        else:
            prompt = trace_entry.get('prompt', '')

        return {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "prompt_len": prompt_len,
            "request_id": f"bootstrap-{self.requests_sent}"
        }

    def _generate_dummy_prompt(self, num_tokens: int) -> str:
        """Generate dummy prompt with approximately num_tokens tokens.

        Uses simple heuristic: ~4 chars per token for English text.
        """
        chars_needed = num_tokens * 4
        return (self.dummy_token + " ") * (chars_needed // 2)

    async def run(self):
        """Run bootstrap benchmark until num_samples collected."""
        import asyncio
        import aiohttp

        interval = 1.0 / self.qps

        async with aiohttp.ClientSession() as session:
            while self.samples_collected < self.num_samples:
                # Send request
                request = self._get_next_request()
                self.requests_sent += 1

                try:
                    async with session.post(
                        f"{self.endpoint}",
                        json={"requests": [request], "scheduling_mode": "bootstrap"}
                    ) as response:
                        result = await response.json()

                        # Count samples from response
                        if result.get('responses'):
                            self.samples_collected += len(result['responses'])

                except Exception as e:
                    logger.warning(f"Request failed: {e}")

                # Progress
                if self.requests_sent % 100 == 0:
                    print(f"Progress: {self.samples_collected}/{self.num_samples} samples "
                          f"({self.requests_sent} requests)")

                # Rate limiting
                await asyncio.sleep(interval)

        print(f"Bootstrap complete: {self.samples_collected} samples from {self.requests_sent} requests")


# CLI interface
def main():
    parser = argparse.ArgumentParser(description="CARA Bootstrap Benchmark")

    parser.add_argument("--endpoint", required=True,
                       help="CARA scheduler endpoint URL")
    parser.add_argument("--trace-file", required=True,
                       help="Trace file with length distributions")
    parser.add_argument("--qps", type=float, default=4.0,
                       help="Queries per second")
    parser.add_argument("--num-samples", type=int, default=10000,
                       help="Number of training samples to collect")
    parser.add_argument("--use-dummy-prompts", action="store_true", default=True,
                       help="Generate dummy prompts from lengths")

    args = parser.parse_args()

    runner = BootstrapBenchmarkRunner(
        endpoint=args.endpoint,
        trace_file=args.trace_file,
        qps=args.qps,
        num_samples=args.num_samples,
        use_dummy_prompts=args.use_dummy_prompts
    )

    asyncio.run(runner.run())


if __name__ == "__main__":
    main()
```

### Files to Create/Modify

1. **NEW**: `block/benchmark/bootstrap_benchmark.py`
   - BootstrapBenchmarkRunner class
   - CLI interface

2. Alternatively, add to existing `block/benchmark_serving.py`
   - Add `--mode bootstrap` flag
   - Add bootstrap-specific options

### Testing Strategy

1. Unit test: Verify dummy prompt generation
2. Unit test: Verify trace shuffling and cycling
3. Integration test: Run against mock endpoint

---

## Dependencies Between Tasks

```
Task #4 (DataCollectionService) ──┐
                                  ├──> Task #6 (deploy_cara.py)
Task #5 (ServiceRateTracker)     ──┘

Task #6 (deploy_cara.py) ──────────> Task #9 (E2E Pipeline)

Task #2 (XGBoost target field) ────> Task #10 (Evaluation plots)
                                     Task #11 (Learning curves)

Task #12 (Bootstrap benchmark) ────> Task #9 (E2E Pipeline)
```

## Implementation Order Recommendation

1. **Phase 1 (Core fixes)**:
   - Task #4: DataCollectionService (foundation)
   - Task #2: XGBoost target field (cleanup)
   - Task #5: ServiceRateTracker decouple (data quality)

2. **Phase 2 (Deployment)**:
   - Task #6: deploy_cara.py
   - Task #12: Bootstrap benchmark

3. **Phase 3 (Evaluation)**:
   - Task #10: CDF plots
   - Task #11: Learning curves
   - Task #9: E2E pipeline

4. **Phase 4 (API & Metrics)**:
   - Task #1: Streaming batch response
   - Task #3: Real vs predicted length
   - Task #7: Predictor overhead
   - Task #8: State sync time

---

## References

- `cara_paper/claude/LATENCY_PREDICTOR_DESIGN.md`
- `cara_paper/codex/cara-objective-design.md`
- `cara_paper/claude/review_comments_for_implementation_predictor_v1.md`
- `cara_paper/claude/IMPLEMENTATION_FIX_PLAN.md`
