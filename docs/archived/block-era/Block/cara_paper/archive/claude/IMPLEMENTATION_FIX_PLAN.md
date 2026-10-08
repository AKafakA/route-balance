# CARA Predictor Implementation Fix Plan

**Date**: January 26, 2026
**Status**: Superseded by IMPLEMENTATION_TASKS_V2_DESIGN.md
**Reference**: `review_comments_for_implementation_predictor_v1.md`

---

> **⚠️ Note**: This document has been **superseded** by [IMPLEMENTATION_TASKS_V2_DESIGN.md](IMPLEMENTATION_TASKS_V2_DESIGN.md), which provides more comprehensive and detailed designs for all 12 pending tasks identified from review comments. Please refer to that document for the authoritative implementation plan.

---

## Executive Summary

The current implementation has **significant deviations** from the agreed design documents. This plan identifies all inconsistencies and proposes specific fixes prioritized by impact.

---

## Critical Issues (Must Fix)

### Issue 1: batch_completions Endpoint - Wrong Execution Model

**Files**: `block/global_scheduler/cara/cara_serve.py:77-257`

**Current Behavior**:
- Waits for ALL requests in batch to complete before returning ANY response
- Lines 198-230: `await future` for each request sequentially
- Introduces significant delay for faster requests

**Design Requirement** (from `LATENCY_PREDICTOR_DESIGN.md` Section 13):
- Individual requests → Global queue → Batch scheduling with dynamic batch size
- Return response IMMEDIATELY when each request completes (no waiting for slowest)
- Priority-Based Streaming Scheduler (from `new_greedy_cara.tex`)

**Proposed Fix**:
```python
# Current (WRONG):
for request_id, inst_id, prompt_tokens, future in response_futures:
    response = await future  # Waits for each one sequentially
    responses.append(response)

# Proposed (streaming dispatch with immediate return):
class StreamingBatchScheduler:
    """Global request queue with streaming batch dispatch."""

    def __init__(self, instances, scheduling_policy):
        self.global_queue = asyncio.Queue()
        self.pending_responses = {}  # request_id -> Future

    async def enqueue_request(self, request) -> asyncio.Future:
        """Add request to global queue, return future for response."""
        response_future = asyncio.Future()
        self.pending_responses[request.request_id] = response_future
        await self.global_queue.put(request)
        return response_future

    async def batch_scheduler_loop(self):
        """Background loop: collect batch → schedule → dispatch."""
        while True:
            batch = await self._collect_batch()  # Dynamic batch size
            assignments = await self._schedule_batch(batch)

            # Dispatch all requests (fire-and-forget with callback)
            for inst_id, requests in assignments.items():
                for req in requests:
                    asyncio.create_task(self._dispatch_and_notify(inst_id, req))

    async def _dispatch_and_notify(self, inst_id, request):
        """Execute request and resolve its future immediately."""
        response = await instance_map[inst_id].query_instance(request)
        self.pending_responses[request.request_id].set_result(response)
```

**Endpoint should support both modes**:
- `wait_all=true`: Current behavior for compatibility
- `wait_all=false` (default): Return immediately as responses arrive

---

### Issue 2: Bootstrap Mode - Should Be Streaming, Not Batched

**Files**: `block/predictor/cara/scheduling_policy.py:163-236`

**Current Behavior**:
- Bootstrap mode uses `route_batch()` same as CARA mode
- All strategies (random, round_robin, least_loaded) process requests in batch

**Design Requirement** (from review comments):
> "rr/random and least_load scheduling besides act as the bootstrap for training data of latency predictor, it will also act as the baseline over cara during evaluation, and always suppose to be dispatched in streaming way (one by one) for fairness per-batch is not their requirement"

**Proposed Fix**:
```python
class SchedulingPolicy:
    """Base interface for scheduling policies."""

    @abstractmethod
    async def route_request(self, request: PredictRequest, instance_ids: List[str]) -> str:
        """Route a SINGLE request - streaming dispatch."""
        pass

    @abstractmethod
    async def route_batch(self, requests: List[PredictRequest], instance_ids: List[str]) -> Dict[str, List[PredictRequest]]:
        """Route a batch - only for CARA mode."""
        pass


class StreamingBaselinePolicy(SchedulingPolicy):
    """Baseline policies: random, round_robin, least_loaded - streaming dispatch."""

    async def route_request(self, request, instance_ids):
        """Route single request immediately (streaming)."""
        if self.strategy == "random":
            return random.choice(instance_ids)
        elif self.strategy == "round_robin":
            target = instance_ids[self._rr_index % len(instance_ids)]
            self._rr_index += 1
            return target
        elif self.strategy == "least_loaded":
            return self._find_least_loaded(instance_ids)

    async def route_batch(self, requests, instance_ids):
        # For baseline policies, route one-by-one (streaming)
        assignments = {inst_id: [] for inst_id in instance_ids}
        for req in requests:
            target = await self.route_request(req, instance_ids)
            assignments[target].append(req)
        return assignments


class CARABatchPolicy(SchedulingPolicy):
    """CARA policy: batch scheduling with LPT + multi-objective."""

    async def route_batch(self, requests, instance_ids):
        # Step 1: LPT sort (configurable)
        if self.config.enable_lpt_sort:
            requests = sorted(requests,
                              key=lambda r: r.num_prompt_tokens + r.num_predicted_output_tokens,
                              reverse=True)  # Descending (big first)

        # Step 2: Refresh states
        states = await self.state_manager.refresh_all()

        # Step 3: Greedy assignment with multi-objective scoring
        assignments = {inst_id: [] for inst_id in instance_ids}
        for request in requests:
            best_instance = self._select_best_instance(request, states)
            assignments[best_instance].append(request)
            self.state_manager.apply_assignment(best_instance, request)

        return assignments
```

---

### Issue 3: Remove Automatic BOOTSTRAP → CARA Conversion

**Files**: `block/predictor/cara/scheduling_policy.py:146-161`

**Current Behavior**:
```python
def record_training_sample(self):
    self._training_samples += 1
    if (self._mode == SchedulingMode.BOOTSTRAP and
        self._training_samples >= self.min_training_samples and ...):
        self._mode = SchedulingMode.CARA  # Auto-convert
```

**Design Requirement** (from review comments):
> "the automatical conversation between bootstrapping and cara can not be required"

The modes are separate:
- Bootstrap: For data collection AND as baseline for evaluation
- CARA: For production with trained predictors
- Switching should be explicit, not automatic

**Proposed Fix**:
- Remove the automatic conversion in `record_training_sample()`
- Keep tracking samples for metrics/logging only
- Mode switching only via explicit `set_mode()` or config

---

### Issue 4: Missing Multi-Objective Scoring

**Files**: `block/predictor/cara/scheduling_policy.py:239-303`

**Current Behavior**:
- `_route_cara()` only considers `latency_prediction` (single objective)
- No quality, cost, or memory balance scoring

**Design Requirement** (from `cara-objective-design.md`):
```
score = α·lat_term + β·cost_term + γ·qual_term
```

With:
- Latency (P̂_ri): predicted server-side E2E
- Quality (Q̂_r,model(i)): best-route quality score
- Cost (Ĉ_ri): token-based proxy
- Memory balance (kv_cache_utilization)

**Proposed Fix**:
```python
@dataclass
class ScoringConfig:
    """Multi-objective scoring weights."""
    w_latency: float = 1.0      # α
    w_cost: float = 0.1         # β
    w_quality: float = 0.5      # γ
    w_memory_balance: float = 0.1

    normalization: str = "minmax"  # "minmax", "percentile", "regret"
    enable_constraints: bool = True
    k_sigma: float = 1.0  # Uncertainty margin


class MultiObjectiveScorer:
    """Score instances for a request using multiple objectives."""

    def score(self, request, instance_id, state, config: ScoringConfig) -> float:
        # Get raw predictions
        latency = self.predictor.predict(state, request)
        cost = self._compute_cost(request, instance_id)
        quality = self._get_quality_estimate(request, instance_id)  # Optional
        mem_balance = state.kv_cache_utilization

        # Normalize (per-request min-max by default)
        lat_norm = self._normalize(latency, self.lat_min, self.lat_max)
        cost_norm = self._normalize(cost, self.cost_min, self.cost_max)
        qual_norm = 1.0 - quality  # Higher quality = lower score (better)

        # Compute score (lower is better)
        score = (
            config.w_latency * lat_norm +
            config.w_cost * cost_norm +
            config.w_quality * qual_norm +
            config.w_memory_balance * mem_balance
        )

        return score
```

---

### Issue 5: Missing LPT Sorting

**Files**: `block/predictor/cara/scheduling_policy.py:239-303`

**Current Behavior**:
- Requests processed in arrival order (FCFS)
- No sorting before greedy assignment

**Design Requirement** (from `LATENCY_PREDICTOR_DESIGN.md` Section 13.3):
> "Graham (1969) showed LPT is a 4/3-approximation for makespan minimization"
> "Placing large jobs first leaves flexibility to fill gaps with smaller jobs"

**Proposed Fix**:
```python
async def _route_cara(self, requests, instance_ids):
    # LPT Sort: Big Rocks First
    if self.config.enable_lpt_sort:
        requests = sorted(requests,
                          key=lambda r: r.num_prompt_tokens + r.num_predicted_output_tokens,
                          reverse=True)  # Descending

    # ... rest of greedy assignment
```

Config option for ablation:
```json
{
    "enable_lpt_sort": true,  // Can disable for ablation study
    "sort_key": "total_tokens"  // or "prompt_only", "output_only"
}
```

---

## Medium Priority Issues

### Issue 6: XGBoost Predictor - Target Field Naming

**Files**: `block/predictor/cara/xgboost_predictor.py:171-176`

**Current Behavior**:
```python
return {
    "target_metric": float(prediction),
    "latency_prediction": float(prediction),
    ...
}
```

**Design Requirement** (from review comments):
> "this can be wrong if the predicted metrics is ttft instead of e2e latency, we should have the target_feature only and also have target_feature_type as part of shared predictor config fields"

**Proposed Fix**:
```python
@dataclass
class XGBoostPredictorConfig:
    target_feature_type: str = "e2e_latency"  # or "ttft", "tpot"
    ...

class XGBoostLatencyPredictor:
    async def predict(self, target_request, state=None) -> Dict:
        prediction = self._predict_internal(state, target_request)
        return {
            "prediction": float(prediction),
            "target_feature_type": self.config.target_feature_type,
            "predictor_type": "xgboost",
            "instance_type": self.instance_type
        }
```

---

### Issue 7: Support Real vs Predicted Decode Length

**Files**: `block/predictor/cara/xgboost_predictor.py`

**Current Behavior**:
- Only uses `num_predicted_output_tokens` from request

**Design Requirement** (from review comments):
> "we should also have the option to use either 1) real decode length logged in the call back or 2) predicted length in the request for the latency predictions"

**Proposed Fix**:
Add config option and support for both:
```python
@dataclass
class PredictRequest:
    num_predicted_output_tokens: int  # From request
    actual_output_length: Optional[int] = None  # From callback (if available)

    def get_output_tokens(self, use_actual: bool = False) -> int:
        if use_actual and self.actual_output_length is not None:
            return self.actual_output_length
        return self.num_predicted_output_tokens
```

---

### Issue 8: DummyCARAPredictor Rename/Refactor

**Files**: `block/predictor/cara/dummy_cara_predictor.py`

**Current Behavior**:
- Named as "predictor" but actually does data collection
- Uses predict interface but doesn't make real predictions

**Design Requirement** (from review comments):
> "the predictor instance and the data collection serve is decoupled, and this class is used for the data collection purpose but still using the predict interface and api which is incorrect"

**Proposed Fix**:
Rename to `DataCollectionService`:
```python
# block/predictor/cara/data_collection_service.py

class DataCollectionService:
    """Instance-side service for collecting training data.

    NOT a predictor - just records (state, request) → actual metrics pairs.
    """

    async def record_request(self, request_id: str, num_prompt_tokens: int,
                             num_predicted_output_tokens: int):
        """Record incoming request and current backend state."""
        state = await self.schedule_client.fetch_schedule_trace()
        await self.data_collector.log_prediction(
            request_id=request_id,
            num_prompt_tokens=num_prompt_tokens,
            num_predicted_output_tokens=num_predicted_output_tokens,
            schedule_state=state
        )

    async def record_completion(self, request_id: str, e2e_latency: float,
                                ttft: float, tpot: float, actual_output_length: int):
        """Record actual metrics after completion."""
        await self.data_collector.log_actual_result(
            request_id=request_id,
            e2e_latency=e2e_latency,
            ttft=ttft,
            tpot=tpot,
            actual_output_length=actual_output_length
        )
```

---

### Issue 9: ServiceRateTracker Alpha Concern

**Files**: `block/predictor/cara/service_rate_tracker.py`

**Current Behavior**:
- Alpha is a hardcoded config parameter
- Affects EMA computation which affects training data

**Design Requirement** (from review comments):
> "self.alpha not convince me, it means we need to manually set it before the data collection and then could affect the ground truth of training data? That is hard to debugging"

**Proposed Fix**:
Option 1: Decouple by recording raw rates, let model learn:
```python
class ServiceRateTracker:
    def update(self, instance_id: str, request_result: dict):
        # Store raw rates, not EMA
        self._raw_rates[instance_id].append({
            'timestamp': time.time(),
            'prefill_rate': prompt_tokens / ttft if ttft > 0 else None,
            'decode_rate': output_tokens / decode_time if decode_time > 0 else None
        })

        # Also compute EMA for real-time use
        # But training data includes raw rates
```

Option 2: Record both raw and EMA in training data for model to choose.

---

### Issue 10: Missing deploy_cara.py Updates

**Files**: `block/exp/cara/deploy_cara.py`

**Current Status**: Not updated for new architecture

**Design Requirement** (from review comments):
> "we used to agree to have another major revision and updates over the deploy scripts (deploy_cara.py) but seems missed here"

**Required Updates**:
1. Remove `predictor_ports` → use `data_collection_port` (single)
2. Derive `instance_type` at deployment time
3. Update for DataCollectionService (not predictor)
4. Support scheduler-side predictor config

---

## Low Priority / Future Work

### Issue 11: Extensible Scheduling Policy Interface

Create proper interface hierarchy:
```python
class BaseSchedulingPolicy(ABC):
    @abstractmethod
    async def select_instance(self, request, instances) -> str: pass

class StreamingPolicy(BaseSchedulingPolicy):
    """For baseline: random, round_robin, least_loaded"""
    pass

class BatchPolicy(BaseSchedulingPolicy):
    """For CARA: batch with LPT + multi-objective"""
    pass
```

### Issue 12: Track Predictor Overhead

Add optional timing measurement:
```python
if self.config.track_prediction_overhead:
    start = time.perf_counter()
    prediction = self.model.predict(...)
    self.prediction_times.append(time.perf_counter() - start)
```

### Issue 13: Dynamic Batch Size (AdaptiveBatcher)

Implement load-aware adaptive batching from design:
```python
class AdaptiveBatcher:
    def get_batch_params(self, num_busy_instances, total_instances):
        load_ratio = num_busy_instances / total_instances
        if load_ratio < 0.3:  # Low load
            return BatchParams(max_batch=8, timeout_ms=20)
        elif load_ratio < 0.7:  # Medium load
            return BatchParams(max_batch=32, timeout_ms=50)
        else:  # High load
            return BatchParams(max_batch=64, timeout_ms=100)
```

---

## Implementation Priority

### Phase 1 (Critical - Must fix before any evaluation)
1. **Issue 1**: Streaming batch dispatch (don't wait for all)
2. **Issue 2**: Bootstrap mode streaming dispatch
3. **Issue 3**: Remove auto BOOTSTRAP→CARA conversion
4. **Issue 5**: Add LPT sorting

### Phase 2 (Important - Before paper results)
5. **Issue 4**: Multi-objective scoring
6. **Issue 6**: Target field naming
7. **Issue 8**: DummyCARAPredictor rename

### Phase 3 (Nice to have)
8. **Issue 7**: Real vs predicted decode length
9. **Issue 9**: ServiceRateTracker decouple
10. **Issue 10**: deploy_cara.py updates
11. Issues 11-13: Future enhancements

---

## Testing Plan

After each fix:
1. Run existing unit tests: `pytest tests/predictor/cara/`
2. Run integration test with mock instances
3. Verify behavior matches design document examples

New tests needed:
- Test streaming dispatch (responses return out of order)
- Test LPT sort produces correct ordering
- Test multi-objective scoring with different weights
- Test bootstrap mode routes one-by-one

---

## References

- `cara_paper/claude/LATENCY_PREDICTOR_DESIGN.md` - Primary design document
- `cara_paper/codex/cara-objective-design.md` - Multi-objective scoring
- `cara_paper/claude/review_comments_for_implementation_predictor_v1.md` - Review feedback
- `cara_paper/new_greedy_cara.tex` - Priority-Based Streaming Scheduler algorithm
