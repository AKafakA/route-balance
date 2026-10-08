# CARA Predictor Implementation Tasks - Detailed Specifications

**Date**: January 26, 2026
**Status**: Phase 1 Tasks Complete (1.1, 1.2, 1.4) - Phase 2 in Planning
**Reviewer**: User

---

> **📖 Related Documents**:
> - **IMPLEMENTATION_TASKS_V2_DESIGN.md**: Detailed designs for 12 additional tasks from review comments (Phase 2)
> - This document covers Phase 1 tasks (1.1-1.4) which are now mostly complete

---

## Overview

This document contains detailed task specifications for fixing the implementation to align with the agreed design. Each task includes:
- Problem statement with code references
- Design requirement with document references
- Detailed modification plan
- Unit test specifications
- Acceptance criteria

**Please review each task and confirm alignment before implementation.**

---

# Phase 1: Critical Fixes

---

## Task 1.1: CARA Scheduling Architecture with Global Queue and Dynamic Batching

### Problem Statement

**Current Location**: `block/global_scheduler/cara/cara_serve.py:77-257`

**Current Behavior**:
- `/v1/batch_completions` endpoint receives a fixed batch from client
- Waits for ALL responses before returning ANY
- No internal queue or dynamic batching
- Missing the core CARA architecture

```python
# Current: batch comes from HTTP request, not internal queue
batch_requests = request_json.get("requests", [])
# ...
# Wait for all responses - BLOCKING
for request_id, inst_id, prompt_tokens, future in response_futures:
    response = await future  # Waits for EACH, introduces tail latency
```

**Issues**:
- No global request queue for internal batching
- No dynamic batch size based on cluster load
- No background scheduler loop
- Client-driven batching instead of server-driven
- Fast requests wait for slow requests

### Design Requirement

**Reference**: `LATENCY_PREDICTOR_DESIGN.md` Section 13.4 (Dynamic Batch Size)

**Reference**: `new_greedy_cara.tex` line 161:
> "Requests are appended to a global queue"

**Reference**: `review_comments_for_implementation_predictor_v1.md` lines 36-37:
> "our design is still allow the client to send request individually but for cara batch scheduler, it will append it to a global request queue and scheduling it in batch by itself (with dynamic batch size)"

**Reference**: `usenix.tex` line 149:
> "The set of requests to be scheduled in the current batch, where the batch will be selected as FCFS order with a dynamic batch size"

### Expected Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CARA Scheduler                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   /v1/completions (BOOTSTRAP mode ONLY)                             │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │  • Routes ONE request at a time via route_single()           │   │
│   │  • Uses random/round_robin/least_loaded                      │   │
│   │  • Dispatch immediately → wait → return response             │   │
│   │  • THROWS EXCEPTION if server mode is CARA                   │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│   /v1/cara_completions (CARA mode ONLY) ← NEW ENDPOINT              │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │  • Accepts single request OR array of requests               │   │
│   │  • Enqueues to Global Request Queue                          │   │
│   │  • Waits for response (same format as baseline)              │   │
│   │  • THROWS EXCEPTION if server mode is BOOTSTRAP              │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│                              ▼                                       │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                   Global Request Queue                       │   │
│   │                   (asyncio.Queue)                            │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│                              ▼ (Background Scheduler Loop)           │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │   AdaptiveBatcher.collect_batch()                            │   │
│   │   - Collect up to dynamic_batch_size requests                │   │
│   │   - OR timeout_ms expires (whichever comes first)            │   │
│   │   - Batch size based on cluster load ratio                   │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│                              ▼                                       │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │   CARABatchPolicy.route_batch()                              │   │
│   │   - LPT sort (big requests first)                            │   │
│   │   - Multi-objective scoring                                  │   │
│   │   - Greedy assignment with local state updates               │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│                              ▼                                       │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │   Streaming Dispatch                                         │   │
│   │   - Fire-and-forget to instances                             │   │
│   │   - Resolve request's Future when response arrives           │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Server Mode**: Set at startup via CLI arg `--scheduling-mode bootstrap|cara`
- BOOTSTRAP mode: Only `/v1/completions` allowed
- CARA mode: Only `/v1/cara_completions` allowed

**Busy Instance Definition** (for load calculation):
```python
def is_busy(state, max_num_seqs):
    return state.num_running > max_num_seqs + 1 or state.num_waiting > 1
```

### Proposed Modifications

#### 1.1.1 Add AdaptiveBatcher Class

**New File**: `block/global_scheduler/cara/adaptive_batcher.py`

```python
"""
Adaptive batching for CARA scheduler.

Dynamically adjusts batch size and timeout based on cluster load.
"""
import asyncio
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import logging

from block.predictor.cara.data_structures import SimpleScheduleState

logger = logging.getLogger(__name__)


@dataclass
class BatchParams:
    """Parameters for batch collection."""
    max_batch_size: int
    timeout_ms: float


@dataclass
class AdaptiveBatcherConfig:
    """Configuration for adaptive batching."""
    min_batch_size: int = 4
    max_batch_size: int = 64
    min_timeout_ms: float = 10.0
    max_timeout_ms: float = 100.0

    # Load thresholds
    low_load_threshold: float = 0.3
    high_load_threshold: float = 0.7


class AdaptiveBatcher:
    """Adaptive batcher that adjusts batch size based on cluster load.

    Key insight from design: When instances are busy, batching delay at
    scheduler is "free" - requests would queue at the instance anyway.
    Larger batches improve global optimization.

    Load ratio calculation:
    - Instance is "busy" if: num_running > max_num_seqs + 1 OR num_waiting > 1
    - load_ratio = busy_instances / total_instances

    Batch params by load:
    - Low load (< 0.3): small batch, short timeout → minimize latency
    - Medium load: interpolate
    - High load (> 0.7): large batch, long timeout → improve throughput
    """

    def __init__(self, config: Optional[AdaptiveBatcherConfig] = None):
        self.config = config or AdaptiveBatcherConfig()
        logger.info(
            f"AdaptiveBatcher initialized: batch_size=[{self.config.min_batch_size}, "
            f"{self.config.max_batch_size}], timeout=[{self.config.min_timeout_ms}, "
            f"{self.config.max_timeout_ms}]ms"
        )

    def is_instance_busy(
        self,
        state: SimpleScheduleState,
        max_num_seqs: int = 256
    ) -> bool:
        """Determine if an instance is busy.

        Busy if:
        - num_running > max_num_seqs + 1 (processing more than capacity)
        - OR num_waiting > 1 (requests already queuing at instance)
        """
        return state.num_running > max_num_seqs + 1 or state.num_waiting > 1

    def compute_load_ratio(
        self,
        states: Dict[str, SimpleScheduleState],
        max_num_seqs: int = 256
    ) -> float:
        """Compute cluster load ratio.

        Returns:
            Ratio of busy instances to total instances [0.0, 1.0]
        """
        if not states:
            return 0.0

        busy_count = sum(
            1 for state in states.values()
            if self.is_instance_busy(state, max_num_seqs)
        )
        return busy_count / len(states)

    def get_batch_params(self, load_ratio: float) -> BatchParams:
        """Get batch parameters based on cluster load.

        Args:
            load_ratio: Ratio of busy instances [0.0, 1.0]

        Returns:
            BatchParams with max_batch_size and timeout_ms
        """
        cfg = self.config

        if load_ratio < cfg.low_load_threshold:
            # Low load: small batch, short timeout → minimize latency
            return BatchParams(
                max_batch_size=cfg.min_batch_size,
                timeout_ms=cfg.min_timeout_ms
            )

        elif load_ratio > cfg.high_load_threshold:
            # High load: large batch, long timeout → maximize throughput
            return BatchParams(
                max_batch_size=cfg.max_batch_size,
                timeout_ms=cfg.max_timeout_ms
            )

        else:
            # Medium load: interpolate linearly
            # Normalize load to [0, 1] within the medium range
            range_size = cfg.high_load_threshold - cfg.low_load_threshold
            normalized = (load_ratio - cfg.low_load_threshold) / range_size

            batch_size = int(
                cfg.min_batch_size +
                (cfg.max_batch_size - cfg.min_batch_size) * normalized
            )
            timeout = (
                cfg.min_timeout_ms +
                (cfg.max_timeout_ms - cfg.min_timeout_ms) * normalized
            )

            return BatchParams(
                max_batch_size=batch_size,
                timeout_ms=timeout
            )

    async def collect_batch(
        self,
        queue: asyncio.Queue,
        states: Dict[str, SimpleScheduleState],
        max_num_seqs: int = 256
    ) -> List:
        """Collect a batch of requests from queue.

        Collects requests until EITHER:
        - max_batch_size reached
        - timeout_ms elapsed

        Args:
            queue: Global request queue
            states: Current instance states (for load calculation)
            max_num_seqs: Max sequences per instance (for busy calculation)

        Returns:
            List of requests (may be empty if queue empty and timeout)
        """
        # Calculate dynamic params based on current load
        load_ratio = self.compute_load_ratio(states, max_num_seqs)
        params = self.get_batch_params(load_ratio)

        logger.debug(
            f"Collecting batch: load_ratio={load_ratio:.2f}, "
            f"max_batch={params.max_batch_size}, timeout={params.timeout_ms}ms"
        )

        batch = []
        deadline = time.time() + params.timeout_ms / 1000

        # Always wait for at least one request (blocking)
        if queue.empty():
            try:
                first_request = await asyncio.wait_for(
                    queue.get(),
                    timeout=params.timeout_ms / 1000
                )
                batch.append(first_request)
            except asyncio.TimeoutError:
                return batch  # Empty batch

        # Collect more requests up to batch size or timeout
        while len(batch) < params.max_batch_size:
            remaining = deadline - time.time()
            if remaining <= 0:
                break

            try:
                request = await asyncio.wait_for(queue.get(), timeout=remaining)
                batch.append(request)
            except asyncio.TimeoutError:
                break

        logger.debug(f"Collected batch of {len(batch)} requests")
        return batch
```

#### 1.1.2 Add CARASchedulerLoop Class

**New File**: `block/global_scheduler/cara/cara_scheduler_loop.py`

```python
"""
CARA Scheduler Loop - Background task for batch scheduling.

Continuously:
1. Collects requests from global queue (with dynamic batching)
2. Runs LPT + multi-objective scheduling
3. Dispatches to instances
4. Resolves response futures
"""
import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
import logging

from block.predictor.cara.data_structures import PredictRequest, SimpleScheduleState
from block.global_scheduler.cara.adaptive_batcher import AdaptiveBatcher, AdaptiveBatcherConfig

logger = logging.getLogger(__name__)


@dataclass
class PendingRequest:
    """A request waiting in the global queue."""
    request_id: str
    predict_request: PredictRequest
    original_request: dict
    enqueue_time: float
    response_future: asyncio.Future = field(default_factory=asyncio.Future)


class CARASchedulerLoop:
    """Background scheduler loop for CARA mode.

    Architecture:
    - Maintains global request queue
    - Runs continuous background loop
    - Uses AdaptiveBatcher for dynamic batch sizing
    - Dispatches and resolves response futures

    Usage:
        loop = CARASchedulerLoop(
            instances=instances,
            state_manager=state_manager,
            scheduling_policy=policy
        )

        # Start background loop
        await loop.start()

        # Enqueue requests (from HTTP endpoint)
        future = await loop.enqueue(request)
        response = await future  # Wait for this specific request

        # Shutdown
        await loop.stop()
    """

    def __init__(
        self,
        instances: List,  # List of Instance objects
        state_manager,  # StateManager
        scheduling_policy,  # CARABatchPolicy
        service_rate_tracker = None,
        batcher_config: Optional[AdaptiveBatcherConfig] = None,
        max_num_seqs: int = 256
    ):
        self.instances = instances
        self.instance_map = {inst._instance_id: inst for inst in instances}
        self.state_manager = state_manager
        self.scheduling_policy = scheduling_policy
        self.service_rate_tracker = service_rate_tracker
        self.max_num_seqs = max_num_seqs

        # Adaptive batcher
        self.batcher = AdaptiveBatcher(batcher_config)

        # Global request queue
        self.global_queue: asyncio.Queue[PendingRequest] = asyncio.Queue()

        # Background task
        self._running = False
        self._loop_task: Optional[asyncio.Task] = None

        # Stats
        self._batches_processed = 0
        self._requests_processed = 0
        self._total_queue_time_ms = 0.0

        logger.info("CARASchedulerLoop initialized")

    async def start(self):
        """Start the background scheduler loop."""
        if self._running:
            logger.warning("Scheduler loop already running")
            return

        self._running = True
        self._loop_task = asyncio.create_task(self._scheduler_loop())
        logger.info("CARA scheduler loop started")

    async def stop(self):
        """Stop the background scheduler loop."""
        self._running = False
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
        logger.info("CARA scheduler loop stopped")

    async def enqueue(self, request_dict: dict) -> asyncio.Future:
        """Enqueue a request and return a Future for its response.

        Args:
            request_dict: Original request dictionary

        Returns:
            Future that will be resolved when response is ready
        """
        request_id = request_dict.get("request_id", f"req-{time.time()}")
        request_dict["request_id"] = request_id

        # Create PredictRequest for scheduling
        predict_request = PredictRequest(
            request_id=request_id,
            num_prompt_tokens=request_dict.get("prompt_len", 0),
            num_predicted_output_tokens=request_dict.get("max_tokens", 256)
        )

        # Create pending request with future
        pending = PendingRequest(
            request_id=request_id,
            predict_request=predict_request,
            original_request=request_dict,
            enqueue_time=time.time()
        )

        # Enqueue
        await self.global_queue.put(pending)
        logger.debug(f"Enqueued request {request_id}, queue size: {self.global_queue.qsize()}")

        return pending.response_future

    async def _scheduler_loop(self):
        """Main scheduler loop - runs continuously."""
        logger.info("Scheduler loop running...")

        while self._running:
            try:
                # Step 1: Refresh instance states
                if self.state_manager:
                    await self.state_manager.refresh_all()
                    states = self.state_manager.states

                    # Augment with service rate tracker if available
                    if self.service_rate_tracker:
                        self.service_rate_tracker.augment_states(states)
                else:
                    states = {}

                # Step 2: Collect batch from queue (with dynamic sizing)
                batch = await self.batcher.collect_batch(
                    self.global_queue,
                    states,
                    self.max_num_seqs
                )

                if not batch:
                    # Empty batch (timeout with no requests), continue
                    continue

                # Step 3: Schedule batch
                predict_requests = [p.predict_request for p in batch]
                instance_ids = list(self.instance_map.keys())

                assignments = await self.scheduling_policy.route_batch(
                    predict_requests,
                    instance_ids
                )

                # Step 4: Dispatch and resolve futures
                await self._dispatch_batch(batch, assignments)

                # Update stats
                self._batches_processed += 1
                self._requests_processed += len(batch)

            except asyncio.CancelledError:
                logger.info("Scheduler loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}", exc_info=True)
                await asyncio.sleep(0.1)  # Brief pause on error

    async def _dispatch_batch(
        self,
        batch: List[PendingRequest],
        assignments: Dict[str, List[PredictRequest]]
    ):
        """Dispatch batch to instances and resolve futures."""

        # Build request_id -> PendingRequest lookup
        pending_map = {p.request_id: p for p in batch}

        # Dispatch each assignment
        for inst_id, assigned_requests in assignments.items():
            instance = self.instance_map.get(inst_id)
            if not instance:
                continue

            for pred_req in assigned_requests:
                pending = pending_map.get(pred_req.request_id)
                if not pending:
                    continue

                # Record queue time
                queue_time_ms = (time.time() - pending.enqueue_time) * 1000
                self._total_queue_time_ms += queue_time_ms

                # Dispatch asynchronously (fire-and-forget)
                asyncio.create_task(
                    self._dispatch_and_resolve(instance, inst_id, pending)
                )

    async def _dispatch_and_resolve(
        self,
        instance,
        instance_id: str,
        pending: PendingRequest
    ):
        """Execute request on instance and resolve its future."""
        try:
            max_tokens = pending.original_request.get("max_tokens", 256)

            response = await instance.query_instance(
                pending.original_request,
                predicted_num_decode_tokens=max_tokens
            )

            # Update service rate tracker
            if self.service_rate_tracker and isinstance(response, dict):
                ttft = response.get('ttft', response.get('time_to_first_token', 0))
                e2e = response.get('e2e_latency', response.get('latency', 0))
                output_tokens = response.get('output_tokens',
                                response.get('usage', {}).get('completion_tokens', 0))

                if e2e > 0:
                    self.service_rate_tracker.update(instance_id, {
                        'ttft': ttft,
                        'e2e_latency': e2e,
                        'num_prompt_tokens': pending.predict_request.num_prompt_tokens,
                        'output_tokens': output_tokens
                    })

            # Resolve future
            pending.response_future.set_result(response)

        except Exception as e:
            logger.error(f"Request {pending.request_id} failed on {instance_id}: {e}")
            pending.response_future.set_exception(e)

    def get_stats(self) -> Dict:
        """Get scheduler loop statistics."""
        avg_queue_time = (
            self._total_queue_time_ms / self._requests_processed
            if self._requests_processed > 0 else 0
        )
        return {
            "running": self._running,
            "queue_size": self.global_queue.qsize(),
            "batches_processed": self._batches_processed,
            "requests_processed": self._requests_processed,
            "avg_queue_time_ms": avg_queue_time,
        }
```

#### 1.1.3 Modify cara_serve.py - Add New Endpoint and Mode Enforcement

**File**: `block/global_scheduler/cara/cara_serve.py`

**Changes**:

1. Add server mode enum and global:
```python
from enum import Enum

class ServerMode(Enum):
    BOOTSTRAP = "bootstrap"
    CARA = "cara"

# Global server mode (set at startup)
server_mode: ServerMode = ServerMode.BOOTSTRAP

# CARA scheduler loop (only initialized in CARA mode)
cara_scheduler_loop: Optional[CARASchedulerLoop] = None
```

2. Modify `/v1/completions` to enforce BOOTSTRAP mode:
```python
@app.post("/v1/completions")
async def completion(request: Request) -> Response:
    global num_requests, server_mode

    # Enforce mode
    if server_mode != ServerMode.BOOTSTRAP:
        return JSONResponse(
            content={
                "error": f"Endpoint /v1/completions only available in BOOTSTRAP mode. "
                         f"Current mode: {server_mode.value}. Use /v1/cara_completions instead."
            },
            status_code=400
        )

    # ... existing bootstrap logic using route_single() ...
```

3. Add new `/v1/cara_completions` endpoint:
```python
@app.post("/v1/cara_completions")
async def cara_completions(request: Request) -> Response:
    """CARA completion endpoint with global queue and dynamic batching.

    Request body:
    {
        "prompt": "...",
        "max_tokens": 256,
        "prompt_len": 128,
        ...
    }
    OR array:
    {
        "requests": [
            {"prompt": "...", "max_tokens": 256, ...},
            ...
        ]
    }

    Response: Same format as /v1/completions for easy comparison.
    """
    global num_requests, server_mode, cara_scheduler_loop

    # Enforce mode
    if server_mode != ServerMode.CARA:
        return JSONResponse(
            content={
                "error": f"Endpoint /v1/cara_completions only available in CARA mode. "
                         f"Current mode: {server_mode.value}. Use /v1/completions instead."
            },
            status_code=400
        )

    if cara_scheduler_loop is None:
        return JSONResponse(
            content={"error": "CARA scheduler loop not initialized"},
            status_code=500
        )

    request_json = await request.json()

    # Support both single request and array of requests
    if "requests" in request_json:
        requests_list = request_json["requests"]
    else:
        requests_list = [request_json]

    # Enqueue all requests and collect futures
    futures = []
    for req in requests_list:
        num_requests += 1
        req["request_id"] = req.get("request_id", f"cara-{num_requests}")
        served_requests.append(req["request_id"])

        future = await cara_scheduler_loop.enqueue(req)
        futures.append((req["request_id"], future))

    # Wait for all responses
    responses = []
    for request_id, future in futures:
        try:
            response = await future
            responses.append(response)
        except Exception as e:
            responses.append({
                "request_id": request_id,
                "success": False,
                "error": str(e)
            })

    # Return single response or array based on input
    if len(responses) == 1 and "requests" not in request_json:
        return JSONResponse(content=responses[0])
    else:
        return JSONResponse(content={"responses": responses})
```

4. Update `init_app` to initialize based on mode:
```python
async def init_app(args: Namespace, instances_list: Optional[List[Instance]] = None) -> FastAPI:
    global server_mode, cara_scheduler_loop

    # ... existing init code ...

    # Set server mode from CLI arg
    mode_str = getattr(args, 'scheduling_mode', 'bootstrap').lower()
    server_mode = ServerMode(mode_str)
    logger.info(f"Server mode: {server_mode.value}")

    if server_mode == ServerMode.CARA:
        # Initialize CARA components
        await _init_scheduler_side_components(args, model_dict)

        # Initialize and start scheduler loop
        cara_scheduler_loop = CARASchedulerLoop(
            instances=instances,
            state_manager=state_manager,
            scheduling_policy=scheduling_policy,
            service_rate_tracker=service_rate_tracker,
            batcher_config=AdaptiveBatcherConfig(
                min_batch_size=getattr(args, 'min_batch_size', 4),
                max_batch_size=getattr(args, 'max_batch_size', 64),
                min_timeout_ms=getattr(args, 'min_timeout_ms', 10.0),
                max_timeout_ms=getattr(args, 'max_timeout_ms', 100.0),
            )
        )
        await cara_scheduler_loop.start()
        logger.info("CARA scheduler loop started")

    else:
        # BOOTSTRAP mode - initialize streaming policy
        scheduling_policy = StreamingBaselinePolicy(
            strategy=StreamingBaselinePolicy.Strategy(args.scheduling),
            state_manager=state_manager
        )
        logger.info(f"BOOTSTRAP mode with strategy: {args.scheduling}")
```

5. Add CLI args:
```python
parser.add_argument("--scheduling-mode", type=str, default="bootstrap",
                    choices=["bootstrap", "cara"],
                    help="Server scheduling mode: bootstrap (baseline) or cara")
parser.add_argument("--min-batch-size", type=int, default=4,
                    help="Minimum batch size for CARA mode")
parser.add_argument("--max-batch-size", type=int, default=64,
                    help="Maximum batch size for CARA mode")
parser.add_argument("--min-timeout-ms", type=float, default=10.0,
                    help="Minimum batch timeout in ms for CARA mode")
parser.add_argument("--max-timeout-ms", type=float, default=100.0,
                    help="Maximum batch timeout in ms for CARA mode")
```

### Implementation Sub-Tasks and Dependencies

The Task 1.1 changes require updates to multiple modules. Here's the implementation order:

| Sub-task | File | Description | Status |
|----------|------|-------------|--------|
| 1.1.a | `block/predictor/cara/cara_predictor_config.py` | Add `AdaptiveBatcherConfig`, `CARASchedulerConfig` | ✅ |
| 1.1.b | `block/predictor/cara/data_structures.py` | Add `max_num_seqs` to `SimpleScheduleState` if missing | ✅ (already exists) |
| 1.1.c | `block/global_scheduler/cara/adaptive_batcher.py` | **NEW FILE**: AdaptiveBatcher class | ✅ |
| 1.1.d | `block/global_scheduler/cara/cara_scheduler_loop.py` | **NEW FILE**: CARASchedulerLoop class | ✅ |
| 1.1.e | `block/predictor/cara/state_manager.py` | Verify `refresh_all()` returns proper states dict | ✅ (already correct) |
| 1.1.f | `block/global_scheduler/cara/cara_serve.py` | Add `ServerMode`, endpoints, CLI args | ✅ |
| 1.1.g | `block/exp/cara/deploy_cara.py` | Add `--scheduling-mode` CLI arg support | ⏳ (deferred) |

**Dependency Graph**:
```
1.1.a (configs) ─┬─► 1.1.c (AdaptiveBatcher)
1.1.b (data)    ─┤
                 ├─► 1.1.d (CARASchedulerLoop) ──► 1.1.f (cara_serve.py)
1.1.e (state)   ─┘                                        │
                                                          ▼
                                                   1.1.g (deploy)
```

### Unit Test Specifications

#### Test File 1: `tests/global_scheduler/cara/test_adaptive_batcher.py`

```python
"""
Unit tests for AdaptiveBatcher.
"""
import pytest
import asyncio
from unittest.mock import MagicMock

from block.global_scheduler.cara.adaptive_batcher import (
    AdaptiveBatcher,
    AdaptiveBatcherConfig,
    BatchParams,
)
from block.predictor.cara.data_structures import SimpleScheduleState


class TestAdaptiveBatcher:
    """Tests for AdaptiveBatcher."""

    @pytest.fixture
    def default_config(self):
        return AdaptiveBatcherConfig(
            min_batch_size=4,
            max_batch_size=64,
            min_timeout_ms=10.0,
            max_timeout_ms=100.0,
            low_load_threshold=0.3,
            high_load_threshold=0.7
        )

    @pytest.fixture
    def batcher(self, default_config):
        return AdaptiveBatcher(default_config)

    def test_is_instance_busy_by_running(self, batcher):
        """Test instance is busy when num_running > max_num_seqs + 1."""
        max_num_seqs = 10

        # Not busy: num_running <= max_num_seqs + 1
        state_not_busy = SimpleScheduleState(instance_id="a", num_running=10, num_waiting=0)
        assert batcher.is_instance_busy(state_not_busy, max_num_seqs) is False

        # Busy: num_running > max_num_seqs + 1
        state_busy = SimpleScheduleState(instance_id="b", num_running=12, num_waiting=0)
        assert batcher.is_instance_busy(state_busy, max_num_seqs) is True

    def test_is_instance_busy_by_waiting(self, batcher):
        """Test instance is busy when num_waiting > 1."""
        max_num_seqs = 256

        # Not busy: num_waiting <= 1
        state_not_busy = SimpleScheduleState(instance_id="a", num_running=5, num_waiting=1)
        assert batcher.is_instance_busy(state_not_busy, max_num_seqs) is False

        # Busy: num_waiting > 1
        state_busy = SimpleScheduleState(instance_id="b", num_running=5, num_waiting=2)
        assert batcher.is_instance_busy(state_busy, max_num_seqs) is True

    def test_compute_load_ratio_all_idle(self, batcher):
        """Test load ratio when all instances are idle."""
        states = {
            "a": SimpleScheduleState(instance_id="a", num_running=5, num_waiting=0),
            "b": SimpleScheduleState(instance_id="b", num_running=3, num_waiting=0),
            "c": SimpleScheduleState(instance_id="c", num_running=0, num_waiting=0),
        }
        assert batcher.compute_load_ratio(states, max_num_seqs=256) == 0.0

    def test_compute_load_ratio_all_busy(self, batcher):
        """Test load ratio when all instances are busy."""
        states = {
            "a": SimpleScheduleState(instance_id="a", num_running=5, num_waiting=5),
            "b": SimpleScheduleState(instance_id="b", num_running=3, num_waiting=3),
            "c": SimpleScheduleState(instance_id="c", num_running=0, num_waiting=2),
        }
        assert batcher.compute_load_ratio(states, max_num_seqs=256) == 1.0

    def test_compute_load_ratio_mixed(self, batcher):
        """Test load ratio with mixed busy/idle instances."""
        states = {
            "a": SimpleScheduleState(instance_id="a", num_running=5, num_waiting=5),  # Busy
            "b": SimpleScheduleState(instance_id="b", num_running=3, num_waiting=0),  # Idle
            "c": SimpleScheduleState(instance_id="c", num_running=0, num_waiting=0),  # Idle
        }
        # 1 out of 3 busy = 0.333...
        assert abs(batcher.compute_load_ratio(states, max_num_seqs=256) - 1/3) < 0.01

    def test_get_batch_params_low_load(self, batcher, default_config):
        """Test batch params for low load (< 0.3)."""
        params = batcher.get_batch_params(load_ratio=0.1)

        assert params.max_batch_size == default_config.min_batch_size  # 4
        assert params.timeout_ms == default_config.min_timeout_ms  # 10

    def test_get_batch_params_high_load(self, batcher, default_config):
        """Test batch params for high load (> 0.7)."""
        params = batcher.get_batch_params(load_ratio=0.9)

        assert params.max_batch_size == default_config.max_batch_size  # 64
        assert params.timeout_ms == default_config.max_timeout_ms  # 100

    def test_get_batch_params_medium_load_interpolates(self, batcher):
        """Test batch params interpolates for medium load."""
        # At 0.5 load (middle of 0.3-0.7 range), should be ~50% between min and max
        params = batcher.get_batch_params(load_ratio=0.5)

        # 0.5 is at 50% of range [0.3, 0.7]
        # Expected batch: 4 + (64-4) * 0.5 = 4 + 30 = 34
        assert 30 <= params.max_batch_size <= 38

        # Expected timeout: 10 + (100-10) * 0.5 = 10 + 45 = 55
        assert 50 <= params.timeout_ms <= 60

    @pytest.mark.asyncio
    async def test_collect_batch_respects_max_size(self, batcher):
        """Test that collect_batch stops at max_batch_size."""
        queue = asyncio.Queue()

        # Add many requests
        for i in range(100):
            await queue.put({"request_id": f"req-{i}"})

        # With low load, max_batch_size = 4
        states = {}  # Empty = 0 load
        batch = await batcher.collect_batch(queue, states)

        assert len(batch) == 4  # min_batch_size

    @pytest.mark.asyncio
    async def test_collect_batch_respects_timeout(self, batcher):
        """Test that collect_batch returns on timeout even if batch not full."""
        queue = asyncio.Queue()

        # Add only 2 requests (less than min_batch_size of 4)
        await queue.put({"request_id": "req-1"})
        await queue.put({"request_id": "req-2"})

        states = {}  # Low load = 10ms timeout
        batch = await batcher.collect_batch(queue, states)

        # Should return 2 requests after timeout
        assert len(batch) == 2

    @pytest.mark.asyncio
    async def test_collect_batch_empty_queue_timeout(self, batcher):
        """Test that collect_batch returns empty on empty queue timeout."""
        queue = asyncio.Queue()
        states = {}

        batch = await batcher.collect_batch(queue, states)

        assert batch == []
```

#### Test File 2: `tests/global_scheduler/cara/test_cara_scheduler_loop.py`

```python
"""
Unit tests for CARASchedulerLoop.
"""
import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch

from block.global_scheduler.cara.cara_scheduler_loop import (
    CARASchedulerLoop,
    PendingRequest,
)
from block.global_scheduler.cara.adaptive_batcher import AdaptiveBatcherConfig
from block.predictor.cara.data_structures import PredictRequest, SimpleScheduleState


class MockInstance:
    """Mock instance for testing."""

    def __init__(self, instance_id: str, delay: float = 0.01):
        self._instance_id = instance_id
        self.delay = delay
        self.queries = []

    async def query_instance(self, request: dict, predicted_num_decode_tokens: int):
        self.queries.append(request["request_id"])
        await asyncio.sleep(self.delay)
        return {
            "request_id": request["request_id"],
            "success": True,
            "instance_id": self._instance_id,
            "e2e_latency": self.delay
        }


class MockStateManager:
    """Mock state manager for testing."""

    def __init__(self, states: dict = None):
        self.states = states or {}
        self.refresh_count = 0

    async def refresh_all(self):
        self.refresh_count += 1


class MockSchedulingPolicy:
    """Mock scheduling policy for testing."""

    def __init__(self, assignment_strategy="round_robin"):
        self.strategy = assignment_strategy
        self._rr_idx = 0

    async def route_batch(self, requests, instance_ids):
        assignments = {inst_id: [] for inst_id in instance_ids}

        for req in requests:
            if self.strategy == "round_robin":
                target = instance_ids[self._rr_idx % len(instance_ids)]
                self._rr_idx += 1
            else:
                target = instance_ids[0]
            assignments[target].append(req)

        return assignments


class TestCARASchedulerLoop:
    """Tests for CARASchedulerLoop."""

    @pytest.fixture
    def instances(self):
        return [
            MockInstance("inst-0", delay=0.01),
            MockInstance("inst-1", delay=0.02),
        ]

    @pytest.fixture
    def state_manager(self):
        return MockStateManager(states={
            "inst-0": SimpleScheduleState(instance_id="inst-0"),
            "inst-1": SimpleScheduleState(instance_id="inst-1"),
        })

    @pytest.fixture
    def scheduling_policy(self):
        return MockSchedulingPolicy()

    @pytest.fixture
    def scheduler_loop(self, instances, state_manager, scheduling_policy):
        return CARASchedulerLoop(
            instances=instances,
            state_manager=state_manager,
            scheduling_policy=scheduling_policy,
            batcher_config=AdaptiveBatcherConfig(
                min_batch_size=2,
                max_batch_size=4,
                min_timeout_ms=50,
                max_timeout_ms=100
            )
        )

    @pytest.mark.asyncio
    async def test_enqueue_returns_future(self, scheduler_loop):
        """Test that enqueue returns a Future."""
        future = await scheduler_loop.enqueue({"prompt": "test", "max_tokens": 10})

        assert isinstance(future, asyncio.Future)
        assert scheduler_loop.global_queue.qsize() == 1

    @pytest.mark.asyncio
    async def test_enqueue_assigns_request_id(self, scheduler_loop):
        """Test that enqueue assigns request_id if missing."""
        request = {"prompt": "test", "max_tokens": 10}
        await scheduler_loop.enqueue(request)

        assert "request_id" in request

    @pytest.mark.asyncio
    async def test_start_stop(self, scheduler_loop):
        """Test starting and stopping the scheduler loop."""
        await scheduler_loop.start()
        assert scheduler_loop._running is True
        assert scheduler_loop._loop_task is not None

        await scheduler_loop.stop()
        assert scheduler_loop._running is False

    @pytest.mark.asyncio
    async def test_request_processed_and_response_returned(self, scheduler_loop):
        """Test full flow: enqueue → schedule → dispatch → response."""
        await scheduler_loop.start()

        try:
            # Enqueue request
            future = await scheduler_loop.enqueue({
                "prompt": "test",
                "max_tokens": 10,
                "prompt_len": 50
            })

            # Wait for response with timeout
            response = await asyncio.wait_for(future, timeout=5.0)

            assert response["success"] is True
            assert "instance_id" in response

        finally:
            await scheduler_loop.stop()

    @pytest.mark.asyncio
    async def test_multiple_requests_batched(self, scheduler_loop, instances):
        """Test that multiple requests are batched together."""
        await scheduler_loop.start()

        try:
            # Enqueue multiple requests quickly
            futures = []
            for i in range(4):
                future = await scheduler_loop.enqueue({
                    "prompt": f"test-{i}",
                    "max_tokens": 10,
                    "prompt_len": 50
                })
                futures.append(future)

            # Wait for all responses
            responses = await asyncio.gather(*[
                asyncio.wait_for(f, timeout=5.0) for f in futures
            ])

            assert len(responses) == 4
            assert all(r["success"] for r in responses)

            # Check stats
            stats = scheduler_loop.get_stats()
            assert stats["requests_processed"] == 4
            # Should have been processed in batches (batch size 2-4)
            assert stats["batches_processed"] >= 1

        finally:
            await scheduler_loop.stop()

    @pytest.mark.asyncio
    async def test_state_refreshed_each_batch(self, scheduler_loop, state_manager):
        """Test that state is refreshed before each batch."""
        await scheduler_loop.start()

        try:
            # Enqueue and wait for processing
            future = await scheduler_loop.enqueue({"prompt": "test", "max_tokens": 10})
            await asyncio.wait_for(future, timeout=5.0)

            # State should have been refreshed
            assert state_manager.refresh_count >= 1

        finally:
            await scheduler_loop.stop()

    @pytest.mark.asyncio
    async def test_get_stats(self, scheduler_loop):
        """Test statistics tracking."""
        stats = scheduler_loop.get_stats()

        assert "running" in stats
        assert "queue_size" in stats
        assert "batches_processed" in stats
        assert "requests_processed" in stats
        assert "avg_queue_time_ms" in stats

        assert stats["running"] is False
        assert stats["batches_processed"] == 0


class TestModeEnforcement:
    """Tests for endpoint mode enforcement."""

    @pytest.mark.asyncio
    async def test_completions_rejected_in_cara_mode(self):
        """Test /v1/completions returns error in CARA mode."""
        # This would be an integration test with the actual endpoint
        # Placeholder for the test structure
        pass

    @pytest.mark.asyncio
    async def test_cara_completions_rejected_in_bootstrap_mode(self):
        """Test /v1/cara_completions returns error in BOOTSTRAP mode."""
        # This would be an integration test with the actual endpoint
        # Placeholder for the test structure
        pass
```

### Acceptance Criteria

#### Core Components
- [ ] `AdaptiveBatcher` class created with dynamic batch sizing
- [ ] `CARASchedulerLoop` class created with background loop
- [ ] Global request queue (`asyncio.Queue`) implemented
- [ ] Busy instance detection: `num_running > max_num_seqs + 1 OR num_waiting > 1`

#### Endpoint Behavior
- [ ] `/v1/completions` only works in BOOTSTRAP mode (throws exception otherwise)
- [ ] `/v1/cara_completions` only works in CARA mode (throws exception otherwise)
- [ ] `/v1/cara_completions` accepts single request OR array of requests
- [ ] Response format matches `/v1/completions` for easy comparison

#### Dynamic Batching
- [ ] Low load (< 30%): small batch (4), short timeout (10ms)
- [ ] High load (> 70%): large batch (64), long timeout (100ms)
- [ ] Medium load: interpolates between min/max

#### Server Initialization
- [ ] `--scheduling-mode bootstrap|cara` CLI argument added
- [ ] CARA mode initializes scheduler loop automatically
- [ ] BOOTSTRAP mode uses `StreamingBaselinePolicy` with `route_single()`

#### Testing
- [ ] All `test_adaptive_batcher.py` tests pass
- [ ] All `test_cara_scheduler_loop.py` tests pass
- [ ] Service rate tracker updated on completion
- [ ] Error handling captures instance failures gracefully

#### Statistics
- [ ] `get_stats()` returns queue_size, batches_processed, requests_processed, avg_queue_time_ms

---

## Task 1.2: Bootstrap Mode - Streaming Dispatch

### Problem Statement

**Current Location**: `block/predictor/cara/scheduling_policy.py:163-237`

**Current Behavior**:
Bootstrap mode (random, round_robin, least_loaded) uses the same `route_batch()` method as CARA, processing all requests as a batch.

```python
async def route_batch(self, requests, instance_ids):
    if self._mode == SchedulingMode.BOOTSTRAP:
        return await self._route_bootstrap(requests, instance_ids)  # Still batched!
    else:
        return await self._route_cara(requests, instance_ids)
```

### Design Requirement

**Reference**: `review_comments_for_implementation_predictor_v1.md` lines 9-11:

> "rr/random and least_load scheduling besides act as the bootstrap for training data of latency predictor, it will also act as the baseline over cara during evaluation, and always suppose to be dispatched in streaming way (one by one) for fairness per-batch is not their requirement"

**Expected Behavior**:
- Bootstrap policies route ONE request at a time (streaming)
- No batch-level coordination or optimization
- Fair comparison baseline for CARA evaluation

### Proposed Modifications

#### 1.2.1 Add Abstract Base Class for Policies

**Modified File**: `block/predictor/cara/scheduling_policy.py`

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, TYPE_CHECKING
from enum import Enum
import random
import logging

if TYPE_CHECKING:
    from block.predictor.cara.state_manager import StateManager
    from block.predictor.cara.predictor_registry import PredictorRegistry

from block.predictor.cara.data_structures import SimpleScheduleState, PredictRequest

logger = logging.getLogger(__name__)


class SchedulingMode(Enum):
    """Scheduling modes."""
    BOOTSTRAP = "bootstrap"  # Streaming baseline (random/rr/least_loaded)
    CARA = "cara"            # Batch with LPT + multi-objective


class BaseSchedulingPolicy(ABC):
    """Abstract base class for scheduling policies.

    All policies must implement:
    - route_single(): Route one request (streaming)
    - route_batch(): Route multiple requests

    Streaming policies (bootstrap) route one-by-one.
    Batch policies (CARA) optimize batch assignment.
    """

    @abstractmethod
    async def route_single(
        self,
        request: PredictRequest,
        instance_ids: List[str]
    ) -> str:
        """Route a single request to an instance.

        Args:
            request: Request to route
            instance_ids: Available instances

        Returns:
            Selected instance_id
        """
        pass

    @abstractmethod
    async def route_batch(
        self,
        requests: List[PredictRequest],
        instance_ids: List[str]
    ) -> Dict[str, List[PredictRequest]]:
        """Route a batch of requests.

        Args:
            requests: Requests to route
            instance_ids: Available instances

        Returns:
            Dict mapping instance_id to assigned requests
        """
        pass

    @abstractmethod
    def get_mode(self) -> SchedulingMode:
        """Get current scheduling mode."""
        pass

    @abstractmethod
    def get_stats(self) -> Dict:
        """Get policy statistics."""
        pass


class StreamingBaselinePolicy(BaseSchedulingPolicy):
    """Baseline policies with streaming (one-by-one) dispatch.

    Strategies:
    - RANDOM: Random instance selection
    - ROUND_ROBIN: Cycle through instances
    - LEAST_LOADED: Pick instance with fewest requests

    These policies route each request independently without
    batch-level optimization. Used for:
    1. Training data collection (bootstrap phase)
    2. Baseline comparison for CARA evaluation
    """

    class Strategy(Enum):
        RANDOM = "random"
        ROUND_ROBIN = "round_robin"
        LEAST_LOADED = "least_loaded"

    def __init__(
        self,
        strategy: 'StreamingBaselinePolicy.Strategy' = None,
        state_manager: Optional['StateManager'] = None
    ):
        """Initialize streaming baseline policy.

        Args:
            strategy: Routing strategy (default: ROUND_ROBIN)
            state_manager: For least_loaded strategy (optional)
        """
        self.strategy = strategy or self.Strategy.ROUND_ROBIN
        self.state_manager = state_manager
        self._rr_index = 0
        self._requests_routed = 0

        logger.info(f"StreamingBaselinePolicy initialized: strategy={self.strategy.value}")

    def get_mode(self) -> SchedulingMode:
        return SchedulingMode.BOOTSTRAP

    async def route_single(
        self,
        request: PredictRequest,
        instance_ids: List[str]
    ) -> str:
        """Route single request using configured strategy."""
        if not instance_ids:
            raise ValueError("No instances available")

        self._requests_routed += 1

        if self.strategy == self.Strategy.RANDOM:
            return random.choice(instance_ids)

        elif self.strategy == self.Strategy.ROUND_ROBIN:
            target = instance_ids[self._rr_index % len(instance_ids)]
            self._rr_index += 1
            return target

        elif self.strategy == self.Strategy.LEAST_LOADED:
            return self._select_least_loaded(instance_ids)

        else:
            # Fallback
            return random.choice(instance_ids)

    def _select_least_loaded(self, instance_ids: List[str]) -> str:
        """Select instance with fewest requests."""
        if self.state_manager is None:
            # No state info, fall back to random
            return random.choice(instance_ids)

        min_load = float('inf')
        best = instance_ids[0]

        for inst_id in instance_ids:
            state = self.state_manager.get_state(inst_id)
            if state is not None:
                load = state.num_running + state.num_waiting
            else:
                load = 0  # Assume empty if no state

            if load < min_load:
                min_load = load
                best = inst_id

        return best

    async def route_batch(
        self,
        requests: List[PredictRequest],
        instance_ids: List[str]
    ) -> Dict[str, List[PredictRequest]]:
        """Route batch by streaming (one-by-one).

        NOTE: This is NOT batch optimization. Each request is
        routed independently using route_single().
        """
        assignments = {inst_id: [] for inst_id in instance_ids}

        for request in requests:
            target = await self.route_single(request, instance_ids)
            assignments[target].append(request)

        return assignments

    def get_stats(self) -> Dict:
        return {
            "mode": self.get_mode().value,
            "strategy": self.strategy.value,
            "requests_routed": self._requests_routed,
            "rr_index": self._rr_index,
        }


class CARABatchPolicy(BaseSchedulingPolicy):
    """CARA policy with batch optimization.

    Features:
    - LPT sorting (optional, for ablation)
    - Multi-objective scoring (latency, cost, quality, memory)
    - Per-batch state refresh
    - Local state updates during assignment
    """

    def __init__(
        self,
        state_manager: 'StateManager',
        predictor_registry: Optional['PredictorRegistry'] = None,
        config: Optional[Dict] = None
    ):
        self.state_manager = state_manager
        self.predictor_registry = predictor_registry
        self.config = config or {}

        # Config options
        self.enable_lpt_sort = self.config.get('enable_lpt_sort', True)
        self.sort_key = self.config.get('sort_key', 'total_tokens')

        # Stats
        self._batches_processed = 0
        self._requests_routed = 0
        self._rr_fallback_count = 0
        self._rr_index = 0

        logger.info(
            f"CARABatchPolicy initialized: lpt_sort={self.enable_lpt_sort}, "
            f"sort_key={self.sort_key}"
        )

    def get_mode(self) -> SchedulingMode:
        return SchedulingMode.CARA

    async def route_single(
        self,
        request: PredictRequest,
        instance_ids: List[str]
    ) -> str:
        """Route single request (for API compatibility).

        Wraps request in batch and returns single assignment.
        """
        assignments = await self.route_batch([request], instance_ids)
        for inst_id, reqs in assignments.items():
            if reqs:
                return inst_id
        return instance_ids[0]  # Fallback

    async def route_batch(
        self,
        requests: List[PredictRequest],
        instance_ids: List[str]
    ) -> Dict[str, List[PredictRequest]]:
        """Route batch with LPT + greedy assignment."""
        self._batches_processed += 1

        if not requests:
            return {inst_id: [] for inst_id in instance_ids}

        # Step 1: LPT sort if enabled
        sorted_requests = self._apply_lpt_sort(requests)

        # Step 2: Greedy assignment
        assignments = {inst_id: [] for inst_id in instance_ids}

        for request in sorted_requests:
            best_instance = await self._select_best_instance(request, instance_ids)
            assignments[best_instance].append(request)
            self._requests_routed += 1

            # Update local state for next assignment
            self.state_manager.apply_assignment(best_instance, request)

        return assignments

    def _apply_lpt_sort(
        self,
        requests: List[PredictRequest]
    ) -> List[PredictRequest]:
        """Sort requests by size (LPT: Longest Processing Time First)."""
        if not self.enable_lpt_sort:
            return requests

        if self.sort_key == 'total_tokens':
            key_fn = lambda r: r.num_prompt_tokens + r.num_predicted_output_tokens
        elif self.sort_key == 'prompt_only':
            key_fn = lambda r: r.num_prompt_tokens
        elif self.sort_key == 'output_only':
            key_fn = lambda r: r.num_predicted_output_tokens
        else:
            key_fn = lambda r: r.num_prompt_tokens + r.num_predicted_output_tokens

        return sorted(requests, key=key_fn, reverse=True)  # Descending

    async def _select_best_instance(
        self,
        request: PredictRequest,
        instance_ids: List[str]
    ) -> str:
        """Select best instance using predictor or fallback."""
        if self.predictor_registry is None:
            # Fallback to round-robin
            self._rr_fallback_count += 1
            target = instance_ids[self._rr_index % len(instance_ids)]
            self._rr_index += 1
            return target

        best_instance = None
        best_score = float('inf')

        for inst_id in instance_ids:
            state = self.state_manager.get_state(inst_id)
            if state is None:
                continue

            try:
                # Get prediction
                instance_type = self._get_instance_type(inst_id)
                prediction = await self.predictor_registry.predict(
                    instance_type=instance_type,
                    state=state,
                    request=request
                )
                score = prediction.get('prediction', prediction.get('latency_prediction', float('inf')))

                # TODO: Add multi-objective scoring here (Task 1.4)

                if score < best_score:
                    best_score = score
                    best_instance = inst_id

            except Exception as e:
                logger.warning(f"Prediction failed for {inst_id}: {e}")

        if best_instance is None:
            # Fallback
            self._rr_fallback_count += 1
            best_instance = instance_ids[self._rr_index % len(instance_ids)]
            self._rr_index += 1

        return best_instance

    def _get_instance_type(self, instance_id: str) -> str:
        """Get instance type from ID (placeholder for actual lookup)."""
        return instance_id

    def get_stats(self) -> Dict:
        return {
            "mode": self.get_mode().value,
            "enable_lpt_sort": self.enable_lpt_sort,
            "sort_key": self.sort_key,
            "batches_processed": self._batches_processed,
            "requests_routed": self._requests_routed,
            "rr_fallback_count": self._rr_fallback_count,
        }


# Factory function for backward compatibility
def create_policy_from_config(
    state_manager: 'StateManager',
    predictor_registry: Optional['PredictorRegistry'] = None,
    config: Optional[Dict] = None
) -> BaseSchedulingPolicy:
    """Create scheduling policy from config.

    Args:
        state_manager: StateManager instance
        predictor_registry: PredictorRegistry (None for bootstrap only)
        config: Configuration dict with:
            - mode: "bootstrap" or "cara"
            - bootstrap_strategy: "random", "round_robin", "least_loaded"
            - enable_lpt_sort: bool
            - sort_key: "total_tokens", "prompt_only", "output_only"

    Returns:
        Configured scheduling policy
    """
    config = config or {}
    mode = config.get('mode', 'bootstrap').lower()

    if mode == 'bootstrap':
        strategy_str = config.get('bootstrap_strategy', 'round_robin').lower()
        try:
            strategy = StreamingBaselinePolicy.Strategy(strategy_str)
        except ValueError:
            logger.warning(f"Unknown strategy '{strategy_str}', using round_robin")
            strategy = StreamingBaselinePolicy.Strategy.ROUND_ROBIN

        return StreamingBaselinePolicy(
            strategy=strategy,
            state_manager=state_manager
        )

    elif mode == 'cara':
        return CARABatchPolicy(
            state_manager=state_manager,
            predictor_registry=predictor_registry,
            config=config
        )

    else:
        logger.warning(f"Unknown mode '{mode}', using bootstrap")
        return StreamingBaselinePolicy(state_manager=state_manager)
```

### Unit Test Specifications

**Test File**: `tests/predictor/cara/test_scheduling_policy_v2.py`

```python
"""
Unit tests for refactored scheduling policies.

Tests streaming baseline vs batch CARA policies.
"""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from block.predictor.cara.scheduling_policy import (
    SchedulingMode,
    BaseSchedulingPolicy,
    StreamingBaselinePolicy,
    CARABatchPolicy,
    create_policy_from_config,
)
from block.predictor.cara.data_structures import PredictRequest, SimpleScheduleState


class TestStreamingBaselinePolicy:
    """Tests for streaming baseline policies."""

    @pytest.fixture
    def instance_ids(self):
        return ["inst-0", "inst-1", "inst-2"]

    @pytest.fixture
    def requests(self):
        return [
            PredictRequest(request_id="r1", num_prompt_tokens=100, num_predicted_output_tokens=50),
            PredictRequest(request_id="r2", num_prompt_tokens=200, num_predicted_output_tokens=100),
            PredictRequest(request_id="r3", num_prompt_tokens=150, num_predicted_output_tokens=75),
        ]

    def test_mode_is_bootstrap(self):
        """Test that streaming policy returns BOOTSTRAP mode."""
        policy = StreamingBaselinePolicy()
        assert policy.get_mode() == SchedulingMode.BOOTSTRAP

    @pytest.mark.asyncio
    async def test_random_strategy_selects_from_instances(self, instance_ids, requests):
        """Test random strategy selects valid instances."""
        policy = StreamingBaselinePolicy(strategy=StreamingBaselinePolicy.Strategy.RANDOM)

        selected = set()
        for req in requests:
            target = await policy.route_single(req, instance_ids)
            assert target in instance_ids
            selected.add(target)

        # With randomness, might not hit all, but should hit some
        assert len(selected) >= 1

    @pytest.mark.asyncio
    async def test_round_robin_cycles_through_instances(self, instance_ids, requests):
        """Test round-robin cycles through all instances in order."""
        policy = StreamingBaselinePolicy(strategy=StreamingBaselinePolicy.Strategy.ROUND_ROBIN)

        targets = []
        for req in requests:
            target = await policy.route_single(req, instance_ids)
            targets.append(target)

        # Should cycle: inst-0, inst-1, inst-2
        assert targets == ["inst-0", "inst-1", "inst-2"]

    @pytest.mark.asyncio
    async def test_round_robin_wraps_around(self, instance_ids):
        """Test round-robin wraps after all instances."""
        policy = StreamingBaselinePolicy(strategy=StreamingBaselinePolicy.Strategy.ROUND_ROBIN)

        # Route 5 requests through 3 instances
        requests = [
            PredictRequest(request_id=f"r{i}", num_prompt_tokens=100, num_predicted_output_tokens=50)
            for i in range(5)
        ]

        targets = []
        for req in requests:
            target = await policy.route_single(req, instance_ids)
            targets.append(target)

        # Should cycle: 0, 1, 2, 0, 1
        assert targets == ["inst-0", "inst-1", "inst-2", "inst-0", "inst-1"]

    @pytest.mark.asyncio
    async def test_least_loaded_selects_min_load(self, instance_ids):
        """Test least-loaded selects instance with minimum load."""
        # Mock state manager
        state_manager = MagicMock()
        state_manager.get_state.side_effect = lambda inst_id: {
            "inst-0": SimpleScheduleState(instance_id="inst-0", num_running=5, num_waiting=3),
            "inst-1": SimpleScheduleState(instance_id="inst-1", num_running=2, num_waiting=1),  # Least: 3
            "inst-2": SimpleScheduleState(instance_id="inst-2", num_running=4, num_waiting=2),
        }.get(inst_id)

        policy = StreamingBaselinePolicy(
            strategy=StreamingBaselinePolicy.Strategy.LEAST_LOADED,
            state_manager=state_manager
        )

        request = PredictRequest(request_id="r1", num_prompt_tokens=100, num_predicted_output_tokens=50)
        target = await policy.route_single(request, instance_ids)

        assert target == "inst-1"  # Lowest load (2+1=3)

    @pytest.mark.asyncio
    async def test_route_batch_routes_one_by_one(self, instance_ids, requests):
        """Test that route_batch calls route_single for each request."""
        policy = StreamingBaselinePolicy(strategy=StreamingBaselinePolicy.Strategy.ROUND_ROBIN)

        assignments = await policy.route_batch(requests, instance_ids)

        # Each instance should get one request (round-robin)
        assert len(assignments["inst-0"]) == 1
        assert len(assignments["inst-1"]) == 1
        assert len(assignments["inst-2"]) == 1

        # Verify correct assignment
        assert assignments["inst-0"][0].request_id == "r1"
        assert assignments["inst-1"][0].request_id == "r2"
        assert assignments["inst-2"][0].request_id == "r3"

    def test_stats_tracks_requests(self):
        """Test that stats tracks routed requests."""
        policy = StreamingBaselinePolicy()

        stats = policy.get_stats()
        assert stats["requests_routed"] == 0

        # Would need to call route_single to increment


class TestCARABatchPolicy:
    """Tests for CARA batch policy with LPT."""

    @pytest.fixture
    def state_manager(self):
        manager = MagicMock()
        manager.get_state.return_value = SimpleScheduleState(
            instance_id="test",
            num_running=5,
            num_waiting=2
        )
        manager.apply_assignment = MagicMock()
        return manager

    @pytest.fixture
    def predictor_registry(self):
        registry = MagicMock()
        # Return different latencies for different instances
        async def mock_predict(instance_type, state, request):
            latencies = {"inst-0": 1.0, "inst-1": 0.5, "inst-2": 0.8}
            return {"latency_prediction": latencies.get(instance_type, 1.0)}

        registry.predict = AsyncMock(side_effect=mock_predict)
        return registry

    @pytest.fixture
    def instance_ids(self):
        return ["inst-0", "inst-1", "inst-2"]

    def test_mode_is_cara(self, state_manager):
        """Test that CARA policy returns CARA mode."""
        policy = CARABatchPolicy(state_manager=state_manager)
        assert policy.get_mode() == SchedulingMode.CARA

    @pytest.mark.asyncio
    async def test_lpt_sort_orders_by_total_tokens_desc(self, state_manager, predictor_registry, instance_ids):
        """Test LPT sorts requests by total tokens descending."""
        policy = CARABatchPolicy(
            state_manager=state_manager,
            predictor_registry=predictor_registry,
            config={'enable_lpt_sort': True, 'sort_key': 'total_tokens'}
        )

        requests = [
            PredictRequest(request_id="small", num_prompt_tokens=50, num_predicted_output_tokens=25),  # 75
            PredictRequest(request_id="large", num_prompt_tokens=200, num_predicted_output_tokens=100),  # 300
            PredictRequest(request_id="medium", num_prompt_tokens=100, num_predicted_output_tokens=50),  # 150
        ]

        sorted_requests = policy._apply_lpt_sort(requests)

        # Should be large, medium, small (descending by total tokens)
        assert sorted_requests[0].request_id == "large"
        assert sorted_requests[1].request_id == "medium"
        assert sorted_requests[2].request_id == "small"

    @pytest.mark.asyncio
    async def test_lpt_disabled_preserves_order(self, state_manager, predictor_registry, instance_ids):
        """Test that disabling LPT preserves original order."""
        policy = CARABatchPolicy(
            state_manager=state_manager,
            predictor_registry=predictor_registry,
            config={'enable_lpt_sort': False}
        )

        requests = [
            PredictRequest(request_id="first", num_prompt_tokens=50, num_predicted_output_tokens=25),
            PredictRequest(request_id="second", num_prompt_tokens=200, num_predicted_output_tokens=100),
            PredictRequest(request_id="third", num_prompt_tokens=100, num_predicted_output_tokens=50),
        ]

        sorted_requests = policy._apply_lpt_sort(requests)

        # Order should be preserved
        assert sorted_requests[0].request_id == "first"
        assert sorted_requests[1].request_id == "second"
        assert sorted_requests[2].request_id == "third"

    @pytest.mark.asyncio
    async def test_selects_instance_with_lowest_latency(self, state_manager, predictor_registry, instance_ids):
        """Test that best instance is selected by lowest predicted latency."""
        policy = CARABatchPolicy(
            state_manager=state_manager,
            predictor_registry=predictor_registry,
            config={'enable_lpt_sort': False}
        )

        request = PredictRequest(request_id="r1", num_prompt_tokens=100, num_predicted_output_tokens=50)

        assignments = await policy.route_batch([request], instance_ids)

        # inst-1 has lowest latency (0.5)
        assert len(assignments["inst-1"]) == 1
        assert assignments["inst-1"][0].request_id == "r1"

    @pytest.mark.asyncio
    async def test_apply_assignment_called_for_each_request(self, state_manager, predictor_registry, instance_ids):
        """Test that state is updated after each assignment."""
        policy = CARABatchPolicy(
            state_manager=state_manager,
            predictor_registry=predictor_registry
        )

        requests = [
            PredictRequest(request_id="r1", num_prompt_tokens=100, num_predicted_output_tokens=50),
            PredictRequest(request_id="r2", num_prompt_tokens=200, num_predicted_output_tokens=100),
        ]

        await policy.route_batch(requests, instance_ids)

        # apply_assignment should be called for each request
        assert state_manager.apply_assignment.call_count == 2

    @pytest.mark.asyncio
    async def test_fallback_to_round_robin_without_predictor(self, state_manager, instance_ids):
        """Test fallback to round-robin when no predictor available."""
        policy = CARABatchPolicy(
            state_manager=state_manager,
            predictor_registry=None  # No predictor
        )

        requests = [
            PredictRequest(request_id=f"r{i}", num_prompt_tokens=100, num_predicted_output_tokens=50)
            for i in range(3)
        ]

        assignments = await policy.route_batch(requests, instance_ids)

        # Should use round-robin fallback
        assert policy.get_stats()["rr_fallback_count"] == 3


class TestCreatePolicyFromConfig:
    """Tests for factory function."""

    def test_creates_bootstrap_policy_by_default(self):
        """Test default is bootstrap policy."""
        state_manager = MagicMock()
        policy = create_policy_from_config(state_manager, config={})

        assert isinstance(policy, StreamingBaselinePolicy)
        assert policy.get_mode() == SchedulingMode.BOOTSTRAP

    def test_creates_cara_policy_when_specified(self):
        """Test CARA mode creates batch policy."""
        state_manager = MagicMock()
        policy = create_policy_from_config(
            state_manager,
            config={'mode': 'cara'}
        )

        assert isinstance(policy, CARABatchPolicy)
        assert policy.get_mode() == SchedulingMode.CARA

    def test_respects_bootstrap_strategy_config(self):
        """Test bootstrap strategy is configurable."""
        state_manager = MagicMock()
        policy = create_policy_from_config(
            state_manager,
            config={'mode': 'bootstrap', 'bootstrap_strategy': 'random'}
        )

        assert isinstance(policy, StreamingBaselinePolicy)
        assert policy.strategy == StreamingBaselinePolicy.Strategy.RANDOM

    def test_respects_lpt_config(self):
        """Test LPT config is passed to CARA policy."""
        state_manager = MagicMock()
        policy = create_policy_from_config(
            state_manager,
            config={
                'mode': 'cara',
                'enable_lpt_sort': False,
                'sort_key': 'prompt_only'
            }
        )

        assert isinstance(policy, CARABatchPolicy)
        assert policy.enable_lpt_sort is False
        assert policy.sort_key == 'prompt_only'
```

### Acceptance Criteria

- [ ] `BaseSchedulingPolicy` abstract class defined
- [ ] `StreamingBaselinePolicy` implements streaming (one-by-one) dispatch
- [ ] `CARABatchPolicy` implements batch with LPT sorting
- [ ] Factory function `create_policy_from_config` updated
- [ ] Old `SchedulingPolicy` class replaced or deprecated
- [ ] All unit tests pass
- [ ] Bootstrap mode routes independently (no batch coordination)
- [ ] CARA mode applies LPT sort when enabled

---

## Task 1.3: Remove Automatic BOOTSTRAP → CARA Conversion

### Problem Statement

**Current Location**: `block/predictor/cara/scheduling_policy.py:146-161`

**Current Behavior**:
```python
def record_training_sample(self):
    self._training_samples += 1
    # Auto-convert to CARA after enough samples
    if (self._mode == SchedulingMode.BOOTSTRAP and
        self._training_samples >= self.min_training_samples and
        self.predictor_registry is not None and
        self.predictor_registry.list_instance_types()):
        logger.info(f"Transitioning to CARA mode after {self._training_samples} samples")
        self._mode = SchedulingMode.CARA
```

### Design Requirement

**Reference**: `review_comments_for_implementation_predictor_v1.md` lines 11-12:

> "the automatical conversation between bootstrapping and cara can not be required, as in our current design/plan, for the latency prediction, no real prompts/responses is needed"

**Expected Behavior**:
- Mode switching is EXPLICIT only (via config or API)
- `record_training_sample()` only updates metrics
- Bootstrap mode persists for baseline evaluation

### Proposed Modifications

In the refactored `scheduling_policy.py` (Task 1.2), the auto-conversion is already removed since we have separate policy classes. However, we should also:

1. Remove `min_training_samples` parameter (no longer needed for auto-conversion)
2. Keep sample counting for metrics/logging only
3. Add explicit mode-switching methods for testing

**Changes to StreamingBaselinePolicy**:
```python
class StreamingBaselinePolicy(BaseSchedulingPolicy):
    def record_training_sample(self):
        """Record a completed training sample (metrics only, no mode change)."""
        self._training_samples += 1
```

**Changes to CARABatchPolicy**:
```python
class CARABatchPolicy(BaseSchedulingPolicy):
    def record_training_sample(self):
        """Record a completed training sample for metrics."""
        self._training_samples += 1
```

### Unit Test Specifications

```python
class TestNoAutoModeConversion:
    """Tests verifying no automatic mode conversion."""

    @pytest.mark.asyncio
    async def test_bootstrap_remains_bootstrap_after_many_samples(self):
        """Test bootstrap mode doesn't auto-convert to CARA."""
        policy = StreamingBaselinePolicy()

        # Record many training samples
        for _ in range(10000):
            policy.record_training_sample()

        # Should still be bootstrap
        assert policy.get_mode() == SchedulingMode.BOOTSTRAP

    def test_mode_only_changes_via_explicit_factory(self):
        """Test that mode is determined at creation, not runtime."""
        state_manager = MagicMock()

        # Create bootstrap policy
        bootstrap = create_policy_from_config(
            state_manager,
            config={'mode': 'bootstrap'}
        )
        assert bootstrap.get_mode() == SchedulingMode.BOOTSTRAP

        # Create CARA policy
        cara = create_policy_from_config(
            state_manager,
            config={'mode': 'cara'}
        )
        assert cara.get_mode() == SchedulingMode.CARA

        # Policies are separate, no conversion
```

### Acceptance Criteria

- [ ] `record_training_sample()` only updates counters
- [ ] No automatic mode conversion logic exists
- [ ] Mode determined at policy creation only
- [ ] Unit tests verify no auto-conversion

---

## Task 1.4: Add Multi-Objective Scoring

### Problem Statement

**Current Location**: `block/predictor/cara/scheduling_policy.py:264-289`

**Current Behavior**:
```python
# Only considers latency
latency = prediction.get('latency_prediction', float('inf'))
if latency < best_latency:
    best_latency = latency
    best_instance = inst_id
```

### Design Requirement

**Reference**: `cara_paper/codex/cara-objective-design.md`:

```
score = α·lat_term + β·cost_term + γ·qual_term
```

With normalization (min-max, percentile, or regret) and constraint handling.

### Proposed Modifications

#### 1.4.1 Create MultiObjectiveScorer Class

**New File**: `block/predictor/cara/multi_objective_scorer.py`

```python
"""
Multi-objective scoring for CARA instance selection.

Combines latency, cost, quality, and memory balance signals
with configurable weights and normalization.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging

from block.predictor.cara.data_structures import SimpleScheduleState, PredictRequest

logger = logging.getLogger(__name__)


class NormalizationMethod(Enum):
    """Normalization methods for multi-objective scoring."""
    MINMAX = "minmax"          # Per-request min-max scaling
    PERCENTILE = "percentile"  # CDF-based (requires rolling histograms)
    REGRET = "regret"          # Regret vs best candidate


@dataclass
class ScoringConfig:
    """Configuration for multi-objective scoring."""

    # Objective weights (lower score is better)
    w_latency: float = 1.0      # α - latency weight
    w_cost: float = 0.1         # β - cost weight
    w_quality: float = 0.0      # γ - quality weight (0 = disabled)
    w_memory: float = 0.1       # Memory balance weight

    # Normalization
    normalization: NormalizationMethod = NormalizationMethod.MINMAX

    # Constraint handling
    enable_constraints: bool = False
    latency_constraint: Optional[float] = None  # Max acceptable latency
    quality_constraint: Optional[float] = None  # Min acceptable quality

    # Uncertainty handling
    k_sigma: float = 1.0  # Uncertainty margin multiplier

    # Penalty for constraint violations
    constraint_penalty: float = 10.0


@dataclass
class CandidateScore:
    """Score for a single candidate instance."""
    instance_id: str

    # Raw values
    latency: float = 0.0
    cost: float = 0.0
    quality: float = 0.0
    memory_util: float = 0.0

    # Normalized values
    lat_norm: float = 0.0
    cost_norm: float = 0.0
    qual_norm: float = 0.0
    mem_norm: float = 0.0

    # Final score
    score: float = 0.0

    # Constraint violations
    lat_violation: float = 0.0
    qual_violation: float = 0.0
    is_feasible: bool = True


class MultiObjectiveScorer:
    """Scores instances for request assignment using multiple objectives.

    Features:
    - Configurable weights for latency, cost, quality, memory
    - Multiple normalization methods
    - Constraint handling with penalties
    - Uncertainty-aware bounds

    Usage:
        scorer = MultiObjectiveScorer(config)

        # Score all candidates for a request
        scores = scorer.score_candidates(
            request=request,
            candidates=[(inst_id, state, prediction), ...],
            quality_estimates={inst_id: quality, ...}  # optional
        )

        # Get best instance
        best = scorer.select_best(scores)
    """

    def __init__(self, config: ScoringConfig):
        self.config = config
        logger.info(
            f"MultiObjectiveScorer initialized: "
            f"w_lat={config.w_latency}, w_cost={config.w_cost}, "
            f"w_qual={config.w_quality}, w_mem={config.w_memory}, "
            f"norm={config.normalization.value}"
        )

    def score_candidates(
        self,
        request: PredictRequest,
        candidates: List[Tuple[str, SimpleScheduleState, Dict]],  # (inst_id, state, prediction)
        quality_estimates: Optional[Dict[str, float]] = None,
        cost_factors: Optional[Dict[str, float]] = None
    ) -> List[CandidateScore]:
        """Score all candidate instances for a request.

        Args:
            request: Request to assign
            candidates: List of (instance_id, state, prediction_dict)
            quality_estimates: Optional quality scores per instance
            cost_factors: Optional cost multipliers per instance

        Returns:
            List of CandidateScore objects
        """
        if not candidates:
            return []

        quality_estimates = quality_estimates or {}
        cost_factors = cost_factors or {}

        # Step 1: Compute raw scores for each candidate
        raw_scores = []
        for inst_id, state, prediction in candidates:
            latency = prediction.get('prediction',
                      prediction.get('latency_prediction', float('inf')))

            # Cost proxy: tokens * cost factor
            cost = self._compute_cost(request, cost_factors.get(inst_id, 1.0))

            # Quality (optional)
            quality = quality_estimates.get(inst_id, 1.0)  # Default: max quality

            # Memory utilization
            memory_util = state.kv_cache_utilization if state else 0.0

            raw_scores.append(CandidateScore(
                instance_id=inst_id,
                latency=latency,
                cost=cost,
                quality=quality,
                memory_util=memory_util
            ))

        # Step 2: Normalize
        self._normalize_scores(raw_scores)

        # Step 3: Check constraints and compute violations
        self._apply_constraints(raw_scores)

        # Step 4: Compute final weighted score
        for score in raw_scores:
            base_score = (
                self.config.w_latency * score.lat_norm +
                self.config.w_cost * score.cost_norm +
                self.config.w_quality * score.qual_norm +
                self.config.w_memory * score.mem_norm
            )

            # Add constraint penalty if violated
            penalty = 0.0
            if not score.is_feasible:
                penalty = self.config.constraint_penalty * (
                    score.lat_violation + score.qual_violation
                )

            score.score = base_score + penalty

        return raw_scores

    def select_best(self, scores: List[CandidateScore]) -> Optional[str]:
        """Select instance with lowest score.

        Prefers feasible candidates over infeasible.
        """
        if not scores:
            return None

        # Sort: feasible first, then by score
        sorted_scores = sorted(
            scores,
            key=lambda s: (not s.is_feasible, s.score)
        )

        return sorted_scores[0].instance_id

    def _compute_cost(self, request: PredictRequest, cost_factor: float) -> float:
        """Compute cost proxy for request."""
        total_tokens = request.num_prompt_tokens + request.num_predicted_output_tokens
        return total_tokens * cost_factor

    def _normalize_scores(self, scores: List[CandidateScore]):
        """Normalize scores using configured method."""
        if self.config.normalization == NormalizationMethod.MINMAX:
            self._normalize_minmax(scores)
        elif self.config.normalization == NormalizationMethod.REGRET:
            self._normalize_regret(scores)
        # PERCENTILE requires rolling histograms - future work

    def _normalize_minmax(self, scores: List[CandidateScore]):
        """Per-request min-max normalization to [0, 1]."""
        eps = 1e-9

        # Latency: lower is better
        lat_min = min(s.latency for s in scores)
        lat_max = max(s.latency for s in scores)
        lat_range = lat_max - lat_min + eps

        # Cost: lower is better
        cost_min = min(s.cost for s in scores)
        cost_max = max(s.cost for s in scores)
        cost_range = cost_max - cost_min + eps

        # Quality: higher is better, so invert
        qual_min = min(s.quality for s in scores)
        qual_max = max(s.quality for s in scores)
        qual_range = qual_max - qual_min + eps

        # Memory: lower is better (prefer less utilized)
        mem_min = min(s.memory_util for s in scores)
        mem_max = max(s.memory_util for s in scores)
        mem_range = mem_max - mem_min + eps

        for s in scores:
            s.lat_norm = (s.latency - lat_min) / lat_range
            s.cost_norm = (s.cost - cost_min) / cost_range
            s.qual_norm = 1.0 - (s.quality - qual_min) / qual_range  # Invert
            s.mem_norm = (s.memory_util - mem_min) / mem_range

    def _normalize_regret(self, scores: List[CandidateScore]):
        """Regret normalization vs best candidate."""
        eps = 1e-9

        best_lat = min(s.latency for s in scores)
        best_cost = min(s.cost for s in scores)
        best_qual = max(s.quality for s in scores)
        best_mem = min(s.memory_util for s in scores)

        for s in scores:
            s.lat_norm = (s.latency - best_lat) / (best_lat + eps)
            s.cost_norm = (s.cost - best_cost) / (best_cost + eps)
            s.qual_norm = (best_qual - s.quality) / (best_qual + eps)
            s.mem_norm = (s.memory_util - best_mem) / (best_mem + eps)

    def _apply_constraints(self, scores: List[CandidateScore]):
        """Check constraints and compute violations."""
        cfg = self.config

        for s in scores:
            s.is_feasible = True

            # Latency constraint
            if cfg.latency_constraint is not None:
                if s.latency > cfg.latency_constraint:
                    s.is_feasible = False
                    s.lat_violation = (s.latency - cfg.latency_constraint) / cfg.latency_constraint

            # Quality constraint
            if cfg.quality_constraint is not None:
                if s.quality < cfg.quality_constraint:
                    s.is_feasible = False
                    s.qual_violation = (cfg.quality_constraint - s.quality) / cfg.quality_constraint
```

#### 1.4.2 Integrate with CARABatchPolicy

**Modify**: `block/predictor/cara/scheduling_policy.py`

```python
from block.predictor.cara.multi_objective_scorer import (
    MultiObjectiveScorer,
    ScoringConfig,
    NormalizationMethod
)

class CARABatchPolicy(BaseSchedulingPolicy):
    def __init__(
        self,
        state_manager: 'StateManager',
        predictor_registry: Optional['PredictorRegistry'] = None,
        config: Optional[Dict] = None
    ):
        # ... existing init ...

        # Initialize multi-objective scorer
        scoring_config = ScoringConfig(
            w_latency=config.get('w_latency', 1.0),
            w_cost=config.get('w_cost', 0.1),
            w_quality=config.get('w_quality', 0.0),
            w_memory=config.get('w_memory', 0.1),
            normalization=NormalizationMethod(config.get('normalization', 'minmax')),
            enable_constraints=config.get('enable_constraints', False),
            latency_constraint=config.get('latency_constraint'),
            quality_constraint=config.get('quality_constraint'),
        )
        self.scorer = MultiObjectiveScorer(scoring_config)

    async def _select_best_instance(
        self,
        request: PredictRequest,
        instance_ids: List[str]
    ) -> str:
        """Select best instance using multi-objective scoring."""

        # Gather predictions for all candidates
        candidates = []
        for inst_id in instance_ids:
            state = self.state_manager.get_state(inst_id)
            if state is None:
                continue

            try:
                instance_type = self._get_instance_type(inst_id)
                prediction = await self.predictor_registry.predict(
                    instance_type=instance_type,
                    state=state,
                    request=request
                )
                candidates.append((inst_id, state, prediction))
            except Exception as e:
                logger.warning(f"Prediction failed for {inst_id}: {e}")

        if not candidates:
            # Fallback
            return self._round_robin_fallback(instance_ids)

        # Score using multi-objective scorer
        scores = self.scorer.score_candidates(
            request=request,
            candidates=candidates,
            quality_estimates=None,  # Future: integrate quality model
            cost_factors=None  # Future: per-instance cost factors
        )

        best = self.scorer.select_best(scores)
        return best or self._round_robin_fallback(instance_ids)
```

### Unit Test Specifications

**Test File**: `tests/predictor/cara/test_multi_objective_scorer.py`

```python
"""
Unit tests for multi-objective scoring.
"""
import pytest
from block.predictor.cara.multi_objective_scorer import (
    MultiObjectiveScorer,
    ScoringConfig,
    CandidateScore,
    NormalizationMethod,
)
from block.predictor.cara.data_structures import PredictRequest, SimpleScheduleState


class TestMultiObjectiveScorer:
    """Tests for MultiObjectiveScorer."""

    @pytest.fixture
    def default_config(self):
        return ScoringConfig(
            w_latency=1.0,
            w_cost=0.1,
            w_quality=0.0,
            w_memory=0.1
        )

    @pytest.fixture
    def request(self):
        return PredictRequest(
            request_id="test",
            num_prompt_tokens=100,
            num_predicted_output_tokens=50
        )

    def test_selects_lowest_latency_when_only_latency_weighted(self, request):
        """Test that with only latency weight, lowest latency wins."""
        config = ScoringConfig(w_latency=1.0, w_cost=0.0, w_quality=0.0, w_memory=0.0)
        scorer = MultiObjectiveScorer(config)

        candidates = [
            ("inst-0", SimpleScheduleState(instance_id="inst-0"), {"latency_prediction": 1.0}),
            ("inst-1", SimpleScheduleState(instance_id="inst-1"), {"latency_prediction": 0.5}),
            ("inst-2", SimpleScheduleState(instance_id="inst-2"), {"latency_prediction": 0.8}),
        ]

        scores = scorer.score_candidates(request, candidates)
        best = scorer.select_best(scores)

        assert best == "inst-1"  # Lowest latency

    def test_memory_balance_affects_selection(self, request):
        """Test that memory utilization affects selection."""
        config = ScoringConfig(w_latency=0.5, w_cost=0.0, w_quality=0.0, w_memory=0.5)
        scorer = MultiObjectiveScorer(config)

        # inst-1 has slightly higher latency but much lower memory util
        candidates = [
            ("inst-0", SimpleScheduleState(instance_id="inst-0", kv_cache_utilization=0.9),
             {"latency_prediction": 0.5}),
            ("inst-1", SimpleScheduleState(instance_id="inst-1", kv_cache_utilization=0.2),
             {"latency_prediction": 0.6}),  # Higher latency but lower memory
        ]

        scores = scorer.score_candidates(request, candidates)
        best = scorer.select_best(scores)

        # inst-1 should win due to better memory balance
        assert best == "inst-1"

    def test_constraint_violation_penalized(self, request):
        """Test that constraint violations add penalty."""
        config = ScoringConfig(
            w_latency=1.0,
            enable_constraints=True,
            latency_constraint=0.6,  # Max 0.6s
            constraint_penalty=10.0
        )
        scorer = MultiObjectiveScorer(config)

        candidates = [
            ("inst-0", SimpleScheduleState(instance_id="inst-0"),
             {"latency_prediction": 0.5}),  # Within constraint
            ("inst-1", SimpleScheduleState(instance_id="inst-1"),
             {"latency_prediction": 0.4}),  # Within constraint, best latency
            ("inst-2", SimpleScheduleState(instance_id="inst-2"),
             {"latency_prediction": 1.0}),  # Violates constraint
        ]

        scores = scorer.score_candidates(request, candidates)

        # inst-2 should be marked infeasible
        inst2_score = next(s for s in scores if s.instance_id == "inst-2")
        assert inst2_score.is_feasible is False
        assert inst2_score.lat_violation > 0

        # Best should be feasible with lowest latency
        best = scorer.select_best(scores)
        assert best == "inst-1"

    def test_minmax_normalization_range(self, request):
        """Test min-max normalization produces [0, 1] range."""
        config = ScoringConfig(normalization=NormalizationMethod.MINMAX)
        scorer = MultiObjectiveScorer(config)

        candidates = [
            ("a", SimpleScheduleState(instance_id="a"), {"latency_prediction": 1.0}),
            ("b", SimpleScheduleState(instance_id="b"), {"latency_prediction": 2.0}),
            ("c", SimpleScheduleState(instance_id="c"), {"latency_prediction": 3.0}),
        ]

        scores = scorer.score_candidates(request, candidates)

        # Normalized latencies should be 0, 0.5, 1
        lat_norms = sorted([s.lat_norm for s in scores])
        assert abs(lat_norms[0] - 0.0) < 0.01
        assert abs(lat_norms[1] - 0.5) < 0.01
        assert abs(lat_norms[2] - 1.0) < 0.01

    def test_regret_normalization(self, request):
        """Test regret normalization computes correctly."""
        config = ScoringConfig(normalization=NormalizationMethod.REGRET)
        scorer = MultiObjectiveScorer(config)

        candidates = [
            ("a", SimpleScheduleState(instance_id="a"), {"latency_prediction": 1.0}),  # Best
            ("b", SimpleScheduleState(instance_id="b"), {"latency_prediction": 2.0}),  # 100% regret
        ]

        scores = scorer.score_candidates(request, candidates)

        a_score = next(s for s in scores if s.instance_id == "a")
        b_score = next(s for s in scores if s.instance_id == "b")

        # Best candidate has 0 regret
        assert abs(a_score.lat_norm - 0.0) < 0.01
        # b has 100% regret (1.0 / 1.0)
        assert abs(b_score.lat_norm - 1.0) < 0.01

    def test_empty_candidates_returns_empty(self, request, default_config):
        """Test handling of empty candidates."""
        scorer = MultiObjectiveScorer(default_config)

        scores = scorer.score_candidates(request, [])

        assert scores == []
        assert scorer.select_best(scores) is None
```

### Acceptance Criteria

- [ ] `MultiObjectiveScorer` class created
- [ ] `ScoringConfig` dataclass with configurable weights
- [ ] Min-max normalization implemented and tested
- [ ] Regret normalization implemented and tested
- [ ] Constraint handling with penalties
- [ ] Integrated with `CARABatchPolicy._select_best_instance()`
- [ ] All unit tests pass
- [ ] Configurable via scheduler config JSON

---

## Summary Checklist

### Phase 1 Tasks

| Task | Description | Status |
|------|-------------|--------|
| 1.1 | CARA Scheduling Architecture (Global Queue + Dynamic Batching) | Pending Review |
| 1.2 | Bootstrap Streaming Policy | Pending Review |
| 1.3 | Remove Auto Mode Conversion | Pending Review |
| 1.4 | Multi-Objective Scoring | Pending Review |

### Files to Create

**Task 1.1 - CARA Architecture**:
- `block/global_scheduler/cara/adaptive_batcher.py` - Dynamic batch sizing
- `block/global_scheduler/cara/cara_scheduler_loop.py` - Background scheduler loop
- `tests/global_scheduler/cara/test_adaptive_batcher.py`
- `tests/global_scheduler/cara/test_cara_scheduler_loop.py`

**Task 1.2 - Scheduling Policies**:
- `tests/predictor/cara/test_scheduling_policy_v2.py`

**Task 1.4 - Multi-Objective Scoring**:
- `block/predictor/cara/multi_objective_scorer.py`
- `tests/predictor/cara/test_multi_objective_scorer.py`

### Files to Modify

- `block/global_scheduler/cara/cara_serve.py` - Add `/v1/cara_completions`, mode enforcement
- `block/predictor/cara/scheduling_policy.py` - Refactor to separate classes

### CLI Argument Changes

```bash
# New arguments for cara_serve.py
--scheduling-mode bootstrap|cara    # Server mode (default: bootstrap)
--min-batch-size 4                  # CARA mode: min batch size
--max-batch-size 64                 # CARA mode: max batch size
--min-timeout-ms 10                 # CARA mode: min batch timeout
--max-timeout-ms 100                # CARA mode: max batch timeout
```

### Config Changes

New scheduler config options:
```json
{
    "mode": "cara",
    "bootstrap_strategy": "round_robin",
    "enable_lpt_sort": true,
    "sort_key": "total_tokens",
    "w_latency": 1.0,
    "w_cost": 0.1,
    "w_quality": 0.0,
    "w_memory": 0.1,
    "normalization": "minmax",
    "enable_constraints": false,
    "latency_constraint": null,
    "quality_constraint": null,
    "constraint_penalty": 10.0
}
```

### Endpoint Summary

| Endpoint | Mode | Behavior |
|----------|------|----------|
| `/v1/completions` | BOOTSTRAP only | Routes single request via `route_single()` |
| `/v1/cara_completions` | CARA only | Enqueues to global queue, batched scheduling |

---

**Please review these task specifications and confirm alignment with the design before implementation.**
