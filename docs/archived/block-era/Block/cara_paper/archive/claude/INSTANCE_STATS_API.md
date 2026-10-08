# Instance Stats API Documentation

## Overview

The `/instance_stats` API endpoint provides aggregated scheduler statistics for monitoring and load balancing purposes. It returns a compact summary of the vLLM scheduler state, enabling external load balancers to make informed routing decisions.

**Endpoint:** `GET /instance_stats`

**Response Format:** JSON

---

## Response Fields

### Batch State Counts

| Field | Type | Description |
|-------|------|-------------|
| `num_running` | int | Number of requests currently in the running queue (actively being processed) |
| `num_waiting` | int | Number of requests in the waiting queue (includes new requests and preempted requests) |
| `num_active_decode_seqs` | int | Number of requests that have completed prefill and are actively generating tokens |

### Context Length Distribution (Decode Sequences Only)

These statistics describe the context lengths of requests in the decode phase:

| Field | Type | Description |
|-------|------|-------------|
| `decode_ctx_mean` | int | Mean context length across all decode sequences |
| `decode_ctx_p50` | int | Median (50th percentile) context length |
| `decode_ctx_p95` | int | 95th percentile context length |
| `decode_ctx_max` | int | Maximum context length among decode sequences |
| `total_decode_context_tokens` | int | Sum of all context lengths for decode sequences |

### Backlog in Tokens

| Field | Type | Description |
|-------|------|-------------|
| `pending_prefill_tokens` | int | Total tokens waiting to be prefilled (prompt tokens not yet computed) |
| `pending_decode_tokens` | int | Estimated remaining decode tokens across all requests |

### Memory Pressure

| Field | Type | Description |
|-------|------|-------------|
| `kv_cache_utilization` | float | KV cache utilization ratio (0.0 to 1.0) |
| `kv_free_blocks` | int | Number of free KV cache blocks available |

### Scheduler Configuration (Static)

| Field | Type | Description |
|-------|------|-------------|
| `token_budget_per_iter` | int | Maximum tokens that can be scheduled per iteration (`max_num_batched_tokens`) |
| `max_num_seqs` | int | Maximum number of sequences that can run concurrently (`max_num_seqs`) |

### Metadata

| Field | Type | Description |
|-------|------|-------------|
| `num_preempted` | int | Cumulative count of preemption events since server start |
| `timestamp` | float | Unix timestamp when stats were collected |

---

## Example Response

```json
{
  "num_running": 8,
  "num_waiting": 3,
  "num_active_decode_seqs": 6,
  "decode_ctx_mean": 512,
  "decode_ctx_p50": 480,
  "decode_ctx_p95": 1024,
  "decode_ctx_max": 1200,
  "total_decode_context_tokens": 3072,
  "pending_prefill_tokens": 1500,
  "pending_decode_tokens": 2400,
  "kv_cache_utilization": 0.65,
  "kv_free_blocks": 1400,
  "token_budget_per_iter": 4096,
  "max_num_seqs": 256,
  "num_preempted": 12,
  "timestamp": 1706123456.789
}
```

---

## Key Concepts

### `predicted_decode_tokens` Field

The API accepts an optional `predicted_decode_tokens` field in chat/completion requests. This provides a **soft estimate** of expected output tokens for scheduling purposes.

**Request Field (OpenAI API Compatible):**
```json
{
  "model": "llama-3",
  "messages": [...],
  "max_tokens": 1000,
  "predicted_decode_tokens": 200
}
```

| Field | Purpose |
|-------|---------|
| `max_tokens` | **Hard upper bound** - Request will stop at this limit |
| `predicted_decode_tokens` | **Soft estimate** - Used for load estimation only, does not affect generation |

**Fallback Behavior:**
- If `predicted_decode_tokens` is not provided, it defaults to `max_tokens`
- This ensures backward compatibility while allowing callers to provide more accurate estimates

### Auto-Extension Logic

When a request's actual output exceeds its predicted decode tokens, the scheduler automatically extends the prediction:

```
AUTO_EXTEND_BUFFER = 10

if num_output_tokens >= predicted_decode_tokens:
    predicted_decode_tokens = num_output_tokens + AUTO_EXTEND_BUFFER
```

**Why Auto-Extension?**
- Prevents `pending_decode_tokens` from becoming 0 or negative
- Provides a buffer to continue accurate load estimation
- Persists the update to handle ongoing decode

### Waiting Queue and Preempted Requests

The waiting queue contains two types of requests:

1. **New requests** - Just arrived, no output generated yet
2. **Preempted requests** - Were running but got preempted due to memory pressure

For preempted requests:
- `num_computed_tokens` is reset to 0 (will re-prefill)
- `num_output_tokens` is preserved (already generated tokens)

The `pending_decode_tokens` calculation accounts for this:

```python
# For each waiting request (including preempted):
remaining_decode = predicted_decode_tokens - num_output_tokens
pending_decode += remaining_decode
```

---

## Calculation Logic

### `pending_prefill_tokens`

```python
pending_prefill = 0

# Waiting requests: full prompt needs prefill
for req in waiting:
    pending_prefill += req.num_prompt_tokens

# Running requests: only remaining prompt tokens
for req in running:
    remaining = req.num_prompt_tokens - req.num_computed_tokens
    if remaining > 0:
        pending_prefill += remaining
```

### `pending_decode_tokens`

```python
pending_decode = 0

# Waiting requests (with auto-extension)
for req in waiting:
    if req.num_output_tokens >= req.predicted_decode_tokens:
        req.predicted_decode_tokens = req.num_output_tokens + AUTO_EXTEND_BUFFER
    remaining = req.predicted_decode_tokens - req.num_output_tokens
    pending_decode += remaining

# Running requests (with auto-extension)
for req in running:
    if req.num_output_tokens >= req.predicted_decode_tokens:
        req.predicted_decode_tokens = req.num_output_tokens + AUTO_EXTEND_BUFFER
    remaining = req.predicted_decode_tokens - req.num_output_tokens
    pending_decode += remaining
```

### `num_active_decode_seqs`

A request is in "decode phase" when it has completed prefill:

```python
decode_ctx_lengths = []
for req in running:
    if req.num_computed_tokens >= req.num_prompt_tokens:
        decode_ctx_lengths.append(req.num_computed_tokens)

num_active_decode_seqs = len(decode_ctx_lengths)
```

---

## Use Cases for Load Balancing

### 1. Least-Backlog Routing

Route new requests to the instance with the lowest pending work:

```python
def get_instance_load(stats):
    return stats["pending_prefill_tokens"] + stats["pending_decode_tokens"]

best_instance = min(instances, key=lambda i: get_instance_load(i.stats))
```

### 2. Memory-Aware Routing

Avoid routing to instances with high memory pressure:

```python
def is_instance_available(stats):
    return stats["kv_cache_utilization"] < 0.9  # 90% threshold

available = [i for i in instances if is_instance_available(i.stats)]
```

### 3. Capacity-Based Routing

Check if instance can accept more requests:

```python
def has_capacity(stats):
    return stats["num_running"] + stats["num_waiting"] < stats["max_num_seqs"]
```

### 4. Weighted Load Score

Combine multiple factors for sophisticated routing:

```python
def compute_load_score(stats):
    # Normalize each factor
    memory_pressure = stats["kv_cache_utilization"]
    queue_pressure = (stats["num_running"] + stats["num_waiting"]) / stats["max_num_seqs"]
    token_backlog = (stats["pending_prefill_tokens"] + stats["pending_decode_tokens"]) / stats["token_budget_per_iter"]

    # Weighted combination
    return 0.4 * memory_pressure + 0.3 * queue_pressure + 0.3 * token_backlog
```

---

## Implementation Notes

### vLLM Version Differences

| Aspect | v0 (0.6.x) | v1 (0.9+) |
|--------|------------|-----------|
| Endpoint | `/scheduler_stats` | `/instance_stats` |
| Request Class | `SequenceGroup` | `Request` |
| Memory Manager | `block_manager` | `kv_cache_manager` |
| Engine Architecture | `LLMEngine` | `EngineCore` (separate process) |

### Thread Safety

The stats collection is performed synchronously within the scheduler's context. The scheduler is single-threaded, so no additional locking is required.

### Performance Considerations

- Stats collection iterates through waiting and running queues
- Complexity: O(num_running + num_waiting)
- Lightweight operation suitable for frequent polling (e.g., every 100ms)

---

## Related Files

| File | Purpose |
|------|---------|
| `vllm/v1/core/sched/scheduler.py` | `get_aggregated_stats()` implementation |
| `vllm/v1/engine/async_llm.py` | Async wrapper for stats retrieval |
| `vllm/v1/engine/core_client.py` | Client interface for engine communication |
| `vllm/entrypoints/openai/api_server.py` | HTTP endpoint definition |
| `vllm/entrypoints/openai/protocol.py` | `predicted_decode_tokens` field definition |
| `vllm/v1/request.py` | Request class with prediction handling |

---

## Changelog

| Date | Change |
|------|--------|
| 2025-01-24 | Added auto-extension logic for `predicted_decode_tokens` |
| 2025-01-24 | Fixed waiting queue handling for preempted requests |
| 2025-01-24 | Initial implementation of `/instance_stats` endpoint |
