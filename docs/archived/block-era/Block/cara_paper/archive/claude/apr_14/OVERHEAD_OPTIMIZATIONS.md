# Apr 14 — Scheduler Overhead Optimizations

This doc captures the two optimizations landed today on the 4× A30 CloudLab cluster, plus the negative-result experiments that eliminated several "folk wisdom" options. Updates paper sections §4 (system design) + §5 overhead analysis.

## TL;DR

| Stage | total_sched_ms p50 (4 QPS) | Delta |
|-------|:-:|:-:|
| Baseline (pre-Apr-14) | 251.4 | — |
| + busy-signal fix (Task #20) | 180.2 | −28% |
| + SDPA + torch.compile (Task #22) | **161.5** | **−36% cumulative** |

Both are now DEFAULTS in the codebase. Orthogonal, both stack.

Measured @ 4 QPS × 50 reqs on `cluster = 2×Qwen-2.5-3B + 2×Qwen-2.5-7B`, workload from `data/cara/best-route-v3-test-500.jsonl`, filter=`cara_tiered`, same before/after.

## 1. Busy-Signal Fix (`cara_serve.py` `BatchQueue.adapt_params`)

### Bug

Prior `adapt_params()` (cara_serve.py:184-208) defined:
```python
busy = sum(1 for inst if stats.get("num_running", 0) > 0)
ratio = busy / len(all_instances)
```

`num_running > 0` is trivially true on any serving cluster — once ANY request is in flight on each instance (which at serve-time is always), `ratio ≈ 1.0` → adaptive policy jumps to overload preset:

```
busy_ratio < 0.3  → max_batch=4,  timeout=50ms   (idle)
busy_ratio 0.3-0.7 → max_batch=16, timeout=100ms  (medium)
busy_ratio > 0.7  → max_batch=32, timeout=200ms  (overload)
```

Result: at 4 QPS with 4 instances each holding exactly 1 request, ratio = 1.0 → 200 ms timeout → every batch waits ~100 ms before flushing, even though the cluster is far from saturated.

### Fix (committed in `block/global_scheduler/cara/cara_serve.py`)

```python
pressures = []
for inst in all_instances:
    s = stats_map.get(inst._instance_id, {}) or {}
    max_seqs = max(1, s.get("max_num_seqs", 256))
    waiting = s.get("num_waiting", 0)
    running = s.get("num_running", 0)
    seq_pressure = max(waiting, max(0, running - max_seqs / 2)) / max_seqs
    kv_pressure = float(s.get("kv_cache_utilization", 0.0))
    pressures.append(min(1.0, max(seq_pressure, kv_pressure)))
ratio = sum(pressures) / len(pressures)
```

`seq_pressure` is non-zero only when there's a queue backlog OR `num_running` exceeds half the instance's max_num_seqs (i.e. real capacity pressure). `kv_pressure` adds memory-pressure signal. Take the max per-instance, average across cluster.

### Measured

| 4 QPS, 50 reqs | batch_wait_ms p50 | total_sched_ms p50 | batch_size (avg) |
|---|:-:|:-:|:-:|
| Before | 111.9 | 251.4 | ~1 |
| After  | 50.9 | 180.2 | 1-4 (idle preset active) |

## 2. SDPA + `torch.compile` (default in `load_fused_model`)

### Setup

Original `MultiHeadEncoder.__init__` (`train_fused_bert_predictor.py`) hardcoded `attn_implementation="eager"` (the naive attention Python loop). That path is needed for some training configurations but the **trained weights are independent of attention kernel** — the state_dict is interchangeable with any valid `attn_implementation`.

### Fix (committed)

`train_fused_bert_predictor.py`:
- `MultiHeadEncoder.__init__` accepts new `attn_impl: str = "eager"` kwarg. Default preserves training behavior.
- `MultiHeadEncoder.predict()` updated to use `self(ids, mask)` (triggers `__call__` → torch.compile wrapper) instead of `self.forward(ids, mask)` (bypasses wrapper).
- `load_fused_model()` accepts `attn_impl: str = "sdpa"`, `compile_mode: Optional[str] = "reduce-overhead"` — both defaulting to fast inference path. Apply `torch.compile(model, mode=compile_mode)` after load.

`block/predictor/cara/model_estimator.py`:
- `_predict_buckets_fused`, `_predict_buckets_fused_batch`, and `estimate_length` patched to call the fused model via `__call__` (`self._fused_bucket(**inputs)`) instead of `.predict(**inputs)`. `.predict()` on a torch.compile-wrapped module calls the inner module's forward, bypassing the compiled graph.

### Measured (controlled bench on idle node1, 5 reps × 60 iters, `torch.set_num_threads(16)`, paired t-test)

| Batch | A eager (median) | D sdpa+compile (median) | Speedup | p |
|:-:|:-:|:-:|:-:|:-:|
| 1 | 95.6 ms | 69.4 ms | **1.38×** | ≈ 0 |
| 4 | 226.9 ms | 181.9 ms | **1.25×** | ≈ 0 |
| 8 | 386.5 ms | 331.8 ms | **1.17×** | ≈ 0 |
| 16 | 1010.2 ms | 620.5 ms | **1.63×** | ≈ 0 |

Parity: max_abs_diff = 7.2e-6 across 3 random inputs.

### In-scheduler measured (4 QPS × 50 reqs, with busy-fix active)

| | estimator_ms p50 |
|---|:-:|
| eager | 99.7 |
| sdpa+compile | **87.1** |

In-scheduler speedup (~10%) is smaller than controlled bench (38% at batch=1) due to:
- async scheduler event loop overhead
- shape-varying prompts → recompile on each new `seq_len`
- thread contention (cara_serve + sidecar + stats fetch share CPU)

Still a statistically real reduction, and combined with busy-fix gets total overhead to 161.5 ms.

## 3. Negative Results (documented for reproducibility)

### ONNX Runtime (CPU)

Full dual-runtime path implemented and exported (`cara_paper/smoke_test_apr_13/scripts/export_fused_to_onnx.py`). Parity check passed (max_abs_diff 3.8e-5). **Benchmarked slower than PyTorch 2.9** across all batch sizes:

| Batch | PyTorch | ONNX | Speedup |
|:-:|:-:|:-:|:-:|
| 1 | 98 ms | 102 ms | 0.96× |
| 4 | 220 ms | 370 ms | 0.59× |
| 8 | 369 ms | 560 ms | 0.66× |
| 16 | 673 ms | 1056 ms | 0.64× |

Root causes:
1. **No fused Attention op** — `onnxruntime.transformers.optimizer(model_type="bert")` fused only `LayerNormalization`, not attention. ModernBERT's rotary position embeddings + alternating global/local attention don't match vanilla BERT attention pattern.
2. **Unfused attention = 4 separate matmul+softmax ops** — PyTorch 2.9 SDPA is a single fused kernel.
3. **MLAS (ORT default matmul) vs MKL-DNN (PyTorch default)** — ~10-20% slower on Intel CPU.
4. **Rotary PE ONNX explosion** — 10+ small ops per attention block × 22 blocks = ~220 extra dispatches.

### INT8 Dynamic Quantization via ONNX

Failed with shape-inference error on the fused per-model heads (`Inferred shape and existing shape differ in dimension 0: (768) vs (16)`). Fixable with manual head layer pre-processing but not pursued.

### `torch.compile` alone (eager attention)

0.97-1.02× — torch.compile needs SDPA to expose fusable patterns in the attention path. Without SDPA, compile has less to optimize over.

### SDPA alone (no compile)

~1.0× — HF's ModernBERT implementation falls back to eager when the attention mask is non-standard (which it often is with padded batches).

## 4. When ONNX Would Actually Win

Retained the ONNX export + dual-runtime infrastructure for:
- **GPU deployment** (CUDAExecutionProvider typically 2-4× faster than PyTorch on GPU due to cuBLAS + cuDNN fusion)
- **Vanilla encoder swap** (if someone replaces ModernBERT with BERT/RoBERTa, attention fusion works)
- **Rust-based serving** (e.g. vLLM Semantic Router style — ORT is the natural path)
- **Post-training quantization pipelines** (ONNX + TensorRT INT8/FP8 are well-supported when fused heads are pre-processed)

## 5. Apr 26 Implications

All optimizations now DEFAULT — no flag to enable. Apr 26 full-scale evaluation should:
1. Report the 161.5 ms scheduler overhead as the main-paper number (previously paper cited 60 ms which was a single-threaded bench, not in-scheduler).
2. Skip the "ONNX projected 4-5×" framing — it was speculative and didn't materialize for this model.
3. Keep the "intrinsic overhead per batch" framing: at batch_size=16 (medium load preset), amortized estimator = 87/16 ≈ 5.4 ms/request.
4. Per-request scheduling overhead at Apr-26 scale (18 nodes, batch up to 32) should be ~10-20 ms/request intrinsic + batch_wait (queue-dependent).

## Files changed

| File | Change |
|------|--------|
| `block/global_scheduler/cara/cara_serve.py:184-218` | `adapt_params` busy-signal fix |
| `block/predictor/cara/offline_training/train_fused_bert_predictor.py` | `MultiHeadEncoder` accepts `attn_impl`; `load_fused_model` defaults to sdpa+compile; `predict()` uses `__call__` |
| `block/predictor/cara/model_estimator.py` | Fused-model call sites use `__call__` directly (wrappers-friendly) |
| `block/exp/cara/setup.sh` | pip installs onnxruntime==1.23.2 + onnx==1.17.0 + onnxscript (for export infrastructure) |
| `cara_paper/smoke_test_apr_13/scripts/patch_nodes_runtime.sh` | same installs for running clusters |

## Reproducibility

```bash
# Variance-controlled bench (use an idle node)
ssh asdwb@d7525-10s10317.wisc.cloudlab.us "cd ~/Block && python3 \
  cara_paper/smoke_test_apr_13/scripts/bench_variants_v2.py \
  --model-dir ~/Block/models/cara/length_bucket/deploy \
  --threads 16 --iters 60 --reps 5"

# In-scheduler measurement (4 QPS × 50 reqs)
python3 block/exp/cara/p100_smoke_client.py \
  --host <node0-ip> --port 8200 \
  --num-prompts 50 --request-rate 4 \
  --dataset data/cara/best-route-v3-test-500.jsonl \
  --output /tmp/overhead_check.json
# Then parse scheduling_overhead_breakdown from each result record
```
