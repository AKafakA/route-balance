# CARA Implementation Plan

## Executive Summary

- The CARA skeleton is running: scheduler server, instance adapters (vLLM/Ollama), predictor sidecars, deployment and benchmarking. Recent fixes stabilized deployment, predictor lifecycle, training‑data collection, and reduced transient client resets.
- Next work centers on quality‑aware scheduling (use pre‑trained best‑route), a greedy mini‑batch scheduler, RSO/constraints, adaptive batching, and evolving the sidecar predictor from dummy to a learned model.
- Data pipeline updated: balanced dataset pool (ultrachat, squad, cnn_dailymail, ai2_arc, gsm8k, reward_bench, code_ultra_feedback, mix_instruct, beaver_tails), tokenizer‑based prompt filtering (no truncation), equal ratios by default, per‑run distribution reporting. 32B dropped due to repetition/cursor bug.

## Current Functionality

- Scheduler (`block/global_scheduler/cara/cara_serve.py`)
  - FastAPI on `/v1/completions`, vLLM‑compatible; routing across heterogeneous backends (vLLM, Ollama).
  - Queries per‑instance predictors in parallel; returns server E2E latency for overhead accounting.
  - Strategies: random, round‑robin. Health endpoint `/health`. Flushes predictors on shutdown.
- Instances
  - Base: `block/global_scheduler/cara/cara_instance/Instance.py` (predictor query + feedback, per‑request mapping).
  - vLLM: `.../vllm_instance.py` with SSE parsing, ttft/itl/usage extraction.
  - Ollama: `.../ollama_instance.py` with NDJSON streaming.
- Predictor Sidecar (`block/predictor/cara/cara_predictor_api_server.py`)
  - Endpoints: `/predict`, `/log_actual`, `/stats`, `/flush`; dummy predictor collects training data.
  - Fallback: collects data with minimal state when `/schedule_trace` is unavailable.
- Deploy & Bench
  - Multi‑host deploy (`block/exp/cara/deploy_cara.py`) with `--deploy-services` gating (model_instance/predictor), predictor process verification, safe cleanup.
  - Benchmark (`block/benchmark/cara/benchmark_serving.py`) integrates CARA endpoint; light retries for transient resets.

## Stability Fixes Implemented

- SSH deploy: anchored `pkill` patterns, predictor verification, service‑level gating via `--deploy-services`.
- Predictor config: filter unknown keys; avoid backend port collision; interpreter autodetect and override.
- Data collection: minimal state fallback, `/flush` endpoint, scheduler flush‑all on shutdown.
- Networking: keep‑alive 60s on server/predictor; client retries for ECONNRESET.
- Imports: lazy‑load heavy modules in `server_utils` to avoid sklearn for CARA.

## Gaps To Implement

- Greedy mini‑batch scheduler with configurable scoring.
- RSO parsing + constraint handling.
- Quality model integration (pre‑trained best‑route).
- Length estimation service (baseline acceptable initially).
- Learned performance predictor + periodic retraining & hot reload.
- Adaptive batching; optional beam search.

## Quality Model Integration (Best‑Route)

- Client module: `block/model_estimator/best_route_client.py`
  - `predict_quality(prompt: str, models: list[str], timeout_s: float) -> dict[str, float]`.
  - Config: `block/config/cara/estimator_config.json` (endpoint, timeout, top_k, cache_size, max_chars).
  - Caching: LRU keyed by (model, sha256(truncated_normalized_prompt)); TTL.
- Scheduler wiring (in `cara_serve.py` decision path):
  - Fetch quality scores in parallel to predictor queries; map instance→model names; robust timeouts.
  - On failure/timeout, fall back to per‑model priors from config.
- Score integration (normalized):
  - `score_ri = w_lat·P_ri + w_cost·C_ri − w_qual·Q_r,model(i)`.
  - Normalize P (seconds), C (token‑cost proxy), Q ([0,1]); weights in `block/config/cara/scheduler_config.json`.
- Latency controls:
  - Top‑K models per request to bound estimator calls.
  - Truncate prompts (max_chars); async with strict timeout.

## Greedy Mini‑Batch Scheduler

- New: `block/global_scheduler/cara/scheduler.py`
  - `select_assignment(minibatch, instances, P, Q, C, weights, constraints)`.
  - `score_fn(req, inst, P_ri, Q_r_model, C_ri, weights, constraints)`.
- Integration: in `cara_serve.py`, group arrivals into mini‑batches (size C, timeout); call scheduler; dispatch via `query_instance`.
- Beam variant (later): keep top‑K placements per mini‑batch.

## RSO & Constraints

- Request payload extension: `request_specific_objective` with `latency_ms`, `model_quality_min`, `token_budget`, `relax_order`.
- Parse in `cara_serve.py`; pass to `score_fn`.
- Start with soft penalties; optional hard filters. Add confidence bounds using predictor MAE/percentiles.

## Adaptive Batching

- Signals per instance: outstanding requests, rolling serving_time, free GPU blocks (if available).
- Policy: increase mini‑batch size and batching timeout under high load; keep small under low load.
- Expose CLI knobs and config defaults.

## Performance Predictor Roadmap

- Train: `block/predictor/cara/train_lstm.py` using JSONL from `training_data/cara` (target = server_latency or ttft+itl aggregate).
- Runtime: `block/predictor/cara/lstm_cara_predictor.py`; extend `LSTMPredictorConfig` (checkpoint path, device); add `/reload_model`.
- Feedback: periodic aggregation + training job; tag predictor versions in `/stats`.

## Length Estimation Service (Baseline)

- Service: `block/model_estimator/server.py` (`/predict_length`) with heuristic baseline; upgrade later.
- Client: `block/model_estimator/client.py`.
- Use lengths for cost proxy and capacity checks (KV estimate) in `score_fn`.

## Data Pipeline & Training

- Collection: keep `/flush`; add periodic auto‑save timer in `TrainingDataCollector`.
- Aggregation: `block/predictor/cara/tools/aggregate_training_data.py` to consolidate JSONL and trigger training.
- Governance: add `version` to `TrainingExample`; minimal tests for schema stability.

## Deployment & Ops

- Extend deploy to manage estimator service; gate via `--deploy-services`.
- Unify configs (scheduler/estimator/priors/cost) under `block/config/cara/`.
- Observability: add `/metrics` stubs for scheduler and predictors; structured logs with `request_id`, chosen instance, decision time.

### Experimental Setup Snapshot (Current)

- Models: Qwen 2.5 {3B, 7B, 14B, 72B}; 32B excluded (repetition bug).
- Instance mix: 72B×2, 14B×3, 7B×5, 3B×8.
- CloudLab mapping:
  - 72B on d8515 (A100×4): TP=4 per node.
  - 14B on c4130 (V100×4): 1 per node, TP/precision per VRAM.
  - 7B on 5×d7525 (A30×1 each).
  - 3B on 3×d7525 (A30×1) + 5×c240g5 (P100×1, vLLM 0.4 w/o FlashAttention).

## Testing & Validation

- Unit: `score_fn`, batch partitioning, RSO parsing, estimator client (timeouts/caching).
- Integration: random dataset runs across QPS; verify predictor calls, training files, overhead trends.
- A/B: random vs round_robin vs greedy+quality across 3B/7B/14B/32B mix; report p50/p95 latency, throughput, and quality‑weighted throughput (ΣQ/sec).

## Milestones

- v0.1 (Greedy Core, 3–5 days)
  - `scheduler.py` (greedy), `score_fn`, C/timeout knobs, RSO parsing (soft penalties), feature‑flag `--scheduling greedy`.
- v0.2 (Quality Integration, 3–5 days)
  - `best_route_client`, caching + priors, `scheduler_config.json` weights; initial calibration.
- v0.3 (Predictor v1, 1–2 weeks)
  - LSTM predictor train/infer + hot reload; periodic training + artifact versioning.
- v0.4 (Adaptive/Beam, 1–2 weeks)
  - Adaptive batching; optional beam; confidence‑bounded scoring; expanded observability.

## Open Questions

- best‑route interface: transport (HTTP/gRPC), prompt normalization (chat formatting), per‑model name mapping, latency budget (target p95).
- Quality scale calibration: raw logits vs calibrated probabilities; per‑model normalization.
- Cost model: per‑instance GPU cost factors, token price proxy, or hybrid.
- Length estimation ownership: rely on best‑route or keep a separate lightweight estimator.

## Immediate Next Steps (Suggested)

- Scaffold `scheduler.py`, add `scheduler_config.json`, wire `--scheduling greedy` into `cara_serve.py` with a simple score_fn that consumes latency (P) now; stub hooks for quality (Q) and cost (C).
- Add `best_route_client.py` and `estimator_config.json`; integrate async quality fetch (behind a feature flag), with caching and priors fallback.
- Add basic adaptive batching knobs in `cara_serve.py` (C and timeout based on outstanding requests per instance).
