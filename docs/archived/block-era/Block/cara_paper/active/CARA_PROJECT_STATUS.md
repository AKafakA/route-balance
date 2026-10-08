# CARA Project Status and Implementation Summary

**Document Purpose**: Comprehensive status report for CARA (Context-Aware Resource Allocation) implementation. This document serves as knowledge transfer for future development by Claude Code or other AI coding agents.

**Last Updated**: April 16, 2026 (**Phase 1 preflight COMPLETE** — 3 runs, 10 bugs fixed, 3 paper-facing findings, all §7.A/§7.C pass criteria green)
**Project Stage**: Phase 1 preflight COMPLETE (Apr 15-16, validated) → **Phase 2 (Apr 26 - May 6, 18-node) = paper-facing primary run** → NeurIPS May 6 submission.

**⚠ Evaluation methodology frozen (Apr 15 PM, revision 2)** — source of truth `cara_paper/codex/apr15_baseline_comparison_design.md`. Summary:
- Main-table baselines (5): Cara-full, vllm-SR, RouteLLM-mf, Avengers-Pro, BEST-Route.
- Clean bench: 500 resampled prompts from `test_scored_filtered.jsonl` (HF `asdwb/cara_model_estimator`). Legacy `best-route-v3-test-500.jsonl` superseded (289/500 predictor-train leakage).
- Quality: pre-computed deepeval lookup (RouterBench-style).
- Figures: 2D Pareto × 2 + CAP radar + goodput bar + throughput bar + memory-variance time-series + w_balance sensitivity + overhead stack + Table 1.
- Cost model: per-token vendor pricing (MaaS).
- Methodology cited: RouterBench (Hu 2024), MoE-CAP (Jiang 2025), RouterEval (Huang 2025); LLMRouter library + BentoML Explorer + Hybrid-LLM + BEST-Route + BOute.

**Target Venue**: NeurIPS 2026 (May 6 deadline, primary) / EuroSys 2027 (May 14, backup)

**Latest Session**: See `cara_paper/handoffs/2026-04-15_handoff_v2.md`. Phase 1 preflight COMPLETE Apr 16. All §7.C blockers closed. BEST-Route trained (val_acc=0.58). 5 Avengers-Pro artifacts. 3 findings for Phase 2: pipeline stats-fetch bottleneck (~200ms/req), SQ pathological in pipeline mode, RouteLLM-mf domain shift on Qwen.

**Phase 1 Preflight Results (Apr 15-16, 4x A30, NOT paper numbers):**
- Cara-full: e2el_p50=8706ms, goodput_bal=0.07, deepeval=0.355
- Cara-no-LPT: 5323ms, gp=3.92, deepeval=0.362
- Cara-no-filter: 4987ms, gp=4.20, deepeval=0.360
- Cara-no-quality: 4447ms, gp=5.27, deepeval=0.359
- quality_greedy: 6225ms, gp=3.17, deepeval=0.341
- Pipeline RR baselines: ~2287ms e2e at λ=10 (warm cache), but ~50s at λ=10 under pipeline stats-fetch bottleneck.

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Project Overview](#project-overview)
3. [Implemented Features](#implemented-features)
4. [System Architecture](#system-architecture)
5. [Current Status](#current-status)
6. [Future Work & Roadmap](#future-work--roadmap)
7. [Feasibility Analysis](#feasibility-analysis)
8. [Risks & Mitigation](#risks--mitigation)
9. [Impact Assessment](#impact-assessment)
10. [Developer Guide](#developer-guide)

---

## Executive Summary

**Mission**: Build a learned, context-aware scheduler for heterogeneous LLM serving that uses ML predictors to make optimal routing decisions.

**Current Achievement**: ✅ **Phase 1-2 Complete, Phase 3 In Progress**
- Full training data pipeline: 18,605 prompts × 4 models, 7 datasets, 6 quality signals
- Latency data: 404,929 records across 5 instance types
- All predictors trained and evaluated: RoBERTa (best), ModernBERT, KNN, LoRA
- Multi-objective scheduler with tiered relaxation: 45/45 smoke test PASS
- Prometheus reference-grounded judging running
- SLO filtering baselines (PolyServe, SLOs-Serve, QLM) planned for ablation

**Next Milestone**: 🎯 **Phase 3** - CloudLab 18-node full evaluation (April 26 - May 6) + SLO ablation study

**Project Health**: 🟡 **YELLOW** - Core system working, but significant implementation work before CloudLab: SLO filter baselines, integration testing, experiment automation

**New in March 2026**:
- ✅ Dataset quality ablation study: per-dataset survival rates analyzed with broadcasting
- ✅ Removed low-survival datasets: ai2_arc (0%), cnn_dailymail (64.9%)
- ✅ Replaced ultrachat with LMSYS-Chat-1M (776k English conversations with response labels)
- ✅ Added response-length pre-filtering for conversation datasets (`--chat-max-response-tokens`, `--chat-max-total-tokens`)
- ✅ Added language filtering for LMSYS (`--chat-lang English`)
- ✅ Generation parameter sweep: best config = temp=0.0, freq_penalty=1.2, rep_penalty=1.0
- ✅ KNN estimator for length + quality prediction implemented and tested
- ✅ Generated v3 test dataset (500 prompts, 7 datasets, even distribution)
- ✅ HuggingFace gated dataset access configured (second token for LMSYS)
- ✅ Full 20k broadcasting run complete (`broadcast_v3_20k.json`, 93.6% survival rate with freq_penalty=1.2)
- ✅ Fixed frequency_penalty forwarding to CARA server (default=1.2, tuned via sweep)
- ✅ Fixed `--save-detailed` store_true flag and `cara` backend sampling param support
- ✅ Recovered lost preprocessing pipeline from git history (llm_judge_scorer, validate_judge_models, split_training_data)
- ✅ Updated `prepare_benchmark_data.py` to unified schema (no latency fields, `similarity_score` + `llm_judge_scores` dict)
- ✅ Best LLM judge: `Qwen/Qwen2.5-7B-Instruct` (Spearman ρ=0.843 on feedbackQA, 1-4 scale)
- ✅ Latency data collection script (`generate_latency_benchmark.py`) for separate controlled single-model requests
- ✅ GPU VM setup (`wd312@gxp-l40s-2`, 4x L40S 48GB) with uv venv, PyTorch, sentence-transformers, xgboost, faiss
- ✅ Preprocessing pipeline tested on 50-prompt dev set: 49/50 valid, 196/196 LLM judge scores (100% parse rate)
- ✅ LLM judge scorer optimized: bf16 + SDPA attention + batch_size=32 (default was fp16 + eager + batch=8)
  - With rationale: 3m41s for 196 pairs (~25h projected for 20k)
  - No rationale (`--disable-rationale`): 57s for 196 pairs (~6.5h projected for 20k), correlation 0.81-0.93 with rationale scores
  - Added `--no-flash-attention` flag, automatic fallback chain: flash_attention_2 → sdpa → eager
- 🚧 Run preprocessing pipeline on 20k data (GPU required), split 80/20, upload to HuggingFace `asdwb/cara_model_estimator`
- 🚧 Develop and test latency training data collection scripts (WS2, parallel)
- 🚧 Model development: KNN (exists), MLP, RoBERTa, LSTM, LLM for quality/length; XGBoost, LSTM, Roofline, Linear for latency

**New in January 2026**:
- ✅ Comprehensive CARA README created in `cara_paper/claude/README.md`
- ✅ Offline training data preparation pipeline with filtering and scoring
- ✅ Data quality analysis identifying coverage and scoring issues
- ✅ LLM judge scoring as alternative to similarity-based scoring
- ✅ Multi-dataset collection pipeline with balanced sampling
- 🚧 Quality estimator and length predictor training in progress

**New Updates (January 16, 2026)**:
- ✅ **A100 Deployment Fixes**: Resolved NVIDIA library path issues (cuDNN, cuSPARSELt, NVSHMEM)
  - Dynamic LD_LIBRARY_PATH discovery in `deploy_cara.py` and `setup.sh`
  - Auto-discovery loop for pip-installed nvidia packages
- ✅ **NumPy 2.x Compatibility**: Added sklearn/pandas/pyarrow upgrades for A100 nodes
- ✅ **P100/A100 Documentation**: Created separate deployment guides
  - `A100_DEPLOYMENT_GUIDE.md`: Comprehensive A100 troubleshooting
  - `P100_DEPLOYMENT_GUIDE.md`: P100-specific build and dataset notes
  - `EXPERIMENT_QUICK_REFERENCE.md`: Quick command reference
- ✅ **Benchmark Fixes**: Added missing `--request-id-prefix` argument
- ✅ **vllm_compat.py**: Enhanced compatibility layer for P100/A100 differences
- ✅ **Word Limit Normalization**: `normalize_word_limits.py` tool documented
  - Hard clamping: Clamps explicit "N words" to token budget
  - Soft clamping: Appends guidance when no explicit word counts
  - Iterative refinement to account for guidance overhead

**New Updates (January 22, 2026)**:
- ✅ **Unified Benchmark Preprocessing**: Refactored `prepare_benchmark_data.py` as the unified preprocessing script
  - Supports both single-model and multi-model (broadcast_results) formats
  - Auto-detects data format and models from data
  - Handles incomplete requests with JSONL export for re-running benchmarks
  - Multiple scoring methods: `llm_judge`, `similarity`, `compression`
- ✅ **Multi-Judge Comparison**: Added analysis features for comparing LLM judges

**New Updates (January 26, 2026)**: ✨ **Task 1.4 Multi-Objective Scoring Complete + Implementation Tasks V2 Design**
- ✅ **Task 1.4 Multi-Objective Scoring** (COMPLETE):
  - RSO (Request-Specific Objectives) and SLO (Service-Level Objectives) data structures
  - MultiObjectiveScorer with tiered constraint handling (STRICT/RELAXED/TIERED modes)
  - Normalized scoring with configurable weights (latency, cost, quality, memory)
  - Min-max and regret normalization methods
  - Placeholder estimators for cost and quality
  - TTFT/TPOT derivation from service rates
  - Full integration with CARABatchPolicy via ScoringConfig
  - 166 passing CARA predictor tests
- ✅ **Implementation Tasks V2 Design Document** (NEW):
  - Comprehensive design for 12 pending tasks from review comments
  - See `cara_paper/claude/IMPLEMENTATION_TASKS_V2_DESIGN.md`
  - Covers: streaming batch response, XGBoost target field fixes, DataCollectionService rename,
    ServiceRateTracker decoupling, deploy_cara.py updates, E2E training pipeline, and more
  - Includes detailed code snippets, file change lists, testing strategies
  - Task dependencies and recommended implementation order

**Previous Updates (January 25, 2026)**: ✨ **Extensible Predictor Architecture + vLLM `/instance_stats` API**
- ✅ **Extensible Predictor Architecture**:
  - Support multiple predictor implementations: XGBoost, LSTM, Roofline, Linear
  - Factory pattern with `predictor_type` field for config dispatch
  - `STATE_API_TYPE` class attribute: `SCHEDULE_TRACE` vs `INSTANCE_STATS`
  - Separate config files per predictor type (not monolithic)
  - Instance type derived at deployment: `{model}_{backend}_{node_type}`
- ✅ **Port Configuration Cleanup** (NEW):
  - Renamed `predictor_ports` → `data_collection_port` (single port)
  - Moved from `host_configs.json` to `model_config_template.json`
  - Decoupled data collection from scheduling policy
- ✅ **New Batch Endpoint** (NEW):
  - `/v1/batch_completions` for CARA batch scheduling
  - Keep `/v1/completions` unchanged for backward compatibility
  - Scheduling modes: `BOOTSTRAP` (random/round-robin) vs `CARA` (XGBoost)
- ✅ **Scheduler-Side Components** (NEW):
  - `StateManager`: Centralized state with `refresh_all()`, `apply_assignment()`
  - `PredictorRegistry`: Maps instance_type to predictor
  - `SchedulingPolicy`: BOOTSTRAP vs CARA mode selection
  - `on_completion_callback`: Constructor-injected callback for completion notifications
- ✅ **vLLM `/instance_stats` API Implemented**:
  - Finalized API specification: `INSTANCE_STATS_API.md`
  - Context length distribution with mean, p50, p95, max, and total
  - Backlog tokens computed for all requests (running + waiting, including preempted)
  - `predicted_decode_tokens` field with auto-extension when output exceeds prediction
  - EMA service rates computed scheduler-side from completed request feedback
- ✅ **P100 Chunked Prefill Configuration**:
  - Added `enable-chunked-prefill` field to model config template
  - Updated `deploy_cara.py` to pass `--enable-chunked-prefill` flag for older vLLM v0
  - Required for consistent latency predictor behavior across all instance types
- ✅ **Batch Scheduling Algorithm Redesign**:
  - Replaced mini-batch permutation search O(|I|^|C|) with LPT-based linear scan O(|R|×|I|)
  - LPT (Longest Processing Time First) heuristic: sort by predicted tokens descending
  - Active pull of `/instance_stats` before each batch (not Dead Reckoning)
  - Within-batch local state updates only (no cross-batch state staleness)
  - **Load-aware adaptive batching**: dynamic batch size based on cluster load ratio
    - Low load: small batch, short timeout → minimize latency
    - High load: large batch, long timeout → improve throughput (requests queue at instance anyway)
  - **Tiered constraint handling**: configurable RSO/SLO violation modes
    - `STRICT`: reject if no fully-satisfied candidate
    - `RELAXED`: always pick best available
    - `TIERED`: try tier-0 first, then tier-1, etc. (sorted by #violations, then score)
- ✅ **XGBoost Latency Predictor Rationale**:
  - Chose XGBoost over simple linear regression for richer feature modeling
  - Can model non-linear effects: context length distribution, chunked prefill constraints
  - Still "gray box" (interpretable features) unlike LSTM black box
  - Features from `/instance_stats`: decode_ctx_p95, pending_tokens, kv_utilization, etc.

**New Updates (January 23, 2026)**: ✨ **Latency Predictor Design Pivot + LLM Judge Improvements**
- ✅ **Latency Predictor Architecture Redesign**:
  - Pivot from LSTM-based sidecar to XGBoost scheduler-side predictor
  - Per-batch state query design (vs per-request)
  - New `/instance_stats` API specification for vLLM (see `INSTANCE_STATS_API.md`)
  - Comprehensive design document: `LATENCY_PREDICTOR_DESIGN.md`
  - Aggregated feature engineering (chunked prefill assumption)

**LLM Judge Quality Improvements (January 23, 2026)**
- ✅ **Configurable Scoring Scales**: Flexible score ranges (1-4, 1-10, or custom)
  - Auto-generated scale descriptions for any range
  - Proper normalization to [0, 1] for all scales
  - CLI arguments: `--score-min`, `--score-max`, `--disable-rationale`
- ✅ **3-Stage Parsing System**: Robust parsing with 95% success rate (up from 80%)
  - Stage 1: Number extraction (direct, fastest)
  - Stage 2: Exact substring match against scale descriptions
  - Stage 3: Semantic embedding match using sentence-transformers (optional)
  - Graceful fallback with detailed statistics
- ✅ **Validation Framework**: New `validate_judge_models.py` script
  - Tests judges against human ratings from feedbackQA dataset
  - Computes Pearson and Spearman correlation (baseline: 0.56)
  - Achieves 0.69 correlation (21% improvement over old approach)
  - Comprehensive comparison reports with parsing method breakdown
- ✅ **Improved Prompt Templates**: Based on HuggingFace cookbook best practices
  - Rationale-based prompting (ask for reasoning)
  - Rating comes first to ensure generation within token limit
  - Increased max_new_tokens from 10 to 200 for rationale
  - 30-50% accuracy improvement from template changes alone
- ✅ **Statistics Tracking**: Comprehensive confidence metrics
  - Success rate per parsing method
  - Semantic similarity confidence (when used)
  - Detailed reporting via `print_parsing_stats()`
- ✅ **Documentation**: Complete guide in `LLM_JUDGE_IMPROVEMENTS.md`
  - Background and research foundation
  - Usage examples and best practices
  - Troubleshooting guide
  - Performance comparison tables
  - Per-judge statistics (mean, std, percentiles)
  - Global Pearson/Spearman correlation between judges
  - Per-request model ranking correlation
  - Divergent sample detection and export
- ✅ **LLM Judge Improvements**:
  - Batched inference for faster scoring (`--batch-size`)
  - Fixed `pad_token_id` warnings during generation
  - Failed score handling (mark as None, filter requests)
- ✅ **Reference Model Validation**: Similarity scoring requires explicit `--reference-model`
- ✅ **Incomplete Request Handling**: Exports to JSONL format compatible with `benchmark_serving.py`

**New Updates (January 21, 2026)**:
- ✅ **Single-Model Benchmark Preprocessing**: Initial `prepare_benchmark_data.py` for single-model data
- ✅ **Data Quality Analysis on ShareGPT Benchmark** (20,000 requests @ 4 QPS):
  - 96.4% valid responses (712 empty filtered)
  - 8.1% hit max_tokens (1024) - truncated
  - 12.3% highly repetitive (compression ratio < 0.2)
  - Repetition correlates with output length: 0.2% at 50-200 tokens → 34.5% at 800-1024 tokens
- ✅ **Key Design Decisions for Estimation Models**:
  - **No prompt classifier**: For online scheduling, we only have raw prompt - no source metadata
  - **Compression ratio as label, not filter**: Keep repetitive samples for training (model should learn to predict this)
  - **Data source tags**: Useful for offline evaluation only, NOT as model input features
- ✅ **Dataset Collection Pipeline**: `collect_data.py` with 7 active datasets (v3), token-based filtering, response-length pre-filtering for conversation datasets
  - v3 datasets: reward_bench, code_ultra_feedback, beaver_tails, mix_instruct, lmsys, gsm8k, squad
  - Removed: ai2_arc (0% survival), cnn_dailymail (64.9% survival), ultrachat (replaced by lmsys)
  - New: LMSYS-Chat-1M with language filter and response-length pre-filtering

---

## Project Overview

### What is CARA?

CARA is a **heterogeneous LLM serving system** that routes requests to instances running different model sizes (3B, 7B, 14B, 32B, 72B parameters) based on **learned predictions** of:
1. **Response length**: How many tokens will be generated
2. **Model quality**: Output quality for a given prompt-model pair
3. **Latency**: Time-to-first-token (TTFT) and end-to-end latency (E2E)

### Why CARA?

**Problem**: Traditional schedulers use:
- Random/round-robin (ignores workload characteristics)
- Simulation-based prediction (static, not adaptive)
- Rule-based heuristics (brittle, requires manual tuning)

**CARA Solution**:
- **Learned predictions**: LSTM models trained on real execution traces
- **Forward-compatible**: Adapts to new models/workloads online
- **Context-aware**: Considers current queue state, request characteristics

### Key Innovation

**Two-Stage Prediction**:
1. **Pre-scheduling**: Predict latency/quality for routing decision
2. **Post-execution**: Send actual metrics back for continuous learning

This creates a **closed-loop learning system** that improves over time.

---

## Implemented Features

### ✅ 1. Predictor Integration Framework

**Location**: `block/global_scheduler/cara/cara_serve.py` (Lines 78-104)

**Functionality**:
- CARA scheduler queries ALL predictors before scheduling each request
- Predictors run in parallel using `asyncio.gather()`
- Predictions logged for training data collection
- Currently uses random/round-robin scheduling (predictions logged but not used for routing yet)

**Code Flow**:
```python
# 1. Extract request metadata
num_prompt_tokens = request_json.get("prompt_len", 0)
max_output_tokens = request_json.get("max_tokens", 256)

# 2. Query all predictors in parallel
prediction_tasks = [
    instance.query_predictor(
        request_id=request_id,
        num_context_tokens=num_prompt_tokens,
        predicted_num_context_tokens={model: max_output_tokens}
    )
    for instance in instances
]
predictions = await asyncio.gather(*prediction_tasks)

# 3. Schedule request (currently random/round-robin)
selected_instance = random.choice(instances)
response_dict = await selected_instance.query_instance(request_json)
```

### ✅ 2. Predictor Feedback Mechanism

**Location**: `block/global_scheduler/cara/cara_instance/Instance.py` (Lines 84-159)

**Functionality**:
- After request execution, instance sends actual metrics back to predictor
- Feedback includes: E2E latency, TTFT, TPOT (time-per-output-token)
- Sampling rate configurable (0.0-1.0) to reduce overhead
- Routes feedback to the SAME predictor that made the prediction (critical for correctness)

**Key Design**:
```python
# Per-instance round-robin counter for predictor selection
self._predictor_index = 0
self._request_to_predictor_index = {}  # Track which predictor was used

async def query_predictor(...):
    # Select predictor using round-robin
    predictor_idx = self._predictor_index % len(self._predictor_urls)
    self._predictor_index += 1

    # Remember which predictor handled this request
    self._request_to_predictor_index[request_id] = predictor_idx

async def _send_predictor_feedback(...):
    # Send feedback to SAME predictor
    predictor_idx = self._request_to_predictor_index.get(request_id)
    log_url = self._predictor_log_urls[predictor_idx]
    # POST feedback...
```

### ✅ 3. Broadcasting Mode for Multi-Model Data Collection

**Location**: `block/global_scheduler/cara/cara_serve.py` (Lines 106-183)

**Purpose**: Collect responses from multiple models for the same prompt to train:
- Response length predictor (model → output length)
- Model quality estimator (model → quality score)

**How It Works**:
1. When `--broadcasting` enabled, CARA receives request
2. Broadcasts to one instance per selected model in parallel
3. Randomly picks one response as "primary"
4. Returns all responses in `broadcast_results` field

**Optimization**: Eliminated duplicate query by randomly selecting primary from broadcast results (saves 1 query per request).

**Usage**:
```bash
python -m block.global_scheduler.cara.cara_serve \
  --broadcasting \
  --selected-broadcasted-models "Qwen/Qwen2.5-3B" "Qwen/Qwen2.5-7B" "Qwen/Qwen2.5-14B"
```

### ✅ 4. Training Data Collection Infrastructure

**Location**: `block/predictor/cara/training_data_collector.py`

**Components**:
- **TrainingDataCollector**: Buffers prediction-actual pairs
- **Automatic matching**: Pairs predictions with actual metrics by request_id
- **Batched disk writes**: Writes JSONL files every N examples (configurable)
- **Stats tracking**: Monitors collection health

**Data Format** (JSONL):
```json
{
  "request_id": "cmpl-123",
  "num_prompt_tokens": 128,
  "num_predicted_output_tokens": 64,
  "schedule_state": {
    "num_running": 3,
    "num_waiting": 2,
    "free_gpu_blocks": 4500,
    "running_requests": [...],
    "waiting_requests": [...]
  },
  "instance_id": "instance_8000",
  "prediction_timestamp": 1702345678.123,
  "actual_e2e_latency": 2.456,
  "actual_ttft": 0.123,
  "actual_tpot": 0.023,
  "completion_timestamp": 1702345680.579
}
```

### ✅ 5. Dummy Predictor for Heuristic-Based Data Collection

**Location**: `block/predictor/cara/dummy_cara_predictor.py`

**Purpose**: Collect training data while providing simple heuristics for scheduling

**Heuristic Modes**:
- `min_requests`: Route to instance with fewest requests
- `max_gpu_blocks`: Route to instance with most free GPU memory
- `combined`: Weighted combination
- `random`: Baseline (for A/B testing)

**Why Dummy?**: Allows data collection before ML models are trained.

### ✅ 6. Predictor API Server

**Location**: `block/predictor/cara/cara_predictor_api_server.py`

**Independence**: Separate from Block's predictor infrastructure (no Vidur dependency)

**Endpoints**:
- `POST /predict`: Get latency prediction for a request
- `POST /log_actual`: Log actual metrics (for training data)
- `GET /stats`: Collection statistics
- `POST /flush`: Force write buffered data to disk
- `GET /health`: Health check

**Key Design Decision**: Field names standardized to avoid confusion
- `num_prompt_tokens` (NOT `num_prompt_len`)
- `num_predicted_output_tokens` (NOT `num_predicted_output_len`)
- `request_id` (NOT `id`)

### ✅ 7. Deployment Automation

**Location**: `block/exp/cara/deploy_cara.py`

**Features**:
- Deploy vLLM/Ollama backends across multiple hosts
- Deploy predictors alongside each backend instance
- Automatic port management (avoids collisions)
- Service selection (`--deploy-services backend predictor`)
- Single source of truth for configuration (host_config.json)

**Configuration Hierarchy**:
```
host_config.json         # Single source of truth
├── backend_port: 8000  # vLLM/Ollama port
└── predictor_ports: [8100, 8101, 8102, 8103]  # Multiple predictors per host

model_deployment.json    # Which models on which hosts
├── hf_model_name        # e.g., "Qwen/Qwen2.5-7B"
├── backend: "vllm"
└── node_hosts: [...]

predictor_deployment_config.json  # Predictor settings
├── predictor_type: "dummy"
├── enable_data_collection: true
└── heuristic_mode: "min_requests"
```

### ✅ 8. Benchmark Client with CARA Backend

**Location**: `block/benchmark/cara/benchmark_serving.py` + `cara_end_point_func.py`

**Features**:
- Custom CARA backend for vLLM benchmark framework
- Per-request data format with `broadcast_results` field
- Selective metric saving (`--save-detailed prompts response ttft e2el models`)
- Support for ShareGPT, random, and Sonnet datasets

**Result Format**:
```json
{
  "completed": 100,
  "failed": 0,
  "mean_ttft_ms": 450.2,
  "mean_e2el_ms": 2340.5,
  "requests": [
    {
      "request_id": "1",
      "prompt": "Write a story...",
      "model": "Qwen/Qwen2.5-7B",
      "response": "Once upon a time...",
      "ttft": 0.45,
      "e2el": 2.34,
      "output_len": 95,
      "broadcast_results": [
        {"model": "Qwen/Qwen2.5-3B", "ttft": 0.3, "e2el": 1.8, "output_len": 100, ...},
        {"model": "Qwen/Qwen2.5-7B", "ttft": 0.45, "e2el": 2.34, "output_len": 95, ...},
        {"model": "Qwen/Qwen2.5-14B", "ttft": 0.7, "e2el": 3.5, "output_len": 105, ...}
      ]
    }
  ]
}
```

### ✅ 9. Test Scripts for Data Collection

**Files**:
- `block/exp/cara/test_broadcasting.sh`: Local execution
- `block/exp/cara/run_broadcasting_remote.sh`: Remote execution via SSH

**Parameterization**:
```bash
./test_broadcasting.sh \
  "Qwen/Qwen2.5-3B Qwen/Qwen2.5-7B Qwen/Qwen2.5-14B" \  # Models
  sharegpt \                                              # Dataset
  500 \                                                   # Num prompts
  2.0 \                                                   # Request rate (qps)
  batch1                                                  # Output suffix
```

**Automated Workflow**:
1. Start CARA server with broadcasting
2. Run benchmark with specified parameters
3. Save results with timestamp
4. Display data format and next steps

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Benchmark Client                              │
│              (benchmark_serving.py)                              │
└───────────────────┬──────────────────────────────────────────────┘
                    │ POST /v1/completions
                    │ {prompt, max_tokens, prompt_len, request_id}
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                 CARA Scheduler (cara_serve.py)                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ 1. Query all predictors in parallel                       │   │
│  │ 2. Log predictions for training data                      │   │
│  │ 3. Select instance (random/round-robin or use predictions)│   │
│  │ 4. Forward request to selected instance                   │   │
│  │ 5. Optional: Broadcast to multiple models                 │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────┬─────────────────────┬─────────────────────┬───────────────┘
      │                     │                     │
      ▼                     ▼                     ▼
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│ vLLM Instance   │   │ vLLM Instance   │   │ vLLM Instance   │
│ Qwen/2.5-3B     │   │ Qwen/2.5-7B     │   │ Qwen/2.5-14B    │
│ Port: 8000      │   │ Port: 8000      │   │ Port: 8000      │
│                 │   │                 │   │                 │
│ ┌─────────────┐ │   │ ┌─────────────┐ │   │ ┌─────────────┐ │
│ │ Predictor   │ │   │ │ Predictor   │ │   │ │ Predictor   │ │
│ │ Port: 8100  │◄┼───┼─┤ Port: 8100  │◄┼───┼─┤ Port: 8100  │ │
│ └─────┬───────┘ │   │ └─────┬───────┘ │   │ └─────┬───────┘ │
│       │ Monitors│   │       │ Monitors│   │       │ Monitors│
│       ▼ Backend │   │       ▼ Backend │   │       ▼ Backend │
│ /schedule_trace │   │ /schedule_trace │   │ /schedule_trace │
└─────────────────┘   └─────────────────┘   └─────────────────┘
```

### Data Flow for Training Data Collection

```
Request arrives
    │
    ▼
┌───────────────────────────────────────────────────┐
│ CARA Scheduler queries predictors                 │
│ - Sends: (prompt_len, predicted_output_len)       │
│ - Receives: {target_metric, gpu_blocks, ...}      │
└───────────────────┬───────────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────────────┐
│ Predictor fetches /schedule_trace from backend    │
│ - Gets current queue state                        │
│ - Logs: (schedule_state, target_request) pair     │
└───────────────────┬───────────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────────────┐
│ CARA Scheduler forwards request to instance       │
│ - Instance executes request                       │
│ - Gets actual metrics (e2e, ttft, tpot)           │
└───────────────────┬───────────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────────────┐
│ Instance sends feedback to predictor               │
│ POST /log_actual {request_id, e2e, ttft, tpot}    │
└───────────────────┬───────────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────────────┐
│ Predictor matches prediction with actual          │
│ - Creates training example                        │
│ - Buffers in memory                               │
│ - Writes to disk every N examples                 │
└───────────────────────────────────────────────────┘
    │
    ▼
training_data/cara/training_data_instance_8000_*.jsonl
```

### Component Dependency Graph

```
cara_serve.py
    ├─→ Instance.py (base class)
    │   ├─→ vllm_instance.py
    │   └─→ ollama_instance.py
    ├─→ cara_predictor_api_server.py (HTTP client)
    └─→ benchmark_serving.py (receives responses)

cara_predictor_api_server.py
    ├─→ base_predictor.py (interface)
    │   └─→ dummy_cara_predictor.py (implementation)
    │       ├─→ schedule_trace_client.py (fetch queue state)
    │       └─→ training_data_collector.py (log data)
    └─→ data_structures.py (PredictRequest, ScheduleState)

deploy_cara.py
    ├─→ model_deployment.json (which models where)
    ├─→ host_config.json (ports, IPs)
    └─→ predictor_deployment_config.json (predictor settings)
```

---

## Documentation

### ✅ New Comprehensive README

**Location**: `cara_paper/claude/README.md`

**Contents**:
- Complete overview of CARA architecture and key features
- Detailed quick start guide for local and multi-node deployment
- Component documentation (scheduler, predictors, instances, training pipeline)
- Configuration reference with examples
- Data collection and training pipeline walkthrough
- API reference for scheduler and predictor endpoints
- Troubleshooting guide with common issues and solutions
- Development roadmap and status

**Purpose**: Primary documentation for new developers and users to understand and deploy CARA.

## Current Status (as of March 31, 2026)

### Complete ✅

1. **Data Pipeline**
   - ✅ 18,605 prompts × 4 models (Qwen2.5-3B/7B/14B/72B), 7 datasets
   - ✅ 6 quality signals: similarity, blind judge, reference_score, reference_similarity, safety (protectai), prometheus (in progress)
   - ✅ 404,929 latency records across 5 instance types (3b_p100, 3b_a30, 7b_a30, 14b_v100, 72b_a100)
   - ✅ HuggingFace datasets: `asdwb/cara_model_estimator`, `asdwb/cara_latency_prediction` (needs re-upload)

2. **Predictor Training & Evaluation**
   - ✅ **RoBERTa fused multi-head** (5 targets, 5 epochs) — BEST on all targets (avg ρ=0.559)
   - ✅ **ModernBERT fused multi-head** (5 targets, 5 epochs) — 2nd best (avg ρ=0.529)
   - ✅ **KNN** with FAISS index — competitive baseline (avg ρ=0.452)
   - ✅ **LoRA 0.5B** (Qwen) — NOT VIABLE (avg ρ=0.291)
   - ✅ **XGBoost latency** — deployed, 1.2-6.2% MAPE
   - ✅ Full evaluation framework: per-model metrics, cross-model ranking, CSV output, bucket filtering
   - ✅ CSD3: 100-epoch fused RoBERTa submitted (PENDING, Q2 budget April 1)

3. **Scheduler**
   - ✅ Multi-objective scoring (latency + quality + cost + balance)
   - ✅ Tiered constraint relaxation (TTFT → TPOT → quality → budget)
   - ✅ Bucket CDF filtering: P(tokens ≤ budget) ≥ θ
   - ✅ Batch scheduling with all gaps closed
   - ✅ CloudLab smoke test: 45/45 PASS on 2×A30

4. **Infrastructure**
   - ✅ Automated deployment (deploy_cara.py)
   - ✅ Benchmark client with per-request metrics
   - ✅ P-D disaggregation prototype (cara_pd/)
   - ✅ route-balance repo (github.com/AKafakA/route-balance)

### In Progress 🚧

1. **Prometheus Reference-Grounded Judging**
   - 🚧 Qwen2.5-7b-it_reference scoring running on Vast (~25%, ETA ~12:30 UTC)
   - ⏳ Prometheus-7b-v2 model scoring (after current run)
   - ⏳ Encoder training for prometheus target
   - Score stored in `llm_judge_scores.{model}_{mode}` (blind/reference)

2. **LSTM v2 Latency Predictor**
   - ✅ Architecture written (per-request queue shapes)
   - ⏳ Training on 359K data (needs GPU)

3. **CSD3 Training**
   - ⏳ 5 jobs PENDING: fused RoBERTa 100ep, ModernBERT long, RoBERTa/DeBERTa long, LoRA per-model

### Not Started ❌

1. **SLO Filtering Baselines** (Task #17)
   - ❌ SLOs-Serve filter (binary admission control)
   - ❌ PolyServe filter (cumulative deadline)
   - ❌ QLM filter (Normal confidence bounds)
   - ❌ Cara internal ablation (hard reject vs tiered relaxation sensitivity)
   - Papers to read: SLOs-Serve (2504.08784), PolyServe (2507.17769), QLM (2407.00047)

2. **Pre-CloudLab Integration**
   - ❌ P100 continuous smoke test nodes
   - ❌ Experiment configs for 75+ ablation runs
   - ❌ Paper skeleton (intro, methodology, related work)

3. **Extensions**
   - ❌ TimeBill baseline
   - ❌ Conformal prediction / quantile regression for TTFT/TPOT
   - ❌ Quality CDF filtering in scheduler (beyond budget)
   - ❌ HF dataset rename (cara → route_balance)

---

## Recent Updates (January 2026)

### Unified Benchmark Preprocessing Pipeline (January 22, 2026)

**Script**: `block/predictor/cara/offline_training/prepare_benchmark_data.py`

The preprocessing pipeline has been refactored into a unified script that handles both single-model and multi-model data formats.

**Key Features**:
- **Auto-detection**: Automatically detects data format (single-model vs multi-model)
- **Model collection**: Scans ALL requests to collect models (not just first request)
- **Incomplete handling**: Exports incomplete requests in JSONL format for re-running
- **Unified scoring** (`--scoring-method all`): Runs both similarity + LLM judge in a single pass
- **No latency fields**: Latency from broadcasting is meaningless (concurrent contention); collected separately via `generate_latency_benchmark.py`
- **Multi-judge comparison**: Compare multiple LLM judges with correlation analysis
- **Divergent samples**: Detect and export samples where judges disagree

**Usage Examples**:
```bash
# Default: both similarity + LLM judge scoring (recommended)
python -m block.predictor.cara.offline_training.prepare_benchmark_data \
    --input data/cara/broadcast_v3_20k.json \
    --scoring-method all \
    --include-response \
    --device cuda

# Similarity only (fast, CPU-friendly)
python -m block.predictor.cara.offline_training.prepare_benchmark_data \
    --input data/cara/broadcast_v3_20k.json \
    --scoring-method similarity \
    --reference-model "Qwen/Qwen2.5-72B"

# LLM judge only
python -m block.predictor.cara.offline_training.prepare_benchmark_data \
    --input data/cara/broadcast_v3_20k.json \
    --scoring-method llm_judge \
    --judge-models Qwen/Qwen2.5-7B-Instruct \
    --device cuda \
    --batch-size 8
```

**Output Files**:
- `{output}.json`: Training data with quality scores
- `{output}_incomplete.jsonl`: Incomplete requests for re-running (benchmark format)
- `{output}_divergent.json`: Samples where judges disagree (if `--compare-judges`)

**Output Schema** (model-related fields only, no latency):
```json
{
  "dataset_name": "benchmark",
  "scoring_method": "all",
  "num_requests": 18700,
  "models": ["Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-14B", "Qwen/Qwen2.5-72B"],
  "requests": [
    {
      "request_id": "bench-xxx",
      "prompt": "...",
      "input_len": 128,
      "models": {
        "Qwen/Qwen2.5-3B": {
          "output_length": 256,
          "response": "generated text...",
          "similarity_score": 0.85,
          "llm_judge_scores": {
            "Qwen_Qwen2.5-7B-Instruct": 7
          },
          "compression_ratio": 0.45,
          "is_truncated": false
        }
      }
    }
  ]
}
```

**Key Design Decisions**:
- `similarity_score`: Cosine similarity to 72B reference response (sentence-transformers, fast/lightweight)
- `llm_judge_scores`: Dict of `{judge_key: score}` for extensibility; current best judge is `Qwen/Qwen2.5-7B-Instruct` (Spearman ρ=0.724 on feedbackQA with 1-10 scale, 2,701 samples). Scores normalized via `rating/10` in JSONL splits
- `response`: Full text included with `--include-response` (needed for training)
- No latency fields: Broadcasting sends prompts to all 4 models concurrently, causing contention; latency data is collected separately via controlled single-model requests

**LLM Judge Scorer** (`llm_judge_scorer.py`):
- 3-stage score parsing: number extraction → exact substring match → semantic embedding match
- Configurable scales (`--score-min`, `--score-max`), rationale-based prompting (`--disable-rationale`)
- Best model: `Qwen/Qwen2.5-7B-Instruct` (validated via `validate_judge_models.py` against feedbackQA)
- Batched inference for faster scoring (`--batch-size`)
- Failed scores marked as `None` and filtered out
- Validation errors raise exceptions instead of silently continuing

### Design Decisions for Estimation Models

**Critical Question**: What features are available at online inference time?

| Feature | Available Online? | Recommendation |
|---------|-------------------|----------------|
| Raw prompt text | ✅ Yes | Use as input (tokenize/embed) |
| Prompt length (tokens) | ✅ Yes | Use as input |
| Data source tag (sharegpt, gsm8k, etc.) | ❌ No | Don't use - not available in production |
| Prompt type (summarization, code, etc.) | ❌ No | Would require classifier overhead |
| Model name | ✅ Yes | Use as input (for multi-model) |

**Key Principle**: Only use features available at inference time. Data source tags and prompt classifications are useful for offline analysis/evaluation but NOT as model input features.

**Repetition Handling**:
- **Don't filter**: Keep repetitive samples for training
- **Why**: Model should learn to predict when outputs will be repetitive
- **Store**: `compression_ratio` as a quality metric/label
- **Analysis**: Repetition correlates with output length (34.5% at 800+ tokens vs 0.2% at 50-200 tokens)

**Filtering Strategy**:
```python
# Mandatory filtering (data quality issues)
- Remove errors/failures
- Remove empty responses (output_len <= 1)

# Length filtering
- min_output_tokens = 3  (too short to be useful)
- max_output_tokens = 1024  (match benchmark setting)

# Disabled by default (keep for training data)
- filter_high_repetition = False  # Model should learn this
- filter_truncated = False  # Flag instead of filter
```

### Benchmark Data Quality Analysis (January 21, 2026)

**Dataset**: ShareGPT-based benchmark with Qwen2.5-3B
**Configuration**: 20,000 prompts @ 4 QPS, presence_penalty=1.2, max_tokens=1024

**Summary Statistics**:
| Metric | Value |
|--------|-------|
| Total requests | 20,000 |
| Completed | 20,000 (100%) |
| Failed/errors | 0 |
| Valid responses (>1 token) | 19,288 (96.4%) |
| Empty responses | 712 (3.6%) |
| Avg input length | 151 tokens |
| Avg output length | 262.7 tokens |

**Output Length Distribution**:
| Bucket | Count | Percentage |
|--------|-------|------------|
| Very short (≤5 tokens) | 3,455 | 17.3% |
| Short (6-50 tokens) | 2,949 | 14.7% |
| Medium (51-200 tokens) | 4,879 | 24.4% |
| Long (201-500 tokens) | 5,259 | 26.3% |
| Very long (>500 tokens) | 3,458 | 17.3% |

**Truncation**: 1,614 (8.1%) hit max_tokens=1024

**Repetition Analysis**:
| Output Length | Repetition Rate |
|---------------|-----------------|
| 50-200 tokens | 0.2% |
| 200-400 tokens | 3.3% |
| 400-600 tokens | 11.3% |
| 600-800 tokens | 21.2% |
| 800-1024 tokens | 34.5% |

**Implication**: Repetition is model-inherent for long outputs, not a data quality bug. The presence_penalty=1.2 helps but doesn't eliminate it. Model should learn to predict this pattern.

**Input-Output Correlation**: r=0.047 (very weak)
- Length prediction cannot rely on input length alone
- Need semantic features from prompt

### Data Collection Improvements

**Issue Identified**: Initial broadcasting experiments revealed critical data quality issues:
- Poor model coverage (3B: 20%, 32B: 82% with repetition bugs)
- Similarity scoring fails on adversarial prompts (41.6% U-shape patterns)
- No quality differentiation between models on safety prompts

**Solutions Implemented**:
1. **LLM Judge Scoring**: Alternative to similarity-based scoring for mixed datasets
   - Uses `Unbabel/M-Prometheus-7B` to evaluate quality
   - Better handling of safety-divergent responses
   - File: `block/predictor/cara/offline_training/llm_judge_scorer.py`

2. **Model Exclusion**: Exclude problematic models from training
   - `--exclude-models "Qwen/Qwen2.5-32B"` to skip 32B (repetition bug)
   - Document notes to exclude 32B until bug is fixed

3. **Coverage Requirements**: Ensure all models present
   - `--require-all-models` flag filters requests with incomplete coverage
   - Ensures consistent training data quality

4. **Multi-Dataset Collection**: Balanced sampling from diverse sources
   - 9 datasets: ultrachat, mix_instruct, code_ultra_feedback, squad, cnn_dailymail, ai2_arc, gsm8k, reward_bench, beaver_tails
   - Token-based filtering (max 700 tokens by default)
   - Equal ratios for balanced distribution

**Data Quality Analysis**: `data/cara/data_quality_analysis.md`
- Detailed breakdown of coverage issues
- Example inspection of U-shape patterns
- Recommendations for fixing similarity scoring problems

### Documentation Updates

**New Comprehensive README**: `cara_paper/claude/README.md`
- 1000+ line comprehensive guide to CARA
- Architecture diagrams and component documentation
- Quick start for local and multi-node deployment
- API reference for scheduler and predictor
- Troubleshooting guide with common issues
- Development roadmap and status tracking

**Purpose**: Make CARA accessible to new developers and deployment teams without requiring deep dive into code.

## Evaluation Results (March 31, 2026)

### Predictor Architecture Comparison (Spearman ρ, 5 epochs, n=3,642)
| Target | RoBERTa | ModernBERT | KNN | LoRA-0.5B |
|--------|---------|------------|-----|-----------|
| Length | **0.696** | 0.679 | 0.576 | 0.307 |
| Bucket | **0.691** | 0.671 | — | — |
| Judge | **0.399** | 0.349 | 0.340 | 0.126 |
| Similarity | **0.419** | 0.394 | 0.384 | 0.353 |
| Ref score | **0.592** | 0.550 | 0.508 | 0.379 |

### Latency Predictor Baselines (45K test)
| Instance | XGBoost MAPE | Roofline MAPE |
|----------|-------------|---------------|
| 3b_p100 | ~2% | 10.1% |
| 7b_a30 | ~3% | 9.6% |
| 14b_v100 | ~6% | 22.1% |
| 72b_a100 | ~1.5% | 4.1% |

### Budget Filtering (CDF-based, bucket classifier)
| Mode | θ | Compliance | False Accept | False Reject |
|------|---|-----------|-------------|-------------|
| Relaxed | 0.7 | 81.2% | 18.8% | 36.6% |
| Balanced | 0.7 | 60.4% | 39.6% | 41.7% |

---

## Future Work & Roadmap

> **Apr 15 update:** evaluation plan superseded by the corrected design in `cara_paper/codex/apr15_baseline_comparison_design.md` and the pinned block at the top of `cara_paper/ROADMAP_TO_SUBMISSION.md`. The items below are kept for historical reference; live plan items are the tasks #51–#55 in the tracker and the retired-items strike-throughs in the roadmap.

### Before CloudLab (April 26) — current live items
1. **Cara-full ingredient toggles** — tasks #51 (LPT on/off) + #52 (assignment_strategy `{scoring, shortest_queue, llumnix_minus}` over all instances) + #53 (verify per-term zero-weights including **`w_balance`** via `/v1/config`).
2. **deepeval post-hoc scoring pipeline** — task #54.
3. **Main + ablation sweep script** — task #55, gated on #51–#54.
4. **Jaeger per-request SR overhead fetch** — task #49.
5. **Paper skeleton** — intro, methodology, related work ready before experiments.

### CloudLab Evaluation (April 26 - May 6) — per corrected Apr 15 design
1. **Main table (revision 2):** `Cara-full` + `vllm_sr-wrapper` + `RouteLLM-wrapper (mf)` (pretrained, α=0.11593) + `Avengers-Pro-wrapper` (α sweep) + `BEST-Route-wrapper` (Qwen+armoRM trained, t sweep). Wrappers paired with RR + shortest_queue dispatch. Retired: `best_route-wrapper` argmax-quality (renamed to `quality_greedy_ablation`) and "strongest composed baseline".
2. **Primary ablation table:** Cara-full ingredients — `Cara-no-LPT`, `Cara-no-filter`, `Cara-point-filter` (planned — needs unified bucket filter, see design §5.11), `Cara-no-quality-term`, `Cara-no-latency-term`, `Cara-no-cost-term`, `Cara-no-balance-term`, `Cara-simple-shortestQ`, `Cara-simple-llumnix`.
3. **Supporting study 1:** routing-signal swap inside a fixed pluggable shell.
4. **Supporting study 2:** filter swap inside a fixed Cara shell.
5. **Latency predictor internal ablation:** XGBoost vs LSTM v2 vs Roofline (Cara-internal, supplemental).

### Retired (superseded Apr 15)
- ~~SLO filtering baselines delivered as an external comparison separate from Cara-full.~~ Filters are now a supporting-study axis inside a fixed Cara shell (see above).
- ~~Cara sensitivity sweep over θ, relaxation order, step size (~75-90 runs).~~ Replaced by single-term zero-out ablations (`Cara-no-*-term` rows) that isolate each scorer term cleanly.
- ~~Main comparison framed as "best external baseline vs full Cara".~~ Replaced by "Cara-full vs practical wrapper baselines + ingredient ablations + supporting studies".

### Post-Evaluation
1. Upload final datasets to HuggingFace (rename cara → route_balance)
2. Sync to route-balance repo
3. Paper results, figures, tables

_Below this point is legacy content from earlier phases, kept for reference._

---

### [LEGACY] Phase 2: Model Training (COMPLETE - March 2026)

**Priority**: 🔥 **HIGH** - Critical path to CARA value proposition

#### 2.1 Response Length Predictor

**Input**: `(prompt, model_name, num_prompt_tokens)`
**Output**: `predicted_output_length` (integer)

**Training Data Source**: `broadcast_results[].output_len`

**Model Architecture**:
```
Prompt Embedding (BERT/RoBERTa)
    │
    ▼
Concatenate [prompt_embedding, model_embedding, num_prompt_tokens]
    │
    ▼
2-layer MLP
    │
    ▼
Output: predicted_output_length
```

**Success Metrics**:
- MAE (Mean Absolute Error) < 15 tokens
- RMSE < 25 tokens
- Inference latency < 5ms

**Implementation Steps**:
1. ✅ Extract training data from `broadcast_results` (via `prepare_training_data.py`)
2. 🚧 Build prompt tokenizer + model embeddings (KNN baseline)
3. ❌ Train regression model (MLP/GBDT)
4. ❌ Evaluate on held-out test set
5. ❌ Export to ONNX for fast inference

**Files**:
- `block/predictor/cara/length_predictor.py` (to be created)
- Training data: `data/cara/best-route_*_model_estimation_training.json` (✅ available)

#### 2.2 Model Quality Estimator

**Input**: `(prompt, model_name, response)`
**Output**: `quality_score` (0-1 float)

**Training Data Source**: `broadcast_results[]` with human/GPT-4 quality labels

**Challenge**: Need quality labels - options:
1. GPT-4 as judge (automated but expensive)
2. Human labeling (accurate but slow)
3. Proxy metrics (ROUGE, perplexity - fast but crude)

**Model Architecture**:
```
[Prompt, Response] → BERT → Quality Score
```

**Success Metrics**:
- Correlation with human judgments > 0.7
- Inference latency < 10ms

**Implementation Steps**:
1. ✅ Label collected data using LLM judge (Unbabel/M-Prometheus-7B)
2. ❌ Train quality predictor (KNN baseline or lightweight encoder)
3. ❌ Validate on held-out test set
4. ❌ Deploy for routing decisions

**Files**:
- `block/predictor/cara/quality_estimator.py` (to be created)
- Training data: `data/cara/best-route_llm_judge_model_estimation_training.json` (can be generated)
- Scoring: `block/predictor/cara/offline_training/llm_judge_scorer.py` (✅ complete)

#### 2.3 Extensible Latency Predictor (Architecture Update - January 25, 2026)

> **📖 Full Design Document**: [LATENCY_PREDICTOR_DESIGN.md](LATENCY_PREDICTOR_DESIGN.md) (v2.0)

**Architecture Update**: Extensible scheduler-side predictor supporting multiple implementations for **ablation studies**.

| Predictor Type | State API | Use Case |
|----------------|-----------|----------|
| `xgboost` | `INSTANCE_STATS` | Primary CARA predictor |
| `lstm` | `SCHEDULE_TRACE` | Research baseline |
| `roofline` | `INSTANCE_STATS` | Ablation baseline |
| `linear` | `INSTANCE_STATS` | Ablation baseline |
| `dummy` | `SCHEDULE_TRACE` | Data collection only |

**Key Design Decisions**:
- **Factory pattern**: `predictor_type` field dispatches to correct config/predictor class
- **State API mapping**: Each predictor declares `STATE_API_TYPE` class attribute
- **Instance type derivation**: `{model}_{backend}_{node_type}` derived at deployment
- **Decoupled data collection**: Separate from scheduling policy
- **Single-writer**: No predictor replicas needed (sequential scheduling within batch)

**Configuration Example** (`predictor_config_xgboost.json`):
```json
{
  "predictor_type": "xgboost",
  "model_dir": "./models/cara/xgboost",
  "enable_data_collection": true,
  "data_output_dir": "./training_data/cara"
}
```

**Success Metrics**:
- E2E MAE < 200ms
- Per-request inference < 100μs
- Per-batch state query < 50ms

**Implementation Tasks - Phase 1 (Architecture)** (14 total):

| # | Task | Status |
|---|------|--------|
| 1 | Port cleanup: `predictor_ports` → `data_collection_port` | ✅ complete |
| 2 | Extend predictor interface (STATE_API_TYPE, factory pattern) | ✅ complete |
| 3 | StateManager (refresh_all, apply_assignment) | ✅ complete |
| 4 | PredictorRegistry (instance_type → predictor) | ✅ complete |
| 5 | SchedulingPolicy (BOOTSTRAP vs CARA) | ✅ complete |
| 6 | XGBoostPredictor class | ✅ complete |
| 7 | XGBoost training pipeline | ✅ complete |
| 8 | Integration (/v1/batch_completions, callback) | ✅ complete |
| 9 | EMA service rate tracking | ✅ complete |
| 10 | Unit tests | ✅ complete |
| 11 | RooflinePredictor (ablation) | ✅ complete |
| 12 | Evaluation framework | ✅ complete |
| 13 | Documentation update | ✅ complete |
| 14 | Task 1.4 Multi-Objective Scoring | ✅ complete |

**Implementation Tasks - Phase 2 (Review Fixes)** (12 total - See `IMPLEMENTATION_TASKS_V2_DESIGN.md`):

| # | Task | Status |
|---|------|--------|
| 1 | Fix cara_serve batch_completions streaming response | ❌ pending |
| 2 | Fix XGBoost predictor target field naming | ❌ pending |
| 3 | Add real vs predicted decode length option | ❌ pending |
| 4 | Rename DummyCARAPredictor to DataCollectionService | ❌ pending |
| 5 | Decouple ServiceRateTracker raw rates from EMA | ❌ pending |
| 6 | Update deploy_cara.py for new architecture | ❌ pending |
| 7 | Add predictor overhead tracking (optional) | ❌ pending |
| 8 | Add state sync time tracking to StateManager | ❌ pending |
| 9 | Create E2E training pipeline shell script | ❌ pending |
| 10 | Add CDF error distribution plots to evaluation | ❌ pending |
| 11 | Add training learning curve reporting | ❌ pending |
| 12 | Update benchmark_serving for bootstrap mode | ❌ pending |

**Files**:
- Design: `cara_paper/claude/LATENCY_PREDICTOR_DESIGN.md` ✅ (v2.0)
- vLLM API Spec: `cara_paper/claude/INSTANCE_STATS_API.md` ✅
- Predictors: `block/predictor/cara/{xgboost,roofline,linear,lstm}_predictor.py` (to be created)
- Scheduler: `block/predictor/cara/{state_manager,predictor_registry,scheduling_policy}.py` (to be created)
- Training: `block/predictor/cara/offline_training/train_xgboost.py` (to be created)
- Evaluation: `block/predictor/cara/offline_training/evaluate_predictor.py` (to be created)
- vLLM endpoint: `/instance_stats` ✅ (implemented in separate repo)

### Phase 3: Production Integration (Q2 2026)

**Priority**: 🔥 **HIGH** - Deploy trained models

#### 3.1 Predictor Replacement

- Replace `DummyCARAPredictor` with trained models
- Load checkpoints from disk
- Fast inference (< 20ms overhead)

#### 3.2 Scheduling Algorithm Development

**Options**:
1. **Simple**: Route to instance with lowest predicted latency
2. **Multi-objective**: Balance latency + quality + load
3. **Thompson Sampling**: Exploration-exploitation tradeoff
4. **Reinforcement Learning**: Learn optimal policy

**File**: `block/global_scheduler/cara/scheduling_policy.py`

#### 3.3 A/B Testing Framework

Compare:
- Random baseline
- Round-robin baseline
- Heuristic-based (min_requests, max_gpu_blocks)
- CARA with predictions

**Metrics**:
- Mean/P50/P99 latency
- Throughput (requests/sec)
- Quality (average score)

### Phase 4: Advanced Features (Q3 2026)

**Priority**: 🟡 **MEDIUM** - Nice to have

#### 4.1 Online Learning

- Continuous model updates from production data
- Periodic retraining
- A/B test new models before deployment

#### 4.2 Multi-Tenant Support

- Per-user fairness constraints
- Priority queues
- SLA enforcement

#### 4.3 Cost Optimization

- Incorporate instance cost ($/token) into routing
- Minimize cost subject to latency SLA

#### 4.4 Auto-Scaling

- Predict load spikes
- Trigger instance scaling
- Load balancing during transitions

---

## Feasibility Analysis

### Technical Feasibility: 🟢 **HIGH**

**Strengths**:
1. ✅ **Core infrastructure complete**: Data collection, deployment, testing
2. ✅ **Well-defined interfaces**: Clear separation between components
3. ✅ **Proven approach**: Similar to DistServe, Clockwork, Nexus
4. ✅ **Data availability**: Can collect unlimited training data

**Risks**:
- ⚠️ Prediction accuracy depends on data quality
- ⚠️ Inference overhead must be < 20ms (tight constraint)
- ⚠️ Model generalization to new workloads uncertain

**Mitigation**:
- Collect diverse training data (multiple datasets, prompt lengths)
- Optimize inference (ONNX, quantization, batching)
- Continuous evaluation on production traffic

### Schedule Feasibility: 🟡 **MEDIUM**

**Phase 2 Estimate**: 6-8 weeks (one researcher)
- Week 1-2: Length predictor + data preprocessing
- Week 3-4: Quality estimator + labeling
- Week 5-6: LSTM latency predictor
- Week 7-8: Integration + testing

**Phase 3 Estimate**: 4-6 weeks
- Week 1-2: Predictor replacement + deployment
- Week 3-4: Scheduling algorithm + A/B testing

**Critical Path**: Quality labeling (if using human labels)

**Acceleration Options**:
- Use GPT-4 as judge (faster but less accurate)
- Parallelize length + quality predictor development
- Reuse existing LSTM architectures from literature

### Resource Feasibility: 🟢 **HIGH**

**Computing**:
- ✅ Training: Single GPU sufficient (model sizes < 100M params)
- ✅ Inference: CPU-only for predictors
- ✅ Data collection: Already running on CloudLab

**Data**:
- ✅ 500 prompts × 5 models = 2500 examples (initial batch)
- ✅ Can scale to 10K+ examples easily
- ✅ ShareGPT dataset available

**Cost**:
- Training: ~$50 (GPU hours)
- Labeling: ~$500 if using GPT-4 (5000 examples × $0.10 each)
- Deployment: Zero marginal cost (uses existing infrastructure)

---

## Risks & Mitigation

### Risk 1: Prediction Accuracy

**Risk**: Predictions inaccurate → bad routing → worse latency than baseline

**Likelihood**: 🟡 **MEDIUM**
**Impact**: 🔴 **HIGH** (undermines project value)

**Mitigation**:
1. Fallback to heuristics if prediction confidence low
2. A/B test before full rollout
3. Continuous monitoring of prediction error
4. Collect data from diverse workloads

### Risk 2: Inference Overhead

**Risk**: Predictor inference too slow → adds latency instead of reducing it

**Likelihood**: 🟢 **LOW** (models are small)
**Impact**: 🟠 **MEDIUM**

**Mitigation**:
1. Optimize with ONNX Runtime
2. Batch predictions across instances
3. Cache predictions for similar requests
4. Use quantized models (INT8)

### Risk 3: Training Data Bias

**Risk**: Collected data not representative of production workloads

**Likelihood**: 🟡 **MEDIUM**
**Impact**: 🟠 **MEDIUM**

**Mitigation**:
1. Collect data from multiple datasets (ShareGPT, Sonnet, custom)
2. Vary prompt lengths, topics, complexity
3. Periodically retrain on production data
4. Monitor distribution shift

### Risk 4: Model Generalization

**Risk**: Models overfit to specific models/workloads

**Likelihood**: 🟡 **MEDIUM**
**Impact**: 🟠 **MEDIUM**

**Mitigation**:
1. Train/val/test split by prompt source
2. Test on unseen models (e.g., train on 3B/7B/14B, test on 32B)
3. Cross-validation
4. Regularization (dropout, weight decay)

### Risk 5: Quality Labeling Subjectivity

**Risk**: Quality scores inconsistent or noisy

**Likelihood**: 🟡 **MEDIUM** (if using GPT-4)
**Impact**: 🟠 **MEDIUM**

**Mitigation**:
1. Use GPT-4 with detailed rubric
2. Validate on human-labeled subset
3. Ensemble multiple judges
4. Focus on clear cases (very good vs very bad)

### Risk 6: Deployment Complexity

**Risk**: Production deployment breaks existing system

**Likelihood**: 🟢 **LOW** (good separation of concerns)
**Impact**: 🔴 **HIGH**

**Mitigation**:
1. Shadow mode: run predictors without affecting routing
2. Gradual rollout (10% → 50% → 100%)
3. Automated rollback on regressions
4. Comprehensive monitoring

---

## Impact Assessment

### Performance Impact: 🔥 **HIGH POTENTIAL**

**Expected Improvements** (based on related work):
- **Latency**: 20-40% reduction in P99 latency (vs random)
- **Throughput**: 30-50% increase (better load balancing)
- **Quality**: 10-20% improvement (routing to better models)

**Comparison to Baselines**:
| Metric | Random | Round-Robin | Heuristic | CARA (Projected) |
|--------|--------|-------------|-----------|-------------------|
| P99 Latency | Baseline | -5% | -15% | **-30%** |
| Throughput | Baseline | +10% | +20% | **+40%** |
| Quality | Baseline | 0% | +5% | **+15%** |

### Scientific Impact: 🔥 **HIGH**

**Novelty**:
1. **First** learned scheduler for heterogeneous LLM serving
2. **First** to jointly optimize latency + quality + cost
3. **First** to use broadcasting for multi-model training data

**Publication Potential**:
- Top-tier systems conference (OSDI, SOSP, NSDI, EuroSys)
- ML systems workshop (MLSys, SysML)

**Open Source Value**:
- Production-ready code (not just prototype)
- Reusable components (predictor framework, broadcasting mode)
- Training data release (if permitted)

### Practical Impact: 🟡 **MEDIUM-HIGH**

**Adoption Scenarios**:
1. **Cloud providers**: AWS Bedrock, Azure OpenAI
2. **LLM companies**: OpenAI, Anthropic, Cohere
3. **Enterprise deployments**: Companies running multiple models

**Barriers to Adoption**:
- Requires multiple model sizes (not all users have this)
- Needs initial training data (cold start problem)
- Complexity vs simple baselines

**Enablers**:
- Open source release
- Docker containers for easy deployment
- Pre-trained models for common LLMs

---

## Developer Guide

### For Future Claude Code Sessions

**Context Loading**:
1. Read this document first
2. Check `block/predictor/cara/README.md` for API details
3. Review `cara_serve.py` for scheduler logic
4. Understand `training_data_collector.py` for data format

**Key Files to Modify**:
- **Adding predictor**: `block/predictor/cara/cara_predictor.py` (new file)
- **Changing scheduling**: `block/global_scheduler/cara/cara_serve.py` (line 173-183)
- **Training pipeline**: `block/predictor/cara/train.py` (new file)

**Common Tasks**:

#### Task 1: Add New Predictor Type

```python
# 1. Create new config class
class MyPredictorConfig(CARABasePredictorConfig):
    predictor_type: str = "my_predictor"
    model_path: str
    # ... other config fields

# 2. Implement predictor
class MyCARAPredictor(CARABasePredictor):
    def __init__(self, config: MyPredictorConfig, ...):
        super().__init__(config, ...)
        self.model = load_model(config.model_path)

    async def predict(self, request: PredictRequest) -> dict:
        # Run inference
        ...

# 3. Register in cara_predictor_api_server.py
elif config.predictor_type == "my_predictor":
    predictor = MyCARAPredictor(config, ...)
```

#### Task 2: Implement Predictor-Based Scheduling

```python
# In cara_serve.py, replace lines 173-178:

# OLD (random scheduling):
selected_instance = random.choice(instances)

# NEW (prediction-based):
# 1. Find instance with lowest predicted latency
best_instance = min(
    zip(instances, predictions),
    key=lambda x: x[1].get('target_metric', float('inf'))
)[0]

selected_instance = best_instance
```

#### Task 3: Add New Metric to Training Data

```python
# 1. Update TrainingExample in data_structures.py
@dataclass
class TrainingExample:
    # ... existing fields ...
    actual_my_metric: float = 0.0  # New field

# 2. Update feedback in Instance.py
feedback_data = {
    "request_id": str(request_id),
    "e2e_latency": e2e_latency,
    "my_metric": response_dict.get("my_metric"),  # New
}

# 3. Update collector in training_data_collector.py
example.actual_my_metric = actual_metrics.get("my_metric", 0.0)
```

### Common Pitfalls to Avoid

1. **Port Configuration**: Always use host_config.json as single source of truth
2. **Field Names**: Use `num_prompt_tokens` NOT `num_prompt_len`
3. **Round-Robin**: Use per-instance counter, NOT global request_id
4. **Feedback Routing**: Send to SAME predictor that made prediction
5. **Async Safety**: Always use `asyncio.create_task()` for fire-and-forget

### Testing Checklist

Before claiming a feature is complete:

- [ ] Unit tests pass
- [ ] End-to-end test with dummy predictor works
- [ ] Broadcasting collects data correctly
- [ ] Training data format validated
- [ ] Predictor feedback reaches collectors
- [ ] No port collisions
- [ ] Logs show expected behavior
- [ ] Benchmark results look reasonable

---

## Conclusion

**Project Status**: 🟢 **HEALTHY** - Phase 1 complete, ready for Phase 2

**Key Achievements**:
- Robust data collection infrastructure ✅
- Broadcasting for multi-model training ✅
- Deployment automation ✅
- Clear roadmap for ML model development ✅

**Next Critical Steps**:
1. Implement response length predictor (2 weeks)
2. Implement quality estimator with GPT-4 labeling (2 weeks)
3. Train LSTM latency predictor (2 weeks)
4. Deploy and A/B test (2 weeks)

**Recommended Priority**: Focus on response length predictor first (easiest, most immediate value).

**Long-Term Vision**: CARA becomes the default scheduler for heterogeneous LLM serving, adopted by major cloud providers and LLM companies.

---

## Quick Reference Links

**Primary Documentation**:
- **CARA README**: `cara_paper/claude/README.md` - Comprehensive user and developer guide
- **Project Status**: `cara_paper/active/CARA_PROJECT_STATUS.md` - This document
- **Deployment Workflow**: `cara_paper/claude/DEPLOYMENT_WORKFLOW.md` - Step-by-step deployment guide
- **Latency Predictor Design**: `cara_paper/claude/LATENCY_PREDICTOR_DESIGN.md` - Extensible predictor architecture (v2.0)
- **Instance Stats API**: `cara_paper/claude/INSTANCE_STATS_API.md` - vLLM `/instance_stats` endpoint spec
- **Submission Plan**: `cara_paper/active/SUBMISSION_PLAN.md` - Venue analysis and paper strategy
- **Implementation Tasks V2 Design**: `cara_paper/claude/IMPLEMENTATION_TASKS_V2_DESIGN.md` - Detailed design for 12 pending review fixes ✨ **NEW**

**Deployment Guides (NEW - January 16, 2026)**:
- **A100 Deployment**: `cara_paper/claude/A100_DEPLOYMENT_GUIDE.md` - A100-specific deployment, NVIDIA library fixes
- **P100 Deployment**: `cara_paper/claude/P100_DEPLOYMENT_GUIDE.md` - P100-specific build and dataset notes
- **Experiment Quick Ref**: `cara_paper/claude/EXPERIMENT_QUICK_REFERENCE.md` - Quick commands for experiments

**Implementation Documentation**:
- **Codex Plan**: `cara_paper/codex/codex-cara-plan.md` - Implementation roadmap and milestones
- **Objective Design**: `cara_paper/codex/cara-objective-design.md` - Multi-objective scoring function design
- **Data & Experiments**: `cara_paper/codex/cara-data-and-experiments.md` - Data collection and experiment plan

**Component READMEs**:
- **Predictor**: `block/predictor/cara/README.md` - Predictor API and data structures
- **Offline Training**: `block/predictor/cara/offline_training/README.md` - Training data preparation pipeline

**Data Analysis**:
- **Quality Analysis**: `data/cara/data_quality_analysis.md` - Data quality issues and fixes

**Key Source Files**:
- Scheduler: `block/global_scheduler/cara/cara_serve.py`
- Instance adapters: `block/global_scheduler/cara/cara_instance/{Instance,vllm_instance,ollama_instance}.py`
- Predictor server: `block/predictor/cara/cara_predictor_api_server.py`
- **Data preparation (unified, recommended)**: `block/predictor/cara/offline_training/prepare_benchmark_data.py`
- Data preparation (legacy multi-model): `block/predictor/cara/offline_training/prepare_training_data.py`
- LLM judge scorer: `block/predictor/cara/offline_training/llm_judge_scorer.py`
- Similarity scorer: `block/predictor/cara/offline_training/similarity_scorer.py`
- Dataset collection: `block/data/collect_data.py`
- Deployment: `block/exp/cara/deploy_cara.py`
- Setup: `block/exp/cara/setup.sh`
- vLLM Compatibility: `block/benchmark/cara/vllm_compat.py`
- Dataset Handler: `block/benchmark/cara/dataset.py`
- Word Limit Normalization: `block/benchmark/cara/tools/normalize_word_limits.py`

---

**Document Maintenance**:
- ✅ Updated January 26, 2026 with Task 1.4 Multi-Objective Scoring completion:
  - RSO/SLO data structures and tiered constraint handling
  - MultiObjectiveScorer with ScoringConfig integration
  - CandidateScore for normalized scoring
  - TTFT/TPOT derivation and placeholder estimators
  - Full integration with CARABatchPolicy
  - 166 passing tests
  - Created IMPLEMENTATION_TASKS_V2_DESIGN.md with detailed designs for 12 pending review fixes
- ✅ Updated January 25, 2026 with extensible predictor architecture v2.0:
  - Factory pattern for multiple predictor types (XGBoost, LSTM, Roofline, Linear)
  - Port configuration cleanup: `predictor_ports` → `data_collection_port`
  - New `/v1/batch_completions` endpoint design
  - Scheduler-side components: StateManager, PredictorRegistry, SchedulingPolicy
  - Instance type derivation at deployment time
  - Completion callback mechanism
  - 14 implementation tasks documented
- ✅ Updated January 25, 2026 with `/instance_stats` API implementation and P100 chunked prefill config
- ✅ Updated January 23, 2026 with latency predictor design pivot (XGBoost scheduler-side architecture)
- ✅ Updated March 13, 2026: Dataset v3 uploaded to HuggingFace (asdwb/cara_model_estimator). Split script now does post-processing: strips chat template tags (multi-format), recalibrates scores from (r-1)/9 to r/10. Output as JSONL. Judge validation: 5 models on feedbackQA (2,701 samples), Qwen 7B best (Pearson=0.730)
- ✅ Updated March 12, 2026 with unified schema (no latency, similarity_score + llm_judge_scores), recovered pipeline code, frequency_penalty fix, 20k broadcasting complete
- ✅ Updated January 22, 2026 with unified preprocessing pipeline, multi-judge comparison, and incomplete request handling
- ✅ Updated January 21, 2026 with single-model benchmark preprocessing pipeline and design decisions
- ✅ Updated January 16, 2026 with A100 deployment fixes and new guides
- ✅ Updated January 15, 2026 with Phase 2 progress and new documentation
- Update this document after each major milestone
- Keep architecture diagrams in sync with code
- Log all breaking changes in "Breaking Changes" section (to be added)
- Version control this document alongside code

**Contact**: For questions about this document or CARA project:
- Start with `cara_paper/claude/README.md` for comprehensive overview
- Check git history for recent changes
- File issues in the repository issue tracker
