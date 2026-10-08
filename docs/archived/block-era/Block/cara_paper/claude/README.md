# CARA: Co-optimizing Super Heterogeneous LLM Serving

**CARA** (Context-Aware Resource Allocation) is a learned, quality-aware scheduler for heterogeneous large language model (LLM) serving systems. It dynamically routes requests across instances running different model sizes and GPU types, using machine learning predictors to optimize for latency, quality, and cost.

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Architecture](#architecture)
4. [Quick Start](#quick-start)
5. [Components](#components)
6. [Configuration](#configuration)
7. [Data Collection Pipeline](#data-collection-pipeline)
8. [Training Estimators](#training-estimators)
9. [Deployment](#deployment)
10. [Benchmarking](#benchmarking)
11. [Development Status](#development-status)
12. [API Reference](#api-reference)
13. [Troubleshooting](#troubleshooting)
14. [References](#references)

---

## Overview

### What is CARA?

CARA is a global scheduler for heterogeneous LLM serving that routes requests to instances running different model sizes (3B, 7B, 14B, 32B, 72B parameters) and GPU types (A100, V100, A30, P100) based on learned predictions of:

1. **Latency**: Predicted server-side end-to-end latency (E2E), time-to-first-token (TTFT), and time-per-output-token (TPOT)
2. **Quality**: Model output quality for a given prompt-model pair
3. **Cost**: Token-based cost proxy considering prompt and predicted output lengths

### Why CARA?

**Traditional schedulers** use simple heuristics:
- Random/round-robin: Ignores workload characteristics and model capabilities
- Load-based: Only considers current queue depth, not predicted execution time
- Static rules: Brittle and requires manual tuning per deployment

**CARA's Approach**:
- **Learned predictions**: Uses machine learning models trained on real execution traces
- **Quality-aware routing**: Routes requests to models that provide the best quality for a given prompt
- **Multi-objective optimization**: Balances latency, quality, and cost with per-request constraints
- **Adaptive**: Learns from production traffic and adapts to new workloads

### Key Innovation

CARA implements a **closed-loop learning system**:

1. **Pre-scheduling prediction**: Before routing, CARA queries predictors for latency/quality estimates
2. **Routing decision**: Uses predictions to select optimal instance based on objective function
3. **Execution**: Selected instance executes the request
4. **Feedback loop**: Actual metrics (latency, quality) are sent back to predictors
5. **Continuous learning**: Predictors use feedback to improve predictions over time

This creates a self-improving system that adapts to changing workloads, new models, and infrastructure updates.

---

## Key Features

### Phase 1: Data Collection Infrastructure (✅ Completed)

- **Predictor Integration**: CARA scheduler queries all predictors before making routing decisions
- **Automatic Feedback Loop**: Instances automatically send actual metrics back to predictors
- **Broadcasting Mode**: Collect responses from multiple models for the same prompt to train quality estimators
- **Training Data Collection**: Automatic logging of (prediction context, actual metrics) pairs to JSONL files
- **Deployment Automation**: Deploy vLLM/Ollama backends and predictors across multi-host clusters
- **End-to-end Testing**: Benchmark framework with CARA-specific data collection

### Phase 2: Model Training (🚧 In Progress)

- **Offline Training Pipeline**: Prepare training data from broadcast results with filtering and scoring
- **Quality Estimation**: Similarity-based and LLM-judge scoring for model quality
- **Length Prediction**: KNN and regression-based output length prediction
- **Latency Prediction**: XGBoost-based latency prediction with scheduler-side architecture (see [LATENCY_PREDICTOR_DESIGN.md](LATENCY_PREDICTOR_DESIGN.md))
- **Multi-Objective Scoring**: ✅ Completed - RSO/SLO constraints, tiered handling, normalized scoring with configurable weights

### Phase 3: Production Integration (📋 Planned)

- **Learned Scheduling**: Replace dummy heuristics with trained ML models
- **Multi-objective Optimization**: Greedy mini-batch scheduler with configurable scoring
- **Request-Specific Objectives (RSO)**: Per-request constraints on latency, quality, and budget
- **Adaptive Batching**: Dynamic batch size and timeout based on load

---

## Architecture

### High-Level System Architecture

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
│  │ 1. Query all predictors in parallel (optional)           │   │
│  │ 2. Query quality estimator (future)                      │   │
│  │ 3. Compute scores for each instance                      │   │
│  │ 4. Select instance via scheduling policy                 │   │
│  │ 5. Forward request to selected instance                  │   │
│  │ 6. Optional: Broadcast to multiple models                │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────┬─────────────────────┬─────────────────────┬───────────────┘
      │                     │                     │
      ▼                     ▼                     ▼
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│ vLLM Instance   │   │ vLLM Instance   │   │ vLLM Instance   │
│ Qwen/2.5-72B    │   │ Qwen/2.5-14B    │   │ Qwen/2.5-7B     │
│ A100×4 (TP=4)   │   │ V100×4 (TP=1)   │   │ A30×1           │
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
Request Arrives
    │
    ▼
┌───────────────────────────────────────────────────┐
│ CARA Scheduler queries predictors (optional)      │
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
│ Instance sends feedback to predictor (optional)   │
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
cara_serve.py (CARA Scheduler)
    ├─→ Instance.py (base class)
    │   ├─→ vllm_instance.py (vLLM backend adapter)
    │   └─→ ollama_instance.py (Ollama backend adapter)
    ├─→ HTTP client for predictor API
    └─→ benchmark_serving.py (receives responses)

cara_predictor_api_server.py (Predictor API Server)
    ├─→ base_predictor.py (interface)
    │   └─→ dummy_cara_predictor.py (heuristic-based predictor)
    │       ├─→ schedule_trace_client.py (fetch queue state)
    │       └─→ training_data_collector.py (log training data)
    └─→ data_structures.py (PredictRequest, ScheduleState, TrainingExample)

deploy_cara.py (Deployment Automation)
    ├─→ model_deployment.json (which models on which hosts)
    ├─→ host_config.json (ports, IPs, SSH config)
    └─→ predictor_deployment_config.json (predictor settings)

offline_training/ (Training Pipeline)
    ├─→ prepare_benchmark_data.py (unified preprocessing: filter, score, export)
    │   ├─→ similarity_scorer.py (cosine similarity to 72B reference)
    │   └─→ llm_judge_scorer.py (Qwen2.5-7B-Instruct, 3-stage parsing)
    ├─→ validate_judge_models.py (validate judges against feedbackQA)
    ├─→ split_training_data.py (hash-based 80/20 split)
    ├─→ generate_latency_benchmark.py (controlled latency data collection)
    └─→ train_knn_estimator.py (KNN quality+length predictor)
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- CUDA 12.6+ (for GPU inference)
- Multiple GPU nodes with different SKUs (for heterogeneous deployment)
- vLLM 0.7.2+ or Ollama
- SSH access to all nodes

### 1. Installation

```bash
# Clone repository
cd /path/to/Block

# Install dependencies
pip install -r requirements.txt

# Install vLLM (custom build with /schedule_trace endpoint)
# See https://anonymous.4open.science/r/vllm-644A for CARA-compatible vLLM
```

### 2. Configuration

#### Step 2a: Generate Host Configuration

Generate host configuration from CloudLab manifest:

```bash
# If using CloudLab, download manifest.xml
python block/exp/generate_config.py \
  --username YOUR_SSH_USERNAME \
  --num_predictors 1

# This generates:
# - block/config/host_configs.json (host IPs, ports, SSH config)
# - block/config/{ampere,volta,pascal}_hosts (hostname lists)
```

Or manually create configuration files following examples in `block/config/`.

#### Step 2b: Configure Model Deployment

Edit the model deployment template to specify which models to deploy and how many instances:

```bash
# Edit the template to set model counts per node type
vi block/config/cara/model_config_template.json

# Example: Deploy 2 instances of Qwen 2.5-3B on c240g5 nodes
# Change "count": 0 to "count": 2 for the models you want
```

**Template Structure** (`model_config_template.json`):
```json
{
  "Qwen-2.5-7B": {
    "hf_model_name": "Qwen/Qwen2.5-7B",
    "precision": "fp16",
    "backend": "vllm",
    "node_type": {
      "d7525": {
        "count": 5,              // Number of instances on this node type
        "max-model-len": 8192
      }
    }
  }
}
```

**Key Fields**:
- `count`: Number of instances to deploy on this node type (0 = skip)
- `node_type`: Target node type (must match hostnames from `generate_config.py`)
- Other fields: vLLM/Ollama-specific parameters
### 2. Deploy Infrastructure

```bash
# Option 1: Use automated setup script
cd block/exp/cara
./setup.sh

# Option 2: Manual deployment with deploy_cara.py
cd block/exp/cara

# Step 1: Generate deployment config (assigns hosts)
python deploy_cara.py \
  --config ../../config/cara/model_config_template.json \
  --output ../../config/cara/model_deployment.json \
  --hosts ../../config/hosts \
  --deploy-services none  # Just generate config, don't deploy yet

# Step 2: Review the generated config
cat ../../config/cara/model_deployment.json

# Step 3: Deploy backends and predictors
python deploy_cara.py \
  --config ../../config/cara/model_config_template.json \
  --output ../../config/cara/model_deployment.json \
  --hosts ../../config/hosts \
  --deploy-services model_instance predictor \
  --hf-token YOUR_HF_TOKEN  # Required for downloading models
```

**Deployment Options**:
- `--deploy-services model_instance`: Deploy only vLLM/Ollama backends
- `--deploy-services predictor`: Deploy only predictors
- `--deploy-services model_instance predictor`: Deploy both (default)
- `--deploy-services none`: Just generate config without deployment
- `--models "Qwen-2.5-3B,Qwen-2.5-7B"`: Deploy only specific models (preserves existing deployments)

### 4. Start CARA Scheduler

```bash
python -m block.global_scheduler.cara.cara_serve \
  --host 0.0.0.0 \
  --port 8200 \
  --model_config_path block/config/cara/model_deployment.json \
  --host_config block/config/host_configs.json \
  --scheduling random \
  --enable-predictor-feedback
```

### 5. Run Benchmark

```bash
python -m block.benchmark.cara.benchmark_serving \
  --backend cara \
  --base-url http://localhost:8200 \
  --model dummy \
  --dataset-name sharegpt \
  --dataset-path data/trace_data/sharegpt/sharegpt_gpt4.jsonl \
  --num-prompts 100 \
  --request-rate 2.0
```

### 6. Collect Training Data (Broadcasting Mode)

```bash
# Start CARA in broadcasting mode to collect multi-model responses
python -m block.global_scheduler.cara.cara_serve \
  --host 0.0.0.0 \
  --port 8200 \
  --model_config_path block/config/cara/model_deployment.json \
  --host_config block/config/host_configs.json \
  --broadcasting \
  --selected-broadcasted-models "Qwen/Qwen2.5-3B" "Qwen/Qwen2.5-7B" "Qwen/Qwen2.5-14B" "Qwen/Qwen2.5-72B" \
  --enable-predictor-feedback

# Run benchmark - responses from all models will be collected
python -m block.benchmark.cara.benchmark_serving \
  --backend cara \
  --base-url http://localhost:8200 \
  --model dummy \
  --dataset-name sharegpt \
  --dataset-path data/trace_data/sharegpt/sharegpt_gpt4.jsonl \
  --num-prompts 500 \
  --request-rate 2.0 \
  --save-detailed prompts response ttft e2el models \
  --save-result
```

---

## Components

### 1. CARA Scheduler

**Location**: `block/global_scheduler/cara/cara_serve.py`

**Responsibilities**:
- Accept requests via `/v1/completions` endpoint (vLLM-compatible)
- Query predictors for latency estimates (optional, for training data collection)
- Select target instance via scheduling policy (random, round-robin, or learned)
- Forward request and collect response
- Optional: Broadcast to multiple models and collect all responses

**Scheduling Policies**:
- `random`: Random instance selection
- `round_robin`: Round-robin across instances
- `greedy_lpt`: (Planned) LPT-based batch scheduling with XGBoost predictions
  - Active pull of `/instance_stats` before each batch
  - Sort requests by predicted tokens descending (LPT heuristic)
  - Greedy assignment with multi-objective scoring (latency + quality + cost)
  - O(|R| × |I|) complexity (linear scan, no permutation search)

**Key Features**:
- Parallel predictor queries using `asyncio.gather()`
- Automatic predictor feedback via instance adapters
- Broadcasting mode for multi-model data collection
- Server-side latency tracking for overhead measurement

### 2. Instance Adapters

**Location**: `block/global_scheduler/cara/cara_instance/`

**Base Class**: `Instance.py`
- Abstract interface for backend instances
- Predictor query and feedback management
- Request-to-predictor mapping (ensures feedback goes to correct predictor)

**Implementations**:
- `vllm_instance.py`: Adapter for vLLM instances with SSE streaming
- `ollama_instance.py`: Adapter for Ollama instances with NDJSON streaming

**Predictor Integration**:
- `query_predictor()`: Send prediction request before routing
- `_send_predictor_feedback()`: Send actual metrics after execution
- Round-robin predictor selection per instance
- Configurable feedback sampling rate (0.0-1.0)

### 3. Predictor Sidecar

**Location**: `block/predictor/cara/`

**Architecture**: Independent from Block's Vidur-based predictor
- Lightweight API server co-located with each backend instance
- Monitors backend queue state via `/schedule_trace` endpoint
- Collects training data automatically
- Simple interface: `PredictRequest` → `prediction_dict`

**Dummy Predictor** (`dummy_cara_predictor.py`):
- Heuristic-based scheduling for data collection phase
- Modes: `min_requests`, `max_gpu_blocks`, `combined`, `random`
- Automatic training data collection to JSONL files

**API Endpoints**:
- `POST /predict`: Get latency prediction for a request
- `POST /log_actual`: Log actual metrics (feedback from instance)
- `GET /stats`: Collection statistics
- `POST /flush`: Force write buffered data to disk
- `GET /health`: Health check

**Data Collection**:
- `TrainingDataCollector`: Buffers (prediction, actual) pairs in memory
- Automatic matching by `request_id`
- Batched disk writes every N examples (configurable)
- Stats tracking: predictions, completions, pending, buffered

### 4. Training Data Pipeline

**Location**: `block/predictor/cara/offline_training/`

**Purpose**: Prepare clean training datasets from broadcasting results for model quality/length prediction. Latency data is collected separately.

**Main Script**: `prepare_benchmark_data.py` (unified preprocessing)

**Key Features**:
- Auto-detects data format (single-model vs multi-model with `broadcast_results`)
- Auto-detects all models from data (scans all requests)
- Handles incomplete requests (exports to JSONL for re-running)
- Unified scoring (`--scoring-method all`): runs both similarity + LLM judge in one pass
- No latency fields — broadcasting latency is meaningless (concurrent contention); collected separately via `generate_latency_benchmark.py`
- Judge comparison analysis with correlation metrics

**Pipeline**:
1. **Format Detection**: Single-model or multi-model format
2. **Model Collection**: Scan all requests to find all models
3. **Response Filtering**:
   - Remove errors/empty responses
   - Filter by length (min/max tokens)
   - Optional: filter truncated or repetitive responses
4. **Quality Scoring**:
   - **similarity**: Cosine similarity to 72B reference response (sentence-transformers, fast)
   - **llm_judge**: `Qwen/Qwen2.5-7B-Instruct` with 3-stage score parsing (Spearman ρ=0.843)
   - **all** (default): Runs both similarity + llm_judge
   - **compression**: Fast heuristic using zlib ratio
5. **Incomplete Export**: Export incomplete requests as JSONL for re-running
6. **Split**: 80/20 hash-based deterministic split via `split_training_data.py`

**Output Schema**:
```json
{
  "models": {
    "Qwen/Qwen2.5-3B": {
      "output_length": 256, "response": "...",
      "similarity_score": 0.85,
      "llm_judge_scores": {"Qwen_Qwen2.5-7B-Instruct": 7},
      "compression_ratio": 0.45, "is_truncated": false
    }
  }
}
```

**Usage**:
```bash
# Default: both similarity + LLM judge (recommended, requires GPU)
python -m block.predictor.cara.offline_training.prepare_benchmark_data \
  --input data/cara/broadcast_v3_20k.json \
  --scoring-method all \
  --include-response \
  --device cuda

# Similarity only (CPU-friendly)
python -m block.predictor.cara.offline_training.prepare_benchmark_data \
  --input data/cara/broadcast_v3_20k.json \
  --scoring-method similarity \
  --reference-model "Qwen/Qwen2.5-72B"

# LLM judge only
python -m block.predictor.cara.offline_training.prepare_benchmark_data \
  --input data/cara/broadcast_v3_20k.json \
  --scoring-method llm_judge \
  --judge-models Qwen/Qwen2.5-7B-Instruct \
  --device cuda
```

**Output**:
- `{output}.json`: Clean training data with per-model quality scores and output lengths
- `{output}_incomplete.jsonl`: Incomplete requests in benchmark format for re-running
- `{output}_divergent.json`: Samples where judges disagree (if `--compare-judges`)

### 5. Benchmark Client

**Location**: `block/benchmark/cara/benchmark_serving.py`

**Features**:
- CARA-specific backend adapter (`cara_end_point_func.py`)
- Support for ShareGPT, random, and Sonnet datasets
- Selective metric saving: `--save-detailed prompts response ttft e2el models`
- Broadcast result collection in `broadcast_results` field

**Metrics Collected**:
- Per-request: TTFT, E2E latency, output length, model name, instance ID, host
- Aggregate: Mean/P50/P95/P99 latency, throughput, success rate
- Broadcasting: All model responses with individual metrics

### 6. Deployment Automation

**Location**: `block/exp/cara/deploy_cara.py`

**Features**:
- Multi-host deployment via SSH
- Automatic port management (avoids collisions)
- Service-level deployment control: `--deploy-services model_instance predictor`
- Process verification and health checks
- Safe cleanup with anchored kill patterns

**Configuration Hierarchy**:
```
host_config.json               # SSH, IPs, ports (single source of truth)
├── hostname → IP mapping
├── backend_port: 8000
└── predictor_ports: [8100, 8101, ...]

model_deployment.json          # Model → host mapping
├── hf_model_name              # e.g., "Qwen/Qwen2.5-7B"
├── backend: "vllm" or "ollama"
├── precision: "fp16"
└── node_type → deployment settings

predictor_deployment_config.json  # Predictor settings
├── predictor_type: "dummy"
├── enable_data_collection: true
├── heuristic_mode: "min_requests"
└── data_output_dir: "./training_data/cara"
```

---

## Configuration

### Model Deployment Configuration

CARA uses two configuration files for model deployment:

#### 1. Template (`model_config_template.json`) - **You Edit This**

This is the input file where you specify what to deploy:

```json
{
  "Qwen-2.5-7B": {
    "hf_model_name": "Qwen/Qwen2.5-7B",
    "precision": "fp16",
    "backend": "vllm",
    "node_type": {
      "d7525": {
        "count": 5,              // EDIT THIS: How many instances
        "max-model-len": 8192
      }
    }
  }
}
```

**Set `count` for each model**:
- `count: 0` = Don't deploy this model
- `count: N` = Deploy N instances on nodes of this type

#### 2. Deployment Config (`model_deployment.json`) - **Auto-Generated**

This is generated by `deploy_cara.py` and contains actual hostnames:

**File**: `block/config/cara/model_deployment.json`

Example for heterogeneous deployment (generated from template):

```json
{
  "Qwen-2.5-72B": {
    "hf_model_name": "Qwen/Qwen2.5-72B",
    "precision": "fp16",
    "backend": "vllm",
    "node_type": {
      "d8545": {
        "count": 2,
        "gpu-memory-utilization": 0.95,
        "max-model-len": 8192,
        "enforce-eager": true,
        "tensor-parallel-size": 4
      }
    }
  },
  "Qwen-2.5-14B": {
    "hf_model_name": "Qwen/Qwen2.5-14B",
    "precision": "fp16",
    "backend": "vllm",
    "node_type": {
      "c4130": {
        "count": 3,
        "max-model-len": 8192,
        "gpu-memory-utilization": 0.85
      }
    }
  },
  "Qwen-2.5-7B": {
    "hf_model_name": "Qwen/Qwen2.5-7B",
    "precision": "fp16",
    "backend": "vllm",
    "node_type": {
      "d7525": {
        "count": 5,
        "max-model-len": 8192
      }
    }
  },
  "Qwen-2.5-3B": {
    "hf_model_name": "Qwen/Qwen2.5-3B",
    "precision": "fp16",
    "backend": "vllm",
    "node_type": {
      "c240g5": {
        "count": 8,
        "serve_with_v0": true,
        "dtype": "float16",
        "attention_backend": "XFORMERS",
        "enable-chunked-prefill": true   // Required for older vLLM v0 instances
      }
    }
  }
}
```

**Note**: Qwen 2.5 32B is currently excluded due to a repetition/cursor bug that causes malformed outputs.

**Important**:
- **Edit** `model_config_template.json` to set deployment counts
- **Run** `deploy_cara.py` to generate `model_deployment.json` with assigned hosts
- **Use** `model_deployment.json` when starting the CARA scheduler

**Workflow**:
```
model_config_template.json (count=5)
         ↓ (deploy_cara.py)
model_deployment.json (node_hosts=[...])
         ↓ (cara_serve.py)
CARA Scheduler
```

### Predictor Configuration

**File**: `block/config/cara/predictor_deployment_config.json`

```json
{
  "predictor_type": "dummy",
  "backend_port": 8000,
  "schedule_trace_timeout": 5,
  "enable_data_collection": true,
  "data_collection_sample_rate": 1.0,
  "data_output_dir": "./training_data/cara",
  "save_batch_size": 100,
  "heuristic_mode": "min_requests"
}
```

**Heuristic Modes**:
- `min_requests`: Route to instance with fewest total requests
- `max_gpu_blocks`: Route to instance with most free GPU blocks
- `combined`: Weighted combination of both
- `random`: Random baseline

### Host Configuration

**File**: `block/config/host_configs.json`

Generated by `generate_config.py` from CloudLab manifest or manually created.

```json
{
  "hosts": {
    "d8545-node1.cloudlab.us": "192.168.1.1",
    "c4130-node2.cloudlab.us": "192.168.1.2",
    ...
  },
  "backend_port": 8000,
  "predictor_ports": [8100, 8101, 8102, 8103],
  "ssh_username": "your_username"
}
```

---

## Data Collection Pipeline

### Overview

CARA uses a multi-stage data collection pipeline to gather training data for estimators:

1. **Broadcasting**: Collect responses from multiple models for the same prompt
2. **Filtering**: Remove low-quality responses (truncated, repetitive, failed)
3. **Scoring**: Compute quality scores (similarity or LLM judge)
4. **Training**: Use cleaned data to train estimators

### Stage 1: Collect Diverse Prompts

**Location**: `block/data/collect_data.py`

Mix prompts from multiple datasets with balanced ratios:

```bash
python block/data/collect_data.py \
  --datasets ultrachat mix_instruct code_ultra_feedback squad cnn_dailymail ai2_arc gsm8k reward_bench beaver_tails \
  --reward-bench-split filtered \
  --total-n 20000 \
  --max-prompt-length 700 \
  --tokenizer Qwen/Qwen2.5-3B \
  --stats-output data/cara/collection.stats.json \
  -o data/cara/prompts.jsonl
```

**Supported Datasets**:
- `ultrachat`: Benign chat conversations
- `mix_instruct`: General instructions + inputs
- `code_ultra_feedback`: Programming tasks
- `squad`: Question answering with context
- `cnn_dailymail`: Summarization tasks
- `ai2_arc`: Multiple-choice reasoning (Challenge + Easy)
- `gsm8k`: Math word problems
- `reward_bench`: AllenAI evaluation prompts (filtered/raw splits)
- `beaver_tails`: Harmful prompts (for safety testing, keep small share)

**Token-Based Filtering**:
- Filter prompts by token length BEFORE allocation (no truncation)
- Default max: 700 tokens
- Uses real tokenizer for accurate counting

### Stage 2: Run Broadcasting Benchmark

Collect responses from all models for each prompt:

```bash
# Option 1: Local execution
cd block/exp/cara
./test_broadcasting.sh \
  "Qwen/Qwen2.5-3B Qwen/Qwen2.5-7B Qwen/Qwen2.5-14B Qwen/Qwen2.5-72B" \
  sharegpt \
  500 \
  2.0 \
  batch1

# Option 2: Remote execution via SSH
./run_broadcasting_remote.sh \
  "node1.cloudlab.us" \
  "Qwen/Qwen2.5-3B Qwen/Qwen2.5-7B Qwen/Qwen2.5-14B Qwen/Qwen2.5-72B" \
  sharegpt \
  500 \
  2.0 \
  batch1
```

**Output**: Benchmark result JSON with `broadcast_results` field containing all model responses.

### Stage 3: Prepare Training Data

Use the unified preprocessing script to filter, score, and prepare responses:

```bash
# Single-model data with compression scoring (fast, no dependencies)
python -m block.predictor.cara.offline_training.prepare_benchmark_data \
  --input cara_result/cara-benchmark.json \
  --scoring-method compression \
  --dataset-name sharegpt-3b

# Multi-model with similarity scoring
python -m block.predictor.cara.offline_training.prepare_benchmark_data \
  --input data/cara/cara-best-route-training.json \
  --scoring-method similarity \
  --reference-model "Qwen/Qwen2.5-72B" \
  --device cuda

# Multi-model with LLM judge scoring (most accurate)
python -m block.predictor.cara.offline_training.prepare_benchmark_data \
  --input data/cara/cara-best-route-training.json \
  --scoring-method llm_judge \
  --judge-models Qwen/Qwen2.5-0.5B \
  --device cuda \
  --batch-size 8

# Compare multiple judges
python -m block.predictor.cara.offline_training.prepare_benchmark_data \
  --input data/cara/cara-best-route-training.json \
  --judge-models Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-3B \
  --compare-judges \
  --divergence-threshold 0.3
```

**Filtering**:
- Error checking: Remove failed requests
- Empty/short detection: Remove responses with ≤1 or < min_output_tokens
- Truncation detection: Flag (optionally filter) responses hitting max_length
- Repetition detection: Flag (optionally filter) responses with compression ratio < 0.2
- Model coverage: By default, require all expected models (use `--allow-incomplete` to relax)

**Scoring Methods**:

1. **Compression Scoring** (default for single-model):
   - Uses zlib compression ratio as quality proxy
   - Fast, no external dependencies
   - **Best for**: Quick processing, no GPU needed

2. **Similarity Scoring** (requires multi-model data):
   - Uses sentence-transformers embeddings
   - Computes cosine similarity to reference model
   - Requires `--reference-model` argument
   - **Best for**: Comparing model outputs semantically

3. **LLM Judge Scoring** (most accurate):
   - Uses LLM to evaluate quality
   - Supports batched inference (`--batch-size`)
   - Supports multiple judges for comparison (`--compare-judges`)
   - **Best for**: Mixed datasets, safety-critical evaluation

**Important**: Similarity scoring may fail on adversarial prompts where some models refuse and others comply. Use LLM judge for safety-mixed datasets.

**📖 New: LLM Judge Improvements** - See [LLM_JUDGE_IMPROVEMENTS.md](LLM_JUDGE_IMPROVEMENTS.md) for:
- Configurable scoring scales (1-4, 1-10, custom)
- 3-stage parsing with semantic fallback (95% success rate)
- Validation framework against human ratings (feedbackQA)
- Comprehensive statistics tracking
- 21% improvement in correlation with human judges

**Incomplete Requests**: Requests missing expected models are exported to `{output}_incomplete.jsonl` in benchmark format for re-running.

### Stage 4: Analyze Data Quality

```bash
python analyze_training_data.py \
  --output data/cara/best-route_similarity_model_estimation_training.json \
  --input data/cara/cara-best-route-training.json \
  --num-u-shape-inspect 5
```

**Checks**:
- Model coverage (% of requests with each model)
- Score distributions (mean, std per model)
- Pattern analysis (monotonic vs U-shape)
- Example inspection for debugging

**Quality Metrics**:
- Model coverage should be >95% for all models
- Monotonic patterns should be >70%
- U-shape patterns should be <10%

See `data/cara/data_quality_analysis.md` for detailed analysis of common data quality issues.

---

## Training Estimators

### Overview

CARA uses three types of estimators for routing decisions:

1. **Quality Estimator**: Predict model output quality for a given prompt
2. **Length Predictor**: Predict output token length for a given prompt-model pair
3. **Latency Predictor**: Predict E2E latency given queue state and request characteristics

### 1. Quality Estimator (Planned)

**Input**: `(prompt, model_name)`
**Output**: `quality_score` ∈ [0, 1]

**Model Options**:
- KNN baseline: Use prompt embeddings, average quality scores from top-k neighbors
- Lightweight encoder: Frozen embeddings + small MLP/GBDT per model
- Multi-head classifier: Shared encoder + per-model quality heads

**Training Data**: Output from `prepare_training_data.py` with quality scores

**Evaluation Metrics**:
- Best-model selection accuracy vs 72B reference
- Monotonicity by model size (3B < 7B < 14B < 72B)
- Per-dataset breakdown (benign vs adversarial)

### 2. Length Predictor (Planned)

**Input**: `(prompt, model_name, num_prompt_tokens)`
**Output**: `predicted_output_length` (integer)

**Model Options**:
- KNN baseline: Top-k neighbors → E[output_len] and Q90[output_len]
- Regressor: Frozen embeddings + MLP/GBDT
- Quantile regression: Predict both mean and Q90/Q95 for conservative bounds

**Training Data**: Output from `prepare_training_data.py` with output lengths

**Evaluation Metrics**:
- MAE (Mean Absolute Error) < 15 tokens
- RMSE < 25 tokens
- Inference latency < 5ms

**Use Cases**:
- E[output_len] for cost/latency scoring
- MaxSafe[output_len] (Q90-Q95) to enforce 1024-token caps per model

### 3. Latency Predictor (XGBoost with Scheduler-Side Architecture)

> **📖 Full Design Document**: [LATENCY_PREDICTOR_DESIGN.md](LATENCY_PREDICTOR_DESIGN.md)
> **📖 vLLM API Specification**: [INSTANCE_STATS_API.md](INSTANCE_STATS_API.md)

**Architecture Change**: The latency predictor has been redesigned from LSTM-based sidecar to **XGBoost-based scheduler-side** with **per-batch state queries**.

| Aspect | Old Design | New Design |
|--------|------------|------------|
| Model | LSTM (sequence) | XGBoost (tree-based) |
| Location | Sidecar (instance-side) | Scheduler-side |
| State Query | Per-request | Per-batch |
| Features | Per-request details | Aggregated instance state |

**Input**: `(aggregated_schedule_state, target_request)`
**Output**: `predicted_e2e_latency`

**Aggregated State Features** (from `/instance_stats` endpoint):
```python
{
    # Batch state
    "num_running": int,                    # Running requests
    "num_waiting": int,                    # Pending requests
    "num_active_decode_seqs": int,         # Sequences in decode phase

    # Context length distribution (decode sequences)
    "decode_ctx_mean": int,                # Mean context length
    "decode_ctx_p50": int,                 # Median context length
    "decode_ctx_p95": int,                 # 95th percentile
    "decode_ctx_max": int,                 # Max context length
    "total_decode_context_tokens": int,    # Sum of all context lengths

    # Backlog in tokens (all requests)
    "pending_prefill_tokens": int,         # Unprocessed prefill work
    "pending_decode_tokens": int,          # Remaining decode work

    # Memory pressure
    "kv_cache_utilization": float,         # 0.0 - 1.0
    "kv_free_blocks": int,                 # Free KV cache blocks

    # Scheduler config
    "token_budget_per_iter": int,          # Max tokens per iteration
    "max_num_seqs": int,                   # Max sequences in batch

    # Metadata
    "num_preempted": int,
    "timestamp": float,
}
```

> **Note**: EMA service rates (`ema_decode_tok_per_s`, `ema_prefill_tok_per_s`) are computed **scheduler-side** from completed request feedback, not provided by vLLM. See [INSTANCE_STATS_API.md](INSTANCE_STATS_API.md) for details.

**Per-Batch Query Flow**:
```
Batch arrives → Query all instances in parallel (one RTT)
     │
     ▼
Update local state cache → Predict locally for each request (~μs)
     │
     ▼
Route requests to best instances
```

**Why XGBoost over LSTM?**
- With **chunked prefill enabled**, latency is predictable from aggregate stats
- **Fixed-size features** (no variable-length sequences)
- **Fast inference** (~μs vs ~ms)
- **Simpler deployment** (single model file)

**Success Metrics**:
- E2E MAE < 200ms
- Inference latency < 100μs per request
- Per-batch query < 50ms

**Implementation**:
- vLLM endpoint: `/instance_stats` ✅ (see [INSTANCE_STATS_API.md](INSTANCE_STATS_API.md))
- Predictor: `block/predictor/cara/xgboost_predictor.py` (to be created)
- Scheduler service: `block/predictor/cara/scheduler_predictor_service.py` (to be created)
- Training: `block/predictor/cara/offline_training/train_xgboost.py` (to be created)

---

## Deployment

### Single-Node Testing (Local)

For development and testing on a single machine:

```bash
# 1. Start vLLM instance
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B \
  --port 8000 \
  --dtype float16 \
  --max-model-len 8192

# 2. Start predictor
python -m block.predictor.cara.cara_predictor_api_server \
  --host 0.0.0.0 \
  --port 8100 \
  --backend-port 8000 \
  --config-path block/config/cara/predictor_config_dummy.json

# 3. Start CARA scheduler
python -m block.global_scheduler.cara.cara_serve \
  --host 0.0.0.0 \
  --port 8200 \
  --model_config_path block/config/cara/model_deployment.json \
  --host_config block/config/host_configs.json

# 4. Run benchmark
python -m block.benchmark.cara.benchmark_serving \
  --backend cara \
  --base-url http://localhost:8200 \
  --model dummy \
  --dataset-name sharegpt \
  --dataset-path data/trace_data/sharegpt/sharegpt_gpt4.jsonl \
  --num-prompts 10 \
  --request-rate 1.0
```

### Multi-Node Deployment (CloudLab)

For heterogeneous deployment across multiple GPU types:

#### Step 1: Generate Configuration

```bash
# Download CloudLab manifest.xml
# Place in block/ directory

python block/exp/generate_config.py \
  --username YOUR_SSH_USERNAME \
  --num_predictors 1
```

This generates:
- `block/config/host_configs.json`: Host IPs, ports, SSH config
- `block/config/{ampere,volta,pascal}_hosts`: Node type lists

#### Step 2: Edit Template and Generate Deployment Config

```bash
cd block/exp/cara

# 1. Edit the template to set deployment counts
vi ../../config/cara/model_config_template.json
# Set "count": N for models you want to deploy

# 2. Generate deployment config (assigns actual hosts)
python deploy_cara.py \
  --config ../../config/cara/model_config_template.json \
  --output ../../config/cara/model_deployment.json \
  --hosts ../../config/hosts \
  --deploy-services none

# 3. Review the generated config
cat ../../config/cara/model_deployment.json
# Verify that node_hosts are assigned correctly
```

#### Step 3: Deploy Infrastructure

```bash
# Deploy backends and predictors
python deploy_cara.py \
  --config ../../config/cara/model_config_template.json \
  --output ../../config/cara/model_deployment.json \
  --hosts ../../config/hosts \
  --deploy-services model_instance predictor \
  --hf-token YOUR_HF_TOKEN

# Verify deployment
python e2e/check_deployment.py \
  --model_config_path ../../config/cara/model_deployment.json \
  --host_config ../../config/host_configs.json
```

#### Step 4: Start CARA Scheduler

On the scheduler node (e.g., head node):

```bash
python -m block.global_scheduler.cara.cara_serve \
  --host 0.0.0.0 \
  --port 8200 \
  --model_config_path block/config/cara/model_deployment.json \
  --host_config block/config/host_configs.json \
  --scheduling random \
  --enable-predictor-feedback
```

#### Step 5: Run Experiments

```bash
# Simple test
cd block/exp/cara
./test.sh

# Broadcasting data collection
./test_broadcasting.sh \
  "Qwen/Qwen2.5-3B Qwen/Qwen2.5-7B Qwen/Qwen2.5-14B Qwen/Qwen2.5-72B" \
  sharegpt \
  500 \
  2.0 \
  batch1

# Full end-to-end experiment
./e2e/experiment.sh
```

### CloudLab Node Type Mapping

Current deployment targets:

| Model | GPU Type | Node Type | Count | TP | Notes |
|-------|----------|-----------|-------|-----|-------|
| Qwen 2.5 72B | A100×4 | d8545 | 2 | 4 | In-node tensor parallelism |
| Qwen 2.5 14B | V100×4 | c4130 | 3 | 1 | Adjust precision if needed |
| Qwen 2.5 7B | A30×1 | d7525 | 5 | 1 | Standard config |
| Qwen 2.5 3B | P100×1 | c240g5 | 8 | 1 | vLLM v0, `enable-chunked-prefill: true` |

**Note**: Qwen 2.5 32B excluded due to repetition/cursor bug.

**P100 Configuration**: P100 nodes use older vLLM v0 which requires manual chunked prefill enablement. Set `"enable-chunked-prefill": true` in the model config for consistent latency predictor behavior.

### Deployment Troubleshooting

**Issue**: Deployment script hangs or fails

```bash
# Check SSH connectivity
ssh user@node1.cloudlab.us "echo OK"

# Verify host_config.json has correct IPs and ports
cat block/config/host_configs.json

# Check for port collisions
ssh user@node1.cloudlab.us "netstat -tuln | grep 8000"
```

**Issue**: Predictor not responding

```bash
# Check predictor logs
ssh user@node1.cloudlab.us "tail -100 experiment_output/logs/predictor_output.log"

# Verify predictor process is running
ssh user@node1.cloudlab.us "ps aux | grep cara_predictor_api_server"

# Test predictor health endpoint
curl http://node1.cloudlab.us:8100/health
```

**Issue**: vLLM instance not serving

```bash
# Check vLLM logs
ssh user@node1.cloudlab.us "tail -100 experiment_output/logs/vllm_output.log"

# Verify GPU availability
ssh user@node1.cloudlab.us "nvidia-smi"

# Test vLLM health endpoint
curl http://node1.cloudlab.us:8000/health
```

---

## Benchmarking

### Basic Benchmarking

```bash
python -m block.benchmark.cara.benchmark_serving \
  --backend cara \
  --base-url http://localhost:8200 \
  --model dummy \
  --dataset-name sharegpt \
  --dataset-path data/trace_data/sharegpt/sharegpt_gpt4.jsonl \
  --num-prompts 100 \
  --request-rate 2.0
```

### Save Detailed Results

```bash
python -m block.benchmark.cara.benchmark_serving \
  --backend cara \
  --base-url http://localhost:8200 \
  --model dummy \
  --dataset-name sharegpt \
  --dataset-path data/trace_data/sharegpt/sharegpt_gpt4.jsonl \
  --num-prompts 100 \
  --request-rate 2.0 \
  --save-detailed prompts response ttft e2el models \
  --save-result
```

**Detailed Fields**:
- `prompts`: Save input prompts
- `response`: Save generated text
- `ttft`: Time-to-first-token
- `e2el`: End-to-end latency
- `models`: Model names and instance IDs

### Broadcasting Benchmark

Collect responses from multiple models:

```bash
# Start CARA in broadcasting mode
python -m block.global_scheduler.cara.cara_serve \
  --host 0.0.0.0 \
  --port 8200 \
  --model_config_path block/config/cara/model_deployment.json \
  --host_config block/config/host_configs.json \
  --broadcasting \
  --selected-broadcasted-models "Qwen/Qwen2.5-3B" "Qwen/Qwen2.5-7B" "Qwen/Qwen2.5-14B" "Qwen/Qwen2.5-72B"

# Run benchmark
python -m block.benchmark.cara.benchmark_serving \
  --backend cara \
  --base-url http://localhost:8200 \
  --model dummy \
  --dataset-name sharegpt \
  --dataset-path data/trace_data/sharegpt/sharegpt_gpt4.jsonl \
  --num-prompts 500 \
  --request-rate 2.0 \
  --save-detailed prompts response ttft e2el models \
  --save-result
```

**Output Format**:
```json
{
  "completed": 500,
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
        {
          "model": "Qwen/Qwen2.5-3B",
          "response": "...",
          "ttft": 0.3,
          "e2el": 1.8,
          "output_len": 100,
          "instance_id": "Qwen-2.5-3B_0",
          "host": "c240g5-1.cloudlab.us"
        },
        {
          "model": "Qwen/Qwen2.5-7B",
          "response": "...",
          "ttft": 0.45,
          "e2el": 2.34,
          "output_len": 95,
          "instance_id": "Qwen-2.5-7B_0",
          "host": "d7525-2.cloudlab.us"
        },
        ...
      ]
    }
  ]
}
```

### Metrics Collected

**Per-Request Metrics**:
- `ttft`: Time-to-first-token (seconds)
- `e2el`: End-to-end latency (seconds)
- `output_len`: Number of output tokens
- `model`: Model name that served the request
- `instance_id`: Backend instance identifier
- `host`: Hostname of the instance
- `server_latency`: Server-side latency excluding network/client overhead

**Aggregate Metrics**:
- Mean/P50/P95/P99 TTFT
- Mean/P50/P95/P99 E2E latency
- Throughput (requests/sec)
- Success rate
- Inter-token latency (ITL) statistics

---

## Development Status

### Phase 1: Data Collection Infrastructure (✅ Complete)

**Completed Features**:
- Predictor integration into CARA scheduler
- Automatic feedback loop (predictions → execution → actual metrics → training data)
- Broadcasting mode for multi-model response collection
- Training data collector with automatic matching and batched disk writes
- Deployment automation for multi-host clusters
- End-to-end testing scripts with remote execution

**Key Files**:
- `block/global_scheduler/cara/cara_serve.py`: Scheduler with predictor queries
- `block/global_scheduler/cara/cara_instance/Instance.py`: Predictor feedback mechanism
- `block/predictor/cara/cara_predictor_api_server.py`: Predictor API server
- `block/predictor/cara/training_data_collector.py`: Training data logging
- `block/exp/cara/deploy_cara.py`: Deployment automation

### Phase 2: Model Training (🚧 In Progress)

**Current Work**:
- Offline training data preparation with filtering and scoring
- Quality estimation using similarity and LLM judge scoring (✨ **newly improved**)
- Judge model validation against human ratings (✨ **new**)
- Length prediction infrastructure
- Data quality analysis and validation

**In Progress**:
- KNN-based quality estimator
- Length predictor training pipeline
- LSTM latency predictor architecture

**Key Files**:
- `block/predictor/cara/offline_training/prepare_training_data.py`: Data preparation
- `block/predictor/cara/offline_training/similarity_scorer.py`: Similarity scoring
- `block/predictor/cara/offline_training/llm_judge_scorer.py`: LLM judge scoring (✨ **enhanced**)
- `block/predictor/cara/offline_training/validate_judge_models.py`: Judge validation (✨ **new**)
- `data/cara/data_quality_analysis.md`: Data quality insights
- `cara_paper/claude/LLM_JUDGE_IMPROVEMENTS.md`: Judge improvements documentation (✨ **new**)

### Phase 3: Production Integration (📋 Planned)

**Upcoming Features**:
- Greedy mini-batch scheduler with multi-objective optimization
- Request-specific objectives (RSO) with per-request constraints
- Learned predictor integration (replace dummy heuristics)
- Adaptive batching based on load
- A/B testing framework

**Planned Files**:
- `block/global_scheduler/cara/batch_scheduler.py`: LPT-based greedy batch scheduler
- `block/predictor/cara/xgboost_predictor.py`: XGBoost latency predictor
- `block/predictor/cara/offline_training/train_xgboost.py`: XGBoost training pipeline
- `block/predictor/cara/service_rate_tracker.py`: Scheduler-side EMA rate tracking
- `block/model_estimator/best_route_client.py`: Quality estimator client
- `block/config/cara/scheduler_config.json`: Scheduling weights and constraints

### Roadmap

**Q1 2026**:
- ✅ Complete `/instance_stats` API in vLLM
- Complete quality estimator (KNN + lightweight encoder)
- Complete length predictor (KNN/regressor)
- Train XGBoost latency predictor (replaced LSTM)
- Implement LPT-based greedy batch scheduler (replaced mini-batch permutation)

**Q2 2026**:
- Replace dummy predictor with trained models
- Implement RSO parsing and constraint handling
- Load-aware adaptive batching
- A/B testing vs baselines

**Q3 2026**:
- Online learning and periodic retraining
- Multi-tenant support with fairness constraints
- Cost optimization
- Auto-scaling based on predicted load

---

## API Reference

### CARA Scheduler API

**Base URL**: `http://localhost:8200` (default)

#### POST /v1/completions

Submit a completion request to CARA scheduler.

**Request**:
```json
{
  "request_id": "demo-1",
  "prompt": "Write a haiku about GPU schedulers.",
  "prompt_len": 7,
  "max_tokens": 64,
  "temperature": 0.7,
  "top_p": 0.9
}
```

**Response** (non-broadcasting):
```json
{
  "request_id": "demo-1",
  "generated_text": "Silicon minds think\nDistributing work with care\nLatency reduced",
  "ttft": 0.045,
  "itl": [0.012, 0.013, 0.011, ...],
  "output_tokens": 15,
  "success": true,
  "model": "Qwen/Qwen2.5-7B",
  "server_latency": 0.234,
  "instance_id": "Qwen-2.5-7B_0",
  "host": "d7525-2.cloudlab.us"
}
```

**Response** (broadcasting mode):
```json
{
  "request_id": "demo-1",
  "generated_text": "...",
  "ttft": 0.045,
  "output_tokens": 15,
  "model": "Qwen/Qwen2.5-7B",
  "broadcast_results": [
    {
      "model": "Qwen/Qwen2.5-3B",
      "generated_text": "...",
      "ttft": 0.032,
      "output_tokens": 18,
      "instance_id": "Qwen-2.5-3B_0"
    },
    {
      "model": "Qwen/Qwen2.5-7B",
      "generated_text": "...",
      "ttft": 0.045,
      "output_tokens": 15,
      "instance_id": "Qwen-2.5-7B_0"
    },
    ...
  ]
}
```

#### GET /health

Health check endpoint.

**Response**:
```json
{
  "status": "healthy",
  "num_instances": 18,
  "scheduling_policy": "random"
}
```

### Predictor API

**Base URL**: `http://localhost:8100` (default)

#### POST /predict

Get latency prediction for a request.

**Request**:
```json
{
  "request_id": "123",
  "num_prompt_tokens": 128,
  "num_predicted_output_tokens": 64
}
```

**Response**:
```json
{
  "target_metric": 5.0,
  "gpu_blocks": 4500,
  "num_requests": 5,
  "num_preempted": 0,
  "predictor_type": "dummy_cara",
  "time_to_predict": 12.34
}
```

#### POST /log_actual

Log actual metrics (feedback from instance).

**Request**:
```json
{
  "request_id": "123",
  "e2e_latency": 2.456,
  "ttft": 0.123,
  "tpot": 0.023
}
```

**Response**:
```json
{
  "status": "logged"
}
```

#### GET /stats

Get predictor statistics.

**Response**:
```json
{
  "total_predictions": 1000,
  "total_completed": 950,
  "total_saved": 900,
  "pending": 50,
  "buffered": 50,
  "last_save_time": 1702345680.579
}
```

#### POST /flush

Force write buffered training data to disk.

**Response**:
```json
{
  "status": "flushed",
  "examples_written": 50
}
```

#### GET /health

Health check endpoint.

**Response**:
```json
{
  "status": "healthy",
  "predictor_type": "dummy_cara",
  "backend_port": 8000
}
```

---

## Troubleshooting

### Common Issues

#### 0. Model Config Not Found / Wrong Format

**Symptoms**:
- `deploy_cara.py` fails with "node_type not found"
- CARA scheduler can't find model instances
- Config has `count` fields instead of `node_hosts`

**Diagnosis**:
```bash
# Check if you're using template instead of generated config
cat block/config/cara/model_deployment.json | grep -c "node_type"
# Should be 0 (no node_type fields)

cat block/config/cara/model_deployment.json | grep -c "node_hosts"
# Should be > 0 (has node_hosts)
```

**Solution**:
```bash
# 1. Edit template to set deployment counts
vi block/config/cara/model_config_template.json
# Set "count": N for models you want

# 2. Generate actual deployment config
python block/exp/cara/deploy_cara.py \
  --config block/config/cara/model_config_template.json \
  --output block/config/cara/model_deployment.json \
  --hosts block/config/hosts \
  --deploy-services none

# 3. Verify generated config has node_hosts
cat block/config/cara/model_deployment.json

# 4. Use model_deployment.json (NOT template) for scheduler
python -m block.global_scheduler.cara.cara_serve \
  --model_config_path block/config/cara/model_deployment.json \
  ...
```

**Key Point**:
- `model_config_template.json` = INPUT (you edit, has `count` fields)
- `model_deployment.json` = OUTPUT (generated, has `node_hosts` arrays)
- Always use `model_deployment.json` for the scheduler!

#### 1. Predictor Not Receiving Feedback

**Symptoms**:
- `total_completed` in `/stats` not increasing
- Training data files empty or not created

**Diagnosis**:
```bash
# Check if feedback is enabled
grep "enable_predictor_feedback" block/config/cara/predictor_config_dummy.json

# Check predictor logs
tail -100 experiment_output/logs/predictor_output.log

# Verify predictor ports in instance creation
grep "predictor_ports" block/config/host_configs.json
```

**Solutions**:
- Ensure `enable_predictor_feedback=True` when creating instances
- Verify predictor ports match between host config and instance initialization
- Check predictor is running: `curl http://localhost:8100/health`

#### 2. Training Data Not Being Saved

**Symptoms**:
- `buffered` count in `/stats` increasing but `total_saved` not
- No JSONL files in `training_data/cara/`

**Diagnosis**:
```bash
# Check data collection config
grep "enable_data_collection" block/config/cara/predictor_config_dummy.json

# Check output directory permissions
ls -ld training_data/cara/

# Force flush
curl -X POST http://localhost:8100/flush
```

**Solutions**:
- Verify `enable_data_collection: true` in predictor config
- Ensure `data_output_dir` exists and is writable
- Reduce `save_batch_size` to flush more frequently
- Call `/flush` endpoint before shutdown

#### 3. /schedule_trace Timeout

**Symptoms**:
- Predictor logs show "schedule_trace timeout"
- Predictions fail or use fallback heuristics

**Diagnosis**:
```bash
# Test /schedule_trace endpoint directly
curl http://localhost:8000/schedule_trace

# Check vLLM logs for errors
tail -100 experiment_output/logs/vllm_output.log
```

**Solutions**:
- Increase `schedule_trace_timeout` in predictor config
- Verify vLLM instance is running and healthy
- Check network connectivity between predictor and backend
- Ensure vLLM version includes `/schedule_trace` endpoint (custom build required)

#### 4. Broadcasting Returns Incomplete Results

**Symptoms**:
- `broadcast_results` has fewer models than expected
- Some models consistently missing from results

**Diagnosis**:
```bash
# Check model deployment status
python block/exp/cara/e2e/check_deployment.py \
  --model_config_path block/config/cara/model_deployment.json \
  --host_config block/config/host_configs.json

# Check scheduler logs
tail -100 experiment_output/logs/cara_serve.log
```

**Solutions**:
- Verify all target models are deployed and healthy
- Check `--selected-broadcasted-models` matches deployed model names
- Ensure model names use consistent format (HuggingFace names)
- Check for instance failures in deployment logs

#### 5. Poor Data Quality (U-Shape Patterns)

**Symptoms**:
- High percentage of U-shape score patterns in analysis
- Quality scores don't correlate with model size
- Low model coverage (<95%)

**Diagnosis**:
```bash
# Run data quality analysis
python analyze_training_data.py \
  --output data/cara/best-route_similarity_model_estimation_training.json \
  --num-u-shape-inspect 5

# Check dataset composition
cat data/cara/best-route.stats.json
```

**Solutions**:
- For adversarial/safety prompts: Use `--scoring-method llm_judge` instead of similarity
- Ensure all models deployed (check 3B coverage issue)
- Fix 32B repetition bug (update prompt template or model config)
- Filter to only requests with all models: `--require-all-models`
- See `data/cara/data_quality_analysis.md` for detailed troubleshooting

#### 6. Deployment Script Hangs

**Symptoms**:
- `deploy_cara.py` hangs during remote deployment
- SSH commands not returning

**Diagnosis**:
```bash
# Test SSH connectivity
ssh user@node1.cloudlab.us "echo OK"

# Check for stuck processes
ssh user@node1.cloudlab.us "ps aux | grep python"
```

**Solutions**:
- Verify SSH keys are set up correctly (passwordless login)
- Check network connectivity to all nodes
- Use `--deploy-services` to deploy incrementally (first `model_instance`, then `predictor`)
- Kill stuck processes manually and retry

### Logging and Debugging

**Enable Debug Logging**:

```bash
# CARA Scheduler
python -m block.global_scheduler.cara.cara_serve \
  --log-level DEBUG \
  ...

# Predictor
python -m block.predictor.cara.cara_predictor_api_server \
  --debug \
  ...

# Data Preparation
python -m block.predictor.cara.offline_training.prepare_training_data \
  --debug \
  ...
```

**Log Locations**:
- CARA Scheduler: `experiment_output/logs/cara_serve.log`
- Predictor: `experiment_output/logs/predictor_output.log`
- vLLM: `experiment_output/logs/vllm_output.log`
- Deployment: `experiment_output/logs/deploy.log`

**Check System Resources**:

```bash
# GPU usage
nvidia-smi

# Memory usage
free -h

# Disk space
df -h

# Network connections
netstat -tuln | grep -E '8000|8100|8200'
```

---

## References

### Related Projects

- **Block**: Predictive scheduling for LLM serving with Vidur simulation
  - Repository: `block/` (parent directory)
  - Paper: "Block: Balance Loader of LLM Serving with Context and Knowledge"

- **vLLM**: High-performance LLM inference engine
  - CARA-compatible fork: https://anonymous.4open.science/r/vllm-644A
  - Custom `/schedule_trace` endpoint for queue state monitoring

- **Vidur**: LLM serving simulator
  - Used in Block for simulation-based prediction
  - CARA uses learned predictors instead

### Documentation

- **CARA Project Status**: `cara_paper/active/CARA_PROJECT_STATUS.md`
  - Comprehensive implementation status and future roadmap
  - Developer guide for common tasks

- **Codex Documents**: `cara_paper/codex/`
  - `codex-cara-plan.md`: Implementation roadmap and milestones
  - `cara-objective-design.md`: Objective function design for multi-objective optimization
  - `cara-data-and-experiments.md`: Data collection and experimental plan

- **Predictor README**: `block/predictor/cara/README.md`
  - Predictor API documentation
  - Training data format specification

- **Offline Training README**: `block/predictor/cara/offline_training/README.md`
  - Data preparation pipeline documentation
  - Filtering and scoring methods

- **Data Quality Analysis**: `data/cara/data_quality_analysis.md`
  - Detailed analysis of common data quality issues
  - Recommendations for fixing coverage and scoring problems

- **LLM Judge Improvements**: `cara_paper/claude/LLM_JUDGE_IMPROVEMENTS.md`
  - Configurable scoring scales (1-4, 1-10, custom)
  - 3-stage parsing system (number, exact match, semantic)
  - Validation framework against human ratings (feedbackQA)
  - Comprehensive statistics and confidence metrics
  - Performance comparison and best practices

- **Latency Predictor Design**: `cara_paper/claude/LATENCY_PREDICTOR_DESIGN.md`
  - XGBoost-based scheduler-side architecture
  - Per-batch query design (vs per-request)
  - Aggregated feature engineering
  - Training pipeline and implementation plan

- **Instance Stats API**: `cara_paper/claude/INSTANCE_STATS_API.md`
  - vLLM `/instance_stats` endpoint specification
  - Context length distribution (mean, p50, p95, max)
  - Backlog calculation for all requests (running + waiting + preempted)
  - `predicted_decode_tokens` field with auto-extension
  - EMA service rates computed scheduler-side

- **Implementation Tasks V2 Design**: `cara_paper/claude/IMPLEMENTATION_TASKS_V2_DESIGN.md` ✨ **NEW**
  - Detailed designs for 12 pending review fixes
  - Streaming batch response, XGBoost target field fixes
  - DataCollectionService rename, ServiceRateTracker decoupling
  - deploy_cara.py updates, E2E training pipeline
  - CDF error plots, learning curve reporting
  - Bootstrap benchmark mode with dummy prompts

- **Submission Plan**: `cara_paper/active/SUBMISSION_PLAN.md`
  - Paper positioning and novelty analysis
  - Venue analysis (SoCC, ATC, EuroSys, NSDI)
  - Reviewer concerns and responses
  - Evaluation plan and required experiments
  - Submission timeline and action items

### Papers and Research

- **CARA (This Work)**: Co-optimizing Super Heterogeneous LLM Serving
  - Multi-objective routing with quality, latency, and cost optimization
  - Learned predictors for forward-compatible scheduling
  - Closed-loop learning from production traffic

- **Related Work**:
  - DistServe: Disaggregated serving for LLMs
  - Clockwork: Predictable scheduling for ML inference
  - Nexus: Multi-model serving with learned predictions
  - Best-Route: LLM routing based on prompt quality estimation

### Contact and Support

For questions, issues, or contributions:

- **Issue Tracker**: File issues in the repository
- **Documentation**: Check `cara_paper/active/CARA_PROJECT_STATUS.md` for latest status
- **Code Review**: All code under `block/global_scheduler/cara/` and `block/predictor/cara/`

---

## Acknowledgments

CARA is built on top of the Block framework and leverages vLLM for high-performance inference. The training data pipeline is inspired by Microsoft's Best-Route prompt routing work.

**License**: MIT (see `LICENSE` file)

**Version**: Phase 2 In Progress (January 26, 2026)

---

**Happy Scheduling!**