# Block Profiling Guide

This document describes how to profile new models and hardware configurations for use with Block's Vidur-based latency simulator.

## Overview

Block uses profiling data to accurately predict LLM inference latency. The profiling captures:
- **Compute profiles**: MLP and attention kernel execution times
- **Network profiles**: Collective communication times (all_reduce, send_recv)

Profiling must be done on actual hardware, but once complete, simulations can run on CPUs only.

---

## Structure of Profiling Data

Profiling data is stored in the `data/profiling` directory:

```
data/profiling/
├── compute/
│   ├── a100_40gb/
│   │   └── meta-llama/
│   │       └── Llama-2-70b-hf/
│   │           ├── mlp.csv
│   │           └── attention.csv
│   └── a30/
│       └── meta-llama/
│           └── Llama-2-7b-hf/
│               ├── mlp.csv
│               └── attention.csv
└── network/
    ├── a100_40gb_pairwise_nvlink/
    │   ├── all_reduce.csv
    │   └── send_recv.csv
    └── a30_pcie/
        ├── all_reduce.csv
        └── send_recv.csv
```

### Naming Conventions

- **Compute profiles**: Named by GPU SKU (e.g., `a100_40gb`, `a30`, `h100`)
  - Same compute data can be used regardless of network topology
- **Network profiles**: Named by GPU SKU + network topology (e.g., `a100_40gb_pairwise_nvlink`, `a30_pcie`)
  - Different topologies have different collective performance characteristics

---

## Prerequisites

### 1. Hardware Requirements

| Profile Type | Minimum GPUs | Notes |
|-------------|--------------|-------|
| MLP | 1 | Can profile any TP degree with 1 GPU |
| Attention | 1 | Can profile any TP degree with 1 GPU |
| all_reduce | N | Requires N GPUs for TP=N profiling |
| send_recv | 2 | Point-to-point between 2 GPUs |

### 2. Software Requirements

- CUDA 12.6+
- PyTorch 2.6.0+
- Block repository with Vidur profiling code
- sarathi-serve-for-block (vidur branch)

---

## Setup Instructions

### Step 1: Install sarathi-serve-for-block

The profiling scripts require sarathi-serve. Use the forked version with fixed dependencies:

```bash
# Clone sarathi-serve-for-block (vidur branch)
cd ~
git clone https://github.com/AKafakA/sarathi-serve-for-block.git sarathi-serve
cd sarathi-serve
git checkout vidur

# Fix pyproject.toml for PyTorch 2.6 compatibility
# Change "torch == 2.3" to "torch >= 2.3" in build-system requires
sed -i 's/"torch == 2.3"/"torch >= 2.3"/g' pyproject.toml

# Install sarathi-serve (use --no-build-isolation to use installed torch)
pip install --no-cache-dir --no-build-isolation -e .

# Verify installation
python -c "import sarathi; print('sarathi-serve installed successfully')"
```

### Step 2: Setup Block Environment

```bash
cd ~/Block
export PYTHONPATH=.
export HF_TOKEN="YOUR_HUGGINGFACE_TOKEN"  # Required for gated models
```

### Step 3: Create Output Directories

```bash
# Replace {gpu_sku} and {model_path} with your values
mkdir -p data/profiling/compute/{gpu_sku}/{model_path}
mkdir -p data/profiling/network/{network_device}

# Example for A100-40GB with Llama-70B:
mkdir -p data/profiling/compute/a100_40gb/meta-llama/Llama-2-70b-hf
mkdir -p data/profiling/network/a100_40gb_pairwise_nvlink
```

---

## Compute Profiling

### MLP Profiling

Profiles MLP kernel execution times (up_proj, act, down_proj, etc.).

```bash
python vidur/profiling/mlp/main.py \
    --models "meta-llama/Llama-2-70b-hf" \
    --num_gpus 4 \
    --num_tensor_parallel_workers 4 \
    --max_tokens 4096 \
    --output_dir "./profiling_tmp/mlp"
```

**Arguments:**
| Argument | Description | Example |
|----------|-------------|---------|
| `--models` | HuggingFace model ID | `meta-llama/Llama-2-70b-hf` |
| `--num_gpus` | Number of GPUs to use | `4` |
| `--num_tensor_parallel_workers` | TP degree to profile | `4` |
| `--max_tokens` | Maximum tokens to profile | `4096` |
| `--output_dir` | Output directory | `./profiling_tmp/mlp` |

**Copy results:**
```bash
cp ./profiling_tmp/mlp/mlp/*/meta-llama/Llama-2-70b-hf/mlp.csv \
   data/profiling/compute/a100_40gb/meta-llama/Llama-2-70b-hf/mlp.csv
```

### Attention Profiling

Profiles attention kernel execution times (prefill and decode).

```bash
python vidur/profiling/attention/main.py \
    --models "meta-llama/Llama-2-70b-hf" \
    --num_gpus 4 \
    --num_tensor_parallel_workers 4 \
    --max_model_len 4096 \
    --output_dir "./profiling_tmp/attention"
```

**Important:** Use `--max_model_len` (not `--max_tokens`) for attention profiling.

**Arguments:**
| Argument | Description | Example |
|----------|-------------|---------|
| `--models` | HuggingFace model ID | `meta-llama/Llama-2-70b-hf` |
| `--num_gpus` | Number of GPUs to use | `4` |
| `--num_tensor_parallel_workers` | TP degree to profile | `4` |
| `--max_model_len` | Maximum sequence length | `4096` |
| `--output_dir` | Output directory | `./profiling_tmp/attention` |

**Copy results:**
```bash
cp ./profiling_tmp/attention/attention/*/meta-llama/Llama-2-70b-hf/attention.csv \
   data/profiling/compute/a100_40gb/meta-llama/Llama-2-70b-hf/attention.csv
```

---

## Network (Collectives) Profiling

Network profiling is **model-independent** - the same data can be used for all models on the same hardware configuration.

### all_reduce Profiling (for Tensor Parallelism)

Profiles all_reduce collective operations used in tensor parallelism.

```bash
python vidur/profiling/collectives/main.py \
    --num_workers_per_node_combinations 4 \
    --collective all_reduce \
    --output_dir "./profiling_tmp/collectives"
```

**Arguments:**
| Argument | Description | Example |
|----------|-------------|---------|
| `--num_workers_per_node_combinations` | Number of GPUs per node | `4` (for 4-GPU node) |
| `--collective` | Type of collective | `all_reduce` |
| `--output_dir` | Output directory | `./profiling_tmp/collectives` |

**Copy results:**
```bash
cp ./profiling_tmp/collectives/collective/*/all_reduce.csv \
   data/profiling/network/a100_40gb_pairwise_nvlink/all_reduce.csv
```

### send_recv Profiling (for Pipeline Parallelism)

Profiles point-to-point send/recv operations used in pipeline parallelism.

**Important:** send_recv requires exactly 2 workers.

```bash
python vidur/profiling/collectives/main.py \
    --num_workers_per_node_combinations 2 \
    --collective send_recv \
    --output_dir "./profiling_tmp/collectives"
```

**Copy results:**
```bash
cp ./profiling_tmp/collectives/collective/*/send_recv.csv \
   data/profiling/network/a100_40gb_pairwise_nvlink/send_recv.csv
```

---

## Quick Reference

### Complete Profiling Workflow

```bash
# 1. Setup environment
cd ~/Block
export PYTHONPATH=.
export HF_TOKEN="your_token"

# 2. Create directories
mkdir -p data/profiling/compute/a100_40gb/meta-llama/Llama-2-70b-hf
mkdir -p data/profiling/network/a100_40gb_pairwise_nvlink

# 3. MLP profiling (~5 min)
python vidur/profiling/mlp/main.py \
    --models "meta-llama/Llama-2-70b-hf" \
    --num_gpus 4 --num_tensor_parallel_workers 4 \
    --max_tokens 4096 --output_dir "./profiling_tmp/mlp"

# 4. Attention profiling (~2 min)
python vidur/profiling/attention/main.py \
    --models "meta-llama/Llama-2-70b-hf" \
    --num_gpus 4 --num_tensor_parallel_workers 4 \
    --max_model_len 4096 --output_dir "./profiling_tmp/attention"

# 5. all_reduce profiling (~5 min)
python vidur/profiling/collectives/main.py \
    --num_workers_per_node_combinations 4 \
    --collective all_reduce --output_dir "./profiling_tmp/collectives"

# 6. send_recv profiling (~5 min)
python vidur/profiling/collectives/main.py \
    --num_workers_per_node_combinations 2 \
    --collective send_recv --output_dir "./profiling_tmp/collectives"

# 7. Copy results
cp ./profiling_tmp/mlp/mlp/*/meta-llama/Llama-2-70b-hf/mlp.csv \
   data/profiling/compute/a100_40gb/meta-llama/Llama-2-70b-hf/
cp ./profiling_tmp/attention/attention/*/meta-llama/Llama-2-70b-hf/attention.csv \
   data/profiling/compute/a100_40gb/meta-llama/Llama-2-70b-hf/
cp ./profiling_tmp/collectives/collective/*/all_reduce.csv \
   data/profiling/network/a100_40gb_pairwise_nvlink/
cp ./profiling_tmp/collectives/collective/*/send_recv.csv \
   data/profiling/network/a100_40gb_pairwise_nvlink/

# 8. Verify
ls -la data/profiling/compute/a100_40gb/meta-llama/Llama-2-70b-hf/
ls -la data/profiling/network/a100_40gb_pairwise_nvlink/
```

### Estimated Profiling Times

| Profile Type | A100-40GB (TP=4) | Notes |
|-------------|------------------|-------|
| MLP | ~5 min | 261 configurations |
| Attention | ~2 min | 4492 configurations |
| all_reduce | ~5 min | ~994 configurations |
| send_recv | ~5 min | ~994 configurations |
| **Total** | **~17 min** | |

---

## Available Profiling Data

### Compute Profiles

| GPU | Model | Status |
|-----|-------|--------|
| A100-40GB | meta-llama/Llama-2-70b-hf | Available |
| A30 | meta-llama/Llama-2-7b-hf | Available |

### Network Profiles

| Network Device | Description | Status |
|----------------|-------------|--------|
| a100_40gb_pairwise_nvlink | CloudLab d8545 (4×A100-40GB SXM4 NVLink) | Available |
| a30_pcie | CloudLab c6525-25g (1×A30 PCIe per node) | Available |

---

## Troubleshooting

### Error: "undefined symbol" or ABI mismatch

This usually means sarathi-serve was built with a different PyTorch version.

```bash
cd ~/sarathi-serve
pip uninstall sarathi -y
rm -rf build/ *.egg-info/

# Ensure pyproject.toml has "torch >= 2.3"
pip install --no-cache-dir --no-build-isolation -e .
```

### Error: send_recv profiling produces no data

send_recv requires exactly 2 workers. Use:
```bash
--num_workers_per_node_combinations 2
```

### Error: "Model not found" (HuggingFace)

Ensure HF_TOKEN is set and has access to the model:
```bash
export HF_TOKEN="your_token"
python -c "from transformers import AutoTokenizer; t = AutoTokenizer.from_pretrained('meta-llama/Llama-2-70b-hf', token='$HF_TOKEN'); print('OK')"
```

### Warning: "cuda_time_total is deprecated"

This is a benign warning from PyTorch profiler. It does not affect profiling accuracy.

---

## Adding a New Model

1. Ensure the model's YAML config exists in `data/model_configs/{org}/{model}.yml`
2. Run MLP profiling with the new model
3. Run attention profiling with the new model
4. Copy results to `data/profiling/compute/{gpu_sku}/{org}/{model}/`
5. Network profiles can be reused if same hardware

---

## Adding a New Hardware Configuration

1. Profile MLP for representative models
2. Profile attention for representative models
3. Profile all_reduce with appropriate worker counts
4. Profile send_recv with 2 workers
5. Copy results to new directories under `data/profiling/`
6. Update simulator config to reference new device names

---

## Linking Profiling to Predictor Config

After profiling, you must create/update the predictor config to use the new profiles.

### Step 1: Get num_blocks from vLLM

Start vLLM and query the `/simple_schedule_trace` endpoint:

```bash
# Start vLLM first, then:
curl -s http://127.0.0.1:8000/simple_schedule_trace | jq '.free_gpu_blocks'
# Example output: 806
```

### Step 2: Create Predictor Config

Create a config file like `block/config/llama70b_a100_40gb_config.json`:

```json
{
    "replica_config": {
        "model_name": "meta-llama/Llama-2-70b-hf",
        "memory_margin_fraction": 0.1,
        "num_pipeline_stages": 1,
        "tensor_parallel_size": 4,
        "device": "a100_40gb",
        "network_device": "a100_40gb_pairwise_nvlink"
    },
    "replica_scheduler_config": {
        "max_tokens_in_batch": 4096,
        "batch_size_cap": 48,
        "chunk_size": 512,
        "num_blocks": 806
    },
    "target_metric": "min_new_request_latency",
    "enable_batch_time_estimation": true
}
```

### Key Fields

| Field | Description | How to Get |
|-------|-------------|------------|
| `device` | GPU SKU name | Must match `data/profiling/compute/{device}/` directory |
| `network_device` | Network topology name | Must match `data/profiling/network/{network_device}/` directory |
| `num_blocks` | KV cache blocks | From vLLM `/simple_schedule_trace` endpoint |
| `tensor_parallel_size` | TP degree | Must match profiling TP degree |
| `batch_size_cap` | Max batch size | Must match vLLM `--max-num-seqs` |
| `chunk_size` | Chunked prefill size | Must match vLLM `--max-num-batched-tokens` |

### Verify Config Links to Profiles

```bash
# Check profiling data exists for the device
ls data/profiling/compute/a100_40gb/meta-llama/Llama-2-70b-hf/
# Should show: mlp.csv, attention.csv

ls data/profiling/network/a100_40gb_pairwise_nvlink/
# Should show: all_reduce.csv, send_recv.csv
```

---

*Last updated: Feb 1, 2026*
