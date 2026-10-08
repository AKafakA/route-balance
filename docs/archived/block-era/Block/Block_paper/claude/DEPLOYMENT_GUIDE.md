# Block Deployment Guide

This document provides comprehensive instructions for deploying Block on both A30 (single-GPU) and A100 (multi-GPU with tensor parallelism) clusters.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Known Issues and Fixes](#2-known-issues-and-fixes)
3. [Prerequisites](#3-prerequisites)
4. [A30 Cluster Deployment](#4-a30-cluster-deployment)
5. [A100 Cluster Deployment](#5-a100-cluster-deployment)
6. [Troubleshooting](#6-troubleshooting)

---

## 1. Overview

Block requires three main components:
- **vLLM**: Inference engine (modified version with Block integration)
- **Predictors**: Simulation-based latency predictors (one or more per node)
- **Global Scheduler**: Request router using predicted metrics

### Deployment Order
1. Start vLLM backend(s)
2. Start Predictors (with cache-aware training)
3. Start Global Scheduler
4. Run benchmarks

---

## 2. Known Issues and Fixes

### 2.1 vLLM ABI Mismatch

**Issue**: When PyTorch and vLLM are built/installed separately, ABI incompatibility can occur:
```
ImportError: undefined symbol: _ZN5torch3jit17parseSchemaOrNameERKSsb
```

**Fix**: Rebuild vLLM with `--no-build-isolation`:
```bash
cd ~/vllm
pip uninstall vllm -y
rm -rf build/ *.egg-info/
pip install --no-cache-dir --no-build-isolation -e .
```

### 2.2 VLLM_USE_V1=0 Required

**Issue**: Block's `get_scheduler_trace` API requires the V0 engine. V1 engine throws:
```
TypeError: Can't instantiate abstract class AsyncLLM with abstract method get_scheduler_trace
```

**Fix**: Always set `VLLM_USE_V1=0` before starting vLLM:
```bash
export VLLM_USE_V1=0
```

### 2.3 Transformers Version (CRITICAL)

**Issue**: vLLM block branch requires `transformers==4.50.3`. Newer versions (5.0.0+) cause:
```
AttributeError: TokenizersBackend has no attribute all_special_tokens_extended
```

**Fix**: Install correct version:
```bash
pip install transformers==4.50.3
```

**Note**: This is now automatically handled in `setup.sh` and `setup_a100.sh`.

### 2.4 Installation Order (CRITICAL)

**Issue**: Installing PyTorch BEFORE vLLM causes ABI mismatch:
```
ImportError: undefined symbol: _ZN5torch3jit17parseSchemaOrNameERKSsb
```

**Fix**: Install vLLM with VLLM_USE_PRECOMPILED=1 FIRST, then PyTorch AFTER:
```bash
# Step 1: vLLM FIRST (with precompiled - ~3 min)
cd ~/vllm
VLLM_USE_PRECOMPILED=1 pip install --editable .

# Step 2: PyTorch AFTER (order matters!)
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126

# Step 3: Fix transformers
pip install transformers==4.50.3
```

**Why**: The precompiled vLLM wheels include .so files built against a specific PyTorch version. Installing PyTorch after ensures ABI compatibility.

### 2.5 VLLM_USE_PRECOMPILED=1

**Issue**: Building vLLM from source takes 30-40 minutes and may fail with OOM on flash attention.

**Fix**: Use precompiled wheels (works for Block's vLLM fork since it only modifies Python layer):
```bash
cd ~/vllm
VLLM_USE_PRECOMPILED=1 pip install --editable .
```

**Note**: This downloads pre-built .so files instead of compiling from source. Reduces install time from ~40 min to ~3 min.

### 2.6 Disk Space for Model Download

**Issue**: HuggingFace cache in home directory may run out of space for large models.

**Fix**: Mount NVMe storage and configure HF cache:
```bash
# Mount NVMe (CloudLab example)
sudo mkfs.ext4 -F /dev/nvme0n1
sudo mkdir -p /mydata
sudo mount /dev/nvme0n1 /mydata
sudo chmod 777 /mydata

# Configure HuggingFace
export HF_HOME=/mydata/huggingface
# And use --download-dir /mydata/huggingface in vLLM
```

### 2.7 Predictor Concurrent Training

**Issue**: Starting multiple predictors simultaneously causes CPU overload as each trains execution time models independently.

**Fix**: Use cache-aware deployment (two options):

**Option A: Copy cache (single training)**
1. Check if cache exists (`~/Block/cache/*.pkl`)
2. If no cache: Start 1 warmup predictor on first node
3. Wait for training (~1-2 min)
4. Copy cache to all other nodes: `rsync -az ~/Block/cache/ node2:~/Block/cache/`
5. Deploy all predictors on all nodes

**Option B: Individual warmup per node (RECOMMENDED)**
1. Start 1 warmup predictor on EACH node in parallel
2. Wait for all to complete training (~1-2 min)
3. Kill all warmup predictors
4. Deploy all predictors on all nodes

Option B is simpler because:
- No inter-node SSH/rsync required (avoids host key issues)
- Each node trains independently in parallel
- Still only 1 predictor per node during training, so no CPU overload

### 2.8 num_blocks Configuration

**Issue**: `num_blocks` in predictor config should match actual free GPU blocks from vLLM.

**Fix**: Get value from vLLM and update config:
```bash
# Get free_gpu_blocks from vLLM
curl -s http://127.0.0.1:8000/simple_schedule_trace | jq '.free_gpu_blocks'

# Update config with this value
```

### 2.9 sarathi-serve Installation (for Profiling)

**Issue**: sarathi-serve has strict PyTorch version requirements.

**Fix**:
```bash
cd ~/sarathi-serve
git checkout vidur
# Edit pyproject.toml: change "torch == 2.3" to "torch >= 2.3"
sed -i 's/"torch == 2.3"/"torch >= 2.3"/g' pyproject.toml
pip install --no-cache-dir --no-build-isolation -e .
```

### 2.10 Profiling Arguments

**Issue**: Wrong arguments for profiling scripts.

**Fixes**:
- Attention profiling uses `--max_model_len`, not `--max_tokens`
- send_recv profiling requires `--num_workers_per_node_combinations 2`

### 2.11 Internal IP for CloudLab

**Issue**: Using public IPs for inter-node traffic is slower and may have firewall restrictions.

**Fix**: Use `generate_config.py` to extract internal IPs from CloudLab manifests:
```bash
python block/exp/generate_config.py \
    --user_name YOUR_USERNAME \
    --manifest_path block/cl_manifest.xml \
    --cluster_type a30  # or a100
```

This generates `host_configs.json` with internal IPs (10.x.x.x) for better inter-node performance.

See: https://docs.cloudlab.us/control-net.html

### 2.12 A30 MIG Mode

**Issue**: A30 GPUs have MIG enabled by default, causing vLLM issues.

**Fix**: Disable MIG mode (already in setup.sh):
```bash
sudo nvidia-smi -mig 0
```

### 2.13 PYTHONPATH Requirement

**Issue**: Block modules not found.

**Fix**: Always set PYTHONPATH:
```bash
cd ~/Block && export PYTHONPATH=.
```

### 2.14 Global Scheduler Parameters

| Parameter | Description |
|-----------|-------------|
| `num_query_predictor` | Number of nodes to query (should equal node count for Block) |
| `num_required_predictor` | Minimum responses needed before decision |
| `num_predictor_ports` | Predictor ports per node |
| `initial_available_instance` | Total vLLM instances |
| `profiling_sampling_rate` | Sample rate for metrics collection (0.1 = 10%) |
| `length_error_pct` | Percentage error for length predictions (for sensitivity testing) |
| `latency_error_pct` | Percentage error for latency predictions (for sensitivity testing) |

### 2.15 Prediction Error Injection (for Sensitivity Testing)

**Issue**: Need to test Block's robustness to prediction errors.

**Solution**: Use `--length_error_pct` and `--latency_error_pct` parameters when starting the global scheduler:

```bash
python block/global_scheduler/api_server.py \
    ... \
    --length_error_pct 25 \
    --latency_error_pct 25
```

This injects Gaussian noise with the specified standard deviation percentage. Values are clamped to [0.1x, 5.0x] of original to prevent invalid predictions.

See `experiment_output/results/SoCC_revision/` for full experimental results showing Block is robust to up to 100% error.

---

## 3. Prerequisites

### 3.1 Software Requirements

| Component | Version |
|-----------|---------|
| Python | 3.10 |
| CUDA | 12.6+ |
| PyTorch | 2.6.0+ |
| transformers | 4.50.3 |
| vLLM | block branch |

### 3.2 Install Dependencies

```bash
# Clone repositories
git clone https://github.com/AKafakA/vllm.git ~/vllm
cd ~/vllm && git checkout block

git clone https://github.com/AKafakA/Block.git ~/Block

# Install vLLM with correct settings
export PATH=$HOME/.local/bin:$PATH
export CUDA_HOME=/usr/local/cuda
pip install --no-cache-dir --no-build-isolation -e .

# Install transformers (correct version)
pip install transformers==4.50.3

# Install Block dependencies
cd ~/Block
pip install -r requirements.txt
```

---

## 4. A30 Cluster Deployment

### 4.1 Configuration Files

**Host list** (`block/config/hosts`):
```
node1.example.com
node2.example.com
...
```

**Host config** (`block/config/host_configs.json`):
```json
{
    "node1.example.com": {
        "backend_port": 8000,
        "ip_address": "10.0.0.1",
        "predictor_ports": [8100]
    }
}
```

**Predictor config** (`block/config/llama_config.json`):
```json
{
    "replica_config": {
        "model_name": "meta-llama/Llama-2-7b-hf",
        "tensor_parallel_size": 1,
        "device": "a30",
        "network_device": "a30_single_gpu"
    },
    "replica_scheduler_config": {
        "batch_size_cap": 48,
        "chunk_size": 512,
        "num_blocks": 1052
    }
}
```

### 4.2 Deployment Steps

```bash
# 1. Start vLLM on all nodes
sh block/exp/run_exp_vllm.sh 48 "meta-llama/Llama-2-7b-hf" false 0 4096 true 1 512 true

# 2. Wait for vLLM to be ready (check health)
parallel-ssh -h block/config/hosts "curl -s http://127.0.0.1:8000/health"

# 3. Start predictors (one per node)
sh block/exp/run_exp_predictor_1.sh block/config/llama_config.json min_new_request_latency true 48 true 1 main 0 2000

# 4. Start global scheduler
sh block/exp/run_exp_global_scheduler.sh scheduler-host 1 1 min_new_request_latency block/config/host_configs.json 1 16 0.5 3600 2000 12 0
```

---

## 5. A100 Cluster Deployment

### 5.1 Configuration Files

**Host list** (`block/config/a100_hosts`):
```
d8545-10s10305.wisc.cloudlab.us
```

**Host config** (`block/config/a100_host_configs.json`):
```json
{
    "d8545-10s10305.wisc.cloudlab.us": {
        "backend_port": 8000,
        "ip_address": "128.105.146.43",
        "predictor_ports": [8100, 8300, 8400, 8500],
        "tensor_parallel_size": 4
    }
}
```

**Predictor config** (`block/config/llama70b_a100_40gb_config.json`):
```json
{
    "replica_config": {
        "model_name": "meta-llama/Llama-2-70b-hf",
        "tensor_parallel_size": 4,
        "device": "a100_40gb",
        "network_device": "a100_40gb_pairwise_nvlink"
    },
    "replica_scheduler_config": {
        "batch_size_cap": 48,
        "chunk_size": 512,
        "num_blocks": 806
    }
}
```

### 5.2 Deployment Steps

```bash
# 1. Mount NVMe storage (if needed)
ssh node "sudo mkfs.ext4 -F /dev/nvme0n1 && sudo mkdir -p /mydata && sudo mount /dev/nvme0n1 /mydata && sudo chmod 777 /mydata"

# 2. Start vLLM with tensor parallelism
sh block/exp/run_exp_vllm_a100.sh 48 "meta-llama/Llama-2-70b-hf" 4096 512 4 "YOUR_HF_TOKEN" true

# 3. Wait for model loading (3-5 minutes for 70B)
sleep 300
ssh node "curl -s http://127.0.0.1:8000/health"

# 4. Get num_blocks and update config
NUM_BLOCKS=$(ssh node "curl -s http://127.0.0.1:8000/simple_schedule_trace" | jq '.free_gpu_blocks')
echo "Update config with num_blocks: $NUM_BLOCKS"

# 5. Start predictors (cache-aware)
sh block/exp/run_exp_predictors_a100.sh block/config/llama70b_a100_40gb_config.json min_new_request_latency 48 1 2000 true

# 6. Start global scheduler
sh block/exp/run_exp_global_scheduler_a100.sh node block/config/a100_host_configs.json min_new_request_latency 4 0.5 3600 2000
```

### 5.3 End-to-End Script

```bash
# Run full A100 experiment with chunked prefill enabled
sh block/exp/end_to_end_exp_scripts/run_a100_llama70b.sh true

# Run without chunked prefill
sh block/exp/end_to_end_exp_scripts/run_a100_llama70b.sh false
```

---

## 6. Troubleshooting

### 6.1 vLLM Won't Start

1. Check CUDA visibility: `nvidia-smi`
2. Verify VLLM_USE_V1=0 is set
3. Check transformers version: `pip show transformers`
4. Check vLLM log: `tail -f ~/Block/experiment_output/logs/vllm.log`

### 6.2 Predictor Training Slow

1. CPU overload from concurrent training - use cache-aware deployment
2. Check cache exists: `ls ~/Block/cache/*.pkl | wc -l`
3. Monitor CPU: `htop`

### 6.3 OOM During Model Loading

1. Reduce batch_size_cap
2. Use tensor parallelism for larger models
3. Mount NVMe for model download

### 6.4 Global Scheduler Connection Issues

1. Verify all predictors are healthy: `curl http://node:PORT/health`
2. Check predictor ports match config
3. Verify network connectivity between nodes

---

## Appendix: Parameter Reference

### vLLM Parameters

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `--max-num-seqs` | Maximum batch size | 48 (7B), 48 (70B) |
| `--max-model-len` | Maximum sequence length | 4096 |
| `--max-num-batched-tokens` | Chunk size for chunked prefill | 512 |
| `--tensor-parallel-size` | Number of GPUs for TP | 1 (7B), 4 (70B) |
| `--enable-chunked-prefill` | Enable chunked prefill | true |

### Predictor Config Parameters

| Parameter | Description |
|-----------|-------------|
| `batch_size_cap` | Maximum batch size (match vLLM) |
| `chunk_size` | Chunk size for chunked prefill |
| `num_blocks` | Number of KV cache blocks (from vLLM) |
| `device` | GPU device identifier |
| `network_device` | Network topology identifier |

---

*Last updated: February 1, 2026*
