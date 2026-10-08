# Block vs Llumnix Comparison Experiment Guide

*Last updated: Feb 2, 2026*

This document provides detailed instructions for running comparison experiments between Block and Llumnix on a 2-node A100-40GB cluster.

---

## Table of Contents

1. [Overview](#overview)
2. [Experiment Status](#experiment-status)
3. [Cluster Configuration](#cluster-configuration)
4. [Block Deployment](#block-deployment)
5. [Llumnix Deployment - Scheduling Only](#llumnix-deployment---scheduling-only)
6. [Llumnix Deployment - Migration + FlashInfer](#llumnix-deployment---migration--flashinfer)
7. [Running Benchmarks](#running-benchmarks)
8. [Experiment Results Summary](#experiment-results-summary)
9. [Known Issues and Solutions](#known-issues-and-solutions)

---

## Overview

**Goal:** Compare Block and Llumnix schedulers on identical hardware with the same model and workload.

**Setup:**
- 2x CloudLab d8545 nodes, each with 4x NVIDIA A100-40GB GPUs
- 8x Llama-2-7B instances total (4 per node, TP=1)
- ShareGPT workload at various QPS levels (16, 20, 24, 28, 32, 36)
- 10,000 requests per experiment

**Key Findings:**
- Block and Llumnix Scheduling Only achieve **comparable performance**
- Both systems scale to ~19k tok/s at QPS=36
- Migration without chunked prefill degrades severely at high load

---

## Experiment Status

### COMPLETED (Feb 1, 2026)

| Configuration | Status | Data Location |
|--------------|--------|---------------|
| Block 16pred | COMPLETE | `experiment_output/benchmark_output/block_16pred/` |
| Llumnix Scheduling Only | COMPLETE | `experiment_output/benchmark_output/llumnix_scheduling_only/` |
| Llumnix Migration + FlashInfer | COMPLETE | `experiment_output/benchmark_output/llumnix_migration_flashinfer/` |

### Configuration Summary

| Config | Attention | Chunked Prefill | Migration | Performance |
|--------|-----------|-----------------|-----------|-------------|
| Block 16pred | FlashInfer | Yes (512) | N/A | GOOD |
| Llumnix Scheduling Only | xFormers | Yes (512) | No | GOOD |
| Llumnix Migration | FlashInfer | No | Yes | DEGRADES at QPS 32+ |

---

## Cluster Configuration

### Node Information

| Node | External Hostname | Internal IP | Role |
|------|-------------------|-------------|------|
| node0 | d8545-10s10305.wisc.cloudlab.us | 10.10.1.1 | Head node |
| node1 | d8545-10s10505.wisc.cloudlab.us | 10.10.1.2 | Worker node |

### Virtualenv Location
```
/mydata/llumnix_venv  # Pre-configured on both nodes
```

---

## Block Deployment

### Using Automation Scripts

```bash
cd ~/Block

# Deploy Block system (vLLM + predictors + global scheduler)
sh block/exp/end_to_end_exp_scripts/a100_supplementary/deploy_block.sh start

# Verify deployment
sh block/exp/end_to_end_exp_scripts/a100_supplementary/deploy_block.sh status

# Stop Block system
sh block/exp/end_to_end_exp_scripts/a100_supplementary/deploy_block.sh stop
```

### Manual Deployment

**Step 1: Start vLLM Instances (8 total)**

On **node0**:
```bash
cd ~/Block
export PYTHONPATH=.
export HF_HOME=/mydata/huggingface
export HF_TOKEN="your_token"
export VLLM_USE_V1=0

for gpu in 0 1 2 3; do
    port=$((8000 + gpu))
    CUDA_VISIBLE_DEVICES=$gpu nohup python -m vllm.entrypoints.api_server \
        --model meta-llama/Llama-2-7b-hf \
        --port $port \
        --max-num-seqs 96 \
        --max-model-len 4096 \
        --enable-chunked-prefill \
        --max-num-batched-tokens 512 \
        > experiment_output/logs/vllm_gpu${gpu}.log 2>&1 &
done
```

On **node1**: Same commands.

**Step 2: Start Predictors (16 total)**

See `block/exp/end_to_end_exp_scripts/a100_supplementary/deploy_block.sh` for full predictor deployment.

**Step 3: Start Global Scheduler**

On **node0**:
```bash
cd ~/Block
export PYTHONPATH=.

python block/global_scheduler/api_server.py \
    --config_path block/config/a100_8x7b_host_configs_16pred.json \
    --metrics_type min_new_request_latency \
    --num_query_predictor 8 \
    --num_required_predictor 1 \
    --workers 1 \
    --num_predictor_ports 4 \
    --initial_available_instance 8
```

---

## Llumnix Deployment - Scheduling Only

This is the **best performing** Llumnix configuration, achieving performance comparable to Block.

### Step 1: Start Ray Cluster

On **node0** (head):
```bash
/mydata/llumnix_venv/bin/ray stop --force 2>/dev/null
rm -rf /tmp/ray/session_*
/mydata/llumnix_venv/bin/ray start --head --port=6380
```

On **node1** (worker):
```bash
/mydata/llumnix_venv/bin/ray stop --force 2>/dev/null
rm -rf /tmp/ray/session_*
/mydata/llumnix_venv/bin/ray start --address=10.10.1.1:6380
```

Verify cluster (should show 8 GPUs):
```bash
/mydata/llumnix_venv/bin/ray status
```

### Step 2: Start Llumnix Server (on node0 only)

```bash
cd ~/Block
mkdir -p experiment_output/logs
export HF_HOME=/mydata/huggingface
export HF_TOKEN="your_token"
export PYTHONUNBUFFERED=1

# NO VLLM_ATTENTION_BACKEND - defaults to xFormers (required for chunked prefill)
# NO --enable-routine-migration (scheduling only)
nohup /mydata/llumnix_venv/bin/python -u -m llumnix.entrypoints.vllm.api_server \
    --host 0.0.0.0 \
    --port 8200 \
    --initial-instances 8 \
    --ray-cluster-port 6380 \
    --model meta-llama/Llama-2-7b-hf \
    --worker-use-ray \
    --trust-remote-code \
    --max-model-len 4096 \
    --max-num-seqs 96 \
    --enable-chunked-prefill \
    --max-num-batched-tokens 512 \
    --dispatch-policy load \
    --migration-backend rayrpc \
    > experiment_output/logs/llumnix_scheduling_only.log 2>&1 &
```

Wait for all 8 instances (~3-5 minutes):
```bash
tail -f ~/Block/experiment_output/logs/llumnix_scheduling_only.log
# Look for: "Polling instance infos of 8 instances"
```

---

## Llumnix Deployment - Migration + FlashInfer

**WARNING**: This configuration degrades severely at high QPS (32+). Only use for migration experiments.

### Step 1: Start Ray Cluster

Same as above (see Scheduling Only section).

### Step 2: Start Llumnix Server with Migration (on node0 only)

```bash
cd ~/Block
mkdir -p experiment_output/logs
export HF_HOME=/mydata/huggingface
export HF_TOKEN="your_token"
export PYTHONUNBUFFERED=1
export VLLM_ATTENTION_BACKEND=FLASHINFER  # Use FlashInfer

# Migration enabled, NO chunked prefill (incompatible with migration)
nohup /mydata/llumnix_venv/bin/python -u -m llumnix.entrypoints.vllm.api_server \
    --host 0.0.0.0 \
    --port 8200 \
    --initial-instances 8 \
    --ray-cluster-port 6380 \
    --model meta-llama/Llama-2-7b-hf \
    --worker-use-ray \
    --trust-remote-code \
    --max-model-len 4096 \
    --max-num-seqs 96 \
    --dispatch-policy load \
    --migration-backend rayrpc \
    --enable-routine-migration \
    > experiment_output/logs/llumnix_migration_flashinfer.log 2>&1 &
```

---

## Running Benchmarks

### Using Automation Script

```bash
# Block benchmark
sh block/exp/end_to_end_exp_scripts/a100_supplementary/run_benchmark.sh block 28 10000

# Llumnix benchmark
sh block/exp/end_to_end_exp_scripts/a100_supplementary/run_benchmark.sh llumnix 28 10000

# Full comparison (all QPS levels)
sh block/exp/end_to_end_exp_scripts/a100_supplementary/full_comparison.sh both
```

### Manual Benchmark Commands

**Block Benchmark:**
```bash
cd ~/Block
export PYTHONPATH=.

python -m block.benchmark.benchmark_serving \
    --ip_ports 10.10.1.1:8200 \
    --tokenizer meta-llama/Llama-2-7b-hf \
    --num_sampled_requests 10000 \
    --dataset_type sharegpt \
    --dataset_path ~/Block/data/trace_data/sharegpt/generate/llama \
    --qps 28 \
    --backend block \
    --output_dir block_qps28_results \
    --trust_remote_code
```

**Llumnix Benchmark:**
```bash
cd ~/Block
export HF_HOME=/mydata/huggingface

/mydata/llumnix_venv/bin/python block/benchmark/benchmark_serving.py \
    --backend llumnix \
    --ip_ports 127.0.0.1:8200 \
    --tokenizer meta-llama/Llama-2-7b-hf \
    --trust_remote_code \
    --dataset_type sharegpt \
    --dataset_path ~/Block/data/trace_data/sharegpt/generate/llama \
    --qps 28 \
    --num_sampled_requests 10000 \
    --max_request_len 4096 \
    --output_dir llumnix_qps28_results \
    --timeout_in_seconds 1800
```

---

## Experiment Results Summary

### Head-to-Head Comparison at QPS=28

| Metric | Block 16pred | Llumnix Sched Only | Llumnix Migration |
|--------|-------------|-------------------|-------------------|
| **Throughput** | 15,009 tok/s | 14,872 tok/s | 15,047 tok/s |
| **Mean E2E Latency** | 6,599 ms | 6,423 ms | 7,635 ms |
| **Mean Token Latency** | 20.97 ms | 20.40 ms | 30.15 ms |
| **Sched Overhead** | 62.89 ms | 34.69 ms | 42.99 ms |

### Performance at High QPS (36)

| Metric | Block 16pred | Llumnix Sched Only | Llumnix Migration |
|--------|-------------|-------------------|-------------------|
| **Throughput** | 18,798 tok/s | 18,843 tok/s | 12,936 tok/s |
| **Mean E2E Latency** | 7,505 ms | 7,458 ms | 89,027 ms |
| **Mean Token Latency** | 24.03 ms | 23.72 ms | 919.84 ms |

### Key Findings

1. **Block and Llumnix Scheduling Only have comparable performance**
   - Both achieve ~19k tok/s at QPS=36
   - Latency within 5% of each other

2. **Migration without chunked prefill degrades at high load**
   - Works at QPS 16-28
   - Collapses at QPS 32+ (latency explodes 7-12x)

3. **Chunked prefill is critical for high-load performance**
   - More important than migration for handling high QPS

See [EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md) for complete results tables.

---

## Known Issues and Solutions

### Issue 1: Llumnix Migration + Chunked Prefill Incompatible

**Error:**
```
ValueError: Unsupported feature: Llumnix does not support "chunked prefill"
when enabling migration currently.
```

**Solution**: Must disable chunked prefill to enable migration. This is a Llumnix 0.1.1 limitation.

### Issue 2: FlashInfer + Chunked Prefill in vLLM 0.6.3.post1

**Error:**
```
AssertionError: Chunked prefill is not supported with flashinfer yet.
```

**Solution**: Use xFormers (unset VLLM_ATTENTION_BACKEND) or disable chunked prefill.

### Issue 3: Port 8200 Already in Use

```bash
# Kill existing processes
pkill -9 -f "llumnix"
pkill -9 -f "ray"
rm -rf /tmp/ray/session_*
```

### Issue 4: Ray Session Conflicts

```bash
# On both nodes
/mydata/llumnix_venv/bin/ray stop --force
rm -rf /tmp/ray/session_*
```

---

## Scripts Reference

| Script | Purpose |
|--------|---------|
| `block/exp/end_to_end_exp_scripts/a100_supplementary/deploy_block.sh` | Deploy Block system |
| `block/exp/end_to_end_exp_scripts/a100_supplementary/deploy_llumnix.sh` | Deploy Llumnix system |
| `block/exp/end_to_end_exp_scripts/a100_supplementary/run_benchmark.sh` | Run benchmarks |
| `block/exp/end_to_end_exp_scripts/a100_supplementary/full_comparison.sh` | Full comparison experiment |

---

*Document location: Block_paper/claude/LLUMNIX_COMPARISON_GUIDE.md*
