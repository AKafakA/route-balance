# P100 Node Deployment Guide

**Purpose**: Guide for deploying CARA on P100 (Pascal) nodes with older vLLM version.

**Last Updated**: January 16, 2026

---

## Table of Contents

1. [Overview](#overview)
2. [Key Constraints](#key-constraints)
3. [Setup Process](#setup-process)
4. [Deployment Commands](#deployment-commands)
5. [Model Configuration](#model-configuration)
6. [Troubleshooting](#troubleshooting)
7. [Quick Reference](#quick-reference)

---

## Overview

P100 nodes (c240g5 on CloudLab) use NVIDIA Pascal architecture with:
- 1× P100 GPU per node
- CUDA compute capability 6.0
- No FlashAttention support (uses XFORMERS instead)
- Custom vLLM branch (`cara_p100_v_6.0`) built from source

**Model Limit**: Only small models (3B parameters) due to single GPU memory constraints.

---

## Key Constraints

### Hardware Limitations

| Constraint | Value | Impact |
|------------|-------|--------|
| GPU Memory | 16GB | Limited to 3B models |
| Compute Capability | sm_60 | No FlashAttention |
| GPUs per Node | 1 | No tensor parallelism |

### Software Requirements

- **vLLM Branch**: `cara_p100_v_6.0` (custom P100-compatible build)
- **Attention Backend**: XFORMERS (not FlashAttention)
- **Build Flags**: Must set `TORCH_CUDA_ARCH_LIST=6.0`
- **CUDA**: 12.8 works but must compile for sm_60

---

## Setup Process

### Step 1: Run setup.sh

The setup.sh handles P100-specific configuration:

```bash
cd block/exp/cara
./setup.sh
```

For P100 nodes, setup.sh performs:

```bash
# 1. Install Build Dependencies
parallel-ssh -t 0 -h block/config/pascal_hosts "sudo apt install -y cmake"

# 2. Clone vLLM and checkout P100 branch
parallel-ssh -t 0 -h block/config/pascal_hosts "git clone https://github.com/AKafakA/vllm.git && cd vllm && git checkout cara_p100_v_6.0"

# 3. Install Xformers (FlashAttention alternative)
parallel-ssh -t 0 -h block/config/pascal_hosts "pip install xformers"

# 4. Build vLLM with Pascal architecture
parallel-ssh -t 0 -h block/config/pascal_hosts "cd vllm && sudo CUDACXX=/usr/local/cuda-12.8/bin/nvcc TORCH_CUDA_ARCH_LIST=6.0 MAX_JOBS=7 CMAKE_BUILD_PARALLEL_LEVEL=7 pip install --editable ."
```

**Build Time**: ~20-30 minutes per node (building from source)

### Step 2: Verify Installation

```bash
# Check vLLM is installed
ssh user@c240g5-node "pip show vllm"

# Check GPU is recognized
ssh user@c240g5-node "python3 -c 'import torch; print(torch.cuda.device_count())'"

# Should output: 1
```

---

## Deployment Commands

### Deploy vLLM on P100 Nodes

```bash
python block/exp/cara/deploy_cara.py \
  --config block/config/cara/model_config_template.json \
  --output block/config/cara/model_deployment.json \
  --hosts block/config/pascal_hosts \
  --hf-token YOUR_HF_TOKEN \
  --deploy-services model_instance predictor
```

### First-Time Deployment (Download Models)

```bash
python block/exp/cara/deploy_cara.py \
  --config block/config/cara/model_config_template.json \
  --output block/config/cara/model_deployment.json \
  --hosts block/config/pascal_hosts \
  --hf-token YOUR_HF_TOKEN \
  --deploy-services model_instance predictor \
  --fresh-deploy
```

**Note**: `--fresh-deploy` adds extra wait time for model download from HuggingFace.

---

## Model Configuration

### model_config_template.json for P100

```json
{
  "Qwen-2.5-3B": {
    "hf_model_name": "Qwen/Qwen2.5-3B",
    "precision": "fp16",
    "backend": "vllm",
    "node_type": {
      "c240g5": {
        "count": 2,
        "serve_with_v0": true,
        "dtype": "float16",
        "attention_backend": "XFORMERS"
      }
    }
  }
}
```

### Key Parameters for P100

| Parameter | Value | Description |
|-----------|-------|-------------|
| `serve_with_v0` | true | Use vLLM v0 API style |
| `dtype` | "float16" | Explicit dtype for old vLLM |
| `attention_backend` | "XFORMERS" | Required (no FlashAttention) |
| `count` | N | Number of P100 nodes to deploy |

**Note**: Don't set `gpu-memory-utilization` or `max-model-len` unless needed - use defaults.

---

## Running Benchmarks on P100

### Important: Dataset Compatibility

P100 nodes use older vLLM without the benchmarks module. Use custom datasets:

```bash
python -m block.benchmark.cara.benchmark_serving \
  --backend cara \
  --base-url http://localhost:8200 \
  --model Qwen/Qwen2.5-3B \
  --dataset-name custom \
  --dataset-path data/cara/benchmark_dataset.jsonl \
  --num-prompts 100 \
  --request-rate 2.0 \
  --save-result
```

**Critical**: Use `--dataset-name custom` with a JSONL file. Other dataset types (sharegpt, sonnet) require the vLLM benchmarks module which isn't available on P100.

### Create Custom Dataset

```bash
# Convert from other formats to JSONL
python -c "
import json
prompts = [
    {'prompt': 'Hello, how are you?', 'output_len': 100},
    {'prompt': 'Write a story about a cat.', 'output_len': 200},
]
with open('data/cara/benchmark_dataset.jsonl', 'w') as f:
    for p in prompts:
        f.write(json.dumps(p) + '\n')
"
```

---

## Troubleshooting

### Error: vLLM Import Fails

```
ModuleNotFoundError: No module named 'vllm'
```

**Solution**: Rebuild vLLM:
```bash
cd ~/vllm
sudo CUDACXX=/usr/local/cuda-12.8/bin/nvcc TORCH_CUDA_ARCH_LIST=6.0 pip install --editable . --force-reinstall
```

### Error: CUDA Architecture Mismatch

```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

**Solution**: vLLM was built for wrong architecture. Rebuild with sm_60:
```bash
cd ~/vllm
pip uninstall vllm -y
sudo CUDACXX=/usr/local/cuda-12.8/bin/nvcc TORCH_CUDA_ARCH_LIST=6.0 MAX_JOBS=7 CMAKE_BUILD_PARALLEL_LEVEL=7 pip install --editable .
```

### Error: Xformers Not Found

```
ModuleNotFoundError: No module named 'xformers'
```

**Solution**:
```bash
pip install xformers
```

### Error: Dataset Requires vLLM Benchmarks Module

```
ValueError: Dataset 'sharegpt' requires vLLM benchmarks module, which is not available on this node
```

**Solution**: Use custom dataset format:
```bash
--dataset-name custom --dataset-path your_dataset.jsonl
```

### Error: Out of Memory

P100 has 16GB VRAM. If OOM:
- Use smaller model (3B only)
- Reduce batch size
- Reduce max_model_len

### Error: CUDA Toolkit Not Found

```
FileNotFoundError: [Errno 2] No such file or directory: 'nvcc'
```

**Solution**: Set CUDA path:
```bash
export PATH=/usr/local/cuda-12.8/bin:$PATH
export CUDACXX=/usr/local/cuda-12.8/bin/nvcc
```

---

## Quick Reference

### Check P100 Node Status

```bash
# GPU status
ssh user@c240g5-node "nvidia-smi"

# vLLM process
ssh user@c240g5-node "ps aux | grep vllm"

# vLLM health
ssh user@c240g5-node "curl http://localhost:8000/health"
```

### Rebuild vLLM on P100

```bash
cd ~/vllm
git pull origin cara_p100_v_6.0
pip uninstall vllm -y
sudo CUDACXX=/usr/local/cuda-12.8/bin/nvcc TORCH_CUDA_ARCH_LIST=6.0 MAX_JOBS=7 CMAKE_BUILD_PARALLEL_LEVEL=7 pip install --editable .
```

### Kill vLLM on P100

```bash
pkill -f vllm.entrypoints
```

### View vLLM Logs

```bash
tail -100 ~/vllm/vllm_server.log
```

### P100 vs A100 at a Glance

| Task | P100 Command | A100 Command |
|------|--------------|--------------|
| Install vLLM | Build from source with sm_60 | Precompiled binary |
| Attention | XFORMERS | FlashAttention |
| Model Size | 3B only | Up to 72B |
| Dataset | Custom JSONL only | Any format |

---

## Comparison: P100 Deploy vs A100 Deploy

### deploy_cara.py Handling

The deployment script automatically handles differences based on model config:

**P100 Model Config**:
```json
{
  "serve_with_v0": true,
  "dtype": "float16",
  "attention_backend": "XFORMERS"
}
```

**A100 Model Config**:
```json
{
  "gpu-memory-utilization": 0.95,
  "max-model-len": 1024,
  "enforce-eager": true
}
```

### Environment Variables

**P100**: deploy_cara.py sets:
```bash
export VLLM_ATTENTION_BACKEND=XFORMERS
```

**A100**: deploy_cara.py sets:
```bash
export LD_LIBRARY_PATH=...(nvidia pip packages)...
```

---

## Related Documents

- **A100_DEPLOYMENT_GUIDE.md**: A100-specific deployment guide
- **DEPLOYMENT_WORKFLOW.md**: Full deployment workflow
- **CARA_PROJECT_STATUS.md**: Project status

---

**Document Maintenance**:
- Updated January 16, 2026
- Update when P100 vLLM branch changes