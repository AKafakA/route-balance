# A100 Node Deployment Guide

**Purpose**: Comprehensive guide for deploying CARA on A100 (Ampere) nodes, including troubleshooting NVIDIA library issues.

**Last Updated**: January 16, 2026

---

## Table of Contents

1. [Overview](#overview)
2. [Key Differences: A100 vs P100](#key-differences-a100-vs-p100)
3. [NVIDIA Library Path Issues](#nvidia-library-path-issues)
4. [Setup Process](#setup-process)
5. [Deployment Commands](#deployment-commands)
6. [Troubleshooting](#troubleshooting)
7. [Quick Reference](#quick-reference)

---

## Overview

A100 nodes (d8545 on CloudLab) use NVIDIA Ampere architecture with:
- 4× A100 GPUs per node
- CUDA 12.8 support
- New vLLM version with FlashAttention support
- pip-installed NVIDIA libraries (cuDNN, cuSPARSELt, NCCL, etc.)

**Critical Difference from P100**: NVIDIA libraries are installed via pip to user's local site-packages (`~/.local/lib/python3.10/site-packages/nvidia/`) rather than system paths.

---

## Key Differences: A100 vs P100

| Aspect | A100 (Ampere) | P100 (Pascal) |
|--------|---------------|---------------|
| **Node Type** | d8545 | c240g5 |
| **GPUs** | 4× A100 | 1× P100 |
| **vLLM Branch** | `cara_v_11` | `cara_p100_v_6.0` |
| **Attention Backend** | FlashAttention (default) | XFORMERS |
| **CUDA Arch** | sm_80 (auto) | sm_60 (explicit) |
| **NVIDIA Libs** | pip site-packages | System paths |
| **Build** | Precompiled | Build from source |
| **Model Size** | 72B (tensor-parallel=4) | 3B only |

### vLLM Installation

**A100**:
```bash
git clone https://github.com/AKafakA/vllm.git && cd vllm && git checkout cara_v_11
sudo VLLM_USE_PRECOMPILED=1 pip install --editable .
```

**P100**:
```bash
git clone https://github.com/AKafakA/vllm.git && cd vllm && git checkout cara_p100_v_6.0
pip install xformers
sudo CUDACXX=/usr/local/cuda-12.8/bin/nvcc TORCH_CUDA_ARCH_LIST=6.0 MAX_JOBS=7 CMAKE_BUILD_PARALLEL_LEVEL=7 pip install --editable .
```

---

## NVIDIA Library Path Issues

### Problem

On A100 nodes, NVIDIA libraries (cuDNN, cuSPARSELt, NCCL, NVSHMEM, etc.) are installed via pip and located in:
```
~/.local/lib/python3.10/site-packages/nvidia/*/lib/
```

When running vLLM or PyTorch, you may encounter errors like:
```
ImportError: libcudnn.so.9: cannot open shared object file: No such file or directory
ImportError: libcusparseLt.so.0: cannot open shared object file: No such file or directory
ImportError: libnvshmem_host.so.3: cannot open shared object file: No such file or directory
```

### Root Cause

SSH sessions don't source `~/.bashrc` by default (non-login shells), so `LD_LIBRARY_PATH` isn't set when deploy_cara.py runs commands remotely.

### Solution

#### 1. In deploy_cara.py (Already Fixed)

The `get_vllm_commands()` function now dynamically discovers nvidia library paths:

```python
cmds.extend([
    "export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}",
    (
        "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:" \
        "$CUDA_HOME/lib64:" \
        "$CUDA_HOME/targets/x86_64-linux/lib:" \
        "/usr/local/cuda-12.8/lib64:" \
        "/usr/lib/x86_64-linux-gnu:" \
        "$(python3 -c 'import site; print(site.getusersitepackages())')/nvidia/cudnn/lib:" \
        "$(python3 -c 'import site; print(site.getusersitepackages())')/nvidia/cusparselt/lib:" \
        "$(python3 -c 'import site; print(site.getusersitepackages())')/nvidia/nccl/lib:" \
        "$(python3 -c 'import site; print(site.getusersitepackages())')/nvidia/nvjitlink/lib:" \
        "$(python3 -c 'import site; print(site.getusersitepackages())')/nvidia/nvshmem/lib:" \
        "$(python3 -c 'import site; print(site.getusersitepackages())')/nvidia/cublas/lib:" \
        "$(python3 -c 'import site; print(site.getusersitepackages())')/nvidia/cuda_runtime/lib:" \
        "/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib:" \
        "/usr/local/lib/python3.10/dist-packages/nvidia/cusparselt/lib:" \
        "/usr/local/lib/python3.10/dist-packages/nvidia/nccl/lib:" \
        "/usr/local/lib/python3.10/dist-packages/nvidia/nvshmem/lib"
    ),
    ...
])
```

#### 2. In setup.sh (Already Fixed)

The setup.sh adds a dynamic loop to `~/.bashrc`:

```bash
parallel-ssh -t 0 -h block/config/hosts "\
  echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc && \
  echo 'export PATH=\$PATH:\$CUDA_HOME/bin:/usr/local/cuda-12.8/bin' >> ~/.bashrc && \
  echo 'export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\$CUDA_HOME/lib64:/usr/local/cuda-12.8/lib64:\$CUDA_HOME/targets/x86_64-linux/lib:/usr/lib/x86_64-linux-gnu' >> ~/.bashrc && \
  echo '# Auto-discover all nvidia pip library paths' >> ~/.bashrc && \
  echo 'for dir in \$HOME/.local/lib/python3.10/site-packages/nvidia/*/lib /usr/local/lib/python3.10/dist-packages/nvidia/*/lib; do [ -d \"\$dir\" ] && export LD_LIBRARY_PATH=\$dir:\$LD_LIBRARY_PATH; done' >> ~/.bashrc && \
  source ~/.bashrc"
```

#### 3. Manual Fix for Existing Nodes

If you already ran setup.sh without these fixes, apply them manually:

```bash
# One-liner to add dynamic nvidia lib discovery to bashrc
echo 'for dir in $HOME/.local/lib/python3.10/site-packages/nvidia/*/lib /usr/local/lib/python3.10/dist-packages/nvidia/*/lib; do [ -d "$dir" ] && export LD_LIBRARY_PATH=$dir:$LD_LIBRARY_PATH; done' >> ~/.bashrc && source ~/.bashrc
```

Or apply to all A100 nodes:

```bash
parallel-ssh -t 0 -h block/config/ampere_hosts "echo 'for dir in \$HOME/.local/lib/python3.10/site-packages/nvidia/*/lib /usr/local/lib/python3.10/dist-packages/nvidia/*/lib; do [ -d \"\$dir\" ] && export LD_LIBRARY_PATH=\$dir:\$LD_LIBRARY_PATH; done' >> ~/.bashrc && source ~/.bashrc"
```

---

## Setup Process

### Step 1: Initial Setup (One-Time)

Run setup.sh which handles A100-specific configuration:

```bash
cd block/exp/cara
./setup.sh
```

This performs:
1. Install CUDA 12.8 toolkit
2. Install `nvidia-open` driver (Ampere-specific)
3. Clone vLLM (cara_v_11 branch)
4. Install vLLM with precompiled binaries
5. Fix NumPy 2.x compatibility (sklearn, pandas, pyarrow)
6. Configure LD_LIBRARY_PATH in bashrc

### Step 2: NumPy 2.x Compatibility Fix

A100 nodes may have NumPy 2.x which causes compatibility issues:

```
AttributeError: _ARRAY_API not found
```

**Fix** (already in setup.sh):
```bash
parallel-ssh -t 0 -h block/config/ampere_hosts "pip install --upgrade scikit-learn pandas pyarrow"
```

### Step 3: Mount Storage (Optional but Recommended)

For HuggingFace model cache:

```bash
# For d8545 nodes (NVMe storage)
sudo mkfs.ext4 -F /dev/nvme0n1 && sudo mkdir -p /mydata && sudo mount /dev/nvme0n1 /mydata && sudo chmod 777 /mydata
```

---

## Deployment Commands

### Deploy vLLM on A100 Nodes

```bash
python block/exp/cara/deploy_cara.py \
  --config block/config/cara/model_config_template.json \
  --output block/config/cara/model_deployment.json \
  --hosts block/config/ampere_hosts \
  --hf-token YOUR_HF_TOKEN \
  --deploy-services model_instance predictor \
  --fresh-deploy
```

### Model Config for A100 (72B Model)

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
        "max-model-len": 1024,
        "enforce-eager": true,
        "use_hf_cache": true
      }
    }
  }
}
```

**Key Parameters**:
- `gpu-memory-utilization`: 0.95 for 72B model (needs most of GPU memory)
- `enforce-eager`: true (disable CUDA graphs for stability)
- `use_hf_cache`: true (use /mydata/hf_cache if mounted)
- `max-model-len`: Context length limit (reduce if OOM)

### Run Benchmark on A100

```bash
python -m block.benchmark.cara.benchmark_serving \
  --backend cara \
  --base-url http://localhost:8200 \
  --model Qwen/Qwen2.5-72B \
  --dataset-name custom \
  --dataset-path data/cara/benchmark_dataset.jsonl \
  --num-prompts 100 \
  --request-rate 2.0 \
  --save-result
```

---

## Troubleshooting

### Error: libcudnn.so.9 not found

```bash
# Find where it is
find ~/.local -name "libcudnn.so.9" 2>/dev/null

# Add to path temporarily
export LD_LIBRARY_PATH=$HOME/.local/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

# Make permanent
echo 'for dir in $HOME/.local/lib/python3.10/site-packages/nvidia/*/lib; do [ -d "$dir" ] && export LD_LIBRARY_PATH=$dir:$LD_LIBRARY_PATH; done' >> ~/.bashrc
source ~/.bashrc
```

### Error: libcusparseLt.so.0 not found

Same solution as above - the dynamic loop covers all nvidia libraries.

### Error: libnvshmem_host.so.3 not found

Same solution - ensure the for loop is in bashrc.

### Error: AttributeError: _ARRAY_API not found

NumPy 2.x incompatibility with sklearn/pandas/pyarrow:

```bash
pip install --upgrade scikit-learn pandas pyarrow
```

### Error: ModuleNotFoundError: No module named 'model_hosting_container_standards'

vLLM not properly installed:

```bash
cd ~/vllm && pip install --editable . --force-reinstall
```

Or re-clone:

```bash
rm -rf ~/vllm
git clone https://github.com/AKafakA/vllm.git
cd vllm && git checkout cara_v_11
VLLM_USE_PRECOMPILED=1 pip install --editable .
```

### Error: Out of GPU Memory

Reduce model parameters:
- Lower `gpu-memory-utilization` (e.g., 0.85)
- Reduce `max-model-len` (e.g., 512)
- Use tensor parallelism across all 4 GPUs (default)

### vLLM Runs but Shows Warnings

NumPy warnings can be ignored if vLLM is serving requests:

```
Some module may need to rebuild instead e.g. with 'pybind11>=2.12'.
If you are a user of the module, the easiest solution will be to
downgrade to 'numpy<2' or try to upgrade the affected module.
```

This doesn't prevent vLLM from working.

---

## Quick Reference

### Find NVIDIA Library Locations

```bash
find ~/.local -name "lib*.so*" -path "*nvidia*" 2>/dev/null | head -20
```

### Test Library Access

```bash
python3 -c "import ctypes; ctypes.CDLL('libcudnn.so.9'); print('cuDNN found')"
python3 -c "import ctypes; ctypes.CDLL('libcusparseLt.so.0'); print('cuSPARSELt found')"
python3 -c "import torch; print(f'PyTorch OK, CUDA: {torch.cuda.is_available()}')"
```

### Check vLLM Health

```bash
curl http://localhost:8000/health
```

### View vLLM Logs

```bash
tail -100 ~/vllm/vllm_server.log
```

### Kill All vLLM Processes

```bash
pkill -f vllm.entrypoints
```

### Common LD_LIBRARY_PATH Export

```bash
export LD_LIBRARY_PATH=$HOME/.local/lib/python3.10/site-packages/nvidia/cudnn/lib:$HOME/.local/lib/python3.10/site-packages/nvidia/cusparselt/lib:$HOME/.local/lib/python3.10/site-packages/nvidia/nccl/lib:$HOME/.local/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$HOME/.local/lib/python3.10/site-packages/nvidia/nvshmem/lib:$LD_LIBRARY_PATH
```

---

## Related Documents

- **DEPLOYMENT_WORKFLOW.md**: Full deployment workflow for all node types
- **CARA_PROJECT_STATUS.md**: Project status and implementation details
- **README.md**: Comprehensive CARA documentation

---

**Document Maintenance**:
- Updated January 16, 2026 with NVIDIA library path fixes
- Update after discovering new library issues