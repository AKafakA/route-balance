# CARA Experiment Quick Reference

**Purpose**: Quick command reference for running experiments on P100 and A100 nodes.

**Last Updated**: January 16, 2026

---

## Quick Start Commands

### A100 (Ampere) Nodes - d8545

```bash
# 1. Setup (one-time, ~45 mins)
cd block/exp/cara && ./setup.sh

# 2. Fix NVIDIA library paths (if not in setup.sh)
parallel-ssh -t 0 -h block/config/ampere_hosts "echo 'for dir in \$HOME/.local/lib/python3.10/site-packages/nvidia/*/lib; do [ -d \"\$dir\" ] && export LD_LIBRARY_PATH=\$dir:\$LD_LIBRARY_PATH; done' >> ~/.bashrc && source ~/.bashrc"

# 3. Deploy vLLM + predictors
python block/exp/cara/deploy_cara.py \
  --config block/config/cara/model_config_template.json \
  --output block/config/cara/model_deployment.json \
  --hosts block/config/ampere_hosts \
  --hf-token YOUR_HF_TOKEN \
  --deploy-services model_instance predictor \
  --fresh-deploy

# 4. Start CARA scheduler
python -m block.global_scheduler.cara.cara_serve \
  --host 0.0.0.0 --port 8200 \
  --model_config_path block/config/cara/model_deployment.json \
  --host_config block/config/host_configs.json \
  --scheduling random --enable-predictor-feedback

# 5. Run benchmark
python -m block.benchmark.cara.benchmark_serving \
  --backend cara --base-url http://localhost:8200 \
  --model Qwen/Qwen2.5-72B \
  --dataset-name custom --dataset-path data/cara/benchmark.jsonl \
  --num-prompts 100 --request-rate 2.0 --save-result
```

### P100 (Pascal) Nodes - c240g5

```bash
# 1. Setup (one-time, ~60 mins - builds from source)
cd block/exp/cara && ./setup.sh

# 2. Deploy vLLM + predictors
python block/exp/cara/deploy_cara.py \
  --config block/config/cara/model_config_template.json \
  --output block/config/cara/model_deployment.json \
  --hosts block/config/pascal_hosts \
  --hf-token YOUR_HF_TOKEN \
  --deploy-services model_instance predictor

# 3. Start CARA scheduler
python -m block.global_scheduler.cara.cara_serve \
  --host 0.0.0.0 --port 8200 \
  --model_config_path block/config/cara/model_deployment.json \
  --host_config block/config/host_configs.json \
  --scheduling random --enable-predictor-feedback

# 4. Run benchmark (MUST use custom dataset on P100)
python -m block.benchmark.cara.benchmark_serving \
  --backend cara --base-url http://localhost:8200 \
  --model Qwen/Qwen2.5-3B \
  --dataset-name custom --dataset-path data/cara/benchmark.jsonl \
  --num-prompts 100 --request-rate 2.0 --save-result
```

---

## Model Configuration Templates

### A100: 72B Model

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

### P100: 3B Model

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

---

## Broadcasting Mode (Multi-Model Data Collection)

```bash
# Start scheduler with broadcasting
python -m block.global_scheduler.cara.cara_serve \
  --host 0.0.0.0 --port 8200 \
  --model_config_path block/config/cara/model_deployment.json \
  --host_config block/config/host_configs.json \
  --broadcasting \
  --selected-broadcasted-models "Qwen/Qwen2.5-3B" "Qwen/Qwen2.5-7B" "Qwen/Qwen2.5-14B" \
  --enable-predictor-feedback

# Run benchmark
./test_broadcasting.sh "Qwen/Qwen2.5-3B Qwen/Qwen2.5-7B Qwen/Qwen2.5-14B" sharegpt 500 2.0 batch1
```

---

## Troubleshooting Quick Fixes

### A100: NVIDIA Library Not Found

```bash
# Find library
find ~/.local -name "libcudnn.so.9" 2>/dev/null

# Fix permanently
echo 'for dir in $HOME/.local/lib/python3.10/site-packages/nvidia/*/lib; do [ -d "$dir" ] && export LD_LIBRARY_PATH=$dir:$LD_LIBRARY_PATH; done' >> ~/.bashrc && source ~/.bashrc
```

### A100: NumPy 2.x Compatibility

```bash
pip install --upgrade scikit-learn pandas pyarrow
```

### P100: vLLM Architecture Error

```bash
cd ~/vllm
pip uninstall vllm -y
sudo CUDACXX=/usr/local/cuda-12.8/bin/nvcc TORCH_CUDA_ARCH_LIST=6.0 MAX_JOBS=7 pip install --editable .
```

### P100: Dataset Error

```
Use --dataset-name custom --dataset-path your_file.jsonl
```

### Check Deployment

```bash
# Health check all nodes
parallel-ssh -h block/config/hosts "curl -s http://localhost:8000/health || echo FAILED"

# View logs
ssh user@node "tail -100 ~/vllm/vllm_server.log"
```

### Kill Processes

```bash
# Kill vLLM on all nodes
parallel-ssh -h block/config/hosts "pkill -f vllm.entrypoints"

# Kill predictors
parallel-ssh -h block/config/hosts "pkill -f cara_predictor"

# Kill scheduler
pkill -f cara_serve
```

---

## Key Differences Summary

| Aspect | A100 | P100 |
|--------|------|------|
| **Node Type** | d8545 | c240g5 |
| **vLLM Branch** | cara_v_11 | cara_p100_v_6.0 |
| **Install** | Precompiled | Build from source |
| **Attention** | FlashAttention | XFORMERS |
| **Max Model** | 72B | 3B |
| **Dataset** | Any | Custom JSONL only |
| **Key Issue** | NVIDIA lib paths | CUDA arch build |

---

## Dataset Preprocessing

### Output Length Clamping

The benchmark uses soft clamping for output lengths:
- Default: `--custom-output-len 1024`
- Per-sample override via `output_len` field in JSONL
- Clamped by `--max-total-len` (default 2048)

### Word Limit Normalization Tool

**File**: `block/benchmark/cara/tools/normalize_word_limits.py`

Normalizes explicit word-count instructions in prompts to fit token budgets:

```bash
# Hard clamp: reduce "500 words" to budget if needed
python -m block.benchmark.cara.tools.normalize_word_limits \
  --infile data/prompts.jsonl \
  --outfile data/prompts_normalized.jsonl \
  --tokenizer Qwen/Qwen2.5-3B \
  --max-total-len 1024 \
  --output-cap 512

# Soft clamp: add guidance when no explicit word counts exist
python -m block.benchmark.cara.tools.normalize_word_limits \
  --infile data/prompts.jsonl \
  --outfile data/prompts_soft.jsonl \
  --tokenizer Qwen/Qwen2.5-3B \
  --max-total-len 1024 \
  --soft-clamp \
  --soft-round 50

# Dry run to see what would change
python -m block.benchmark.cara.tools.normalize_word_limits \
  --infile data/prompts.jsonl \
  --outfile /dev/null \
  --dry-run
```

**Modes**:
| Mode | Description | Example |
|------|-------------|---------|
| Hard clamp | Clamps explicit "N words" to budget | `"500 words"` → `"200 words"` |
| Soft clamp | Appends guidance if no explicit counts | Adds `"Try to answer within 150 words if possible."` |

**Patterns Handled**:
- `between X and Y words`
- `at least N words` / `minimum of N words`
- `no more than N words` / `maximum of N words`
- bare `N words`

**Key Options**:
- `--max-total-len`: Max total tokens (prompt + output)
- `--output-cap`: Max output tokens for budget calculation
- `--tokens-per-word`: Conversion ratio (default 1.3)
- `--soft-clamp`: Enable soft guidance appending
- `--soft-round`: Round soft budget to nearest N words (default 50)
- `--calibrate-tpw-samples N`: Calibrate tokens-per-word from first N samples

### Custom Dataset Format

```json
{"prompt": "Your prompt here", "output_len": 256}
{"prompt": "Another prompt", "output_len": 512}
```

### Using vllm_compat

P100 nodes use `block/benchmark/cara/vllm_compat.py` for compatibility:
- Provides fallback `SampleRequest`, `BenchmarkDataset`, `CustomDataset`
- Implements `is_valid_sequence()` for length validation
- Auto-discovers available vLLM features

---

## Training Data Preparation

### Preprocessing Benchmark Results

**File**: `block/predictor/cara/offline_training/prepare_benchmark_data.py`

Process benchmark results into training data for quality/length estimators:

```bash
# Single-model data with compression-based scoring (fast)
python -m block.predictor.cara.offline_training.prepare_benchmark_data \
  --input cara_result/cara-benchmark.json \
  --scoring-method compression \
  --dataset-name sharegpt-3b

# Multi-model data with similarity scoring
python -m block.predictor.cara.offline_training.prepare_benchmark_data \
  --input data/cara/broadcast_results.json \
  --scoring-method similarity \
  --reference-model "Qwen/Qwen2.5-72B" \
  --device cuda

# Multi-model data with LLM judge scoring
python -m block.predictor.cara.offline_training.prepare_benchmark_data \
  --input data/cara/broadcast_results.json \
  --scoring-method llm_judge \
  --judge-models Qwen/Qwen2.5-0.5B \
  --device cuda \
  --batch-size 8

# Compare multiple judges
python -m block.predictor.cara.offline_training.prepare_benchmark_data \
  --input cara_result/cara-benchmark.json \
  --judge-models Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-3B \
  --compare-judges \
  --divergence-threshold 0.3
```

**Key Features**:
- Auto-detects data format (single-model vs multi-model)
- Auto-detects all models from data
- Exports incomplete requests as JSONL for re-running
- Multiple quality scoring methods

**Scoring Methods**:
| Method | Requirements | Speed | Accuracy |
|--------|-------------|-------|----------|
| `compression` | None | Fast | Basic |
| `similarity` | Multi-model data, `--reference-model` | Medium | Good |
| `llm_judge` | GPU recommended | Slow | Best |

**Output Files**:
- `{output}.json`: Training data with quality scores
- `{output}_incomplete.jsonl`: Requests for re-running (benchmark format)
- `{output}_divergent.json`: Samples where judges disagree (if `--compare-judges`)

---

## File Locations

| File | Purpose |
|------|---------|
| `block/config/cara/model_config_template.json` | Edit deployment counts |
| `block/config/cara/model_deployment.json` | Auto-generated config |
| `block/config/host_configs.json` | Host IPs and ports |
| `block/config/ampere_hosts` | A100 node list |
| `block/config/pascal_hosts` | P100 node list |
| `block/exp/cara/deploy_cara.py` | Deployment script |
| `block/exp/cara/setup.sh` | Initial setup script |
| `experiment_output/results/` | Benchmark results |
| `training_data/cara/` | Predictor training data |

---

## Related Documentation

- **A100_DEPLOYMENT_GUIDE.md**: Detailed A100 deployment
- **P100_DEPLOYMENT_GUIDE.md**: Detailed P100 deployment
- **DEPLOYMENT_WORKFLOW.md**: Full deployment workflow
- **CARA_PROJECT_STATUS.md**: Project status and roadmap
