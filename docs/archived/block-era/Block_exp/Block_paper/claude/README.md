# Block

**Block: Balance Loader of LLM Serving with Context, Knowledge and Predictive Scheduling** ([paper link](https://arxiv.org/abs/2508.03611))

Block is a research prototype that explores *predictive, performance-aware scheduling* for distributed large-language-model (LLM) inference.
It builds on top of Microsoft's [Vidur](https://github.com/microsoft/vidur) simulator, which initially developed for offline evaluation and optimal configuration searching and adds

* a side-car *Predictor* service that forecasts per–instance leading metrics with Vidur at run time,
* a *Global Scheduler* that uses these predictions (or live metrics) to route requests, and
* tooling for training a light-weight length-estimator model so the scheduler can reason about prompts it has never seen before.

Everything needed to reproduce the paper's results—source code, datasets, experiment scripts—lives in this repository.

---

## 1. Architecture at a Glance

![Block Architecture](../../images/block_structure.png)

• **Predictor** (`block/predictor`)
 Co-locates with every inference node. Collects live stats, or spins up a Vidur simulation on-demand, and answers *"What if I got one more request?"*

• **Global Scheduler** (`block/global_scheduler`)
 Receives requests, queries Predictors, and applies the scheduling policy (default: Block; alternatives: LLumnix, round-robin, …).

• **Query Length Tagger** (`block/length_estimation`)
 A RoBERTa-based regressor that predicts the response-token count for unseen (model, prompt) pairs, feeding the scheduler with input-aware cost estimates.

Block is inference-engine agnostic. We provide an implementation for vLLM 0.7.2 (see the sealed repo at https://github.com/AKafakA/vllm/tree/block).

---

## 2. Repository Layout

```
block/
 ├── predictor/             # side-car prediction service
 ├── global_scheduler/      # request router
 ├── length_estimation/     # training & inference of token-length regressor
 ├── benchmark/             # Poisson load-generator (forked from vLLM)
 ├── config/
 │     ├── hosts            # cluster host list
 │     ├── host_configs.json# cluster description
 │     ├── llama_config.json# A30 7B model config
 │     ├── a100_hosts       # A100 cluster host list
 │     ├── a100_host_configs.json # A100 cluster config
 │     └── llama70b_a100_40gb_config.json # A100 70B model config
 ├── exp/
 │     ├── run_exp_vllm.sh         # vLLM deployment (A30)
 │     ├── run_exp_predictor_*.sh  # individual predictor scripts (1-16)
 │     ├── run_exp_global_scheduler.sh
 │     ├── run_exp_vllm_a100.sh    # vLLM deployment (A100, tensor parallelism)
 │     ├── run_exp_predictors_a100.sh # predictor deployment (A100)
 │     ├── run_exp_global_scheduler_a100.sh
 │     ├── end_to_end_exp_scripts/
 │     │     ├── a30_main/         # Main A30 experiments
 │     │     │     ├── main_experiment.sh
 │     │     │     └── warmup.sh
 │     │     ├── a100_supplementary/  # A100 Llumnix comparison & profiling
 │     │     │     ├── deploy_block.sh
 │     │     │     ├── deploy_llumnix.sh
 │     │     │     ├── run_benchmark.sh
 │     │     │     ├── full_comparison.sh
 │     │     │     └── a100_40gb_profiling.sh
 │     │     ├── ablation/         # Ablation studies
 │     │     │     ├── po2_ablation_exp.sh
 │     │     │     ├── burstiness_exp.sh
 │     │     │     ├── error_heatmap_exp.sh
 │     │     │     └── cpu_tracking_experiment.sh
 │     │     └── README.md
 │     └── ...
 └── data/                  # ShareGPT, BurstGPT, ArXiv-Summ datasets

vidur/  # Simulator with Block extensions
 ├── profiling/             # Hardware profiling scripts
 ├── scheduler/
 │     ├── sarathi_replica_scheduler.py
 │     └── simulate_predict_replica_scheduler.py
 └── ...

Block_paper/claude/         # Updated documentation
 ├── README.md              # This file
 ├── DEPLOYMENT_GUIDE.md    # Comprehensive deployment guide
 ├── PROFILING_GUIDE.md     # Hardware profiling instructions
 ├── A100_TESTING_GUIDE.md  # A100 experiment setup
 ├── LLUMNIX_COMPARISON_GUIDE.md  # Block vs Llumnix comparison guide
 ├── EXPERIMENT_RESULTS.md  # Llumnix comparison results
 ├── SOCC_REVISION_PROGRESS.md    # SoCC revision progress tracker
 └── SOCC_REVISION_PLAN.md  # Original revision plan
```

---

## 3. Quick Start

### 3.1 Prerequisites

| Component | Version | Notes |
|-----------|---------|-------|
| Python | 3.10 | |
| CUDA | 12.6+ | |
| PyTorch | 2.6.0+ | |
| transformers | 4.50.3 | **Important**: Must be 4.50.3, not 5.0+ |
| vLLM | block branch | https://github.com/AKafakA/vllm/tree/block |

### 3.2 Installation

```bash
# Clone vLLM block branch
git clone https://github.com/AKafakA/vllm.git ~/vllm
cd ~/vllm && git checkout block

# Install vLLM (IMPORTANT: use --no-build-isolation for ABI compatibility)
pip install --no-cache-dir --no-build-isolation -e .

# Install correct transformers version
pip install transformers==4.50.3

# Clone and setup Block
git clone https://github.com/AKafakA/Block.git ~/Block
cd ~/Block
pip install -r requirements.txt
```

### 3.3 Configuration

Generate cluster configuration:
```bash
# For CloudLab: download manifest.xml and move to block/config/
python block/exp/generate_config.py --username YOUR_SSH_USERNAME

# Or manually create:
# - block/config/hosts (one hostname per line)
# - block/config/host_configs.json (see examples)
```

### 3.4 Deployment

**A30 Single-GPU Cluster (Llama-7B):**
```bash
# Start vLLM (VLLM_USE_V1=0 is required for Block)
sh block/exp/run_exp_vllm.sh 48 "meta-llama/Llama-2-7b-hf" false 0 4096 true 1 512 true

# Start predictors (use experiment.sh which handles this automatically)
# Or manually start individual predictors:
for i in $(seq 1 16); do
  sh block/exp/run_exp_predictor_${i}.sh block/config/llama_config.json min_new_request_latency true 48 true 1 main 0 2000
done

# Start global scheduler
sh block/exp/run_exp_global_scheduler.sh scheduler-host 1 1 min_new_request_latency block/config/host_configs.json 1 16 0.5 3600 2000 12 0
```

**A100 Multi-GPU Cluster (Llama-70B with TP=4):**
```bash
# Full automated experiment
sh block/exp/end_to_end_exp_scripts/a100_supplementary/run_a100_llama70b.sh true  # with chunked prefill
sh block/exp/end_to_end_exp_scripts/a100_supplementary/run_a100_llama70b.sh false # without chunked prefill
```

See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for detailed instructions.

---

## 4. Known Issues and Critical Fixes

### Quick Reference (TL;DR)

```bash
# CRITICAL: Installation order matters!
# 1. vLLM FIRST (with precompiled)
cd ~/vllm && VLLM_USE_PRECOMPILED=1 pip install --editable .

# 2. PyTorch AFTER
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126

# 3. transformers 4.50.3 (not 5.0+)
pip install transformers==4.50.3

# 4. Always set before vLLM
export VLLM_USE_V1=0
export HF_TOKEN=your_token  # For gated models like Llama-2
```

### 4.1 VLLM_USE_V1=0 Required

Block's `get_scheduler_trace` API requires the V0 engine:
```bash
export VLLM_USE_V1=0  # Must be set before starting vLLM
```

### 4.2 transformers Version

vLLM block branch requires transformers==4.50.3:
```bash
pip install transformers==4.50.3  # Not 5.0+
```

Error if wrong version:
```
AttributeError: TokenizersBackend has no attribute all_special_tokens_extended
```

### 4.3 Installation Order (CRITICAL)

Installing PyTorch BEFORE vLLM causes ABI mismatch:
```
ImportError: undefined symbol: _ZN5torch3jit17parseSchemaOrNameERKSsb
```

**Fix**: Install vLLM with VLLM_USE_PRECOMPILED=1 FIRST, then PyTorch AFTER.

### 4.4 VLLM_USE_PRECOMPILED=1

Use precompiled wheels for fast installation (~3 min vs 40 min):
```bash
cd ~/vllm && VLLM_USE_PRECOMPILED=1 pip install --editable .
```

### 4.5 Predictor Concurrent Training

Starting multiple predictors simultaneously causes CPU overload. Use the cache-aware deployment scripts which:
1. Check if cache exists (`~/Block/cache/*.pkl`)
2. If no cache: start 1 predictor, wait for training, then start rest
3. If cache exists: start all predictors in parallel

### 4.6 HuggingFace Token

For gated models (Llama-2), set token:
```bash
export HF_TOKEN=your_token
```

See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) Section 2 for complete list.

---

## 5. Running Experiments

### 5.1 Main Experiments (A30 Cluster)

```bash
# Main QPS sweep experiment
sh block/exp/end_to_end_exp_scripts/a30_main/main_experiment.sh

# Warmup script (downloads model, clears cache)
sh block/exp/end_to_end_exp_scripts/a30_main/warmup.sh meta-llama/Llama-2-7b-hf
```

### 5.2 A100 Experiments (Llumnix Comparison)

```bash
# Deploy Block
sh block/exp/end_to_end_exp_scripts/a100_supplementary/deploy_block.sh start

# Run benchmark
sh block/exp/end_to_end_exp_scripts/a100_supplementary/run_benchmark.sh block 28 10000

# Full comparison (Block vs Llumnix)
sh block/exp/end_to_end_exp_scripts/a100_supplementary/full_comparison.sh both
```

### 5.3 Ablation Studies

```bash
# Power-of-two choices (N=2 vs N=12)
sh block/exp/end_to_end_exp_scripts/ablation/po2_ablation_exp.sh

# Burstiness sensitivity study
sh block/exp/end_to_end_exp_scripts/ablation/burstiness_exp.sh

# Prediction error sensitivity heatmap
sh block/exp/end_to_end_exp_scripts/ablation/error_heatmap_exp.sh

# CPU overhead tracking
sh block/exp/end_to_end_exp_scripts/ablation/cpu_tracking_experiment.sh
```

### 5.4 Plot Results

```bash
# Generate all SoCC revision plots
sh experiments_analysis/plot.sh socc

# Individual plots
sh experiments_analysis/plot.sh po2
sh experiments_analysis/plot.sh burstiness
sh experiments_analysis/plot.sh prediction
sh experiments_analysis/plot.sh cpu
sh experiments_analysis/plot.sh llumnix

# Output: experiment_output/results/SoCC_revision/
```

---

## 6. Hardware Profiling

To profile new GPU/model combinations:

```bash
# See PROFILING_GUIDE.md for detailed instructions

# MLP profiling
python vidur/profiling/mlp/main.py --models "meta-llama/Llama-2-70b-hf" --num_gpus 4

# Attention profiling (note: --max_model_len not --max_tokens)
python vidur/profiling/attention/main.py --models "meta-llama/Llama-2-70b-hf" --max_model_len 4096

# Collectives profiling
python vidur/profiling/collectives/main.py --collective all_reduce --num_workers_per_node_combinations 4
python vidur/profiling/collectives/main.py --collective send_recv --num_workers_per_node_combinations 2
```

---

## 7. Train / Evaluate the Length Estimator

```bash
# Download dataset
wget https://huggingface.co/datasets/shibing624/sharegpt_gpt4/blob/main/sharegpt_gpt4.jsonl

# Train and generate tagged data
python block/length_estimation/sample
python block/length_estimation/train_roberta
python block/length_estimation/eval_roberta --tag-data True
```

---

## 8. Extending Block

### New Scheduling Heuristic
1. Implement metrics export in vLLM
2. Define load scores in `simulate_predictor.py.predict`
3. Add scheduler name to `vidur/types/optimal_global_scheduler_target_metric.py`

### New Inference Engine
Expose metrics via API (see `scheduler_trace` in vLLM `api_server.py`)

### New Model/GPU
1. Run profiling (see [PROFILING_GUIDE.md](PROFILING_GUIDE.md))
2. Add model config to `vidur/config/model_config.py`
3. Create predictor config in `block/config/`

---

## 9. Citation

```bibtex
@misc{da2025blockbalancingloadllm,
      title={Block: Balancing Load in LLM Serving with Context, Knowledge and Predictive Scheduling},
      author={Wei Da and Evangelia Kalyvianaki},
      year={2025},
      eprint={2508.03611},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/abs/2508.03611},
}
```

---

## 10. License

This work is released under the MIT license. See `LICENSE` for details.

Happy scheduling!

---

*Last updated: February 2, 2026*
