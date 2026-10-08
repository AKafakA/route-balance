# CARA Deployment & Experiment Workflow

**Purpose**: Step-by-step guide for deploying and running CARA experiments, from initial setup to full 18-node evaluation.

**Last Updated**: January 15, 2026

---

## Table of Contents

1. [Overview](#overview)
2. [Phase 1: Initial Setup (One-Time)](#phase-1-initial-setup-one-time)
3. [Phase 2: Development Iteration](#phase-2-development-iteration)
4. [Phase 3: Full-Scale Evaluation](#phase-3-full-scale-evaluation)
5. [Quick Reference](#quick-reference)
6. [Troubleshooting](#troubleshooting)

---

## Overview

### CARA Deployment Architecture

```
Local Machine (where you run commands)
    ↓ (generate_config.py)
block/config/host_configs.json, hosts
    ↓ (setup.sh - one time)
Remote Nodes: ~/Block, ~/vllm (cloned repos)
    ↓ (deploy_cara.py)
Remote Nodes: vLLM instances + Predictors running
    ↓ (cara_serve.py)
CARA Scheduler (reads local config on remote node)
    ↓ (benchmark_serving.py)
Experiment Results
```

### Key Concepts

**Config Files**:
- `model_config_template.json`: **YOU EDIT THIS** - Specifies what to deploy (`count` fields)
- `model_deployment.json`: **AUTO-GENERATED** - Contains assigned hostnames (`node_hosts` arrays)
- `host_configs.json`: **AUTO-GENERATED** - Host IPs, ports, SSH config

**Deployment Process**:
1. Edit template → Set deployment counts
2. Run `deploy_cara.py` → Generates config + distributes to nodes + deploys services
3. Start scheduler → Uses deployed instances
4. Run experiments → Collect results

---

## Phase 1: Initial Setup (One-Time)

Run these steps **once per CloudLab reservation** or after node reboot.

### Step 1: Reserve CloudLab Nodes

**For Development (2-4 nodes)**:
- 2× c240g5 (P100×1) - Test Qwen 2.5-3B

**For Full Evaluation (18 nodes)**:
- 2× d8545 (A100×4) - Qwen 2.5-72B
- 3× c4130 (V100×4) - Qwen 2.5-14B
- 5× d7525 (A30×1) - Qwen 2.5-7B
- 8× c240g5 (P100×1) - Qwen 2.5-3B

### Step 2: Generate Host Configuration

```bash
# Download manifest.xml from CloudLab experiment page
# Place it in block/ directory

# Generate host configs
python block/exp/generate_config.py \
  --username YOUR_SSH_USERNAME \
  --num_predictors 1

# Verify generated files
ls -la block/config/
# Should see: host_configs.json, hosts, ampere_hosts, volta_hosts, pascal_hosts
```

**What this creates**:
- `block/config/host_configs.json`: Host IPs, ports mapping
- `block/config/hosts`: All hostnames (for parallel-ssh)
- `block/config/ampere_hosts`: A100 nodes (d8545)
- `block/config/volta_hosts`: V100 nodes (c4130)
- `block/config/pascal_hosts`: P100 nodes (c240g5)

### Step 3: Initial Infrastructure Setup

```bash
# Install CUDA, dependencies, clone repos on ALL nodes
cd block/exp/cara
./setup.sh

# This takes 30-60 minutes and does:
# - Install CUDA 12.8 on all nodes
# - Clone Block repo to ~/Block on each node
# - Clone custom vLLM to ~/vllm on A100/V100/A30 nodes
# - Install vLLM 0.4 on P100 nodes (no FlashAttention)
# - Install Python dependencies

# Monitor progress
tail -f block/exp/cara/setup.log  # If you redirected output
```

**IMPORTANT**:
- Run `setup.sh` **only once** per reservation
- If nodes reboot, you may need to re-run
- This is a one-time infrastructure setup, separate from deployment

### Step 4: Verify Setup

```bash
# Check that Block repo is on all nodes
parallel-ssh -h block/config/hosts "ls -la ~/Block"

# Check that vLLM is installed on A100/V100/A30 nodes
parallel-ssh -h block/config/ampere_hosts "pip show vllm"
parallel-ssh -h block/config/volta_hosts "pip show vllm"

# Check CUDA
parallel-ssh -h block/config/hosts "nvidia-smi"
```

---

## Phase 2: Development Iteration

For rapid development and testing on a subset of nodes (e.g., 2 P100 nodes).

### Step 1: Edit Model Deployment Template

```bash
# Edit the template to deploy only what you need
vi block/config/cara/model_config_template.json
```

**Example: Deploy only Qwen 2.5-3B on 2 P100 nodes**:

```json
{
  "Qwen-2.5-72B": {
    "hf_model_name": "Qwen/Qwen2.5-72B",
    "precision": "fp16",
    "backend": "vllm",
    "node_type": {
      "d8545": {
        "count": 0,  // Don't deploy 72B in dev mode
        "gpu-memory-utilization": 0.95,
        "max-model-len": 8192,
        "enforce-eager": true,
        "use_hf_cache": true
      }
    }
  },
  "Qwen-2.5-14B": {
    "hf_model_name": "Qwen/Qwen2.5-14B",
    "precision": "fp16",
    "backend": "vllm",
    "node_type": {
      "c4130": {
        "count": 0,  // Don't deploy 14B in dev mode
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
        "count": 0  // Don't deploy 7B in dev mode
      }
    }
  },
  "Qwen-2.5-3B": {
    "hf_model_name": "Qwen/Qwen2.5-3B",
    "precision": "fp16",
    "backend": "vllm",
    "node_type": {
      "c240g5": {
        "count": 2,              // Deploy on 2 P100 nodes
        "serve_with_v0": true,   // Use vLLM 0.4 for P100
        "dtype": "float16",
        "attention_backend": "XFORMERS"
      }
    }
  }
}
```

**Key Points**:
- Set `count: 0` for models you don't want to deploy
- Set `count: N` to deploy N instances of that model on that node type
- Node types (`d8545`, `c4130`, `d7525`, `c240g5`) must match what's in your CloudLab reservation

### Step 2: Generate Config and Deploy

```bash
cd block/exp/cara

# Generate deployment config + distribute to nodes + deploy services
python deploy_cara.py \
  --config ../../config/cara/model_config_template.json \
  --output ../../config/cara/model_deployment.json \
  --hosts ../../config/hosts \
  --hf-token YOUR_HF_TOKEN \
  --deploy-services model_instance predictor

# This does 4 things:
# 1. Reads template, allocates hosts from available pool
# 2. Generates model_deployment.json with actual hostnames
# 3. SCPs config to ~/Block/block/config/cara/ on ALL nodes
# 4. SSH to each node, kills old processes, starts vLLM + predictor
```

**Wait for models to load**:
- First deployment with `--fresh-deploy`: 10-20 minutes (downloads from HF)
- Subsequent deployments: 2-5 minutes (uses cached models)

**Options**:
- `--deploy-services model_instance`: Deploy only vLLM/Ollama backends
- `--deploy-services predictor`: Deploy only predictors
- `--deploy-services model_instance predictor`: Deploy both (default)
- `--deploy-services none`: Just generate and distribute config
- `--no-distribute-config`: Skip copying config to remote nodes (local testing only)
- `--fresh-deploy`: Download models from HF (first time or after cache clear)
- `--models "Qwen-2.5-3B,Qwen-2.5-7B"`: Deploy only specific models

### Step 3: Verify Deployment

```bash
# Automated check
python e2e/check_deployment.py \
  --model_config_path ../../config/cara/model_deployment.json \
  --host_config ../../config/host_configs.json

# Manual check on specific node
ssh user@c240g5-1.cloudlab.us "ps aux | grep vllm"
ssh user@c240g5-1.cloudlab.us "curl http://localhost:8000/health"
ssh user@c240g5-1.cloudlab.us "curl http://localhost:8100/health"  # Predictor

# Check generated config
cat ../../config/cara/model_deployment.json
# Should have "node_hosts": ["user@node1", "user@node2"] (not "count")
```

### Step 4: Start CARA Scheduler

**Option A: Run on Local Machine** (for debugging, can see logs):

```bash
# From your local machine
python -m block.global_scheduler.cara.cara_serve \
  --host 0.0.0.0 \
  --port 8200 \
  --model_config_path block/config/cara/model_deployment.json \
  --host_config block/config/host_configs.json \
  --scheduling random \
  --enable-predictor-feedback

# Keep this terminal open to see logs
# Scheduler is at http://localhost:8200
```

**Option B: Run on Remote Node** (for longer experiments):

```bash
# SSH to head node
ssh user@d8545-1.cloudlab.us

# Start scheduler in background
cd ~/Block
nohup python -m block.global_scheduler.cara.cara_serve \
  --host 0.0.0.0 \
  --port 8200 \
  --model_config_path block/config/cara/model_deployment.json \
  --host_config block/config/host_configs.json \
  --scheduling random \
  --enable-predictor-feedback \
  > experiment_output/logs/cara_scheduler.log 2>&1 &

# Check it's running
curl http://localhost:8200/health

# View logs
tail -f experiment_output/logs/cara_scheduler.log
```

**Scheduler Options**:
- `--scheduling random`: Random instance selection
- `--scheduling round_robin`: Round-robin across instances
- `--scheduling greedy`: Use predictor scores (future)
- `--enable-predictor-feedback`: Collect training data
- `--broadcasting`: Broadcast mode (query all models for same prompt)
- `--selected-broadcasted-models "Qwen/Qwen2.5-3B" "Qwen/Qwen2.5-7B"`: Models to broadcast to

### Step 5: Run Experiments

**Simple Test**:

```bash
python -m block.benchmark.cara.benchmark_serving \
  --backend cara \
  --base-url http://localhost:8200 \
  --model dummy \
  --dataset-name sharegpt \
  --dataset-path data/trace_data/sharegpt/sharegpt_gpt4.jsonl \
  --num-prompts 10 \
  --request-rate 1.0 \
  --save-result

# Results in: experiment_output/results/
```

**Broadcasting Mode** (collect multi-model responses):

```bash
# Start scheduler in broadcasting mode
python -m block.global_scheduler.cara.cara_serve \
  --broadcasting \
  --selected-broadcasted-models "Qwen/Qwen2.5-3B" "Qwen/Qwen2.5-7B" \
  ...

# Run benchmark
./test_broadcasting.sh \
  "Qwen/Qwen2.5-3B Qwen/Qwen2.5-7B" \
  sharegpt \
  100 \
  2.0 \
  dev_test
```

**Full End-to-End Script**:

```bash
cd block/exp/cara
./run_cara_e2e.sh true  # true = redeploy backends
```

### Step 6: Iterate on Code Changes

**If you changed model deployment config**:

```bash
# 1. Edit template
vi block/config/cara/model_config_template.json

# 2. Redeploy (will kill old processes, start new ones)
cd block/exp/cara
python deploy_cara.py \
  --config ../../config/cara/model_config_template.json \
  --output ../../config/cara/model_deployment.json \
  --deploy-services model_instance predictor

# 3. Restart scheduler
# Kill old scheduler: Ctrl+C or pkill -f cara_serve
# Start new scheduler: (same command as Step 4)
```

**If you changed CARA scheduler code** (e.g., `cara_serve.py`):

```bash
# 1. Git commit locally
git add block/global_scheduler/cara/cara_serve.py
git commit -m "Fix scheduling bug"
git push origin cara-small

# 2. If scheduler runs on remote node, pull there
ssh scheduler-node "cd ~/Block && git pull origin cara-small"

# 3. Restart scheduler
# Kill old, start new
```

**If you changed predictor code**:

```bash
# 1. Git commit and push
git add block/predictor/cara/dummy_cara_predictor.py
git commit -m "Update predictor heuristic"
git push origin cara-small

# 2. Pull on all deployment nodes
parallel-ssh -h block/config/pascal_hosts "cd ~/Block && git pull origin cara-small"

# 3. Redeploy only predictors (fast)
cd block/exp/cara
python deploy_cara.py \
  --config ../../config/cara/model_config_template.json \
  --deploy-services predictor
```

**If you changed vLLM code**:

```bash
# 1. Commit and push to your vLLM fork
cd ~/vllm
git add ...
git commit -m "Add new feature"
git push origin cara_v_11

# 2. Pull and rebuild on all nodes
parallel-ssh -h block/config/ampere_hosts "cd ~/vllm && git pull && pip install --editable ."
parallel-ssh -h block/config/volta_hosts "cd ~/vllm && git pull && pip install --editable ."

# 3. Redeploy backends
cd ~/Block/block/exp/cara
python deploy_cara.py --deploy-services model_instance
```

---

## Phase 3: Full-Scale Evaluation

When ready for the final 18-node evaluation for the paper.

### Step 1: Reserve Full CloudLab Cluster

Reserve all nodes:
- 2× d8545 (A100×4) - 72B model
- 3× c4130 (V100×4) - 14B model
- 5× d7525 (A30×1) - 7B model
- 8× c240g5 (P100×1) - 3B model

**Total**: 18 nodes, 4 GPU types

### Step 2: Regenerate Configs for Full Cluster

```bash
# Download new manifest.xml (contains all 18 nodes)
# Place in block/

# Generate host configs
python block/exp/generate_config.py \
  --username YOUR_SSH_USERNAME \
  --num_predictors 1

# Verify all node types present
cat block/config/ampere_hosts  # Should have 2 d8545 nodes
cat block/config/volta_hosts   # Should have 3 c4130 nodes
cat block/config/pascal_hosts  # Should have 8 c240g5 nodes
wc -l block/config/hosts        # Should be 18 total
```

### Step 3: Run Initial Setup on All Nodes

```bash
# This takes 30-60 minutes for 18 nodes
cd block/exp/cara
./setup.sh

# Monitor progress (if you redirected output)
# parallel-ssh output is sequential, so be patient
```

### Step 4: Configure Full Deployment

Edit template to deploy all models:

```bash
vi block/config/cara/model_config_template.json
```

**Set all counts** (full deployment):

```json
{
  "Qwen-2.5-72B": {
    "hf_model_name": "Qwen/Qwen2.5-72B",
    "precision": "fp16",
    "backend": "vllm",
    "node_type": {
      "d8545": {
        "count": 2,  // 2 A100 nodes
        "gpu-memory-utilization": 0.95,
        "max-model-len": 8192,
        "enforce-eager": true,
        "tensor-parallel-size": 4  // 4 GPUs per node
      }
    }
  },
  "Qwen-2.5-14B": {
    "hf_model_name": "Qwen/Qwen2.5-14B",
    "precision": "fp16",
    "backend": "vllm",
    "node_type": {
      "c4130": {
        "count": 3,  // 3 V100 nodes
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
        "count": 5,  // 5 A30 nodes
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
        "count": 8,  // All 8 P100 nodes
        "serve_with_v0": true,
        "dtype": "float16",
        "attention_backend": "XFORMERS"
      }
    }
  }
}
```

**Note**: 32B is excluded due to repetition bug (documented in template).

### Step 5: Full Deployment

```bash
cd block/exp/cara

# First deployment: download models from HF (takes ~30 minutes)
python deploy_cara.py \
  --config ../../config/cara/model_config_template.json \
  --output ../../config/cara/model_deployment.json \
  --hosts ../../config/hosts \
  --hf-token YOUR_HF_TOKEN \
  --deploy-services model_instance predictor \
  --fresh-deploy

# Wait for all models to load
# Monitor: ssh to each node and check logs
parallel-ssh -h block/config/hosts "tail -20 experiment_output/logs/vllm_output.log"
```

### Step 6: Verify Full Deployment

```bash
# Automated check all nodes
python e2e/check_deployment.py \
  --model_config_path ../../config/cara/model_deployment.json \
  --host_config ../../config/host_configs.json

# Expected output: 18 instances healthy (2×72B, 3×14B, 5×7B, 8×3B)
```

### Step 7: Collect Training Data (Broadcasting Mode)

```bash
# 1. Start scheduler in broadcasting mode
python -m block.global_scheduler.cara.cara_serve \
  --host 0.0.0.0 \
  --port 8200 \
  --model_config_path block/config/cara/model_deployment.json \
  --host_config block/config/host_configs.json \
  --broadcasting \
  --selected-broadcasted-models "Qwen/Qwen2.5-3B" "Qwen/Qwen2.5-7B" "Qwen/Qwen2.5-14B" "Qwen/Qwen2.5-72B" \
  --enable-predictor-feedback

# 2. Run broadcasting benchmark (collect multi-model responses)
./test_broadcasting.sh \
  "Qwen/Qwen2.5-3B Qwen/Qwen2.5-7B Qwen/Qwen2.5-14B Qwen/Qwen2.5-72B" \
  sharegpt \
  500 \
  2.0 \
  batch1

# Results: experiment_output/results/
# Training data: training_data/cara/*.jsonl
```

### Step 8: Run Full Evaluation Experiments

```bash
# Baseline: Random scheduling
./run_cara_e2e.sh false  # false = don't redeploy

# Baseline: Round-robin scheduling
# Edit experiment.sh to change --scheduling to round_robin
# Re-run

# CARA: With predictors (after training models)
# Edit to use --scheduling greedy
# Re-run

# Collect results for all baselines and CARA variants
```

### Step 9: Process Results for Paper

```bash
# Prepare training data for quality/length estimators
python -m block.predictor.cara.offline_training.prepare_training_data \
  --input data/cara/cara-best-route-training.json \
  --tokenizer Qwen/Qwen2.5-72B \
  --scoring-method llm_judge \
  --output data/cara/best-route_llm_judge_training.json

# Analyze data quality
python analyze_training_data.py \
  --output data/cara/best-route_llm_judge_training.json

# Generate figures for paper (implement as needed)
python plot_results.py \
  --baseline-random experiment_output/results/random/ \
  --baseline-rr experiment_output/results/round_robin/ \
  --cara experiment_output/results/cara/ \
  --output cara_paper/figures/
```

---

## Quick Reference

### Common Commands

| Task | Command |
|------|---------|
| **Generate host configs** | `python block/exp/generate_config.py --username USER` |
| **Initial setup (once)** | `cd block/exp/cara && ./setup.sh` |
| **Deploy everything** | `python deploy_cara.py --config ... --deploy-services model_instance predictor` |
| **Deploy only backends** | `python deploy_cara.py --deploy-services model_instance` |
| **Deploy only predictors** | `python deploy_cara.py --deploy-services predictor` |
| **Just generate config** | `python deploy_cara.py --deploy-services none` |
| **Skip config distribution** | `python deploy_cara.py --no-distribute-config` |
| **First-time deployment** | `python deploy_cara.py ... --fresh-deploy` |
| **Deploy specific models** | `python deploy_cara.py --models "Qwen-2.5-3B,Qwen-2.5-7B"` |
| **Check deployment** | `python e2e/check_deployment.py --model_config_path ... --host_config ...` |
| **Start scheduler** | `python -m block.global_scheduler.cara.cara_serve --host 0.0.0.0 --port 8200 ...` |
| **Run benchmark** | `python -m block.benchmark.cara.benchmark_serving --backend cara ...` |
| **Broadcasting test** | `./test_broadcasting.sh "Qwen/Qwen2.5-3B Qwen/Qwen2.5-7B" sharegpt 100 2.0 test` |

### File Locations

| File | Location | Purpose |
|------|----------|---------|
| **Model template** | `block/config/cara/model_config_template.json` | Edit counts here |
| **Deployment config** | `block/config/cara/model_deployment.json` | Auto-generated |
| **Host configs** | `block/config/host_configs.json` | Auto-generated |
| **Hostnames** | `block/config/hosts`, `*_hosts` | Auto-generated |
| **Predictor config** | `block/config/cara/predictor_deployment_config.json` | Predictor settings |
| **vLLM logs** | `experiment_output/logs/vllm_output.log` | Per-node |
| **Predictor logs** | `experiment_output/logs/predictor_output.log` | Per-node |
| **Scheduler logs** | `experiment_output/logs/cara_scheduler.log` | If running remotely |
| **Benchmark results** | `experiment_output/results/` | JSON files |
| **Training data** | `training_data/cara/*.jsonl` | Predictor training data |

### Node Type Mapping

| Node Type | GPU | CloudLab Name | Model | Instances |
|-----------|-----|---------------|-------|-----------|
| `d8545` | A100×4 | d8545-10s103XX | Qwen 2.5-72B | 2 |
| `c4130` | V100×4 | c4130-XXXX | Qwen 2.5-14B | 3 |
| `d7525` | A30×1 | d7525-XXXX | Qwen 2.5-7B | 5 |
| `c240g5` | P100×1 | c240g5-XXXX | Qwen 2.5-3B | 8 |

---

## Troubleshooting

### Deployment Issues

#### Problem: `deploy_cara.py` fails with "Not enough nodes of type X"

**Diagnosis**:
```bash
# Check available nodes
cat block/config/pascal_hosts  # How many P100 nodes?
cat block/config/ampere_hosts  # How many A100 nodes?

# Check template counts
grep -A 5 "count" block/config/cara/model_config_template.json
```

**Solution**: Reduce `count` in template to match available nodes.

#### Problem: vLLM process not starting

**Diagnosis**:
```bash
# Check vLLM logs
ssh node "tail -100 experiment_output/logs/vllm_output.log"

# Check if process running
ssh node "ps aux | grep vllm"

# Check GPU availability
ssh node "nvidia-smi"
```

**Common Causes**:
- Out of GPU memory: Reduce `--gpu-memory-utilization` or `--max-model-len`
- Model not downloaded: Use `--fresh-deploy` flag
- Port already in use: Check for zombie processes, kill them

**Solution**:
```bash
# Kill all vLLM processes
parallel-ssh -h block/config/hosts "pkill -f vllm"

# Redeploy
python deploy_cara.py ... --fresh-deploy
```

#### Problem: Predictor not starting

**Diagnosis**:
```bash
# Check predictor logs
ssh node "tail -100 experiment_output/logs/predictor_output.log"

# Check if running
ssh node "ps aux | grep cara_predictor"

# Test health endpoint
ssh node "curl http://localhost:8100/health"
```

**Solution**:
```bash
# Redeploy only predictors
python deploy_cara.py --deploy-services predictor
```

#### Problem: Config not found on remote nodes

**Diagnosis**:
```bash
# Check if config exists on remote node
ssh node "cat ~/Block/block/config/cara/model_deployment.json"

# Check if it has node_hosts (not node_type)
ssh node "grep node_hosts ~/Block/block/config/cara/model_deployment.json"
```

**Solution**:
```bash
# Regenerate and redistribute
python deploy_cara.py \
  --config block/config/cara/model_config_template.json \
  --output block/config/cara/model_deployment.json \
  --deploy-services none  # Just distribute config
```

### Scheduler Issues

#### Problem: Scheduler can't connect to instances

**Diagnosis**:
```bash
# Check scheduler logs
tail -100 experiment_output/logs/cara_scheduler.log

# Test instance connectivity from scheduler node
curl http://node1:8000/health
curl http://node1:8100/health  # Predictor
```

**Solution**:
- Verify `host_configs.json` has correct IPs
- Check firewall rules (CloudLab should allow inter-node traffic)
- Ensure instances are running (`ps aux | grep vllm`)

#### Problem: Scheduler uses wrong config

**Diagnosis**:
```bash
# Check which config file scheduler is using
ps aux | grep cara_serve | grep model_config_path
```

**Solution**: Use `model_deployment.json` (NOT template):
```bash
python -m block.global_scheduler.cara.cara_serve \
  --model_config_path block/config/cara/model_deployment.json \
  ...
```

### Experiment Issues

#### Problem: Benchmark fails with connection errors

**Diagnosis**:
```bash
# Check scheduler is running
curl http://scheduler-node:8200/health

# Check scheduler logs for errors
tail -100 experiment_output/logs/cara_scheduler.log
```

**Solution**:
- Ensure scheduler is running and accessible
- Check `--base-url` in benchmark command matches scheduler host:port
- Increase benchmark timeout if instances are slow

#### Problem: No training data being collected

**Diagnosis**:
```bash
# Check predictor stats
curl http://node:8100/stats

# Check if feedback is enabled
grep enable_predictor_feedback block/config/cara/predictor_deployment_config.json

# Check scheduler started with feedback flag
ps aux | grep cara_serve | grep enable-predictor-feedback
```

**Solution**:
- Ensure `enable_data_collection: true` in predictor config
- Start scheduler with `--enable-predictor-feedback` flag
- Check `training_data/cara/` directory exists

#### Problem: Broadcasting returns incomplete results

**Diagnosis**:
```bash
# Check which models deployed
cat block/config/cara/model_deployment.json | grep hf_model_name

# Check scheduler broadcasting flag
ps aux | grep cara_serve | grep broadcasting

# Check scheduler logs
tail -100 experiment_output/logs/cara_scheduler.log | grep broadcast
```

**Solution**:
- Ensure all target models are deployed (not count=0)
- Match `--selected-broadcasted-models` to deployed model names
- Use exact HuggingFace names (e.g., "Qwen/Qwen2.5-3B")

### General Debugging

#### View Logs on All Nodes

```bash
# vLLM logs
parallel-ssh -h block/config/hosts "tail -20 experiment_output/logs/vllm_output.log"

# Predictor logs
parallel-ssh -h block/config/hosts "tail -20 experiment_output/logs/predictor_output.log"
```

#### Check All Processes

```bash
# Check vLLM
parallel-ssh -h block/config/hosts "ps aux | grep vllm | grep -v grep"

# Check predictors
parallel-ssh -h block/config/hosts "ps aux | grep cara_predictor | grep -v grep"
```

#### Health Check All Services

```bash
# Test vLLM health
parallel-ssh -h block/config/hosts "curl -s http://localhost:8000/health || echo FAILED"

# Test predictor health
parallel-ssh -h block/config/hosts "curl -s http://localhost:8100/health || echo FAILED"
```

#### Kill All CARA Processes

```bash
# Kill vLLM
parallel-ssh -h block/config/hosts "pkill -f vllm"

# Kill predictors
parallel-ssh -h block/config/hosts "pkill -f cara_predictor"

# Kill scheduler (if running remotely)
ssh scheduler-node "pkill -f cara_serve"

# Verify all killed
parallel-ssh -h block/config/hosts "ps aux | grep -E 'vllm|cara' | grep -v grep || echo Clean"
```

---

## Best Practices

### Before Each Experiment

1. ✅ **Verify all nodes healthy**: `parallel-ssh -h block/config/hosts "nvidia-smi"`
2. ✅ **Check deployment**: `python e2e/check_deployment.py`
3. ✅ **Test scheduler connectivity**: `curl http://scheduler:8200/health`
4. ✅ **Clear old results**: `rm -rf experiment_output/results/old_*`
5. ✅ **Note experiment parameters**: Document in lab notebook

### During Development

1. **Use small node counts**: Start with 2-4 nodes for faster iteration
2. **Test on small datasets**: Use `--num-prompts 10` for quick tests
3. **Monitor logs in real-time**: `tail -f` scheduler and instance logs
4. **Git commit frequently**: Commit working configs and code
5. **Use meaningful experiment names**: Name result directories clearly

### Before Final Evaluation

1. **Full deployment verification**: All 18 nodes healthy
2. **Baseline data collection**: Collect random, round-robin, heuristic results first
3. **Training data quality check**: Run `analyze_training_data.py`
4. **Predictor accuracy validation**: Check MAE/RMSE on validation set
5. **Dry run**: Full experiment with small dataset first

### After Experiments

1. **Save results**: Copy `experiment_output/results/` to archive
2. **Commit configs**: Git commit final model_deployment.json
3. **Document anomalies**: Note any errors or unexpected behavior
4. **Backup training data**: Copy `training_data/cara/` to safe location
5. **Release CloudLab nodes**: If done experimenting

---

## Notes

- **CloudLab reservations expire**: Typically 16 hours default, extend if needed
- **Models cache location**: `/mydata/hf_cache` (mounted tmpfs, cleared on reboot)
- **Code location**: `~/Block` on each node (separate git repos)
- **Config distribution**: Automatic by default, ensure SSH keys work
- **Parallel-ssh**: Used heavily, install with `pip install parallel-ssh`

For more details, see:
- **README.md**: Comprehensive CARA documentation
- **CARA_PROJECT_STATUS.md**: Implementation status and roadmap
- **Codex documents**: Design decisions and data pipeline

---

**Last Updated**: January 15, 2026
**Maintainer**: Check git history for recent changes
