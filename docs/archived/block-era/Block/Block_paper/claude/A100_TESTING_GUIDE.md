# A100-40GB Llama-70B Testing Guide

This document provides step-by-step instructions for running the Vidur latency prediction accuracy experiment on CloudLab d8545 nodes (A100-40GB).

**Goal:** Validate that Block's Vidur-based simulator accurately predicts latency for Llama-70B models.

**Hardware:** 2× CloudLab d8545 nodes, each with 4× NVIDIA A100-40GB SXM4 GPUs (NVLink)

---

## Security Note

**The following files are gitignored and should NEVER be committed:**
- `block/*.xml` - CloudLab manifest files (contain encrypted tokens)
- `block/config/*_hosts` - SSH targets with usernames
- `block/config/*_host_configs.json` - Node IPs and ports

These files are generated locally and SCP'd to remote hosts by the setup script.

---

## Prerequisites

- CloudLab d8545 nodes provisioned
- CloudLab manifest downloaded locally (DO NOT commit to git)
- HuggingFace token with access to `meta-llama/Llama-2-70b-hf`
- SSH access to nodes

---

## Node Information

After provisioning, get node info from CloudLab web UI or manifest:

| Node | Hostname | Public IP | Internal IP |
|------|----------|-----------|-------------|
| node0 | (from manifest) | (from manifest) | 10.10.1.1 |
| node1 | (from manifest) | (from manifest) | 10.10.1.2 |

---

## Phase 0: Local Setup (on your machine)

### 0.1 Download CloudLab Manifest

1. Go to CloudLab experiment page
2. Click "Manifest" tab
3. Download XML and save to `block/a100_cl_manifest.xml` (gitignored)

### 0.2 Generate A100 Cluster Config

```bash
cd /auto/homes/wd312/Code/llm/Block

# Generate config from CloudLab manifest (manifest is gitignored)
python block/exp/generate_config.py \
    --user_name YOUR_CLOUDLAB_USERNAME \
    --manifest_path block/a100_cl_manifest.xml \
    --cluster_type a100 \
    --tensor_parallel_size 4 \
    --num_predictors 4

# Verify output (these files are also gitignored)
echo "=== A100 Host Config ==="
cat block/config/a100_host_configs.json

echo "=== A100 Hosts File ==="
cat block/config/a100_hosts
```

**Expected output:**
- `block/config/a100_host_configs.json` - JSON with both nodes (gitignored)
- `block/config/a100_hosts` - SSH targets for parallel-ssh (gitignored)

### 0.3 Test SSH Connectivity

```bash
# Get hostnames from generated config
cat block/config/a100_hosts

# Test each node (replace with actual hostnames)
ssh YOUR_USERNAME@NODE0_HOSTNAME "hostname && nvidia-smi -L"
ssh YOUR_USERNAME@NODE1_HOSTNAME "hostname && nvidia-smi -L"
```

**Expected:** Each node shows 4× A100-SXM4-40GB GPUs

### 0.4 Push Code to GitHub (without sensitive files)

```bash
cd /auto/homes/wd312/Code/llm/Block

# Verify sensitive files are gitignored
git status  # Should NOT show *.xml or *_hosts files

git add -A
git commit -m "A100 testing setup"
git push origin a100-test
```

---

## Phase 1: Cluster Setup (run locally, deploys to both nodes)

### 1.1 Run Setup Script

```bash
cd /auto/homes/wd312/Code/llm/Block
sh block/exp/setup_a100.sh
```

**This script does the following on BOTH nodes:**
1. Installs system packages, build tools, and CUDA 12.6
2. Mounts NVMe storage at `/mydata` (for large model downloads)
3. Clones and installs vLLM (block branch) with `VLLM_USE_PRECOMPILED=1` (~3 min)
4. Installs PyTorch 2.6 **AFTER** vLLM (critical for ABI compatibility)
5. Installs `transformers==4.50.3` (not 5.0.0+)
6. Clones and installs Block (a100-test branch)
7. **SCPs local config files to remote hosts** (since configs are gitignored)
8. Verifies installation

**Estimated time:** 15-20 minutes (using precompiled vLLM)

**CRITICAL Installation Order:**
```
1. vLLM with VLLM_USE_PRECOMPILED=1 (downloads compatible .so files)
2. PyTorch 2.6.0+cu126 (AFTER vLLM, not before)
3. transformers==4.50.3 (not 5.0.0+)
```

Installing in wrong order causes ABI mismatch errors.

### 1.2 Verify Installation (run locally)

```bash
# Check both nodes
parallel-ssh -h block/config/a100_hosts -i "nvidia-smi --query-gpu=name,memory.total --format=csv"
parallel-ssh -h block/config/a100_hosts -i "python -c 'import torch; print(torch.cuda.device_count())'"

# Verify configs were copied
parallel-ssh -h block/config/a100_hosts -i "ls -la ~/Block/block/config/"
```

**Expected:** Each node shows 4 GPUs with ~40GB memory, and config files exist

---

## Phase 2: Vidur Profiling (run on node0)

Profiling must be done on actual A100-40GB hardware to get accurate execution time data.

### 2.1 SSH to node0

```bash
ssh asdwb@d8545-10s10305.wisc.cloudlab.us
```

### 2.2 Install sarathi-serve-for-block (Required for Profiling)

The profiling scripts require sarathi-serve. Use the forked version with fixed dependencies:

```bash
# Clone sarathi-serve-for-block (vidur branch)
cd ~
rm -rf sarathi-serve
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

### 2.3 Setup Environment

```bash
cd ~/Block
export PYTHONPATH=.
export HF_TOKEN="YOUR_HUGGINGFACE_TOKEN"
```

### 2.4 Create Profiling Directories

```bash
mkdir -p data/profiling/compute/a100_40gb/meta-llama/Llama-2-70b-hf
mkdir -p data/profiling/network/a100_40gb_pairwise_nvlink
```

### 2.5 Profile MLP Kernels (~5 min)

```bash
python vidur/profiling/mlp/main.py \
    --models "meta-llama/Llama-2-70b-hf" \
    --num_gpus 4 \
    --num_tensor_parallel_workers 4 \
    --max_tokens 4096 \
    --output_dir "./profiling_tmp/mlp"

# Check output
ls -la ./profiling_tmp/mlp/
find ./profiling_tmp/mlp -name "*.csv"
```

**If successful, copy results:**
```bash
cp ./profiling_tmp/mlp/mlp/*/meta-llama/Llama-2-70b-hf/mlp.csv \
   data/profiling/compute/a100_40gb/meta-llama/Llama-2-70b-hf/mlp.csv
```

### 2.6 Profile Attention Kernels (~2 min)

Note: Use `--max_model_len` (not `--max_tokens`) for attention profiling.

```bash
python vidur/profiling/attention/main.py \
    --models "meta-llama/Llama-2-70b-hf" \
    --num_gpus 4 \
    --num_tensor_parallel_workers 4 \
    --max_model_len 4096 \
    --output_dir "./profiling_tmp/attention"

# Check output
ls -la ./profiling_tmp/attention/
find ./profiling_tmp/attention -name "*.csv"
```

**If successful, copy results:**
```bash
cp ./profiling_tmp/attention/attention/*/meta-llama/Llama-2-70b-hf/attention.csv \
   data/profiling/compute/a100_40gb/meta-llama/Llama-2-70b-hf/attention.csv
```

### 2.7 Profile NVLink Collectives (~10 min)

**Note:** Collectives profiling requires separate runs for `all_reduce` and `send_recv`:
- `all_reduce`: Profile with 4 workers (uses all 4 GPUs)
- `send_recv`: Profile with 2 workers (point-to-point between 2 GPUs)

```bash
# Profile all_reduce (4 workers)
python vidur/profiling/collectives/main.py \
    --num_workers_per_node_combinations 4 \
    --collective all_reduce \
    --output_dir "./profiling_tmp/collectives"

# Profile send_recv (2 workers - required for point-to-point)
python vidur/profiling/collectives/main.py \
    --num_workers_per_node_combinations 2 \
    --collective send_recv \
    --output_dir "./profiling_tmp/collectives"

# Check output
find ./profiling_tmp/collectives -name "*.csv"
```

**If successful, copy results:**
```bash
# Find the timestamped directories and copy
cp ./profiling_tmp/collectives/collective/*/all_reduce.csv \
   data/profiling/network/a100_40gb_pairwise_nvlink/all_reduce.csv
cp ./profiling_tmp/collectives/collective/*/send_recv.csv \
   data/profiling/network/a100_40gb_pairwise_nvlink/send_recv.csv
```

### 2.8 Verify All Profiling Data

```bash
echo "=== Compute Profiles ==="
ls -la data/profiling/compute/a100_40gb/meta-llama/Llama-2-70b-hf/

echo "=== Network Profiles ==="
ls -la data/profiling/network/a100_40gb_pairwise_nvlink/

echo "=== Sample MLP data ==="
head -5 data/profiling/compute/a100_40gb/meta-llama/Llama-2-70b-hf/mlp.csv
```

**Expected files:**
- `data/profiling/compute/a100_40gb/meta-llama/Llama-2-70b-hf/mlp.csv`
- `data/profiling/compute/a100_40gb/meta-llama/Llama-2-70b-hf/attention.csv`
- `data/profiling/network/a100_40gb_pairwise_nvlink/all_reduce.csv`
- `data/profiling/network/a100_40gb_pairwise_nvlink/send_recv.csv`

---

## Phase 3: Sanity Test (run on node0)

Before running the full experiment, test each component individually.

### 3.1 Test vLLM with Llama-70B

```bash
cd ~/Block
export PYTHONPATH=.
export HF_TOKEN="YOUR_HUGGINGFACE_TOKEN"

# Start vLLM server (takes ~5 min to load model)
python -m vllm.entrypoints.api_server \
    --model meta-llama/Llama-2-70b-hf \
    --tensor-parallel-size 4 \
    --max-num-seqs 16 \
    --max-model-len 4096 \
    --enable-chunked-prefill \
    --max-num-batched-tokens 512 \
    --port 8000 \
    --trust-remote-code &

# Wait for model to load
sleep 300

# Test endpoint
curl -X POST http://127.0.0.1:8000/generate \
    -H "Content-Type: application/json" \
    -d '{"prompt": "Hello, how are you?", "max_tokens": 20}'
```

**Expected:** JSON response with generated text

### 3.2 Test Predictor

```bash
# In another terminal (or use tmux)
cd ~/Block
export PYTHONPATH=.

python block/predictor/api_server.py \
    --port 8101 \
    --instance-port 8000 \
    --config_path block/config/llama70b_a100_40gb_config.json \
    --predictor_type simulate \
    --metric_type min_new_request_latency \
    --batch_size_cap 16 \
    --enable_chunked_prefill &

sleep 30

# Test predictor
curl -X POST http://127.0.0.1:8101/schedule_trace \
    -H "Content-Type: application/json" \
    -d '{"input_tokens": 100, "expected_output_tokens": 50}'
```

**Expected:** JSON with predicted latency metrics

### 3.3 Cleanup Test Processes

```bash
pkill -f "vllm.entrypoints"
pkill -f "predictor/api_server"
```

---

## Phase 4: Run Experiment (run on node0)

### 4.1 Prepare Experiment

```bash
cd ~/Block
export PYTHONPATH=.
export HF_TOKEN="YOUR_HUGGINGFACE_TOKEN"

# Verify config
cat block/config/llama70b_a100_40gb_config.json
```

### 4.2 Manual Deployment (Alternative to Script)

If running manually instead of using the experiment script:

```bash
# 1. Start vLLM (see Phase 3.1)

# 2. Cache-Aware Predictor Deployment
# Check if cache exists
CACHE_COUNT=$(ls ~/Block/cache/*.pkl 2>/dev/null | wc -l)
echo "Found $CACHE_COUNT cache files"

if [ "$CACHE_COUNT" -lt 10 ]; then
  # Start warmup predictor for training
  echo "Starting warmup predictor..."
  cd ~/Block && export PYTHONPATH=.
  nohup python block/predictor/api_server.py \
    --config_path block/config/llama70b_a100_40gb_config.json \
    --metric_type min_new_request_latency \
    --enable_time_estimation true \
    --batch_size_cap 48 \
    --workers 1 \
    --enable_chunked_prefill \
    --predictor_timeout 2000 \
    --predictor_index 1 \
    --port 8100 > experiment_output/logs/predictor_warmup.log 2>&1 &

  # Wait for training (~1-2 min)
  echo "Waiting for training..."
  while ! curl -s http://127.0.0.1:8100/health > /dev/null 2>&1; do
    sleep 10
    echo -n "."
  done
  echo "Training complete!"

  # Kill warmup and deploy all
  pkill -f "predictor/api_server"
  sleep 2
fi

# Deploy all 4 predictors
for port in 8100 8300 8400 8500; do
  idx=$((port/100 - 80))
  nohup python block/predictor/api_server.py \
    --config_path block/config/llama70b_a100_40gb_config.json \
    --metric_type min_new_request_latency \
    --enable_time_estimation true \
    --batch_size_cap 48 \
    --workers 1 \
    --enable_chunked_prefill \
    --predictor_timeout 2000 \
    --predictor_index $idx \
    --port $port > experiment_output/logs/predictor_$idx.log 2>&1 &
  sleep 2
done

# 3. Start Global Scheduler
python block/global_scheduler/api_server.py \
  --config_path block/config/a100_host_configs.json \
  --metrics_type min_new_request_latency \
  --num_query_predictor 2 \
  --num_required_predictor 1 \
  --workers 1 \
  --num_predictor_ports 4 \
  --profiling_sampling_rate 0.1 \
  --predictor_timeout 2000 \
  --backend_timeout 3600 \
  --initial_available_instance 2 \
  --max_slo_in_seconds 0 > experiment_output/logs/global_scheduler.log 2>&1 &
```

### 4.3 Run Experiment Script

```bash
# Edit TARGET_HOST if needed (should be 127.0.0.1 for single-node test)
sh block/exp/end_to_end_exp_scripts/a100_supplementary/a100_llama70b_exp.sh
```

### 4.4 Monitor Progress

```bash
# In another terminal
tail -f ~/Block/experiment_output/a100_40gb_llama70b/*/benchmark.log
```

---

## Troubleshooting

### Error: "CUDA out of memory"

Llama-70B with TP=4 requires ~35GB per GPU. If you see OOM:
```bash
# Check GPU memory usage
nvidia-smi

# Kill other GPU processes
sudo fuser -v /dev/nvidia* 2>/dev/null | xargs -r kill -9
```

### Error: "Profiling data not found"

```bash
# Verify profiling files exist
ls -la data/profiling/compute/a100_40gb/meta-llama/Llama-2-70b-hf/

# If missing, re-run profiling steps in Phase 2
```

### Error: "Model not found" (HuggingFace)

```bash
# Verify HF token
echo $HF_TOKEN

# Test model access
python -c "from transformers import AutoTokenizer; t = AutoTokenizer.from_pretrained('meta-llama/Llama-2-70b-hf', token='$HF_TOKEN'); print('OK')"
```

### Error: "Connection refused" (vLLM/Predictor)

```bash
# Check if processes are running
ps aux | grep vllm
ps aux | grep predictor

# Check ports
netstat -tlnp | grep -E "8000|8101"
```

### Error: vLLM Installation Failed (Precompiled)

If you see errors like "wheel not found" or "hash mismatch" during vLLM installation:

```bash
# The setup script should auto-fallback to source build, but if needed manually:
cd ~/vllm
pip uninstall vllm -y

# Build from source (takes 20-40 minutes)
pip install --editable .
```

**Common build issues:**

1. **Missing CUDA**: Ensure CUDA 12.6 is installed and in PATH
   ```bash
   nvcc --version  # Should show 12.6
   echo $PATH | grep cuda  # Should include /usr/local/cuda-12.6/bin
   ```

2. **Missing build tools**: Install if needed
   ```bash
   sudo apt install -y build-essential cmake ninja-build python3-dev
   ```

3. **Memory during compilation**: Building vLLM needs ~16GB RAM
   ```bash
   free -h  # Check available memory
   # If low, create swap:
   sudo fallocate -l 16G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

### Error: "undefined symbol" or "ImportError" (vLLM)

This usually means version mismatch between Python code and compiled binaries:

```bash
cd ~/vllm
pip uninstall vllm -y
rm -rf build/ *.egg-info/

# Rebuild from source
pip install --editable .
```

---

## Collecting Results

After experiment completes:

```bash
# On node0
cd ~/Block

# List result directories
ls -la experiment_output/a100_40gb_llama70b/

# Copy results to local machine (run locally)
scp -r asdwb@d8545-10s10305.wisc.cloudlab.us:~/Block/experiment_output/a100_40gb_llama70b/ \
    /auto/homes/wd312/Code/llm/Block/experiment_output/
```

---

## Error Reporting Template

When reporting errors, please include:

```
## Error Report

**Phase:** [0/1/2/3/4]
**Step:** [e.g., 2.4 Profile MLP Kernels]
**Command run:**
```
[paste command here]
```

**Error message:**
```
[paste full error output here]
```

**Environment:**
- Node: [node0/node1]
- GPU memory free: [nvidia-smi output]
- Disk space: [df -h output]
```

---

## Notes

- Profiling only needs to be done once per hardware type
- After profiling, commit the CSV files to the repo for reproducibility
- Both nodes can be used for profiling in parallel if needed (one runs MLP, other runs attention)

---

*Last updated: Feb 1, 2026*
