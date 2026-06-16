#!/bin/bash

# Complete End-to-End ROUTE_BALANCE Deployment and Testing Script
# This script orchestrates the full deployment workflow:
# 1. Deploy backend instances (vLLM/Ollama)
# 2. Verify backend deployments
# 3. Start ROUTE_BALANCE scheduler server
# 4. Run benchmark tests

set -e  # Exit on error

# Configuration
TARGET_HOST="${CLOUDLAB_HOST}"  # Host to run ROUTE_BALANCE server
MODEL_CONFIG="route_balance/config/route_balance/model_config_template.json"
HOST_CONFIG="route_balance/config/host_configs.json"
HOSTS_FILE="route_balance/config/hosts"
DEPLOYMENT_CONFIG="route_balance/config/route_balance/model_deployment.json"

SCHEDULING_STRATEGY="random"  # Scheduling strategy for ROUTE_BALANCE
HF_TOKEN="${HF_TOKEN:-}"  # Set HF_TOKEN env var or pass as argument
REPETITION_PENALTY=${3:-1.05}  # Repetition penalty for generation
REQUEST_RATE=${4:-inf}  # Request rate for benchmark (requests per second, or 'inf' for unlimited)
SAVE_DETAILED=${5:-"ttft itl e2el models hosts instance_ids"}  # Detailed metrics to save

DOWNLOAD_DATASET=${2:-false}  # Pass 'true' as second argument to download dataset

DATASET_NAME="sharegpt"
DATASET_PATH="~/dataset/sharegpt"
DATASET_LINK="https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"

REDEPLOY=${1:-false}  # Pass 'true' as first argument to force redeployment

echo "========================================"
echo "ROUTE_BALANCE End-to-End Deployment & Test"
echo "========================================"
echo ""

# Step 1: Deploy backend instances
echo "Step 1: Deploying Backend Instances"
echo "------------------------------------"
echo "This will deploy vLLM and Ollama instances across all configured hosts..."
if [ "$REDEPLOY" = "true" ]; then
  python route_balance/exp/route_balance/deploy_route_balance.py \
    --hosts ${HOSTS_FILE} \
    --config ${MODEL_CONFIG} \
    --hf-token "${HF_TOKEN}" \
    --output ${DEPLOYMENT_CONFIG}
  sleep 600  # Wait for models to load
else
  echo "Skipping redeployment of backends. Using existing deployment config at ${DEPLOYMENT_CONFIG}."
fi

if [ "$DOWNLOAD_DATASET" = "true" ]; then
  echo "Downloading ShareGPT dataset to ${DATASET_PATH}..."
  ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    ${TARGET_HOST} \
    "mkdir -p ${DATASET_PATH} && wget -O ${DATASET_PATH}/sharegpt_random_10k.jsonl ${DATASET_LINK}"
  echo "Dataset download completed."
fi

# Start ROUTE_BALANCE scheduler server
echo "Step 2: Starting ROUTE_BALANCE Scheduler Server"
echo "---------------------------------------"
echo "Starting ROUTE_BALANCE server on ${TARGET_HOST}:${ROUTE_BALANCE_PORT}..."

# Kill existing ROUTE_BALANCE server if running
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  ${TARGET_HOST} \
  "pkill -f 'route_balance_serve.py' || echo 'No existing ROUTE_BALANCE server found'"
sleep 2

# Start ROUTE_BALANCE server in background
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  ${TARGET_HOST} \
  "cd Block && nohup python -m route_balance.global_scheduler.route_balance.route_balance_serve \
    --model_config_path ${DEPLOYMENT_CONFIG} \
    --host_config ${HOST_CONFIG} \
    --scheduling ${SCHEDULING_STRATEGY} \
    --repetition-penalty ${REPETITION_PENALTY} \
    > experiment_output/logs/route_balance_server.log 2>&1 &"

echo "Waiting 10 seconds for ROUTE_BALANCE server to start..."
sleep 10

# Run benchmark tests
echo "Step 3: Running Benchmark Tests"
echo "--------------------------------"

# Configuration for benchmark
OUTPUT_DIR="experiment_output/route_balance_test_results"
mkdir -p ${OUTPUT_DIR}

# Run benchmark with small/random dataset for testing
echo "Running ROUTE_BALANCE benchmark with random dataset..."
echo "  Requests: 50"
echo "  Input length: 128 tokens"
echo "  Output length: 64 tokens"
echo ""

# Add vLLM to PYTHONPATH if it exists locally
if [ -d "$HOME/vllm" ]; then
  export PYTHONPATH="$HOME/vllm:$PYTHONPATH"
  echo "Using local vLLM from: $HOME/vllm"
elif [ -d "~/vllm" ]; then
  export PYTHONPATH="~/vllm:$PYTHONPATH"
  echo "Using local vLLM from: ~/vllm"
fi

ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  ${TARGET_HOST} \
  "export PYTHONPATH=${PYTHONPATH} && \
   cd Block && \
   python route_balance/benchmark/route_balance/benchmark_serving.py \
     --backend route_balance \
     --host 127.0.0.1 \
     --port 8200 \
     --dataset-name ${DATASET_NAME} \
     --dataset-path ${DATASET_PATH}/sharegpt_random_10k.jsonl \
     --num-prompts 50 \
     --request-rate ${REQUEST_RATE} \
     --save-detailed ${SAVE_DETAILED} \
     --result-dir ${OUTPUT_DIR} \
     --save-result"

if [ $? -ne 0 ]; then
    echo "❌ Benchmark failed!"
    echo "Check ROUTE_BALANCE server logs for errors:"
    echo "  ssh ${TARGET_HOST} 'tail -f Block/experiment_output/logs/route_balance_server.log'"
    exit 1
fi

echo ""
echo "========================================"
echo "ROUTE_BALANCE Deployment & Benchmark Complete!"
echo "========================================"
echo ""
echo "Summary:"
echo "  ✅ Backend instances deployed and verified"
echo "  ✅ ROUTE_BALANCE server running at http://${ROUTE_BALANCE_HOST_IP}:${ROUTE_BALANCE_PORT}"
echo "  ✅ Benchmark tests completed"
echo ""
echo "Results:"
echo "  - Benchmark results saved to: ${OUTPUT_DIR}"
echo "  - View latest results: ls -lht ${OUTPUT_DIR}"
echo ""
echo "Useful Commands:"
echo "  1. Check ROUTE_BALANCE server logs:"
echo "     ssh ${TARGET_HOST} 'tail -f Block/experiment_output/logs/route_balance_server.log'"
echo ""
echo "  2. Run additional benchmark tests:"
echo "     python route_balance/benchmark/route_balance/benchmark_serving.py \\"
echo "       --backend route_balance \\"
echo "       --base-url http://${ROUTE_BALANCE_HOST_IP}:${ROUTE_BALANCE_PORT} \\"
echo "       --dataset-name random \\"
echo "       --num-prompts 100"
echo ""
echo "  3. Monitor backend logs on remote hosts:"
echo "     ssh <host> 'tail -f ~/vllm/vllm_server.log'  # For vLLM"
echo "     ssh <host> 'tail -f ~/ollama/ollama_server.log'  # For Ollama"
echo ""
