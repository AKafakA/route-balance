#!/bin/bash

# Simple ROUTE_BALANCE Test Script
# Assumes backend instances (vLLM/Ollama) are already running
# This script will:
#   1. Start ROUTE_BALANCE scheduler server
#   2. Send 100 random requests to test random scheduling
# Prerequisites: Customized vLLM with route_balance backend at ~/vllm

set -e  # Exit on error

# Configuration
ROUTE_BALANCE_HOST="127.0.0.1"
ROUTE_BALANCE_PORT="8200"
ROUTE_BALANCE_URL="http://${ROUTE_BALANCE_HOST}:${ROUTE_BALANCE_PORT}"
OUTPUT_DIR="experiment_output/route_balance_test_results"
MODEL_CONFIG="route_balance/config/route_balance/model_deployment.json"
HOST_CONFIG="route_balance/config/host_configs.json"
REPETITION_PENALTY=${1:-1.05}  # Repetition penalty for generation
REQUEST_RATE=${2:-inf}  # Request rate for benchmark
SAVE_DETAILED=${3:-"ttft itl e2el models hosts instance_ids"}  # Detailed metrics to save

echo "========================================"
echo "ROUTE_BALANCE Simple Test"
echo "========================================"
echo "ROUTE_BALANCE Server: ${ROUTE_BALANCE_URL}"
echo "Output Directory: ${OUTPUT_DIR}"
echo ""

# Step 1: Start ROUTE_BALANCE scheduler server
echo "Step 1: Starting ROUTE_BALANCE Scheduler Server"
echo "---------------------------------------"

# Kill existing ROUTE_BALANCE server if running
pkill -f 'route_balance_serve.py' || echo "No existing ROUTE_BALANCE server found"
sleep 2

# Create output directory
mkdir -p ${OUTPUT_DIR}
mkdir -p experiment_output/logs

# Start ROUTE_BALANCE server in background
nohup python -m route_balance.global_scheduler.route_balance.route_balance_serve \
  --host ${ROUTE_BALANCE_HOST} \
  --port ${ROUTE_BALANCE_PORT} \
  --model_config_path ${MODEL_CONFIG} \
  --host_config ${HOST_CONFIG} \
  --scheduling random \
  --repetition-penalty ${REPETITION_PENALTY} \
  > experiment_output/logs/route_balance_server.log 2>&1 &

ROUTE_BALANCE_PID=$!
echo "ROUTE_BALANCE server started (PID: ${ROUTE_BALANCE_PID})"
echo "Waiting 10 seconds for ROUTE_BALANCE server to initialize..."
sleep 10

# Verify ROUTE_BALANCE server is running
if ! ps -p ${ROUTE_BALANCE_PID} > /dev/null; then
    echo "❌ ROUTE_BALANCE server failed to start! Check logs:"
    tail -n 20 experiment_output/logs/route_balance_server.log
    exit 1
fi

echo "✅ ROUTE_BALANCE server is running"
echo ""

# Step 2: Run benchmark test
echo "Step 2: Running Benchmark Test"
echo "-------------------------------"
echo "Sending 100 random requests to ROUTE_BALANCE..."

# Add vllm to PYTHONPATH to use the customized version
export PYTHONPATH="$HOME/vllm:$PYTHONPATH"

python route_balance/benchmark/route_balance/benchmark_serving.py \
  --backend route_balance \
  --host ${ROUTE_BALANCE_HOST} \
  --port ${ROUTE_BALANCE_PORT} \
  --dataset-name random \
  --random-input-len 128 \
  --random-output-len 64 \
  --num-prompts 100 \
  --request-rate ${REQUEST_RATE} \
  --save-detailed ${SAVE_DETAILED} \
  --result-dir ${OUTPUT_DIR} \
  --result-filename route_balance_simple_test.json \
  --save-result

echo ""
echo "========================================"
echo "Test Completed!"
echo "========================================"
echo "Results saved to: ${OUTPUT_DIR}/route_balance_simple_test.json"
echo ""
echo "To view results:"
echo "  cat ${OUTPUT_DIR}/route_balance_simple_test.json | jq '.'"
echo ""
echo "Key metrics to check:"
echo "  - completed: Number of successful requests (should be 100)"
echo "  - request_throughput: Requests per second"
echo "  - mean_ttft_ms: Average time to first token"
echo "  - mean_tpot_ms: Average time per output token"
echo ""
echo "ROUTE_BALANCE server is still running (PID: ${ROUTE_BALANCE_PID})"
echo "To stop: kill ${ROUTE_BALANCE_PID}"
echo "To view logs: tail -f experiment_output/logs/route_balance_server.log"
