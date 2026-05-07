#!/bin/bash
# RSO Density: fraction of requests with per-request SLOs
set -euo pipefail
RESULT_DIR=${1:-"experiment_output/sensitivity/rso_density"}
NUM_REQ=${2:-2000}
QPS_LEVELS=${3:-"5 10 15"}
PORT=8200
DATASET="data/route_balance/best-route-v3-test-500.jsonl"
cd ~/RouteBalance; export PYTHONPATH=~/RouteBalance:~/vllm:$PYTHONPATH; mkdir -p $RESULT_DIR

run() { local N=$1; shift; python3 route_balance/benchmark/route_balance/benchmark_serving.py --backend route_balance --host 127.0.0.1 --port $PORT --dataset-name custom --dataset-path $DATASET --num-prompts $NUM_REQ --trust-remote-code --save-result --save-detailed --result-dir $RESULT_DIR --result-filename "${N}.json" "$@" 2>&1 | tail -1; }

echo "RSO Density Sweep — $(date)"
for RATE in "0.0" "0.1" "0.3" "0.5" "1.0"; do
    echo "  rate=$RATE"
    for QPS in $QPS_LEVELS; do
        run "rso_${RATE}_qps${QPS}" --request-rate $QPS --rso-rate $RATE
    done
done
echo "RSO Density COMPLETE"
