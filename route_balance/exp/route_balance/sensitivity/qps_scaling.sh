#!/bin/bash
# QPS Scaling: throughput/latency curve across load levels
set -euo pipefail
RESULT_DIR=${1:-"experiment_output/sensitivity/qps_scaling"}
NUM_REQ=${2:-2000}
PORT=8200
DATASET="data/route_balance/best-route-v3-test-500.jsonl"
cd ~/Block; export PYTHONPATH=~/Block:~/vllm:$PYTHONPATH; mkdir -p $RESULT_DIR

run() { local N=$1; shift; python3 route_balance/benchmark/route_balance/benchmark_serving.py --backend route_balance --host 127.0.0.1 --port $PORT --dataset-name custom --dataset-path $DATASET --num-prompts $NUM_REQ --trust-remote-code --save-result --save-detailed --result-dir $RESULT_DIR --result-filename "${N}.json" "$@" 2>&1 | tail -1; }

echo "QPS Scaling — $(date)"
for QPS in 1 2 5 10 15 20 25 30; do
    echo "  QPS=$QPS"
    run "route_balance_qps${QPS}" --request-rate $QPS
done
echo "QPS Scaling COMPLETE"
