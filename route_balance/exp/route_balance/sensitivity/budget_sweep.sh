#!/bin/bash
# Budget Sweep: budget_tokens + threshold θ
set -euo pipefail
RESULT_DIR=${1:-"experiment_output/sensitivity/budget_sweep"}
NUM_REQ=${2:-2000}
QPS_LEVELS=${3:-"5 10 15"}
PORT=8200
DATASET="data/route_balance/best-route-v3-test-500.jsonl"
cd ~/RouteBalance; export PYTHONPATH=~/RouteBalance:~/vllm:$PYTHONPATH; mkdir -p $RESULT_DIR

run() { local N=$1; shift; python3 route_balance/benchmark/route_balance/benchmark_serving.py --backend route_balance --host 127.0.0.1 --port $PORT --dataset-name custom --dataset-path $DATASET --num-prompts $NUM_REQ --trust-remote-code --save-result --save-detailed --result-dir $RESULT_DIR --result-filename "${N}.json" "$@" 2>&1 | tail -1; }
cfg() { curl -s -X POST http://localhost:$PORT/v1/config -H "Content-Type: application/json" -d "$1" > /dev/null; }

echo "Budget Sweep — $(date)"
echo "--- Budget tokens ---"
for BUDGET in 64 128 256 512 1024; do
    cfg "{\"slo_defaults\":{\"budget_tokens\":$BUDGET}}"
    for QPS in $QPS_LEVELS; do run "budget_${BUDGET}_qps${QPS}" --request-rate $QPS; done
done
cfg '{"slo_defaults":{"budget_tokens":256}}'

echo "--- Budget threshold ---"
for THRESH in "0.0" "0.3" "0.5" "0.7" "0.9"; do
    cfg "{\"slo_defaults\":{\"budget_confidence_threshold\":$THRESH}}"
    for QPS in $QPS_LEVELS; do run "thresh_${THRESH}_qps${QPS}" --request-rate $QPS; done
done
cfg '{"slo_defaults":{"budget_confidence_threshold":0.5}}'
echo "Budget Sweep COMPLETE"
