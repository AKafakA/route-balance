#!/bin/bash
# Batch Size Sweep: batch_size 1/4/8/16/32 × multiple QPS
set -euo pipefail

RESULT_DIR=${1:-"experiment_output/sensitivity/batch_sweep"}
NUM_REQ=${2:-2000}
QPS_LEVELS=${3:-"5 10 15"}
PORT=8200
DATASET="data/route_balance/best-route-v3-test-500.jsonl"

cd ~/Block
export PYTHONPATH=~/Block:~/vllm:$PYTHONPATH
mkdir -p $RESULT_DIR

run() { local N=$1; shift; python3 route_balance/benchmark/route_balance/benchmark_serving.py --backend route_balance --host 127.0.0.1 --port $PORT --dataset-name custom --dataset-path $DATASET --num-prompts $NUM_REQ --trust-remote-code --save-result --save-detailed --result-dir $RESULT_DIR --result-filename "${N}.json" "$@" 2>&1 | tail -1; }
cfg() { curl -s -X POST http://localhost:$PORT/v1/config -H "Content-Type: application/json" -d "$1" > /dev/null; }

echo "Batch Sweep — $(date)"
for BS_TO in "1:1" "4:25" "8:50" "16:100" "32:200"; do
    BS="${BS_TO%%:*}"; TO="${BS_TO#*:}"
    cfg "{\"slo_defaults\":{\"batch_config\":{\"max_batch_size\":$BS,\"batch_timeout_ms\":$TO,\"adaptive_sizing\":false}}}"
    echo "  batch=$BS timeout=${TO}ms"
    for QPS in $QPS_LEVELS; do
        run "batch_${BS}_qps${QPS}" --request-rate $QPS
    done
done
cfg '{"slo_defaults":{"batch_config":{"max_batch_size":8,"batch_timeout_ms":50,"adaptive_sizing":true}}}'
echo "Batch Sweep COMPLETE"
