#!/bin/bash
# Quality Threshold Sweep: quality_min 0.0 - 0.9
set -euo pipefail
RESULT_DIR=${1:-"experiment_output/ablation/quality_threshold"}
NUM_REQ=${2:-2000}
QPS_LEVELS=${3:-"5 10 15"}
PORT=8200
DATASET="data/route_balance/best-route-v3-test-500.jsonl"
cd ~/Block; export PYTHONPATH=~/Block:~/vllm:$PYTHONPATH; mkdir -p $RESULT_DIR

run() { local N=$1; shift; python3 route_balance/benchmark/route_balance/benchmark_serving.py --backend route_balance --host 127.0.0.1 --port $PORT --dataset-name custom --dataset-path $DATASET --num-prompts $NUM_REQ --trust-remote-code --save-result --save-detailed --result-dir $RESULT_DIR --result-filename "${N}.json" "$@" 2>&1 | tail -1; }
cfg() { curl -s -X POST http://localhost:$PORT/v1/config -H "Content-Type: application/json" -d "$1" > /dev/null; }

echo "Quality Threshold Sweep — $(date)"
for QMIN in "0.0" "0.3" "0.5" "0.7" "0.9"; do
    cfg "{\"slo_defaults\":{\"quality_min\":$QMIN}}"
    echo "  quality_min=$QMIN"
    for QPS in $QPS_LEVELS; do run "qualmin_${QMIN}_qps${QPS}" --request-rate $QPS; done
done
cfg '{"slo_defaults":{"quality_min":0.0}}'
echo "Quality Threshold COMPLETE"
