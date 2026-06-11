#!/bin/bash
# Normalization Mode Ablation: two_phase / per_request / topsis / z_score
set -euo pipefail

RESULT_DIR=${1:-"experiment_output/ablation/normalization"}
NUM_REQ=${2:-2000}
QPS_LEVELS=${3:-"5 10 15"}
PORT=8200
DATASET="data/route_balance/best-route-v3-test-500.jsonl"

cd ~/Block
export PYTHONPATH=~/Block:~/vllm:$PYTHONPATH
mkdir -p $RESULT_DIR

run() { local N=$1; shift; python3 route_balance/benchmark/route_balance/benchmark_serving.py --backend route_balance --host 127.0.0.1 --port $PORT --dataset-name custom --dataset-path $DATASET --num-prompts $NUM_REQ --trust-remote-code --save-result --save-detailed --result-dir $RESULT_DIR --result-filename "${N}.json" "$@" 2>&1 | tail -1; }
cfg() { curl -s -X POST http://localhost:$PORT/v1/config -H "Content-Type: application/json" -d "$1" > /dev/null; }

echo "Normalization Ablation — $(date)"
for MODE in two_phase per_request topsis z_score; do
    cfg "{\"slo_defaults\":{\"normalization_mode\":\"$MODE\"}}"
    echo "  mode=$MODE"
    for QPS in $QPS_LEVELS; do
        run "norm_${MODE}_qps${QPS}" --request-rate $QPS
    done
done
cfg '{"slo_defaults":{"normalization_mode":"two_phase"}}'
echo "Normalization Ablation COMPLETE"
