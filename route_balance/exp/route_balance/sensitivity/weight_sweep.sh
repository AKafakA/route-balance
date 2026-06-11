#!/bin/bash
# Weight Sweep: 9 scoring weight configurations × multiple QPS
# Single scheduler deployment (route_balance), runtime config updates only.
#
# Usage: bash route_balance/exp/route_balance/sensitivity/weight_sweep.sh [result_dir] [num_requests]
set -euo pipefail

RESULT_DIR=${1:-"experiment_output/sensitivity/weight_sweep"}
NUM_REQ=${2:-2000}
QPS_LEVELS=${3:-"5 10 15"}
DATASET="data/route_balance/best-route-v3-test-500.jsonl"
PORT=8200

cd ~/Block
export PYTHONPATH=~/Block:~/vllm:$PYTHONPATH
mkdir -p $RESULT_DIR

run() {
    local NAME=$1; shift
    python3 route_balance/benchmark/route_balance/benchmark_serving.py \
      --backend route_balance --host 127.0.0.1 --port $PORT \
      --dataset-name custom --dataset-path $DATASET \
      --num-prompts $NUM_REQ --trust-remote-code \
      --save-result --save-detailed \
      --result-dir $RESULT_DIR --result-filename "${NAME}.json" "$@" \
      2>&1 | tail -1
}

cfg() {
    curl -s -X POST http://localhost:$PORT/v1/config \
      -H "Content-Type: application/json" -d "$1" > /dev/null
}

echo "Weight Sweep — $(date)"

CONFIGS=(
    'balanced:{"w_latency":0.3,"w_cost":0.2,"w_quality":0.3,"w_balance":0.2}'
    'lat_focus:{"w_latency":0.6,"w_cost":0.1,"w_quality":0.2,"w_balance":0.1}'
    'qual_focus:{"w_latency":0.1,"w_cost":0.1,"w_quality":0.7,"w_balance":0.1}'
    'cost_focus:{"w_latency":0.1,"w_cost":0.6,"w_quality":0.2,"w_balance":0.1}'
    'bal_focus:{"w_latency":0.2,"w_cost":0.1,"w_quality":0.2,"w_balance":0.5}'
    'no_quality:{"w_latency":0.4,"w_cost":0.3,"w_quality":0.0,"w_balance":0.3}'
    'no_balance:{"w_latency":0.35,"w_cost":0.25,"w_quality":0.4,"w_balance":0.0}'
    'no_latency:{"w_latency":0.0,"w_cost":0.3,"w_quality":0.5,"w_balance":0.2}'
    'no_cost:{"w_latency":0.4,"w_cost":0.0,"w_quality":0.4,"w_balance":0.2}'
)

for entry in "${CONFIGS[@]}"; do
    NAME="${entry%%:*}"
    WEIGHTS="${entry#*:}"
    cfg "{\"scoring_weights\":$WEIGHTS}"
    echo "  $NAME: $WEIGHTS"
    for QPS in $QPS_LEVELS; do
        run "weight_${NAME}_qps${QPS}" --request-rate $QPS
    done
done

# Reset to balanced
cfg '{"scoring_weights":{"w_latency":0.3,"w_cost":0.2,"w_quality":0.3,"w_balance":0.2}}'

echo "Weight Sweep COMPLETE — $(ls -1 $RESULT_DIR/*.json | wc -l) results"
