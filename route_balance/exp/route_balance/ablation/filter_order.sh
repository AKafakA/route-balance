#!/bin/bash
# Filter Relax Order Ablation: which constraint to relax first
# Requires tight SLOs to trigger filtering
set -euo pipefail
RESULT_DIR=${1:-"experiment_output/ablation/filter_order"}
NUM_REQ=${2:-2000}
QPS_LEVELS=${3:-"5 10 15"}
PORT=8200
DATASET="data/route_balance/best-route-v3-test-500.jsonl"
cd ~/RouteBalance; export PYTHONPATH=~/RouteBalance:~/vllm:$PYTHONPATH; mkdir -p $RESULT_DIR

run() { local N=$1; shift; python3 route_balance/benchmark/route_balance/benchmark_serving.py --backend route_balance --host 127.0.0.1 --port $PORT --dataset-name custom --dataset-path $DATASET --num-prompts $NUM_REQ --trust-remote-code --save-result --save-detailed --result-dir $RESULT_DIR --result-filename "${N}.json" "$@" 2>&1 | tail -1; }
cfg() { curl -s -X POST http://localhost:$PORT/v1/config -H "Content-Type: application/json" -d "$1" > /dev/null; }
stats() { curl -s http://localhost:$PORT/v1/scheduling_stats > "$RESULT_DIR/${1}_stats.json" 2>/dev/null; }

echo "Filter Order Ablation — $(date)"
# Set tight SLOs to ensure all filters active
cfg '{"slo_defaults":{"ttft_slo_ms":100,"tpot_slo_ms":30,"quality_min":0.4,"budget_tokens":128,"budget_confidence_threshold":0.5}}'

for entry in \
    'ttft_first:["ttft","tpot","quality","budget"]' \
    'budget_first:["budget","ttft","tpot","quality"]' \
    'quality_first:["quality","budget","ttft","tpot"]' \
    'tpot_first:["tpot","ttft","quality","budget"]' \
; do
    NAME="${entry%%:*}"; ORD="${entry#*:}"
    cfg "{\"slo_defaults\":{\"relax_order\":$ORD}}"
    echo "  order=$NAME"
    for QPS in $QPS_LEVELS; do run "order_${NAME}_qps${QPS}" --request-rate $QPS; done
    stats "order_${NAME}"
done

cfg '{"slo_defaults":{"relax_order":["ttft","tpot","quality","budget"],"ttft_slo_ms":10000,"tpot_slo_ms":200,"quality_min":0.0,"budget_tokens":256}}'
echo "Filter Order COMPLETE"
