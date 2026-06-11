#!/bin/bash
# Per-Filter Impact Ablation: disable one filter at a time
# Requires moderate SLOs to trigger filtering
set -euo pipefail

RESULT_DIR=${1:-"experiment_output/ablation/filter_impact"}
NUM_REQ=${2:-2000}
QPS_LEVELS=${3:-"5 10 15"}
PORT=8200
DATASET="data/route_balance/best-route-v3-test-500.jsonl"

cd ~/Block
export PYTHONPATH=~/Block:~/vllm:$PYTHONPATH
mkdir -p $RESULT_DIR

run() { local N=$1; shift; python3 route_balance/benchmark/route_balance/benchmark_serving.py --backend route_balance --host 127.0.0.1 --port $PORT --dataset-name custom --dataset-path $DATASET --num-prompts $NUM_REQ --trust-remote-code --save-result --save-detailed --result-dir $RESULT_DIR --result-filename "${N}.json" "$@" 2>&1 | tail -1; }
cfg() { curl -s -X POST http://localhost:$PORT/v1/config -H "Content-Type: application/json" -d "$1" > /dev/null; }
stats() { curl -s http://localhost:$PORT/v1/scheduling_stats > "$RESULT_DIR/${1}_stats.json" 2>/dev/null; }

echo "Filter Impact Ablation — $(date)"

# Set moderate SLOs to ensure filters are active
BASELINE_SLO='{"slo_defaults":{"ttft_slo_ms":200,"tpot_slo_ms":50,"quality_min":0.3,"budget_tokens":128,"budget_confidence_threshold":0.5,"constraint_mode":"TIERED"}}'
cfg "$BASELINE_SLO"

echo "  All filters active"
for QPS in $QPS_LEVELS; do run "all_active_qps${QPS}" --request-rate $QPS; done
stats "all_active"

echo "  No budget filter"
cfg '{"slo_defaults":{"budget_confidence_threshold":0.0}}'
for QPS in $QPS_LEVELS; do run "no_budget_qps${QPS}" --request-rate $QPS; done
stats "no_budget"
cfg '{"slo_defaults":{"budget_confidence_threshold":0.5}}'

echo "  No TTFT filter"
cfg '{"slo_defaults":{"ttft_slo_ms":999999}}'
for QPS in $QPS_LEVELS; do run "no_ttft_qps${QPS}" --request-rate $QPS; done
stats "no_ttft"
cfg '{"slo_defaults":{"ttft_slo_ms":200}}'

echo "  No TPOT filter"
cfg '{"slo_defaults":{"tpot_slo_ms":999999}}'
for QPS in $QPS_LEVELS; do run "no_tpot_qps${QPS}" --request-rate $QPS; done
stats "no_tpot"
cfg '{"slo_defaults":{"tpot_slo_ms":50}}'

echo "  No quality filter"
cfg '{"slo_defaults":{"quality_min":0.0}}'
for QPS in $QPS_LEVELS; do run "no_quality_qps${QPS}" --request-rate $QPS; done
stats "no_quality"
cfg '{"slo_defaults":{"quality_min":0.3}}'

echo "  No filters (RELAXED)"
cfg '{"slo_defaults":{"constraint_mode":"RELAXED"}}'
for QPS in $QPS_LEVELS; do run "no_filters_qps${QPS}" --request-rate $QPS; done
stats "no_filters"

# Reset
cfg '{"slo_defaults":{"constraint_mode":"TIERED","ttft_slo_ms":10000,"tpot_slo_ms":200,"quality_min":0.0,"budget_tokens":256,"budget_confidence_threshold":0.5}}'
echo "Filter Impact Ablation COMPLETE"
