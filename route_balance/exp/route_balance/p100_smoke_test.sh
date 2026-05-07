#!/bin/bash
# P100 Smoke Test: Filter baselines + Overhead study (C1-C4)
# Run from node0 (scheduler node)
#
# Tests:
# 1. Basic e2e: verify scheduler routes requests correctly
# 2. Filter baselines: switch filter type via /v1/config, verify different accept/reject rates
# 3. Overhead study: run C1-C4 configs, measure scheduling latency per config
#
# Usage: bash route_balance/exp/route_balance/p100_smoke_test.sh [NUM_PROMPTS] [QPS]
set -euo pipefail

NUM_PROMPTS=${1:-50}
QPS=${2:-2}
PORT=8200
DATASET="data/route_balance/best-route-v3-test-500.jsonl"
RESULT_BASE="experiment_output/p100_smoke"

cd ~/RouteBalance
export PYTHONPATH=~/RouteBalance:~/vllm:$PYTHONPATH
mkdir -p $RESULT_BASE/logs

echo "============================================================"
echo "  P100 Smoke Test — $(date)"
echo "  Prompts=$NUM_PROMPTS, QPS=$QPS"
echo "============================================================"

# Helper: send requests via simple Python client
run_test() {
    local NAME=$1
    local N=${2:-$NUM_PROMPTS}
    local R=${3:-$QPS}
    echo "  Running: $NAME (n=$N, qps=$R)"
    python3 route_balance/exp/route_balance/p100_smoke_client.py \
        --host 127.0.0.1 --port $PORT \
        --dataset $DATASET \
        --num-prompts $N --request-rate $R \
        --output "$RESULT_BASE/${NAME}.json" \
        2>&1 | tail -5
    # Collect scheduling stats
    curl -sf http://localhost:$PORT/v1/scheduling_stats > "$RESULT_BASE/${NAME}_stats.json" 2>/dev/null
    echo "  Done: $NAME"
}

# Helper: update scheduler config
cfg() {
    curl -sf -X POST http://localhost:$PORT/v1/config \
        -H "Content-Type: application/json" -d "$1" > /dev/null
    echo "  Config updated"
}

# Reset stats
reset_stats() {
    curl -sf -X POST http://localhost:$PORT/v1/scheduling_stats/reset > /dev/null 2>&1 || true
}

# ============================================================
# Phase 1: Basic E2E Sanity (5 requests)
# ============================================================
echo ""
echo "=== Phase 1: Basic E2E Sanity ==="
reset_stats
run_test "sanity" 5 1

# ============================================================
# Phase 2: Filter Baseline Comparison
# ============================================================
echo ""
echo "=== Phase 2: Filter Baselines ==="

# 2a. No filter (accept all)
echo ""
echo "--- 2a: No filter ---"
cfg '{"slo_defaults":{"filter":{"type":"none"}}}'
reset_stats
run_test "filter_none" $NUM_PROMPTS $QPS

# 2b. SLOs-Serve (binary point prediction)
echo ""
echo "--- 2b: SLOs-Serve ---"
cfg '{"slo_defaults":{"filter":{"type":"slos_serve"},"ttft_slo_ms":5000,"tpot_slo_ms":100}}'
reset_stats
run_test "filter_slos_serve" $NUM_PROMPTS $QPS

# 2c. PolyServe (cumulative deadline)
echo ""
echo "--- 2c: PolyServe ---"
cfg '{"slo_defaults":{"filter":{"type":"polyserve"},"ttft_slo_ms":5000,"tpot_slo_ms":100}}'
reset_stats
run_test "filter_polyserve" $NUM_PROMPTS $QPS

# 2d. QLM (Normal confidence)
echo ""
echo "--- 2d: QLM ---"
cfg '{"slo_defaults":{"filter":{"type":"qlm","confidence":0.95},"ttft_slo_ms":5000,"tpot_slo_ms":100}}'
reset_stats
run_test "filter_qlm" $NUM_PROMPTS $QPS

# 2e. RouteBalance CDF hard reject
echo ""
echo "--- 2e: RouteBalance CDF hard ---"
cfg '{"slo_defaults":{"filter":{"type":"route_balance_hard_reject","confidence_threshold":0.7},"ttft_slo_ms":5000,"tpot_slo_ms":100}}'
reset_stats
run_test "filter_route_balance_hard" $NUM_PROMPTS $QPS

# 2f. RouteBalance CDF tiered (default)
echo ""
echo "--- 2f: RouteBalance CDF tiered ---"
cfg '{"slo_defaults":{"filter":{"type":"route_balance_tiered","confidence_threshold":0.7},"ttft_slo_ms":5000,"tpot_slo_ms":100}}'
reset_stats
run_test "filter_route_balance_tiered" $NUM_PROMPTS $QPS

# Reset filter to none for overhead tests
cfg '{"slo_defaults":{"filter":{"type":"none"}}}'

# ============================================================
# Phase 3: Overhead Study (C1-C4 require scheduler restart)
# Phase 3 is documented but run separately per config
# ============================================================
echo ""
echo "=== Phase 2 Complete ==="
echo "Results saved to: $RESULT_BASE/"
echo ""
echo "Phase 3 (Overhead C1-C4) requires scheduler restart per config."
echo "Run: bash route_balance/exp/route_balance/p100_overhead_test.sh"
echo ""
echo "============================================================"
echo "  P100 Smoke Test Complete — $(date)"
echo "============================================================"
