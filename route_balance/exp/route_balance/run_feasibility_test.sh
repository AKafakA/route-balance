#!/bin/bash
# ROUTE_BALANCE Comprehensive Feasibility Test — March 26, 2026
# Tests all schedulers, weight sweeps, batch configs, and collects overhead metrics
#
# Prerequisites: vLLM + predictors running, scheduler will be restarted per strategy
# Usage: bash run_feasibility_test.sh

set -e

SCHEDULER_PORT=8200
NUM_REQUESTS=50
DATASET_PATH="data/route_balance/best-route-v3-test-500.jsonl"
RESULT_DIR="experiment_output/feasibility_results"
PREDICTOR_CONFIG="route_balance/config/route_balance/predictor_config_smoketest.json"
DEPLOYMENT_CONFIG="route_balance/config/route_balance/model_deployment_smoketest.json"
HOST_CONFIG="route_balance/config/host_configs.json"

cd ~/Block
export PYTHONPATH=~/Block:~/vllm:$PYTHONPATH
mkdir -p $RESULT_DIR

run_benchmark() {
    local SCHED=$1
    local QPS=$2
    local RUN_NAME=$3
    local EXTRA_ARGS=${4:-""}

    echo ""
    echo "=== Running: $RUN_NAME (scheduler=$SCHED, qps=$QPS) ==="

    python3 route_balance/benchmark/route_balance/benchmark_serving.py \
      --backend route_balance \
      --host 127.0.0.1 \
      --port $SCHEDULER_PORT \
      --dataset-name custom \
      --dataset-path $DATASET_PATH \
      --num-prompts $NUM_REQUESTS \
      --request-rate $QPS \
      --trust-remote-code \
      --save-result \
      --save-detailed \
      --result-dir $RESULT_DIR \
      --result-filename "${RUN_NAME}.json" \
      --metadata scheduler=$SCHED qps=$QPS $EXTRA_ARGS \
      2>&1 | tail -5

    echo "Done: $RUN_NAME"
}

start_scheduler() {
    local SCHED=$1
    pkill -f 'route_balance_serve' 2>/dev/null || true
    sleep 2

    SCHED_ARGS="--host 0.0.0.0 --port $SCHEDULER_PORT"
    SCHED_ARGS="$SCHED_ARGS --model_config_path $DEPLOYMENT_CONFIG"
    SCHED_ARGS="$SCHED_ARGS --host_config $HOST_CONFIG"
    SCHED_ARGS="$SCHED_ARGS --scheduling $SCHED"
    SCHED_ARGS="$SCHED_ARGS --predictor-config $PREDICTOR_CONFIG"
    SCHED_ARGS="$SCHED_ARGS --chat"
    # Only enable predictor feedback for route_balance (needed for sidecar calls)
    if [ "$SCHED" = "route_balance" ] || [ "$SCHED" = "length_aware" ]; then
        SCHED_ARGS="$SCHED_ARGS --enable-predictor-feedback --feedback-sample-rate 0.0"
    fi

    nohup python3 -u -m route_balance.global_scheduler.route_balance.route_balance_serve $SCHED_ARGS \
      > experiment_output/logs/route_balance_serve_${SCHED}.log 2>&1 &
    sleep 6

    if pgrep -f 'route_balance_serve' > /dev/null; then
        echo "Scheduler started: $SCHED"
    else
        echo "ERROR: Scheduler failed: $SCHED"
        tail -10 experiment_output/logs/route_balance_serve_${SCHED}.log
        return 1
    fi
}

update_config() {
    local JSON=$1
    curl -s -X POST http://localhost:$SCHEDULER_PORT/v1/config \
      -H "Content-Type: application/json" \
      -d "$JSON" > /dev/null
}

echo "========================================"
echo "ROUTE_BALANCE Feasibility Test"
echo "========================================"
echo "Time: $(date)"
echo ""

# ============================================
# PART 1: All 7 schedulers at QPS=2,5
# ============================================
echo ""
echo "======== PART 1: All Schedulers ========"

for SCHED in random round_robin shortest_queue quality_greedy cost_greedy length_aware route_balance; do
    start_scheduler $SCHED
    for QPS in 2 5; do
        run_benchmark $SCHED $QPS "${SCHED}_qps${QPS}"
    done
done

# ============================================
# PART 2: Weight sweep (route_balance only, QPS=5)
# ============================================
echo ""
echo "======== PART 2: Weight Sweep ========"

start_scheduler route_balance

# Balanced (default)
update_config '{"scoring_weights":{"w_latency":0.3,"w_cost":0.2,"w_quality":0.3,"w_balance":0.2}}'
run_benchmark route_balance 5 "weight_balanced"

# Latency focused
update_config '{"scoring_weights":{"w_latency":0.6,"w_cost":0.1,"w_quality":0.2,"w_balance":0.1}}'
run_benchmark route_balance 5 "weight_latency"

# Quality focused
update_config '{"scoring_weights":{"w_latency":0.1,"w_cost":0.1,"w_quality":0.7,"w_balance":0.1}}'
run_benchmark route_balance 5 "weight_quality"

# Cost focused
update_config '{"scoring_weights":{"w_latency":0.1,"w_cost":0.6,"w_quality":0.2,"w_balance":0.1}}'
run_benchmark route_balance 5 "weight_cost"

# Reset to default
update_config '{"scoring_weights":{"w_latency":0.3,"w_cost":0.2,"w_quality":0.3,"w_balance":0.2}}'

# ============================================
# PART 3: Batch config sweep (route_balance, QPS=5)
# ============================================
echo ""
echo "======== PART 3: Batch Config Sweep ========"

# Small fast batches
update_config '{"slo_defaults":{"batch_config":{"max_batch_size":4,"batch_timeout_ms":25,"adaptive_sizing":false}}}'
run_benchmark route_balance 5 "batch_small_fast"

# Medium (default)
update_config '{"slo_defaults":{"batch_config":{"max_batch_size":8,"batch_timeout_ms":50,"adaptive_sizing":true}}}'
run_benchmark route_balance 5 "batch_medium"

# Large slow
update_config '{"slo_defaults":{"batch_config":{"max_batch_size":16,"batch_timeout_ms":200,"adaptive_sizing":false}}}'
run_benchmark route_balance 5 "batch_large_slow"

# LPT min
update_config '{"slo_defaults":{"lpt_sort_key":"min","batch_config":{"max_batch_size":8,"batch_timeout_ms":50,"adaptive_sizing":true}}}'
run_benchmark route_balance 5 "lpt_min"

# LPT mean
update_config '{"slo_defaults":{"lpt_sort_key":"mean"}}'
run_benchmark route_balance 5 "lpt_mean"

# Reset
update_config '{"slo_defaults":{"lpt_sort_key":"max","batch_config":{"max_batch_size":8,"batch_timeout_ms":100,"adaptive_sizing":true}}}'

# ============================================
# PART 4: Higher QPS stress test (route_balance)
# ============================================
echo ""
echo "======== PART 4: Stress Test ========"
run_benchmark route_balance 10 "route_balance_qps10"

# Capture final batch stats
echo "Final batch stats:"
curl -s http://localhost:$SCHEDULER_PORT/v1/batch_stats
echo ""

# Cleanup
pkill -f 'route_balance_serve' 2>/dev/null || true

echo ""
echo "========================================"
echo "Feasibility test complete!"
echo "========================================"
echo "Results: $RESULT_DIR/"
ls -1 $RESULT_DIR/*.json | wc -l
echo "json files generated"
echo ""
echo "Analyze: python3 route_balance/exp/route_balance/analyze_smoketest.py --result-dir $RESULT_DIR"
