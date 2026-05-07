#!/bin/bash
# ROUTE_BALANCE Comprehensive Experiment Runner
# Covers all experiment categories for CloudLab evaluation
# Usage: bash run_all_experiments.sh [result_dir]

set -e

PORT=8200
NUM_REQ=50
DATASET="data/route_balance/best-route-v3-test-500.jsonl"
RESULT_DIR=${1:-"experiment_output/comprehensive_results"}
DEPLOY_CONFIG="route_balance/config/route_balance/model_deployment_smoketest.json"
HOST_CONFIG="route_balance/config/host_configs.json"
PRED_CONFIG="route_balance/config/route_balance/predictor_config_smoketest.json"

cd ~/RouteBalance
export PYTHONPATH=~/RouteBalance:~/vllm:$PYTHONPATH
mkdir -p $RESULT_DIR

run() {
    local NAME=$1; shift
    echo "--- $NAME ---"
    python3 route_balance/benchmark/route_balance/benchmark_serving.py \
      --backend route_balance --host 127.0.0.1 --port $PORT \
      --dataset-name custom --dataset-path $DATASET \
      --num-prompts $NUM_REQ --trust-remote-code \
      --save-result --save-detailed \
      --result-dir $RESULT_DIR --result-filename "${NAME}.json" "$@" \
      2>&1 | grep -E "Throughput|TTFT|E2EL|saved|completed"
}

start_sched() {
    local SCHED=$1
    pkill -f route_balance_serve 2>/dev/null || true; sleep 2
    local ARGS="--host 0.0.0.0 --port $PORT"
    ARGS="$ARGS --model_config_path $DEPLOY_CONFIG --host_config $HOST_CONFIG"
    ARGS="$ARGS --scheduling $SCHED --predictor-config $PRED_CONFIG --chat"
    if [ "$SCHED" = "route_balance" ] || [ "$SCHED" = "length_aware" ]; then
        ARGS="$ARGS --enable-predictor-feedback --feedback-sample-rate 0.0"
    fi
    nohup python3 -u -m route_balance.global_scheduler.route_balance.route_balance_serve $ARGS \
      > experiment_output/logs/route_balance_${SCHED}.log 2>&1 &
    sleep 6
    pgrep -f route_balance_serve > /dev/null || { echo "FAIL: $SCHED"; return 1; }
    # Warmup estimator
    curl -s -X POST http://localhost:$PORT/v1/estimate \
      -H "Content-Type: application/json" -d '{"prompt":"warmup"}' > /dev/null 2>&1
    echo "Scheduler: $SCHED"
}

cfg() {
    curl -s -X POST http://localhost:$PORT/v1/config \
      -H "Content-Type: application/json" -d "$1" > /dev/null
}

echo "======== PART A: All 7 Schedulers (QPS=2,5) ========"
for S in random round_robin shortest_queue quality_greedy cost_greedy length_aware route_balance; do
    start_sched $S
    for Q in 2 5; do
        run "${S}_qps${Q}" --request-rate $Q
    done
done

echo "======== PART B: Weight Sweep (9 configs, QPS=5) ========"
start_sched route_balance
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
    run "weight_${NAME}" --request-rate 5
done
cfg '{"scoring_weights":{"w_latency":0.3,"w_cost":0.2,"w_quality":0.3,"w_balance":0.2}}'

echo "======== PART C: Batch Config Sweep (QPS=5) ========"
for BS_TO in "4:25" "8:50" "16:100" "32:200"; do
    BS="${BS_TO%%:*}"; TO="${BS_TO#*:}"
    cfg "{\"slo_defaults\":{\"batch_config\":{\"max_batch_size\":$BS,\"batch_timeout_ms\":$TO,\"adaptive_sizing\":false}}}"
    run "batch_${BS}_${TO}ms" --request-rate 5
done
# No batching baseline (batch_size=1, timeout=0)
cfg '{"slo_defaults":{"batch_config":{"max_batch_size":1,"batch_timeout_ms":1,"adaptive_sizing":false}}}'
run "batch_nobatch" --request-rate 5
# Reset
cfg '{"slo_defaults":{"batch_config":{"max_batch_size":8,"batch_timeout_ms":50,"adaptive_sizing":true}}}'

echo "======== PART D: LPT Sort Key (QPS=5) ========"
for KEY in max min mean; do
    cfg "{\"slo_defaults\":{\"lpt_sort_key\":\"$KEY\"}}"
    run "lpt_${KEY}" --request-rate 5
done
cfg '{"slo_defaults":{"lpt_sort_key":"max"}}'

echo "======== PART E: Budget Sweep (QPS=5) ========"
for BUDGET in 64 128 256 512 1024; do
    cfg "{\"slo_defaults\":{\"budget_tokens\":$BUDGET}}"
    run "budget_${BUDGET}" --request-rate 5
done
cfg '{"slo_defaults":{"budget_tokens":256}}'

echo "======== PART F: Budget Threshold Sweep (QPS=5) ========"
for THRESH in "0.0" "0.3" "0.5" "0.7" "0.9"; do
    cfg "{\"slo_defaults\":{\"budget_confidence_threshold\":$THRESH}}"
    run "thresh_${THRESH}" --request-rate 5
done
cfg '{"slo_defaults":{"budget_confidence_threshold":0.5}}'

echo "======== PART G: Constraint Mode (QPS=5, tight SLO) ========"
for MODE in STRICT TIERED RELAXED; do
    cfg "{\"slo_defaults\":{\"constraint_mode\":\"$MODE\",\"ttft_slo_ms\":3000}}"
    run "constraint_${MODE}_3s" --request-rate 5
done
cfg '{"slo_defaults":{"constraint_mode":"TIERED","ttft_slo_ms":10000}}'

echo "======== PART H: QPS Scaling (route_balance) ========"
for Q in 1 2 5 10 15 20; do
    run "route_balance_qps${Q}" --request-rate $Q
done

echo "======== PART I: Quality Threshold Sweep (QPS=5) ========"
for QMIN in "0.0" "0.3" "0.5" "0.7" "0.9"; do
    cfg "{\"slo_defaults\":{\"quality_min\":$QMIN}}"
    run "qualmin_${QMIN}" --request-rate 5
done
cfg '{"slo_defaults":{"quality_min":0.0}}'

echo "======== PART J: Filter Order Ablation (QPS=5, tight SLOs) ========"
# Tight SLOs to ensure filters are active
cfg '{"slo_defaults":{"ttft_slo_ms":100,"tpot_slo_ms":30,"quality_min":0.4,"budget_tokens":128,"budget_confidence_threshold":0.5,"constraint_mode":"TIERED"}}'

# Default order: ttft → tpot → quality → budget
for ORDER in \
    'ttft_first:["ttft","tpot","quality","budget"]' \
    'budget_first:["budget","ttft","tpot","quality"]' \
    'quality_first:["quality","budget","ttft","tpot"]' \
    'tpot_first:["tpot","ttft","quality","budget"]' \
; do
    NAME="${ORDER%%:*}"; ORD="${ORDER#*:}"
    cfg "{\"slo_defaults\":{\"relax_order\":$ORD}}"
    run "relax_order_${NAME}" --request-rate 5
done
cfg '{"slo_defaults":{"relax_order":["ttft","tpot","quality","budget"]}}'
# Reset SLOs
cfg '{"slo_defaults":{"ttft_slo_ms":10000,"tpot_slo_ms":200,"quality_min":0.0,"budget_tokens":256,"budget_confidence_threshold":0.5}}'

echo "======== PART K: Per-Filter Impact (QPS=5) ========"
# Disable one filter at a time to measure each filter's contribution
# Baseline: all filters active with moderate SLOs
cfg '{"slo_defaults":{"ttft_slo_ms":200,"tpot_slo_ms":50,"quality_min":0.3,"budget_tokens":128,"budget_confidence_threshold":0.5,"constraint_mode":"TIERED"}}'
run "filter_all_active" --request-rate 5

# No budget filter (threshold=0)
cfg '{"slo_defaults":{"budget_confidence_threshold":0.0}}'
run "filter_no_budget" --request-rate 5
cfg '{"slo_defaults":{"budget_confidence_threshold":0.5}}'

# No TTFT filter (very loose)
cfg '{"slo_defaults":{"ttft_slo_ms":999999}}'
run "filter_no_ttft" --request-rate 5
cfg '{"slo_defaults":{"ttft_slo_ms":200}}'

# No TPOT filter (very loose)
cfg '{"slo_defaults":{"tpot_slo_ms":999999}}'
run "filter_no_tpot" --request-rate 5
cfg '{"slo_defaults":{"tpot_slo_ms":50}}'

# No quality filter (min=0)
cfg '{"slo_defaults":{"quality_min":0.0}}'
run "filter_no_quality" --request-rate 5
cfg '{"slo_defaults":{"quality_min":0.3}}'

# No filters at all (RELAXED mode)
cfg '{"slo_defaults":{"constraint_mode":"RELAXED"}}'
run "filter_none" --request-rate 5

# Reset
cfg '{"slo_defaults":{"constraint_mode":"TIERED","ttft_slo_ms":10000,"tpot_slo_ms":200,"quality_min":0.0,"budget_tokens":256,"budget_confidence_threshold":0.5}}'

echo "======== PART L: RSO Density Sweep (QPS=5) ========"
for RATE in "0.0" "0.1" "0.3" "0.5" "1.0"; do
    run "rso_rate_${RATE}" --request-rate 5 --rso-rate $RATE
done

# Final stats
echo "" && echo "=== Final batch stats ===" && curl -s http://localhost:$PORT/v1/batch_stats
echo "" && echo "=== Final scheduling stats ===" && curl -s http://localhost:$PORT/v1/scheduling_stats | python3 -m json.tool
pkill -f route_balance_serve 2>/dev/null || true

echo ""
echo "========================================"
echo "ALL EXPERIMENTS COMPLETE"
echo "========================================"
ls -1 $RESULT_DIR/*.json | wc -l
echo "result files in $RESULT_DIR/"
echo "Analyze: python3 route_balance/exp/route_balance/analyze_smoketest.py --result-dir $RESULT_DIR"
