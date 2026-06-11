#!/bin/bash
# E2E Baseline Comparison: 8 scheduling strategies × multiple QPS levels
# Redeploys scheduler per strategy, keeps vLLM + monitor running.
#
# Usage: bash route_balance/exp/route_balance/e2e/run.sh [result_dir] [num_requests] [qps_levels]
# Example: bash route_balance/exp/route_balance/e2e/run.sh experiment_output/e2e 2000 "2 5 10 15 20"
set -euo pipefail

RESULT_DIR=${1:-"experiment_output/e2e"}
NUM_REQ=${2:-2000}
QPS_LEVELS=${3:-"2 5 10 15 20"}
DATASET="data/route_balance/best-route-v3-test-500.jsonl"
PORT=8200

# Configs — override via environment variables or adjust defaults per cluster
MODEL_DEPLOY=${MODEL_DEPLOY:-"route_balance/config/route_balance/model_deployment.json"}
SCHEDULER_CONFIG=${SCHEDULER_CONFIG:-"route_balance/config/route_balance/scheduler_config.json"}
PREDICTOR_CONFIG=${PREDICTOR_CONFIG:-""}

# All strategies to test
STRATEGIES="random round_robin shortest_queue quality_greedy cost_greedy length_aware route_balance"

cd ~/Block
export PYTHONPATH=~/Block:~/vllm:$PYTHONPATH
mkdir -p $RESULT_DIR experiment_output/logs

PASS=0; FAIL=0

run() {
    local NAME=$1; shift
    echo "  [$NAME] $NUM_REQ requests..."
    if python3 route_balance/benchmark/route_balance/benchmark_serving.py \
      --backend route_balance --host 127.0.0.1 --port $PORT \
      --dataset-name custom --dataset-path $DATASET \
      --num-prompts $NUM_REQ --trust-remote-code \
      --save-result --save-detailed \
      --result-dir $RESULT_DIR --result-filename "${NAME}.json" "$@" \
      2>&1 | tail -3; then
        [ -f "$RESULT_DIR/${NAME}.json" ] && { echo "  [$NAME] PASS"; PASS=$((PASS+1)); } \
          || { echo "  [$NAME] FAIL (no output)"; FAIL=$((FAIL+1)); }
    else
        echo "  [$NAME] FAIL (benchmark error)"; FAIL=$((FAIL+1))
    fi
}

deploy_scheduler() {
    local SCHED=$1
    local EXTRA_ARGS=""
    if [ "$SCHED" = "route_balance" ] || [ "$SCHED" = "length_aware" ]; then
        EXTRA_ARGS="--enable-predictor-feedback --feedback-sample-rate 0.0"
    fi

    # Kill previous scheduler by PID file (pkill -f would match this script too)
    if [ -f /tmp/scheduler_pid ]; then
        kill $(cat /tmp/scheduler_pid) 2>/dev/null || true
        # Wait for port to be released
        for i in $(seq 1 10); do
            curl -sf http://localhost:$PORT/health > /dev/null 2>&1 || break
            sleep 1
        done
    fi
    sleep 1

    # Use scheduler_config if available, else fall back to predictor_config (legacy)
    local CONFIG_ARGS=""
    if [ -f "$SCHEDULER_CONFIG" ]; then
        CONFIG_ARGS="--scheduler-config $SCHEDULER_CONFIG"
    fi
    if [ -f "$PREDICTOR_CONFIG" ]; then
        CONFIG_ARGS="$CONFIG_ARGS --predictor-config $PREDICTOR_CONFIG"
    fi

    nohup python3 -u -m route_balance.global_scheduler.route_balance.route_balance_serve \
      --host 0.0.0.0 --port $PORT \
      --model_config_path $MODEL_DEPLOY \
      --scheduling $SCHED \
      $CONFIG_ARGS \
      --chat $EXTRA_ARGS \
      > experiment_output/logs/route_balance_${SCHED}.log 2>&1 &
    echo $! > /tmp/scheduler_pid
    sleep 8

    # Health check (use /health which works for all strategies)
    for i in $(seq 1 12); do
        curl -sf http://localhost:$PORT/health > /dev/null 2>&1 && break
        sleep 5
    done
    if ! curl -sf http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "  FAIL: scheduler $SCHED did not start"
        return 1
    fi
    echo "  Scheduler: $SCHED"
}

echo "========================================================"
echo "E2E Baseline Comparison — $(date)"
echo "Strategies: $STRATEGIES"
echo "QPS levels: $QPS_LEVELS"
echo "Requests per run: $NUM_REQ"
echo "========================================================"

for SCHED in $STRATEGIES; do
    echo ""
    echo "--- Strategy: $SCHED ---"
    deploy_scheduler $SCHED || continue

    for QPS in $QPS_LEVELS; do
        run "${SCHED}_qps${QPS}" --request-rate $QPS
    done

    # Collect scheduling stats
    curl -s http://localhost:$PORT/v1/scheduling_stats > "$RESULT_DIR/${SCHED}_stats.json" 2>/dev/null
done

pkill -f '^python.*route_balance_serve' 2>/dev/null || true

echo ""
echo "========================================================"
echo "E2E COMPLETE — $(date)"
echo "PASS: $PASS  FAIL: $FAIL"
echo "Results: $(ls -1 $RESULT_DIR/*.json 2>/dev/null | wc -l) files in $RESULT_DIR/"
echo "========================================================"
