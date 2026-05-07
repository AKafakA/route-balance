#!/bin/bash
# P100 Overhead Study: Compare scheduling overhead across C1-C4 estimator configs
# Each config requires a scheduler restart (different model_estimator loaded)
#
# C1: ModernBERT bucket (no regression) + KNN judge
# C2: ModernBERT bucket + regression + KNN judge
# C3: RoBERTa bucket (no regression) + KNN judge
# C4: RoBERTa bucket + KNN reference_score
#
# Usage: bash route_balance/exp/route_balance/p100_overhead_test.sh [NUM_PROMPTS] [QPS]
set -euo pipefail

NUM_PROMPTS=${1:-50}
QPS=${2:-2}
PORT=8200
DATASET="data/route_balance/best-route-v3-test-500.jsonl"
MODEL_DEPLOY="route_balance/config/route_balance/model_deployment_p100_smoketest.json"
RESULT_BASE="experiment_output/p100_overhead"

cd ~/RouteBalance
export PYTHONPATH=~/RouteBalance:~/vllm:$PYTHONPATH
mkdir -p $RESULT_BASE/logs

echo "============================================================"
echo "  P100 Overhead Study — $(date)"
echo "  Prompts=$NUM_PROMPTS, QPS=$QPS"
echo "============================================================"

deploy_scheduler() {
    local CONFIG_NAME=$1
    local CONFIG_PATH=$2
    echo ""
    echo "--- Deploying $CONFIG_NAME ($CONFIG_PATH) ---"

    # Kill existing scheduler
    pkill -f 'route_balance_serve' 2>/dev/null || true
    sleep 3

    # Start scheduler
    nohup python3 -u -m route_balance.global_scheduler.route_balance.route_balance_serve \
      --host 0.0.0.0 --port $PORT \
      --model_config_path $MODEL_DEPLOY \
      --scheduling route_balance --chat \
      --scheduler-config $CONFIG_PATH \
      > "$RESULT_BASE/logs/scheduler_${CONFIG_NAME}.log" 2>&1 &

    # Wait for ready
    local ready=0
    for i in $(seq 1 24); do
        sleep 5
        if curl -sf http://localhost:$PORT/v1/batch_stats > /dev/null 2>&1; then
            ready=1
            echo "  Scheduler ready after ${i}x5s"
            break
        fi
    done
    if [ $ready -eq 0 ]; then
        echo "  FAIL: Scheduler did not start!"
        tail -10 "$RESULT_BASE/logs/scheduler_${CONFIG_NAME}.log"
        return 1
    fi
}

run_test() {
    local NAME=$1
    echo "  Running benchmark: $NAME"
    python3 route_balance/exp/route_balance/p100_smoke_client.py \
        --host 127.0.0.1 --port $PORT \
        --dataset $DATASET \
        --num-prompts $NUM_PROMPTS --request-rate $QPS \
        --output "$RESULT_BASE/${NAME}.json" \
        2>&1 | tail -8
    curl -sf http://localhost:$PORT/v1/scheduling_stats > "$RESULT_BASE/${NAME}_stats.json" 2>/dev/null
}

# Run each config
for CONFIG in c1 c2 c3 c4; do
    CONFIG_PATH="route_balance/config/route_balance/scheduler_config_p100_${CONFIG}.json"
    if [ ! -f "$CONFIG_PATH" ]; then
        echo "  WARNING: $CONFIG_PATH not found, skipping"
        continue
    fi
    deploy_scheduler "$CONFIG" "$CONFIG_PATH" || continue
    run_test "overhead_${CONFIG}"
done

# Cleanup
pkill -f 'route_balance_serve' 2>/dev/null || true

echo ""
echo "============================================================"
echo "  Overhead Study Complete — $(date)"
echo "  Results in: $RESULT_BASE/"
echo "============================================================"

# Summary
echo ""
echo "=== Summary ==="
for CONFIG in c1 c2 c3 c4; do
    FILE="$RESULT_BASE/overhead_${CONFIG}.json"
    if [ -f "$FILE" ]; then
        python3 -c "
import json
d = json.load(open('$FILE'))
s = d['stats']
print(f'  ${CONFIG}: TTFT P50={s[\"ttft_p50_ms\"]:.0f}ms P95={s[\"ttft_p95_ms\"]:.0f}ms | E2E P50={s[\"e2e_p50_ms\"]:.0f}ms | {s[\"num_successful\"]}/{s[\"num_requests\"]} OK')
" 2>/dev/null || echo "  ${CONFIG}: parse error"
    else
        echo "  ${CONFIG}: no results"
    fi
done
