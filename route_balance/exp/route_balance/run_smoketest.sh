#!/bin/bash
# ROUTE_BALANCE E2E Smoke Test — March 26, 2026
# Tests the full pipeline: multiple schedulers, benchmark, metrics collection
#
# Prerequisites:
#   - vLLM instances running on both nodes
#   - Predictor sidecars running on both nodes
#   - Test data at data/route_balance/best-route-v3-test-500.jsonl
#
# Usage: bash run_smoketest.sh [scheduler_node_hostname]

set -e

SCHEDULER_NODE=${1:-"d7525-10s10337.cluster.example"}
SCHEDULER_IP="128.105.146.28"
SCHEDULER_PORT=8200
NUM_REQUESTS=50
QPS_LEVELS="2 5"
RESULT_DIR="experiment_output/smoketest_results"
DATASET_PATH="data/route_balance/best-route-v3-test-500.jsonl"

# Schedulers to test
# route_balance = batch scheduling (new), others = per-request baselines
SCHEDULERS="random round_robin shortest_queue route_balance"

echo "========================================"
echo "ROUTE_BALANCE E2E Smoke Test"
echo "========================================"
echo "Scheduler node: $SCHEDULER_NODE"
echo "Schedulers: $SCHEDULERS"
echo "QPS levels: $QPS_LEVELS"
echo "Requests per run: $NUM_REQUESTS"
echo ""

cd ~/RouteBalance
export PYTHONPATH=~/RouteBalance:~/vllm:$PYTHONPATH
mkdir -p $RESULT_DIR

for SCHED in $SCHEDULERS; do
  for QPS in $QPS_LEVELS; do
    RUN_NAME="${SCHED}_qps${QPS}"
    echo ""
    echo "========================================"
    echo "Running: $RUN_NAME"
    echo "========================================"

    # Restart scheduler with this strategy
    pkill -f 'route_balance_serve' 2>/dev/null || true
    sleep 2

    SCHED_ARGS="--host 0.0.0.0 --port $SCHEDULER_PORT"
    SCHED_ARGS="$SCHED_ARGS --model_config_path route_balance/config/route_balance/model_deployment_smoketest.json"
    SCHED_ARGS="$SCHED_ARGS --host_config route_balance/config/host_configs.json"
    SCHED_ARGS="$SCHED_ARGS --scheduling $SCHED"
    SCHED_ARGS="$SCHED_ARGS --predictor-config route_balance/config/route_balance/predictor_config_smoketest.json"
    SCHED_ARGS="$SCHED_ARGS --chat"

    nohup python3 -u -m route_balance.global_scheduler.route_balance.route_balance_serve $SCHED_ARGS \
      > experiment_output/logs/route_balance_serve_${RUN_NAME}.log 2>&1 &

    echo "Waiting for scheduler to start..."
    sleep 8

    # Verify scheduler is running
    if ! pgrep -f 'route_balance_serve' > /dev/null; then
      echo "ERROR: Scheduler failed to start for $RUN_NAME"
      tail -10 experiment_output/logs/route_balance_serve_${RUN_NAME}.log
      continue
    fi

    # Quick health check
    curl -s http://localhost:$SCHEDULER_PORT/v1/batch_stats > /dev/null 2>&1 || {
      echo "ERROR: Scheduler not responding for $RUN_NAME"
      continue
    }

    echo "Scheduler running ($SCHED). Starting benchmark..."

    # Run benchmark
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
      --metadata scheduler=$SCHED qps=$QPS num_requests=$NUM_REQUESTS \
      2>&1 | tee $RESULT_DIR/${RUN_NAME}_console.log

    echo ""
    echo "Completed: $RUN_NAME"

    # If route_balance scheduler, capture batch stats
    if [ "$SCHED" = "route_balance" ]; then
      echo "Batch stats:"
      curl -s http://localhost:$SCHEDULER_PORT/v1/batch_stats 2>/dev/null
      echo ""
    fi

    echo "---"
  done
done

# Cleanup
pkill -f 'route_balance_serve' 2>/dev/null || true

echo ""
echo "========================================"
echo "All runs completed!"
echo "========================================"
echo "Results in: $RESULT_DIR/"
ls -la $RESULT_DIR/*.json 2>/dev/null
echo ""
echo "To analyze: python3 route_balance/exp/route_balance/analyze_smoketest.py --result-dir $RESULT_DIR"
