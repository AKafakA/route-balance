#!/bin/bash
# ROUTE_BALANCE Smoke Test — Fused vs PFS Comparison
# Tests the fused model estimator vs PFS baseline on 2xA30 CloudLab
#
# Prerequisites:
#   - vLLM instances running on both nodes (7B on node0, 3B on node1)
#   - Test data at data/route_balance/best-route-v3-test-500.jsonl
#
# Usage: bash run_smoketest_compare.sh

set -e

# Node0 = scheduler + 7B, Node1 = 3B
SCHEDULER_IP=$(hostname -I | awk '{print $1}')
SCHEDULER_PORT=8200
NUM_REQUESTS=50
QPS_LEVELS="2 5"
RESULT_DIR="experiment_output/smoketest_fused_compare"
DATASET_PATH="data/route_balance/best-route-v3-test-500.jsonl"

# Model estimator configs to compare
declare -A CONFIGS
CONFIGS[route_balance_fused]="route_balance/config/route_balance/predictor_config_smoketest_fused.json"
CONFIGS[route_balance_pfs]="route_balance/config/route_balance/predictor_config_smoketest_pfs.json"

# Also test baselines (no model estimator needed)
BASELINE_SCHEDULERS="random round_robin shortest_queue"

echo "========================================"
echo "ROUTE_BALANCE Smoke Test — Fused vs PFS"
echo "========================================"
echo "Scheduler: $(hostname)"
echo "QPS levels: $QPS_LEVELS"
echo "Requests per run: $NUM_REQUESTS"
echo ""

cd ~/RouteBalance
export PYTHONPATH=~/RouteBalance:~/vllm:$PYTHONPATH
mkdir -p $RESULT_DIR experiment_output/logs

# 1. Test baseline schedulers (no model estimator)
for SCHED in $BASELINE_SCHEDULERS; do
  for QPS in $QPS_LEVELS; do
    RUN_NAME="${SCHED}_qps${QPS}"
    echo ""
    echo "=== $RUN_NAME ==="

    pkill -f 'route_balance_serve' 2>/dev/null || true
    sleep 2

    nohup python3 -u -m route_balance.global_scheduler.route_balance.route_balance_serve \
      --host 0.0.0.0 --port $SCHEDULER_PORT \
      --model_config_path route_balance/config/route_balance/model_deployment_smoketest.json \
      --host_config route_balance/config/host_configs.json \
      --scheduling $SCHED --chat \
      > experiment_output/logs/route_balance_serve_${RUN_NAME}.log 2>&1 &

    sleep 8
    if ! pgrep -f 'route_balance_serve' > /dev/null; then
      echo "ERROR: Scheduler failed to start"
      tail -10 experiment_output/logs/route_balance_serve_${RUN_NAME}.log
      continue
    fi

    python3 route_balance/benchmark/route_balance/benchmark_serving.py \
      --backend route_balance --host 127.0.0.1 --port $SCHEDULER_PORT \
      --dataset-name custom --dataset-path $DATASET_PATH \
      --num-prompts $NUM_REQUESTS --request-rate $QPS \
      --trust-remote-code --save-result --save-detailed \
      --result-dir $RESULT_DIR --result-filename "${RUN_NAME}.json" \
      --metadata scheduler=$SCHED qps=$QPS \
      2>&1 | tee $RESULT_DIR/${RUN_NAME}_console.log

    echo "Completed: $RUN_NAME"
  done
done

# 2. Test ROUTE_BALANCE with different model estimators
for CONFIG_NAME in route_balance_fused route_balance_pfs; do
  CONFIG_PATH=${CONFIGS[$CONFIG_NAME]}
  for QPS in $QPS_LEVELS; do
    RUN_NAME="${CONFIG_NAME}_qps${QPS}"
    echo ""
    echo "=== $RUN_NAME (config: $CONFIG_PATH) ==="

    pkill -f 'route_balance_serve' 2>/dev/null || true
    sleep 2

    nohup python3 -u -m route_balance.global_scheduler.route_balance.route_balance_serve \
      --host 0.0.0.0 --port $SCHEDULER_PORT \
      --model_config_path route_balance/config/route_balance/model_deployment_smoketest.json \
      --host_config route_balance/config/host_configs.json \
      --scheduling route_balance --chat \
      --predictor-config $CONFIG_PATH \
      > experiment_output/logs/route_balance_serve_${RUN_NAME}.log 2>&1 &

    sleep 15  # fused model needs extra time to load

    if ! pgrep -f 'route_balance_serve' > /dev/null; then
      echo "ERROR: Scheduler failed to start"
      tail -20 experiment_output/logs/route_balance_serve_${RUN_NAME}.log
      continue
    fi

    # Health check
    curl -s http://localhost:$SCHEDULER_PORT/v1/batch_stats > /dev/null 2>&1 || {
      echo "WARNING: Scheduler not fully ready, waiting 10 more seconds..."
      sleep 10
    }

    python3 route_balance/benchmark/route_balance/benchmark_serving.py \
      --backend route_balance --host 127.0.0.1 --port $SCHEDULER_PORT \
      --dataset-name custom --dataset-path $DATASET_PATH \
      --num-prompts $NUM_REQUESTS --request-rate $QPS \
      --trust-remote-code --save-result --save-detailed \
      --result-dir $RESULT_DIR --result-filename "${RUN_NAME}.json" \
      --metadata scheduler=route_balance estimator=$CONFIG_NAME qps=$QPS \
      2>&1 | tee $RESULT_DIR/${RUN_NAME}_console.log

    echo "Batch stats:"
    curl -s http://localhost:$SCHEDULER_PORT/v1/batch_stats 2>/dev/null
    echo ""
    echo "Completed: $RUN_NAME"
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
