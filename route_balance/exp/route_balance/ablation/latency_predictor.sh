#!/bin/bash
# Latency Predictor Ablation: xgboost / roofline / static_tpot
# Requires scheduler restart per predictor type (different predictor_config)
set -euo pipefail
RESULT_DIR=${1:-"experiment_output/ablation/latency_predictor"}
NUM_REQ=${2:-2000}
QPS_LEVELS=${3:-"5 10 15"}
PORT=8200
DATASET="data/route_balance/best-route-v3-test-500.jsonl"
MODEL_DEPLOY=${MODEL_DEPLOY:-"route_balance/config/route_balance/model_deployment.json"}
SCHEDULER_CONFIG=${SCHEDULER_CONFIG:-"route_balance/config/route_balance/scheduler_config.json"}

cd ~/RouteBalance; export PYTHONPATH=~/RouteBalance:~/vllm:$PYTHONPATH
mkdir -p $RESULT_DIR experiment_output/logs

run() { local N=$1; shift; python3 route_balance/benchmark/route_balance/benchmark_serving.py --backend route_balance --host 127.0.0.1 --port $PORT --dataset-name custom --dataset-path $DATASET --num-prompts $NUM_REQ --trust-remote-code --save-result --save-detailed --result-dir $RESULT_DIR --result-filename "${N}.json" "$@" 2>&1 | tail -1; }

deploy_with_predictor() {
    local NAME=$1 PRED_CONFIG=$2
    if [ -f /tmp/scheduler_pid ]; then
        kill $(cat /tmp/scheduler_pid) 2>/dev/null || true
        for i in $(seq 1 10); do curl -sf http://localhost:$PORT/health > /dev/null 2>&1 || break; sleep 1; done
    fi
    sleep 1

    local CONFIG_ARGS="--scheduler-config $SCHEDULER_CONFIG --predictor-config $PRED_CONFIG"
    nohup python3 -u -m route_balance.global_scheduler.route_balance.route_balance_serve \
      --host 0.0.0.0 --port $PORT \
      --model_config_path $MODEL_DEPLOY \
      --scheduling route_balance --chat $CONFIG_ARGS \
      > experiment_output/logs/route_balance_latpred_${NAME}.log 2>&1 &
    echo $! > /tmp/scheduler_pid
    sleep 8
    for i in $(seq 1 12); do curl -sf http://localhost:$PORT/health > /dev/null 2>&1 && break; sleep 5; done
    curl -sf http://localhost:$PORT/health > /dev/null 2>&1 || { echo "FAIL: $NAME"; return 1; }
    echo "  Deployed: $NAME"
}

echo "Latency Predictor Ablation — $(date)"
for entry in \
    "xgboost:route_balance/config/route_balance/predictor_config_smoketest_fused.json" \
    "roofline:route_balance/config/route_balance/predictor_config_smoketest_roofline.json" \
    "static_tpot:route_balance/config/route_balance/predictor_config_smoketest_static_tpot.json" \
; do
    NAME="${entry%%:*}"; CONFIG="${entry#*:}"
    [ ! -f "$CONFIG" ] && { echo "  SKIP: $CONFIG not found"; continue; }
    deploy_with_predictor $NAME $CONFIG || continue
    for QPS in $QPS_LEVELS; do run "latpred_${NAME}_qps${QPS}" --request-rate $QPS; done
done

test -f /tmp/scheduler_pid && kill $(cat /tmp/scheduler_pid) 2>/dev/null || true
echo "Latency Predictor COMPLETE"
