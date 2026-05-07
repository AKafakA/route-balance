#!/bin/bash
# Estimator Type Ablation: default (fused) / knn / pfs
# Requires scheduler restart per estimator type (different scheduler_config)
set -euo pipefail

RESULT_DIR=${1:-"experiment_output/ablation/estimator"}
NUM_REQ=${2:-2000}
QPS_LEVELS=${3:-"5 10 15"}
PORT=8200
DATASET="data/route_balance/best-route-v3-test-500.jsonl"
MODEL_DEPLOY=${MODEL_DEPLOY:-"route_balance/config/route_balance/model_deployment.json"}

cd ~/RouteBalance
export PYTHONPATH=~/RouteBalance:~/vllm:$PYTHONPATH
mkdir -p $RESULT_DIR experiment_output/logs

run() { local N=$1; shift; python3 route_balance/benchmark/route_balance/benchmark_serving.py --backend route_balance --host 127.0.0.1 --port $PORT --dataset-name custom --dataset-path $DATASET --num-prompts $NUM_REQ --trust-remote-code --save-result --save-detailed --result-dir $RESULT_DIR --result-filename "${N}.json" "$@" 2>&1 | tail -1; }

deploy_with_config() {
    local NAME=$1
    local SCHED_CONFIG=$2
    pkill -f '^python.*route_balance_serve' 2>/dev/null || true; sleep 2
    nohup python3 -u -m route_balance.global_scheduler.route_balance.route_balance_serve \
      --host 0.0.0.0 --port $PORT \
      --model_config_path $MODEL_DEPLOY \
      --scheduling route_balance --chat \
      --scheduler-config $SCHED_CONFIG \
      > experiment_output/logs/route_balance_estimator_${NAME}.log 2>&1 &
    sleep 8
    for i in $(seq 1 12); do
        curl -sf http://localhost:$PORT/v1/batch_stats > /dev/null 2>&1 && break; sleep 5
    done
    curl -sf http://localhost:$PORT/v1/batch_stats > /dev/null 2>&1 || { echo "FAIL: $NAME"; return 1; }
    echo "  Deployed: $NAME"
}

echo "Estimator Ablation — $(date)"

# Each estimator type needs a scheduler_config with different model_estimator.type
# Create these configs if they don't exist, or use pre-made ones
for TYPE in default knn pfs; do
    CONFIG="route_balance/config/route_balance/scheduler_config_${TYPE}.json"
    if [ ! -f "$CONFIG" ]; then
        echo "  WARNING: $CONFIG not found, skipping $TYPE"
        continue
    fi
    deploy_with_config $TYPE $CONFIG || continue
    for QPS in $QPS_LEVELS; do
        run "estimator_${TYPE}_qps${QPS}" --request-rate $QPS
    done
done

pkill -f '^python.*route_balance_serve' 2>/dev/null || true
echo "Estimator Ablation COMPLETE"
