#!/bin/bash
# Validate ALL experiment scripts with minimal requests on 2-node CloudLab
# Deploys route_balance scheduler once, runs each script with 10 req at QPS=5
set -euo pipefail

PORT=8200
NUM_REQ=10
QPS="5"
MODEL_DEPLOY=${MODEL_DEPLOY:-"route_balance/config/route_balance/model_deployment_smoketest_v2.json"}
SCHEDULER_CONFIG=${SCHEDULER_CONFIG:-"route_balance/config/route_balance/scheduler_config_smoketest.json"}
export MODEL_DEPLOY SCHEDULER_CONFIG

cd ~/Block
export PYTHONPATH=~/Block:~/vllm:$PYTHONPATH
mkdir -p experiment_output/logs

PASS=0; FAIL=0; SKIP=0

check() {
    local NAME=$1 DIR=$2
    local COUNT=$(ls $DIR/*.json 2>/dev/null | wc -l)
    if [ "$COUNT" -gt 0 ]; then
        echo "  ✓ $NAME: $COUNT results"; PASS=$((PASS+1))
    else
        echo "  ✗ $NAME: NO results"; FAIL=$((FAIL+1))
    fi
}

# Start route_balance scheduler for runtime-config experiments
if [ -f /tmp/scheduler_pid ]; then kill $(cat /tmp/scheduler_pid) 2>/dev/null || true; sleep 2; fi
nohup python3 -u -m route_balance.global_scheduler.route_balance.route_balance_serve \
  --host 0.0.0.0 --port $PORT \
  --model_config_path $MODEL_DEPLOY \
  --scheduling route_balance --chat \
  --scheduler-config $SCHEDULER_CONFIG \
  > experiment_output/logs/route_balance_validate.log 2>&1 &
echo $! > /tmp/scheduler_pid
sleep 8
curl -sf http://localhost:$PORT/health > /dev/null 2>&1 || { echo "Scheduler failed to start"; exit 1; }
echo "Scheduler running"

echo ""
echo "===== Runtime Config Scripts (no restart needed) ====="

echo "--- Weight Sweep ---"
bash route_balance/exp/route_balance/sensitivity/weight_sweep.sh experiment_output/validate/weight $NUM_REQ "$QPS" 2>&1 | tail -1
check "weight_sweep" "experiment_output/validate/weight"

echo "--- Batch Sweep ---"
bash route_balance/exp/route_balance/sensitivity/batch_sweep.sh experiment_output/validate/batch $NUM_REQ "$QPS" 2>&1 | tail -1
check "batch_sweep" "experiment_output/validate/batch"

echo "--- Budget Sweep ---"
bash route_balance/exp/route_balance/sensitivity/budget_sweep.sh experiment_output/validate/budget $NUM_REQ "$QPS" 2>&1 | tail -1
check "budget_sweep" "experiment_output/validate/budget"

echo "--- RSO Density ---"
bash route_balance/exp/route_balance/sensitivity/rso_density.sh experiment_output/validate/rso $NUM_REQ "$QPS" 2>&1 | tail -1
check "rso_density" "experiment_output/validate/rso"

echo "--- QPS Scaling (3 levels only) ---"
bash route_balance/exp/route_balance/sensitivity/qps_scaling.sh experiment_output/validate/qps $NUM_REQ 2>&1 | tail -1
check "qps_scaling" "experiment_output/validate/qps"

echo "--- Normalization ---"
bash route_balance/exp/route_balance/ablation/normalization.sh experiment_output/validate/norm $NUM_REQ "$QPS" 2>&1 | tail -1
check "normalization" "experiment_output/validate/norm"

echo "--- LPT Sort ---"
bash route_balance/exp/route_balance/ablation/lpt_sort.sh experiment_output/validate/lpt $NUM_REQ "$QPS" 2>&1 | tail -1
check "lpt_sort" "experiment_output/validate/lpt"

echo "--- Filter Order ---"
bash route_balance/exp/route_balance/ablation/filter_order.sh experiment_output/validate/forder $NUM_REQ "$QPS" 2>&1 | tail -1
check "filter_order" "experiment_output/validate/forder"

echo "--- Quality Threshold ---"
bash route_balance/exp/route_balance/ablation/quality_threshold.sh experiment_output/validate/qualmin $NUM_REQ "$QPS" 2>&1 | tail -1
check "quality_threshold" "experiment_output/validate/qualmin"

echo "--- Filter Impact ---"
bash route_balance/exp/route_balance/ablation/filter_impact.sh experiment_output/validate/fimpact $NUM_REQ "$QPS" 2>&1 | tail -1
check "filter_impact" "experiment_output/validate/fimpact"

# Kill scheduler before restart-requiring scripts
kill $(cat /tmp/scheduler_pid) 2>/dev/null || true; sleep 2

echo ""
echo "===== Restart-Required Scripts ====="

echo "--- E2E Baselines ---"
bash route_balance/exp/route_balance/e2e/run.sh experiment_output/validate/e2e $NUM_REQ "$QPS" 2>&1 | grep -E "PASS|FAIL|COMPLETE" | tail -5
check "e2e" "experiment_output/validate/e2e"

echo "--- Estimator Ablation ---"
bash route_balance/exp/route_balance/ablation/estimator.sh experiment_output/validate/estimator $NUM_REQ "$QPS" 2>&1 | grep -E "Deployed|SKIP|COMPLETE" | tail -5
check "estimator" "experiment_output/validate/estimator"

echo "--- Latency Predictor ---"
bash route_balance/exp/route_balance/ablation/latency_predictor.sh experiment_output/validate/latpred $NUM_REQ "$QPS" 2>&1 | grep -E "Deployed|SKIP|COMPLETE" | tail -5
check "latency_predictor" "experiment_output/validate/latpred"

# Cleanup
test -f /tmp/scheduler_pid && kill $(cat /tmp/scheduler_pid) 2>/dev/null || true

echo ""
echo "========================================================"
echo "VALIDATION COMPLETE — $(date)"
echo "PASS: $PASS  FAIL: $FAIL  SKIP: $SKIP"
TOTAL=$(find experiment_output/validate -name "*.json" 2>/dev/null | wc -l)
echo "Total result files: $TOTAL"
echo "========================================================"
