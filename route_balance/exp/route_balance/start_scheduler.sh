#!/bin/bash
# Start ROUTE_BALANCE scheduler with batch scheduling
# Usage: ./start_scheduler.sh [scheduling_strategy]
STRATEGY=${1:-route_balance}

cd ~/Block
export PYTHONPATH=~/Block:$PYTHONPATH

pkill -f 'route_balance_serve' 2>/dev/null
sleep 1

mkdir -p experiment_output/logs

echo "Starting ROUTE_BALANCE scheduler (strategy=$STRATEGY)..."
nohup python3 -u -m route_balance.global_scheduler.route_balance.route_balance_serve \
  --host 0.0.0.0 \
  --port 8200 \
  --model_config_path route_balance/config/route_balance/model_deployment_smoketest.json \
  --host_config route_balance/config/host_configs.json \
  --scheduling "$STRATEGY" \
  --predictor-config route_balance/config/route_balance/predictor_config_smoketest.json \
  --enable-predictor-feedback \
  --feedback-sample-rate 0.0 \
  --chat \
  --debugging_logs \
  > experiment_output/logs/route_balance_scheduler.log 2>&1 &

sleep 5
if pgrep -f 'route_balance_serve' > /dev/null; then
  echo "Scheduler started OK (pid=$(pgrep -f 'route_balance_serve'))"
  echo "Logs: experiment_output/logs/route_balance_scheduler.log"
  tail -5 experiment_output/logs/route_balance_scheduler.log
else
  echo "Scheduler FAILED to start"
  tail -30 experiment_output/logs/route_balance_scheduler.log
  exit 1
fi
