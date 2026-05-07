#!/bin/bash
# Start ROUTE_BALANCE predictor sidecar
# Usage: ./start_predictor.sh <hostname> <instance_type>
HOSTNAME=${1:-$(hostname -f)}
INSTANCE_TYPE=${2:-unknown}

cd ~/RouteBalance
export PYTHONPATH=~/RouteBalance:$PYTHONPATH

pkill -f 'route_balance_predictor_api_server' 2>/dev/null
sleep 1

mkdir -p experiment_output/logs

nohup python3 -u -m route_balance.predictor.route_balance.route_balance_predictor_api_server \
  --host 0.0.0.0 \
  --port 8300 \
  --backend-port 8000 \
  --hostname "$HOSTNAME" \
  --config-path route_balance/config/route_balance/predictor_config_smoketest.json \
  --instance-type "$INSTANCE_TYPE" \
  > experiment_output/logs/predictor_8300.log 2>&1 &

sleep 5
if pgrep -f 'route_balance_predictor_api_server.*8300' > /dev/null; then
  echo "Predictor started OK (pid=$(pgrep -f 'route_balance_predictor_api_server.*8300'))"
else
  echo "Predictor FAILED to start"
  tail -20 experiment_output/logs/predictor_8300.log
  exit 1
fi
