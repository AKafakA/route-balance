#!/bin/bash

# ROUTE_BALANCE End-to-End Experiment Script
# Complete workflow: deploy backends -> verify -> start ROUTE_BALANCE -> run tests

set -e  # Exit on error

# Configuration
TARGET_HOST="asdwb@d8545-10s10301.wisc.cloudlab.us"
HOST_CONFIG_PATH="route_balance/config/host_configs.json"
MODEL_CONFIG_PATH="route_balance/config/route_balance/model_config_template.json"
DEPLOYMENT_CONFIG="route_balance/config/route_balance/model_deployment.json"
HOSTS_FILE="route_balance/config/hosts"
ROUTE_BALANCE_PORT=8200
HF_TOKEN="${HF_TOKEN:-}"  # Set HF_TOKEN environment variable before running

echo "========================================"
echo "ROUTE_BALANCE End-to-End Experiment"
echo "========================================"
echo "Target ROUTE_BALANCE Host: ${TARGET_HOST}"
echo "ROUTE_BALANCE Port: ${ROUTE_BALANCE_PORT}"
echo ""

# Step 1: Deploy backend instances
echo "Step 1: Deploying Backend Instances"
echo "------------------------------------"
python route_balance/exp/route_balance/deploy_route_balance.py \
  --hosts ${HOSTS_FILE} \
  --config ${MODEL_CONFIG_PATH} \
  --hf-token "${HF_TOKEN}" \
  --output ${DEPLOYMENT_CONFIG}

if [ $? -ne 0 ]; then
    echo "❌ Backend deployment failed!"
    exit 1
fi
echo "✅ Backend deployment completed"
echo ""

# Step 2: Wait for backends to initialize
echo "Step 2: Waiting for Backends to Initialize"
echo "-------------------------------------------"
echo "Waiting 60 seconds for models to load..."
sleep 60
echo ""

# Step 3: Verify backend deployments
echo "Step 3: Verifying Backend Deployments"
echo "--------------------------------------"
python route_balance/exp/route_balance/e2e/check_deployment.py \
  --config ${DEPLOYMENT_CONFIG} \
  --output route_balance/config/route_balance/verified_hosts.json

if [ $? -ne 0 ]; then
    echo "⚠️  Warning: Some backends may not be fully operational"
    echo "Continuing with available backends..."
fi
echo "✅ Backend verification completed"
echo ""

# Step 4: Start ROUTE_BALANCE scheduler server on target host
echo "Step 4: Starting ROUTE_BALANCE Scheduler Server"
echo "---------------------------------------"

# Kill existing ROUTE_BALANCE server if running
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  ${TARGET_HOST} \
  "pkill -f 'route_balance_serve.py' || echo 'No existing ROUTE_BALANCE server found'"

sleep 2

# Create log directory on target host
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  ${TARGET_HOST} \
  "mkdir -p Block/experiment_output/logs"

# Start ROUTE_BALANCE server in background
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  ${TARGET_HOST} \
  "cd Block && nohup python -m route_balance.global_scheduler.route_balance.route_balance_serve \
    --host 0.0.0.0 \
    --port ${ROUTE_BALANCE_PORT} \
    --model_config_path ${DEPLOYMENT_CONFIG} \
    --host_config ${HOST_CONFIG_PATH} \
    --scheduling random \
    > experiment_output/logs/route_balance_server.log 2>&1 &"

echo "Waiting 10 seconds for ROUTE_BALANCE server to start..."
sleep 10

# Extract IP from target host
ROUTE_BALANCE_HOST_IP=$(echo ${TARGET_HOST} | cut -d'@' -f2)

echo "✅ ROUTE_BALANCE server started at http://${ROUTE_BALANCE_HOST_IP}:${ROUTE_BALANCE_PORT}"
echo ""

# Step 5: Run simple benchmark test
echo "Step 5: Running Benchmark Test"
echo "-------------------------------"

# Update test.sh with correct ROUTE_BALANCE host and run it on target host
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  ${TARGET_HOST} \
  "cd Block && \
   sed -i 's/ROUTE_BALANCE_HOST=\"127.0.0.1\"/ROUTE_BALANCE_HOST=\"127.0.0.1\"/' route_balance/exp/route_balance/test.sh && \
   bash route_balance/exp/route_balance/test.sh"

echo ""
echo "========================================"
echo "ROUTE_BALANCE Experiment Complete!"
echo "========================================"
echo ""
echo "Summary:"
echo "  ✅ Backends deployed and verified"
echo "  ✅ ROUTE_BALANCE server running at http://${ROUTE_BALANCE_HOST_IP}:${ROUTE_BALANCE_PORT}"
echo "  ✅ Benchmark test completed"
echo ""
echo "View results:"
echo "  ssh ${TARGET_HOST} 'cat Block/experiment_output/route_balance_test_results/route_balance_simple_test.json | jq .'"
echo ""
echo "Check ROUTE_BALANCE server logs:"
echo "  ssh ${TARGET_HOST} 'tail -f Block/experiment_output/logs/route_balance_server.log'"
echo ""
