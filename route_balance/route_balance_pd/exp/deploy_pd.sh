#!/bin/bash
# Deploy P-D disaggregated vLLM instances + P-D scheduler.
#
# Starts:
#   node0 (PREFILL_HOST): vLLM kv_producer on PREFILL_PORT
#   node1 (DECODE_HOST):  vLLM kv_consumer on DECODE_PORT
#   node0: P-D scheduler on SCHEDULER_PORT
#
# Usage:
#   bash route_balance/route_balance_pd/exp/deploy_pd.sh
#
# Override defaults via env vars:
#   PREFILL_HOST=anon@node0 DECODE_HOST=anon@node1 bash route_balance/route_balance_pd/exp/deploy_pd.sh
set -euo pipefail

# --- Config (override via env) ---
PREFILL_HOST=${PREFILL_HOST:-"anon@d7525-10s10317.cluster.example"}
DECODE_HOST=${DECODE_HOST:-"anon@d7525-10s10319.cluster.example"}
PREFILL_IP=${PREFILL_IP:-"10.10.1.1"}
DECODE_IP=${DECODE_IP:-"10.10.1.2"}
PREFILL_PORT=${PREFILL_PORT:-7100}
DECODE_PORT=${DECODE_PORT:-7200}
SCHEDULER_PORT=${SCHEDULER_PORT:-8200}
MODEL=${MODEL:-"Qwen/Qwen2.5-7B"}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.85}
# HF_HOME: node0 uses /mydata/hf_cache, node1 uses default (~/.cache/huggingface)
PREFILL_HF_HOME=${PREFILL_HF_HOME:-"/mydata/hf_cache"}
DECODE_HF_HOME=${DECODE_HF_HOME:-""}
PD_CONFIG=${PD_CONFIG:-"route_balance/route_balance_pd/config/pd_config_smoketest.json"}
NIXL_PORT=${NIXL_PORT:-14579}

SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10"

echo "========================================================"
echo "P-D Deployment — $(date)"
echo "Prefill: $PREFILL_HOST ($PREFILL_IP:$PREFILL_PORT) kv_producer"
echo "Decode:  $DECODE_HOST ($DECODE_IP:$DECODE_PORT) kv_consumer"
echo "Model:   $MODEL  max_model_len=$MAX_MODEL_LEN"
echo "========================================================"

# --- Step 1: Kill existing processes on both nodes ---
echo ""
echo "--- Step 1: Cleanup ---"
for HOST in "$PREFILL_HOST" "$DECODE_HOST"; do
    echo "  Cleaning $HOST..."
    ssh $SSH_OPTS "$HOST" "
        pkill -f 'python.*vllm\.entrypoints' || true
        pkill -f 'python.*block\.route_balance_pd\.route_balance_pd_serve' || true
        sleep 2
        pgrep -u \$(whoami) -f 'vllm|route_balance_pd' -a 2>/dev/null && echo '  WARNING: processes still running' || echo '  Clean'
    " 2>/dev/null || echo "  Warning: cleanup on $HOST had issues"
done

# --- Step 2: Verify GPUs are clean ---
echo ""
echo "--- Step 2: GPU check ---"
for HOST in "$PREFILL_HOST" "$DECODE_HOST"; do
    echo "  $HOST:"
    ssh $SSH_OPTS "$HOST" "nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader" 2>/dev/null
done

# --- Step 3: Start vLLM prefiller (kv_producer) on node0 ---
echo ""
echo "--- Step 3: Start prefiller on $PREFILL_HOST ---"
PREFILL_HF_CMD=""
if [ -n "$PREFILL_HF_HOME" ]; then
    PREFILL_HF_CMD="export HF_HOME=$PREFILL_HF_HOME &&"
fi

ssh $SSH_OPTS "$PREFILL_HOST" "
    cd ~/vllm &&
    export LD_LIBRARY_PATH=\$(python3 -c 'import nvidia, os, glob; print(\":\".join(glob.glob(os.path.join(nvidia.__path__[0], \"*\", \"lib\"))))' 2>/dev/null):\$LD_LIBRARY_PATH &&
    $PREFILL_HF_CMD
    nohup python3 -u -m vllm.entrypoints.openai.api_server \
        --model $MODEL \
        --port $PREFILL_PORT \
        --max-model-len $MAX_MODEL_LEN \
        --gpu-memory-utilization $GPU_MEM_UTIL \
        --kv-transfer-config '{\"kv_connector\": \"NixlConnector\", \"kv_role\": \"kv_producer\", \"kv_ip\": \"$PREFILL_IP\", \"kv_port\": $NIXL_PORT}' \
        > /tmp/vllm_prefiller.log 2>&1 &
    echo \$! > /tmp/vllm_prefiller.pid
    echo \"Prefiller PID: \$(cat /tmp/vllm_prefiller.pid)\"
" 2>/dev/null
echo "  Prefiller starting..."

# --- Step 4: Start vLLM decoder (kv_consumer) on node1 ---
echo ""
echo "--- Step 4: Start decoder on $DECODE_HOST ---"
DECODE_HF_CMD=""
if [ -n "$DECODE_HF_HOME" ]; then
    DECODE_HF_CMD="export HF_HOME=$DECODE_HF_HOME &&"
fi

ssh $SSH_OPTS "$DECODE_HOST" "
    cd ~/vllm &&
    export LD_LIBRARY_PATH=\$(python3 -c 'import nvidia, os, glob; print(\":\".join(glob.glob(os.path.join(nvidia.__path__[0], \"*\", \"lib\"))))' 2>/dev/null):\$LD_LIBRARY_PATH &&
    $DECODE_HF_CMD
    nohup python3 -u -m vllm.entrypoints.openai.api_server \
        --model $MODEL \
        --port $DECODE_PORT \
        --max-model-len $MAX_MODEL_LEN \
        --gpu-memory-utilization $GPU_MEM_UTIL \
        --kv-transfer-config '{\"kv_connector\": \"NixlConnector\", \"kv_role\": \"kv_consumer\", \"kv_ip\": \"$DECODE_IP\", \"kv_port\": $NIXL_PORT}' \
        > /tmp/vllm_decoder.log 2>&1 &
    echo \$! > /tmp/vllm_decoder.pid
    echo \"Decoder PID: \$(cat /tmp/vllm_decoder.pid)\"
" 2>/dev/null
echo "  Decoder starting..."

# --- Step 5: Wait for vLLM health checks ---
echo ""
echo "--- Step 5: Waiting for vLLM instances (up to 180s) ---"

wait_for_health() {
    local HOST=$1 IP=$2 PORT=$3 LABEL=$4
    for i in $(seq 1 36); do
        if ssh $SSH_OPTS "$HOST" "curl -sf http://$IP:$PORT/health > /dev/null 2>&1" 2>/dev/null; then
            echo "  $LABEL ready! (attempt $i)"
            return 0
        fi
        echo "  $LABEL attempt $i/36..."
        sleep 5
    done
    echo "  FAIL: $LABEL did not start"
    ssh $SSH_OPTS "$HOST" "tail -20 /tmp/vllm_*.log" 2>/dev/null
    return 1
}

# Wait in parallel
wait_for_health "$PREFILL_HOST" "$PREFILL_IP" "$PREFILL_PORT" "Prefiller" &
PID_W1=$!
wait_for_health "$DECODE_HOST" "$DECODE_IP" "$DECODE_PORT" "Decoder" &
PID_W2=$!

FAIL=0
wait $PID_W1 || FAIL=1
wait $PID_W2 || FAIL=1

if [ $FAIL -ne 0 ]; then
    echo "DEPLOYMENT FAILED — vLLM instances did not start"
    exit 1
fi

# --- Step 6: Start P-D scheduler on node0 ---
echo ""
echo "--- Step 6: Start P-D scheduler on $PREFILL_HOST ---"
ssh $SSH_OPTS "$PREFILL_HOST" "
    cd ~/RouteBalance &&
    export PYTHONPATH=~/RouteBalance:~/vllm:\$PYTHONPATH &&
    nohup python3 -u -m route_balance.route_balance_pd.route_balance_pd_serve \
        --pd-config $PD_CONFIG \
        --port $SCHEDULER_PORT \
        > /tmp/route_balance_pd_scheduler.log 2>&1 &
    echo \$! > /tmp/route_balance_pd_scheduler.pid
    echo \"Scheduler PID: \$(cat /tmp/route_balance_pd_scheduler.pid)\"
" 2>/dev/null

# Wait for scheduler health
echo "  Waiting for scheduler..."
for i in $(seq 1 12); do
    if ssh $SSH_OPTS "$PREFILL_HOST" "curl -sf http://localhost:$SCHEDULER_PORT/health" 2>/dev/null | grep -q "ok"; then
        echo "  Scheduler ready!"
        break
    fi
    sleep 5
done

# --- Step 7: Verify ---
echo ""
echo "--- Step 7: Verification ---"
echo "  Prefiller:"
ssh $SSH_OPTS "$PREFILL_HOST" "curl -sf http://$PREFILL_IP:$PREFILL_PORT/health 2>/dev/null && echo ' OK' || echo ' FAIL'" 2>/dev/null
echo "  Decoder:"
ssh $SSH_OPTS "$DECODE_HOST" "curl -sf http://$DECODE_IP:$DECODE_PORT/health 2>/dev/null && echo ' OK' || echo ' FAIL'" 2>/dev/null
echo "  Scheduler:"
ssh $SSH_OPTS "$PREFILL_HOST" "curl -sf http://localhost:$SCHEDULER_PORT/health 2>/dev/null || echo 'FAIL'" 2>/dev/null

# Quick smoke test: single request through the P-D scheduler
echo ""
echo "--- Smoke test: single request ---"
ssh $SSH_OPTS "$PREFILL_HOST" "
    curl -sf -X POST http://localhost:$SCHEDULER_PORT/v1/completions \
        -H 'Content-Type: application/json' \
        -d '{\"prompt\": \"Hello world\", \"max_tokens\": 16, \"model\": \"$MODEL\", \"prompt_len\": 4}' \
        2>/dev/null | python3 -m json.tool 2>/dev/null | head -20
" 2>/dev/null

echo ""
echo "========================================================"
echo "P-D DEPLOYMENT COMPLETE — $(date)"
echo "Scheduler: $PREFILL_HOST:$SCHEDULER_PORT"
echo "Run benchmark: bash route_balance/route_balance_pd/exp/benchmark_pd.sh"
echo "========================================================"
