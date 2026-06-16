#!/bin/bash
# Benchmark P-D disaggregated serving at multiple QPS levels.
# Run AFTER deploy_pd.sh has set up the cluster.
#
# Uses the standard RouteBalance benchmark client (--backend route_balance) pointing at
# the P-D scheduler, which returns benchmark-compatible response format.
#
# Usage:
#   bash route_balance/route_balance_pd/exp/benchmark_pd.sh
#   bash route_balance/route_balance_pd/exp/benchmark_pd.sh experiment_output/pd 50 "1 3 5"
#
# Override via env vars:
#   SCHEDULER_HOST=${CLOUDLAB_USER}@node0 NUM_REQ=100 bash route_balance/route_balance_pd/exp/benchmark_pd.sh
set -euo pipefail

RESULT_DIR=${1:-"experiment_output/pd_benchmark"}
NUM_REQ=${2:-50}
QPS_LEVELS=${3:-"1 3 5"}
DATASET=${DATASET:-"data/route_balance/best-route-v3-test-500.jsonl"}

# Where the P-D scheduler is running (SSH host for running benchmark)
SCHEDULER_HOST=${SCHEDULER_HOST:-"${CLOUDLAB_HOST}"}
SCHEDULER_PORT=${SCHEDULER_PORT:-8200}

SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10"

echo "========================================================"
echo "P-D Benchmark — $(date)"
echo "Scheduler: $SCHEDULER_HOST:$SCHEDULER_PORT"
echo "QPS levels: $QPS_LEVELS"
echo "Requests per run: $NUM_REQ"
echo "Results: $RESULT_DIR"
echo "========================================================"

# Verify scheduler is up
echo ""
echo "--- Checking scheduler health ---"
HEALTH=$(ssh $SSH_OPTS "$SCHEDULER_HOST" "curl -sf http://localhost:$SCHEDULER_PORT/health 2>/dev/null" 2>/dev/null || echo "FAIL")
echo "  $HEALTH"
if echo "$HEALTH" | grep -q "FAIL"; then
    echo "ERROR: P-D scheduler not running. Run deploy_pd.sh first."
    exit 1
fi

# Run benchmark on the scheduler host
echo ""
echo "--- Running benchmarks ---"
PASS=0; FAIL=0

for QPS in $QPS_LEVELS; do
    NAME="pd_qps${QPS}"
    echo "  [$NAME] $NUM_REQ requests at QPS=$QPS..."

    if ssh $SSH_OPTS "$SCHEDULER_HOST" "
        cd ~/Block &&
        export PYTHONPATH=~/Block:~/vllm:\$PYTHONPATH &&
        mkdir -p $RESULT_DIR &&
        python3 route_balance/benchmark/route_balance/benchmark_serving.py \
            --backend route_balance \
            --host 127.0.0.1 --port $SCHEDULER_PORT \
            --dataset-name custom --dataset-path $DATASET \
            --num-prompts $NUM_REQ \
            --request-rate $QPS \
            --trust-remote-code \
            --save-result --save-detailed \
            --result-dir $RESULT_DIR \
            --result-filename ${NAME}.json \
            2>&1 | tail -5
    " 2>/dev/null; then
        echo "  [$NAME] PASS"
        PASS=$((PASS+1))
    else
        echo "  [$NAME] FAIL"
        FAIL=$((FAIL+1))
    fi
done

# Collect P-D stats
echo ""
echo "--- Collecting P-D stats ---"
ssh $SSH_OPTS "$SCHEDULER_HOST" "
    curl -sf http://localhost:$SCHEDULER_PORT/v1/pd_stats 2>/dev/null | python3 -m json.tool
    mkdir -p $RESULT_DIR
    curl -sf http://localhost:$SCHEDULER_PORT/v1/pd_stats > ~/$RESULT_DIR/pd_stats.json 2>/dev/null
" 2>/dev/null

# Fetch results back to local machine
echo ""
echo "--- Fetching results ---"
mkdir -p "$RESULT_DIR"
scp $SSH_OPTS "$SCHEDULER_HOST:~/$RESULT_DIR/*.json" "$RESULT_DIR/" 2>/dev/null || echo "  Warning: scp failed (results may still be on remote)"

echo ""
echo "========================================================"
echo "P-D BENCHMARK COMPLETE — $(date)"
echo "PASS: $PASS  FAIL: $FAIL"
echo "Results: $RESULT_DIR/"
ls -la "$RESULT_DIR/"*.json 2>/dev/null || echo "  (results on remote: $SCHEDULER_HOST:~/$RESULT_DIR/)"
echo "========================================================"
