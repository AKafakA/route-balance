#!/bin/bash
# GPU VM Long Training Study — 3 fused runs at 50 epochs
# Compares MSE vs log-transform vs bucket on learning curves
#
# Usage: bash route_balance/exp/route_balance/run_gpu_vm_long_training.sh 2>&1 | tee /tmp/long_training.log
cd ~/Code/llm/RouteBalance
export PYTHONPATH=.
PYTHON=.venv/bin/python
TRAIN=data/route_balance/training_data/train_fixed.jsonl
TEST=data/route_balance/training_data/test_fixed.jsonl

EPOCHS=100
PATIENCE=10

check_disk() {
    local avail_gb=$(df -BG /home/anon --output=avail | tail -1 | tr -d ' G')
    if [ "$avail_gb" -lt 50 ]; then
        echo "[$(date)] DISK LOW: ${avail_gb}GB free. STOPPING."
        exit 1
    fi
    echo "[$(date)] Disk: ${avail_gb}GB free"
}

check_disk
echo "=========================================="
echo "Long Training: 3 fused runs × ${EPOCHS} epochs, early_stopping=${PATIENCE}"
echo "=========================================="

echo ""
echo "=========================================="
echo "1. ModernBERT fused length MSE (${EPOCHS} ep)"
echo "=========================================="
echo "[$(date)]"
$PYTHON -m route_balance.predictor.route_balance.offline_training.train_bert_predictor \
    --input $TRAIN --test-input $TEST \
    --regression-model-name answerdotai/ModernBERT-base \
    --target length --lr 1e-5 --epochs $EPOCHS \
    --early-stopping-patience $PATIENCE \
    --loss-type mse --device cuda \
    --output-dir models/route_balance/long_study/modernbert_fused_length_mse \
    || echo "[$(date)] MSE length FAILED"

check_disk
echo ""
echo "=========================================="
echo "2. ModernBERT fused length log-transform (${EPOCHS} ep)"
echo "=========================================="
echo "[$(date)]"
$PYTHON -m route_balance.predictor.route_balance.offline_training.train_bert_predictor \
    --input $TRAIN --test-input $TEST \
    --regression-model-name answerdotai/ModernBERT-base \
    --target length --lr 1e-5 --epochs $EPOCHS \
    --early-stopping-patience $PATIENCE \
    --log-transform --device cuda \
    --output-dir models/route_balance/long_study/modernbert_fused_length_logtransform \
    || echo "[$(date)] Log-transform length FAILED"

check_disk
echo ""
echo "=========================================="
echo "3. ModernBERT fused length_bucket CE (${EPOCHS} ep)"
echo "=========================================="
echo "[$(date)]"
$PYTHON -m route_balance.predictor.route_balance.offline_training.train_bert_predictor \
    --input $TRAIN --test-input $TEST \
    --regression-model-name answerdotai/ModernBERT-base \
    --target length_bucket --lr 1e-5 --epochs $EPOCHS \
    --early-stopping-patience $PATIENCE \
    --device cuda \
    --output-dir models/route_balance/long_study/modernbert_fused_length_bucket \
    || echo "[$(date)] Bucket CE FAILED"

echo "=========================================="
echo "[$(date)] LONG TRAINING DONE"
echo "=========================================="
