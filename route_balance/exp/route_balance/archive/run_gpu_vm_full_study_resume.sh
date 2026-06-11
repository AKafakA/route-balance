#!/bin/bash
# Resume full study from RoBERTa onwards (experiments 4-10)
# Skips failures and continues to next experiment
cd ~/Code/llm/Block
export PYTHONPATH=.
PYTHON=.venv/bin/python
LATENCY_DIR=data/route_balance/latency_data/tagged/
TRAIN=data/route_balance/training_data/train_fixed.jsonl
TEST=data/route_balance/training_data/test_fixed.jsonl

check_disk() {
    local avail_gb=$(df -BG /home/wd312 --output=avail | tail -1 | tr -d ' G')
    if [ "$avail_gb" -lt 50 ]; then
        echo "[$(date)] DISK LOW: ${avail_gb}GB free. STOPPING."
        exit 1
    fi
    echo "[$(date)] Disk: ${avail_gb}GB free"
}

# Clean up incomplete roberta from previous run
rm -rf models/route_balance/study/roberta_fused_length 2>/dev/null

check_disk
echo "=========================================="
echo "4. RoBERTa fused × 3 targets (5 ep)"
echo "=========================================="
for TARGET in length similarity judge_class; do
    OUTDIR=models/route_balance/study/roberta_fused_${TARGET}
    echo "[$(date)] RoBERTa fused ${TARGET}..."
    $PYTHON -m route_balance.predictor.route_balance.offline_training.train_bert_predictor \
        --input $TRAIN --test-input $TEST \
        --regression-model-name roberta-base \
        --target $TARGET --lr 1e-5 --epochs 5 --device cuda \
        --output-dir $OUTDIR \
        || echo "[$(date)] RoBERTa fused ${TARGET} FAILED — skipping"
done

check_disk
echo "=========================================="
echo "5. Qwen-0.5B LoRA × 3 targets (3 ep)"
echo "=========================================="
for TARGET in length similarity judge; do
    OUTDIR=models/route_balance/study/qwen05b_fused_${TARGET}
    echo "[$(date)] Qwen-0.5B LoRA fused ${TARGET}..."
    $PYTHON -m route_balance.predictor.route_balance.offline_training.train_llm_predictor \
        --input $TRAIN --test-input $TEST \
        --target $TARGET --epochs 3 \
        --output-dir $OUTDIR \
        || echo "[$(date)] Qwen-0.5B LoRA ${TARGET} FAILED — skipping"
done

check_disk
echo "=========================================="
echo "6. KNN (~10 min)"
echo "=========================================="
echo "[$(date)] KNN..."
$PYTHON -m route_balance.predictor.route_balance.offline_training.train_knn_estimator \
    --input $TRAIN --test-input $TEST \
    --output-dir models/route_balance/study/knn/ --device cuda \
    || echo "[$(date)] KNN FAILED — skipping"

check_disk
echo "=========================================="
echo "7. MLP (~20 min)"
echo "=========================================="
echo "[$(date)] MLP..."
$PYTHON -m route_balance.predictor.route_balance.offline_training.train_mlp_predictor \
    --input $TRAIN --test-input $TEST \
    --output-dir models/route_balance/study/mlp/ --device cuda \
    || echo "[$(date)] MLP FAILED — skipping"

check_disk
echo "=========================================="
echo "8. LSTM latency"
echo "=========================================="
echo "[$(date)] LSTM latency..."
$PYTHON -m route_balance.predictor.route_balance.offline_training.train_lstm_latency \
    --data-dir $LATENCY_DIR --output-dir models/route_balance/study/lstm_latency/ \
    || echo "[$(date)] LSTM latency FAILED — skipping"

check_disk
echo "=========================================="
echo "9. LSTM quality"
echo "=========================================="
echo "[$(date)] LSTM quality..."
$PYTHON -m route_balance.predictor.route_balance.offline_training.train_lstm_predictor \
    --input $TRAIN --test-input $TEST \
    --output-dir models/route_balance/study/lstm_quality/ \
    || echo "[$(date)] LSTM quality FAILED — skipping"

check_disk
echo "=========================================="
echo "10. Bucket filtering evaluation"
echo "=========================================="
BUCKET_MODEL=models/route_balance/early_study/modernbert_fused_length_bucket_with_squad
if [ ! -d "$BUCKET_MODEL" ]; then
    BUCKET_MODEL=models/route_balance/study/modernbert_fused_length_bucket
fi
echo "[$(date)] Bucket filtering simulation..."
$PYTHON -m route_balance.predictor.route_balance.offline_training.evaluate_bucket_filtering \
    --test-input $TEST \
    --model-dir $BUCKET_MODEL \
    --device cuda \
    --output models/route_balance/study/bucket_filtering_results.json \
    || echo "[$(date)] Bucket filtering eval FAILED — skipping"

echo "=========================================="
echo "[$(date)] RESUME STUDY DONE"
echo "=========================================="
