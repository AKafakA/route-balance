#!/bin/bash
# GPU VM Full Study — runs experiments NOT covered by early study
# Assumes early study already completed: XGBoost E2E/TTFT/TPOT, Linear, Roofline,
# ModernBERT fused × 4 targets (with_squad)
#
# Usage: bash route_balance/exp/route_balance/run_gpu_vm_full_study.sh 2>&1 | tee /tmp/full_study.log
# Don't set -e: skip failures and continue to next experiment
cd ~/Code/llm/RouteBalance
export PYTHONPATH=.
PYTHON=.venv/bin/python
LATENCY_DIR=data/route_balance/latency_data/tagged/
TRAIN=data/route_balance/training_data/train_fixed.jsonl
TEST=data/route_balance/training_data/test_fixed.jsonl

# Disk safety: stop if less than 50GB free
check_disk() {
    local avail_gb=$(df -BG /home/anon --output=avail | tail -1 | tr -d ' G')
    if [ "$avail_gb" -lt 50 ]; then
        echo "[$(date)] DISK LOW: ${avail_gb}GB free. STOPPING to avoid filling disk."
        exit 1
    fi
    echo "[$(date)] Disk: ${avail_gb}GB free"
}

# Ensure tagged latency data with actual_output_tokens is linked
mkdir -p data/route_balance/latency_data/tagged
ln -sf $(pwd)/data/route_balance/latency_data/all/latency_train_tagged.jsonl data/route_balance/latency_data/tagged/latency_train.jsonl
ln -sf $(pwd)/data/route_balance/latency_data/all/latency_test_tagged.jsonl data/route_balance/latency_data/tagged/latency_test.jsonl

check_disk
echo "=========================================="
echo "1. XGBoost E2E with actual output tokens (~5 min)"
echo "=========================================="
echo "[$(date)] XGBoost E2E (actual output tokens)..."
$PYTHON -m route_balance.predictor.route_balance.offline_training.train_xgboost \
    --data-dir $LATENCY_DIR --output-dir models/route_balance/study/xgboost_e2e_actual/ \
    --target e2el --use-actual-output-len

check_disk
echo "=========================================="
echo "2. ModernBERT fused log-transform length (5 ep) (~30 min)"
echo "=========================================="
echo "[$(date)] ModernBERT fused length log-transform..."
$PYTHON -m route_balance.predictor.route_balance.offline_training.train_bert_predictor \
    --input $TRAIN --test-input $TEST \
    --regression-model-name answerdotai/ModernBERT-base \
    --target length --lr 1e-5 --epochs 5 --log-transform --device cuda \
    --output-dir models/route_balance/study/modernbert_fused_length_logtransform

check_disk
echo "=========================================="
echo "3. ModernBERT per-model 7B × 4 targets (5 ep) (~2h)"
echo "=========================================="
for TARGET in length_bucket length similarity judge_class; do
    OUTDIR=models/route_balance/study/modernbert_7b_${TARGET}
    echo "[$(date)] ModernBERT 7B ${TARGET}..."
    $PYTHON -m route_balance.predictor.route_balance.offline_training.train_bert_predictor \
        --input $TRAIN --test-input $TEST \
        --regression-model-name answerdotai/ModernBERT-base \
        --target-models Qwen/Qwen2.5-7B \
        --target $TARGET --lr 1e-5 --epochs 5 --device cuda \
        --output-dir $OUTDIR
done

check_disk
echo "=========================================="
echo "4. RoBERTa fused × 3 targets (5 ep) (~1.5h)"
echo "=========================================="
for TARGET in length similarity judge_class; do
    OUTDIR=models/route_balance/study/roberta_fused_${TARGET}
    echo "[$(date)] RoBERTa fused ${TARGET}..."
    $PYTHON -m route_balance.predictor.route_balance.offline_training.train_bert_predictor \
        --input $TRAIN --test-input $TEST \
        --regression-model-name roberta-base \
        --target $TARGET --lr 1e-5 --epochs 5 --device cuda \
        --output-dir $OUTDIR
done

check_disk
echo "=========================================="
echo "5. Qwen-0.5B LoRA × 3 targets (3 ep) (~3h)"
echo "=========================================="
for TARGET in length similarity judge; do
    OUTDIR=models/route_balance/study/qwen05b_fused_${TARGET}
    echo "[$(date)] Qwen-0.5B LoRA fused ${TARGET}..."
    $PYTHON -m route_balance.predictor.route_balance.offline_training.train_llm_predictor \
        --input $TRAIN --test-input $TEST \
        --target $TARGET --epochs 3 \
        --output-dir $OUTDIR
done

check_disk
echo "=========================================="
echo "6. KNN (~10 min)"
echo "=========================================="
echo "[$(date)] KNN..."
$PYTHON -m route_balance.predictor.route_balance.offline_training.train_knn_estimator \
    --input $TRAIN --test-input $TEST \
    --output-dir models/route_balance/study/knn/ --device cuda

check_disk
echo "=========================================="
echo "7. MLP (~20 min)"
echo "=========================================="
echo "[$(date)] MLP..."
$PYTHON -m route_balance.predictor.route_balance.offline_training.train_mlp_predictor \
    --input $TRAIN --test-input $TEST \
    --output-dir models/route_balance/study/mlp/ --device cuda

check_disk
echo "=========================================="
echo "8. LSTM latency (~2h)"
echo "=========================================="
echo "[$(date)] LSTM latency..."
$PYTHON -m route_balance.predictor.route_balance.offline_training.train_lstm_latency \
    --data-dir $LATENCY_DIR --output-dir models/route_balance/study/lstm_latency/ \
    || echo "LSTM latency FAILED — skipping"

check_disk
echo "=========================================="
echo "9. LSTM quality (~1h)"
echo "=========================================="
echo "[$(date)] LSTM quality..."
$PYTHON -m route_balance.predictor.route_balance.offline_training.train_lstm_predictor \
    --input $TRAIN --test-input $TEST \
    --output-dir models/route_balance/study/lstm_quality/ \
    || echo "LSTM quality FAILED — skipping"

check_disk
echo "=========================================="
echo "10. Bucket filtering evaluation"
echo "=========================================="
# Use the fused bucket model from early study
BUCKET_MODEL=models/route_balance/early_study/modernbert_fused_length_bucket_with_squad
if [ ! -d "$BUCKET_MODEL" ]; then
    # Fallback to full study model
    BUCKET_MODEL=models/route_balance/study/modernbert_fused_length_bucket
fi

echo "[$(date)] Bucket filtering simulation..."
$PYTHON -m route_balance.predictor.route_balance.offline_training.evaluate_bucket_filtering \
    --test-input $TEST \
    --model-dir $BUCKET_MODEL \
    --device cuda \
    --output models/route_balance/study/bucket_filtering_results.json \
    || echo "Bucket filtering eval FAILED — skipping"

echo "=========================================="
echo "[$(date)] ALL DONE"
echo "=========================================="
