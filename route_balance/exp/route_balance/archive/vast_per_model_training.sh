#!/bin/bash
# Vast.ai training: per-model ModernBERT + Qwen LoRA (all targets)
# Run: bash route_balance/exp/route_balance/vast_per_model_training.sh 2>&1 | tee /tmp/vast_training.log
#
# Expects: repo at ~/Code/llm/Block, training data synced, venv ready
# RTX 3090 24GB — ModernBERT-base fits easily

set -o pipefail
cd ~/Code/llm/Block
export PYTHONPATH=.
export HF_HOME=~/.cache/huggingface
PYTHON=python3  # or .venv/bin/python if venv used

TRAIN=data/route_balance/training_data/train_fixed.jsonl
TEST=data/route_balance/training_data/test_fixed.jsonl
TRAIN_REF=data/route_balance/training_data_with_ref/train_fixed.jsonl
TEST_REF=data/route_balance/training_data_with_ref/test_fixed.jsonl
ENCODER=answerdotai/ModernBERT-base
EPOCHS=5
LR=1e-5
BATCH=32
MAX_LEN=1024

check_disk() {
    local avail_gb=$(df -BG / --output=avail | tail -1 | tr -d ' G')
    if [ "$avail_gb" -lt 5 ]; then
        echo "[$(date)] DISK LOW: ${avail_gb}GB free. STOPPING."
        exit 1
    fi
    echo "[$(date)] Disk: ${avail_gb}GB free"
}

echo "=========================================="
echo "Vast.ai Per-Model Training"
echo "Start: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "=========================================="

# ==========================================
# 1. Per-model ModernBERT: length_bucket (×4 sizes, ~40 min)
# ==========================================
check_disk
echo "[$(date)] Per-model length_bucket..."
$PYTHON -m route_balance.predictor.route_balance.offline_training.train_bert_predictor \
    --input $TRAIN --test-input $TEST \
    --regression-model-name $ENCODER --target length_bucket \
    --epochs $EPOCHS --lr $LR --batch-size $BATCH --max-length $MAX_LEN \
    --output-dir models/route_balance/length_bucket/ablation/per_model \
    --seed 42

# ==========================================
# 2. Per-model ModernBERT: length log-transform (×4, ~40 min)
# ==========================================
check_disk
echo "[$(date)] Per-model length log-transform..."
$PYTHON -m route_balance.predictor.route_balance.offline_training.train_bert_predictor \
    --input $TRAIN --test-input $TEST \
    --regression-model-name $ENCODER --target length --log-transform \
    --epochs $EPOCHS --lr $LR --batch-size $BATCH --max-length $MAX_LEN \
    --output-dir models/route_balance/length_regression/ablation/per_model \
    --seed 42

# ==========================================
# 3. Per-model ModernBERT: similarity (×4, ~40 min)
# ==========================================
check_disk
echo "[$(date)] Per-model similarity..."
$PYTHON -m route_balance.predictor.route_balance.offline_training.train_bert_predictor \
    --input $TRAIN --test-input $TEST \
    --regression-model-name $ENCODER --target similarity \
    --epochs $EPOCHS --lr $LR --batch-size $BATCH --max-length $MAX_LEN \
    --output-dir models/route_balance/quality/similarity/ablation/per_model \
    --seed 42

# ==========================================
# 4. Per-model ModernBERT: judge_class (×4, ~60 min)
# ==========================================
check_disk
echo "[$(date)] Per-model judge_class..."
$PYTHON -m route_balance.predictor.route_balance.offline_training.train_bert_predictor \
    --input $TRAIN --test-input $TEST \
    --regression-model-name $ENCODER --target judge_class \
    --epochs $EPOCHS --lr $LR --batch-size $BATCH --max-length $MAX_LEN \
    --output-dir models/route_balance/quality/judge/ablation/per_model \
    --seed 42

# ==========================================
# 5. Per-model ModernBERT: reference_score (×4, ~40 min)
# ==========================================
check_disk
echo "[$(date)] Per-model reference_score..."
$PYTHON -m route_balance.predictor.route_balance.offline_training.train_bert_predictor \
    --input $TRAIN_REF --test-input $TEST_REF \
    --regression-model-name $ENCODER --target reference_score \
    --epochs $EPOCHS --lr $LR --batch-size $BATCH --max-length $MAX_LEN \
    --output-dir models/route_balance/quality/reference_score/ablation/per_model \
    --seed 42

# ==========================================
# 6. Qwen LoRA fused (×5 targets, ~30 min each)
# ==========================================
echo ""
echo "=========================================="
echo "Qwen LoRA Fused Training"
echo "=========================================="

for TARGET in length similarity judge reference_score; do
    check_disk
    if [ "$TARGET" = "reference_score" ]; then
        TR=$TRAIN_REF; TE=$TEST_REF
    else
        TR=$TRAIN; TE=$TEST
    fi
    echo "[$(date)] Qwen LoRA fused ${TARGET}..."
    $PYTHON -m route_balance.predictor.route_balance.offline_training.train_llm_predictor \
        --input $TR --test-input $TE \
        --target $TARGET --mode fused --epochs 3 \
        --output-dir models/route_balance/baselines/qwen05b_lora_fused/${TARGET} \
        || echo "Qwen LoRA fused ${TARGET} FAILED — skipping"
done

# ==========================================
# 7. Qwen LoRA per-model (×5 targets, ~30 min each)
# ==========================================
echo ""
echo "=========================================="
echo "Qwen LoRA Per-Model Training"
echo "=========================================="

for TARGET in length similarity judge reference_score; do
    check_disk
    if [ "$TARGET" = "reference_score" ]; then
        TR=$TRAIN_REF; TE=$TEST_REF
    else
        TR=$TRAIN; TE=$TEST
    fi
    echo "[$(date)] Qwen LoRA per_model ${TARGET}..."
    $PYTHON -m route_balance.predictor.route_balance.offline_training.train_llm_predictor \
        --input $TR --test-input $TE \
        --target $TARGET --mode per_model --epochs 3 \
        --output-dir models/route_balance/baselines/qwen05b_lora_per_model/${TARGET} \
        || echo "Qwen LoRA per_model ${TARGET} FAILED — skipping"
done

echo "=========================================="
echo "[$(date)] ALL DONE"
echo "=========================================="
