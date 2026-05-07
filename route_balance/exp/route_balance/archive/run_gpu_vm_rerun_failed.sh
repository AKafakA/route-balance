#!/bin/bash
# Re-run failed experiments with all fixes applied
cd ~/Code/llm/RouteBalance
export PYTHONPATH=.
PYTHON=.venv/bin/python
TRAIN=data/route_balance/training_data/train_fixed.jsonl
TEST=data/route_balance/training_data/test_fixed.jsonl
LATENCY_DIR=data/route_balance/latency_data/tagged/

check_disk() {
    local avail_gb=$(df -BG /home/anon --output=avail | tail -1 | tr -d ' G')
    if [ "$avail_gb" -lt 50 ]; then
        echo "[$(date)] DISK LOW: ${avail_gb}GB free. STOPPING."
        exit 1
    fi
    echo "[$(date)] Disk: ${avail_gb}GB free"
}

echo "=========================================="
echo "Re-running failed experiments (all fixes applied)"
echo "Seed: 42 (explicit)"
echo "=========================================="

check_disk
echo "[$(date)] 1/5: Qwen-0.5B LoRA judge (was: indentation crash)..."
rm -rf models/route_balance/study/qwen05b_fused_judge 2>/dev/null
$PYTHON -m route_balance.predictor.route_balance.offline_training.train_llm_predictor \
    --input $TRAIN --test-input $TEST \
    --target judge --epochs 3 --seed 42 \
    --output-dir models/route_balance/study/qwen05b_fused_judge \
    || echo "[$(date)] Qwen-0.5B judge FAILED again"

check_disk
echo "[$(date)] 2/5: LSTM latency (was: off-by-one target)..."
rm -rf models/route_balance/study/lstm_latency 2>/dev/null
$PYTHON -m route_balance.predictor.route_balance.offline_training.train_lstm_latency \
    --data-dir $LATENCY_DIR --output-dir models/route_balance/study/lstm_latency/ \
    --device cuda \
    || echo "[$(date)] LSTM latency FAILED again"

check_disk
echo "[$(date)] 3/5: ModernBERT fused judge_class (was: wrong judge labels)..."
rm -rf models/route_balance/study/modernbert_fused_judge_class 2>/dev/null
$PYTHON -m route_balance.predictor.route_balance.offline_training.train_bert_predictor \
    --input $TRAIN --test-input $TEST \
    --regression-model-name answerdotai/ModernBERT-base \
    --target judge_class --lr 1e-5 --epochs 5 --seed 42 --device cuda \
    --output-dir models/route_balance/study/modernbert_fused_judge_class \
    || echo "[$(date)] ModernBERT fused judge_class FAILED"

check_disk
echo "[$(date)] 4/5: ModernBERT 7B judge_class (was: wrong judge labels)..."
rm -rf models/route_balance/study/modernbert_7b_judge_class 2>/dev/null
$PYTHON -m route_balance.predictor.route_balance.offline_training.train_bert_predictor \
    --input $TRAIN --test-input $TEST \
    --regression-model-name answerdotai/ModernBERT-base \
    --target-models Qwen/Qwen2.5-7B \
    --target judge_class --lr 1e-5 --epochs 5 --seed 42 --device cuda \
    --output-dir models/route_balance/study/modernbert_7b_judge_class \
    || echo "[$(date)] ModernBERT 7B judge_class FAILED"

check_disk
echo "[$(date)] 5/5: MLP (was: no seed)..."
rm -rf models/route_balance/study/mlp 2>/dev/null
$PYTHON -m route_balance.predictor.route_balance.offline_training.train_mlp_predictor \
    --input $TRAIN --test-input $TEST \
    --output-dir models/route_balance/study/mlp/ --device cuda \
    || echo "[$(date)] MLP FAILED"

echo "=========================================="
echo "[$(date)] RE-RUNS DONE"
echo "=========================================="
