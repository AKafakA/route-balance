#!/bin/bash
# Re-run remaining failed experiments (Qwen judge already done)
cd ~/Code/llm/Block
export PYTHONPATH=.
PYTHON=.venv/bin/python
TRAIN=data/route_balance/training_data/train_fixed.jsonl
TEST=data/route_balance/training_data/test_fixed.jsonl
LATENCY_DIR=data/route_balance/latency_data/tagged/

check_disk() {
    local avail_gb=$(df -BG /home/wd312 --output=avail | tail -1 | tr -d ' G')
    if [ "$avail_gb" -lt 50 ]; then
        echo "[$(date)] DISK LOW: ${avail_gb}GB free. STOPPING."
        exit 1
    fi
    echo "[$(date)] Disk: ${avail_gb}GB free"
}

echo "=========================================="
echo "Re-running remaining (Qwen judge already done)"
echo "=========================================="

check_disk
echo "[$(date)] 1/4: LSTM latency (--device cuda)..."
rm -rf models/route_balance/study/lstm_latency 2>/dev/null
$PYTHON -m route_balance.predictor.route_balance.offline_training.train_lstm_latency \
    --data-dir $LATENCY_DIR --output-dir models/route_balance/study/lstm_latency/ \
    --device cuda \
    || echo "[$(date)] LSTM latency FAILED"

check_disk
echo "[$(date)] 2/4: ModernBERT fused judge_class..."
rm -rf models/route_balance/study/modernbert_fused_judge_class 2>/dev/null
$PYTHON -m route_balance.predictor.route_balance.offline_training.train_bert_predictor \
    --input $TRAIN --test-input $TEST \
    --regression-model-name answerdotai/ModernBERT-base \
    --target judge_class --lr 1e-5 --epochs 5 --seed 42 --device cuda \
    --output-dir models/route_balance/study/modernbert_fused_judge_class \
    || echo "[$(date)] ModernBERT fused judge_class FAILED"

check_disk
echo "[$(date)] 3/4: ModernBERT 7B judge_class..."
rm -rf models/route_balance/study/modernbert_7b_judge_class 2>/dev/null
$PYTHON -m route_balance.predictor.route_balance.offline_training.train_bert_predictor \
    --input $TRAIN --test-input $TEST \
    --regression-model-name answerdotai/ModernBERT-base \
    --target-models Qwen/Qwen2.5-7B \
    --target judge_class --lr 1e-5 --epochs 5 --seed 42 --device cuda \
    --output-dir models/route_balance/study/modernbert_7b_judge_class \
    || echo "[$(date)] ModernBERT 7B judge_class FAILED"

check_disk
echo "[$(date)] 4/4: MLP..."
rm -rf models/route_balance/study/mlp 2>/dev/null
$PYTHON -m route_balance.predictor.route_balance.offline_training.train_mlp_predictor \
    --input $TRAIN --test-input $TEST \
    --output-dir models/route_balance/study/mlp/ --device cuda \
    || echo "[$(date)] MLP FAILED"

echo "=========================================="
echo "[$(date)] REMAINING RE-RUNS DONE"
echo "=========================================="
