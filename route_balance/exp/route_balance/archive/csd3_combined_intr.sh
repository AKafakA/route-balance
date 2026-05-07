#!/bin/bash
#SBATCH -J route_balance_combined
#SBATCH -A KALYVIANAKI-SL3-GPU
#SBATCH -p ampere
#SBATCH --qos=INTR
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/rds/user/anon/hpc-work/llm/RouteBalance/logs/route_balance_combined_%j.out
#SBATCH --error=/rds/user/anon/hpc-work/llm/RouteBalance/logs/route_balance_combined_%j.err

# Combined 1h intr session:
# 1. Judge 7B (~20 min)
# 2. RoBERTa experiments with downgraded transformers (~35 min)

cd /rds/user/anon/hpc-work/llm/RouteBalance
export PYTHONPATH=.
PYTHON=.venv/bin/python
TRAIN=data/route_balance/training_data/train_fixed.jsonl
TEST=data/route_balance/training_data/test_fixed.jsonl

echo "=========================================="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "=========================================="

# --- Part 1: Judge 7B (~20 min) ---
echo "[$(date)] 1/2: ModernBERT 7B judge_class..."
$PYTHON -m route_balance.predictor.route_balance.offline_training.train_bert_predictor \
    --input $TRAIN --test-input $TEST \
    --regression-model-name answerdotai/ModernBERT-base \
    --target-models Qwen/Qwen2.5-7B \
    --target judge_class --lr 1e-5 --epochs 5 --seed 42 --device cuda \
    --output-dir models/route_balance/study/modernbert_7b_judge_class \
    || echo "[$(date)] Judge 7B FAILED"

# --- Part 2: RoBERTa with downgraded transformers (~35 min) ---
echo "[$(date)] 2/2: Setting up RoBERTa environment..."

# Create temporary venv for RoBERTa (transformers 4.50.3)
if [ ! -d ".venv_roberta" ]; then
    echo "[$(date)] Creating RoBERTa venv..."
    python3 -m venv .venv_roberta
    .venv_roberta/bin/python -m pip install --upgrade pip -q
    .venv_roberta/bin/pip install torch==2.5.1 transformers==4.50.3 scipy scikit-learn numpy -q
    echo "[$(date)] RoBERTa venv created"
else
    echo "[$(date)] RoBERTa venv exists"
fi

RPYTHON=.venv_roberta/bin/python

# Pre-download roberta-base
$RPYTHON -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('roberta-base')" 2>/dev/null

echo "[$(date)] RoBERTa length (MSE)..."
$RPYTHON -m route_balance.predictor.route_balance.offline_training.train_bert_predictor \
    --input $TRAIN --test-input $TEST \
    --regression-model-name roberta-base \
    --target length --lr 1e-5 --epochs 5 --seed 42 --device cuda \
    --output-dir models/route_balance/study/roberta_fused_length \
    || echo "[$(date)] RoBERTa length FAILED"

echo "[$(date)] RoBERTa similarity..."
$RPYTHON -m route_balance.predictor.route_balance.offline_training.train_bert_predictor \
    --input $TRAIN --test-input $TEST \
    --regression-model-name roberta-base \
    --target similarity --lr 1e-5 --epochs 5 --seed 42 --device cuda \
    --output-dir models/route_balance/study/roberta_fused_similarity \
    || echo "[$(date)] RoBERTa similarity FAILED"

echo "[$(date)] RoBERTa judge_class..."
$RPYTHON -m route_balance.predictor.route_balance.offline_training.train_bert_predictor \
    --input $TRAIN --test-input $TEST \
    --regression-model-name roberta-base \
    --target judge_class --lr 1e-5 --epochs 5 --seed 42 --device cuda \
    --output-dir models/route_balance/study/roberta_fused_judge_class \
    || echo "[$(date)] RoBERTa judge_class FAILED"

echo "=========================================="
echo "[$(date)] ALL DONE"
echo "=========================================="
