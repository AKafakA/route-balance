#!/bin/bash
#SBATCH -J route_balance_judge7b
#SBATCH -A KALYVIANAKI-SL3-GPU
#SBATCH -p ampere
#SBATCH --qos=INTR
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/rds/user/wd312/hpc-work/llm/Block/logs/route_balance_judge7b_%j.out
#SBATCH --error=/rds/user/wd312/hpc-work/llm/Block/logs/route_balance_judge7b_%j.err

cd /rds/user/wd312/hpc-work/llm/Block
export PYTHONPATH=.
PYTHON=.venv/bin/python
TRAIN=data/route_balance/training_data/train_fixed.jsonl
TEST=data/route_balance/training_data/test_fixed.jsonl

echo "[$(date)] ModernBERT 7B judge_class..."
$PYTHON -m route_balance.predictor.route_balance.offline_training.train_bert_predictor \
    --input $TRAIN --test-input $TEST \
    --regression-model-name answerdotai/ModernBERT-base \
    --target-models Qwen/Qwen2.5-7B \
    --target judge_class --lr 1e-5 --epochs 5 --seed 42 --device cuda \
    --output-dir models/route_balance/study/modernbert_7b_judge_class

echo "[$(date)] DONE"
