#!/bin/bash
cd ~/Code/llm/Block
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=3

TRAIN=data/route_balance/training_data_with_ref/train_fixed.jsonl
TEST=data/route_balance/training_data_with_ref/test_fixed.jsonl
OUT_BASE=models/route_balance/refscore_study

pkill -f "train_bert_predictor.*reference_score" 2>/dev/null
sleep 1

echo "=== Fused reference_score ==="
python3 route_balance/predictor/route_balance/offline_training/train_bert_predictor.py \
  --input $TRAIN --test-input $TEST \
  --regression-model-name answerdotai/ModernBERT-base \
  --target reference_score \
  --epochs 5 --lr 2e-5 --batch-size 32 --max-length 1024 \
  --output-dir $OUT_BASE/modernbert_fused_reference_score \
  --seed 42

echo "=== Done: $(date) ==="
