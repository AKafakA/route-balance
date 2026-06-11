#!/bin/bash
#SBATCH --job-name=rb_roberta_test
#SBATCH --partition=ampere
#SBATCH --account=KALYVIANAKI-SL3-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/rds/user/wd312/hpc-work/llm/Block/training_logs/roberta_test_%j.log

# INTR test: verify RoBERTa training works with transformers==4.50.3
# Run with: sbatch --qos=INTR route_balance/exp/route_balance/csd3_roberta_intr_test.sh

set -e
cd /rds/user/wd312/hpc-work/llm/Block
export PYTHONPATH=/rds/user/wd312/hpc-work/llm/Block:$PYTHONPATH

# Use SEPARATE venv for RoBERTa (transformers==4.50.3)
# Main venv at /rds/.../venv/ is for ModernBERT (transformers 5.x) — DO NOT TOUCH
source /rds/user/wd312/hpc-work/venv_roberta/bin/activate

echo "=== Environment ==="
python3 -c "import transformers, torch; print(f'tf={transformers.__version__}, torch={torch.__version__}, CUDA={torch.cuda.is_available()}')"

TRAIN_DATA="data/route_balance/training_data/train_fixed.jsonl"
TEST_DATA="data/route_balance/training_data/test_fixed.jsonl"

echo "=== Test 1: RoBERTa MSE regression (3 epochs) ==="
python3 -m route_balance.predictor.route_balance.offline_training.train_bert_predictor \
    --input $TRAIN_DATA --test-input $TEST_DATA \
    --regression-model-name roberta-base \
    --target-model "Qwen/Qwen2.5-7B" \
    --target length \
    --loss-type mse \
    --epochs 3 \
    --max-length 512 \
    --precision fp16 \
    --save-total-limit 2 \
    --output-dir models/route_balance/test/roberta_mse_7b

echo "=== Test 2: RoBERTa log-transform regression (3 epochs) ==="
python3 -m route_balance.predictor.route_balance.offline_training.train_bert_predictor \
    --input $TRAIN_DATA --test-input $TEST_DATA \
    --regression-model-name roberta-base \
    --target-model "Qwen/Qwen2.5-7B" \
    --target length \
    --log-transform \
    --epochs 3 \
    --max-length 512 \
    --precision fp16 \
    --save-total-limit 2 \
    --output-dir models/route_balance/test/roberta_log_7b

echo "=== Test 3: RoBERTa bucket classification (3 epochs) ==="
python3 -m route_balance.predictor.route_balance.offline_training.train_bert_predictor \
    --input $TRAIN_DATA --test-input $TEST_DATA \
    --regression-model-name roberta-base \
    --target-model "Qwen/Qwen2.5-7B" \
    --target length_bucket \
    --epochs 3 \
    --max-length 512 \
    --precision fp16 \
    --save-total-limit 2 \
    --output-dir models/route_balance/test/roberta_bucket_7b

echo "=== Test 4: Fused RoBERTa bucket (1 epoch) ==="
python3 -m route_balance.predictor.route_balance.offline_training.train_fused_bert_predictor \
    --input $TRAIN_DATA --test-input $TEST_DATA \
    --encoder-name roberta-base \
    --target length_bucket \
    --epochs 1 \
    --max-length 512 \
    --precision fp16 \
    --save-total-limit 2 \
    --output-dir models/route_balance/test/roberta_fused_bucket

echo "=== All tests passed ==="
