#!/bin/bash
#SBATCH --job-name=rb_rob_5ep
#SBATCH --partition=ampere
#SBATCH --account=KALYVIANAKI-SL3-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --qos=INTR
#SBATCH --output=/rds/user/wd312/hpc-work/llm/Block/logs/roberta_5ep_%j.log

# RoBERTa fused 5-epoch training for ALL deployment targets
# INTR job — 1 hour, sequential targets on 1 GPU

cd /rds/user/wd312/hpc-work/llm/Block
# Use separate RoBERTa venv (transformers==4.50.3) — NOT main .venv (transformers==5.3.0)
# Main .venv is for ModernBERT which requires transformers 5.x
# RoBERTa/DeBERTa require transformers 4.x
source /rds/user/wd312/hpc-work/venv_roberta/bin/activate

TRAIN_DATA=data/route_balance/training_data_with_ref/train_fixed.jsonl
TEST_DATA=data/route_balance/training_data_with_ref/test_fixed.jsonl
OUTPUT_BASE=models/route_balance/roberta_fused_5ep

mkdir -p $OUTPUT_BASE logs

echo "=== RoBERTa fused 5ep training — $(date) ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Transformers: $(python -c 'import transformers; print(transformers.__version__)')"
echo "Torch: $(python -c 'import torch; print(torch.__version__)')"
# No debug flags needed — venv_roberta has correct transformers version

# Target mapping: script target names
# length with log-transform = length_log equivalent
# judge_class = judge 10-class
# reference_score = reference_score regression
for target in length length_bucket judge_class reference_score similarity; do
    echo ""
    echo "=== Training target=$target — $(date) ==="

    EXTRA_ARGS=""
    if [ "$target" == "length" ]; then
        EXTRA_ARGS="--log-transform"
    fi

    python -u -m route_balance.predictor.route_balance.offline_training.train_fused_bert_predictor \
        --input $TRAIN_DATA \
        --test-input $TEST_DATA \
        --encoder-name roberta-base \
        --target $target \
        --output-dir $OUTPUT_BASE/$target \
        --epochs 5 \
        --batch-size 8 \
        --lr 1e-5 \
        --scheduler polynomial \
        --max-length 512 \
        --device cuda \
        --save-total-limit 1 \
        $EXTRA_ARGS
    echo "=== $target DONE — $(date) ==="
done

echo ""
echo "=== ALL TARGETS COMPLETE — $(date) ==="
ls -lh $OUTPUT_BASE/*/fused_model.pt 2>/dev/null
