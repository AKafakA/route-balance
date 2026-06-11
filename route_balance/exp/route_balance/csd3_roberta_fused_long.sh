#!/bin/bash
#SBATCH --job-name=rb_rob_fused
#SBATCH --partition=ampere
#SBATCH --account=KALYVIANAKI-SL3-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --output=/rds/user/wd312/hpc-work/llm/Block/training_logs/roberta_fused_long_%j.log

# Fused RoBERTa long training — deployment models
# 6 targets, 4 GPUs, 2 rounds of 3 parallel jobs
# Params matched to Block's original: lr=1e-5, batch=8, warmup=0.03, polynomial, 100ep
# Keeps best + last checkpoint for resume
#
# Run with: sbatch route_balance/exp/route_balance/csd3_roberta_fused_long.sh

set -e
cd /rds/user/wd312/hpc-work/llm/Block
export PYTHONPATH=/rds/user/wd312/hpc-work/llm/Block:$PYTHONPATH
source /rds/user/wd312/hpc-work/venv_roberta/bin/activate
mkdir -p training_logs models/route_balance/roberta_fused_long

echo "=== Fused RoBERTa Long Training — $(date) ==="
python3 -c "import transformers, torch; print(f'tf={transformers.__version__}, torch={torch.__version__}, GPUs={torch.cuda.device_count()}')"

TRAIN="data/route_balance/training_data/train_fixed.jsonl"
TEST="data/route_balance/training_data/test_fixed.jsonl"
TRAIN_REF="data/route_balance/training_data/train_with_ref.jsonl"
TEST_REF="data/route_balance/training_data/test_with_ref.jsonl"

# Match Block params: lr=1e-5, batch=8, warmup=0.03, polynomial, fp16
COMMON="--encoder-name roberta-base --lr 1e-5 --batch-size 8 --max-length 512 --precision fp16 --scheduler polynomial --seed 42 --save-total-limit 2 --epochs 100"
FUSED="python3 -m route_balance.predictor.route_balance.offline_training.train_fused_bert_predictor"
OUT="models/route_balance/roberta_fused_long"

# Round 1: length_log + length_mse + length_bucket (3 GPUs)
echo ""
echo "=== Round 1: length targets (3 GPUs) — $(date) ==="

echo "  GPU 0: length log-transform 100ep"
CUDA_VISIBLE_DEVICES=0 $FUSED \
    --input $TRAIN --test-input $TEST \
    --target length --log-transform $COMMON \
    --output-dir $OUT/length_log \
    > training_logs/rob_fused_length_log.log 2>&1 &
PID1=$!

echo "  GPU 1: length MSE 100ep"
CUDA_VISIBLE_DEVICES=1 $FUSED \
    --input $TRAIN --test-input $TEST \
    --target length $COMMON \
    --output-dir $OUT/length_mse \
    > training_logs/rob_fused_length_mse.log 2>&1 &
PID2=$!

echo "  GPU 2: length_bucket 100ep"
CUDA_VISIBLE_DEVICES=2 $FUSED \
    --input $TRAIN --test-input $TEST \
    --target length_bucket $COMMON \
    --output-dir $OUT/length_bucket \
    > training_logs/rob_fused_length_bucket.log 2>&1 &
PID3=$!

echo "  Waiting for Round 1..."
FAIL=0
for PN in "length_log:$PID1" "length_mse:$PID2" "length_bucket:$PID3"; do
    NAME="${PN%%:*}"; PID="${PN#*:}"
    if wait $PID; then echo "  $NAME: SUCCESS"
    else echo "  $NAME: FAILED (exit $?)"; FAIL=$((FAIL+1)); fi
done

# Round 2: judge + similarity + reference_score (3 GPUs)
echo ""
echo "=== Round 2: quality targets (3 GPUs) — $(date) ==="

echo "  GPU 0: judge_class 100ep"
CUDA_VISIBLE_DEVICES=0 $FUSED \
    --input $TRAIN --test-input $TEST \
    --target judge_class $COMMON \
    --output-dir $OUT/judge_class \
    > training_logs/rob_fused_judge.log 2>&1 &
PID4=$!

echo "  GPU 1: similarity 100ep"
CUDA_VISIBLE_DEVICES=1 $FUSED \
    --input $TRAIN --test-input $TEST \
    --target similarity $COMMON \
    --output-dir $OUT/similarity \
    > training_logs/rob_fused_similarity.log 2>&1 &
PID5=$!

echo "  GPU 2: reference_score 100ep"
CUDA_VISIBLE_DEVICES=2 $FUSED \
    --input $TRAIN_REF --test-input $TEST_REF \
    --target reference_score $COMMON \
    --output-dir $OUT/reference_score \
    > training_logs/rob_fused_refscore.log 2>&1 &
PID6=$!

echo "  Waiting for Round 2..."
for PN in "judge:$PID4" "similarity:$PID5" "reference_score:$PID6"; do
    NAME="${PN%%:*}"; PID="${PN#*:}"
    if wait $PID; then echo "  $NAME: SUCCESS"
    else echo "  $NAME: FAILED (exit $?)"; FAIL=$((FAIL+1)); fi
done

echo ""
echo "=== Results ==="
for d in $OUT/*/; do
    name=$(basename $d)
    if [ -f "${d}training_results.json" ]; then
        echo "  $name: OK"
        tail -3 training_logs/rob_fused_${name}*.log 2>/dev/null | grep -E "MAE=|accuracy|spearman" | head -1
    else
        echo "  $name: MISSING"
    fi
done

echo ""
echo "=== Fused RoBERTa Long Training Complete — $(date) ==="
echo "Failures: $FAIL/6"
