#!/bin/bash
# Final unified evaluation using the new evaluation module
cd ~/Code/llm/RouteBalance
export PYTHONPATH=.
PYTHON=.venv/bin/python
TRAIN=data/route_balance/training_data/train_fixed.jsonl
TEST=data/route_balance/training_data/test_fixed.jsonl
OUTDIR=models/route_balance/evaluation
EVAL="$PYTHON -m route_balance.predictor.route_balance.offline_training.evaluation.run_evaluation"

mkdir -p $OUTDIR

echo "=========================================="
echo "Final Unified Evaluation — $(date)"
echo "=========================================="

# --- 1. Length (regression + bucket + filtering) ---
echo ""
echo "[$(date)] === Length ==="
$EVAL --test-input $TEST --train-input $TRAIN --target length \
    --predictors \
        knn:models/route_balance/study/knn \
        mlp:models/route_balance/study/mlp \
        encoder:models/route_balance/early_study/modernbert_fused_length_with_squad \
        encoder:models/route_balance/study/modernbert_fused_length_logtransform \
        encoder:models/route_balance/study/modernbert_7b_length \
        bucket_encoder:models/route_balance/early_study/modernbert_fused_length_bucket_with_squad \
        bucket_encoder:models/route_balance/study/modernbert_7b_length_bucket \
        llm:models/route_balance/study/qwen05b_fused_length \
    --device cuda --output $OUTDIR/eval_length.json \
    || echo "[$(date)] Length eval FAILED"

# Long-training models if available
for dir in models/route_balance/long_study/modernbert_fused_length_mse models/route_balance/long_study/modernbert_fused_length_logtransform; do
    if [ -d "$dir" ] && [ -f "$dir/model.safetensors" ]; then
        name=$(basename $dir)
        echo "[$(date)] Length (100ep: $name)..."
        $EVAL --test-input $TEST --train-input $TRAIN --target length \
            --predictors encoder:$dir \
            --device cuda --output $OUTDIR/eval_length_${name}.json \
            || echo "[$(date)] $name eval FAILED"
    fi
done

if [ -d "models/route_balance/long_study/modernbert_fused_length_bucket" ]; then
    echo "[$(date)] Length bucket (100ep)..."
    $EVAL --test-input $TEST --train-input $TRAIN --target length \
        --predictors bucket_encoder:models/route_balance/long_study/modernbert_fused_length_bucket \
        --device cuda --output $OUTDIR/eval_length_bucket_100ep.json \
        || echo "[$(date)] Bucket 100ep eval FAILED"
fi

# --- 2. Similarity ---
echo ""
echo "[$(date)] === Similarity ==="
$EVAL --test-input $TEST --train-input $TRAIN --target similarity \
    --predictors \
        knn:models/route_balance/study/knn \
        mlp:models/route_balance/study/mlp \
        encoder:models/route_balance/early_study/modernbert_fused_similarity_with_squad \
        encoder:models/route_balance/study/modernbert_7b_similarity \
        llm:models/route_balance/study/qwen05b_fused_similarity \
    --device cuda --output $OUTDIR/eval_similarity.json \
    || echo "[$(date)] Similarity eval FAILED"

# --- 3. Judge (safety-aware) ---
echo ""
echo "[$(date)] === Judge ==="
$EVAL --test-input $TEST --train-input $TRAIN --target judge \
    --predictors \
        knn:models/route_balance/study/knn \
        mlp:models/route_balance/study/mlp \
        encoder:models/route_balance/study/modernbert_fused_judge_class \
        encoder:models/route_balance/study/modernbert_7b_judge_class \
        llm:models/route_balance/study/qwen05b_fused_judge \
    --device cuda --output $OUTDIR/eval_judge.json \
    || echo "[$(date)] Judge eval FAILED"

echo ""
echo "=========================================="
echo "[$(date)] FINAL EVALUATION DONE"
echo "=========================================="
echo "Results:"
ls -la $OUTDIR/*.json 2>/dev/null
