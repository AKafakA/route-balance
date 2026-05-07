#!/bin/bash
# GPU VM Early Study — run all training jobs
# Usage: bash route_balance/exp/route_balance/run_gpu_vm_early_study.sh 2>&1 | tee /tmp/early_study.log
set -e
cd ~/Code/llm/RouteBalance
export PYTHONPATH=.
PYTHON=.venv/bin/python
LATENCY_DIR=data/route_balance/latency_data/tagged/
TRAIN_DATA=data/route_balance/training_data/train_fixed.jsonl
TEST_DATA=data/route_balance/training_data/test_fixed.jsonl
TRAIN_NOSQUAD=data/route_balance/training_data/train_fixed_nosquad.jsonl
TEST_NOSQUAD=data/route_balance/training_data/test_fixed_nosquad.jsonl

# Set up tagged latency data directory
mkdir -p data/route_balance/latency_data/tagged
ln -sf $(pwd)/data/route_balance/latency_data/all/latency_train_tagged.jsonl data/route_balance/latency_data/tagged/latency_train.jsonl
ln -sf $(pwd)/data/route_balance/latency_data/all/latency_test_tagged.jsonl data/route_balance/latency_data/tagged/latency_test.jsonl

echo "=========================================="
echo "Phase 1: CPU Latency Models (~10 min total)"
echo "=========================================="

echo "[$(date)] XGBoost E2E..."
$PYTHON -m route_balance.predictor.route_balance.offline_training.train_xgboost \
    --data-dir $LATENCY_DIR --output-dir models/route_balance/xgboost_e2e/ --target e2el

echo "[$(date)] XGBoost TTFT..."
$PYTHON -m route_balance.predictor.route_balance.offline_training.train_xgboost \
    --data-dir $LATENCY_DIR --output-dir models/route_balance/xgboost_ttft/ --target ttft

echo "[$(date)] XGBoost TPOT..."
$PYTHON -m route_balance.predictor.route_balance.offline_training.train_xgboost \
    --data-dir $LATENCY_DIR --output-dir models/route_balance/xgboost_tpot/ --target tpot

echo "[$(date)] Linear baseline..."
$PYTHON -m route_balance.predictor.route_balance.offline_training.train_linear \
    --data-dir $LATENCY_DIR --output-dir models/route_balance/linear/

echo "[$(date)] Roofline calibration..."
$PYTHON -m route_balance.predictor.route_balance.offline_training.train_roofline \
    --data-dir $LATENCY_DIR --output-dir models/route_balance/roofline/

echo "=========================================="
echo "Phase 2: ModernBERT fused early study (~2h)"
echo "  4 targets x {with_squad, no_squad}"
echo "=========================================="

for VARIANT in with_squad no_squad; do
    if [ "$VARIANT" = "with_squad" ]; then
        TR=$TRAIN_DATA; TE=$TEST_DATA
    else
        TR=$TRAIN_NOSQUAD; TE=$TEST_NOSQUAD
    fi

    for TARGET in length_bucket length similarity judge_class; do
        OUTDIR=models/route_balance/early_study/modernbert_fused_${TARGET}_${VARIANT}
        echo "[$(date)] ModernBERT fused ${TARGET} (${VARIANT})..."
        $PYTHON -m route_balance.predictor.route_balance.offline_training.train_bert_predictor \
            --input $TR --test-input $TE \
            --regression-model-name answerdotai/ModernBERT-base \
            --target $TARGET --lr 1e-5 --epochs 5 --device cuda \
            --output-dir $OUTDIR
    done
done

echo "=========================================="
echo "Phase 3: Qwen-0.5B LoRA fused early study (~2h)"
echo "  2 targets (length, quality) x {with_squad, no_squad}"
echo "=========================================="

for VARIANT in with_squad no_squad; do
    if [ "$VARIANT" = "with_squad" ]; then
        TR=$TRAIN_DATA; TE=$TEST_DATA
    else
        TR=$TRAIN_NOSQUAD; TE=$TEST_NOSQUAD
    fi

    for TARGET in length quality; do
        OUTDIR=models/route_balance/early_study/qwen05b_fused_${TARGET}_${VARIANT}
        echo "[$(date)] Qwen-0.5B LoRA fused ${TARGET} (${VARIANT})..."
        $PYTHON -m route_balance.predictor.route_balance.offline_training.train_llm_predictor \
            --input $TR --test-input $TE \
            --target $TARGET --epochs 3 \
            --output-dir $OUTDIR
    done
done

echo "=========================================="
echo "Phase 4: KNN retrain (~10 min)"
echo "=========================================="

echo "[$(date)] KNN with_squad..."
$PYTHON -m route_balance.predictor.route_balance.offline_training.train_knn_estimator \
    --input $TRAIN_DATA --test-input $TEST_DATA \
    --output-dir models/route_balance/early_study/knn_with_squad/

echo "[$(date)] KNN no_squad..."
$PYTHON -m route_balance.predictor.route_balance.offline_training.train_knn_estimator \
    --input $TRAIN_NOSQUAD --test-input $TEST_NOSQUAD \
    --output-dir models/route_balance/early_study/knn_no_squad/

echo "=========================================="
echo "[$(date)] ALL DONE"
echo "=========================================="
