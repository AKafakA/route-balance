#!/bin/bash
# Full overnight evaluation pipeline for RouteBalance
# Chains: LoRA multi-target → LSTM v2 → comprehensive evaluation
# Run from Block repo root with venv activated
set -e

cd ~/Code/llm/Block
source .venv/bin/activate
export PYTHONPATH=~/Code/llm/Block:$PYTHONPATH

RESULTS_DIR="eval_results/$(date +%Y%m%d)"
mkdir -p "$RESULTS_DIR"

LOG="$RESULTS_DIR/pipeline.log"
exec > >(tee -a "$LOG") 2>&1

echo "=========================================="
echo "  RouteBalance Full Evaluation Pipeline"
echo "  Started: $(date)"
echo "  Results: $RESULTS_DIR"
echo "=========================================="

# ============================================================
# PHASE 1: Train remaining models
# ============================================================

echo ""
echo "=== PHASE 1a: LoRA multi-target (RoBERTa B + ModernBERT B) ==="

for encoder in roberta-base answerdotai/ModernBERT-base; do
    if [ "$encoder" = "roberta-base" ]; then enc_dir="roberta"; else enc_dir="modernbert"; fi
    outdir="models/route_balance/lora_multitarget_${enc_dir}_B"
    if [ -f "${outdir}/lora_multitarget_model.pt" ]; then
        echo "SKIP lora_multitarget_${enc_dir}_B -- already exists"
        continue
    fi
    echo "=== Training lora_multitarget_${enc_dir}_B ==="
    python -m route_balance.predictor.route_balance.offline_training.train_lora_multitarget \
        --input data/route_balance/scored/train_scored_filtered.jsonl \
        --test-input data/route_balance/scored/test_scored_filtered.jsonl \
        --encoder-name "$encoder" \
        --targets 'length_bucket:classification:16:64' 'deepeval:regression:1' \
        --epochs 5 --lr 1e-4 --device cuda \
        --output-dir "$outdir"
done

echo ""
echo "=== PHASE 1b: LSTM v2 on enriched data ==="
LSTM_OUT="models/route_balance/lstm_v2_enriched"
if [ -f "${LSTM_OUT}/lstm_v2_predictor.pt" ]; then
    echo "SKIP LSTM v2 -- already exists"
else
    python -m route_balance.predictor.route_balance.offline_training.train_lstm_latency_v2 \
        --data-dir data/route_balance/latency_data/enriched/ \
        --output-dir "$LSTM_OUT" \
        --target actual_e2e_latency --epochs 50 --device cuda
fi

echo ""
echo "=== PHASE 1 COMPLETE ==="

# ============================================================
# PHASE 2: Quality model evaluation (all models × 6 targets)
# ============================================================

echo ""
echo "=========================================="
echo "  PHASE 2: Quality Model Evaluation"
echo "=========================================="

TRAIN_DATA="data/route_balance/scored/train_scored_filtered.jsonl"
TEST_DATA="data/route_balance/scored/test_scored_filtered.jsonl"

# Define all models to evaluate per target
# Format: type:path
declare -A MODELS_BY_TARGET

# --- Regression targets: length, similarity, judge, reference_score, deepeval ---
for target in length similarity judge reference_score deepeval; do
    PREDICTORS=""

    # RoBERTa fused 5ep
    rdir="models/route_balance/roberta_fused_5ep/${target}"
    [ -d "$rdir" ] || rdir="models/route_balance/deepeval_models/roberta_fused_deepeval"
    if [ "$target" = "deepeval" ]; then rdir="models/route_balance/deepeval_models/roberta_fused_deepeval"; fi
    if [ "$target" = "length" ]; then rdir="models/route_balance/roberta_fused_5ep/length"; fi
    if [ "$target" = "similarity" ]; then rdir="models/route_balance/roberta_fused_5ep/reference_score"; fi  # similarity was under ref_score
    [ -d "$rdir" ] && PREDICTORS="$PREDICTORS encoder:$rdir"

    # KNN
    kdir="models/route_balance/knn"
    [ "$target" = "deepeval" ] && kdir="models/route_balance/deepeval_models/knn_deepeval"
    [ -d "$kdir" ] && PREDICTORS="$PREDICTORS knn:$kdir"

    # MLP
    mdir="models/route_balance/baselines/mlp"
    [ -d "$mdir" ] && PREDICTORS="$PREDICTORS mlp:$mdir"

    # LoRA-encoder RoBERTa
    ldir="models/route_balance/lora_encoder_roberta/${target}"
    [ "$target" = "judge" ] && ldir="models/route_balance/lora_encoder_roberta/judge_class"
    [ -d "$ldir" ] && PREDICTORS="$PREDICTORS lora_encoder:$ldir"

    # LoRA-encoder ModernBERT
    lmdir="models/route_balance/lora_encoder_modernbert/${target}"
    [ "$target" = "judge" ] && lmdir="models/route_balance/lora_encoder_modernbert/judge_class"
    [ -d "$lmdir" ] && PREDICTORS="$PREDICTORS lora_encoder:$lmdir"

    # Qwen LoRA fused
    qdir="models/route_balance/qwen_lora_fused/${target}"
    [ "$target" = "deepeval" ] && qdir="models/route_balance/deepeval_models/qwen_lora_fused_deepeval"
    [ -d "$qdir" ] && PREDICTORS="$PREDICTORS llm:$qdir"

    if [ -n "$PREDICTORS" ]; then
        echo ""
        echo "--- Evaluating target=$target ---"
        echo "  Predictors: $PREDICTORS"
        python -m route_balance.predictor.route_balance.offline_training.evaluation.run_evaluation \
            --test-input "$TEST_DATA" \
            --train-input "$TRAIN_DATA" \
            --predictors $PREDICTORS \
            --target "$target" \
            --device cuda \
            --output "$RESULTS_DIR/quality_${target}.json" \
            --csv "$RESULTS_DIR/quality_${target}.csv" \
            || echo "WARN: evaluation failed for target=$target"
    fi
done

# --- Bucket target: length_bucket ---
echo ""
echo "--- Evaluating target=length_bucket ---"
BUCKET_PREDICTORS=""

# RoBERTa fused bucket
bdir="models/route_balance/roberta_fused_5ep/length_bucket"
[ -d "$bdir" ] && BUCKET_PREDICTORS="$BUCKET_PREDICTORS bucket:$bdir"

# LoRA-encoder RoBERTa bucket
lbdir="models/route_balance/lora_encoder_roberta/length_bucket"
[ -d "$lbdir" ] && BUCKET_PREDICTORS="$BUCKET_PREDICTORS lora_encoder:$lbdir"

# LoRA-encoder ModernBERT bucket
lmbdir="models/route_balance/lora_encoder_modernbert/length_bucket"
[ -d "$lmbdir" ] && BUCKET_PREDICTORS="$BUCKET_PREDICTORS lora_encoder:$lmbdir"

if [ -n "$BUCKET_PREDICTORS" ]; then
    python -m route_balance.predictor.route_balance.offline_training.evaluation.run_evaluation \
        --test-input "$TEST_DATA" \
        --train-input "$TRAIN_DATA" \
        --predictors $BUCKET_PREDICTORS \
        --target length_bucket \
        --device cuda \
        --output "$RESULTS_DIR/quality_length_bucket.json" \
        --csv "$RESULTS_DIR/quality_length_bucket.csv" \
        || echo "WARN: evaluation failed for target=length_bucket"
fi

# --- Multi-target models (evaluate each head separately) ---
echo ""
echo "--- Evaluating multi-target models ---"
for variant in roberta_multitarget_B modernbert_multitarget_B roberta_multitarget_A modernbert_multitarget_A; do
    mtdir="models/route_balance/deepeval_models/${variant}"
    if [ -d "$mtdir" ]; then
        for mt_target in length_bucket deepeval; do
            echo "  Multi-target ${variant} / ${mt_target}"
            python -m route_balance.predictor.route_balance.offline_training.evaluation.run_evaluation \
                --test-input "$TEST_DATA" \
                --train-input "$TRAIN_DATA" \
                --predictors "multitarget:$mtdir" \
                --target "$mt_target" \
                --device cuda \
                --output "$RESULTS_DIR/multitarget_${variant}_${mt_target}.json" \
                || echo "WARN: multi-target eval failed for ${variant}/${mt_target}"
        done
    fi
done

# --- LoRA multi-target ---
for enc_dir in roberta modernbert; do
    lmt="models/route_balance/lora_multitarget_${enc_dir}_B"
    if [ -d "$lmt" ]; then
        for mt_target in length_bucket deepeval; do
            echo "  LoRA multi-target ${enc_dir}_B / ${mt_target}"
            python -m route_balance.predictor.route_balance.offline_training.evaluation.run_evaluation \
                --test-input "$TEST_DATA" \
                --train-input "$TRAIN_DATA" \
                --predictors "lora_multitarget:$lmt" \
                --target "$mt_target" \
                --device cuda \
                --output "$RESULTS_DIR/lora_multitarget_${enc_dir}_B_${mt_target}.json" \
                || echo "WARN: lora multi-target eval failed for ${enc_dir}_B/${mt_target}"
        done
    fi
done

echo ""
echo "=== PHASE 2 COMPLETE ==="

# ============================================================
# PHASE 3: Latency model evaluation
# ============================================================

echo ""
echo "=========================================="
echo "  PHASE 3: Latency Model Evaluation"
echo "=========================================="

LATENCY_TEST="data/route_balance/latency_data/enriched/latency_test_tagged_enriched.jsonl"
LATENCY_TRAIN="data/route_balance/latency_data/enriched/latency_train_tagged_enriched.jsonl"

for lat_target in actual_e2e_latency actual_ttft actual_tpot; do
    echo ""
    echo "--- Latency target=$lat_target ---"

    PREDICTORS_ARG="roofline static median"

    # XGBoost
    if [ "$lat_target" = "actual_e2e_latency" ]; then
        XGB_DIR="models/route_balance/xgboost_enriched"
    elif [ "$lat_target" = "actual_ttft" ]; then
        XGB_DIR="models/route_balance/xgboost_enriched_ttft"
    else
        XGB_DIR="models/route_balance/xgboost_enriched_tpot"
    fi

    XGB_ARG=""
    [ -d "$XGB_DIR" ] && XGB_ARG="--xgboost-dir $XGB_DIR" && PREDICTORS_ARG="xgboost $PREDICTORS_ARG"

    LSTM_ARG=""
    if [ "$lat_target" = "actual_e2e_latency" ] && [ -d "$LSTM_OUT" ]; then
        LSTM_ARG="--lstm-v2-dir $LSTM_OUT"
        PREDICTORS_ARG="$PREDICTORS_ARG lstm_v2"
    fi

    python -m route_balance.predictor.route_balance.offline_training.evaluation.eval_latency_predictors \
        --test-data "$LATENCY_TEST" \
        --train-data "$LATENCY_TRAIN" \
        --target "$lat_target" \
        --predictors $PREDICTORS_ARG \
        $XGB_ARG $LSTM_ARG \
        --output "$RESULTS_DIR/latency_${lat_target}.json" \
        || echo "WARN: latency eval failed for target=$lat_target"
done

echo ""
echo "=== PHASE 3 COMPLETE ==="

# ============================================================
# PHASE 4: XGBoost calibration + SLO filter evaluation
# ============================================================

echo ""
echo "=========================================="
echo "  PHASE 4: SLO Filter Evaluation"
echo "=========================================="

python3 << 'PYEOF'
import json
import numpy as np
import sys
sys.path.insert(0, ".")

from route_balance.predictor.route_balance.estimators.xgboost_predictor import XGBoostLatencyPredictor, build_feature_vector
from route_balance.predictor.route_balance.offline_training.train_xgboost import load_latency_data, prepare_features_and_targets

RESULTS_DIR = sys.argv[1] if len(sys.argv) > 1 else "eval_results"
print("=== Generating XGBoost calibration residuals ===")

# Load enriched test data
test_by_type = load_latency_data("data/route_balance/latency_data/enriched/", instance_types=None)

# Load XGBoost models
xgb = XGBoostLatencyPredictor.load("models/route_balance/xgboost_enriched/")

calibration = {}
for inst_type, records in sorted(test_by_type.items()):
    actuals = []
    preds = []
    for rec in records:
        actual = rec.get("actual_e2e_latency", 0)
        if actual <= 0:
            continue
        ss = rec.get("schedule_state", {})
        if not ss:
            continue
        prompt = int(rec.get("num_prompt_tokens", 0))
        output = int(rec.get("actual_output_tokens") or rec.get("num_predicted_output_tokens", 0))
        if output <= 0:
            continue
        try:
            fv = build_feature_vector(ss, prompt, output)
            result = xgb.predict(inst_type, fv)
            pred_e2e = result.get("e2e_latency", 0)
            if pred_e2e > 0:
                actuals.append(actual)
                preds.append(pred_e2e)
        except Exception:
            continue

    if actuals:
        residuals = np.array(actuals) - np.array(preds)
        calibration[inst_type] = {
            "residuals": residuals.tolist(),
            "mean": float(residuals.mean()),
            "std": float(residuals.std()),
            "n": len(residuals),
            "mae": float(np.abs(residuals).mean()),
        }
        print(f"  {inst_type}: n={len(residuals)}, residual mean={residuals.mean():.4f}s, std={residuals.std():.4f}s")

# Save calibration
with open(f"{RESULTS_DIR}/xgboost_calibration.json", "w") as f:
    json.dump({k: {kk: vv for kk, vv in v.items() if kk != "residuals"} for k, v in calibration.items()}, f, indent=2)

# Evaluate SLO filters
print("\n=== Evaluating SLO filters ===")

SLO_CONFIGS = {
    "ttft_slo_s": [0.5, 1.0, 2.0, 5.0],
    "e2e_slo_s": [5.0, 10.0, 20.0, 30.0],
}

filter_results = {}
for slo_type, thresholds in SLO_CONFIGS.items():
    for thresh in thresholds:
        key = f"{slo_type}={thresh}"
        filter_results[key] = {}

        for inst_type, cal in calibration.items():
            residuals = np.array(cal.get("residuals", []))  # actual - predicted
            n = len(residuals)
            if n == 0:
                continue

            # For each record: would the filter accept or reject?
            # SLOs-Serve: accept if predicted <= SLO
            # QLM: accept if predicted + z*std <= SLO (z=1.96 for 95%)
            # RouteBalanceCDF: accept if P(actual <= SLO | predicted) >= confidence

            test_recs = test_by_type[inst_type]
            results_per_filter = {
                "slos_serve": {"tp": 0, "fp": 0, "tn": 0, "fn": 0},
                "qlm_95": {"tp": 0, "fp": 0, "tn": 0, "fn": 0},
                "route_balance_cdf_90": {"tp": 0, "fp": 0, "tn": 0, "fn": 0},
            }

            std = cal["std"]
            for rec in test_recs:
                actual = rec.get("actual_e2e_latency", 0) if "e2e" in slo_type else rec.get("actual_ttft", 0)
                if actual <= 0:
                    continue
                ss = rec.get("schedule_state", {})
                prompt = int(rec.get("num_prompt_tokens", 0))
                output = int(rec.get("actual_output_tokens") or rec.get("num_predicted_output_tokens", 0))
                if output <= 0:
                    continue
                try:
                    fv = build_feature_vector(ss, prompt, output)
                    result = xgb.predict(inst_type, fv)
                    predicted = result.get("e2e_latency", 0) if "e2e" in slo_type else result.get("ttft", 0)
                except Exception:
                    continue

                meets_slo = actual <= thresh
                residual = actual - predicted

                # SLOs-Serve: accept if predicted <= SLO
                slos_accept = predicted <= thresh
                # QLM: accept if predicted + 1.96*std <= SLO
                qlm_accept = (predicted + 1.96 * std) <= thresh
                # RouteBalanceCDF: accept if P(residual <= SLO - predicted) >= 0.90
                route_balance_accept = np.mean(residuals <= (thresh - predicted)) >= 0.90

                for fname, accepted in [("slos_serve", slos_accept), ("qlm_95", qlm_accept), ("route_balance_cdf_90", route_balance_accept)]:
                    if accepted and meets_slo:
                        results_per_filter[fname]["tp"] += 1
                    elif accepted and not meets_slo:
                        results_per_filter[fname]["fp"] += 1
                    elif not accepted and meets_slo:
                        results_per_filter[fname]["fn"] += 1
                    else:
                        results_per_filter[fname]["tn"] += 1

            for fname, counts in results_per_filter.items():
                tp, fp, tn, fn = counts["tp"], counts["fp"], counts["tn"], counts["fn"]
                precision = tp / max(tp + fp, 1)
                recall = tp / max(tp + fn, 1)
                f1 = 2 * precision * recall / max(precision + recall, 1e-8)
                accuracy = (tp + tn) / max(tp + fp + tn + fn, 1)

                if inst_type not in filter_results[key]:
                    filter_results[key][inst_type] = {}
                filter_results[key][inst_type][fname] = {
                    "precision": round(precision, 4),
                    "recall": round(recall, 4),
                    "f1": round(f1, 4),
                    "accuracy": round(accuracy, 4),
                    "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                }

        # Print summary
        print(f"\n  {key}:")
        for inst_type in sorted(filter_results[key]):
            for fname, m in filter_results[key][inst_type].items():
                print(f"    {inst_type}/{fname}: F1={m['f1']:.3f} P={m['precision']:.3f} R={m['recall']:.3f} Acc={m['accuracy']:.3f}")

with open(f"{RESULTS_DIR}/slo_filter_results.json", "w") as f:
    json.dump(filter_results, f, indent=2)

print(f"\nResults saved to {RESULTS_DIR}/")
PYEOF "$RESULTS_DIR"

echo ""
echo "=== PHASE 4 COMPLETE ==="

# ============================================================
# SUMMARY
# ============================================================

echo ""
echo "=========================================="
echo "  Pipeline Complete: $(date)"
echo "=========================================="
echo "Results in: $RESULTS_DIR/"
ls -la "$RESULTS_DIR/"
echo ""
echo "Quality results:  quality_*.json / quality_*.csv"
echo "Latency results:  latency_*.json"
echo "Filter results:   slo_filter_results.json"
echo "Calibration:      xgboost_calibration.json"
