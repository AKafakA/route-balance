#!/usr/bin/env bash
# BEST-Route's ranker, trained with upstream's own trainer (microsoft/best-route-llm at the commit scripts/setup.sh
# pins) as the paper's runs trained it, and the checkpoint folder the scheduler loads. One GPU.
#
#   train_best_route.sh --family qwen|gemma --upstream DIR --data DIR --out DIR [--python PY] [--gpu N]
#
#   --upstream  the best-route-llm checkout
#   --data      tools/best_route_upstream.py's output folder (train.jsonl and val.jsonl)
#   --out       a new folder. Upstream's trainer writes its run in OUT/run; OUT/checkpoint is the folder a scheduler
#               configuration names as best_route_upstream's "checkpoint": upstream's best checkpoint plus the family's
#               labels file from configs/best_route/ (the candidates' order, the fleet models and the collator lengths).
#   --python    the training environment's python (default python3): Python 3.12 with
#               configs/best_route/requirements-train.txt
#   --gpu       the one GPU the trainer sees (default 0), as the runs trained it: on a node with more GPUs
#               upstream's trainer would otherwise use them all and change the effective batch size
#
# The arguments are the runs': the base model microsoft/deberta-v3-small, the family's four candidates in ladder order,
# the loss prob_nlabels and the quality metric armoRM_scores (tools/best_route_upstream.py puts the DeepEval score
# under that key). Every other setting is upstream's default; its collator lengths (128 and 128) are the labels file's.
# Upstream loads the backbone by name through the Hugging Face cache: with the snapshot configs/scheduler_hub.json pins
# in the cache (scripts/setup.sh's scheduler role), run it with HF_HUB_OFFLINE=1, as the runs did.
set -euo pipefail
die() { echo "train_best_route.sh: $*" >&2; exit 1; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FAMILY="" UPSTREAM="" DATA="" OUT="" PY=python3 GPU=0
while [ $# -gt 0 ]; do
  [ $# -ge 2 ] || die "$1 needs a value"
  case "$1" in
    --family) FAMILY="$2" ;; --upstream) UPSTREAM="$2" ;; --data) DATA="$2" ;; --out) OUT="$2" ;; --python) PY="$2" ;;
    --gpu) GPU="$2" ;;
    *) die "unknown argument $1" ;;
  esac
  shift 2
done
[ -n "$FAMILY" ] && [ -n "$UPSTREAM" ] && [ -n "$DATA" ] && [ -n "$OUT" ] \
  || die "usage: train_best_route.sh --family qwen|gemma --upstream DIR --data DIR --out DIR [--python PY] [--gpu N]"
case "$FAMILY" in
  qwen)  CANDIDATES="Qwen2.5-3B_bo1,Qwen2.5-7B_bo1,Qwen2.5-14B_bo1,Qwen2.5-72B_bo1"
         LABELS="$HERE/../configs/best_route/qwen_upstream_20260928b/route_balance_labels.json" ;;
  gemma) CANDIDATES="gemma-3-1b_bo1,gemma-3-4b_bo1,gemma-3-12b_bo1,gemma-3-27b_bo1"
         LABELS="$HERE/../configs/best_route/gemma_upstream_20261002/route_balance_labels.json" ;;
  *) die "--family is qwen or gemma" ;;
esac
[ -f "$UPSTREAM/train_router.py" ] || die "$UPSTREAM is not a best-route-llm checkout"
[ -s "$DATA/train.jsonl" ] && [ -s "$DATA/val.jsonl" ] || die "$DATA has no train.jsonl and val.jsonl"
[ ! -e "$OUT" ] || die "$OUT exists: give a new folder"
DATA="$(cd "$DATA" && pwd)"
case "$PY" in */*) PY="$(realpath -s -- "$PY")" ;; esac   # absolute, symlinks kept (a venv's python is one)
mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"

(cd "$UPSTREAM" && CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u train_router.py \
   --model_name microsoft/deberta-v3-small --run_name "best_route_upstream_$FAMILY" \
   --train_data_path "$DATA/train.jsonl" --eval_data_path "$DATA/val.jsonl" --test_data_path "$DATA/val.jsonl" \
   --candidate_models "$CANDIDATES" --loss_type prob_nlabels \
   --quality_metric armoRM_scores --output_dir "$OUT/run") > "$OUT/train.log" 2>&1 \
  || die "upstream's training failed (see $OUT/train.log)"
[ -d "$OUT/run/checkpoint-best" ] || die "upstream wrote no checkpoint-best under $OUT/run"
mkdir "$OUT/checkpoint"
cp -r "$OUT/run/checkpoint-best/." "$OUT/checkpoint/"
cp "$LABELS" "$OUT/checkpoint/"
echo "$OUT/checkpoint: upstream's best checkpoint and $(basename "$LABELS")"
