#!/usr/bin/env bash
# Every estimator and router input built from one family's judged training corpus, as the paper's runs built them.
# CPU only; run it in the scheduler's environment (stage 1) from the repository's root.
#
#   build_inputs.sh --family qwen|gemma --corpus SCORED_TRAIN.jsonl --out DIR [--python PY]
#
#   DIR/train_rows.jsonl     the training rows: every prompt whose answers all carry a judged score
#                            (router_training filter); Table A1's training counts
#   DIR/br_data/             BEST-Route's labels and split (router_training best-route-data)
#   DIR/br_upstream/         the same examples in upstream's format (tools/best_route_upstream.py), which
#                            scripts/train_best_route.sh trains on (on a GPU)
#   DIR/knn/                 the k-NN index, k = 50 (tools/knn_index.py); for Qwen also the ablations' indexes: 75, 50
#                            and 25% of the labels (knn_q75, knn_half, knn_q25) and every dataset left out in turn
#                            (knn_lodo_<dataset>), each from its rows subset (tools/row_subsets.py)
#   DIR/ap/                  AvengersPro at its five settings (router_training avengers-pro)
#
# Every output's md5 is printed at the end. A configuration names an artifact by path and md5, so built artifacts are
# pinned as docs/reproduce/02-run-experiments.md (step 2) describes.
set -euo pipefail
die() { echo "build_inputs.sh: $*" >&2; exit 1; }

FAMILY="" CORPUS="" OUT="" PY=python3
while [ $# -gt 0 ]; do
  [ $# -ge 2 ] || die "$1 needs a value"
  case "$1" in
    --family) FAMILY="$2" ;; --corpus) CORPUS="$2" ;; --out) OUT="$2" ;; --python) PY="$2" ;;
    *) die "unknown argument $1" ;;
  esac
  shift 2
done
[ -n "$FAMILY" ] && [ -n "$CORPUS" ] && [ -n "$OUT" ] \
  || die "usage: build_inputs.sh --family qwen|gemma --corpus SCORED_TRAIN.jsonl --out DIR [--python PY]"
case "$FAMILY" in
  qwen)  MODELS="Qwen/Qwen2.5-3B Qwen/Qwen2.5-7B Qwen/Qwen2.5-14B Qwen/Qwen2.5-72B" ;;
  gemma) MODELS="google/gemma-3-1b-it google/gemma-3-4b-it google/gemma-3-12b-it google/gemma-3-27b-it" ;;
  *) die "--family is qwen or gemma" ;;
esac
[ -s "$CORPUS" ] || die "$CORPUS is missing or empty"
[ -f tools/router_training.py ] || die "run from the repository's root"
[ ! -e "$OUT" ] || die "$OUT exists: give a new folder"
FLEET=configs/fleet_$FAMILY.json
KEY=deepeval-llama3.1-8b-it_reference
ts() { date -u +%FT%TZ; }
mkdir -p "$OUT/knn"
ROWS=$OUT/train_rows.jsonl

echo "$(ts) training rows"
"$PY" -m tools.router_training filter --corpus "$CORPUS" --fleet "$FLEET" --judge-key $KEY --out "$ROWS"
echo "$(ts) BEST-Route data and upstream's format"
# shellcheck disable=SC2086
"$PY" -m tools.router_training best-route-data --corpus "$ROWS" --fleet "$FLEET" --judge-key $KEY --models $MODELS \
  --out "$OUT/br_data"
"$PY" -m tools.best_route_upstream --family "$FAMILY" --rows "$ROWS" --split "$OUT/br_data" --out "$OUT/br_upstream"
echo "$(ts) k-NN index, k = 50"
"$PY" -m tools.knn_index --input "$ROWS" --k 50 --output-dir "$OUT/knn/knn" --device cpu 2>&1 | grep -v Batches | tail -4
if [ "$FAMILY" = qwen ]; then
  for f in 0.75:q75 0.5:half 0.25:q25; do
    echo "$(ts) k-NN index on ${f%%:*} of the labels"
    "$PY" -m tools.row_subsets fraction --rows "$ROWS" --fraction "${f%%:*}" --out "$OUT/knn/rows_${f#*:}.jsonl"
    "$PY" -m tools.knn_index --input "$OUT/knn/rows_${f#*:}.jsonl" --k 50 --output-dir "$OUT/knn/knn_${f#*:}" --device cpu \
      2>&1 | grep -v Batches | tail -2
  done
  for ds in beaver_tails code_ultra_feedback gsm8k lmsys mix_instruct reward_bench squad; do
    echo "$(ts) k-NN index without $ds"
    "$PY" -m tools.row_subsets without --rows "$ROWS" --dataset $ds --out "$OUT/knn/rows_lodo_$ds.jsonl"
    "$PY" -m tools.knn_index --input "$OUT/knn/rows_lodo_$ds.jsonl" --k 50 --output-dir "$OUT/knn/knn_lodo_$ds" --device cpu \
      2>&1 | grep -v Batches | tail -2
  done
fi
echo "$(ts) AvengersPro at five settings"
# shellcheck disable=SC2086
"$PY" -m tools.router_training avengers-pro --corpus "$ROWS" --fleet "$FLEET" --judge-key $KEY --models $MODELS \
  --family "$FAMILY" --pw 0.25 0.39 0.53 0.70 0.80 --embedder sentence-transformers/all-MiniLM-L6-v2 \
  --embedder-revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 --n-clusters 32 --out-root "$OUT/ap" 2>&1 | grep -v Batches | tail -6
(cd "$OUT" && find knn ap br_data br_upstream -type f ! -name 'rows_*.jsonl' -exec md5sum {} + | sort -k2)
echo "$(ts) done; next: scripts/train_best_route.sh --family $FAMILY --data $OUT/br_upstream"
