#!/usr/bin/env bash
# The end-to-end run on one GPU (proofs/gpu/e2e.py describes it): build the inputs, start the two engines, every run
# under the ref tree and then under the submission tree, compare, stop the engines. Progress goes to OUT.progress.
#
#   proofs/gpu/run.sh OUT REF_TREE SUBMISSION_TREE
#
# with, in the environment: SCHED_PY (the scheduler environment's python), ENGINE_VENV (vLLM 0.18.1's environment),
# ARTIFACTS (the artifact root the runs loaded), DATA_ROOT (the folder of the Qwen prompt set), HF_HOMES (the Hub
# homes that hold the two Qwen models and the two encoders, colon-separated), AVENGERS_REPO and BEST_ROUTE_REPO (the
# pinned upstream checkouts) and TIKTOKEN (AvengersPro's cl100k_base cache).
set -u
OUT=$1 REF=$2 SUB=$3
E2E="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/e2e.py"
for v in SCHED_PY ENGINE_VENV ARTIFACTS DATA_ROOT HF_HOMES AVENGERS_REPO BEST_ROUTE_REPO TIKTOKEN; do
  [ -n "${!v:-}" ] || { echo "set $v" >&2; exit 2; }
done
[ ! -e "$OUT" ] || { echo "$OUT exists" >&2; exit 2; }
stamp() { echo "$(date -u +%FT%TZ) $*" >> "$OUT.progress"; }
HF_ARGS=()
IFS=: read -r -a homes <<< "$HF_HOMES"
for h in "${homes[@]}"; do HF_ARGS+=(--hf-home "$h"); done
stamp start
"$SCHED_PY" "$E2E" build --tree "$REF" --out "$OUT" --artifacts "$ARTIFACTS" --data-root "$DATA_ROOT" "${HF_ARGS[@]}" \
  > "$OUT.build.log" 2>&1
rc=$?; stamp "build rc=$rc"
[ "$rc" = 0 ] || { stamp done; exit 1; }
"$SCHED_PY" "$E2E" engines-up --out "$OUT" --tree "$REF" --engine-venv "$ENGINE_VENV" > "$OUT.engines.log" 2>&1
rc=$?; stamp "engines rc=$rc"
if [ "$rc" = 0 ]; then
  for kind in ref submission; do
    tree=$REF; [ "$kind" = submission ] && tree=$SUB
    "$SCHED_PY" "$E2E" run --out "$OUT" --tree "$tree" --kind "$kind" --artifacts "$ARTIFACTS" --data-root "$DATA_ROOT" \
      --avengers-repo "$AVENGERS_REPO" --best-route-repo "$BEST_ROUTE_REPO" --tiktoken "$TIKTOKEN" \
      --scheduler-python "$SCHED_PY" > "$OUT.$kind.log" 2>&1
    stamp "run $kind rc=$?"
  done
  "$SCHED_PY" "$E2E" compare --out "$OUT" > "$OUT.compare.log" 2>&1; stamp "compare rc=$?"
fi
"$SCHED_PY" "$E2E" engines-down --out "$OUT" >> "$OUT.engines.log" 2>&1; stamp "engines down rc=$?"
stamp done
