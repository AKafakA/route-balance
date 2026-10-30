#!/usr/bin/env bash
# One shard scored by the G-Eval scorer (tools/geval_score.py) against one judge: the judge Llama-3.1-8B-Instruct served
# by vLLM (configs/fleet_judge.json, configs/engine_args_judge.json), the judge key deepeval-llama3.1-8b-it_reference.
# Resumable: the scorer writes chunks next to OUTPUT and starts at the first missing one.
#
#   judge_one.sh INPUT OUTPUT LOGFILE BASE_URL [extra scorer args...]
#
# JUDGE_PY names the judge environment's python (default python3).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=${JUDGE_PY:-python3}
IN=$(realpath "$1"); OUT=$(realpath -m "$2"); LOG=$(realpath -m "$3"); URL=$4; shift 4
case "$PY" in */*) PY=$(realpath "$PY") ;; esac        # the scorer runs from the output folder
OD=$(dirname "$OUT"); mkdir -p "$OD"
export OPENAI_API_KEY=dummy DEEPEVAL_TELEMETRY_OPT_OUT=YES
exec >> "$LOG" 2>&1
ts(){ date -u +%FT%TZ; }
code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "$URL/models")
[ "$code" = 200 ] || { echo "$(ts) FATAL: judge server $URL not healthy (HTTP $code)"; echo "SCORER_RC=97"; exit 97; }
K=0; while [ -f "$OD/chunks/chunk_$(printf %04d $K).jsonl" ]; do K=$((K+1)); done
echo "$(ts) START in=$IN out=$OUT url=$URL start_chunk=$K args=$* pid=$$"
cd "$OD"
"$PY" -u "$HERE/../tools/geval_score.py" --input "$IN" --output "$OUT" \
  --judge-key deepeval-llama3.1-8b-it_reference --model meta-llama/Llama-3.1-8B-Instruct \
  --base-url "$URL" --start-chunk $K "$@"
RC=$?
echo "$(ts) SCORER_RC=$RC"
exit $RC
