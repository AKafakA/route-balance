#!/usr/bin/env bash
# The judged shards joined back into each corpus, in shard (= file) order, each checked against its unscored input
# (tools/judge_validate.py: same rows, order, fields; every score in [0,1]; exit 2 lists the unscored pairs in
# <out>.missing.json, which the gap-fill step re-judges).
#
#   judge_assemble.sh SHARD_DIR OUT_ROOT PREFIX=UNSCORED.jsonl=SCORED.jsonl [...]
#
# SHARD_DIR and OUT_ROOT are judge_pool.sh's; every shard's claim must hold rc=0. PY names the python (default python3).
# Exits non-zero when a corpus cannot be joined or fails validation (validate's exit 1); exit 2 alone, unscored pairs
# for the gap-fill, is reported and the next corpus is assembled.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=${PY:-python3}
case "$PY" in */*) PY=$(realpath -s -- "$PY") ;; esac   # absolute, symlinks kept (a venv's python is one)
SH=$1; OUT=$2; shift 2
bad=0
for s in "$SH"/shard_*.jsonl; do
  b=$(basename "$s" .jsonl)
  [ "$(cat "$OUT/claims/$b/rc" 2>/dev/null)" = 0 ] || { echo "  $b: not done"; bad=1; }
done
[ $bad = 0 ] || { echo "not every shard finished with rc=0: remove the failed shards' claim folders (OUT_ROOT/claims/<shard>), then run judge_pool.sh again"; exit 1; }
failed=0
for spec in "$@"; do
  pre=${spec%%=*}; rest=${spec#*=}; unscored=$(realpath "${rest%%=*}"); scored=$(realpath -m "${rest#*=}")
  test ! -e "$scored" || { echo "$scored exists"; exit 1; }
  for s in $(ls "$SH" | grep "^shard_${pre}" | sort); do cat "$OUT/${s%.jsonl}/scored.jsonl" || exit 1; done > "$scored" || { echo "$scored: a shard could not be joined"; exit 1; }
  echo "$scored: $(wc -l < "$scored") rows"
  (cd "$HERE/.." && "$PY" -m tools.judge_validate "$unscored" "$scored"); rc=$?
  echo "$scored validate rc=$rc"
  [ $rc = 0 ] || [ $rc = 2 ] || failed=1
done
exit $failed
