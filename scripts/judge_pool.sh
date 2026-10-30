#!/usr/bin/env bash
# Every shard SHARD_DIR/shard_*.jsonl (tools/judge_shards.py) scored once with judge_one.sh against one of the judges
# listed in URLS_FILE (one base URL per line, e.g. http://node:8100/v1). One worker per judge claims the next unclaimed
# shard (an atomic mkdir), so a faster judge takes more shards; shards go in name order. Each shard's output:
# OUT_ROOT/<shard>/scored.jsonl, its log OUT_ROOT/logs/<shard>.log, its claim OUT_ROOT/claims/<shard>/{url,rc}. A shard
# whose scorer failed keeps rc != 0: remove its claim folder and run the pool again to retry it (the scorer resumes at
# its first missing chunk). The paper's judging passed --max-concurrent 64 --chunk-size 200.
#
#   judge_pool.sh SHARD_DIR OUT_ROOT URLS_FILE [scorer args...]
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SH=$1; OUT=$2; URLS=$3; shift 3
mkdir -p "$OUT/claims" "$OUT/logs"
ts(){ date -u +%FT%TZ; }
worker() {
  local url=$1 s b rc; shift
  for s in "$SH"/shard_*.jsonl; do
    b=$(basename "$s" .jsonl)
    mkdir "$OUT/claims/$b" 2>/dev/null || continue
    echo "$url" > "$OUT/claims/$b/url"
    echo "$(ts) CLAIM $b $url"
    "$HERE/judge_one.sh" "$s" "$OUT/$b/scored.jsonl" "$OUT/logs/$b.log" "$url" "$@"
    rc=$?
    echo "$rc" > "$OUT/claims/$b/rc"
    echo "$(ts) DONE $b rc=$rc"
  done
}
while read -r url; do [ -n "$url" ] && worker "$url" "$@" & done < "$URLS"
wait
echo "$(ts) ALL DONE: $(grep -l '^0$' "$OUT"/claims/*/rc 2>/dev/null | wc -l) shards rc=0 of $(ls "$SH"/shard_*.jsonl | wc -l)"
