#!/usr/bin/env bash
# dev-gpu only (the test profile): two small engines for the end-to-end test, and a record in deploy.sh's format
# (tools/engines.py reads it as the lease's), each started line read from the process and the engine as
# scripts/node_ctl.sh's cmd_started does. The engines serve Qwen2.5-0.5B under the test profile's names.
#   engines.sh start RUN_DIR RECORD [pair]      engines.sh stop RUN_DIR [pair]
set -euo pipefail
ROOT="${RB_VSR_TEST_ROOT:?set RB_VSR_TEST_ROOT to the dev-gpu test directory (venv-vllm inside)}"
HF="${RB_VSR_HF_HOME:?set RB_VSR_HF_HOME to the Hub cache holding the test model}"
PY=$ROOT/venv-vllm/bin/python
REV=060db6499f32faf8b98477b0a26969ef7d8b9987
ENGINES=("s0 8611 m-small" "s1 8612 m-small")          # engines.sh ... [pair]: s0 m-small, l0 m-large
for arg in "$@"; do if [ "$arg" = pair ]; then ENGINES=("s0 8611 m-small" "l0 8612 m-large"); fi; done
argv() { echo "--model Qwen/Qwen2.5-0.5B --revision $REV --served-model-name $2 --port $1 --host 0.0.0.0 --dtype float16 --gpu-memory-utilization 0.045 --max-model-len 4096 --enforce-eager"; }

start() {
  local run=$1 record=$2
  mkdir -p "$run"
  [ ! -e "$record" ] || { echo "$record exists" >&2; exit 1; }
  printf '# test-profile engines (vllm-sr-bench/tests/e2e/engines.sh), %s\n' "$(date -u +%FT%TZ)" > "$record.tmp"
  for e in "${ENGINES[@]}"; do
    set -- $e
    ! curl -sf -m 2 "http://127.0.0.1:$2/health" >/dev/null || { echo "port $2 already serves" >&2; exit 1; }
    env -i HOME="$HOME" PATH="$ROOT/venv-vllm/bin:/usr/bin:/bin" CUDA_VISIBLE_DEVICES=0 VLLM_SERVER_DEV_MODE=1 VLLM_HTTP_TIMEOUT_KEEP_ALIVE=600 \
        HF_HOME="$HF" HF_HUB_OFFLINE=1 \
        setsid "$PY" -m vllm.entrypoints.openai.api_server $(argv "$2" "$3") > "$run/engine.$1.log" 2>&1 < /dev/null &
    echo $! > "$run/engine.$1.pid"
    for _ in $(seq 1 240); do curl -sf -m 2 "http://127.0.0.1:$2/health" >/dev/null && break; sleep 2; done
    curl -sf -m 2 "http://127.0.0.1:$2/health" >/dev/null || { echo "$1 did not become healthy" >&2; tail -5 "$run/engine.$1.log"; exit 1; }
    printf '%s\tdevgpu\t127.0.0.1\t%s\t0\ta30\t1\tQwen/Qwen2.5-0.5B\t%s\n' "$1" "$2" "$(argv "$2" "$3")" >> "$record.tmp"
  done
  for e in "${ENGINES[@]}"; do
    set -- $e
    local pid start own cuda ka cmdline
    pid=$(cat "$run/engine.$1.pid")
    start=$(curl -sf -m 10 "http://127.0.0.1:$2/metrics" | awk '$1 == "process_start_time_seconds" {v = $2; n++} END {if (n != 1) exit 1; print v}')
    own=$(awk -v t="$(awk '{print $22}' /proc/$pid/stat)" -v hz="$(getconf CLK_TCK)" '/^btime/ {printf "%.3f", $2 + t / hz}' /proc/stat)
    awk -v a="$start" -v b="$own" 'BEGIN {d = a - b; exit !(d < 0.05 && d > -0.05)}' || { echo "$1: another process answers" >&2; exit 1; }
    cuda=$(tr '\0' '\n' < "/proc/$pid/environ" | sed -n 's/^CUDA_VISIBLE_DEVICES=//p')
    ka=$(tr '\0' '\n' < "/proc/$pid/environ" | sed -n 's/^VLLM_HTTP_TIMEOUT_KEEP_ALIVE=//p')
    cmdline=$(tr '\0' ' ' < "/proc/$pid/cmdline")
    printf 'started\t%s\t%s\t%s\t%s\n' "$1" "$start" "$cuda" "VLLM_HTTP_TIMEOUT_KEEP_ALIVE=$ka ${cmdline% }" >> "$record.tmp"
  done
  mv "$record.tmp" "$record"
  echo "${#ENGINES[@]} engines healthy; record $record"
}

stop() {
  local run=$1
  for e in "${ENGINES[@]}"; do
    set -- $e
    [ -e "$run/engine.$1.pid" ] || continue
    kill -TERM -- "-$(cat "$run/engine.$1.pid")" 2>/dev/null || true
  done
  sleep 5
  for e in "${ENGINES[@]}"; do set -- $e; ! curl -sf -m 2 "http://127.0.0.1:$2/health" >/dev/null || echo "port $2 still serves" >&2; done
}

case "${1:-}" in start) start "$2" "$3" ;; stop) stop "$2" ;; *) echo "usage: $0 start RUN_DIR RECORD | stop RUN_DIR" >&2; exit 2 ;; esac
