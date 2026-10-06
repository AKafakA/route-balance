# shellcheck shell=bash disable=SC2015
# (SC2015: `a && b || fail` is meant: fail when either a or b fails.)
# Node-side engine and scheduler control. Never run by hand: deploy.sh, stop_engines.sh and
# stop_scheduler.sh stream node_layout.sh + this file to `bash -s -- <command> <args>` over ssh, so the
# node always runs the operator's copy. Every command fails loudly (exit 1) and prints why.
#
#   preflight   <gpu_class> <gpus_on_node>
#   launch      <instance_id> <port> <gpus> <model> <revision> <vLLM argv...>
#   wait        <instance_id> <port> <model> <timeout_s>
#   stop        <instance_id>
#   gpus-free   <gpus>                  kill what still computes on these GPUs, then require them idle
#   sched-start <run_dir> <tree> <artifacts> <log_dir> <host> <port> <torch_threads> <max_connections>
#               <request_timeout_s>
#   sched-wait  <port> <timeout_s>
#   sched-stop

set -euo pipefail
fail() { echo "[$(hostname -s)] FAILED: $*" >&2; exit 1; }
note() { echo "[$(hostname -s)] $*"; }

venv_complete() {   # the venv exists and setup.sh finished it
  [ -f "$1/rb-setup.spec" ] && grep -q '^complete ' "$1/rb-setup.spec" || fail "$1 is not a completed setup.sh venv"
}

alive() { [ -n "$1" ] && kill -0 "$1" 2>/dev/null; }

# A pid file holds "<pid> <log> <boot id> <start time>". The pid counts only while it is still that process
# (same boot, same start time): /mydata keeps pid files across reboots, and pids are reused.
BOOT_ID=$(cat /proc/sys/kernel/random/boot_id)
# the child that becomes the process writes its own record (field 22 of /proc/<pid>/stat is its start time)
# shellcheck disable=SC2016
RECORD_AND_EXEC='s=$(cat /proc/$$/stat); s=${s##*) }; read -ra f <<< "$s"; echo "$$ $2 $3 ${f[19]}" > "$1"; shift 3'

start_time() {
  local s f
  s=$(cat "/proc/$1/stat" 2>/dev/null) || return 0
  s=${s##*) }
  read -ra f <<< "$s"
  echo "${f[19]}"
}

pid_of() {           # the recorded pid if that process still runs, else empty
  local pid log boot start
  [ -f "$1" ] || return 0
  read -r pid log boot start < "$1" || return 0
  if [ -n "$pid" ] && [ "$boot" = "$BOOT_ID" ] && [ "$(start_time "$pid")" = "$start" ]; then echo "$pid"; fi
}

log_of() {           # the log file recorded for a process, or empty
  if [ -f "$1" ]; then cut -d' ' -f2 "$1"; fi
}

port_free() {       # no listener on the port; a failed ss fails, never reads as free
  local out
  out=$(ss -Hltn "sport = :$1") || fail "ss failed while checking port $1"
  [ -z "$out" ]
}

gpu_uuids() {        # uuids of the listed GPU indices; fails when nvidia-smi fails or an index has no GPU
  local idx u
  for idx in ${1//,/ }; do
    u=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v i="$idx" '$1 == i {print $2}') || return 1
    [ -n "$u" ] || return 1
    echo "$u"
  done
}

busy_pids() {        # pids of compute processes on the listed GPU indices
  local uuids
  uuids=$(gpu_uuids "$1") || return 1
  nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | while IFS=', ' read -r u p; do
    if grep -qx "$u" <<< "$uuids"; then echo "$p"; fi
  done
}

stop_group() {       # stop_group <pid file> <what>: TERM the process group, KILL after 60 s
  local pidf="$1" pid i
  pid=$(pid_of "$pidf")
  if ! alive "$pid"; then rm -f "$pidf"; note "$2: not running"; return 0; fi
  kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid"
  for i in $(seq 1 60); do alive "$pid" || break; sleep 1; done
  if alive "$pid"; then kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid"; sleep 2; fi
  alive "$pid" && fail "$2 (pid $pid) survived SIGKILL"
  rm -f "$pidf"
  note "$2: stopped (pid $pid, after ${i}s)"
}

cmd_preflight() {
  local cls="$1" n="$2" names driver
  mountpoint -q "$MYDATA" || fail "$MYDATA is not mounted (after a reboot, run setup.sh again to remount it)"
  venv_complete "$ENGINE_VENV"
  names=$(nvidia-smi --query-gpu=name --format=csv,noheader) || fail "nvidia-smi failed"
  [ "$(wc -l <<< "$names")" -eq "$n" ] || fail "expected $n GPU(s), nvidia-smi lists $(wc -l <<< "$names")"
  while read -r line; do [[ "${line^^}" == *"${cls^^}"* ]] || fail "GPU '$line' is not a ${cls^^}"; done <<< "$names"
  driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | sort -u) || fail "nvidia-smi failed"
  [ "$driver" = "$DRIVER_VERSION" ] || fail "driver $driver != $DRIVER_VERSION"
  note "preflight ok: $n x ${cls^^}, driver $driver"
}

cmd_launch() {
  local id="$1" port="$2" gpus="$3" model="$4" rev="$5"; shift 5
  local pidf="$RB_RUN/engine.$id.pid" snap busy log
  mountpoint -q "$MYDATA" || fail "$MYDATA is not mounted"
  venv_complete "$ENGINE_VENV"
  snap="$RB_HF_HOME/hub/models--${model//\//--}/snapshots/$rev"
  [ -f "$snap/config.json" ] || fail "$id: no snapshot $model@$rev under $RB_HF_HOME (run setup.sh --model $model@$rev)"
  alive "$(pid_of "$pidf")" && fail "$id is already running (pid $(pid_of "$pidf")); stop_engines.sh first"
  port_free "$port" || fail "$id: port $port is in use"
  busy=$(busy_pids "$gpus") || fail "nvidia-smi failed"
  [ -z "$busy" ] || fail "$id: GPUs $gpus are in use by pid(s) $(echo "$busy" | tr '\n' ' '); stop_engines.sh first"
  mkdir -p "$RB_LOGS" "$RB_RUN"
  rm -f "$pidf"                        # a stale record (that process is gone), replaced by the new one below
  log="$RB_LOGS/engine.$id.$(date -u +%Y%m%dT%H%M%SZ).log"
  printf '%s\n' "CUDA_VISIBLE_DEVICES=$gpus VLLM_SERVER_DEV_MODE=1 VLLM_HTTP_TIMEOUT_KEEP_ALIVE=$ENGINE_KEEP_ALIVE_S $ENGINE_VENV/bin/python -u -m vllm.entrypoints.openai.api_server $*" \
    > "$RB_RUN/engine.$id.cmd"
  # VLLM_SERVER_DEV_MODE=1 serves vLLM's POST /reset_prefix_cache (routes only; the serving path is unchanged), which every
  # run's warm-up calls to start from an empty prefix cache (tools/direct.py).
  # VLLM_HTTP_TIMEOUT_KEEP_ALIVE (vLLM's own setting, stock 5 s) above the clients' keep-alive (vLLM bench's stock
  # connector, 60 s): at 5 s an engine could close an idle connection just as the scheduler reused it ("Can not write
  # request body", 2 of 4 runs at the 32-call setting); v3 met the same race on its scheduler's inbound hop and raised
  # that server's keep-alive to 600 s, and the engines' is raised to 600 s too.
  # A new session per engine: its pid is its process group, which holds every worker it spawns.
  # (SC2016: the inner script expands in the child; SC2094: $log is recorded in the pid file, not read.)
  # shellcheck disable=SC2016,SC2094
  setsid --fork bash -c "$RECORD_AND_EXEC"'; exec "$@"' _ "$pidf" "$log" "$BOOT_ID" \
    env -i HOME="$HOME" PATH="$RB_PATH" LANG=C.UTF-8 CUDA_HOME="$CUDA_DIR" CUDA_VISIBLE_DEVICES="$gpus" \
    HF_HOME="$RB_HF_HOME" HF_HUB_OFFLINE=1 VLLM_SERVER_DEV_MODE=1 VLLM_HTTP_TIMEOUT_KEEP_ALIVE="$ENGINE_KEEP_ALIVE_S" \
    "$ENGINE_VENV/bin/python" -u -m vllm.entrypoints.openai.api_server "$@" > "$log" 2>&1 < /dev/null
  local i
  for i in $(seq 1 20); do [ -s "$pidf" ] && break; sleep 0.25; done
  alive "$(pid_of "$pidf")" || fail "$id did not start; log $log: $(tail -5 "$log" 2>/dev/null | tr '\n' ' ')"
  note "$id launched: pid $(pid_of "$pidf") gpus $gpus port $port log $log"
}

cmd_wait() {
  local id="$1" port="$2" model="$3" timeout="$4" pidf="$RB_RUN/engine.$1.pid" t0 pid log served
  t0=$(date +%s)
  pid=$(pid_of "$pidf"); log=$(log_of "$pidf")
  alive "$pid" || fail "$id is not running"
  while ! curl -sf -m 5 "http://127.0.0.1:$port/health" > /dev/null; do
    alive "$pid" || fail "$id exited while loading; log $log:
$(tail -30 "$log")"
    [ $(( $(date +%s) - t0 )) -lt "$timeout" ] || fail "$id not healthy after ${timeout}s; log $log:
$(tail -30 "$log")"
    sleep 10
  done
  served=$(curl -sf -m 10 "http://127.0.0.1:$port/v1/models") || fail "$id: GET /v1/models failed"
  grep -qF "\"id\":\"$model\"" <<< "${served// /}" || fail "$id serves $served, not $model"
  note "$id healthy after $(( $(date +%s) - t0 ))s, serving $model on port $port"
}

# What a healthy engine executes, one line for deploy.sh's record:
#   started <id> <process_start_time_seconds> <CUDA_VISIBLE_DEVICES> <command line>
# read from the recorded process (its /proc environment and command line) and from the engine answering on the
# port (its /metrics start time), which must be that process: its /proc start time in epoch seconds (btime +
# starttime / CLK_TCK, the formula of Prometheus's process collector) agrees to within rounding.
cmd_started() {
  local id="$1" port="$2" pidf="$RB_RUN/engine.$1.pid" pid start own cuda ka cmdline
  pid=$(pid_of "$pidf")
  alive "$pid" || fail "$id is not running"
  start=$(curl -sf -m 10 "http://127.0.0.1:$port/metrics" \
          | awk '$1 == "process_start_time_seconds" {v = $2; n++} END {if (n != 1) exit 1; print v}') \
    || fail "$id: /metrics on port $port has no single process_start_time_seconds"
  own=$(awk -v t="$(start_time "$pid")" -v hz="$(getconf CLK_TCK)" '/^btime/ {printf "%.3f", $2 + t / hz}' /proc/stat)
  awk -v a="$start" -v b="$own" 'BEGIN {d = a - b; exit !(d < 0.05 && d > -0.05)}' \
    || fail "$id: the engine on port $port started at $start, its recorded pid $pid at $own: another process answers"
  cuda=$(tr '\0' '\n' < "/proc/$pid/environ" | sed -n 's/^CUDA_VISIBLE_DEVICES=//p')
  [ -n "$cuda" ] || fail "$id: pid $pid has no CUDA_VISIBLE_DEVICES"
  # the keep-alive the process runs with, checked and recorded ahead of its command line (tools/engines.py checks
  # the record's command line ends with the planned arguments; the parity gate compares the whole of it)
  ka=$(tr '\0' '\n' < "/proc/$pid/environ" | sed -n 's/^VLLM_HTTP_TIMEOUT_KEEP_ALIVE=//p')
  [ "$ka" = "$ENGINE_KEEP_ALIVE_S" ] || fail "$id: pid $pid runs with VLLM_HTTP_TIMEOUT_KEEP_ALIVE='$ka', not $ENGINE_KEEP_ALIVE_S"
  cmdline=$(tr '\0' ' ' < "/proc/$pid/cmdline")
  printf 'started\t%s\t%s\t%s\t%s\n' "$id" "$start" "$cuda" "VLLM_HTTP_TIMEOUT_KEEP_ALIVE=$ka ${cmdline% }"
}

cmd_stop() {
  stop_group "$RB_RUN/engine.$1.pid" "$1"
}

cmd_gpus_free() {
  local gpus="$1" busy p
  busy=$(busy_pids "$gpus") || fail "nvidia-smi failed"
  for p in $busy; do
    note "GPUs $gpus: killing leftover compute pid $p ($(ps -o comm= -p "$p" 2>/dev/null))"
    kill -KILL "$p" 2>/dev/null || true
  done
  [ -z "$busy" ] || sleep 5
  busy=$(busy_pids "$gpus") || fail "nvidia-smi failed"
  [ -z "$busy" ] || fail "GPUs $gpus still in use by pid(s) $(echo "$busy" | tr '\n' ' ')"
  note "GPUs $gpus idle"
}

hub_pins_ok() {      # hub_pins_ok <spec file>: offline, a loader naming only the repository reads refs/main
  local spec_line repo rev ref
  [ -s "$1" ] || fail "no verified Hub specs at $1 (setup.sh --role scheduler did not finish)"
  while read -r spec_line; do
    repo=${spec_line%@*}; rev=${spec_line##*@}
    ref="$RB_HF_HOME/hub/models--${repo//\//--}/refs/main"
    [ "$(cat "$ref" 2>/dev/null)" = "$rev" ] || fail "$ref is '$(cat "$ref" 2>/dev/null)', not the pinned $rev"
    [ -d "$RB_HF_HOME/hub/models--${repo//\//--}/snapshots/$rev" ] || fail "no snapshot of $spec_line"
  done < "$1"
}

cmd_sched_start() {
  local run="$1" tree="$2" art="$3" logdir="$4" host="$5" port="$6" threads="$7" conns="$8" timeout="$9"
  local pidf="$RB_RUN/scheduler.pid" log where
  mountpoint -q "$MYDATA" || fail "$MYDATA is not mounted"
  venv_complete "$SCHED_VENV"
  [ -f "$AP_DIR/simple_cluster_router.py" ] || fail "no AvengersPro checkout at $AP_DIR (setup.sh --role scheduler)"
  [ -f "$BR_DIR/hybrid_llm/pair_ranker/ranker.py" ] || fail "no best-route-llm checkout at $BR_DIR (setup.sh --role scheduler)"
  # relative paths are relative to the login directory (where ssh starts), for the checks and the scheduler alike
  case "$art" in /*) ;; *) art="$HOME/$art" ;; esac
  case "$logdir" in /*) ;; *) logdir="$HOME/$logdir" ;; esac
  [ -d "$art" ] || fail "artifact root $art does not exist"
  tree=$(cd "$tree" && pwd -P) || fail "no tree at $2"
  # the venv must have been built from this tree and its current pyproject (setup.sh records both), and
  # its editable install must resolve there (probed from /: `python -c` puts the cwd first on sys.path)
  local spec
  spec="complete scheduler $tree pyproject.sha256=$(sha256sum "$tree/pyproject.toml" | cut -c1-16)"
  [ "$(cat "$SCHED_VENV/rb-setup.spec")" = "$spec" ] \
    || fail "$SCHED_VENV was built as '$(cat "$SCHED_VENV/rb-setup.spec")', not '$spec' (rerun setup.sh --role scheduler from $tree)"
  where=$(cd / && env -i HOME="$HOME" PATH="$RB_PATH" "$SCHED_VENV/bin/python" -c \
    'import os, routebalance; print(os.path.realpath(routebalance.__file__))') || fail "the scheduler venv cannot import routebalance"
  [[ "$where" == "$tree"/* ]] || fail "the scheduler venv imports routebalance from $where, not $tree"
  hub_pins_ok "$SCHED_VENV/rb-setup.hub"
  alive "$(pid_of "$pidf")" && fail "a scheduler is already running (pid $(pid_of "$pidf")); stop_scheduler.sh first"
  port_free "$port" || fail "port $port is in use"
  if [ -e "$logdir" ] && [ -n "$(ls -A "$logdir")" ]; then fail "log dir $logdir is not empty; use a new one per run"; fi
  mkdir -p "$logdir" "$RB_LOGS"
  rm -f "$pidf"                        # a stale record (that process is gone), replaced by the new one below
  log="$RB_LOGS/scheduler.$(date -u +%Y%m%dT%H%M%SZ).log"
  local argv=(--fleet "$run/fleet.json" --hosts "$run/hosts.json" --config "$run/scheduler.json"
              --artifacts "$art" --avengers-repo "$AP_DIR" --best-route-repo "$BR_DIR" --host "$host" --port "$port"
              --request-timeout-s "$timeout" --max-connections "$conns" --log-dir "$logdir")
  [ "$threads" = none ] || argv+=(--torch-threads "$threads")   # none: torch's default thread count (v3)
  printf '%s\n' "CUDA_VISIBLE_DEVICES= $SCHED_VENV/bin/python -u -m routebalance.server ${argv[*]}" > "$run/scheduler.cmd"
  # the open-file limit: the scheduler raises its own soft limit at start as v3's did (set_ulimit, 65535)
  # shellcheck disable=SC2016,SC2094
  (cd "$tree" && setsid --fork bash -c "$RECORD_AND_EXEC"'; exec "$@"' _ "$pidf" "$log" "$BOOT_ID" \
    env -i HOME="$HOME" PATH="$RB_PATH" LANG=C.UTF-8 CUDA_VISIBLE_DEVICES= HF_HOME="$RB_HF_HOME" HF_HUB_OFFLINE=1 \
    TIKTOKEN_CACHE_DIR="$RB_TIKTOKEN" \
    "$SCHED_VENV/bin/python" -u -m routebalance.server "${argv[@]}" > "$log" 2>&1 < /dev/null)
  local i
  for i in $(seq 1 20); do [ -s "$pidf" ] && break; sleep 0.25; done
  alive "$(pid_of "$pidf")" || fail "the scheduler did not start; log $log: $(tail -5 "$log" 2>/dev/null | tr '\n' ' ')"
  note "scheduler launched: pid $(pid_of "$pidf") port $port log $log"   # its open-file limit: the manifest
}

cmd_sched_wait() {
  local port="$1" timeout="$2" pidf="$RB_RUN/scheduler.pid" t0 pid log
  t0=$(date +%s); pid=$(pid_of "$pidf"); log=$(log_of "$pidf")
  alive "$pid" || fail "the scheduler is not running"
  while ! curl -sf -m 5 "http://127.0.0.1:$port/health" > /dev/null; do
    alive "$pid" || fail "the scheduler exited while starting; log $log:
$(tail -30 "$log")"
    [ $(( $(date +%s) - t0 )) -lt "$timeout" ] || fail "the scheduler is not healthy after ${timeout}s; log $log:
$(tail -30 "$log")"
    sleep 5
  done
  curl -sf -m 10 "http://127.0.0.1:$port/v1/config" | env -i PATH="$RB_PATH" "$SCHED_VENV/bin/python" -c \
    'import json, sys; d = json.load(sys.stdin); print("DIGESTS", d["config_digest"], d["fleet_digest"])' \
    || fail "GET /v1/config failed"
  note "scheduler healthy after $(( $(date +%s) - t0 ))s"
}

cmd_sched_stop() {
  stop_group "$RB_RUN/scheduler.pid" scheduler
}

rb_ctl() {
  local cmd="$1"; shift
  case "$cmd" in
    preflight) [ $# -eq 2 ] || fail "usage: preflight <gpu_class> <gpus_on_node>"; cmd_preflight "$@" ;;
    launch) [ $# -gt 5 ] || fail "usage: launch <id> <port> <gpus> <model> <revision> <argv...>"; cmd_launch "$@" ;;
    wait) [ $# -eq 4 ] || fail "usage: wait <id> <port> <model> <timeout_s>"; cmd_wait "$@" ;;
    started) [ $# -eq 2 ] || fail "usage: started <id> <port>"; cmd_started "$@" ;;
    stop) [ $# -eq 1 ] || fail "usage: stop <id>"; cmd_stop "$@" ;;
    gpus-free) [ $# -eq 1 ] || fail "usage: gpus-free <gpus>"; cmd_gpus_free "$@" ;;
    sched-start) [ $# -eq 9 ] || fail "usage: sched-start <run_dir> <tree> <artifacts> <log_dir> <host> <port> <threads> <conns> <timeout>"
                 cmd_sched_start "$@" ;;
    sched-wait) [ $# -eq 2 ] || fail "usage: sched-wait <port> <timeout_s>"; cmd_sched_wait "$@" ;;
    sched-stop) [ $# -eq 0 ] || fail "usage: sched-stop"; cmd_sched_stop ;;
    *) fail "unknown command $cmd" ;;
  esac
}

if [ -z "${BASH_SOURCE[0]:-}" ] || [ "${BASH_SOURCE[0]}" = "$0" ]; then   # streamed (bash -s) or run; tests source it
  rb_ctl "$@"
fi
