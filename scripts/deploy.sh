#!/usr/bin/env bash
# Launch a fleet's vLLM engines, or the scheduler, on the cluster nodes (run from the operator machine).
#
#   deploy.sh engines   --fleet F --hosts H --engine-args E --record FILE
#                       [--health-timeout-s S] [--ssh-user U]
#   deploy.sh scheduler --fleet F --hosts H --node N --config C --launch L --artifacts DIR --tree DIR
#                       --log-dir DIR --record FILE [--health-timeout-s S] [--ssh-user U]
#
# engines: every node is preflighted (mount, venv, GPU class/count, driver), then every instance is launched
#   with exactly the plan's arguments (python -m tools.engine_plan engines ...) and waited for: /health, then
#   GET /v1/models must name the tier's model. Any failure exits 1 and leaves the other engines as they are
#   (stop them with stop_engines.sh). A running engine, a busy port or a busy GPU is refused, never killed.
# scheduler: python -m fmrlb.server on node N with the fleet, hosts and scheduler config copied to
#   the node (sha256-checked), and the scheduler's launch parameters from L (configs/scheduler_launch.example.json
#   documents them; OWNER-MUST-SET values are refused). --artifacts, --tree and --log-dir are paths on the
#   node (--tree: the repository copy the scheduler venv was installed from; --log-dir must be new or empty).
#   Health is /health; the served config and fleet digests must equal the local ones.
# --record FILE receives the rendered plan (engines) or the launch line and digests (scheduler). For engines it
#   then gets one "started" line per engine once all are healthy: its process start time, CUDA_VISIBLE_DEVICES
#   and command line as the node reports them (node_ctl.sh started). tools/engines.py checks every cell's engines
#   against it: an engine not launched by that deploy run is refused.
# --health-timeout-s defaults to 1800 (engines: loading 72B weights) and 600 (scheduler).

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib_operator.sh
. "$HERE/lib_operator.sh"
set -euo pipefail

MODE="${1:-}"; [ $# -gt 0 ] && shift
FLEET="" HOSTS="" EA="" RECORD="" NODE="" CONFIG="" LAUNCH="" ARTIFACTS="" TREE="" LOGDIR="" HEALTH=""
while [ $# -gt 0 ]; do
  [ $# -ge 2 ] || die "$1 needs a value"
  case "$1" in
    --fleet) FLEET="$2" ;; --hosts) HOSTS="$2" ;; --engine-args) EA="$2" ;; --record) RECORD="$2" ;;
    --node) NODE="$2" ;; --config) CONFIG="$2" ;; --launch) LAUNCH="$2" ;; --artifacts) ARTIFACTS="$2" ;;
    --tree) TREE="$2" ;; --log-dir) LOGDIR="$2" ;; --health-timeout-s) HEALTH="$2" ;; --ssh-user) SSH_USER="$2" ;;
    *) die "unknown argument $1" ;;
  esac
  shift 2
done
need() { while [ $# -gt 0 ]; do [ -n "$1" ] || die "$MODE needs $2"; shift 2; done; }
require_py
existing FLEET HOSTS EA CONFIG LAUNCH   # local files (--artifacts, --tree, --log-dir are paths on the node)
absolute RECORD
# shellcheck source=scripts/node_layout.sh
. "$HERE/node_layout.sh"     # the node paths ($RB_RUN) the remote side uses

deploy_engines() {
  need "$FLEET" --fleet "$HOSTS" --hosts "$EA" --engine-args "$RECORD" --record
  local timeout="${HEALTH:-1800}" lines nodes
  [[ "$timeout" =~ ^[0-9]+$ ]] || die "--health-timeout-s must be whole seconds"
  lines=$(plan engines --fleet "$FLEET" --hosts "$HOSTS" --engine-args "$EA") || die "invalid plan"
  nodes=$(plan nodes --fleet "$FLEET" --hosts "$HOSTS") || die "invalid fleet/hosts"
  mkdir -p "$(dirname "$RECORD")"
  { echo "# deploy.sh engines $(date -u +%FT%TZ) fleet=$FLEET engine_args=$EA"
    echo "# fleet sha256 $(sha256sum "$FLEET" | cut -c1-64)"
    echo "# engine args sha256 $(sha256sum "$EA" | cut -c1-64)"
    echo "$lines"; } > "$RECORD"
  say "plan: $(wc -l <<< "$lines") engine(s) on $(wc -l <<< "$nodes") node(s); recorded in $RECORD"

  local node addr cls n
  while IFS=$'\t' read -r node addr _ cls n; do
    rb_ctl 120 "$addr" preflight "$cls" "$n" || die "preflight failed on $node ($addr)"
  done <<< "$nodes"

  local id port gpus model argv rev
  while IFS=$'\t' read -r id node addr port gpus cls n model argv; do
    read -r -a args <<< "$argv"
    rev="${args[3]}"                               # plan argv: --model M --revision R --port P ...
    [ "${args[2]}" = --revision ] || die "unexpected plan argv for $id"
    rb_ctl 120 "$addr" launch "$id" "$port" "$gpus" "$model" "$rev" "${args[@]}" || die "launch of $id failed"
  done <<< "$lines"

  say "waiting for $(wc -l <<< "$lines") engine(s), up to ${timeout}s each"
  local pids=() ids=() failed=()
  while IFS=$'\t' read -r id node addr port gpus cls n model argv; do
    rb_ctl $((timeout + 120)) "$addr" wait "$id" "$port" "$model" "$timeout" &
    pids+=("$!"); ids+=("$id")
  done <<< "$lines"
  local i
  for i in "${!pids[@]}"; do wait "${pids[$i]}" || failed+=("${ids[$i]}"); done
  [ "${#failed[@]}" -eq 0 ] || die "not healthy: ${failed[*]} (the others keep running; stop_engines.sh stops all)"
  local started
  while IFS=$'\t' read -r id node addr port gpus cls n model argv; do     # what each engine executes
    started=$(rb_ctl 60 "$addr" started "$id" "$port") || die "cannot read what $id executes"
    echo "$started" >> "$RECORD"
  done <<< "$lines"
  say "ALL $(wc -l <<< "$lines") ENGINES HEALTHY; each one's start time, GPUs and command line recorded in $RECORD"
}

deploy_scheduler() {
  need "$FLEET" --fleet "$HOSTS" --hosts "$NODE" --node "$CONFIG" --config "$LAUNCH" --launch \
       "$ARTIFACTS" --artifacts "$TREE" --tree "$LOGDIR" --log-dir "$RECORD" --record
  local timeout="${HEALTH:-600}" launch host port threads conns rtimeout addr digests want run f
  [[ "$timeout" =~ ^[0-9]+$ ]] || die "--health-timeout-s must be whole seconds"
  launch=$(plan launch --launch "$LAUNCH") || die "invalid launch file $LAUNCH"
  IFS=$'\t' read -r host port threads conns rtimeout <<< "$launch"
  want=$(cd "$RB_REPO" && "$PY" -c 'import sys
from fmrlb.config import config_digest, fleet_digest, load_fleet, load_scheduler
print(config_digest(load_scheduler(sys.argv[3])), fleet_digest(load_fleet(sys.argv[1], sys.argv[2])))' \
    "$FLEET" "$HOSTS" "$CONFIG") || die "invalid fleet, hosts or scheduler config"
  addr=$(plan nodes --fleet "$FLEET" --hosts "$HOSTS" | awk -F'\t' -v n="$NODE" '$1 == n {print $2}') \
    || die "invalid fleet/hosts"
  [ -n "$addr" ] || die "node $NODE serves no instance of $FLEET (the scheduler runs on a fleet node)"

  run="$RB_RUN/scheduler.$(date -u +%Y%m%dT%H%M%SZ)"
  rb_ssh 60 "$addr" mkdir -p "$run" || die "cannot create $run on $NODE"
  for f in "$FLEET:fleet.json" "$HOSTS:hosts.json" "$CONFIG:scheduler.json" "$LAUNCH:launch.json"; do
    timeout 120 scp -q "${SSH_OPTS[@]}" "${f%%:*}" "$(ssh_target "$addr"):$run/${f#*:}" || die "copy ${f%%:*} to $NODE"
    [ "$(rb_ssh 60 "$addr" sha256sum "$run/${f#*:}" | cut -c1-64)" = "$(sha256sum "${f%%:*}" | cut -c1-64)" ] \
      || die "$run/${f#*:} on $NODE differs from ${f%%:*}"
  done
  rb_ctl 180 "$addr" sched-start "$run" "$TREE" "$ARTIFACTS" "$LOGDIR" "$host" "$port" "$threads" "$conns" "$rtimeout" \
    || die "scheduler start failed on $NODE"
  digests=$(rb_ctl $((timeout + 60)) "$addr" sched-wait "$port" "$timeout") || die "the scheduler did not become healthy"
  echo "$digests"
  [ "$(grep '^DIGESTS ' <<< "$digests" | cut -d' ' -f2-)" = "$want" ] \
    || die "the scheduler serves digests '$(grep '^DIGESTS ' <<< "$digests")', expected '$want'"
  mkdir -p "$(dirname "$RECORD")"
  { echo "# deploy.sh scheduler $(date -u +%FT%TZ) node=$NODE run_dir=$run"
    echo "launch	$launch"
    echo "digests	$want"
    rb_ssh 60 "$addr" cat "$run/scheduler.cmd"; } > "$RECORD"
  say "SCHEDULER HEALTHY on $NODE:$port (config/fleet digests $want); recorded in $RECORD"
}

case "$MODE" in
  engines) deploy_engines ;;
  scheduler) deploy_scheduler ;;
  *) die "usage: deploy.sh engines|scheduler ... (see the header of $0)" ;;
esac
