#!/usr/bin/env bash
# Restart the scheduler on its own node, as a fresh process, before a cell.
#
#   scripts/restart_scheduler.sh RUN_DIR LOG_DIR
#
# RUN_DIR is the run directory deploy.sh scheduler made on this node (its fleet.json, hosts.json, scheduler.json,
# launch.json and scheduler.cmd, sha256-checked when copied); LOG_DIR must be new. The running scheduler is stopped
# and a new one started from RUN_DIR with the same launch values (launch.json), the same artifact root (as
# scheduler.cmd recorded it), this tree, and LOG_DIR. Returns once it is healthy and serves the digests of RUN_DIR's
# configuration and fleet; any failure exits non-zero.
set -euo pipefail
[ $# -eq 2 ] || { echo "usage: $0 RUN_DIR LOG_DIR" >&2; exit 2; }
run=$1 logdir=$2
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -n "${PY:-}" ] || { echo "set PY to the scheduler venv python" >&2; exit 2; }   # env-ok: the interpreter path, not a setting
for f in fleet.json hosts.json scheduler.json launch.json scheduler.cmd; do
  [ -f "$run/$f" ] || { echo "$run has no $f (not a deploy.sh scheduler run directory)" >&2; exit 2; }
done
[ ! -e "$logdir" ] || { echo "log dir $logdir exists; every restart gets a new one" >&2; exit 2; }
ctl() { cat "$REPO/scripts/node_layout.sh" "$REPO/scripts/node_ctl.sh" | bash -s -- "$@"; }
read -r host port threads conns rtimeout < <("$PY" -c 'import json, sys
l = json.load(open(sys.argv[1]))
print(l["host"], l["port"], "none" if l.get("torch_threads") is None else l["torch_threads"], l["max_connections"],
      l["request_timeout_s"])' "$run/launch.json")
art=$(grep -o -- '--artifacts [^ ]*' "$run/scheduler.cmd" | cut -d' ' -f2)
[ -n "$art" ] || { echo "$run/scheduler.cmd records no --artifacts" >&2; exit 2; }
want=$(cd "$REPO" && "$PY" -c 'import sys
from routebalance.config import config_digest, fleet_digest, load_fleet, load_scheduler
print(config_digest(load_scheduler(sys.argv[3])), fleet_digest(load_fleet(sys.argv[1], sys.argv[2])))' \
  "$run/fleet.json" "$run/hosts.json" "$run/scheduler.json")
ctl sched-stop
ctl sched-start "$run" "$REPO" "$art" "$logdir" "$host" "$port" "$threads" "$conns" "$rtimeout"
got=$(ctl sched-wait "$port" 600 | grep '^DIGESTS ' | cut -d' ' -f2-)
[ "$got" = "$want" ] || { echo "the restarted scheduler serves digests '$got', expected '$want'" >&2; exit 1; }
echo "scheduler restarted: log dir $logdir, digests $got"
