#!/usr/bin/env bash
# Run the cells of some lease rows in the plan's order, resuming: a cell already published is never run again
# (valid ones are done; invalid or ungated ones are reported and left for the owner to account for or retire).
#
#   scripts/run_scope.sh --render DIR --rows G1,W2,... --max-invalid N [--restart "CMD"] \
#                        --scheduler URL --scheduler-log-dir DIR --fleet FLEET.json --hosts HOSTS.json \
#                        --engines-record FILE --data-root DIR --cells-root DIR --mirror DEST
#
# --engines-record is the --record file of the deploy.sh engines run that launched the fleet's engines (D48).
#
# The rows are run in the plan's order whatever order they are given in; every requested cell must belong to
# the fleet given (a topology or model-family change is a redeploy, then a run of its own rows). Before every
# cell the scheduler's code on disk must be the code it started with (its manifest reports both): after an
# rsync of the code the scheduler must be restarted. With --restart the command is run (it must restart the
# scheduler and return once it is launched) and the check repeated; without it the run stops.
# A cell that could not run (exit 2) or whose mirror failed (exit 3) stops the run; an invalid cell (exit 1)
# counts toward --max-invalid and the run stops when more than that many have been seen. Before the first cell and
# after the last, the bookkeeping directories (_attempts, _groups, _retired) are copied to the second location
# and every published cell of the rows is copied and proven identical there (a copy that failed on an earlier
# run is completed; a mirror copy of another run of the cell, e.g. a retired one, is never written over: the
# run stops). The run succeeds
# only when every rendered cell of the rows is done (published, valid, gated) and every group of theirs whose
# parity ran has passed it; a requested row with a pending block is refused before any run. The manifest of the
# whole render is printed at the end.
set -euo pipefail

[ -n "${PY:-}" ] || { echo "set PY to the scheduler venv python" >&2; exit 2; }   # env-ok: the interpreter path, not a setting
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

usage() { echo "usage: $0 --render DIR --rows R1,R2 --max-invalid N [--restart CMD] --scheduler URL" \
               "--scheduler-log-dir DIR --fleet F --hosts H --engines-record FILE --data-root DIR --cells-root DIR" \
               "--mirror DEST" >&2; exit 2; }
RENDER= ROWS= MAXINV= RESTART= URL= LOGDIR= FLEET= HOSTS= ENGINES= DATA= ROOT= MIRROR=
while [ $# -gt 0 ]; do
  [ $# -ge 2 ] || usage          # every flag takes a value
  case "$1" in
    --render) RENDER=$2 ;; --rows) ROWS=$2 ;; --max-invalid) MAXINV=$2 ;; --restart) RESTART=$2 ;;
    --scheduler) URL=$2 ;; --scheduler-log-dir) LOGDIR=$2 ;; --fleet) FLEET=$2 ;; --hosts) HOSTS=$2 ;;
    --data-root) DATA=$2 ;; --cells-root) ROOT=$2 ;; --mirror) MIRROR=$2 ;; --engines-record) ENGINES=$2 ;;
    *) usage ;;
  esac
  shift 2
done
for v in RENDER ROWS MAXINV URL LOGDIR FLEET HOSTS ENGINES DATA ROOT MIRROR; do [ -n "${!v}" ] || usage; done
case "$MAXINV" in ''|*[!0-9]*) usage ;; esac

source "$REPO/scripts/mirror.sh"
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
drv() { "$PY" -m tools.cell_driver "$@"; }

mirror_root() {  # the bookkeeping directories, then every published cell of the rows: never over another run
  local d published cell
  for d in "$ROOT"/_*/; do
    [ -d "$d" ] || continue
    rsync -a --checksum "$d" "$MIRROR/$(basename "$d")/" || return 1
  done
  published=$(drv published --render "$RENDER" --rows "$ROWS" --cells-root "$ROOT") || return 1
  while read -r cell; do
    [ -n "$cell" ] || continue
    mirror_cell "$ROOT" "$MIRROR" "$cell" || return 1
  done <<< "$published"
}

scheduler_current() {
  drv code --scheduler "$URL" && return 0
  [ -n "$RESTART" ] || { echo "$(ts) [run_scope] the scheduler runs stale code; restart it (or pass --restart)"; return 1; }
  echo "$(ts) [run_scope] restarting the scheduler: $RESTART"
  bash -c "$RESTART" || { echo "$(ts) [run_scope] the restart command failed"; return 1; }
  for _ in $(seq 1 120); do
    curl -sf "$URL/health" >/dev/null 2>&1 && break
    sleep 2
  done
  drv code --scheduler "$URL"
}

mkdir -p "$ROOT"
mirror_root || { echo "$(ts) [run_scope] the second location at $MIRROR does not match $ROOT"; exit 3; }
PLAN=$(drv plan --render "$RENDER" --rows "$ROWS" --cells-root "$ROOT" --fleet "$FLEET" --hosts "$HOSTS") \
  || { echo "$(ts) [run_scope] the plan could not be read (see above)"; exit 2; }
CELLS=()
[ -z "$PLAN" ] || mapfile -t CELLS <<< "$PLAN"
echo "$(ts) [run_scope] ${#CELLS[@]} cells to run in rows $ROWS"
INVALID=0
for cell in "${CELLS[@]}"; do
  [ -n "$cell" ] || continue
  scheduler_current || exit 2
  set +e
  scripts/run_cell.sh --render "$RENDER" --cell "$cell" --scheduler "$URL" --scheduler-log-dir "$LOGDIR" \
      --fleet "$FLEET" --hosts "$HOSTS" --engines-record "$ENGINES" --data-root "$DATA" --cells-root "$ROOT" \
      --mirror "$MIRROR"
  rc=$?
  set -e
  case "$rc" in
    0) ;;
    1) INVALID=$((INVALID + 1))
       echo "$(ts) [run_scope] $cell INVALID ($INVALID so far, at most $MAXINV)"
       [ "$INVALID" -le "$MAXINV" ] || { echo "$(ts) [run_scope] stopping: more than $MAXINV invalid cells"; exit 1; } ;;
    *) echo "$(ts) [run_scope] stopping: $cell exited $rc"; exit "$rc" ;;
  esac
done
ROW_GROUPS=$(drv groups --render "$RENDER" --rows "$ROWS") || { echo "$(ts) [run_scope] the groups could not be read"; exit 2; }
for group in $ROW_GROUPS; do        # a group completed by a cell whose driver stopped early still gets its parity
  set +e
  drv parity --render "$RENDER" --cells-root "$ROOT" --group "$group"
  set -e
done
mirror_root || { echo "$(ts) [run_scope] the second location at $MIRROR does not match $ROOT"; exit 3; }
set +e
"$PY" -m analysis.manifest --scope "$RENDER/scope.json" --cells "$ROOT"
drv status --render "$RENDER" --rows "$ROWS" --cells-root "$ROOT"
DONE=$?
set -e
echo "$(ts) [run_scope] rows $ROWS: $INVALID cells invalid in this run"
[ "$DONE" = 0 ] || { echo "$(ts) [run_scope] rows $ROWS are not done (listed above)"; exit 1; }
echo "$(ts) [run_scope] rows $ROWS done"
