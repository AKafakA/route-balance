#!/usr/bin/env bash
# Run one campaign cell end to end, fail-loud. Every step is a subcommand of tools/cell_driver.py or a gate;
# the cell directory layout is documented there.
#
#   scripts/run_cell.sh --render DIR --cell CELL_ID --scheduler URL --scheduler-log-dir DIR \
#                       --fleet FLEET.json --hosts HOSTS.json --engines-record FILE --data-root DIR \
#                       --cells-root DIR --mirror DEST
#
#   --render             a campaign render (python -m tools.campaign render): scope.json, configs/, plan.json
#   --scheduler-log-dir  the running scheduler's --log-dir (decisions.jsonl, latency.<digest>.jsonl)
#   --engines-record     the --record file of the deploy.sh engines run that launched the fleet (D48)
#   --cells-root         the first location (this node); DEST (rsync syntax, local or host:path) the second
#
# Order: drain the engines and the scheduler; POST the cell's configuration (every cell: it resets the
# dispatcher's RNG, the LRU and lowest_latency state) and check the digests it returns; preflight; warm the
# engines up directly (never through the scheduler) and drain again; read every engine as it reports itself
# (start time, KV cache, model, context) with its launch command; check the scheduler's code and launch
# values against the scope; snapshot stats and manifest once the logs are settled; run the bench runner; snapshot
# again and cut this cell's window out of the decision log (and the latency sweep's log); gates arm and
# zero_call; the verdict; publish the attempt as the cell; mirror it; parity when the cell completes its group.
# The mirror root (DEST) must exist. This script's own log is <cells-root>/_attempts/<cell>.<stamp>/driver.log.
#
# Exit: 0 the cell is valid and passed its gates; 1 it ran and is invalid, or a gate failed (published, kept);
#       2 it could not run (no record; the attempt stays under <cells-root>/_attempts); 3 the mirror failed.
set -euo pipefail

[ -n "${PY:-}" ] || { echo "set PY to the scheduler venv python" >&2; exit 2; }   # env-ok: the interpreter path, not a setting
PY=$(command -v "$PY") || { echo "PY=$PY is not an executable" >&2; exit 2; }   # one absolute interpreter for all steps
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

usage() { echo "usage: $0 --render DIR --cell ID --scheduler URL --scheduler-log-dir DIR --fleet F --hosts H" \
               "--engines-record FILE --data-root DIR --cells-root DIR --mirror DEST" >&2; exit 2; }
RENDER= CELL= URL= LOGDIR= FLEET= HOSTS= ENGINES= DATA= ROOT= MIRROR=
RC=2                    # fail()'s exit code: a cell that could not run (never taken from the caller's environment)
while [ $# -gt 0 ]; do
  [ $# -ge 2 ] || usage          # every flag takes a value
  case "$1" in
    --render) RENDER=$2 ;; --cell) CELL=$2 ;; --scheduler) URL=$2 ;; --scheduler-log-dir) LOGDIR=$2 ;;
    --fleet) FLEET=$2 ;; --hosts) HOSTS=$2 ;; --data-root) DATA=$2 ;; --cells-root) ROOT=$2 ;; --mirror) MIRROR=$2 ;;
    --engines-record) ENGINES=$2 ;;
    *) usage ;;
  esac
  shift 2
done
for v in RENDER CELL URL LOGDIR FLEET HOSTS ENGINES DATA ROOT MIRROR; do [ -n "${!v}" ] || usage; done

source "$REPO/scripts/mirror.sh"
source "$REPO/scripts/node_layout.sh"   # RB_HF_HOME: the Hub snapshots setup.sh pinned (the runner's tokenizer)
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
drv() { "$PY" -m tools.cell_driver "$@"; }
fail() { echo "$(ts) [run_cell] $CELL: $*" >&2; exit "$RC"; }

# ---- the attempt directory, the cell's configuration and group
PREP=$(drv prepare --render "$RENDER" --cell "$CELL" --cells-root "$ROOT" --fleet "$FLEET" --hosts "$HOSTS") \
  || fail "prepare refused"
eval "$PREP"            # ATTEMPT CONFIG_ID CONFIG_DIGEST FLEET_DIGEST GROUP SWEEP INSTANCES
mkdir -p "$ATTEMPT/gates"
exec > >(tee -a "$(dirname "$ATTEMPT")/driver.log") 2>&1      # beside the attempt: it stays under _attempts
echo "$(ts) [run_cell] $CELL: config $CONFIG_ID ($CONFIG_DIGEST), group $GROUP, attempt $ATTEMPT"

# ---- 1. drain  2. POST the configuration  3. preflight
drv drain --render "$RENDER" --cell "$CELL" --fleet "$FLEET" --hosts "$HOSTS" --scheduler "$URL" || fail "drain"
drv fresh --scheduler "$URL" || fail "the scheduler is not a fresh process (D132: run_scope.sh --fresh-scheduler)"
drv post-config --scheduler "$URL" --attempt "$ATTEMPT" --config-digest "$CONFIG_DIGEST" \
    --fleet-digest "$FLEET_DIGEST" || fail "POST /v1/config"
"$PY" -m gates.preflight --scheduler "$URL" --fleet "$FLEET" --hosts "$HOSTS" \
    --expect-config-digest "$CONFIG_DIGEST" --expect-fleet-digest "$FLEET_DIGEST" > "$ATTEMPT/gates/preflight.log" 2>&1 \
  || { cat "$ATTEMPT/gates/preflight.log"; fail "preflight failed"; }

# ---- 4. the engines as they report themselves (D48), with their serving counters, BEFORE the warm-up, so the
#         traffic check covers the reset, its proof and the whole run (D132); then the warm-up straight to the
#         engines (the proven cache reset, D128), then drain
drv engines --render "$RENDER" --cell "$CELL" --fleet "$FLEET" --hosts "$HOSTS" --engines-record "$ENGINES" \
    --out "$ATTEMPT/engines_before.json" || fail "the engines could not be read"
drv warmup --render "$RENDER" --cell "$CELL" --fleet "$FLEET" --hosts "$HOSTS" --attempt "$ATTEMPT" || fail "warm-up"

# ---- 5. stats and manifest before (settled logs), the scheduler's code and launch values
drv before --scheduler "$URL" --attempt "$ATTEMPT" --log-dir "$LOGDIR" --config-digest "$CONFIG_DIGEST" \
    --sweep "$SWEEP" || fail "snapshot before"
drv check-scheduler --render "$RENDER" --attempt "$ATTEMPT" || fail "restart the scheduler (see above)"

# ---- 6. the bench runner (writes <cell_id>.cell.json into the attempt directory); offline, from the pinned
#         snapshots, in a clean environment like the scheduler and the engines (node_ctl.sh)
set +e
env -i HOME="$HOME" PATH="$RB_PATH" LANG=C.UTF-8 HF_HOME="$RB_HF_HOME" HF_HUB_OFFLINE=1 \
    "$PY" -m bench.runner --scope "$RENDER/scope.json" --cell "$CELL" --scheduler-url "$URL" --data-root "$DATA" \
    --out-dir "$ATTEMPT"
BENCH=$?
set -e
[ -f "$ATTEMPT/$CELL.cell.json" ] || fail "the runner wrote no record (exit $BENCH)"

# ---- 7. stats and manifest after, and this cell's slice of the logs; the engines again (a restart shows)
drv after --scheduler "$URL" --attempt "$ATTEMPT" --config-digest "$CONFIG_DIGEST" \
  || { echo "$(ts) [run_cell] $CELL: the logs could not be sliced; the cell is kept but cannot pass its gates"; }
drv engines --render "$RENDER" --cell "$CELL" --fleet "$FLEET" --hosts "$HOSTS" --engines-record "$ENGINES" \
    --out "$ATTEMPT/engines_after.json" \
  || { echo "$(ts) [run_cell] $CELL: the engines could not be read after the cell; the cell is kept and invalid"; }

# ---- 8. gates on the cell, the verdict, publish
GATE_ARGS=()
for g in arm zero_call; do
  set +e
  if [ "$g" = arm ]; then
    "$PY" -m gates.arm --cell "$ATTEMPT" --config "$ATTEMPT/config.json" > "$ATTEMPT/gates/$g.log" 2>&1
  else
    "$PY" -m gates.zero_call --cell "$ATTEMPT" > "$ATTEMPT/gates/$g.log" 2>&1
  fi
  rc=$?
  set -e
  echo "$(ts) [run_cell] gate $g: exit $rc"
  GATE_ARGS+=(--gate "$g=$rc")
done
set +e
drv verdict --render "$RENDER" --attempt "$ATTEMPT" "${GATE_ARGS[@]}"
VALID=$?
set -e
CELLDIR=$(drv publish --attempt "$ATTEMPT" --cells-root "$ROOT") || fail "publish"
echo "$(ts) [run_cell] $CELL published at $CELLDIR"

# ---- 9. the second location: copy, then prove it identical (checksums, nothing extra on the mirror). The cell
#         directory has no writer left (this script logs beside the attempt, under _attempts). A mirror that
#         already holds another run of this cell (a retired one) is never written over.
mirror_cell "$ROOT" "$MIRROR" "$CELL" || { RC=3 fail "the second location refused or failed (see above)"; }
echo "$(ts) [run_cell] $CELL mirrored to $MIRROR/$CELL"

# ---- 10. parity once the cell completes its group (every run kept under _groups/<group>/), mirrored too
set +e
drv parity --render "$RENDER" --cells-root "$ROOT" --group "$GROUP"
PARITY=$?
set -e
if [ -d "$ROOT/_groups" ]; then
  rsync -a --checksum "$ROOT/_groups/" "$MIRROR/_groups/" || { RC=3 fail "the mirror of _groups failed"; }
fi
[ "$PARITY" = 0 ] || { echo "$(ts) [run_cell] group $GROUP failed its parity gate (or the gate could not run)"; VALID=1; }
[ "$VALID" = 0 ] || { echo "$(ts) [run_cell] $CELL is INVALID (kept; see gates/verdict.json)"; exit 1; }
echo "$(ts) [run_cell] $CELL valid"
