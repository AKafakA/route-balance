#!/usr/bin/env bash
# Run setup.sh on every node of a fleet, in parallel (run from the operator machine).
#
#   setup_fleet.sh --fleet F --hosts H --engine-args E --scheduler-node N --hf-token-file LOCAL_PATH
#                  --tree REMOTE_DIR --log-dir LOCAL_DIR [--hub configs/scheduler_hub.json] [--ssh-user U]
#                  [--data-disks FILE] [--with-tests]
#
# --data-disks: {"<node>": "/dev/<disk>"} for nodes whose scratch disk is not the node type's default (some d7525
# have no NVMe disk); setup.sh gets --data-disk for those nodes, and refuses the others' missing default.
#
# Per node: the git-tracked files of this repository go to REMOTE_DIR (rsync), the token to
# ~/.rb-setup/hf_token (mode 600, over ssh stdin: never on a command line), then
# `setup.sh --role engine` with the node's --model specs (python -m tools.engine_plan models). Exit 3
# (REBOOT REQUIRED) reboots the node, waits for it (up to 30 min) and runs the same command again, once.
# The scheduler node then also runs `setup.sh --role scheduler` with the --hub specs. One log per node
# under LOG_DIR. Finally every engine node's venv freeze must be identical (one engine stack, D24).
# --with-tests: the scheduler node's setup.sh also builds the test venv (setup.sh --with-tests).
# Exits 1 if any node failed.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib_operator.sh
. "$HERE/lib_operator.sh"
set -euo pipefail

FLEET="" HOSTS="" EA="" SCHED="" TOKEN="" TREE="" LOGDIR="" HUB="$RB_REPO/configs/scheduler_hub.json" DISKS=""
TEST_ARGS=()
while [ $# -gt 0 ]; do
  if [ "$1" = --with-tests ]; then TEST_ARGS=(--with-tests); shift; continue; fi
  [ $# -ge 2 ] || die "$1 needs a value"
  case "$1" in
    --fleet) FLEET="$2" ;; --hosts) HOSTS="$2" ;; --engine-args) EA="$2" ;; --scheduler-node) SCHED="$2" ;;
    --hf-token-file) TOKEN="$2" ;; --tree) TREE="$2" ;; --log-dir) LOGDIR="$2" ;; --hub) HUB="$2" ;;
    --ssh-user) SSH_USER="$2" ;; --data-disks) DISKS="$2" ;;
    *) die "unknown argument $1" ;;
  esac
  shift 2
done
for v in FLEET HOSTS EA SCHED TOKEN TREE LOGDIR; do [ -n "${!v}" ] || die "missing argument ($v); see the header of $0"; done
[ -f "$TOKEN" ] && [ $(( 8#$(stat -c %a "$TOKEN") & 8#077 )) -eq 0 ] || die "$TOKEN must exist with mode 600"
[[ "$TREE" =~ ^[A-Za-z0-9._/-]+$ ]] || die "--tree must be a plain path"
require_py
existing FLEET HOSTS EA TOKEN HUB
absolute LOGDIR
nodes=$(plan nodes --fleet "$FLEET" --hosts "$HOSTS") || die "invalid fleet/hosts"
if [ -n "$DISKS" ]; then       # every key a fleet node, every value a /dev path
  existing DISKS
  RB_NODES="$nodes" "$PY" - "$DISKS" <<'PYEOF' || die "invalid --data-disks $DISKS"
import json, os, re, sys
d = json.load(open(sys.argv[1]))
nodes = {line.split("\t")[0] for line in os.environ["RB_NODES"].splitlines() if line}
assert isinstance(d, dict) and d, "a non-empty {node: /dev/disk} object"
for k, v in d.items():
    assert k in nodes, f"{k} is not a node of the fleet"
    assert isinstance(v, str) and re.fullmatch(r"/dev/[a-z0-9]+", v), f"{k}: {v!r} is not a /dev disk"
PYEOF
fi
plan engines --fleet "$FLEET" --hosts "$HOSTS" --engine-args "$EA" > /dev/null || die "invalid engine args"
awk -F'\t' -v n="$SCHED" '$1 == n {found = 1} END {exit !found}' <<< "$nodes" \
  || die "scheduler node $SCHED serves no instance of $FLEET"
hub_args=()
while IFS=$'\t' read -r flag spec; do hub_args+=("$flag" "$spec"); done < <(plan hub --file "$HUB" || echo FAIL)
[ "${hub_args[0]:-}" != FAIL ] && [ "${#hub_args[@]}" -gt 0 ] || die "invalid hub file $HUB"
mkdir -p "$LOGDIR"
REMOTE_TOKEN=.rb-setup/hf_token

run_setup() {    # run_setup <addr> <log> <setup.sh args...>: 0 ready, 3 reboot required, else failed
  local addr="$1" log="$2"; shift 2
  local q; q=$(printf '%q ' "$@")
  rb_ssh 14400 "$addr" "bash $TREE/scripts/setup.sh $q" >> "$log" 2>&1
}

setup_node() {   # setup_node <node> <addr>
  local node="$1" addr="$2" log="$LOGDIR/setup.$1.log" rc models=() spec i disk disk_args=()
  while read -r spec; do models+=(--model "$spec"); done < <(plan models --fleet "$FLEET" --hosts "$HOSTS" --engine-args "$EA" --node "$node")
  [ "${#models[@]}" -gt 0 ] || { echo "no models for $node" >> "$log"; return 1; }
  if [ -n "$DISKS" ]; then
    disk=$("$PY" -c 'import json, sys; print(json.load(open(sys.argv[1])).get(sys.argv[2], ""))' "$DISKS" "$node")
    [ -z "$disk" ] || disk_args=(--data-disk "$disk")
  fi
  {
    say "sync tree -> $node:$TREE"
    (cd "$RB_REPO" && git ls-files -z | timeout 600 rsync -a --from0 --files-from=- \
      -e "ssh ${SSH_OPTS[*]}" ./ "$(ssh_target "$addr"):$TREE/") || { echo "rsync failed"; return 1; }
    timeout 60 ssh "${SSH_OPTS[@]}" "$(ssh_target "$addr")" \
      "umask 077 && mkdir -p .rb-setup && cat > $REMOTE_TOKEN" < "$TOKEN" || { echo "token copy failed"; return 1; }
  } >> "$log" 2>&1
  for i in 1 2; do
    rc=0; run_setup "$addr" "$log" --role engine --hf-token-file "$REMOTE_TOKEN" "${models[@]}" ${disk_args[@]+"${disk_args[@]}"} || rc=$?
    [ "$rc" -eq 3 ] && [ "$i" -eq 1 ] || break
    say "$node: REBOOT REQUIRED -> rebooting" >> "$log"
    rb_ssh 30 "$addr" sudo reboot >> "$log" 2>&1 || true
    sleep 120
    local up=0
    for _ in $(seq 1 28); do
      if rb_ssh 30 "$addr" true > /dev/null 2>&1; then up=1; break; fi
      sleep 60
    done
    [ "$up" -eq 1 ] || { say "$node did not come back after the reboot" >> "$log"; return 1; }
  done
  [ "$rc" -eq 0 ] || { say "$node: engine setup failed (exit $rc)" >> "$log"; return 1; }
  if [ "$node" = "$SCHED" ]; then
    run_setup "$addr" "$log" --role scheduler --hf-token-file "$REMOTE_TOKEN" "${hub_args[@]}" ${disk_args[@]+"${disk_args[@]}"} \
      ${TEST_ARGS[@]+"${TEST_ARGS[@]}"} || { say "$node: scheduler setup failed" >> "$log"; return 1; }
  fi
  say "$node: ready" >> "$log"
}

pids=() names=()
while IFS=$'\t' read -r node addr hw _; do
  setup_node "$node" "$addr" &
  pids+=("$!"); names+=("$node")
  say "setup started on $node ($hw); log $LOGDIR/setup.$node.log"
done <<< "$nodes"
failed=()
for i in "${!pids[@]}"; do wait "${pids[$i]}" || failed+=("${names[$i]}"); done
[ "${#failed[@]}" -eq 0 ] || die "setup failed on: ${failed[*]} (logs under $LOGDIR)"

# one engine stack: every engine venv must have the same freeze
# shellcheck source=scripts/node_layout.sh
. "$HERE/node_layout.sh"
declare -A freeze=()
while IFS=$'\t' read -r node addr _; do
  freeze[$node]=$(rb_ssh 60 "$addr" sha256sum "$ENGINE_VENV/rb-setup.freeze" | cut -c1-64) || die "no freeze on $node"
done <<< "$nodes"
if [ "$(printf '%s\n' "${freeze[@]}" | sort -u | wc -l)" -ne 1 ]; then
  for node in "${!freeze[@]}"; do echo "  $node ${freeze[$node]}"; done
  die "the engine venvs differ between nodes (pip freeze); compare $ENGINE_VENV/rb-setup.freeze"
fi
say "ALL $(wc -l <<< "$nodes") NODES READY; engine freeze sha256 ${freeze[$SCHED]}"
