#!/usr/bin/env bash
# Stop a fleet's vLLM engines and leave their GPUs idle (run from the operator machine).
#
#   stop_engines.sh --fleet F --hosts H --engine-args E [--ssh-user U]
#
# Each engine's process group gets SIGTERM, then SIGKILL after 60 s. Then, per node, anything still computing
# on the fleet's GPUs of that node is killed (retitled engine workers survive a name-based pkill), and the
# GPUs must be idle. Exits 1 if any instance or GPU could not be freed.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib_operator.sh
. "$HERE/lib_operator.sh"
set -euo pipefail

FLEET="" HOSTS="" EA=""
while [ $# -gt 0 ]; do
  [ $# -ge 2 ] || die "$1 needs a value"
  case "$1" in
    --fleet) FLEET="$2" ;; --hosts) HOSTS="$2" ;; --engine-args) EA="$2" ;; --ssh-user) SSH_USER="$2" ;;
    *) die "unknown argument $1" ;;
  esac
  shift 2
done
[ -n "$FLEET" ] && [ -n "$HOSTS" ] && [ -n "$EA" ] || die "usage: stop_engines.sh --fleet F --hosts H --engine-args E"
require_py
existing FLEET HOSTS EA

lines=$(plan engines --fleet "$FLEET" --hosts "$HOSTS" --engine-args "$EA") || die "invalid plan"
failed=()
declare -A node_addr=() node_gpus=()
while IFS=$'\t' read -r id node addr _ gpus _; do
  rb_ctl 180 "$addr" stop "$id" || failed+=("$id")
  node_addr[$node]="$addr"
  node_gpus[$node]="${node_gpus[$node]:+${node_gpus[$node]},}$gpus"
done <<< "$lines"
for node in "${!node_addr[@]}"; do
  rb_ctl 120 "${node_addr[$node]}" gpus-free "${node_gpus[$node]}" || failed+=("$node GPUs")
done
[ "${#failed[@]}" -eq 0 ] || die "could not stop: ${failed[*]}"
say "ALL $(wc -l <<< "$lines") ENGINES STOPPED, their GPUs idle"
