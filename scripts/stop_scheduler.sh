#!/usr/bin/env bash
# Stop the scheduler that deploy.sh started on a node (run from the operator machine).
#
#   stop_scheduler.sh --fleet F --hosts H --node N [--ssh-user U]

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib_operator.sh
. "$HERE/lib_operator.sh"
set -euo pipefail

FLEET="" HOSTS="" NODE=""
while [ $# -gt 0 ]; do
  [ $# -ge 2 ] || die "$1 needs a value"
  case "$1" in
    --fleet) FLEET="$2" ;; --hosts) HOSTS="$2" ;; --node) NODE="$2" ;; --ssh-user) SSH_USER="$2" ;;
    *) die "unknown argument $1" ;;
  esac
  shift 2
done
[ -n "$FLEET" ] && [ -n "$HOSTS" ] && [ -n "$NODE" ] || die "usage: stop_scheduler.sh --fleet F --hosts H --node N"
require_py
existing FLEET HOSTS
addr=$(plan nodes --fleet "$FLEET" --hosts "$HOSTS" | awk -F'\t' -v n="$NODE" '$1 == n {print $2}') || die "invalid fleet/hosts"
[ -n "$addr" ] || die "node $NODE serves no instance of $FLEET"
rb_ctl 120 "$addr" sched-stop || die "could not stop the scheduler on $NODE"
