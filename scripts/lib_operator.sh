# shellcheck shell=bash
# Operator-side helpers for deploy.sh, stop_engines.sh, stop_scheduler.sh and setup_fleet.sh.
# Every remote call is bounded (timeout + ssh keep-alives), non-interactive (BatchMode) and multiplexed
# over one ControlMaster connection per host. Addresses come from local/hosts.json via tools.engine_plan.

LIB_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RB_REPO="$(cd "$LIB_HERE/.." && pwd)"
PY="${PY:-$RB_REPO/.venv/bin/python}"      # the operator's dev venv (tools.engine_plan, routebalance.config)
SSH_USER=""                                 # set by --ssh-user; empty = ssh's own configuration

ts() { date -u +%H:%M:%SZ; }
say() { echo "[$(ts)] $*"; }
die() { echo "[$(ts)] FAILED: $*" >&2; exit 1; }

require_py() {
  case "$PY" in /*) ;; *) PY="$PWD/$PY" ;; esac      # plan() runs from the repository root
  [ -x "$PY" ] || die "no python at $PY (make the dev venv, or set PY=<python with this repo's dependencies>)"
}

# absolute VAR...: make each named local path absolute against the caller's directory, once, before any use
# (plan() runs from the repository root, scp and sha256sum from here). Symlinks are kept as they are.
absolute() {
  local v
  for v in "$@"; do
    case "${!v}" in ""|/*) ;; *) printf -v "$v" '%s' "$PWD/${!v}" ;; esac
  done
}

# existing VAR...: absolute, and the file must exist
existing() {
  local v
  absolute "$@"
  for v in "$@"; do [ -z "${!v}" ] || [ -e "${!v}" ] || die "no such file: ${!v}"; done
}

plan() {   # tools.engine_plan <args>, from the repository root
  (cd "$RB_REPO" && "$PY" -m tools.engine_plan "$@")
}

ssh_target() { if [ -n "$SSH_USER" ]; then echo "$SSH_USER@$1"; else echo "$1"; fi; }

SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=15 -o ServerAliveCountMax=4
          -o StrictHostKeyChecking=accept-new -o ControlMaster=auto -o "ControlPath=$HOME/.ssh/sockets/rb-%C"
          -o ControlPersist=900)
mkdir -p "$HOME/.ssh/sockets"

# rb_ssh <timeout_s> <address> <remote command...>: stdin is closed
rb_ssh() {
  local t="$1" addr="$2"; shift 2
  timeout "$t" ssh -n "${SSH_OPTS[@]}" "$(ssh_target "$addr")" "$@"
}

# rb_ctl <timeout_s> <address> <node_ctl command> <args...>: runs the operator's node_ctl.sh on the node
rb_ctl() {
  local t="$1" addr="$2"; shift 2
  local q
  q=$(printf '%q ' "$@")
  cat "$LIB_HERE/node_layout.sh" "$LIB_HERE/node_ctl.sh" \
    | timeout "$t" ssh "${SSH_OPTS[@]}" "$(ssh_target "$addr")" "bash -s -- $q"
}
