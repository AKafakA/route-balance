#!/usr/bin/env bash
# shellcheck disable=SC2015
# (SC2015: `a && b || die` is meant: die when either a or b fails.)
# One lease node's environment, from the v3 recipe (driver + CUDA 12.8.0 local repo) adapted to official
# vLLM in a fresh venv. Runs on the node itself; idempotent, fail-loud, resumable across the driver reboot.
#
#   setup.sh --role engine    --hf-token-file PATH --model ORG/NAME@REV [--model ...] [--data-disk DEV]
#   setup.sh --role scheduler --hf-token-file PATH --model ORG/NAME@REV [...] [--tokenizer ORG/NAME@REV ...]
#                             [--data-disk DEV] [--with-tests]
#   setup.sh --role vsr       --vsr-models DIR
#
#   /mydata goes on the node type's scratch disk (c4130 /dev/sdb, d7525/d8545 /dev/nvme0n1, as v3). Some d7525
#   have no NVMe disk: then --data-disk must name one, and without it setup fails and lists the candidates (whole
#   disks without partitions, mounts or the root filesystem). A disk with any of those is always refused.
#
#   engine     driver + CUDA toolkit, python3.10-dev, $ENGINE_VENV with vllm==$VLLM_VERSION, the weights of
#              every --model (the tiers this node serves: python -m tools.engine_plan models ... --node N)
#   scheduler  python3.10-dev, $SCHED_VENV with this repository (pip install -e, its pinned dependencies),
#              the pinned upstream AvengersPro and best-route-llm checkouts, every --model (sentence encoders,
#              BEST-Route's backbone) and --tokenizer
#              (tokenizer files only; configs/scheduler_hub.json lists both). No driver: the scheduler is CPU-only.
#              --with-tests also builds $TEST_VENV: this repository with the dev extras (pytest), for running the
#              test suite on the node against the same pins and Hub snapshots; nothing serves from it.
#   vsr        (the scheduler's node, after its scheduler role) Docker from Ubuntu's archive (docker.io
#              $DOCKER_IO_VERSION), the lease user in the docker group, $VSR_VENV with vllm-sr and
#              the versions it was tested with (vllm-sr-bench/requirements-vsr.txt), the four images vLLM-SR 0.4.0 runs,
#              pulled by their pinned digests and tagged as its CLI names them (vllm-sr-bench/vsrbench/constants.py),
#              and the router's classifier weights at --vsr-models checked file by file (vllm-sr-bench/models.sha256).
#              No token and no --model.
#   Every role ends with a self-check that prints the versions and exits non-zero on any mismatch.
#
# Exit status: 0 ready; 1 failed; 3 REBOOT REQUIRED (the driver needs a reboot: reboot the node, then run
# the same command again; it continues where it stopped). A node with the scheduler runs both roles.
# The token file is read by Python only; the token is never printed, logged or put on a command line.
# Nothing is ever installed into the system Python or into a virtualenv this script did not create.
# Log: ~/.rb-setup/setup.<role>.<UTC time>.log (the root disk: it exists before /mydata does).

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd -P)"
# shellcheck source=scripts/node_layout.sh
. "$HERE/node_layout.sh"

STATE_DIR="$HOME/.rb-setup"           # survives the reboot that unmounts /mydata
PIP_VERSION=24.2
CUDA_DEB=cuda-repo-ubuntu2204-12-8-local_12.8.0-570.86.10-1_amd64.deb
CUDA_URL=https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/$CUDA_DEB
CUDA_DEB_MD5=72f6916d375c91ad1ab456011425bf94     # NVIDIA's published md5sum.txt for 12.8.0
BASE_PACKAGES="python3.10-dev python3.10-venv build-essential ccache ninja-build cmake pciutils git curl"
AP_URL=https://github.com/ZhangYiqun018/AvengersPro.git
AP_COMMIT=dfb60b156c4b6011b89edfc8516aa16d8e4f4d38
AP_FILE=simple_cluster_router.py
AP_UPSTREAM_MD5=a845b36763db4ba4c22f87d78753c4bb  # $AP_FILE at $AP_COMMIT
AP_PATCHED_MD5=658d5d1a874d71a6dcb77700e1f2a2c4   # after the local patch below (the copy v3 served)
BR_URL=https://github.com/microsoft/best-route-llm.git
BR_COMMIT=fc9b896a814acf7c00565fbea0803a8c20812702  # BEST-Route's authors' code, unpatched
LLM_BLENDER_COMMIT=33204d2712944b6b17996f7c079e74cd963ccc7c   # == the pyproject pin (tests check)
SPEC_RE='^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*@[0-9a-f]{40}$'

# PCI device id of the node's GPUs -> GPU class, CloudLab hardware type, GPUs per node, driver flavour,
# /mydata disk. Must agree with routebalance/hw.py (tests/unit/test_scripts.py checks it).
gpu_row() {
  case "$1" in
    1db1) echo "v100 c4130 4 volta /dev/sdb" ;;          # Tesla V100-SXM2-16GB
    20b7) echo "a30 d7525 1 ampere /dev/nvme0n1" ;;      # A30
    20b0) echo "a100 d8545 4 ampere /dev/nvme0n1" ;;     # A100-SXM4-40GB
    *) return 1 ;;
  esac
}

installed() {   # installed PKG...: every package is fully installed (dpkg -s also accepts unpacked/half-configured)
  local p
  for p in "$@"; do
    [ "$(dpkg-query -W -f='${Status}' "$p" 2>/dev/null)" = "install ok installed" ] || return 1
  done
}

ts() { date -u +%H:%M:%SZ; }
say() { echo "[$(ts)] $*"; }
step() { echo; echo "===== [$(ts)] $* ====="; }
die() { echo "[$(ts)] SETUP FAILED: $*" >&2; exit 1; }

ROLE="" TOKEN_FILE="" MODELS=() TOKENIZERS=() DATA_DISK="" WITH_TESTS=0 VSR_MODELS_SRC=""
# Ubuntu 22.04's docker.io (jammy-updates) as the lease nodes' archive offered it on 30 Sep
DOCKER_IO_VERSION="29.1.3-0ubuntu3~22.04.2"

parse_args() {
  while [ $# -gt 0 ]; do
    case "$1" in
      --role) [ $# -ge 2 ] || die "--role needs a value"; ROLE="$2"; shift 2 ;;
      --hf-token-file) [ $# -ge 2 ] || die "--hf-token-file needs a value"; TOKEN_FILE="$2"; shift 2 ;;
      --model) [ $# -ge 2 ] || die "--model needs a value"; MODELS+=("$2"); shift 2 ;;
      --tokenizer) [ $# -ge 2 ] || die "--tokenizer needs a value"; TOKENIZERS+=("$2"); shift 2 ;;
      --data-disk) [ $# -ge 2 ] || die "--data-disk needs a value"; DATA_DISK="$2"; shift 2 ;;
      --with-tests) WITH_TESTS=1; shift ;;
      --vsr-models) [ $# -ge 2 ] || die "--vsr-models needs a value"; VSR_MODELS_SRC="$2"; shift 2 ;;
      -h|--help) sed -n '4,/^$/p' "${BASH_SOURCE[0]}"; exit 0 ;;
      *) die "unknown argument $1" ;;
    esac
  done
  case "$ROLE" in
    engine|scheduler) ;;
    vsr)
      [ -n "$VSR_MODELS_SRC" ] || die "--vsr-models is required for the vsr role"
      [ -z "$TOKEN_FILE" ] && [ "${#MODELS[@]}" -eq 0 ] && [ "${#TOKENIZERS[@]}" -eq 0 ] && [ "$WITH_TESTS" = 0 ] \
        && [ -z "$DATA_DISK" ] || die "the vsr role takes only --vsr-models"
      return 0 ;;
    *) die "--role must be engine, scheduler or vsr (got '${ROLE}')" ;;
  esac
  [ -z "$VSR_MODELS_SRC" ] || die "--vsr-models is for the vsr role"
  [ -n "$TOKEN_FILE" ] || die "--hf-token-file is required"
  [ "${#MODELS[@]}" -ge 1 ] || die "at least one --model ORG/NAME@REV is required"
  if [ "$ROLE" = engine ] && [ "${#TOKENIZERS[@]}" -gt 0 ]; then die "--tokenizer is for the scheduler role"; fi
  if [ "$ROLE" = engine ] && [ "$WITH_TESTS" = 1 ]; then die "--with-tests is for the scheduler role"; fi
  local s
  for s in "${MODELS[@]}" ${TOKENIZERS[@]+"${TOKENIZERS[@]}"}; do
    [[ "$s" =~ $SPEC_RE ]] || die "'$s' is not ORG/NAME@<40-hex Hub commit>"
  done
}

check_token_file() {
  [ -f "$TOKEN_FILE" ] && [ -s "$TOKEN_FILE" ] || die "token file $TOKEN_FILE is missing or empty"
  local mode
  mode=$(stat -c %a "$TOKEN_FILE") || die "cannot stat $TOKEN_FILE"
  [ $(( 8#$mode & 8#077 )) -eq 0 ] || die "token file $TOKEN_FILE is readable by others (mode $mode): chmod 600 it"
}

# ---------------------------------------------------------------- system

preflight() {
  step "preflight"
  # shellcheck source=/dev/null
  . /etc/os-release
  [ "${ID:-}" = ubuntu ] && [ "${VERSION_ID:-}" = 22.04 ] || die "Ubuntu 22.04 expected (found ${ID:-?} ${VERSION_ID:-?})"
  [ "$(id -u)" -ne 0 ] || die "run as the lease user, not root (sudo is used where needed)"
  sudo -n true 2>/dev/null || die "passwordless sudo is required"
  say "role=$ROLE repo=$REPO kernel=$(uname -r) models=${MODELS[*]} tokenizers=${TOKENIZERS[*]:-none}"
}

base_packages() {
  step "apt: full-upgrade once, then $BASE_PACKAGES"
  # finish whatever an interrupted earlier run left unpacked or half-configured (a DKMS build, say)
  sudo DEBIAN_FRONTEND=noninteractive dpkg --configure -a || die "dpkg --configure -a"
  if [ ! -f "$STATE_DIR/apt-upgrade.done" ]; then
    sudo apt-get update || die "apt-get update"
    sudo DEBIAN_FRONTEND=noninteractive apt-get full-upgrade -y || die "apt-get full-upgrade"
    date -u +%FT%TZ > "$STATE_DIR/apt-upgrade.done"
  fi
  # shellcheck disable=SC2086
  if ! installed $BASE_PACKAGES; then
    # shellcheck disable=SC2086
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y $BASE_PACKAGES || die "apt-get install $BASE_PACKAGES"
  fi
  python3.10 -c 'import os, sys, sysconfig; sys.exit(0 if os.path.isfile(os.path.join(sysconfig.get_paths()["include"], "Python.h")) else 1)' \
    || die "Python.h missing after installing python3.10-dev (vLLM's startup JIT needs it)"
}

GPU_CLASS="" HW_TYPE="" GPU_COUNT="" ARCH="" MYDATA_DEV=""

detect_gpu() {
  step "GPU detection (lspci)"
  local lines ids row
  lines=$(lspci -nn -d 10de: | grep -E '\[(0300|0302)\]') || die "no NVIDIA display/3D controller on this node"
  echo "$lines"
  ids=$(echo "$lines" | grep -oE '\[10de:[0-9a-f]{4}\]' | cut -c7-10 | sort -u)
  [ "$(echo "$ids" | wc -l)" -eq 1 ] || die "mixed GPU models on one node: $(echo "$ids" | tr '\n' ' ')"
  row=$(gpu_row "$ids") || die "GPU device id 10de:$ids is not one of this lease's (V100/A30/A100); see gpu_row"
  read -r GPU_CLASS HW_TYPE GPU_COUNT ARCH MYDATA_DEV <<< "$row"
  [ "$(echo "$lines" | wc -l)" -eq "$GPU_COUNT" ] || die "$HW_TYPE should have $GPU_COUNT GPU(s), lspci shows $(echo "$lines" | wc -l)"
  if [ -n "$DATA_DISK" ]; then
    MYDATA_DEV="$DATA_DISK"
  elif [ ! -b "$MYDATA_DEV" ] && ! [ -s "$STATE_DIR/mydata.uuid" ] && ! mountpoint -q "$MYDATA"; then
    die "$HW_TYPE's scratch disk $MYDATA_DEV does not exist on this node; pass --data-disk one of: $(candidate_disks)"
  fi
  say "gpu=$GPU_CLASS x$GPU_COUNT hw=$HW_TYPE arch=$ARCH /mydata disk=$MYDATA_DEV"
}

candidate_disks() {   # whole disks with no partitions, no mounts and not the root filesystem's
  local root name
  root=$(lsblk -nro PKNAME "$(findmnt -no SOURCE /)" 2>/dev/null)
  while read -r name type; do
    [ "$type" = disk ] || continue
    [ "$name" = "$root" ] && continue
    [ "$(lsblk -nro NAME "/dev/$name" | wc -l)" -eq 1 ] || continue            # has partitions
    lsblk -nro MOUNTPOINT "/dev/$name" | grep -q . && continue
    printf '/dev/%s(%s) ' "$name" "$(lsblk -dnro SIZE "/dev/$name")"
  done < <(lsblk -dnro NAME,TYPE)
}

mount_mydata() {
  step "/mydata"
  local uuid
  if mountpoint -q "$MYDATA"; then
    uuid=$(findmnt -no UUID "$MYDATA")
    [ -s "$STATE_DIR/mydata.uuid" ] || echo "$uuid" > "$STATE_DIR/mydata.uuid"
    [ "$(cat "$STATE_DIR/mydata.uuid")" = "$uuid" ] || die "$MYDATA is mounted from filesystem $uuid, not the one this setup created"
  elif [ -s "$STATE_DIR/mydata.uuid" ]; then     # after a reboot: remount, never format again
    uuid=$(cat "$STATE_DIR/mydata.uuid")
    sudo mkdir -p "$MYDATA" && sudo mount "UUID=$uuid" "$MYDATA" || die "cannot remount $MYDATA (UUID=$uuid)"
  else                                           # first run: format the scratch disk, as v3 did
    [ -b "$MYDATA_DEV" ] || die "$MYDATA_DEV is not a block device"
    [ "$(lsblk -dnro TYPE "$MYDATA_DEV")" = disk ] || die "$MYDATA_DEV is not a whole disk"
    if lsblk -nro MOUNTPOINT "$MYDATA_DEV" | grep -q .; then
      die "$MYDATA_DEV (or a partition of it) is in use: $(lsblk -nro NAME,MOUNTPOINT "$MYDATA_DEV" | tr '\n' ' ')"
    fi
    [ "$(lsblk -nro NAME "$MYDATA_DEV" | wc -l)" -eq 1 ] || die "$MYDATA_DEV has partitions; refusing to format it"
    [ "$(lsblk -nro PKNAME "$(findmnt -no SOURCE /)")" != "$(basename "$MYDATA_DEV")" ] || die "$MYDATA_DEV holds /"
    sudo mkfs.ext4 -F -q "$MYDATA_DEV" || die "mkfs.ext4 $MYDATA_DEV"
    uuid=$(sudo blkid -s UUID -o value "$MYDATA_DEV") && [ -n "$uuid" ] || die "no UUID on $MYDATA_DEV"
    echo "$uuid" > "$STATE_DIR/mydata.uuid"
    sudo mkdir -p "$MYDATA" && sudo mount "$MYDATA_DEV" "$MYDATA" || die "mount $MYDATA_DEV"
  fi
  sudo chmod 777 "$MYDATA"
  mkdir -p "$RB_ROOT" "$RB_HF_HOME" "$RB_LOGS" "$RB_RUN" "$RB_DOWNLOADS"
  df -h "$MYDATA" | tail -1
}

cuda_toolkit() {
  step "CUDA 12.8.0 toolkit (local repo, driver $DRIVER_VERSION)"
  if installed cuda-toolkit-12-8 && [ -x "$CUDA_DIR/bin/nvcc" ]; then say "already installed"; return; fi
  sudo tee /etc/apt/preferences.d/cuda-repository-pin-600 >/dev/null <<'PIN'
Package: nsight-compute
Pin: origin *ubuntu.com*
Pin-Priority: -1

Package: nsight-systems
Pin: origin *ubuntu.com*
Pin-Priority: -1

Package: *
Pin: release l=NVIDIA CUDA
Pin-Priority: 600
PIN
  local deb="$RB_DOWNLOADS/$CUDA_DEB"
  if [ ! -s "$deb" ] || [ "$(md5sum "$deb" | cut -d' ' -f1)" != "$CUDA_DEB_MD5" ]; then
    curl -fL --retry 3 -o "$deb.part" "$CUDA_URL" && mv "$deb.part" "$deb" || die "download $CUDA_URL"
  fi
  [ "$(md5sum "$deb" | cut -d' ' -f1)" = "$CUDA_DEB_MD5" ] || die "$deb md5 != $CUDA_DEB_MD5"
  sudo dpkg -i "$deb" || die "dpkg -i $CUDA_DEB"
  sudo cp /var/cuda-repo-ubuntu2204-12-8-local/cuda-*-keyring.gpg /usr/share/keyrings/ || die "CUDA keyring"
  sudo apt-get update || die "apt-get update (CUDA repo)"
  sudo dpkg --configure -a && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y cuda-toolkit-12-8 \
    || die "apt-get install cuda-toolkit-12-8"
  [ -x "$CUDA_DIR/bin/nvcc" ] || die "no nvcc at $CUDA_DIR after the toolkit install"
}

driver_ok() {
  local v
  v=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | sort -u) || return 1
  [ "$v" = "$DRIVER_VERSION" ]
}

nvidia_driver() {
  local pkg=nvidia-open                 # Ampere: NVIDIA's open kernel modules
  [ "$ARCH" = volta ] && pkg=cuda-drivers   # Volta: the open modules do not support it; proprietary + reboot
  step "driver $DRIVER_VERSION ($pkg)"
  if driver_ok; then say "driver $DRIVER_VERSION loaded"; rm -f "$STATE_DIR/reboot.boot_id"; return; fi
  if ! installed "$pkg"; then
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y "$pkg" || die "apt-get install $pkg"
  fi
  say "dkms: $(dkms status 2>&1 | tr '\n' ' ')"      # for the log; the version is checked once the driver loads
  if [ "$ARCH" = ampere ] && driver_ok; then say "driver $DRIVER_VERSION loaded without a reboot"; return; fi
  local boot; boot=$(cat /proc/sys/kernel/random/boot_id)
  if [ -s "$STATE_DIR/reboot.boot_id" ] && [ "$(cat "$STATE_DIR/reboot.boot_id")" != "$boot" ]; then
    die "the node rebooted after the driver install, but nvidia-smi does not report $DRIVER_VERSION: $(nvidia-smi 2>&1 | head -3 | tr '\n' ' ')"
  fi
  echo "$boot" > "$STATE_DIR/reboot.boot_id"
  echo
  echo "REBOOT REQUIRED: driver $DRIVER_VERSION is installed but not loaded ($ARCH; nouveau holds Volta GPUs)."
  echo "Reboot the node (sudo reboot), then run the same command again: $0 --role $ROLE ..."
  exit 3
}

# mig_enabled: 0 when a GPU has MIG on, 1 when every GPU reports it off; a failed query, a GPU count other than
# GPU_COUNT or a mode other than Enabled/Disabled fails setup (an unread mode is never read as off)
mig_enabled() {
  local modes
  modes=$(nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader) || die "nvidia-smi could not read the MIG mode"
  [ "$(grep -c . <<< "$modes")" -eq "$GPU_COUNT" ] || die "the MIG mode of $(grep -c . <<< "$modes") GPU(s), not $GPU_COUNT"
  if grep -q -v -x -e Enabled -e Disabled <<< "$modes"; then die "unexpected MIG mode: $(tr '\n' ' ' <<< "$modes")"; fi
  grep -q -x Enabled <<< "$modes"
}

mig_off() {
  [ "$ARCH" = ampere ] || return 0
  step "MIG off"
  if mig_enabled; then
    sudo nvidia-smi -mig 0 || die "nvidia-smi -mig 0"
  fi
  if mig_enabled; then
    die "MIG is still enabled: $(nvidia-smi --query-gpu=index,mig.mode.current,mig.mode.pending --format=csv,noheader | tr '\n' ' ')"
  fi
}

# ---------------------------------------------------------------- virtualenvs (only ones made here)

# new_venv DIR SPEC: create DIR for SPEC, or accept it if this script completed it for the same SPEC.
# Returns 0 when the caller must install into it, 1 when it is complete already.
new_venv() {
  local dir="$1" spec="$2" marker="$1/rb-setup.spec"
  if [ -e "$dir" ]; then
    [ -f "$marker" ] || die "$dir exists but was not created by this script; it is never touched (R1)"
    if [ "$(cat "$marker")" = "complete $spec" ]; then say "$dir is complete for: $spec"; return 1; fi
    if [ "$(cat "$marker")" = "installing $spec" ]; then     # this script's own interrupted install
      say "removing the interrupted install at $dir"
      case "$(readlink -f "$dir")" in "$RB_ROOT"/venv-*) rm -rf "$dir" ;; *) die "refusing to remove $dir" ;; esac
    else
      die "$dir was built for '$(cat "$marker")', not '$spec'. Remove it deliberately (rm -rf $dir) and rerun"
    fi
  fi
  mkdir "$dir" || die "mkdir $dir"
  echo "installing $spec" > "$marker"          # ownership first: an interrupted creation is recognised as ours
  python3.10 -m venv "$dir" || die "python3.10 -m venv $dir"
  "$dir/bin/python" -m pip install --quiet "pip==$PIP_VERSION" || die "pip==$PIP_VERSION into $dir"
  return 0
}

finish_venv() {
  local dir="$1" spec="$2"
  "$dir/bin/python" -m pip check || die "pip check failed in $dir"
  "$dir/bin/python" -m pip freeze --all > "$dir/rb-setup.freeze"
  say "freeze sha256 $(sha256sum "$dir/rb-setup.freeze" | cut -c1-16) ($(wc -l < "$dir/rb-setup.freeze") packages)"
  echo "complete $spec" > "$dir/rb-setup.spec"
}

engine_venv() {
  local spec="engine vllm==$VLLM_VERSION"
  step "venv $ENGINE_VENV: vllm==$VLLM_VERSION (official, PyPI)"
  if new_venv "$ENGINE_VENV" "$spec"; then
    "$ENGINE_VENV/bin/python" -m pip install "vllm==$VLLM_VERSION" || die "pip install vllm==$VLLM_VERSION"
    finish_venv "$ENGINE_VENV" "$spec"
  fi
}

scheduler_venv() {
  [ -f "$REPO/pyproject.toml" ] && [ -d "$REPO/routebalance" ] || die "$REPO is not the route-balance tree"
  local spec
  spec="scheduler $REPO pyproject.sha256=$(sha256sum "$REPO/pyproject.toml" | cut -c1-16)"
  step "venv $SCHED_VENV: pip install -e $REPO"
  if new_venv "$SCHED_VENV" "$spec"; then
    "$SCHED_VENV/bin/python" -m pip install -e "$REPO" || die "pip install -e $REPO"
    finish_venv "$SCHED_VENV" "$spec"
  fi
}

test_venv() {
  local spec
  spec="test $REPO pyproject.sha256=$(sha256sum "$REPO/pyproject.toml" | cut -c1-16)"
  step "venv $TEST_VENV: pip install -e ${REPO}[dev], constrained to the scheduler venv's versions (the test suite only)"
  [ -f "$SCHED_VENV/rb-setup.freeze" ] || die "no $SCHED_VENV/rb-setup.freeze: the scheduler venv comes first"
  if new_venv "$TEST_VENV" "$spec"; then
    # every package the scheduler venv resolved, at its version: the test venv is the serving stack plus the dev extras
    "$SCHED_VENV/bin/python" -m tools.venv_parity --constraints "$SCHED_VENV/rb-setup.freeze" > "$TEST_VENV/rb-setup.constraints" \
      || die "cannot write the constraints"
    "$TEST_VENV/bin/python" -m pip install -e "${REPO}[dev]" -c "$TEST_VENV/rb-setup.constraints" \
      || die "pip install -e ${REPO}[dev]"
    "$TEST_VENV/bin/python" -m pip freeze --all > "$TEST_VENV/rb-setup.freeze.new"
    "$TEST_VENV/bin/python" -m tools.venv_parity "$SCHED_VENV/rb-setup.freeze" "$TEST_VENV/rb-setup.freeze.new" \
      || die "the test venv differs from the scheduler venv (not marked complete)"
    rm -f "$TEST_VENV/rb-setup.freeze.new"
    finish_venv "$TEST_VENV" "$spec"
  fi
  # compared on every run: the scheduler venv may have been rebuilt since (the suite compares again at run time)
  "$TEST_VENV/bin/python" -m tools.venv_parity "$SCHED_VENV/rb-setup.freeze" "$TEST_VENV/rb-setup.freeze" \
    || die "the test venv differs from the scheduler venv: remove $TEST_VENV deliberately and rerun"
  local where
  where=$(cd / && env -i HOME="$HOME" PATH="$RB_PATH" "$TEST_VENV/bin/python" -c \
    'import os, pytest, routebalance; print(os.path.realpath(routebalance.__file__), pytest.__version__)') \
    || die "the test venv cannot import pytest and routebalance"
  [[ "$where" == "$(cd "$REPO" && pwd -P)"/* ]] || die "the test venv imports routebalance from $where, not $REPO"
  say "SELF-CHECK OK: tests ($TEST_VENV: $where)"
}

# ---------------------------------------------------------------- the vLLM Semantic Router (vsr only)

docker_engine() {
  step "Docker: docker.io=$DOCKER_IO_VERSION (Ubuntu's archive), data root $DOCKER_ROOT, $USER in the docker group"
  local have conf sim changed
  have=$(dpkg-query -W -f '${Version}' docker.io 2>/dev/null || true)
  [ -z "$have" ] || installed docker.io || die "docker.io $have is not fully installed (unpacked or half-configured): R1"
  # v2's vLLM-SR deployment guide (2026-05-03: "Always set data-root: /mydata/docker before starting Docker"; the root
  # disk is 63 GB), written before the package's first start of the daemon; as v2's storage: Docker 29's
  # containerd image store off, so images and layers stay under the data root in Docker's classic store, as v2 ran
  conf='{"data-root":"'"$DOCKER_ROOT"'","features":{"containerd-snapshotter":false}}'
  if [ "$have" != "$DOCKER_IO_VERSION" ]; then
    [ -z "$have" ] || die "docker.io $have is installed, not $DOCKER_IO_VERSION: it is never replaced here (R1)"
    sudo apt-get update || die "apt-get update"
    # nothing installed is changed here: apt's own dry run first (before anything is written), refused if it would
    # upgrade, downgrade or remove any installed package ("Inst <pkg> [<installed version>] ..." or "Remv")
    sim=$(apt-get -s install --no-remove "docker.io=$DOCKER_IO_VERSION") || die "apt-get -s install docker.io"
    changed=$(printf '%s\n' "$sim" | grep -E '^(Inst [^ ]+ \[|Remv )' || true)
    [ -z "$changed" ] || die "installing docker.io would change installed packages (R1): $changed"
    say "new packages: $(printf '%s\n' "$sim" | awk '/^Inst /{printf "%s=%s ", $2, substr($3, 2)}')"
    if [ -e /etc/docker/daemon.json ]; then
      [ "$(cat /etc/docker/daemon.json)" = "$conf" ] || die "/etc/docker/daemon.json exists and is not $conf (R1)"
    else
      sudo mkdir -p /etc/docker "$DOCKER_ROOT" || die "mkdir /etc/docker $DOCKER_ROOT"
      printf '%s\n' "$conf" | sudo tee /etc/docker/daemon.json >/dev/null || die "writing /etc/docker/daemon.json"
    fi
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-remove -o Dpkg::Options::=--force-confold \
      "docker.io=$DOCKER_IO_VERSION" || die "apt-get install docker.io"
  fi
  [ "$(cat /etc/docker/daemon.json)" = "$conf" ] || die "/etc/docker/daemon.json is not $conf"
  # Docker runs only during the vLLM-SR blocks, so never at boot: started here, its units disabled (and
  # ubuntu-fan's, a package apt's default install brings with docker.io, as v2's install did)
  local u
  for u in docker.service docker.socket containerd.service ubuntu-fan.service; do
    systemctl cat "$u" >/dev/null 2>&1 || continue
    sudo systemctl disable "$u" || die "systemctl disable $u"
  done
  sudo systemctl start docker || die "systemctl start docker"
  if ! id -nG "$USER" | tr ' ' '\n' | grep -qx docker; then
    sudo usermod -aG docker "$USER" || die "usermod -aG docker $USER"
    say "$USER joined the docker group: a new login (or sg docker) picks it up"
  fi
}

vsr_venv() {
  local req="$REPO/vllm-sr-bench/requirements-vsr.txt" spec
  [ -f "$req" ] || die "$req is missing (this tree has no vllm-sr-bench)"
  spec="vsr requirements.sha256=$(sha256sum "$req" | cut -c1-16)"
  step "venv $VSR_VENV: vllm-sr and the versions it was tested with"
  if new_venv "$VSR_VENV" "$spec"; then
    "$VSR_VENV/bin/python" -m pip install --no-deps -r "$req" || die "pip install -r $req"
    finish_venv "$VSR_VENV" "$spec"
  fi
}

vsr_image_list() {   # sets VSR_IMAGES: "<ref> <digest>" per image, from the module's pins; all four, or die
  VSR_IMAGES=$("$VSR_VENV/bin/python" -c 'import sys; sys.path.insert(0, sys.argv[1]); from vsrbench import constants as C
for ref, digest in C.IMAGES.values(): print(ref, digest)' "$REPO/vllm-sr-bench") || die "reading the image pins"
  [ "$(printf '%s\n' "$VSR_IMAGES" | grep -c .)" = 4 ] || die "the module pins $(printf '%s\n' "$VSR_IMAGES" | grep -c .) images, 4 expected"
}

vsr_images() {
  step "the images vLLM-SR 0.4.0 runs, by their pinned digests"
  local ref digest repo
  vsr_image_list
  while read -r ref digest; do
    repo=${ref%:*}
    sudo docker pull "$repo@$digest" || die "docker pull $repo@$digest"
    sudo docker tag "$repo@$digest" "$ref" || die "docker tag $ref"
  done <<< "$VSR_IMAGES"
}

vsr_models() {
  local manifest="$REPO/vllm-sr-bench/models.sha256" n
  step "the router's classifier weights at $VSR_MODELS_SRC, file by file"
  [ -d "$VSR_MODELS_SRC" ] || die "$VSR_MODELS_SRC is not a directory"
  # the model files; the download metadata the router may rewrite (.cache/, the revision receipts) is not compared,
  # as vsrbench/run_cell.py verify_weights does every run
  grep -v -e '/\.cache/' -e 'router-model-revision\.json$' "$manifest" > "$STATE_DIR/vsr-models.sha256" \
    || die "reading $manifest"
  (cd "$VSR_MODELS_SRC" && sha256sum --quiet -c "$STATE_DIR/vsr-models.sha256") || die "the weights differ from $manifest"
  n=$(cd "$VSR_MODELS_SRC" && find . -type f -not -path '*/.cache/*' -not -name '.router-model-revision.json' | wc -l)
  [ "$n" = "$(wc -l < "$STATE_DIR/vsr-models.sha256")" ] \
    || die "$VSR_MODELS_SRC holds $n model files, $manifest lists $(wc -l < "$STATE_DIR/vsr-models.sha256")"
}

self_check_vsr() {
  step "self-check (vsr)"
  local fail=0 ref digest
  installed docker.io && [ "$(dpkg-query -W -f '${Version}' docker.io)" = "$DOCKER_IO_VERSION" ] || { say "docker.io is not fully installed at $DOCKER_IO_VERSION"; fail=1; }
  [ "$(sudo docker info --format '{{.DockerRootDir}}')" = "$DOCKER_ROOT" ] || { say "Docker's data root is not $DOCKER_ROOT"; fail=1; }
  [ "$(sudo docker info --format '{{.Driver}}')" = overlay2 ] || { say "Docker's storage is not overlay2 under $DOCKER_ROOT"; fail=1; }
  local u state
  for u in docker.service docker.socket containerd.service ubuntu-fan.service; do     # never started at boot
    state=$(systemctl is-enabled "$u" 2>&1) || :      # disabled (and a missing unit) exit nonzero; the case reads it
    case "$state" in
      disabled) ;;
      *"No such file"*|*"not found"*|not-found*) [ "$u" = ubuntu-fan.service ] || { say "$u is not installed"; fail=1; } ;;
      *) say "$u is not disabled at boot ($state)"; fail=1 ;;
    esac
  done
  say "Docker's storage: $(sudo docker info --format '{{.Driver}} {{json .DriverStatus}}'); root disk $(df -h / | awk 'NR==2{print $4}') free"
  say "docker server $(sudo docker version --format '{{.Server.Version}}'); containerd $(dpkg-query -W -f '${Version}' containerd 2>/dev/null || echo none); runc $(dpkg-query -W -f '${Version}' runc 2>/dev/null || echo none)"
  "$VSR_VENV/bin/vllm-sr" --version | grep -q "0\.4\.0" || { say "vllm-sr is not 0.4.0"; fail=1; }
  vsr_image_list
  while read -r ref digest; do
    sudo docker image inspect "$ref" --format '{{join .RepoDigests " "}}' | tr ' ' '\n' | grep -qx "${ref%:*}@$digest" \
      || { say "$ref is not $digest"; fail=1; }
  done <<< "$VSR_IMAGES"
  id -nG "$USER" | tr ' ' '\n' | grep -qx docker || { say "$USER is not in the docker group"; fail=1; }
  [ "$fail" = 0 ] || die "self-check (vsr) failed"
  say "self-check (vsr) OK"
}

# ---------------------------------------------------------------- AvengersPro (scheduler only)

ap_patch() {   # v3's one local edit: upstream imports `datasets` for training only; routing never uses it
  git -C "$AP_DIR" apply --whitespace=nowarn - <<'PATCH'
diff --git a/simple_cluster_router.py b/simple_cluster_router.py
--- a/simple_cluster_router.py
+++ b/simple_cluster_router.py
@@ -27,7 +27,10 @@ from concurrent.futures import ThreadPoolExecutor
 import sys
 import os
 
-from datasets import Dataset
+try:
+    from datasets import Dataset
+except ImportError:  # `datasets` only needed for upstream training, not routing
+    Dataset = None
 from sklearn.cluster import KMeans
 from sklearn.preprocessing import Normalizer
 import joblib
PATCH
}

avengers_pro() {
  step "AvengersPro $AP_COMMIT + local patch -> $AP_DIR"
  if [ ! -e "$AP_DIR" ]; then
    rm -rf "$AP_DIR.clone"
    git clone --quiet "$AP_URL" "$AP_DIR.clone" || die "git clone $AP_URL"
    git -C "$AP_DIR.clone" checkout --quiet --detach "$AP_COMMIT" || die "checkout $AP_COMMIT"
    mv "$AP_DIR.clone" "$AP_DIR"
  fi
  [ -d "$AP_DIR/.git" ] || die "$AP_DIR exists but is not a git checkout"
  [ "$(git -C "$AP_DIR" rev-parse HEAD)" = "$AP_COMMIT" ] || die "$AP_DIR is at $(git -C "$AP_DIR" rev-parse HEAD), not $AP_COMMIT"
  local md5; md5=$(md5sum "$AP_DIR/$AP_FILE" | cut -d' ' -f1)
  if [ "$md5" = "$AP_UPSTREAM_MD5" ]; then ap_patch || die "the local patch does not apply"; fi
  [ "$(md5sum "$AP_DIR/$AP_FILE" | cut -d' ' -f1)" = "$AP_PATCHED_MD5" ] || die "$AP_FILE md5 is not the patched v3 copy"
  [ "$(git -C "$AP_DIR" status --porcelain --untracked-files=no)" = " M $AP_FILE" ] \
    || die "$AP_DIR has changes beyond the patch: $(git -C "$AP_DIR" status --porcelain | tr '\n' ' ')"
}

best_route_llm() {
  step "best-route-llm $BR_COMMIT -> $BR_DIR"
  if [ ! -e "$BR_DIR" ]; then
    rm -rf "$BR_DIR.clone"
    git clone --quiet "$BR_URL" "$BR_DIR.clone" || die "git clone $BR_URL"
    git -C "$BR_DIR.clone" checkout --quiet --detach "$BR_COMMIT" || die "checkout $BR_COMMIT"
    mv "$BR_DIR.clone" "$BR_DIR"
  fi
  [ -d "$BR_DIR/.git" ] || die "$BR_DIR exists but is not a git checkout"
  [ "$(git -C "$BR_DIR" rev-parse HEAD)" = "$BR_COMMIT" ] || die "$BR_DIR is at $(git -C "$BR_DIR" rev-parse HEAD), not $BR_COMMIT"
  [ -z "$(git -C "$BR_DIR" status --porcelain --untracked-files=no)" ] \
    || die "$BR_DIR has local changes: $(git -C "$BR_DIR" status --porcelain | tr '\n' ' ')"
}

# ---------------------------------------------------------------- Hugging Face snapshots

# hub_fetch PYTHON MODE: MODE download fetches every spec at its pinned commit; verify only reads the cache.
# Engines name the revision (--revision). The scheduler's processes run offline (HF_HUB_OFFLINE=1) and some
# of its loaders name only the repository (the AvengersPro embedder, the benchmark tokenizer), which resolves
# through the cache's refs/main: for the scheduler role, refs/main must equal the pinned commit. It is written
# once, and only after the Hub confirms that its main IS the pinned commit; an existing refs/main is never
# refetched (so a rerun cannot repoint it), and node_ctl.sh sched-start checks it again before every launch.
hub_fetch() {
  env -i HOME="$HOME" PATH="$RB_PATH" HF_HOME="$RB_HF_HOME" HF_HUB_DISABLE_PROGRESS_BARS=1 "$1" - "$2" "$ROLE" "$TOKEN_FILE" "${#MODELS[@]}" "${MODELS[@]}" ${TOKENIZERS[@]+"${TOKENIZERS[@]}"} <<'PY'
import json, sys
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

mode, role, token_file, n_models = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
specs = [(s, "model") for s in sys.argv[5:5 + n_models]] + [(s, "tokenizer") for s in sys.argv[5 + n_models:]]
WEIGHTS = ["*.safetensors", "*.bin", "*.pt", "*.pth", "*.h5", "*.msgpack", "*.gguf", "*.onnx", "*.ot", "*.tflite"]
token = Path(token_file).read_text().strip() if mode == "download" else None
bad = []
for spec, kind in specs:
    repo, rev = spec.split("@")
    kw = {"ignore_patterns": WEIGHTS} if kind == "tokenizer" else {}
    try:
        if mode == "download":
            snap = Path(snapshot_download(repo, revision=rev, token=token, max_workers=8, **kw))
            if role == "scheduler" and not (snap.parent.parent / "refs" / "main").is_file():
                main = HfApi().model_info(repo, revision="main", token=token).sha
                if main != rev:
                    bad.append(f"{spec}: the Hub's main is {main}, not the pinned commit (pin it in the loaders)")
                    continue
                snapshot_download(repo, revision="main", token=token, max_workers=8, **kw)   # writes refs/main
        path = Path(snapshot_download(repo, revision=rev, local_files_only=True, **kw))
    except Exception as e:   # report and continue, so one run names every missing snapshot
        bad.append(f"{spec}: {type(e).__name__}: {e}")
        continue
    problems = []
    if path.name != rev:
        problems.append(f"snapshot dir {path.name} != {rev}")
    if role == "scheduler":
        ref = path.parent.parent / "refs" / "main"
        main = ref.read_text().strip() if ref.is_file() else None
        if main != rev:
            problems.append(f"refs/main is {main}, not the pinned commit (the Hub's main moved: pin it in the loaders)")
    if not (path / "config.json").is_file() and kind == "model":
        problems.append("no config.json")
    if kind == "model":
        index = path / "model.safetensors.index.json"
        shards = sorted(set(json.loads(index.read_text())["weight_map"].values())) if index.is_file() \
            else [p.name for p in path.glob("*.safetensors")]
        if not shards and role == "scheduler" and (path / "pytorch_model.bin").is_file():
            shards = ["pytorch_model.bin"]   # a scheduler model published only as a PyTorch file (BEST-Route's backbone)
        if not shards:
            problems.append("no safetensors weights")
        missing = [s for s in shards if not (path / s).is_file() or (path / s).stat().st_size == 0]
        if missing:
            problems.append(f"missing or empty shards {missing}")
    size = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    print(f"  {kind:9s} {spec}  {size / 1e9:.2f} GB  {'OK' if not problems else 'BAD: ' + '; '.join(problems)}")
    if problems:
        bad.append(f"{spec}: {'; '.join(problems)}")
if bad:
    print("snapshots not usable:\n  " + "\n  ".join(bad), file=sys.stderr)
    sys.exit(1)
PY
}

# ---------------------------------------------------------------- self-check

self_check_engine() {
  step "self-check (engine)"
  local fail=0 line
  mountpoint -q "$MYDATA" || { echo "FAIL $MYDATA not mounted"; fail=1; }
  nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
  [ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -eq "$GPU_COUNT" ] || { echo "FAIL GPU count != $GPU_COUNT"; fail=1; }
  driver_ok || { echo "FAIL driver != $DRIVER_VERSION"; fail=1; }
  while read -r line; do
    [[ "${line^^}" == *"${GPU_CLASS^^}"* ]] || { echo "FAIL GPU '$line' is not a ${GPU_CLASS^^}"; fail=1; }
  done < <(nvidia-smi --query-gpu=name --format=csv,noheader)
  line=$("$CUDA_DIR/bin/nvcc" --version | grep -o 'release [0-9.]*')
  echo "nvcc: $line"; [ "$line" = "release 12.8" ] || { echo "FAIL nvcc $line != release 12.8"; fail=1; }
  if [ "$ARCH" = ampere ] && mig_enabled; then
    echo "FAIL MIG enabled"; fail=1
  fi
  env -i HOME="$HOME" PATH="$RB_PATH" CUDA_HOME="$CUDA_DIR" "$ENGINE_VENV/bin/python" - "$VLLM_VERSION" "$GPU_COUNT" <<'PY' || fail=1
import platform, sys
want_vllm, want_gpus = sys.argv[1], int(sys.argv[2])
import torch, vllm, transformers
fail = []
print(f"python {platform.python_version()}  vllm {vllm.__version__}  torch {torch.__version__} (CUDA {torch.version.cuda})  "
      f"transformers {transformers.__version__}")
if not platform.python_version().startswith("3.10."):
    fail.append("python is not 3.10")
if vllm.__version__ != want_vllm:
    fail.append(f"vllm {vllm.__version__} != {want_vllm}")
if not torch.__version__.startswith("2.10.0") or torch.version.cuda != "12.8":
    fail.append(f"torch {torch.__version__} / CUDA {torch.version.cuda} is not 2.10.0 / 12.8")
if not torch.cuda.is_available() or torch.cuda.device_count() != want_gpus:
    fail.append(f"torch sees {torch.cuda.device_count() if torch.cuda.is_available() else 0} GPU(s), expected {want_gpus}")
else:
    for d in range(want_gpus):
        major, minor = torch.cuda.get_device_capability(d)
        arch = f"sm_{major}{minor}"
        if arch not in torch.cuda.get_arch_list():
            fail.append(f"GPU {d} is {arch}; this torch build has {torch.cuda.get_arch_list()}")
            continue
        x = torch.ones(1024, device=f"cuda:{d}", dtype=torch.float16)
        if float((x * 2).sum().item()) != 2048.0:
            fail.append(f"GPU {d}: a float16 kernel returned a wrong result")
        print(f"  cuda:{d} {torch.cuda.get_device_name(d)} {arch} ok")
for f in fail:
    print("FAIL", f)
sys.exit(1 if fail else 0)
PY
  hub_fetch "$ENGINE_VENV/bin/python" verify || fail=1
  [ "$fail" -eq 0 ] || die "self-check failed (see FAIL lines above)"
  echo "SELF-CHECK OK: engine node ($GPU_CLASS x$GPU_COUNT, driver $DRIVER_VERSION, vllm $VLLM_VERSION)"
}

self_check_scheduler() {
  step "self-check (scheduler)"
  local fail=0
  mountpoint -q "$MYDATA" || { echo "FAIL $MYDATA not mounted"; fail=1; }
  env -i HOME="$HOME" PATH="$RB_PATH" CUDA_VISIBLE_DEVICES= "$SCHED_VENV/bin/python" - "$REPO" "$AP_DIR" "$BR_DIR" \
    "$LLM_BLENDER_COMMIT" <<'PY' || fail=1
import importlib.metadata as md, json, os, platform, re, sys
repo, ap, br, blender_commit = sys.argv[1:5]
fail = []
import routebalance, routebalance.server, routebalance.estimate.knn, routebalance.baselines.routers  # noqa: F401
import torch
print(f"python {platform.python_version()}  routebalance {md.version('routebalance')} from {routebalance.__file__}  "
      f"torch {torch.__version__}")
if not platform.python_version().startswith("3.10."):
    fail.append("python is not 3.10")
if not os.path.realpath(routebalance.__file__).startswith(os.path.realpath(repo) + os.sep):
    fail.append(f"routebalance is imported from {routebalance.__file__}, not {repo}")
text = open(os.path.join(repo, "pyproject.toml")).read()
deps = re.search(r"^dependencies = \[(.*?)^\]", text, re.S | re.M).group(1)
pins = re.findall(r'^\s*"([A-Za-z0-9_.-]+)==([^"]+)"', deps, re.M)
if not pins:
    fail.append("no pins found in pyproject.toml")
for name, want in pins:
    try:
        have = md.version(name)
    except md.PackageNotFoundError:
        fail.append(f"{name} is not installed")
        continue
    if have != want:
        fail.append(f"{name} {have} != {want}")
print(f"  {len(pins)} pyproject pins checked")
sys.path.insert(0, ap)
import simple_cluster_router  # noqa: E402,F401  upstream's module-level imports must resolve here
if os.path.realpath(os.path.dirname(simple_cluster_router.__file__)) != os.path.realpath(ap):
    fail.append(f"simple_cluster_router came from {simple_cluster_router.__file__}")
sys.path.insert(0, br)
import hybrid_llm.pair_ranker.model_util  # noqa: E402  upstream BEST-Route and LLM-Blender must import here
if not os.path.realpath(hybrid_llm.pair_ranker.model_util.__file__).startswith(os.path.realpath(br) + os.sep):
    fail.append(f"hybrid_llm came from {hybrid_llm.pair_ranker.model_util.__file__}")
url = json.loads(md.distribution("llm-blender").read_text("direct_url.json") or "{}")
if url.get("vcs_info", {}).get("commit_id") != blender_commit:
    fail.append(f"llm-blender is not at {blender_commit}: {url}")
for f in fail:
    print("FAIL", f)
sys.exit(1 if fail else 0)
PY
  [ "$(git -C "$AP_DIR" rev-parse HEAD)" = "$AP_COMMIT" ] || { echo "FAIL AvengersPro HEAD"; fail=1; }
  [ "$(md5sum "$AP_DIR/$AP_FILE" | cut -d' ' -f1)" = "$AP_PATCHED_MD5" ] || { echo "FAIL AvengersPro $AP_FILE md5"; fail=1; }
  [ "$(git -C "$BR_DIR" rev-parse HEAD)" = "$BR_COMMIT" ] || { echo "FAIL best-route-llm HEAD"; fail=1; }
  hub_fetch "$SCHED_VENV/bin/python" verify || fail=1
  env -i HOME="$HOME" PATH="$RB_PATH" TIKTOKEN_CACHE_DIR="$RB_TIKTOKEN" "$SCHED_VENV/bin/python" -c \
    'import tiktoken; tiktoken.get_encoding("cl100k_base")' || { echo "FAIL tiktoken cl100k_base not cached"; fail=1; }
  [ "$fail" -eq 0 ] || die "self-check failed (see FAIL lines above)"
  # the specs node_ctl.sh sched-start re-checks (refs/main == pin) before every scheduler launch
  printf '%s\n' "${MODELS[@]}" ${TOKENIZERS[@]+"${TOKENIZERS[@]}"} > "$SCHED_VENV/rb-setup.hub"
  echo "SELF-CHECK OK: scheduler ($SCHED_VENV, AvengersPro $AP_COMMIT patched, best-route-llm $BR_COMMIT)"
}

main() {
  set -euo pipefail
  parse_args "$@"
  mkdir -p "$STATE_DIR"
  chmod 700 "$STATE_DIR"
  LOG="$STATE_DIR/setup.$ROLE.$(date -u +%Y%m%dT%H%M%SZ).log"
  exec > >(tee -a "$LOG") 2>&1
  say "log: $LOG"
  if [ "$ROLE" = vsr ]; then
    preflight
    # the scheduler role ran on this node first (its apt upgrade, packages and /mydata): this role upgrades nothing
    grep -q '^complete ' "$SCHED_VENV/rb-setup.spec" 2>/dev/null \
      || die "the vsr role runs on the scheduler's node after its scheduler role ($SCHED_VENV is not complete)"
    docker_engine
    vsr_venv
    vsr_images
    vsr_models
    self_check_vsr
    return 0
  fi
  check_token_file
  preflight
  base_packages
  detect_gpu
  mount_mydata
  if [ "$ROLE" = engine ]; then
    cuda_toolkit
    nvidia_driver
    mig_off
    engine_venv
    step "Hub snapshots -> $RB_HF_HOME"
    hub_fetch "$ENGINE_VENV/bin/python" download || die "Hub download"
    self_check_engine
  else
    scheduler_venv
    avengers_pro
    best_route_llm
    step "Hub snapshots -> $RB_HF_HOME"
    hub_fetch "$SCHED_VENV/bin/python" download || die "Hub download"
    # upstream AvengersPro's tokenizer (tiktoken fetches it on first use): cached on /mydata, not /tmp, so
    # the scheduler needs no network at serve time; tiktoken checks the file against its own pinned hash
    env -i HOME="$HOME" PATH="$RB_PATH" TIKTOKEN_CACHE_DIR="$RB_TIKTOKEN" "$SCHED_VENV/bin/python" -c \
      'import tiktoken; tiktoken.get_encoding("cl100k_base")' || die "tiktoken cl100k_base download"
    self_check_scheduler
    if [ "$WITH_TESTS" = 1 ]; then test_venv; fi
  fi
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  main "$@"
fi
