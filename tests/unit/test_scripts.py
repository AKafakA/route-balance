"""The shell scripts: syntax, the pure parts of setup.sh, and the argument checks that fail before any ssh.

Nothing here touches the machine: setup.sh is sourced (its main only runs when executed) and every
script is run with arguments that make it stop before it would reach a node.
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from routebalance.hw import HW
from tools.engine_plan import _SPEC

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = sorted((REPO / "scripts").glob("*.sh"))
SETUP = REPO / "scripts" / "setup.sh"


def bash(script: str, env: dict | None = None, cwd: Path = REPO) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, cwd=cwd,
                          env={**os.environ, **(env or {})}, timeout=60)


def test_scripts_exist_and_parse():
    names = {p.name for p in SCRIPTS}
    assert {"setup.sh", "deploy.sh", "stop_engines.sh", "stop_scheduler.sh", "setup_fleet.sh", "node_ctl.sh",
            "node_layout.sh", "lib_operator.sh"} <= names
    for p in SCRIPTS:
        r = subprocess.run(["bash", "-n", str(p)], capture_output=True, text=True)
        assert r.returncode == 0, f"{p.name}: {r.stderr}"


def test_setup_gpu_table_matches_the_hardware_map():
    rows = {}
    for dev in re.findall(r"^\s+([0-9a-f]{4})\) echo", SETUP.read_text(), re.M):
        r = bash(f"source scripts/setup.sh && gpu_row {dev}")
        assert r.returncode == 0, r.stderr
        gpu, hw, count, arch, disk = r.stdout.split()
        rows[hw] = (gpu, int(count))
        assert arch == ("volta" if gpu == "v100" else "ampere") and disk.startswith("/dev/")
    assert rows == HW
    assert bash("source scripts/setup.sh && gpu_row 1b38").returncode == 1


def test_setup_spec_pattern_is_the_plans():
    (sh,) = re.findall(r"^SPEC_RE='(.+)'$", SETUP.read_text(), re.M)
    assert sh == _SPEC


@pytest.mark.parametrize("args, message", [
    ("", "--role must be engine or scheduler"),
    ("--role worker --hf-token-file t --model a/b@" + "0" * 40, "--role must be"),
    ("--role engine --model a/b@" + "0" * 40, "--hf-token-file is required"),
    ("--role engine --hf-token-file t", "at least one --model"),
    ("--role engine --hf-token-file t --model a/b@main", "40-hex"),
    ("--role engine --hf-token-file t --model a/b", "40-hex"),
    ("--role engine --hf-token-file t --model a/b@" + "0" * 40 + " --tokenizer c/d@" + "1" * 40, "scheduler role"),
    ("--role engine --hf-token-file t --model a/b@" + "0" * 40 + " --bogus", "unknown argument"),
    ("--role engine --hf-token-file", "needs a value"),
    ("--role engine --hf-token-file t --model a/b@" + "0" * 40 + " --data-disk", "needs a value"),
])
def test_setup_refuses_bad_arguments(args, message):
    r = bash(f"source scripts/setup.sh && parse_args {args}")
    assert r.returncode == 1 and message in r.stderr, r.stderr


def test_setup_accepts_good_arguments():
    r = bash("source scripts/setup.sh && parse_args --role scheduler --hf-token-file t --model a/b@" + "0" * 40
             + " --tokenizer Qwen/Qwen2.5-3B@" + "1" * 40 + ' && echo "$ROLE ${#MODELS[@]} ${#TOKENIZERS[@]}"')
    assert r.returncode == 0 and r.stdout.split() == ["scheduler", "1", "1"], r.stderr


def test_setup_token_file_must_be_private(tmp_path):
    tok = tmp_path / "token"
    tok.write_text("not-a-real-token\n")
    for mode, ok in ((0o644, False), (0o640, False), (0o600, True), (0o400, True)):
        tok.chmod(mode)
        r = bash(f"source scripts/setup.sh && TOKEN_FILE={tok} && check_token_file && echo ok")
        assert (r.returncode == 0) == ok, (oct(mode), r.stderr)
        assert "not-a-real-token" not in r.stdout + r.stderr
    empty = tmp_path / "empty"
    empty.write_text("")
    empty.chmod(0o600)
    assert bash(f"source scripts/setup.sh && TOKEN_FILE={empty} && check_token_file").returncode == 1


def test_setup_never_installs_outside_its_own_venvs():
    """R1: every pip call goes through a venv this script made; nothing uses the system pip or sudo pip."""
    code = "\n".join(ln for ln in SETUP.read_text().splitlines() if not ln.lstrip().startswith("#"))
    pythons = re.findall(r"(\S+) -m pip ", code)
    assert len(pythons) >= 5, pythons                       # pip itself, vllm, the repo, check, freeze
    assert set(pythons) == {'"$dir/bin/python"', '"$ENGINE_VENV/bin/python"', '"$SCHED_VENV/bin/python"'}
    assert not re.search(r"(^|[;&|(]|\bsudo)\s*(pip3?|python3(\.10)?\s+-m\s+pip)\s", code, re.M)


def test_node_ctl_fails_loudly_on_bad_commands():
    stream = "cat scripts/node_layout.sh scripts/node_ctl.sh | bash -s -- "
    r = bash(stream + "bogus")
    assert r.returncode == 1 and "unknown command bogus" in r.stderr
    r = bash(stream + "launch only-an-id")
    assert r.returncode == 1 and "usage: launch" in r.stderr


def test_node_ctl_pid_records_survive_only_their_own_process(tmp_path):
    """A pid file names a process only while that process lives: same boot, same start time. A record from
    before a reboot, or one whose pid was reused, is not a running engine, and stop never signals it."""
    lib = "export RB_CTL_AS_LIBRARY=1; source scripts/node_layout.sh; source scripts/node_ctl.sh; "
    pidf = tmp_path / "x.pid"
    script = lib + f"""
setsid --fork bash -c "$RECORD_AND_EXEC"'; exec "$@"' _ {pidf} LOG "$BOOT_ID" sleep 30 < /dev/null > /dev/null 2>&1
for i in $(seq 1 40); do [ -s {pidf} ] && break; sleep 0.1; done
pid=$(pid_of {pidf}); [ -n "$pid" ] || {{ echo "no live pid"; exit 1; }}
read -r p log boot start < {pidf}
[ "$log" = LOG ] && [ "$p" = "$pid" ] || {{ echo "bad record"; exit 1; }}
echo "$p $log $boot 1" > {pidf}.start;  [ -z "$(pid_of {pidf}.start)" ] || {{ echo "start time ignored"; exit 1; }}
echo "$p $log other-boot $start" > {pidf}.boot; [ -z "$(pid_of {pidf}.boot)" ] || {{ echo "boot ignored"; exit 1; }}
stop_group {pidf}.boot stale; kill -0 "$pid" || {{ echo "stale record stopped a live process"; exit 1; }}
stop_group {pidf} sleeper; ! kill -0 "$pid" 2>/dev/null || {{ echo "not stopped"; exit 1; }}
[ ! -e {pidf} ] && echo DONE
"""
    r = bash(script)
    assert r.returncode == 0 and r.stdout.strip().endswith("DONE"), r.stdout + r.stderr


def test_node_ctl_requires_refs_main_at_the_pin(tmp_path):
    lib = f"export RB_CTL_AS_LIBRARY=1; source scripts/node_layout.sh; source scripts/node_ctl.sh; RB_HF_HOME={tmp_path}; "
    rev, other = "1" * 40, "2" * 40
    repo = tmp_path / "hub" / "models--org--enc"
    (repo / "refs").mkdir(parents=True)
    (repo / "snapshots" / rev).mkdir(parents=True)
    specs = tmp_path / "hub.txt"
    specs.write_text(f"org/enc@{rev}\n")
    for ref, ok in ((rev, True), (other, False)):
        (repo / "refs" / "main").write_text(ref)
        r = bash(lib + f"hub_pins_ok {specs} && echo OK")
        assert (r.returncode == 0 and r.stdout.strip() == "OK") == ok, r.stdout + r.stderr
    (repo / "refs" / "main").unlink()
    assert bash(lib + f"hub_pins_ok {specs}").returncode == 1
    assert bash(lib + f"hub_pins_ok {tmp_path}/missing").returncode == 1


@pytest.mark.parametrize("cmd, message", [
    ("scripts/deploy.sh", "usage: deploy.sh"),
    ("scripts/deploy.sh engines --fleet configs/fleet_qwen.json", "engines needs --hosts"),
    ("scripts/deploy.sh engines --fleet configs/fleet_qwen.json --hosts configs/hosts.example.json "
     "--engine-args configs/engine_args_gemma.json --record REC", "invalid plan"),
    ("scripts/deploy.sh scheduler --fleet configs/fleet_qwen.json --hosts configs/hosts.example.json --node a100-0 "
     "--config configs/fleet_qwen.json --launch configs/scheduler_launch.example.json --artifacts A --tree T --log-dir L "
     "--record REC",
     "invalid launch file"),
    ("scripts/stop_engines.sh --fleet configs/fleet_qwen.json", "usage: stop_engines.sh"),
    ("scripts/stop_engines.sh --fleet configs/fleet_qwen.json --hosts local/none.json "
     "--engine-args configs/engine_args_qwen.json", "no such file: " + str(REPO / "local/none.json")),
    ("scripts/stop_scheduler.sh --fleet configs/fleet_qwen.json --hosts configs/hosts.example.json --node a30-99",
     "serves no instance"),
    ("scripts/setup_fleet.sh --fleet configs/fleet_qwen.json", "missing argument"),
])
def test_operator_scripts_stop_before_ssh(tmp_path, cmd, message):
    r = bash(f"bash {cmd}", env={"PY": sys.executable, "HOME": str(tmp_path)})
    assert r.returncode == 1 and message in r.stderr, r.stdout + r.stderr
    assert "REC" not in {p.name for p in REPO.iterdir()}


@pytest.mark.parametrize("disks, message", [
    ({"a30-99": "/dev/sdc"}, "is not a node of the fleet"),
    ({"a30-0": "sdc"}, "is not a /dev disk"),
    ({}, "non-empty"),
])
def test_setup_fleet_checks_the_data_disk_table_before_ssh(tmp_path, disks, message):
    token = tmp_path / "tok"
    token.write_text("x")
    token.chmod(0o600)
    table = tmp_path / "disks.json"
    table.write_text(json.dumps(disks))
    r = bash(f"bash scripts/setup_fleet.sh --fleet configs/fleet_qwen.json --hosts configs/hosts.example.json "
             f"--engine-args configs/engine_args_qwen.json --scheduler-node a100-0 --hf-token-file {token} "
             f"--tree rb --log-dir {tmp_path}/logs --data-disks {table}", env={"PY": sys.executable, "HOME": str(tmp_path)})
    assert r.returncode == 1 and "invalid --data-disks" in r.stderr and message in r.stdout + r.stderr, r.stdout + r.stderr


def test_setup_documents_the_data_disk_rule():
    text = (REPO / "scripts/setup.sh").read_text()
    assert "--data-disk" in text and "candidate_disks" in text and "has partitions; refusing" in text
