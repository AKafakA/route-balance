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
    ("--role engine --hf-token-file t --model a/b@" + "0" * 40 + " --with-tests", "--with-tests is for the scheduler"),
])
def test_setup_refuses_bad_arguments(args, message):
    r = bash(f"source scripts/setup.sh && parse_args {args}")
    assert r.returncode == 1 and message in r.stderr, r.stderr


def test_setup_accepts_good_arguments():
    r = bash("source scripts/setup.sh && parse_args --role scheduler --hf-token-file t --model a/b@" + "0" * 40
             + " --tokenizer Qwen/Qwen2.5-3B@" + "1" * 40 + ' && echo "$ROLE ${#MODELS[@]} ${#TOKENIZERS[@]}"')
    assert r.returncode == 0 and r.stdout.split() == ["scheduler", "1", "1"], r.stderr
    r = bash("source scripts/setup.sh && parse_args --role scheduler --hf-token-file t --model a/b@" + "0" * 40
             + ' --with-tests && echo "$WITH_TESTS"')
    assert r.returncode == 0 and r.stdout.split() == ["1"], r.stderr


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
    assert set(pythons) == {'"$dir/bin/python"', '"$ENGINE_VENV/bin/python"', '"$SCHED_VENV/bin/python"',
                            '"$TEST_VENV/bin/python"'}      # each made by new_venv (--with-tests: the test venv)
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
    lib = "source scripts/node_layout.sh; source scripts/node_ctl.sh; "
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
    lib = f"source scripts/node_layout.sh; source scripts/node_ctl.sh; RB_HF_HOME={tmp_path}; "
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
    ("scripts/setup_fleet.sh --with-tests --fleet configs/fleet_qwen.json", "missing argument"),   # a flag, no value
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
    r = bash(f"bash scripts/setup_fleet.sh --fleet configs/fleet_qwen.json --hub configs/scheduler_hub.json "
             f"--hosts configs/hosts.example.json "
             f"--engine-args configs/engine_args_qwen.json --scheduler-node a100-0 --hf-token-file {token} "
             f"--tree rb --log-dir {tmp_path}/logs --data-disks {table}", env={"PY": sys.executable, "HOME": str(tmp_path)})
    assert r.returncode == 1 and "invalid --data-disks" in r.stderr and message in r.stdout + r.stderr, r.stdout + r.stderr


def test_setup_documents_the_data_disk_rule():
    text = (REPO / "scripts/setup.sh").read_text()
    assert "--data-disk" in text and "candidate_disks" in text and "has partitions; refusing" in text


SERVE_OWN_START = r"""
import http.server, os, sys
ticks = os.sysconf("SC_CLK_TCK")
btime = next(int(l.split()[1]) for l in open("/proc/stat") if l.startswith("btime"))
start = btime + int(open("/proc/self/stat").read().rsplit(")", 1)[1].split()[19]) / ticks
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = f"# TYPE process_start_time_seconds gauge\nprocess_start_time_seconds {start}\n".encode()
        self.send_response(200); self.end_headers(); self.wfile.write(body)
    def log_message(self, *a): pass
http.server.HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
"""


def test_node_ctl_started_reports_what_the_recorded_engine_process_executes(tmp_path):
    """D48: the start time on the engine's port must be the recorded pid's, and the record gets its GPUs and
    command line from /proc; another process answering on the port is refused."""
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    (tmp_path / "srv.py").write_text(SERVE_OWN_START)
    lib = f"source scripts/node_layout.sh; source scripts/node_ctl.sh; RB_RUN={tmp_path}; "
    launch = (f'setsid --fork bash -c "$RECORD_AND_EXEC"\'; exec "$@"\' _ {tmp_path}/engine.{{id}}.pid LOG "$BOOT_ID" '
              f'env CUDA_VISIBLE_DEVICES=2,3 {{cmd}} < /dev/null > /dev/null 2>&1\n'
              f'for i in $(seq 1 50); do [ -s {tmp_path}/engine.{{id}}.pid ] && break; sleep 0.1; done\n')
    cleanup = (f"trap 'stop_group {tmp_path}/engine.E.pid E > /dev/null; "
               f"stop_group {tmp_path}/engine.S.pid S > /dev/null' EXIT\n")
    script = lib + cleanup + launch.format(id="E", cmd=f"{sys.executable} {tmp_path}/srv.py {port}") + "sleep 1\n" + \
        launch.format(id="S", cmd="sleep 30") + f"""
for i in $(seq 1 50); do curl -sf -m 1 http://127.0.0.1:{port}/metrics > /dev/null && break; sleep 0.1; done
cmd_started E {port}
( cmd_started S {port} ) 2>&1 | tail -1 || true
"""
    r = bash(script)
    assert r.returncode == 0, r.stdout + r.stderr
    lines = r.stdout.strip().splitlines()
    fields = lines[0].split("\t")
    assert fields[:2] == ["started", "E"] and float(fields[2]) > 1.7e9 and fields[3] == "2,3", r.stdout + r.stderr
    assert fields[4] == f"{sys.executable} {tmp_path}/srv.py {port}"
    assert "another process answers" in lines[-1]


def test_the_best_route_pins_agree_across_setup_pyproject_hub_and_presets():
    """D79: setup.sh's LLM-Blender commit is pyproject's; the backbone revision the presets load is the one the
    scheduler's Hub pins fetch; the checkout setup.sh clones is the one the presets pin by digest (the digest of
    fc9b896's modules was computed from that checkout, 09-26)."""
    setup = SETUP.read_text()
    commit = re.search(r"^LLM_BLENDER_COMMIT=([0-9a-f]{40})", setup, re.M).group(1)
    assert f"LLM-Blender.git@{commit}\"" in (REPO / "pyproject.toml").read_text()
    presets = json.loads((REPO / "configs" / "presets.json").read_text())
    kw = presets["parts"]["br.router"]["kwargs"]
    hub = json.loads((REPO / "configs" / "scheduler_hub.json").read_text())
    assert f"{kw['base_model']}@{kw['base_revision']}" in hub["models"]
    assert re.search(r"^BR_COMMIT=fc9b896a814acf7c00565fbea0803a8c20812702 ", setup, re.M)
    assert kw["upstream_sha256"] == "85279e1f6080d87db95dcbded46d7713fcf2728b82339c3b6a30ef3b56579005"



@pytest.mark.parametrize("modes,rc,message", [
    ("Disabled Disabled", 1, ""), ("Disabled Enabled", 0, ""),
    ("Disabled", 1, "the MIG mode of 1 GPU(s), not 2"), ("Disabled [N/A]", 1, "unexpected MIG mode"),
    ("FAIL", 1, "could not read the MIG mode"), ("", 1, "the MIG mode of 0 GPU(s), not 2"),
])
def test_setup_never_reads_an_unread_mig_mode_as_off(tmp_path, modes, rc, message):
    smi = tmp_path / "nvidia-smi"
    smi.write_text("#!/bin/bash\n" + ("exit 9\n" if modes == "FAIL" else "".join(f"echo '{m}'\n" for m in modes.split())))
    smi.chmod(0o755)
    r = bash(f"source scripts/setup.sh && PATH={tmp_path}:$PATH && GPU_COUNT=2 && mig_enabled")
    assert r.returncode == rc and message in r.stderr
    if not message:                                                # MIG off or on is read, never a failed setup
        assert "SETUP FAILED" not in r.stderr
