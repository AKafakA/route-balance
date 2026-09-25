"""The campaign scripts refuse to start without every argument, and a cell the render does not know never runs."""
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _run(script, *args, env_py=True):
    env = {**os.environ}
    if env_py:
        env["PY"] = sys.executable
    else:
        env.pop("PY", None)
    return subprocess.run(["bash", str(REPO / "scripts" / script), *args], capture_output=True, text=True, env=env,
                          cwd=REPO, timeout=60)


def test_the_scripts_parse():
    for script in ("run_cell.sh", "run_scope.sh"):
        assert subprocess.run(["bash", "-n", str(REPO / "scripts" / script)]).returncode == 0


def test_every_argument_is_required():
    r = _run("run_cell.sh", "--render", "x")
    assert r.returncode == 2 and "usage" in r.stderr
    r = _run("run_scope.sh", "--render", "x", "--rows", "G1")
    assert r.returncode == 2 and "usage" in r.stderr
    assert _run("run_cell.sh", env_py=False).returncode != 0          # PY must be named


def test_a_bad_max_invalid_is_refused():
    args = ["--render", "r", "--rows", "G1", "--max-invalid", "many", "--scheduler", "u", "--scheduler-log-dir", "l",
            "--fleet", "f", "--hosts", "h", "--data-root", "d", "--cells-root", "c", "--mirror", "m"]
    assert _run("run_scope.sh", *args).returncode == 2


def _bash(tmp_path, body):
    script = tmp_path / "t.sh"
    script.write_text("set -euo pipefail\n" + (REPO / "scripts" / "mirror.sh").read_text() + body)
    return subprocess.run(["bash", str(script)], capture_output=True, text=True, cwd=tmp_path, timeout=60)


def test_the_mirror_never_writes_over_another_run_and_proves_identity(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    (src / "a").write_text("1")
    (src / "b").write_text("2")
    r = _bash(tmp_path, 'fresh src dst/cell && echo absent-ok\nmirror src dst/cell && echo copied\n'
                        'fresh src dst/cell && echo identical-ok\n'
                        'echo other > dst/cell/a\nfresh src dst/cell || echo changed-refused\n'
                        'cp src/a dst/cell/a\necho x > dst/cell/extra\nfresh src dst/cell || echo extra-refused\n'
                        'mirror src dst/cell || echo mirror-refuses-extra\n')
    assert r.stdout.split() == ["absent-ok", "copied", "identical-ok", "changed-refused", "extra-refused",
                                "mirror-refuses-extra"], r.stderr
    r = _bash(tmp_path, 'mirror src nowhere/deeper/cell || echo failed-copy-refused\n')
    assert "failed-copy-refused" in r.stdout
    (tmp_path / "root" / "c1").mkdir(parents=True)
    (tmp_path / "root" / "c1" / "r").write_text("new run")
    (tmp_path / "mir" / "c1").mkdir(parents=True)
    (tmp_path / "mir" / "c1" / "r").write_text("old run")        # a retired run's copy
    r = _bash(tmp_path, 'mirror_cell root mir c1 || echo refused\n')
    assert "refused" in r.stdout and "holds another run" in r.stderr
    assert (tmp_path / "mir" / "c1" / "r").read_text() == "old run"


def test_an_unknown_render_stops_before_anything_is_sent(tmp_path):
    args = ["--render", str(tmp_path / "none"), "--cell", "x__l12__s1", "--scheduler", "http://127.0.0.1:9",
            "--scheduler-log-dir", str(tmp_path), "--fleet", "f", "--hosts", "h", "--data-root", str(tmp_path),
            "--cells-root", str(tmp_path / "cells"), "--mirror", str(tmp_path / "mirror")]
    r = _run("run_cell.sh", *args)
    assert r.returncode == 2 and "prepare refused" in r.stderr
    assert not (tmp_path / "cells").exists() and not (tmp_path / "mirror").exists()
