"""The campaign scripts refuse to start without every argument, and a cell the render does not know never runs."""
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _run(script, *args, env_py=True, extra_env=None):
    env = {**os.environ, **(extra_env or {})}
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
            "--fleet", "f", "--hosts", "h", "--engines-record", "e", "--data-root", "d", "--cells-root", "c",
            "--mirror", "m"]
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
    r = _bash(tmp_path, 'rm -r dst/cell\nmirror src dst/cell\nrm dst/cell/b\ntouch -d 2001-01-01 dst/cell\n'
                        'fresh src dst/cell && echo interrupted-copy-resumes\n')   # a partial copy's directory time
    assert "interrupted-copy-resumes" in r.stdout, r.stderr
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
            "--scheduler-log-dir", str(tmp_path), "--fleet", "f", "--hosts", "h", "--engines-record", "e",
            "--data-root", str(tmp_path), "--cells-root", str(tmp_path / "cells"), "--mirror", str(tmp_path / "mirror")]
    r = _run("run_cell.sh", *args)
    assert r.returncode == 2 and "prepare refused" in r.stderr
    exported = _run("run_cell.sh", *args, extra_env={"RC": "0"})     # an RC in the caller's shell changes nothing
    assert exported.returncode == 2 and "prepare refused" in exported.stderr
    without = [a for a in args if a not in ("--engines-record", "e")]            # the deploy record is required
    assert "usage" in _run("run_cell.sh", *without).stderr
    assert not (tmp_path / "cells").exists() and not (tmp_path / "mirror").exists()


def test_the_runner_reads_its_tokenizer_offline_from_the_pinned_snapshots():
    """The benchmark's tokenizer is the one setup.sh pinned under RB_HF_HOME, never a fresh download of main, and
    the runner starts in a clean environment (nothing inherited from the caller, like the scheduler)."""
    text = (REPO / "scripts" / "run_cell.sh").read_text()
    lines = text.splitlines()
    (i,) = [k for k, ln in enumerate(lines) if "-m bench.runner" in ln]
    assert lines[i - 1].strip() == ('env -i HOME="$HOME" PATH="$RB_PATH" LANG=C.UTF-8 HF_HOME="$RB_HF_HOME" '
                                    'HF_HUB_OFFLINE=1 \\')
    assert lines[i].strip().startswith('"$PY" -m bench.runner')
    assert "\nRC=2 " in text and 'exit "$RC"' in text and "${RC:-" not in text   # fail()'s code is never inherited
    assert 'source "$REPO/scripts/node_layout.sh"' in text
    assert "RB_HF_HOME=$MYDATA/hf" in (REPO / "scripts" / "node_layout.sh").read_text()


def test_every_cell_reads_its_engines_before_and_after():
    """D48: run_cell.sh reads every engine before the warm-up (D132: so the traffic check covers the cache reset and
    its proof) and again after the runner."""
    text = (REPO / "scripts" / "run_cell.sh").read_text()
    outs = [ln for ln in text.splitlines() if "--out \"$ATTEMPT/engines_" in ln]
    assert [o.split("engines_")[1].split(".json")[0] for o in outs] == ["before", "after"]
    assert text.index("engines_before.json") < text.index("drv warmup") < text.index("-m bench.runner") \
        < text.index("engines_after.json")


def test_every_cell_checks_for_a_fresh_scheduler_before_its_configuration():
    """D132: run_cell.sh refuses a used scheduler after the drain, before it posts the cell's configuration."""
    text = (REPO / "scripts" / "run_cell.sh").read_text()
    assert text.index("drv drain") < text.index("drv fresh") < text.index("drv post-config")
