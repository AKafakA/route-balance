"""The runner's command line refuses incomplete arguments, a cell the render does not know never runs, the second
location never takes another run's place, and the benchmark client starts offline in a clean environment."""
import subprocess
import sys
from pathlib import Path

from runner import cell, mirror

REPO = Path(__file__).resolve().parents[2]


def _runner(*args):
    return subprocess.run([sys.executable, "-m", "runner", *args], capture_output=True, text=True, cwd=REPO, timeout=60)


SETUP = ["--scheduler", "u", "--scheduler-log-dir", "l", "--fleet", "f", "--hosts", "h", "--engines-record", "e",
         "--data-root", "d", "--cells-root", "c", "--mirror", "m"]


def test_every_argument_is_required():
    r = _runner("cell", "--render", "x")
    assert r.returncode == 2 and "required" in r.stderr
    r = _runner("rows", "--render", "x", "--rows", "G1")
    assert r.returncode == 2 and "required" in r.stderr


def test_a_bad_max_invalid_or_a_relative_log_dir_with_a_fresh_scheduler_is_refused():
    assert _runner("rows", "--render", "r", "--rows", "G1", "--max-invalid", "many", *SETUP).returncode == 2
    r = _runner("rows", "--render", "r", "--rows", "G1", "--max-invalid", "3", "--fresh-scheduler", "run", *SETUP)
    assert r.returncode == 2 and "absolute" in r.stderr


def test_an_unknown_render_stops_before_anything_is_sent(tmp_path):
    args = ["--render", str(tmp_path / "none"), "--cell", "x__l12__s1", "--scheduler", "http://127.0.0.1:9",
            "--scheduler-log-dir", str(tmp_path), "--fleet", "f", "--hosts", "h", "--engines-record", "e",
            "--data-root", str(tmp_path), "--cells-root", str(tmp_path / "cells"), "--mirror", str(tmp_path / "mirror")]
    r = _runner("cell", *args)
    assert r.returncode == 2 and "prepare refused" in r.stdout
    without = [a for a in args if a not in ("--engines-record", "e")]            # the deploy record is required
    assert "required" in _runner("cell", *without).stderr
    assert not (tmp_path / "cells").exists() and not (tmp_path / "mirror").exists()


def test_the_mirror_never_writes_over_another_run_and_proves_identity(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    (src / "a").write_text("1")
    (src / "b").write_text("2")
    cellm = str(dst / "cell")
    assert mirror.fresh(str(src), cellm)                                  # absent
    assert mirror.mirror(str(src), cellm)
    assert mirror.fresh(str(src), cellm)                                  # identical
    (dst / "cell" / "a").write_text("other")
    assert not mirror.fresh(str(src), cellm)                              # a file differs
    (dst / "cell" / "a").write_text("1")
    (dst / "cell" / "extra").write_text("x")
    assert not mirror.fresh(str(src), cellm) and not mirror.mirror(str(src), cellm)   # an extra file
    (dst / "cell" / "extra").unlink()
    (dst / "cell" / "b").unlink()
    subprocess.run(["touch", "-d", "2001-01-01", cellm], check=True)
    assert mirror.fresh(str(src), cellm)                                  # an interrupted copy resumes
    assert not mirror.mirror(str(src), str(tmp_path / "nowhere" / "deeper" / "cell"))
    (tmp_path / "root" / "c1").mkdir(parents=True)
    (tmp_path / "root" / "c1" / "r").write_text("new run")
    (tmp_path / "mir" / "c1").mkdir(parents=True)
    (tmp_path / "mir" / "c1" / "r").write_text("old run")                # a retired run's copy
    said = []
    assert not mirror.mirror_cell(tmp_path / "root", str(tmp_path / "mir"), "c1", said.append)
    assert any("holds another run" in s for s in said)
    assert (tmp_path / "mir" / "c1" / "r").read_text() == "old run"


def test_the_client_runs_offline_from_the_pinned_snapshots_in_a_clean_environment():
    """The benchmark's tokenizer is the one setup.sh pinned under RB_HF_HOME, never a fresh download, and the client
    inherits nothing from the caller, like the scheduler and the engines (scripts/node_ctl.sh)."""
    path = "/usr/local/cuda-12.8/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"     # the node layout's
    assert cell.node_env() == {"HOME": str(Path.home()), "PATH": path, "LANG": "C.UTF-8", "HF_HOME": "/mydata/hf",
                               "HF_HUB_OFFLINE": "1"}
    layout = (REPO / "scripts" / "node_layout.sh").read_text()
    assert "RB_HF_HOME=$MYDATA/hf" in layout and "RB_PATH=$CUDA_DIR/bin:" in layout


def test_a_row_list_that_names_no_row_is_refused():
    assert _runner("rows", "--render", "r", "--rows", ",", "--max-invalid", "3", *SETUP).returncode == 2


def test_an_empty_required_value_is_refused():
    """An empty --cells-root would be the repository itself: refused like a missing one."""
    r = _runner("rows", "--render", "r", "--rows", "G1", "--max-invalid", "3",
                *[v if v != "c" else "" for v in SETUP])
    assert r.returncode == 2 and "must not be empty" in r.stderr


def test_exit_statuses_are_the_shells(tmp_path):
    """A gate killed by a signal records 128 + the signal; an unwritable log and a missing command are failures
    (1 and 127), never a crash of the runner."""
    assert cell._to_file(["bash", "-c", "kill -9 $$"], tmp_path / "killed.log") == 137
    assert cell._to_file(["true"], tmp_path / "no-such-dir" / "x.log") == 1
    assert cell._to_file([str(tmp_path / "no-such-command")], tmp_path / "missing.log") == 127


def test_a_mirror_without_rsync_fails_instead_of_raising(tmp_path, monkeypatch):
    (tmp_path / "root" / "c1").mkdir(parents=True)
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    said = []
    assert not mirror.mirror_cell(tmp_path / "root", str(tmp_path / "mir"), "c1", said.append)
    assert any("exit 127" in s for s in said)
