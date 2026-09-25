"""The hygiene gate must fail on every class of leak it exists to catch, and pass a clean tree.

Leak fixtures are assembled at runtime from fragments, so this file contains no leak itself and the
gate needs no exemption for it (a pasted real credential here would be caught like anywhere else).
"""
import subprocess
from pathlib import Path

import pytest

from gates.hygiene import main, scan_tree

J = "".join
CODENAME = J(["ca", "ra"])
NODE = J(["d75", "25-", "99s", "99999"])          # CloudLab node-name shape, fictional number
NODE6 = J(["c24", "0g5-", "999999"])
LEAKS = {
    "codename_word": (f"from pkg.{CODENAME} import x\n", "codename"),
    "codename_camel": (f"class {CODENAME.capitalize()}Scheduler: pass\n", "codename"),
    "codename_env": (f'{{"env": "{CODENAME.upper()}_STATIC_LAT"}}\n', "codename"),
    "host_domain": (J(["ssh node0.", "cloud", "lab.us\n"]), "host"),
    "host_node": (f"ssh {NODE}\n", "host"),
    "host_node6": (f"ssh {NODE6}\n", "host"),
    "host_campus": (J(["ssh login-", "icelake.hpc.", "cam", ".ac.uk\n"]), "host"),
    "ip_private": (J(['{"ip": "10.', '0.0.7"}\n']), "ip"),
    "ip_campus": (J(['{"ip": "128.', '105.0.7"}\n']), "ip"),
    "secret_hf": (J(["TOKEN=hf_", "a" * 34, "\n"]), "secret"),
    "secret_sk_proj": (J(["OPENAI_API_KEY=sk-", "proj-", "b" * 40, "\n"]), "secret"),
    "secret_gho": (J(["gh", "o_", "c" * 36, "\n"]), "secret"),
    "path_home": (J(['p = "/ho', 'me/someone"\n']), "abs_path"),
    "path_users": (J(["cd /us", "ers/someone/x\n"]), "abs_path"),
    "path_mac": (J(["cd /Us", "ers/Someone/x\n"]), "abs_path"),
    "path_rds": (J(["cd /rds", "/user/someone/x\n"]), "abs_path"),
    "path_scratch": (J(["cd /local", "/scratch/tmp/x\n"]), "abs_path"),
}


def _tree(root: Path, files: dict) -> Path:
    for rel, content in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        (p.write_bytes if isinstance(content, bytes) else p.write_text)(content)
    return root


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


def test_clean_tree_passes(tmp_path):
    root = _tree(tmp_path, {"routebalance/x.py": "def f():\n    return 1\n", "README.md": "RouteBalance\n"})
    assert scan_tree(root) == []


@pytest.mark.parametrize("name", sorted(LEAKS))
def test_each_leak_class_fails(tmp_path, name):
    text, kind = LEAKS[name]
    root = _tree(tmp_path, {"scripts/f.txt": text})
    assert kind in {f.kind for f in scan_tree(root)}


@pytest.mark.parametrize("name", [
    "configs/x.json.bak", "configs/x.json.bak2", "configs/x.json.bak.20260606", "configs/y.STALE_v1",
    "routebalance/score_v2.py", "routebalance/score_v3.py", "routebalance/score.py.orig", "routebalance/score.py~",
    "archive/old.py", "configs.bak/x.json", "routebalance_v2/x.py",
])
def test_forbidden_names_fail_for_files_and_dirs(tmp_path, name):
    root = _tree(tmp_path, {name: "x\n"})
    assert "forbidden_file" in {f.kind for f in scan_tree(root)}


def test_leaks_in_file_paths_fail(tmp_path):
    root = _tree(tmp_path, {f"configs/{NODE}.json": "{}\n", f"routebalance/{CODENAME}_pd/serve.py": "x = 1\n"})
    kinds = {f.kind for f in scan_tree(root)}
    assert {"host", "codename"} <= kinds


def test_secret_is_located_but_never_printed(tmp_path):
    text, _ = LEAKS["secret_hf"]
    root = _tree(tmp_path, {"scripts/t.sh": text})
    f = [x for x in scan_tree(root) if x.kind == "secret"][0]
    assert (f.path, f.line) == ("scripts/t.sh", 1)
    assert text.strip().split("=")[1] not in f.message


def test_undecodable_bytes_do_not_hide_a_leak(tmp_path):
    ip = J(["10.", "0.0.7"]).encode()
    root = _tree(tmp_path, {"configs/hosts.txt": b"# caf\xe9\nHOST=" + ip + b"\n"})
    assert "ip" in {f.kind for f in scan_tree(root)}


def test_caches_are_skipped_outside_git(tmp_path):
    root = _tree(tmp_path, {".venv/lib/x.py": CODENAME + "\n", "__pycache__/x.txt": CODENAME + "\n"})
    assert scan_tree(root) == []


def test_git_repo_scans_tracked_files_even_under_ignored_dirs(tmp_path):
    root = _tree(tmp_path, {".gitignore": "/local/\n", "local/hosts.json": f'{{"n": "{NODE}"}}\n',
                            "routebalance/models/estimator.py": "x = 1\n"})
    _git(root, "init", "-q")
    assert scan_tree(root) == []                       # ignored and untracked: not part of the repo
    _git(root, "add", "-f", "local/hosts.json")       # force-staged: must be checked
    assert "host" in {f.kind for f in scan_tree(root)}


def test_git_repo_skips_ignored_untracked_logs(tmp_path):
    root = _tree(tmp_path, {".gitignore": "*.log\n", "run.log": f"{NODE}\n", "README.md": "ok\n"})
    _git(root, "init", "-q")
    assert scan_tree(root) == []


def test_missing_root_is_a_usage_error(tmp_path):
    assert main([str(tmp_path / "absent")]) == 2


def test_this_repository_is_clean():
    repo = Path(__file__).resolve().parents[2]
    assert scan_tree(repo) == []


HOST_UNDERSCORE = [f"NODE_{NODE} = 1\n", f'"{NODE}_log"\n']


@pytest.mark.parametrize("text", HOST_UNDERSCORE)
def test_host_next_to_underscore_fails(tmp_path, text):
    root = _tree(tmp_path, {"scripts/f.txt": text})
    assert "host" in {f.kind for f in scan_tree(root)}


@pytest.mark.parametrize("text,kind", [
    (J(["ssh ", "someone@", "142.", "93.38.181\n"]), "ip"),
    (J(["#SBATCH -A ", "KALYV", "IANAKI-SL3-GPU\n"]), "identity"),
    (J(["squeue -u ", "wd", "312\n"]), "identity"),
])
def test_identity_and_public_login_fail(tmp_path, text, kind):
    root = _tree(tmp_path, {"scripts/f.sh": text})
    assert kind in {f.kind for f in scan_tree(root)}


@pytest.mark.parametrize("text", ["nvidia-curand-cu12==10.3.9.90\n", 'RUN = "a100-260924"\n',
                                  "scheduler = task-queue-manager-for-requests-x\n"])
def test_version_pins_and_run_ids_are_not_leaks(tmp_path, text):
    root = _tree(tmp_path, {"constraints/x.txt": text})
    assert scan_tree(root) == []


def test_assignment_ip_is_still_a_leak(tmp_path):
    root = _tree(tmp_path, {"scripts/env.sh": J(["HOST=10.", "10.1.4\n"])})
    assert "ip" in {f.kind for f in scan_tree(root)}


@pytest.mark.parametrize("name", ["configs/x.json.bak_20260606", "configs/x.json.old", "routebalance/score.py.backup",
                                  "routebalance/score_v2_fixed.py", "routebalance/score-v2.py",
                                  "routebalance/score_old.py", "old/x.py", "backup/x.py"])
def test_more_backup_names_fail(tmp_path, name):
    root = _tree(tmp_path, {name: "x\n"})
    assert "forbidden_file" in {f.kind for f in scan_tree(root)}


def test_gpu_names_in_file_names_are_fine(tmp_path):
    root = _tree(tmp_path, {"configs/calib/tpot_v100.json": "{}\n", "configs/calib/tpot_a30.json": "{}\n"})
    assert scan_tree(root) == []


def test_staged_version_is_checked_even_if_disk_is_clean(tmp_path):
    root = _tree(tmp_path, {"configs/env.sh": J(["HOST=10.", "10.1.5\n"])})
    _git(root, "init", "-q")
    _git(root, "add", "configs/env.sh")
    (root / "configs/env.sh").write_text("HOST=example\n")      # clean on disk, dirty in the index
    assert "ip" in {f.kind for f in scan_tree(root)}


def test_staged_symlink_target_is_checked(tmp_path):
    root = tmp_path
    _git(root, "init", "-q")
    (root / "data_link").symlink_to(J(["/ho", "me/someone/Block/data"]))   # dangling on purpose
    _git(root, "add", "data_link")
    assert "abs_path" in {f.kind for f in scan_tree(root)}


def test_secret_in_file_name_is_never_printed(tmp_path, capsys):
    secret = J(["sk-", "proj-", "a" * 40])
    root = _tree(tmp_path, {f"configs/{secret}.json": "{}\n"})
    assert main([str(root)]) == 1
    assert secret not in capsys.readouterr().out


@pytest.mark.parametrize("text", [J(["The head node is 10.", "10.1.1.\n"]), J(["ssh root@142.", "93.38.181.\n"])])
def test_ip_before_full_stop_fails(tmp_path, text):
    root = _tree(tmp_path, {"docs/notes.md": text})
    assert "ip" in {f.kind for f in scan_tree(root)}


@pytest.mark.parametrize("name", [J(["configs/10.", "0.0.7.json"]), J(["configs/node_10.", "0.0.7_log.json"])])
def test_ip_in_file_name_fails(tmp_path, name):
    root = _tree(tmp_path, {name: "{}\n"})
    assert "ip" in {f.kind for f in scan_tree(root)}


@pytest.mark.parametrize("name", ["routebalance/scheduler_backup.py", "routebalance/scheduler_bak.py",
                                  "routebalance/scheduler-backup.py", "backup_20260924.tar.gz"])
def test_backup_word_names_fail(tmp_path, name):
    root = _tree(tmp_path, {name: "x\n"})
    assert "forbidden_file" in {f.kind for f in scan_tree(root)}


def test_broken_git_repo_is_an_error_not_a_fallback(tmp_path):
    root = _tree(tmp_path, {"README.md": "ok\n"})
    _git(root, "init", "-q")
    (root / ".git" / "HEAD").write_text("garbage\n")
    assert main([str(root)]) == 2


def test_unreadable_secret_named_file_is_redacted(tmp_path, capsys):
    import os
    secret = J(["sk-", "proj-", "d" * 40])
    root = _tree(tmp_path, {f"configs/{secret}.json": "{}\n"})
    os.chmod(root / "configs" / f"{secret}.json", 0)
    try:
        rc = main([str(root)])
    finally:
        os.chmod(root / "configs" / f"{secret}.json", 0o600)
    captured = capsys.readouterr()
    assert rc in (1, 2) and secret not in captured.out + captured.err
