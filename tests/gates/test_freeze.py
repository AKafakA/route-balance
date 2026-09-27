"""The freeze gate: after the tag, scope and presets change only with a logged owner decision."""
import hashlib
import json
import subprocess

from gates import freeze


def _git(root, *args):
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, timeout=60)


def _repo(tmp_path):
    (tmp_path / "configs").mkdir()
    for f in freeze.FROZEN:
        (tmp_path / f).write_text("{}\n")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "c")
    return tmp_path


def test_before_the_tag_and_unchanged_after_it_pass(tmp_path):
    root = _repo(tmp_path)
    (root / "configs" / "scope.json").write_text('{"x": 1}\n')
    assert freeze.findings(root, "lease-x") == []                 # no tag yet: nothing is frozen
    _git(root, "checkout", "-q", "--", "configs/scope.json")
    _git(root, "tag", "lease-x")
    assert freeze.findings(root, "lease-x") == []


def test_a_change_after_the_tag_needs_its_logged_decision(tmp_path):
    root = _repo(tmp_path)
    _git(root, "tag", "lease-x")
    (root / "configs" / "presets.json").write_text('{"y": 2}\n')
    (found,) = freeze.findings(root, "lease-x")
    assert "configs/presets.json changed since lease-x" in found
    digest = hashlib.sha256((root / "configs" / "presets.json").read_bytes()).hexdigest()
    entry = {"file": "configs/presets.json", "sha256": digest, "decision": "D91", "owner": "the owner's words",
             "gpu_cost": "none"}
    (root / freeze.EXCEPTIONS).write_text(json.dumps([{**entry, "gpu_cost": ""}]))
    assert "non-empty strings" in freeze.findings(root, "lease-x")[0]
    (root / freeze.EXCEPTIONS).write_text(json.dumps([entry]))
    assert freeze.findings(root, "lease-x") == []
    (root / "configs" / "presets.json").write_text('{"y": 3}\n')      # the entry names the exact content
    assert len(freeze.findings(root, "lease-x")) == 1


def test_a_deployed_copy_without_git_is_named_and_a_broken_repository_raises(tmp_path, capsys):
    import pytest
    from gates.freeze import findings
    copy = tmp_path / "copy"
    copy.mkdir()
    assert findings(copy, "lease-2026-09-28") == []
    assert "a deployed copy" in capsys.readouterr().err
    broken = tmp_path / "broken"
    (broken / ".git").mkdir(parents=True)                  # a .git that is not a repository
    with pytest.raises(RuntimeError, match="not a git repository"):
        findings(broken, "lease-2026-09-28")
    with pytest.raises(RuntimeError, match="not a directory"):
        findings(tmp_path / "missing", "lease-2026-09-28")
