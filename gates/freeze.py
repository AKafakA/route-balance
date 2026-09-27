"""Freeze gate (owner, 09-26): after the freeze tag, the lease's scope and presets change only by a logged owner decision.

A change to configs/scope.json or configs/presets.json since the tag passes only when configs/after_freeze.json lists
the file's current sha256 with the decision (its DECISION_LOG id), the owner's words and the GPU cost:

    [{"file": "configs/presets.json", "sha256": "<64 hex>", "decision": "D91",
      "owner": "<the owner's words>", "gpu_cost": "<runs and hours, on or off the must path>"}]

Before the tag exists there is nothing to compare, and the gate passes.

Usage: python -m gates.freeze <repo-root> --tag lease-2026-09-28   (exit 1 on findings, 2 on usage or git errors)
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

from bench.schema import loads_strict
from routebalance.config import ConfigError

FROZEN = ("configs/scope.json", "configs/presets.json")
EXCEPTIONS = "configs/after_freeze.json"
FIELDS = {"file", "sha256", "decision", "owner", "gpu_cost"}


def _git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=60)


def findings(root: Path, tag: str) -> list[str]:
    if not (Path(root) / ".git").exists():
        # a deployed copy of the tree (setup.sh rsyncs it to the nodes), not the repository: the tag and the history
        # live in the repository, and run_scope checks the copy's code digest
        print(f"freeze: {root} has no .git: a deployed copy, not the repository; the freeze is checked there",
              file=sys.stderr)
        return []
    repo = _git(root, "rev-parse", "--git-dir")
    if repo.returncode != 0:
        raise RuntimeError(f"{root} is not a git repository: {repo.stderr.strip()}")
    t = _git(root, "rev-parse", "--verify", "--quiet", f"refs/tags/{tag}")
    if t.returncode == 1:
        return []                                             # before the freeze: the tag does not exist yet
    if t.returncode != 0:
        raise RuntimeError(f"git rev-parse refs/tags/{tag}: {t.stderr.strip()}")
    changed = []
    for f in FROZEN:
        r = _git(root, "diff", "--quiet", tag, "--", f)
        if r.returncode not in (0, 1):
            raise RuntimeError(f"git diff {tag} -- {f}: {r.stderr.strip()}")
        if r.returncode == 1:
            changed.append(f)
    if not changed:
        return []
    listed: dict[tuple[str, str], dict] = {}
    path = root / EXCEPTIONS
    if path.is_file():
        entries = loads_strict(path.read_text())
        if not isinstance(entries, list):
            return [f"{EXCEPTIONS} must be a list of entries"]
        for e in entries:
            if not (isinstance(e, dict) and set(e) == FIELDS
                    and all(isinstance(e[k], str) and e[k].strip() for k in FIELDS)):
                return [f"{EXCEPTIONS}: every entry has exactly the non-empty strings {sorted(FIELDS)}"]
            listed[(e["file"], e["sha256"])] = e
    out = []
    for f in changed:
        digest = hashlib.sha256((root / f).read_bytes()).hexdigest()
        if (f, digest) not in listed:
            out.append(f"{f} changed since {tag} (sha256 {digest}) without an entry in {EXCEPTIONS} naming the "
                       "owner's decision and the GPU cost")
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("root", type=Path)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args(argv)
    try:
        found = findings(a.root, a.tag)
    except (RuntimeError, OSError, ConfigError, ValueError) as e:
        print(f"freeze: {e}", file=sys.stderr)
        return 2
    for f in found:
        print(f"freeze: {f}")
    print(f"freeze: {len(found)} finding(s)")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
