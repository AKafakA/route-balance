"""Repository hygiene gate.

Fails on anything that must never be committed: the internal codename, cluster or personal host names, private IPs,
personal identifiers, credentials, personal absolute paths, and backup/stale/versioned copies. The identity terms
(codename, development hosts, people, cluster paths) come from a private --terms file, never from this file: a
release ships without them. File contents and file paths are both checked. Credentials are never printed, not even
when they appear in a file name.

Scope. In a git repository: every tracked or staged file is checked in BOTH its staged (index)
version and its working-tree version, and every untracked file that is not ignored is checked on
disk. A force-added file under an ignored directory is therefore still checked. Symlinks are
checked by their link-target text, which is what git stores. Outside git: every file except caches.
There are no exempt files: tests build their leak fixtures at runtime.

Usage: python -m gates.hygiene [--terms FILE] <repo-root>   (exit 1 on findings, 2 on usage or I/O errors)
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

CACHE_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache"}


def _alt(*words: str) -> str:
    return "|".join(re.escape(w) for w in words)


# The identity terms (the internal codename, the development hosts, the people, the cluster paths) are not in this
# file: a release ships without them. They come from a private --terms file (kind<TAB>term per line, # comments),
# each kind matched as below. Without it the gate checks everything else and says that the terms were not checked.
TERM_KINDS = ("codename", "host", "identity", "abs_path")
_HW = r"(?:d7525|c4130|d8545|c240g\d|r7525|m510|xl170|c6525|c6420|c220g\d|d430|d710|d820|c8220)"
_NB, _NA = r"(?<![A-Za-z0-9])", r"(?![A-Za-z0-9])"   # word boundaries that also split on '_'
SECRET = re.compile(_NB + (
    r"(?:hf_[A-Za-z0-9]{30,}|gh[opusr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}"
    r"|sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_-]{30,}|xox[abprs]-[A-Za-z0-9-]{10,}"
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----)"))
PATTERNS = [
    ("host", re.compile(r"(?i)" + _NB + _HW + r"[a-z0-9]*-(?:\d{2}s\d{5}|\d{6})" + _NA)),
    # private and campus IPs (also inside file names and before a sentence's full stop), but not version pins
    # such as "==10.3.9.90"; and user@<any IPv4>
    ("ip", re.compile(r"(?<![A-Za-z0-9.])(?<!==)(?<!>=)(?<!<=)(?<!~=)(?<!!=)"
                      r"(?:10\.\d{1,3}|192\.168|172\.(?:1[6-9]|2\d|3[01])|128\.105)\.\d{1,3}\.\d{1,3}"
                      r"(?![A-Za-z0-9]|\.\d)"
                      r"|[A-Za-z0-9_.-]+@\d{1,3}(?:\.\d{1,3}){3}(?!\.?\d)")),
    ("secret", SECRET),
    ("abs_path", re.compile(r"/home/[A-Za-z0-9_.-]+|/[Uu]sers/[A-Za-z0-9_.-]+")),
]


def parse_terms(text: str, where: str = "terms") -> list[tuple[str, re.Pattern]]:
    """The term patterns of a --terms file's text, one per kind. A file missing a kind is refused."""
    terms: dict[str, list[str]] = {k: [] for k in TERM_KINDS}
    for n, line in enumerate(text.splitlines(), start=1):
        if not line.strip() or line.startswith("#"):
            continue
        kind, sep, term = line.partition("\t")
        if not sep or kind not in terms or not term:
            raise ValueError(f"{where}:{n}: expected kind<TAB>term, kind one of {TERM_KINDS}")
        terms[kind].append(term)
    missing = [k for k in TERM_KINDS if not terms[k]]
    if missing:
        raise ValueError(f"{where}: no {', '.join(missing)} terms: every kind needs at least one")
    build = {"codename": lambda w: r"(?i)" + _alt(*w), "host": lambda w: r"(?i)" + _alt(*w),
             "identity": lambda w: r"(?i)" + _NB + "(?:" + _alt(*w) + ")" + _NA, "abs_path": lambda w: _alt(*w)}
    return [(k, re.compile(build[k](terms[k]))) for k in TERM_KINDS]
FORBIDDEN_PART = re.compile(
    r"(?i)\.bak|\.orig$|~$|\.stale|\.old$|\.backup$|(?:^|[_-])v\d{1,2}(?:[_.-]|$)"
    r"|(?:^|[_-])(?:old|bak|backup)(?:[_.-]|$)|^archive$")


@dataclass(frozen=True)
class Finding:
    kind: str
    path: str
    line: int
    message: str

    def __str__(self) -> str:
        return f"{self.kind:14s} {SECRET.sub('<redacted>', self.path)}:{self.line}  {SECRET.sub('<redacted>', self.message)}"


def _git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True)


def _in_git_root(root: Path) -> bool:
    top = _git(root, "rev-parse", "--show-toplevel")
    if top.returncode != 0:
        if (root / ".git").exists():
            raise OSError(f"{root} has a .git but git cannot read it: {top.stderr.decode().strip()}")
        return False
    return Path(top.stdout.decode().strip()).resolve() == root.resolve()


def _disk_text(p: Path) -> str | None:
    if p.is_symlink():
        return os.readlink(p)          # what git stores for a symlink
    if p.is_file():
        return p.read_bytes().decode("latin-1")  # every byte decodes; the patterns are ASCII
    return None


def _versions(root: Path) -> list[tuple[str, str, str]]:
    """(relative path, where, text) for every version of every file in scope."""
    out: list[tuple[str, str, str]] = []
    if _in_git_root(root):
        listings = []
        for args in (("ls-files", "-z", "--cached"), ("ls-files", "-z", "--others", "--exclude-standard")):
            r = _git(root, *args)
            if r.returncode != 0:
                raise OSError(f"git {' '.join(args)} failed: {r.stderr.decode().strip()}")
            listings.append([p for p in r.stdout.decode().split("\0") if p])
        cached, others = listings
        for rel in sorted(set(cached)):
            staged = _git(root, "show", f":{rel}")
            if staged.returncode != 0:
                raise OSError(f"cannot read staged {rel}: {staged.stderr.decode().strip()}")
            out.append((rel, "staged", staged.stdout.decode("latin-1")))  # the version that will be committed
            disk = _disk_text(root / rel)
            if disk is not None and disk != out[-1][2]:
                out.append((rel, "working tree", disk))
        for rel in sorted(set(others) - set(cached)):
            disk = _disk_text(root / rel)
            if disk is not None:
                out.append((rel, "untracked", disk))
        return out
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if CACHE_DIRS.intersection(rel.parts):
            continue
        disk = _disk_text(p)
        if disk is not None:
            out.append((rel.as_posix(), "file", disk))
    return out


def _check(kind: str, rx: re.Pattern, text: str) -> str | None:
    m = rx.search(text)
    if not m:
        return None
    return "<redacted>" if kind == "secret" else m.group(0)


def scan_tree(root: Path, terms: list[tuple[str, re.Pattern]] = ()) -> list[Finding]:
    root = Path(root)
    patterns = PATTERNS + list(terms)
    findings: list[Finding] = []
    seen_paths: set[str] = set()
    for rel, where, text in _versions(root):
        if rel not in seen_paths:
            seen_paths.add(rel)
            if any(FORBIDDEN_PART.search(part) for part in Path(rel).parts):
                findings.append(Finding("forbidden_file", rel, 0, "backup/stale/versioned copy or archive"))
            for kind, rx in patterns:
                shown = _check(kind, rx, rel)
                if shown:
                    findings.append(Finding(kind, rel, 0, f"file path matched {shown}"))
        for lineno, line in enumerate(text.splitlines(), start=1):
            for kind, rx in patterns:
                shown = _check(kind, rx, line)
                if shown:
                    findings.append(Finding(kind, rel, lineno, f"{where}: matched {shown}"))
    return findings


def main(argv: list[str]) -> int:
    terms_file = None
    if len(argv) == 3 and argv[0] == "--terms":
        terms_file, argv = Path(argv[1]), argv[2:]
    if len(argv) != 1:
        print("usage: python -m gates.hygiene [--terms FILE] <repo-root>", file=sys.stderr)
        return 2
    root = Path(argv[0])
    if not root.is_dir():
        print(f"hygiene: {SECRET.sub('<redacted>', str(root))} is not a directory", file=sys.stderr)
        return 2
    try:
        terms = parse_terms(terms_file.read_text(), str(terms_file)) if terms_file else []
        findings = scan_tree(root, terms)
    except (OSError, ValueError) as e:
        print(f"hygiene: {SECRET.sub('<redacted>', str(e))}", file=sys.stderr)
        return 2
    for f in findings:
        print(f)
    if terms_file is None:
        print("hygiene: no --terms file, so the identity terms were not checked", file=sys.stderr)
    print(f"hygiene: {len(findings)} finding(s)")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
