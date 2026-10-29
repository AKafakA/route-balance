#!/usr/bin/env python3
"""Checks for the paper data pack, run by reproduce_results.sh.

    python3 verify_results.py manifest          every file of the pack against MANIFEST.sha256, and no file outside it
    python3 verify_results.py compare OUT       every reference output under expected/ against OUT, byte for byte

Exit 0 only when every check passes.
"""
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SKIP = {"MANIFEST.sha256"}


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def manifest() -> int:
    listed = {}
    for line in (HERE / "MANIFEST.sha256").read_text().splitlines():
        digest, rel = line.split("  ", 1)
        listed[rel] = digest
    bad = [rel for rel, d in sorted(listed.items()) if not (HERE / rel).is_file() or sha256(HERE / rel) != d]
    present = {str(p.relative_to(HERE)) for p in HERE.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
    extra = sorted(present - set(listed) - SKIP)
    for rel in bad:
        print(f"MANIFEST: {rel} is missing or differs")
    for rel in extra:
        print(f"MANIFEST: {rel} is not listed")
    if bad or extra:
        print(f"FAIL: {len(bad)} listed files differ or are missing, {len(extra)} files are not listed")
        return 1
    print(f"manifest: {len(listed)} files match MANIFEST.sha256")
    return 0


def compare(out: Path) -> int:
    ref = HERE / "expected"
    listed = [line.split("  ", 1)[1] for line in (HERE / "MANIFEST.sha256").read_text().splitlines()]
    refs = sorted(rel[len("expected/"):] for rel in listed if rel.startswith("expected/"))
    if not refs or any(not (ref / rel).is_file() for rel in refs):
        print("FAIL: the manifest lists no reference outputs, or one is missing")
        return 1
    differ, checked = [], 0
    for rel in refs:
        checked += 1
        got = out / rel
        if got.is_file() and got.read_bytes() == (ref / rel).read_bytes():
            print(f"same     {rel}")
        else:
            print(f"DIFFERS  {rel}" + ("" if got.is_file() else " (not written)"))
            differ.append(rel)
    if differ:
        print(f"FAIL: {len(differ)} of {checked} reference outputs differ")
        return 1
    print(f"PASS: {checked} reference outputs match")
    return 0


def main(argv: list[str]) -> int:
    if argv[:1] == ["manifest"] and len(argv) == 1:
        return manifest()
    if argv[:1] == ["compare"] and len(argv) == 2:
        return compare(Path(argv[1]).resolve())
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
