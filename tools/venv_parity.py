"""The test venv a node runs the suite in (scripts/setup.sh --with-tests) must be the serving stack plus the dev
extras: every package of the scheduler venv at the same version, and beyond it only the dev extras' packages. The
test venv is built constrained to the scheduler venv's versions, then compared.

    python -m tools.venv_parity SCHED_FREEZE TEST_FREEZE     exit 1 with the differences if the stacks differ
    python -m tools.venv_parity --constraints SCHED_FREEZE   the scheduler venv's pins, as a pip constraints file

A freeze is `pip freeze --all` output: `name==version` pins, `name @ url` direct installs (compared, never used as
constraints: the pyproject pins them itself) and `-e` editable lines (this repository: not compared, setup.sh and
the suite check where it imports from).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Iterable

DEV_EXTRAS = frozenset({"pytest", "ruff", "iniconfig", "pluggy", "tomli", "exceptiongroup"})
_LINE = re.compile(r"^([A-Za-z0-9_.-]+)(==| @ )(\S+)$")


def _name(raw: str) -> str:
    return raw.lower().replace("_", "-")


def packages(lines: Iterable[str]) -> dict[str, str]:
    """name -> version (a pin) or url (a direct install); editable lines, comments and blank lines are left out,
    and any other line is refused."""
    out = {}
    for line in lines:
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("-e "):
            continue
        m = _LINE.match(s)
        if not m:
            raise ValueError(f"unreadable freeze line {s!r}")
        out[_name(m.group(1))] = m.group(3)
    return out


def differences(sched: dict[str, str], test: dict[str, str]) -> list[str]:
    changed = [f"{n}: {v} in the scheduler venv, {test.get(n)} in the test venv" for n, v in sorted(sched.items())
               if test.get(n) != v]
    extra = [f"{n} {test[n]} only in the test venv" for n in sorted(set(test) - set(sched) - DEV_EXTRAS)]
    return changed + extra


def constraints(lines: Iterable[str]) -> list[str]:
    """The `name==version` pins of a freeze, for `pip install -c`."""
    kept = [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith(("#", "-e "))]
    matched = []
    for ln in kept:
        m = _LINE.match(ln)
        if not m:
            raise ValueError(f"unreadable freeze line {ln!r}")
        matched.append(m)
    return [f"{m.group(1)}=={m.group(3)}" for m in matched if m.group(2) == "=="]


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[0] == "--constraints":
        try:
            print("\n".join(constraints(Path(argv[1]).read_text().splitlines())))
        except ValueError as e:
            print(f"venv_parity: {e}", file=sys.stderr)
            return 2
        return 0
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    try:
        sched, test = (packages(Path(p).read_text().splitlines()) for p in argv)
    except ValueError as e:
        print(f"venv_parity: {e}", file=sys.stderr)
        return 2
    diff = differences(sched, test)
    for d in diff:
        print(f"FAIL {d}")
    if not diff:
        print(f"  the test venv has the scheduler venv's {len(sched)} packages at the same versions, plus "
              f"{sorted(set(test) - set(sched))}")
    return 1 if diff else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
