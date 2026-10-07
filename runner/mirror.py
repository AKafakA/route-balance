"""The second location: every published cell and the bookkeeping directories (_attempts, _groups, _retired) are
copied there with rsync (a local path or host:path) and proven identical by checksum. Nothing here ever deletes or
writes over a different run on the mirror."""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable, Iterable, Optional

# the lines of an itemized rsync dry run that leave a run on the mirror as it is: nothing to do, a directory to
# create, a file to create, a directory's time to set
_HARMLESS = (lambda ln: ln == "", lambda ln: ln.startswith("created directory "), lambda ln: "+++++++++" in ln,
             lambda ln: ln.startswith(".d..t...... "))

Say = Callable[[str], None]


def _rsync(say: Optional[Say], *args: str) -> subprocess.CompletedProcess:
    try:
        r = subprocess.run(["rsync", "-a", "--checksum", *args], capture_output=True, text=True)
    except OSError as e:            # rsync cannot start: a failed copy, as a shell's 127
        r = subprocess.CompletedProcess(["rsync", *args], 127, "", str(e))
    if r.returncode != 0 and say is not None:
        say(f"rsync {' '.join(args)}: exit {r.returncode}: {r.stderr.strip()[-1000:]}")
    return r


def _dry_run(src: str, dst: str, say: Optional[Say]) -> subprocess.CompletedProcess:
    return _rsync(say, "--dry-run", "--itemize-changes", "--delete", f"{src}/", f"{dst}/")


def fresh(src: str, dst: str, say: Optional[Say] = None) -> bool:
    """A copy from src to dst would only create files or set a directory's time: dst is absent, partial or
    identical."""
    plan = _dry_run(src, dst, say)
    return plan.returncode == 0 and all(any(ok(ln) for ok in _HARMLESS) for ln in plan.stdout.split("\n"))


def mirror(src: str, dst: str, say: Optional[Say] = None) -> bool:
    """Copy, then a dry run must succeed and list no difference and no extra file."""
    if _rsync(say, f"{src}/", f"{dst}/").returncode != 0:
        return False
    check = _dry_run(src, dst, say)
    return check.returncode == 0 and check.stdout == ""


def mirror_cell(root: Path, dest: str, cell: str, say: Say) -> bool:
    """One published cell: never over another run of it on the mirror, then identical."""
    src, dst = f"{root}/{cell}", f"{dest}/{cell}"
    if not fresh(src, dst, say):
        say(f"{dst} holds another run of {cell}: move it aside (retire it there too) first")
        return False
    if not mirror(src, dst, say):
        say(f"the mirror of {cell} at {dst} is not identical")
        return False
    return True


def mirror_bookkeeping(root: Path, dest: str, say: Say, names: Iterable[str] = ()) -> bool:
    """The bookkeeping directories (every _<name> directory under the cells root, or only `names`), by checksum."""
    names = list(names)
    dirs = [Path(root) / n for n in names] if names else sorted(Path(root).glob("_*"))
    return all(_rsync(say, f"{d}/", f"{dest}/{d.name}/").returncode == 0 for d in dirs if d.is_dir())
