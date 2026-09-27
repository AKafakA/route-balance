"""Artifact files are read once and checked against a pinned md5 before anything parses them."""
from __future__ import annotations

import hashlib
from pathlib import Path


class ArtifactError(ValueError):
    pass


def read_verified(path: Path, md5: str) -> bytes:
    """The file's bytes, if their md5 is `md5`. Parse these bytes, never the file again (no check-then-reopen)."""
    try:
        data = Path(path).read_bytes()
    except OSError as e:
        raise ArtifactError(f"cannot read {path}: {e}") from e
    got = hashlib.md5(data).hexdigest()
    if got != md5:
        raise ArtifactError(f"{path}: md5 {got} does not match the pinned {md5}")
    return data


def md5_tree(root: Path, rel: str) -> dict[str, str]:
    """{"<rel>/<file>": md5} for every regular file under root/rel, in path order: the identity of an
    artifact directory that a library loads by path (a router checkpoint), recorded in the execution manifest."""
    top = Path(root) / rel
    if not top.is_dir():
        raise ArtifactError(f"{top} is not a directory")
    out = {}
    for path in sorted(p for p in top.rglob("*") if p.is_file()):
        h = hashlib.md5()
        try:
            with open(path, "rb") as fh:
                while chunk := fh.read(1 << 20):
                    h.update(chunk)
        except OSError as e:
            raise ArtifactError(f"cannot read {path}: {e}") from e
        out[f"{rel}/{path.relative_to(top).as_posix()}"] = h.hexdigest()
    if not out:
        raise ArtifactError(f"{top} holds no files")
    return out
