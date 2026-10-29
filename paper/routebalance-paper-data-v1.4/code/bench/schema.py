"""Strict schemas for the files the benchmark and the analysis read and write.

Every key is required, unknown keys are rejected, nothing is coerced (no "1" for 1, no true for 1), and
non-finite numbers are refused. JSON text goes through routebalance.config.loads_strict, which also
rejects duplicate keys.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Sequence

from pydantic import BaseModel, ConfigDict

from routebalance.config import ConfigError, loads_strict, parse, read_json   # noqa: F401 (re-exported)


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)


def read_jsonl(path: Path) -> Iterator[tuple[int, object]]:
    """(line number, record) for every line; the final newline is optional, any other empty line is an error."""
    try:
        lines = Path(path).read_bytes().decode("utf-8").split("\n")   # not splitlines(): U+2028 may sit in a string
    except (OSError, UnicodeDecodeError) as e:
        raise ConfigError(f"cannot read {path}: {e}") from e
    if lines and lines[-1] == "":
        lines.pop()                                        # the final newline
    for lineno, line in enumerate(lines, start=1):
        try:
            yield lineno, loads_strict(line)
        except ConfigError as e:
            raise ConfigError(f"{path}:{lineno}: {e}") from e


def write_new_jsonl(path: Path, rows: Sequence[object]) -> None:
    """One JSON line per row, to a file that must not exist yet, in a directory that must exist."""
    if path.exists():
        raise ConfigError(f"{path} exists; write a new file")
    with open(path, "x", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
