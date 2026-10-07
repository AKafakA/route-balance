"""The run archive, read with the standard library only, so the same reader serves every code tree.

A runs root holds one folder per family (qwen, gemma), and in it one folder per run, named by the run's id
(`<configuration>__<trace>__s<seed>`), with `config.json` (the scheduler document the run used), the record
`<run>.cell.json`, `decisions.jsonl` (one line per request, written when the request ended), the manifests, engine
and stats readings, and `gates/`. Folders that start with `_` and the vLLM Semantic Router's runs are not runs of the
scheduler.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterator, Optional


def runs(root: Path, families: tuple[str, ...] = ("qwen", "gemma"), only: Optional[set[str]] = None) -> list[Path]:
    out = []
    for fam in families:
        for d in sorted((Path(root) / fam).iterdir()):
            if d.is_dir() and not d.name.startswith("_") and ".vsr." not in d.name and (only is None or d.name in only):
                out.append(d)
    return out


def family(run: Path) -> str:
    return run.parent.name


def config_name(run: Path) -> str:
    return run.name.split("__")[0]


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def record(run: Path) -> dict:
    return read_json(run / f"{run.name}.cell.json")


def config(run: Path) -> dict:
    return read_json(run / "config.json")


def decisions(run: Path) -> list[dict]:
    return [json.loads(line) for line in (run / "decisions.jsonl").read_text().splitlines() if line.strip()]


def in_send_order(rec: dict) -> list[dict]:
    """The record's requests in the order the client sent them."""
    return sorted(rec["requests"], key=lambda r: r["index"])


def rounds(lines: list[dict], rec: dict) -> list[tuple[int, list[dict], list[dict]]]:
    """RouteBalance's rounds: (round number, its decision lines in dispatch order, its requests in send order, the
    order the round collected them in). Lines without a round (a request that failed before a round) are left out."""
    by_id = {r["request_id"]: r for r in rec["requests"]}
    grouped: dict[int, list[dict]] = {}
    for line in lines:
        if line.get("round") is not None:
            grouped.setdefault(line["round"], []).append(line)
    out = []
    for number in sorted(grouped):
        members = sorted(grouped[number], key=lambda line: line["round_pos"])
        batch = sorted((by_id[line["request_id"]] for line in members), key=lambda r: r["index"])
        out.append((number, members, batch))
    return out


def hexf(x: float) -> str:
    """A float, exactly."""
    return float(x).hex()


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


class Digest:
    """The running sha256 of a run's outputs, unit by unit."""

    def __init__(self) -> None:
        self._h = hashlib.sha256()
        self.units = 0

    def add(self, obj) -> None:
        self._h.update(canonical(obj).encode())
        self._h.update(b"\n")
        self.units += 1

    def hexdigest(self) -> str:
        return self._h.hexdigest()


def write_line(f, obj) -> None:
    f.write(canonical(obj) + "\n")
    f.flush()


def read_lines(path: Path) -> Iterator[dict]:
    for line in Path(path).read_text().splitlines():
        if line.strip():
            yield json.loads(line)
