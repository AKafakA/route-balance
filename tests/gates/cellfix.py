"""Recorded cell directories (tests/gates/fixtures, written by tests/contract/cell_recorder.py) and helpers that
seed one violation into a copy of them."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Callable

FIXTURES = Path(__file__).parent / "fixtures"
CELLS = sorted(p.name for p in (FIXTURES / "cells").iterdir())
RB_MAXDIV, RB_FIXEDREF = "rb-maxdiv__l400__s1", "rb-fixedref__l400__s1"
RB_MIN, RB_MEAN, RB_OFF = "rb-min__l400__s1", "rb-mean__l400__s1", "rb-off__l400__s1"
LOR, DECOUPLED, SWEEP = "lor__l400__s1", "decoupled__l400__s1", "sweep__l400__s1"
RB_POLL, LOR_PROBE = "rb-poll__l400__s1", "lor-probe__l400__s1"    # the read-mode controls
W4_REF, W4_NODR = "w4-ref__l400__s1", "w4-nodr__l400__s1"
RB_ROUND = "rb-round__l400__s1"                                   # normalizes over the round
G1 = [RB_MAXDIV, RB_ROUND, RB_FIXEDREF, RB_MIN, RB_MEAN, RB_OFF, RB_POLL, LOR, LOR_PROBE, DECOUPLED, SWEEP]   # a main group
W4 = [W4_REF, W4_NODR]                                                    # an ablation group (dead reckoning off)


def copy_cell(tmp_path: Path, name: str) -> Path:
    dst = tmp_path / name
    shutil.copytree(FIXTURES / "cells" / name, dst)
    return dst


def copy_all(tmp_path: Path) -> list[Path]:
    return [copy_cell(tmp_path, name) for name in CELLS]


def edit_json(path: Path, fn: Callable[[dict], None]) -> None:
    data = json.loads(path.read_text())
    fn(data)
    path.write_text(json.dumps(data))


def edit_manifests(cell: Path, fn: Callable[[dict], None]) -> None:
    """The same change before and after (an execution difference that held for the whole cell)."""
    for when in ("before", "after"):
        edit_json(cell / f"manifest_{when}.json", fn)


def edit_lines(cell: Path, fn: Callable[[list[dict]], None]) -> None:
    path = cell / "decisions.jsonl"
    lines = [json.loads(x) for x in path.read_text().splitlines()]
    fn(lines)
    path.write_text("".join(json.dumps(x) + "\n" for x in lines))


def record(cell: Path) -> Path:
    return cell / f"{cell.name}.cell.json"
