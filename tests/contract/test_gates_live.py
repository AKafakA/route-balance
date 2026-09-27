"""The gates against a live run: the real scheduler behind fake vLLM engines, driven by the real runner.

Every cell recorded live passes arm and zero_call, and the group passes parity. Preflight passes the live fleet
and scheduler over HTTP and fails each seeded violation. The committed fixtures of tests/gates keep the live
format (same keys), so the unit tests there cannot drift from what the server writes.
"""
import asyncio
import json

import pytest

import gates.preflight as preflight_gate
from gates.arm import arm_problems
from gates.cell import open_cell
from gates.parity import parity_problems
from gates.zero_call import zero_call_problems
from routebalance.config import load_scheduler
from tests.contract.cell_recorder import GROUPS, LiveFleet, install_stand_ins
from tests.gates.cellfix import FIXTURES


@pytest.fixture
def stand_ins(monkeypatch):
    install_stand_ins(monkeypatch.setitem, monkeypatch.setattr)


def test_live_cells_pass_every_gate(tmp_path, stand_ins):
    async def scenario():
        async with LiveFleet(tmp_path / "work") as live:
            out = {}
            for g, (_, cfgs) in GROUPS.items():
                out[g] = [await live.record_cell(cid, tmp_path / "cells", group=g) for cid in cfgs]
            return out
    groups = asyncio.run(scenario())
    for cells in groups.values():
        for cell in cells:
            assert arm_problems(open_cell(cell), load_scheduler(cell / "config.json"))[0] == [], cell.name
            assert zero_call_problems(open_cell(cell))[0] == [], cell.name
        assert parity_problems([open_cell(c) for c in cells])[0] == []
        _same_format(cells)


def _keys(obj, prefix=""):
    if isinstance(obj, dict):
        return set().union(*[_keys(v, f"{prefix}.{k}") for k, v in obj.items()], {prefix})
    return {prefix}


def _same_format(live_cells):
    """Key sets of the live files equal the committed fixtures' (values differ: timings, digests, pools)."""
    for cell in live_cells:
        fixture = FIXTURES / "cells" / cell.name
        for name in ("manifest_before.json", "stats_after.json"):
            live, committed = (json.loads((d / name).read_text()) for d in (cell, fixture))
            drop = ("artifacts", "threadpools", "versions")          # keyed by what is installed or loaded
            assert _keys({k: v for k, v in live.items() if k not in drop}) == \
                   _keys({k: v for k, v in committed.items() if k not in drop}), (cell.name, name)
        live_line, committed_line = (json.loads((d / "decisions.jsonl").read_text().splitlines()[0])
                                     for d in (cell, fixture))
        assert set(live_line) == set(committed_line), cell.name
        assert (cell / "group.json").read_text() == (fixture / "group.json").read_text(), cell.name


async def _preflight(live, **kw):
    args = dict(scheduler_url=live.url, fleet=live.fleet, expect_config=live.service.digest,
                expect_fleet=live.service.fleet_digest) | kw
    return await preflight_gate.preflight(**args)


def test_preflight_passes_a_live_idle_fleet_and_its_cli_agrees(tmp_path, stand_ins):
    async def scenario():
        async with LiveFleet(tmp_path / "work") as live:
            await live.flushed_stats()
            found = await _preflight(live)
            argv = ["--scheduler", live.url, "--fleet", str(live.fleet_path), "--hosts", str(live.hosts_path),
                    "--expect-config-digest", live.service.digest, "--expect-fleet-digest", live.service.fleet_digest]
            code = await asyncio.get_running_loop().run_in_executor(None, preflight_gate.main, argv)
            return found, code
    found, code = asyncio.run(scenario())
    assert found == [] and code == 0


LIVE_FAIL = [
    ("not drained", lambda live: setattr(live.states["S_0"], "running", 1), {}, "engine S_0: not drained"),
    ("waiting", lambda live: setattr(live.states["S_1"], "waiting", 2), {}, "engine S_1: not drained"),
    ("old vLLM", lambda live: setattr(live.states["S_1"], "version", "0.18.0"), {}, "engine S_1: runs vLLM '0.18.0'"),
    ("wrong model", lambda live: setattr(live.states["L_2"], "served", ("S",)), {}, "engine L_2: serves ['S']"),
    ("two models", lambda live: setattr(live.states["L_2"], "served", ("L", "L-lora")), {}, "serves ['L', 'L-lora']"),
    ("config", None, {"expect_config": "0" * 64}, "/v1/config config_digest is"),
    ("fleet", None, {"expect_fleet": "0" * 64}, "the fleet files describe fleet"),
]


@pytest.mark.parametrize("what,seed,kw,expected", LIVE_FAIL, ids=[w for w, *_ in LIVE_FAIL])
def test_preflight_fails_each_live_violation(tmp_path, stand_ins, what, seed, kw, expected):
    async def scenario():
        async with LiveFleet(tmp_path / "work") as live:
            await live.flushed_stats()
            if seed is not None:
                seed(live)
            return await _preflight(live, **kw)
    found = asyncio.run(scenario())
    assert any(expected in p for p in found), found


def test_preflight_fails_an_unreachable_engine(tmp_path, stand_ins):
    async def scenario():
        async with LiveFleet(tmp_path / "work") as live:
            await live.flushed_stats()
            await live.engines[1].cleanup()                 # the engine is gone
            return await _preflight(live)
    found = asyncio.run(scenario())
    assert any(p.startswith("engine S_1: /health is ") for p in found), found
