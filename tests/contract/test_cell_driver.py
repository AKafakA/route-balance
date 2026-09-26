"""run_cell.sh's steps: the attempt directory, the window cut out of the scheduler's logs, the verdict, publishing
and resuming. The scheduler is a stand-in serving /v1/stats and /v1/manifest over real HTTP."""
import asyncio
import hashlib
import json

import pytest
from aiohttp import web

from routebalance.config import ConfigError, fleet_digest, load_fleet
from tests.contract.test_server import free_port
from tests.unit import test_campaign as tc
from tools import campaign
from tools import cell_driver as cd

FLEET = {"tiers": {"qwen2.5-3b_a30": {"model": "Qwen/Qwen2.5-3B", "gpu": "a30", "cost_in": 6e-8, "cost_out": 6e-8,
                                      "max_num_seqs": 256}},
         "instances": [{"instance_id": "q3b_0", "tier": "qwen2.5-3b_a30", "node": "n1", "port": 8001}]}
HOSTS = {"n1": {"address": "127.0.0.1", "hw": "d7525"}}
CELL = "g1.qwen.ap.pw070.lor__l12__s1"


ENGINES = {"record_sha256": "a" * 64, "instances": {"e_0": {
    "argv_sha256": "b" * 64, "argv": "--model M --port 8000", "gpus": "0",
    "cmdline": "/venv/bin/python -u -m vllm.entrypoints.openai.api_server --model M --port 8000",
    "start_time_s": 1790000000.0,
    "cache_config": {"block_size": "16", "num_gpu_blocks": "8000"}, "model": "M", "max_model_len": 4096}}}


@pytest.fixture
def lease(tmp_path):
    fleet, hosts = tmp_path / "fleet.json", tmp_path / "hosts.json"
    fleet.write_text(json.dumps(FLEET))
    hosts.write_text(json.dumps(HOSTS))
    entry = (hashlib.sha256(fleet.read_bytes()).hexdigest(), fleet_digest(load_fleet(fleet, hosts)))
    result = tc.full_render(fleets={name: entry for name in tc.FLEETS})
    render = tmp_path / "render"
    campaign.write_render(result, render, {})
    root = tmp_path / "cells"
    root.mkdir()
    return render, root, fleet, hosts, result


def test_prepare_makes_an_attempt_with_the_rendered_configuration(lease):
    render, root, fleet, hosts, result = lease
    out = cd.prepare(render, CELL, root, fleet, hosts)
    attempt = cd.Path(out["ATTEMPT"])
    assert attempt.name == CELL and attempt.parent.parent == root / "_attempts"
    assert (attempt / "config.json").read_bytes() == (render / "configs" / "g1.qwen.ap.pw070.lor.json").read_bytes()
    assert json.loads((attempt / "group.json").read_text()) == {"group": "g1", "reference": None,
                                                                 "intended_differences": ["system.*",
                                                                                          "telemetry.dead_reckoning"]}
    assert out["SWEEP"] == "0" and out["CONFIG_DIGEST"] == result.scope["configs"]["g1.qwen.ap.pw070.lor"]["config_digest"]
    assert cd.prepare(render, "qwen.sweep.random__sweep6-36__s1", root, fleet, hosts)["SWEEP"] == "1"
    other = fleet.with_name("other.json")
    other.write_text(json.dumps({**FLEET, "instances": [{**FLEET["instances"][0], "port": 8002}]}))
    with pytest.raises(ConfigError, match="not the"):
        cd.prepare(render, CELL, root, other, hosts)
    (root / CELL).mkdir()
    with pytest.raises(ConfigError, match="never overwritten"):
        cd.prepare(render, CELL, root, fleet, hosts)


class Scheduler:
    """/v1/stats and /v1/manifest from mutable dicts."""

    def __init__(self):
        self.stats = {"active": 0, "logged": {"decisions": 0, "decisions_buffered": 0, "latency": None}}
        self.manifest = {"config_digest": "d" * 64, "code": {"routebalance": "a", "routebalance_now": "a"},
                         "launch": {"torch_threads": 4, "max_connections": 4096, "request_timeout_s": 600.0}}

    async def __aenter__(self):
        async def stats(_request):
            return web.json_response(self.stats)

        async def manifest(_request):
            return web.json_response(self.manifest)
        app = web.Application()
        app.router.add_get("/v1/stats", stats)
        app.router.add_get("/v1/manifest", manifest)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        port = free_port()
        await web.TCPSite(self.runner, "127.0.0.1", port).start()
        self.url = f"http://127.0.0.1:{port}"
        return self

    async def __aexit__(self, *exc):
        await self.runner.cleanup()


def _append(path, lines):
    with open(path, "a") as f:
        f.write("".join(json.dumps(x) + "\n" for x in lines))


def test_the_window_holds_exactly_the_cells_decision_lines(tmp_path):
    logs, attempt = tmp_path / "logs", tmp_path / "a" / CELL
    logs.mkdir()
    attempt.mkdir(parents=True)
    _append(logs / "decisions.jsonl", [{"config_digest": "e" * 64, "request_id": "old.0"}])

    async def scenario():
        async with Scheduler() as s:
            s.stats["logged"]["decisions"] = 1
            await cd.snapshot_before(s.url, attempt, logs, "d" * 64, False)
            ours = [{"config_digest": "d" * 64, "request_id": f"{CELL}.r.{i}", "instance_id": "q3b_0"} for i in range(3)]
            _append(logs / "decisions.jsonl", ours)
            s.stats["logged"]["decisions"] = 4
            await cd.snapshot_after(s.url, attempt, "d" * 64)
            return ours
    ours = asyncio.run(scenario())
    assert [json.loads(x) for x in (attempt / "decisions.jsonl").read_text().splitlines()] == ours
    for name in ("stats_before.json", "stats_after.json", "manifest_before.json", "manifest_after.json"):
        assert (attempt / name).exists()
    window = json.loads((attempt / "window.json").read_text())
    assert window["decisions"]["end"] - window["decisions"]["start"] == len((attempt / "decisions.jsonl").read_bytes())


def test_foreign_or_unflushed_lines_fail_the_window(tmp_path):
    logs, attempt = tmp_path / "logs", tmp_path / "a" / CELL
    logs.mkdir()
    attempt.mkdir(parents=True)

    async def scenario(extra, logged):
        async with Scheduler() as s:
            await cd.snapshot_before(s.url, attempt, logs, "d" * 64, False)
            _append(logs / "decisions.jsonl", extra)
            s.stats["logged"]["decisions"] = logged
            await cd.snapshot_after(s.url, attempt, "d" * 64)
    with pytest.raises(ConfigError, match="foreign traffic"):
        asyncio.run(scenario([{"config_digest": "e" * 64, "request_id": "x"}], 1))
    for name in ("stats_before.json", "manifest_before.json", "window.json"):
        (attempt / name).unlink()
    cd.SETTLE_S = 0.3
    with pytest.raises(ConfigError, match="did not settle"):
        asyncio.run(scenario([{"config_digest": "d" * 64, "request_id": "y"}], 5))


def test_the_sweep_window_waits_for_every_latency_line(tmp_path):
    logs, attempt = tmp_path / "logs", tmp_path / "a" / CELL
    logs.mkdir()
    attempt.mkdir(parents=True)
    digest = "d" * 64

    async def scenario():
        async with Scheduler() as s:
            s.stats["logged"]["latency"] = 0
            await cd.snapshot_before(s.url, attempt, logs, digest, True)
            _append(logs / "decisions.jsonl", [{"config_digest": digest, "request_id": f"r{i}", "instance_id": "i"}
                                               for i in range(2)] + [{"config_digest": digest, "request_id": "r2"}])
            _append(logs / f"latency.{digest[:16]}.jsonl", [{"tier": "t", "n": i} for i in range(2)])
            s.stats["logged"].update(decisions=3, latency=2)
            await cd.snapshot_after(s.url, attempt, digest)
    asyncio.run(scenario())
    assert len((attempt / "latency.jsonl").read_text().splitlines()) == 2   # the unassigned request has none


def test_a_clean_record_with_every_gate_passed_is_valid(lease, monkeypatch):
    """Found 09-25: the gates' source directory once shadowed the gate list, and every valid cell's verdict raised."""
    render, root, fleet, hosts, _ = lease
    attempt = cd.Path(cd.prepare(render, CELL, root, fleet, hosts)["ATTEMPT"])
    (attempt / "gates").mkdir()
    monkeypatch.setattr(cd, "record_problems", lambda scope, exp, d: [])
    for when in ("before", "after"):
        (attempt / f"engines_{when}.json").write_text(json.dumps(ENGINES))
    assert cd.GATES == ("arm", "zero_call")
    assert cd.verdict(render, attempt, {"arm": 0, "zero_call": 0})["valid"] is True
    for gates in ({"arm": 0}, {"arm": 0, "zero_call": 1}):
        (attempt / "gates" / "verdict.json").unlink()
        assert cd.verdict(render, attempt, gates)["valid"] is False


def test_an_engine_restart_during_the_cell_invalidates_it(lease, monkeypatch):
    """D48: the engines are read before and after; any change, a restart first, makes the cell invalid."""
    render, root, fleet, hosts, _ = lease
    attempt = cd.Path(cd.prepare(render, CELL, root, fleet, hosts)["ATTEMPT"])
    (attempt / "gates").mkdir()
    monkeypatch.setattr(cd, "record_problems", lambda scope, exp, d: [])
    restarted = json.loads(json.dumps(ENGINES))
    restarted["instances"]["e_0"]["start_time_s"] += 60.0
    (attempt / "engines_before.json").write_text(json.dumps(ENGINES))
    (attempt / "engines_after.json").write_text(json.dumps(restarted))
    v = cd.verdict(render, attempt, {"arm": 0, "zero_call": 0})
    assert not v["valid"] and v["bench_problems"] == ["an engine restarted or changed during the cell: "
                                                      "['e_0.start_time_s']"]
    (attempt / "engines_after.json").unlink()
    (attempt / "gates" / "verdict.json").unlink()
    assert "the engine record is missing" in cd.verdict(render, attempt, {"arm": 0, "zero_call": 0})["bench_problems"][0]


def test_status_verdict_publish_and_resume(lease, tmp_path):
    render, root, fleet, hosts, result = lease
    scope, _ = cd.load(render)
    run, report = cd.cells_to_run(render, ["G1"], root, fleet, hosts)
    assert len(run) == 44 and run[0].startswith("g1.qwen.rb.") and report == []
    attempt = cd.Path(cd.prepare(render, CELL, root, fleet, hosts)["ATTEMPT"])
    with pytest.raises(ConfigError, match="no cell record"):
        cd.publish(attempt, root)
    (attempt / f"{CELL}.cell.json").write_text("{}")                 # an unreadable record: invalid, kept
    (attempt / "gates").mkdir()
    v = cd.verdict(render, attempt, {"arm": 0, "zero_call": 0})
    assert not v["valid"] and v["bench_problems"]
    published = cd.publish(attempt, root)
    assert published == root / CELL and cd.status(scope, cd.expected(scope, CELL), root) == "invalid"
    run, report = cd.cells_to_run(render, ["G1"], root, fleet, hosts)
    assert CELL not in run and any(CELL in line and line.startswith("invalid") for line in report)
    retired = cd.retire(root, CELL)
    assert retired.parent == root / "_retired" and CELL in cd.cells_to_run(render, ["G1"], root, fleet, hosts)[0]
    with pytest.raises(ConfigError, match="not cell rows"):
        cd.cells_to_run(render, ["W1"], root, fleet, hosts)


def test_the_scheduler_must_run_its_own_code_and_the_scopes_launch_values(lease):
    render, *_ = lease
    _, plan = cd.load(render)
    m = Scheduler().manifest
    assert cd.check_code(m) == [] and cd.check_launch(plan, m) == []
    assert cd.check_code({"code": {"routebalance": "a", "routebalance_now": "b"}}) == ["routebalance"]
    wrong = cd.check_launch(plan, {"launch": {**m["launch"], "torch_threads": 8}})
    assert wrong == ["torch_threads: the scheduler runs 8, the scope says 4"]
    assert "OWNER-MUST-SET" in cd.check_launch({**plan, "server": None}, m)[0]


def test_an_unpublished_record_is_never_run_again(lease):
    render, root, fleet, hosts, _ = lease
    attempt = cd.Path(cd.prepare(render, CELL, root, fleet, hosts)["ATTEMPT"])
    assert CELL in cd.cells_to_run(render, ["G1"], root, fleet, hosts)[0]        # no record yet: run it
    (attempt / f"{CELL}.cell.json").write_text("{}")                          # the driver stopped after the runner
    run, report = cd.cells_to_run(render, ["G1"], root, fleet, hosts)
    assert CELL not in run and any(line.startswith("unpublished") and CELL in line for line in report)


def test_the_rows_are_done_only_when_every_cell_is_and_parity_passed(lease, monkeypatch):
    render, root, *_ = lease
    problems, published = cd.rows_status(render, ["G1"], root)
    assert len(problems) == 44 and published == []
    (root / CELL / "gates").mkdir(parents=True)
    (root / CELL / "gates" / "verdict.json").write_text(json.dumps({"valid": False}))
    problems, published = cd.rows_status(render, ["G1"], root)
    assert published == [CELL] and f"invalid: {CELL}" in problems
    assert cd.main(["status", "--render", str(render), "--rows", "G1", "--cells-root", str(root)]) == 1
    assert cd.main(["published", "--render", str(render), "--rows", "G1", "--cells-root", str(root)]) == 0
    # a complete group: parity must have passed over exactly its current cells
    monkeypatch.setattr(cd, "status", lambda scope, exp, root: "done")
    scope, _ = cd.load(render)
    for exp in [c for c in cd.expected_cells(scope) if c.group == "g1"]:
        (root / exp.cell_id).mkdir(exist_ok=True)
        (root / exp.cell_id / f"{exp.cell_id}.cell.json").write_text("{}")
    problems, _ = cd.rows_status(render, ["G1"], root)
    assert problems == ["group g1 is complete but parity has not run over its current cells with the current gate code"]
    assert cd.parity(render, root, "g1") == 1                  # gates.parity is not in this tree: the gate fails
    problems, _ = cd.rows_status(render, ["G1"], root)
    assert len(problems) == 1 and problems[0].startswith("group g1 failed its parity gate")
    monkeypatch.setattr(cd, "gate_code", lambda: "f" * 64)       # a record made by other gate code proves nothing
    problems, _ = cd.rows_status(render, ["G1"], root)
    assert problems == ["group g1 is complete but parity has not run over its current cells with the current gate code"]


def test_parity_runs_once_per_set_of_cells_and_keeps_every_run(lease, monkeypatch):
    render, root, *_ = lease
    assert cd.parity(render, root, "g1") is None               # not ready: nothing runs
    for name in ("a", "b", "c"):
        (root / name).mkdir()
        (root / name / f"{name}.cell.json").write_text("run 1")
    cells = [str(root / "a"), str(root / "b")]
    monkeypatch.setattr(cd, "group_ready", lambda r, c, g: list(cells))
    runs = []
    monkeypatch.setattr(cd.subprocess, "run", lambda argv, **kw: runs.append(argv) or type("P", (), {"returncode": 0})())
    assert cd.parity(render, root, "g1") == 0 and cd.parity(render, root, "g1") == 0
    assert len(runs) == 1 and runs[0][-2:] == cells and runs[0][1:4] == ["-m", "gates.parity", "--cells"]
    cells.append(str(root / "c"))                              # a later render adds a cell: parity runs again
    cd.time.sleep(1.1)
    assert cd.parity(render, root, "g1") == 0 and len(runs) == 2
    assert len(list((root / "_groups" / "g1").glob("parity.*.json"))) == 2
    (root / "b" / "b.cell.json").write_text("run 2")           # b retired and run again, at the same path
    cd.time.sleep(1.1)
    assert cd.parity(render, root, "g1") == 0 and len(runs) == 3   # the gate sees the new run
    monkeypatch.setattr(cd, "gate_code", lambda: "f" * 64)       # the gate's code was fixed: its verdict is redone
    cd.time.sleep(1.1)
    assert cd.parity(render, root, "g1") == 0 and len(runs) == 4
    assert cd.parity(render, root, "g1") == 0 and len(runs) == 4   # and then reused under the new code


def test_prepare_refuses_a_cell_whose_attempt_holds_a_record(lease):
    render, root, fleet, hosts, _ = lease
    attempt = cd.Path(cd.prepare(render, CELL, root, fleet, hosts)["ATTEMPT"])
    (attempt / f"{CELL}.cell.json").write_text("{}")
    with pytest.raises(cd.ConfigError, match="already hold a record"):
        cd.prepare(render, CELL, root, fleet, hosts)


def test_a_group_is_ready_for_parity_only_when_every_cell_is_done(lease):
    render, root, *_ = lease
    assert cd.group_ready(render, root, "g1") == []
    (root / CELL / "gates").mkdir(parents=True)                     # a verdict without a valid record is not done
    (root / CELL / "gates" / "verdict.json").write_text(json.dumps({"valid": True}))
    scope, _ = cd.load(render)
    assert cd.status(scope, cd.expected(scope, CELL), root) == "invalid" and cd.group_ready(render, root, "g1") == []
