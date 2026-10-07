"""The runner's steps: the attempt directory, the window cut out of the scheduler's logs, the verdict, publishing
and resuming. The scheduler is a stand-in serving /v1/stats and /v1/manifest over real HTTP."""
import asyncio
import hashlib
import json
import shutil
import time
from pathlib import Path

import pytest
from aiohttp import web

from bench.scope import expected_cells
from routebalance.config import ConfigError, fleet_digest, load_fleet
from runner import render as rd, scheduler as sched, store
from runner.__main__ import main
from tests.contract.test_server import free_port
from tests.gates.cellfix import FIXTURES

FLEET = {"tiers": {"qwen2.5-3b_a30": {"model": "Qwen/Qwen2.5-3B", "gpu": "a30", "cost_in": 6e-8, "cost_out": 6e-8,
                                      "max_num_seqs": 256, "stop_token_ids": []}},
         "instances": [{"instance_id": "q3b_0", "tier": "qwen2.5-3b_a30", "node": "n1", "port": 8001}]}
HOSTS = {"n1": {"address": "127.0.0.1", "hw": "d7525"}}
CELL = "g1.qwen.ap.pw070.lor__l12__s1"


ENGINES = {"record_sha256": "a" * 64, "instances": {"e_0": {
    "argv_sha256": "b" * 64, "argv": "--model M --port 8000", "gpus": "0",
    "cmdline": "/venv/bin/python -u -m vllm.entrypoints.openai.api_server --model M --port 8000",
    "start_time_s": 1790000000.0,
    "cache_config": {"block_size": "16", "num_gpu_blocks": "8000"}, "model": "M", "max_model_len": 4096,
    "served": {"requests": 0, "prompt_tokens": 0, "generation_tokens": 0, "prefix_queries": 0, "prefix_hits": 0}}}}


@pytest.fixture
def lease(tmp_path):
    fleet, hosts = tmp_path / "fleet.json", tmp_path / "hosts.json"
    fleet.write_text(json.dumps(FLEET))
    hosts.write_text(json.dumps(HOSTS))
    render = tmp_path / "render"                     # the shipped render, pinned to this test fleet (the guide's step 1)
    shutil.copytree(Path(__file__).resolve().parents[2] / "configs" / "render36", render)
    plan = json.loads((render / "plan.json").read_text())
    for entry in plan["fleets"].values():
        entry["fleet_sha256"] = hashlib.sha256(fleet.read_bytes()).hexdigest()
        entry["fleet_digest"] = fleet_digest(load_fleet(fleet, hosts))
    (render / "plan.json").write_text(json.dumps(plan, indent=1) + "\n")
    scope = json.loads((render / "scope.json").read_text())
    root = tmp_path / "cells"
    root.mkdir()
    return render, root, fleet, hosts, scope


def prepare(render, cell, root, fleet, hosts) -> Path:
    """The attempt of a cell, as the runner makes it: the render read, the fleet checked, the attempt created."""
    scope, plan = rd.load(render)
    exp = rd.expected(scope, cell)
    rd.fleet_of(plan, exp.group, fleet, hosts)
    attempt = store.prepare(scope, exp, render, root)
    (attempt / "gates").mkdir()
    return attempt


def test_prepare_makes_an_attempt_with_the_rendered_configuration(lease):
    render, root, fleet, hosts, scope = lease
    attempt = prepare(render, CELL, root, fleet, hosts)
    assert attempt.name == CELL and attempt.parent.parent == root / "_attempts"
    assert (attempt / "config.json").read_bytes() == (render / "configs" / "g1.qwen.ap.pw070.lor.json").read_bytes()
    assert json.loads((attempt / "group.json").read_text()) == {"group": "g1", "reference": None,
                                                                 "intended_differences": ["system.*",
                                                                                          "telemetry.dead_reckoning"]}
    assert not rd.is_sweep(json.loads((attempt / "config.json").read_text()))
    sweep = prepare(render, "qwen.sweep.random__sweep6-36__s1", root, fleet, hosts)
    assert rd.is_sweep(json.loads((sweep / "config.json").read_text()))
    other = fleet.with_name("other.json")
    other.write_text(json.dumps({**FLEET, "instances": [{**FLEET["instances"][0], "port": 8002}]}))
    with pytest.raises(ConfigError, match="not the"):
        prepare(render, CELL, root, other, hosts)
    (root / CELL).mkdir()
    with pytest.raises(ConfigError, match="never overwritten"):
        prepare(render, CELL, root, fleet, hosts)


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
            await sched.snapshot_before(s.url, attempt, logs, "d" * 64, False)
            ours = [{"config_digest": "d" * 64, "request_id": f"{CELL}.r.{i}", "instance_id": "q3b_0"} for i in range(3)]
            _append(logs / "decisions.jsonl", ours)
            s.stats["logged"]["decisions"] = 4
            await sched.snapshot_after(s.url, attempt, "d" * 64)
            return ours
    ours = asyncio.run(scenario())
    assert [json.loads(x) for x in (attempt / "decisions.jsonl").read_text().splitlines()] == ours
    for name in ("stats_before.json", "stats_after.json", "manifest_before.json", "manifest_after.json"):
        assert (attempt / name).exists()
    window = json.loads((attempt / "window.json").read_text())
    assert window["decisions"]["end"] - window["decisions"]["start"] == len((attempt / "decisions.jsonl").read_bytes())


def test_foreign_or_unflushed_lines_fail_the_window(tmp_path, monkeypatch):
    logs, attempt = tmp_path / "logs", tmp_path / "a" / CELL
    logs.mkdir()
    attempt.mkdir(parents=True)

    async def scenario(extra, logged):
        async with Scheduler() as s:
            await sched.snapshot_before(s.url, attempt, logs, "d" * 64, False)
            _append(logs / "decisions.jsonl", extra)
            s.stats["logged"]["decisions"] = logged
            await sched.snapshot_after(s.url, attempt, "d" * 64)
    with pytest.raises(ConfigError, match="foreign traffic"):
        asyncio.run(scenario([{"config_digest": "e" * 64, "request_id": "x"}], 1))
    for name in ("stats_before.json", "manifest_before.json", "window.json"):
        (attempt / name).unlink()
    with pytest.raises(ConfigError, match="not the running scheduler's"):   # a log directory with another's lines
        asyncio.run(scenario([], 0))
    (logs / "decisions.jsonl").unlink()
    monkeypatch.setattr(sched, "SETTLE_S", 0.3)
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
            await sched.snapshot_before(s.url, attempt, logs, digest, True)
            _append(logs / "decisions.jsonl", [{"config_digest": digest, "request_id": f"r{i}", "instance_id": "i"}
                                               for i in range(2)] + [{"config_digest": digest, "request_id": "r2"}])
            _append(logs / f"latency.{digest[:16]}.jsonl", [{"tier": "t", "n": i} for i in range(2)])
            s.stats["logged"].update(decisions=3, latency=2)
            await sched.snapshot_after(s.url, attempt, digest)
    asyncio.run(scenario())
    assert len((attempt / "latency.jsonl").read_text().splitlines()) == 2   # the unassigned request has none


def _manifest(attempt, instances=("e_0",)):
    """A recorded cell's manifest with the attempt's fleet (the engine check reads the fleet from it)."""
    m = json.loads((FIXTURES / "cells" / "rb-maxdiv__l400__s1" / "manifest_before.json").read_text())
    m["fleet"] = [dict(m["fleet"][0], instance_id=i) for i in instances]
    (attempt / "manifest_before.json").write_text(json.dumps(m))


def test_a_clean_record_with_every_gate_passed_is_valid(lease, monkeypatch):
    """A clean record whose two run gates passed is valid; a missing or failed gate makes it invalid."""
    render, root, fleet, hosts, _ = lease
    scope, _ = rd.load(render)
    attempt = prepare(render, CELL, root, fleet, hosts)
    monkeypatch.setattr(store, "record_problems", lambda scope, exp, d: [])
    monkeypatch.setattr(store, "isolation_problems", lambda d: [])   # its own tests: tests/gates/test_isolation.py
    for when in ("before", "after"):
        (attempt / f"engines_{when}.json").write_text(json.dumps(ENGINES))
    _manifest(attempt)
    assert store.GATES == ("arm", "zero_call")
    assert store.verdict(scope, attempt, {"arm": 0, "zero_call": 0})["valid"] is True
    for gates in ({"arm": 0}, {"arm": 0, "zero_call": 1}):
        (attempt / "gates" / "verdict.json").unlink()
        assert store.verdict(scope, attempt, gates)["valid"] is False


def test_an_engine_restart_during_the_cell_invalidates_it(lease, monkeypatch):
    """the engines are read before and after; any change, a restart first, makes the cell invalid."""
    render, root, fleet, hosts, _ = lease
    scope, _ = rd.load(render)
    attempt = prepare(render, CELL, root, fleet, hosts)
    monkeypatch.setattr(store, "record_problems", lambda scope, exp, d: [])
    monkeypatch.setattr(store, "isolation_problems", lambda d: [])   # its own tests: tests/gates/test_isolation.py
    restarted = json.loads(json.dumps(ENGINES))
    restarted["instances"]["e_0"]["start_time_s"] += 60.0
    (attempt / "engines_before.json").write_text(json.dumps(ENGINES))
    (attempt / "engines_after.json").write_text(json.dumps(restarted))
    _manifest(attempt)
    v = store.verdict(scope, attempt, {"arm": 0, "zero_call": 0})
    assert not v["valid"] and v["bench_problems"] == ["an engine restarted or changed during the cell: "
                                                      "['e_0.start_time_s']"]
    (attempt / "gates" / "verdict.json").unlink()
    _manifest(attempt, ("e_0", "e_1"))                         # an engine of the fleet the record never read
    assert store.verdict(scope, attempt, {"arm": 0, "zero_call": 0})["bench_problems"][0] == (
        "the engine record does not cover the cell's fleet: engines ['e_0'] vs the fleet ['e_0', 'e_1']")
    _manifest(attempt)
    (attempt / "engines_after.json").unlink()
    (attempt / "gates" / "verdict.json").unlink()
    assert "the engine record is missing" in store.verdict(scope, attempt, {"arm": 0, "zero_call": 0})["bench_problems"][0]


def test_status_verdict_publish_and_resume(lease):
    render, root, fleet, hosts, _ = lease
    scope, plan = rd.load(render)
    run, report = store.cells_to_run(scope, plan, ["G1"], root)
    assert len(run) == 66 and run[0].startswith("g1.qwen.rb.") and report == []
    attempt = prepare(render, CELL, root, fleet, hosts)
    with pytest.raises(ConfigError, match="no cell record"):
        store.publish(attempt, root)
    (attempt / f"{CELL}.cell.json").write_text("{}")                 # an unreadable record: invalid, kept
    v = store.verdict(scope, attempt, {"arm": 0, "zero_call": 0})
    assert not v["valid"] and v["bench_problems"]
    published = store.publish(attempt, root)
    assert published == root / CELL and store.status(scope, rd.expected(scope, CELL), root) == "invalid"
    run, report = store.cells_to_run(scope, plan, ["G1"], root)
    assert CELL not in run and any(CELL in line and line.startswith("invalid") for line in report)
    retired = store.retire(root, CELL)
    assert retired.parent == root / "_retired" and CELL in store.cells_to_run(scope, plan, ["G1"], root)[0]
    with pytest.raises(ConfigError, match="not cell rows"):
        store.cells_to_run(scope, plan, ["W1"], root)


def test_the_scheduler_must_run_its_own_code_and_the_plans_launch_values(lease):
    render, *_ = lease
    _, plan = rd.load(render)
    m = Scheduler().manifest
    plan = {**plan, "server": {k: m["launch"][k] for k in plan["server"]}}   # the stand-in's own launch values
    assert sched.code_changes(m) == [] and sched.launch_problems(plan, m) == []
    assert sched.code_changes({"code": {"routebalance": "a", "routebalance_now": "b"}}) == ["routebalance"]
    wrong = sched.launch_problems(plan, {"launch": {**m["launch"], "torch_threads": 8}})
    assert wrong == ["torch_threads: the scheduler runs 8, the plan says 4"]
    assert "not set" in sched.launch_problems({**plan, "server": None}, m)[0]


def test_an_unpublished_record_is_never_run_again(lease):
    render, root, fleet, hosts, _ = lease
    scope, plan = rd.load(render)
    attempt = prepare(render, CELL, root, fleet, hosts)
    assert CELL in store.cells_to_run(scope, plan, ["G1"], root)[0]        # no record yet: run it
    (attempt / f"{CELL}.cell.json").write_text("{}")                      # the runner stopped after the client
    run, report = store.cells_to_run(scope, plan, ["G1"], root)
    assert CELL not in run and any(line.startswith("unpublished") and CELL in line for line in report)


def test_the_rows_are_done_only_when_every_cell_is_and_parity_passed(lease, monkeypatch):
    monkeypatch.setattr(store, "isolation_problems", lambda d: [])   # its own tests: tests/gates/test_isolation.py
    render, root, *_ = lease
    scope, plan = rd.load(render)
    problems, published = store.rows_status(scope, plan, ["G1"], root)
    assert len(problems) == 66 and published == []
    (root / CELL / "gates").mkdir(parents=True)
    (root / CELL / "gates" / "verdict.json").write_text(json.dumps({"valid": False}))
    problems, published = store.rows_status(scope, plan, ["G1"], root)
    assert published == [CELL] and f"invalid: {CELL}" in problems
    assert main(["status", "--render", str(render), "--rows", "G1", "--cells-root", str(root)]) == 1
    # a complete group: parity must have passed over exactly its current cells
    monkeypatch.setattr(store, "status", lambda scope, exp, root: "done")
    for exp in [c for c in expected_cells(scope) if c.group == "g1"]:
        (root / exp.cell_id / "gates").mkdir(parents=True, exist_ok=True)
        (root / exp.cell_id / "gates" / "verdict.json").write_text(json.dumps({"valid": True}))
        (root / exp.cell_id / f"{exp.cell_id}.cell.json").write_text("{}")
    problems, _ = store.rows_status(scope, plan, ["G1"], root)
    assert problems == ["group g1 is complete but parity has not run over its current cells with the current gate code"]
    assert store.parity(scope, plan, root, "g1") == 1          # the records are empty: the parity gate fails
    problems, _ = store.rows_status(scope, plan, ["G1"], root)
    assert len(problems) == 1 and problems[0].startswith("group g1 failed its parity gate")
    monkeypatch.setattr(store, "gate_code", lambda: "f" * 64)    # a record made by other gate code proves nothing
    problems, _ = store.rows_status(scope, plan, ["G1"], root)
    assert problems == ["group g1 is complete but parity has not run over its current cells with the current gate code"]


def test_parity_runs_once_per_set_of_cells_and_keeps_every_run(lease, monkeypatch):
    render, root, *_ = lease
    scope, plan = rd.load(render)
    assert store.parity(scope, plan, root, "g1") is None       # not ready: nothing runs
    for name in ("a", "b", "c"):
        (root / name).mkdir()
        (root / name / f"{name}.cell.json").write_text("run 1")
    cells = [str(root / "a"), str(root / "b")]
    monkeypatch.setattr(store, "group_ready", lambda s, p, r, g: list(cells))
    runs = []
    monkeypatch.setattr(store.subprocess, "run",
                        lambda argv, **kw: runs.append(argv) or type("P", (), {"returncode": 0})())
    assert store.parity(scope, plan, root, "g1") == 0 and store.parity(scope, plan, root, "g1") == 0
    assert len(runs) == 1 and runs[0][-2:] == cells and runs[0][1:4] == ["-m", "gates.parity", "--cells"]
    cells.append(str(root / "c"))                              # a later render adds a cell: parity runs again
    time.sleep(1.1)
    assert store.parity(scope, plan, root, "g1") == 0 and len(runs) == 2
    assert len(list((root / "_groups" / "g1").glob("parity.*.json"))) == 2
    (root / "b" / "b.cell.json").write_text("run 2")           # b retired and run again, at the same path
    time.sleep(1.1)
    assert store.parity(scope, plan, root, "g1") == 0 and len(runs) == 3   # the gate sees the new run
    monkeypatch.setattr(store, "gate_code", lambda: "f" * 64)    # the gate's code was fixed: its verdict is redone
    time.sleep(1.1)
    assert store.parity(scope, plan, root, "g1") == 0 and len(runs) == 4
    assert store.parity(scope, plan, root, "g1") == 0 and len(runs) == 4   # and then reused under the new code


def test_prepare_refuses_a_cell_whose_attempt_holds_a_record(lease):
    render, root, fleet, hosts, _ = lease
    attempt = prepare(render, CELL, root, fleet, hosts)
    (attempt / f"{CELL}.cell.json").write_text("{}")
    with pytest.raises(ConfigError, match="already hold a record"):
        prepare(render, CELL, root, fleet, hosts)


def test_a_group_is_ready_for_parity_only_when_every_cell_is_done(lease):
    render, root, *_ = lease
    scope, plan = rd.load(render)
    assert store.group_ready(scope, plan, root, "g1") == []
    (root / CELL / "gates").mkdir(parents=True)                     # a verdict without a valid record is not done
    (root / CELL / "gates" / "verdict.json").write_text(json.dumps({"valid": True}))
    assert store.status(scope, rd.expected(scope, CELL), root) == "invalid"
    assert store.group_ready(scope, plan, root, "g1") == []


def test_a_requested_row_with_a_pending_block_is_refused_and_an_unknown_group_raises(lease):
    render, root, *_ = lease
    scope, plan = rd.load(render)
    for r in plan["rows"]:
        if r["id"] == "G1":
            r["pending"] = {"G1-x": "pins missing: ap.qwen.pw080"}
    with pytest.raises(ConfigError, match="G1/G1-x: pins missing"):
        store.cells_to_run(scope, plan, ["G1"], root)
    with pytest.raises(ConfigError, match="has no cells"):
        store.group_ready(scope, plan, root, "no-such-group")


@pytest.mark.parametrize("rows", ["", ","])
def test_a_row_selection_that_names_no_row_is_refused(lease, rows):
    render, root, *_ = lease
    with pytest.raises(SystemExit) as e:
        main(["status", "--render", str(render), "--rows", rows, "--cells-root", str(root)])
    assert e.value.code == 2


def test_a_cell_is_refused_before_it_runs_on_a_scheduler_that_has_served_requests():
    """A scheduler that was not restarted must cost no run, so the runner refuses it up front."""
    async def scenario():
        async with Scheduler() as s:
            s.stats["logged"]["decisions"] = 0
            await sched.require_fresh(s.url)
            s.stats["logged"]["decisions"] = 3
            with pytest.raises(ConfigError, match="not a fresh process"):
                await sched.require_fresh(s.url)
            s.stats["logged"].update(decisions=0, decisions_buffered=1)      # written, not yet flushed
            with pytest.raises(ConfigError, match="not a fresh process"):
                await sched.require_fresh(s.url)
    asyncio.run(scenario())


# ---------------------------------------------------------------- one cell's steps, in order

class Steps:
    """Every step of a cell that reaches outside the cells root, faked and recorded in call order."""

    def __init__(self, monkeypatch, plan, *, fresh=True, record=True, valid=True, mirrored=True, parity=None,
                 launch=None, manifest=True):
        from runner import cell, mirror
        self.calls, self.env, self.posted, self.preflight = [], None, None, None
        server = dict(plan["server"]) if launch is None else launch

        def note(name, result=None):
            async def step(*a, **kw):
                self.calls.append(name)
                return result
            return step

        async def post_config(url, attempt, config_digest, fleet_digest):
            self.calls.append("post_config")
            self.posted = (config_digest, fleet_digest)

        async def require_fresh(url):
            self.calls.append("fresh")
            if not fresh:
                raise ConfigError("the scheduler has logged 3 decisions: not a fresh process")

        async def engines(instances, record_path, out):
            self.calls.append(out.name)
            out.write_text("{}")

        async def before(url, attempt, log_dir, digest, is_sweep):
            self.calls.append("snapshot_before")
            (attempt / "manifest_before.json").write_text(json.dumps(
                {"code": {"routebalance": "a", "routebalance_now": "a"}, **({"launch": server} if manifest else {})}))

        def to_file(cmd, path):
            self.calls.append(cmd[2])                       # the module after -m: gates.preflight, gates.arm, ...
            if cmd[2] == "gates.preflight":
                self.preflight = (cmd[cmd.index("--expect-config-digest") + 1], cmd[cmd.index("--expect-fleet-digest") + 1])
            return 0

        def relay(cmd, log, env=None):
            self.calls.append(cmd[2])
            self.env = env
            if record:
                (Path(cmd[cmd.index("--out-dir") + 1]) / f"{cmd[cmd.index('--cell') + 1]}.cell.json").write_text("{}")
            return 0

        def verdict(scope, attempt, gates):
            self.calls.append("verdict")
            return {"cell_id": attempt.name, "valid": valid, "bench_problems": [], "gates": gates}

        def mirror_cell(root, dest, cell_id, say):
            self.calls.append("mirror_cell")
            return mirrored

        def parity_(scope, plan, root, group):
            self.calls.append("parity")
            return parity

        def bookkeeping(root, dest, say, names=()):
            self.calls.append(f"mirror{list(names)}")
            return True
        monkeypatch.setattr(sched, "drain", note("drain", 0.0))
        monkeypatch.setattr(sched, "require_fresh", require_fresh)
        monkeypatch.setattr(sched, "post_config", post_config)
        monkeypatch.setattr(sched, "snapshot_before", before)
        monkeypatch.setattr(sched, "snapshot_after", note("snapshot_after"))
        monkeypatch.setattr(cell, "_engines", engines)
        monkeypatch.setattr(cell, "_warm_up", note("warm_up"))
        monkeypatch.setattr(cell, "_to_file", to_file)
        monkeypatch.setattr(cell, "_relay", relay)
        monkeypatch.setattr(store, "verdict", verdict)
        monkeypatch.setattr(store, "parity", parity_)
        monkeypatch.setattr(mirror, "mirror_cell", mirror_cell)
        monkeypatch.setattr(mirror, "mirror_bookkeeping", bookkeeping)


def _run_cell(lease, tmp_path):
    from runner import cell
    render, root, fleet, hosts, _ = lease
    setup = cell.Setup(render=render, scheduler="http://127.0.0.1:9", scheduler_log_dir=tmp_path / "logs",
                       fleet=fleet, hosts=hosts, engines_record=tmp_path / "engines.record", data_root=tmp_path,
                       cells_root=root, mirror=str(tmp_path / "mirror"))
    return cell.run(setup, CELL)


ORDER = ["drain", "fresh", "post_config", "gates.preflight", "engines_before.json", "warm_up", "snapshot_before",
         "bench.runner", "snapshot_after", "engines_after.json", "gates.arm", "gates.zero_call", "verdict", "mirror_cell",
         "parity", "mirror['_groups']"]


def test_a_cell_runs_its_steps_in_order_and_is_published(lease, tmp_path, monkeypatch):
    """The engines are read before the warm-up (the traffic check covers the cache reset) and after the client; a
    used scheduler is refused before the configuration is posted; the client runs offline in a clean environment."""
    from runner import cell
    render, root, *_ = lease
    scope, plan = rd.load(render)
    steps = Steps(monkeypatch, plan)
    assert _run_cell(lease, tmp_path) == cell.VALID
    assert steps.calls == ORDER
    assert steps.env == cell.node_env() and steps.env["HF_HUB_OFFLINE"] == "1"
    # the scheduler is posted the rendered configuration and fleet, and preflight expects exactly those digests
    want = (rd.expected(scope, CELL).config.config_digest, plan["fleets"][plan["groups"]["g1"]["fleet"]]["fleet_digest"])
    assert steps.posted == steps.preflight == want
    assert (root / CELL / f"{CELL}.cell.json").exists()
    (log,) = (root / "_attempts").glob(f"{CELL}.*/driver.log")      # the runner's log stays beside the attempt
    assert "valid" in log.read_text().splitlines()[-1]


@pytest.mark.parametrize("fault,rc,last", [
    ({"fresh": False}, 2, "fresh"),                    # a used scheduler: refused before the configuration
    ({"record": False}, 2, "bench.runner"),            # no record: the cell did not run
    ({"launch": {"torch_threads": 8, "max_connections": 4096, "request_timeout_s": 21600.0}}, 2, "snapshot_before"),
    ({"manifest": False}, 2, "snapshot_before"),       # a manifest the check cannot read: the cell does not run
    ({"valid": False}, 1, "mirror['_groups']"),        # invalid: published and kept
    ({"mirrored": False}, 3, "mirror_cell"),
    ({"parity": 1}, 1, "mirror['_groups']"),           # its group failed parity: the cell counts as invalid
])
def test_a_cells_faults_stop_it_where_they_happen(lease, tmp_path, monkeypatch, fault, rc, last):
    render, root, *_ = lease
    _, plan = rd.load(render)
    steps = Steps(monkeypatch, plan, **fault)
    assert _run_cell(lease, tmp_path) == rc
    assert steps.calls == ORDER[:ORDER.index(last) + 1]
    assert (root / CELL).is_dir() == (rc != 2)          # a cell that did not run stays an attempt


def test_a_malformed_manifest_is_not_current_and_stops_without_a_restart_command():
    """A manifest that cannot be read counts as stale code: with no --restart the rows stop."""
    said = []

    async def scenario():
        async with Scheduler() as s:
            s.manifest = {}
            return await asyncio.get_running_loop().run_in_executor(None, sched.current, s.url, None, said.append)
    assert asyncio.run(scenario()) is False
    assert any("could not be read" in m for m in said) and any("restart it" in m for m in said)


def test_a_cell_that_cannot_be_prepared_never_reaches_the_scheduler(lease, tmp_path, monkeypatch):
    """Another fleet, or a cell already published: refused before anything is sent (exit 2, no attempt)."""
    from runner import cell
    render, root, fleet, hosts, _ = lease
    _, plan = rd.load(render)
    steps = Steps(monkeypatch, plan)
    other = fleet.with_name("other.json")
    other.write_text(json.dumps({**FLEET, "instances": [{**FLEET["instances"][0], "port": 8002}]}))
    setup = cell.Setup(render=render, scheduler="http://127.0.0.1:9", scheduler_log_dir=tmp_path / "logs",
                       fleet=other, hosts=hosts, engines_record=tmp_path / "engines.record", data_root=tmp_path,
                       cells_root=root, mirror=str(tmp_path / "mirror"))
    assert cell.run(setup, CELL) == cell.NOT_RUN and steps.calls == [] and not (root / "_attempts").exists()
    (root / CELL).mkdir()
    assert _run_cell(lease, tmp_path) == cell.NOT_RUN and steps.calls == [] and not (root / "_attempts").exists()


def test_the_rows_stop_on_the_invalid_budget_or_a_cell_that_could_not_run(lease, tmp_path, monkeypatch):
    """The rows run their cells in the plan's order; more than --max-invalid invalid cells, or one that could not
    run, stops them with its code; rows that end with cells not done exit 1."""
    from runner import cell, mirror, rows
    render, root, fleet, hosts, _ = lease
    setup = cell.Setup(render=render, scheduler="http://127.0.0.1:9", scheduler_log_dir=tmp_path / "logs",
                       fleet=fleet, hosts=hosts, engines_record=tmp_path / "engines.record", data_root=tmp_path,
                       cells_root=root, mirror=str(tmp_path / "mirror"))
    monkeypatch.setattr(mirror, "mirror_bookkeeping", lambda *a, **k: True)
    monkeypatch.setattr(mirror, "mirror_cell", lambda *a, **k: True)
    monkeypatch.setattr(sched, "current", lambda *a: True)
    scope, plan = rd.load(render)
    order = [c.cell_id for c in rd.requested(scope, plan, ["G1"])]

    def script(codes):
        ran = []
        monkeypatch.setattr(cell, "run", lambda s, c: ran.append(c) or (codes[len(ran) - 1] if len(ran) <= len(codes)
                                                                          else cell.VALID))
        return ran
    ran = script([cell.INVALID, cell.INVALID])
    assert rows.run(setup, ["G1"], 1, None, None) == cell.INVALID and ran == order[:2]
    ran = script([cell.INVALID, cell.NOT_RUN])
    assert rows.run(setup, ["G1"], 5, None, None) == cell.NOT_RUN and ran == order[:2]
    ran = script([cell.MIRROR_FAILED])
    assert rows.run(setup, ["G1"], 5, None, None) == cell.MIRROR_FAILED and ran == order[:1]
    ran = script([])                                   # every cell "valid", but none was published: not done
    assert rows.run(setup, ["G1"], 5, None, None) == cell.INVALID and ran == order
