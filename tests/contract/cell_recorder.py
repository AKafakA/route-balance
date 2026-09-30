"""Record real cell directories for the gates. The parts are real where it matters:
  - the scheduler (routebalance/server.py, with the real load_components) behind fake vLLM engines;
  - the runner that drives it (bench/runner.py);
  - the layout, which follows scripts/run_cell.sh (gates/cell.py).

The recorder is used in two places. The contract tests make a live run that every gate must pass. The same
code re-records the fixtures committed under tests/gates:
    python -m tests.contract.cell_recorder tests/gates/fixtures
Only the parts that would need downloaded artifacts are stand-ins: the KNN estimator (a per-prompt stub) and
the encoder snapshot hashes. The latency tables, the routers, the dispatchers and the artifact bookkeeping all
run the real code.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
import socket
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import aiohttp
from aiohttp import web

import routebalance.server as server
from tests.traces import constant_trace
from bench.client import HttpSchedulerClient
from bench.dataset import load_prompts
from bench.records import Sampling, write_cell
from bench.runner import CellSpec, run_cell
from routebalance.config import config_digest, load_fleet, parse_scheduler
from tests.contract.fake_vllm import FakeState, make_app
from tests.contract.scheduler_http import ServedScheduler
from tests.unit.fakes import SAMPLING, FakeTokenizer, write_dataset
from tools.engines import engines

# group -> (reference config id or None, {config id: its intended differences}): the group.json of each cell.
# g1 is a main comparison group of five systems, with RouteBalance on the pollers' cache (D54) and v3's
# per-request probe arm (D55, dead reckoning off); w4-dr an ablation group (dead reckoning off vs its reference).
GROUPS = {
    "g1": (None, {cid: ["system.*", "telemetry.dead_reckoning"]
                  for cid in ("rb-maxdiv", "rb-round", "rb-fixedref", "rb-min", "rb-mean", "rb-off", "rb-poll", "lor",
                              "lor-probe", "decoupled", "sweep")}),
    "w4-dr": ("w4-ref", {"w4-ref": [], "w4-nodr": ["telemetry.dead_reckoning"]}),
}
N_REQUESTS, RATE = 16, 400.0          # bursts that fill RouteBalance's rounds
TIERS = {"small_a30": {"model": "S", "gpu": "a30", "cost_in": 1e-7, "cost_out": 1e-7, "max_num_seqs": 8,
                       "stop_token_ids": []},
         "large_a100": {"model": "L", "gpu": "a100", "cost_in": 4e-7, "cost_out": 4e-7, "max_num_seqs": 8,
                        "stop_token_ids": []}}
# (instance, tier, node, hardware type, loopback address): one address per hardware type, as load_fleet requires
INSTANCES = [("S_0", "small_a30", "n0", "d7525", "127.0.0.1"), ("S_1", "small_a30", "n1", "d7525", "127.0.0.1"),
             ("L_2", "large_a100", "n2", "d8545", "127.0.0.2")]
TABLE = {"unit": "ms_per_token", "constant": {"small_a30": 10.0, "large_a100": 40.0},
         "curve": {"small_a30": [[1, 10.0], [8, 30.0]], "large_a100": [[1, 40.0], [8, 90.0]]},
         "provenance": {"source": "tests.contract.cell_recorder"}}
PREFILL = {"unit": "ms", "prefill": {"small_a30": {"intercept_ms": 5.0, "per_token_ms": 0.05, "n": 64, "r2": 0.9},
                                     "large_a100": {"intercept_ms": 20.0, "per_token_ms": 0.1, "n": 64, "r2": 0.9}},
           "provenance": {"source": "tests.contract.cell_recorder"}}
ENCODER = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass(frozen=True)
class Est:
    quality: float
    length: float
    bucket_probs: tuple = (1.0,)
    p_under_budget: float = 1.0


class VaryingEstimator:
    """Per-prompt qualities and lengths, so a round holds distinct LPT keys."""
    index = types.SimpleNamespace(encoder_name=ENCODER)

    def estimate_batch(self, prompts, budget_tokens):
        out = []
        for p in prompts:
            h = int(hashlib.sha256(p.encode()).hexdigest()[:8], 16)
            out.append({"S": Est(0.3 + (h % 7) / 20, 8.0 + h % 40), "L": Est(0.8, 12.0 + (h >> 8) % 60)})
        return out


def fake_knn_module() -> types.ModuleType:
    mod = types.ModuleType("routebalance.estimate.knn")
    mod.INDEX_FILE = "knn_estimator.pkl"
    mod.load_knn = lambda cfg, root, models: VaryingEstimator()
    return mod


def fake_model_files(role: str, name: str, revision) -> dict[str, str]:
    return {f"{role}@hf:{name}/model.safetensors": hashlib.md5(f"{name}@{revision}".encode()).hexdigest()}


def install_stand_ins(setitem, setattr_) -> None:
    """setitem/setattr_: monkeypatch.setitem / monkeypatch.setattr in tests, or plain assignment."""
    setitem(sys.modules, "routebalance.estimate.knn", fake_knn_module())
    setattr_(server, "model_files", fake_model_files)


def configs(table_md5: str, prefill_md5: str) -> dict[str, dict]:
    base = {
        "dead_reckoning": True,
        "latency": {"source": "constant", "envelope": "own_length", "normalization": "max_divide",
                    "table": "calib/tpot.json", "table_md5": table_md5, "prefill": None},
        "estimator": {"index_dir": "knn/x", "index_md5": "0" * 32, "encoder_revision": "a" * 40, "k": 50,
                      "score_type": "deepeval", "prior": "knn"},
        "budget": {"admission_filter": False},
        "telemetry": {"poll_ms": 20.0, "timeout_s": 2.0},
    }
    weights = {"w_quality": 0.5, "w_latency": 0.3, "w_cost": 0.2}
    curve = {"source": "profiled_curve", "envelope": "batch_aware", "normalization": "fixed_ref",     # with prefill (D118)
             "table": "calib/tpot.json", "table_md5": table_md5,
             "prefill": {"table": "calib/prefill.json", "table_md5": prefill_md5}}

    def rb(**kw):
        return base | {"mode": "routebalance", "normalize_over": "round", "weights": weights, "telemetry_read": "per_round",
                       "pipeline": None} | kw

    def pipe(router, dispatch, **kw):
        return base | {"mode": "pipeline", "normalize_over": None, "weights": None, "rounds": {"adaptive": True},
                       "lpt": "max",
                       "pending_decode": None, "telemetry_read": "poll", "dead_reckoning": False,
                       "pipeline": {"router": router, "dispatch": dispatch}} | kw
    passthrough = {"type": "passthrough", "kwargs": {}}
    maxdiv = rb(rounds={"adaptive": False, "max_batch": 4, "timeout_ms": 30.0}, lpt="max",
                pending_decode="own_length",               # cost and latency heavy: both tiers win requests
                weights={"w_quality": 0.2, "w_latency": 0.4, "w_cost": 0.4},
                normalize_over="request")                  # v3's per-request scale (D137: "round" is the main one)
    return {
        "rb-maxdiv": maxdiv,
        "rb-round": maxdiv | {"normalize_over": "round", "lpt": "off", "pending_decode": "lpt_key"},   # D137: the
        #           round's scales; arrival order, so a later request can raise the running latency maximum
        "w4-ref": maxdiv,
        "w4-nodr": maxdiv | {"dead_reckoning": False},
        "rb-fixedref": rb(rounds={"adaptive": True}, lpt="max", pending_decode="lpt_key", latency=curve),
        "rb-min": rb(rounds={"adaptive": False, "max_batch": 8, "timeout_ms": 30.0}, lpt="min",
                     pending_decode="lpt_key"),               # the LPT-key ablation (D41); decode = the key
        "rb-mean": rb(rounds={"adaptive": False, "max_batch": 8, "timeout_ms": 30.0}, lpt="mean",
                      pending_decode="own_length", latency=curve),
        "rb-off": rb(rounds={"adaptive": False, "max_batch": 8, "timeout_ms": 30.0}, lpt="off", normalize_over="request",
                     pending_decode="lpt_key"),               # arrival order; lpt_key and decode = the max
        "rb-poll": maxdiv | {"telemetry_read": "poll"},       # D54: RouteBalance on the pollers' cache
        "lor": pipe(passthrough, {"type": "least_outstanding", "seed": 1, "count": "engine"}),   # the main grid (D84)
        "lor-probe": pipe(passthrough, {"type": "least_outstanding", "seed": 1, "count": "engine"},
                          telemetry_read="per_request",
                          dead_reckoning=False),           # D55: v3's per-request probe on the engine's counts
        "decoupled": pipe({"type": "rb_decoupled", "kwargs": {}, "concurrency": 32}, {"type": "round_robin", "seed": 1},
                          weights=weights, latency=curve),
        "sweep": pipe(passthrough, {"type": "random", "seed": 7}, pending_decode="own_length",
                       dead_reckoning=True),   # the sweep logs RouteBalance's view (its random dispatch reads no load)
    }


def free_port(address: str = "127.0.0.1") -> int:
    with socket.socket() as s:
        s.bind((address, 0))
        return s.getsockname()[1]


class LiveFleet:
    """Fake engines on local ports, the fleet and hosts files that describe them, and the real scheduler."""

    def __init__(self, work: Path):
        self.work = work
        self.states = {iid: FakeState(served=(TIERS[tier]["model"],)) for iid, tier, *_ in INSTANCES}

    async def __aenter__(self) -> "LiveFleet":
        w = self.work
        (w / "artifacts/calib").mkdir(parents=True)
        table, prefill = json.dumps(TABLE).encode(), json.dumps(PREFILL).encode()
        (w / "artifacts/calib/tpot.json").write_bytes(table)
        (w / "artifacts/calib/prefill.json").write_bytes(prefill)
        self.configs = configs(hashlib.md5(table).hexdigest(), hashlib.md5(prefill).hexdigest())
        ports = {iid: free_port(address) for iid, _, _, _, address in INSTANCES}
        self.fleet_path, self.hosts_path = w / "fleet.json", w / "hosts.json"
        self.fleet_path.write_text(json.dumps({"tiers": TIERS, "instances": [
            {"instance_id": iid, "tier": tier, "node": node, "port": ports[iid]} for iid, tier, node, *_ in INSTANCES]}))
        self.hosts_path.write_text(json.dumps({node: {"address": address, "hw": hw}
                                               for _, _, node, hw, address in INSTANCES}))
        self.fleet = load_fleet(self.fleet_path, self.hosts_path)
        self.engines_record = w / "engines.record"            # as deploy.sh engines --record writes it (D48)
        argv = {i.instance_id: f"--model {i.model} --port {i.port} --max-num-seqs {i.tier.max_num_seqs}" for i in self.fleet}
        self.engines_record.write_text("# deploy.sh engines (recorder)\n" + "".join(
            f"{i.instance_id}\t{i.node}\t{i.address}\t{i.port}\t0\t{i.tier.gpu}\t1\t{i.model}\t{argv[i.instance_id]}\n"
            for i in self.fleet) + "".join(
            f"started\t{i.instance_id}\t{self.states[i.instance_id].start_time_s}\t0\t"
            f"/venv/bin/python -u -m vllm.entrypoints.openai.api_server {argv[i.instance_id]}\n" for i in self.fleet))
        self.engines = []
        for inst in self.fleet:
            r = web.AppRunner(make_app(self.states[inst.instance_id]))
            await r.setup()
            await web.TCPSite(r, inst.address, inst.port).start()
            self.engines.append(r)
        self.service = server.Service(self.fleet, artifact_root=w / "artifacts", avengers_repo=None,
                                      request_timeout_s=5.0, max_connections=64, torch_threads=1, log_dir=w / "logs")
        await self.service.start(parse_scheduler(self.configs["lor"]))
        self.port = free_port()
        self.app = await ServedScheduler(self.service, self.port).start()
        self.url = f"http://127.0.0.1:{self.port}"
        self.http = aiohttp.ClientSession()
        return self

    async def get(self, path: str):
        async with self.http.get(self.url + path) as r:
            assert r.status == 200, (path, r.status, await r.text())
            return await r.json()

    async def post_config(self, doc: dict) -> None:
        async with self.http.post(self.url + "/v1/config", data=json.dumps(doc)) as r:
            assert r.status == 200, await r.text()

    async def flushed_stats(self) -> dict:
        await self.service.decisions.flush()        # run_cell.sh polls until decisions_buffered is 0
        return await self.get("/v1/stats")

    def log_lines(self) -> list[str]:
        path = self.work / "logs/decisions.jsonl"
        return path.read_text().splitlines() if path.exists() else []

    async def record_cell(self, config_id: str, cells_root: Path, *, group: str, run_id: str = "r1") -> Path:
        doc = self.configs[config_id]
        reference, declared = GROUPS[group]
        digest = config_digest(parse_scheduler(doc))
        await self.post_config(doc)
        engines_before = await engines(self.fleet, self.engines_record, 5.0)
        stats_before = await self.flushed_stats()
        manifest_before = await self.get("/v1/manifest")
        offset = len(self.log_lines())
        trace = constant_trace(N_REQUESTS, RATE, 1.0)
        cell_id = f"{config_id}__l400__s1"
        data = self.work / "data"
        data.mkdir(exist_ok=True)
        prompts = load_prompts(write_dataset(data, N_REQUESTS), N_REQUESTS, "fixed", 1, FakeTokenizer(), "tok-id")
        labels = {"block": group, "group": group, "config_id": config_id, "trace_id": "l400", "system": config_id,
                  "knob": "k", "dispatch": "d"}
        spec = CellSpec(cell_id=cell_id, labels=labels, trace=trace, seed=1, sampling=Sampling(**SAMPLING),
                        expected_config_digest=digest)
        async with HttpSchedulerClient(self.url, 64, 10.0) as client:
            record = await run_cell(spec, prompts, client, run_id)
        stats_after = await self.flushed_stats()
        manifest_after = await self.get("/v1/manifest")
        engines_after = await engines(self.fleet, self.engines_record, 5.0)
        window = [json.loads(x) for x in self.log_lines()[offset:]]
        cell = cells_root / cell_id
        cell.mkdir(parents=True)
        write_cell(record, cell)
        (cell / "config.json").write_text(json.dumps(doc, indent=1))
        (cell / "group.json").write_text(json.dumps({"group": group, "reference": reference,
                                                     "intended_differences": declared[config_id]}))
        (cell / "decisions.jsonl").write_text("".join(json.dumps(x) + "\n" for x in window
                                                      if x["config_digest"] == digest))
        for name, obj in (("stats_before", stats_before), ("stats_after", stats_after),
                          ("manifest_before", manifest_before), ("manifest_after", manifest_after),
                          ("engines_before", engines_before), ("engines_after", engines_after)):
            (cell / f"{name}.json").write_text(json.dumps(obj, indent=1))
        return cell

    async def __aexit__(self, *exc) -> None:
        await self.http.close()
        await self.app.stop()
        await self.service.stop()
        for r in self.engines:
            await r.cleanup()


    async def record_preflight(self, out: Path) -> None:
        """The replies preflight reads, from one engine and from the scheduler, and the files and digests it is
        given: the committed fixtures of tests/gates/test_preflight.py."""
        out.mkdir(parents=True)
        inst = self.fleet[0]
        for path, name in (("/version", "version.json"), ("/v1/models", "models.json"), ("/metrics", "metrics.txt")):
            async with self.http.get(inst.base_url + path) as r:
                assert r.status == 200
                (out / f"engine_{name}").write_text(await r.text())
        for path, name in (("/v1/config", "config.json"), ("/v1/manifest", "manifest.json"), ("/v1/stats", "stats.json")):
            (out / f"scheduler_{name}").write_text(json.dumps(await self.get(path), indent=1))
        shutil.copy(self.fleet_path, out / "fleet.json")
        shutil.copy(self.hosts_path, out / "hosts.json")
        (out / "expect.json").write_text(json.dumps({"config_digest": self.service.digest,
                                                     "fleet_digest": self.service.fleet_digest}))


async def record_all(work: Path, out: Path) -> list[Path]:
    async with LiveFleet(work) as live:
        cells = [await live.record_cell(cid, out / "cells", group=g) for g, (_, cfgs) in GROUPS.items() for cid in cfgs]
        await live.post_config(live.configs["lor"])          # a polling system: preflight checks its cache ages (D53)
        await live.flushed_stats()
        await live.record_preflight(out / "preflight")
        return cells


def main(argv: list[str]) -> int:
    """Re-record the committed fixtures: <out>/cells/<cell_id>/ for every configuration, and <out>/preflight/."""
    out = Path(argv[0])
    install_stand_ins(lambda d, k, v: d.__setitem__(k, v), setattr)
    work = out / ".recording"
    for d in (work, out / "cells", out / "preflight"):
        if d.exists():
            shutil.rmtree(d)
    work.mkdir(parents=True)
    cells = asyncio.run(record_all(work, out))
    shutil.rmtree(work)
    print("\n".join(str(c) for c in cells))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
