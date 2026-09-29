"""The execution manifest (GET /v1/manifest) and the decision-line evidence the gates read, over real HTTP."""
import asyncio
import hashlib
import subprocess

import aiohttp

from routebalance.server import PACKAGE_DIR, py_tree_digest
from routebalance.server import Service
from tests.contract.test_server import RB, Harness, body, factory, fleet_states, pipeline_cfg, read_jsonl

MANIFEST_KEYS = {"format", "config_digest", "fleet_digest", "fleet", "system", "budget", "telemetry", "loaded",
                 "artifacts", "code", "threadpools", "versions", "launch"}


def shell_tree_digest(directory) -> str:
    """The shell command documented in routebalance.server.py_tree_digest."""
    listing = subprocess.run("find . -name '*.py' -printf '%P\\n' | LC_ALL=C sort | xargs sha256sum", shell=True,
                             cwd=directory, capture_output=True, check=True).stdout
    return hashlib.sha256(listing).hexdigest()


def test_the_code_digest_is_the_documented_shell_listing(tmp_path):
    (tmp_path / "pkg/sub").mkdir(parents=True)
    (tmp_path / "pkg/a.py").write_text("A = 1\n")
    (tmp_path / "pkg/sub/b.py").write_text("B = 2\n")
    (tmp_path / "pkg/notes.txt").write_text("not code\n")
    assert py_tree_digest(tmp_path / "pkg") == shell_tree_digest(tmp_path / "pkg")
    assert py_tree_digest(PACKAGE_DIR) == shell_tree_digest(PACKAGE_DIR)
    before = py_tree_digest(tmp_path / "pkg")
    (tmp_path / "pkg/sub/b.py").write_text("B = 3\n")
    assert py_tree_digest(tmp_path / "pkg") != before


def test_the_manifest_reports_what_this_process_executes(tmp_path):
    async def scenario():
        async with Harness(tmp_path, fleet_states()) as h:
            await h.start(RB)
            status, man = await h.get("/v1/manifest")
            _, ident = await h.get("/v1/config")
            return status, man, ident, h.fleet
    status, man, ident, fleet = asyncio.run(scenario())
    assert status == 200 and man["format"] == "rb-manifest/1" and set(man) == MANIFEST_KEYS
    assert (man["config_digest"], man["fleet_digest"]) == (ident["config_digest"], ident["fleet_digest"])
    assert [f["instance_id"] for f in man["fleet"]] == [i.instance_id for i in fleet]
    assert man["fleet"][2] == {"instance_id": "L_2", "tier": "large_a100", "model": "L", "gpu": "a100",
                               "cost_in": 4e-7, "cost_out": 4e-7, "max_num_seqs": 8, "node": "n2"}   # no host or port
    assert man["system"] == {k: RB[k] for k in ("mode", "weights", "rounds", "lpt", "pending_decode", "latency",
                                                "estimator")} | {"router": None, "dispatch": None,
                                                                 "telemetry_read": "per_round"}   # D53
    assert man["budget"] == RB["budget"]
    assert man["telemetry"] == RB["telemetry"] | {"dead_reckoning": True}
    assert man["loaded"] == {"estimator": "StubEstimator", "latency_source": "constant",
                             "latency_envelope": "own_length", "router": None, "dispatcher": None, "round_loop": True,
                             "telemetry_pollers": False}
    assert man["artifacts"] == {}                  # the test factory loads no files
    digest = shell_tree_digest(PACKAGE_DIR)
    assert man["code"] == {"routebalance": digest, "routebalance_now": digest, "avengers_pro": None,
                           "avengers_pro_now": None}
    assert all(isinstance(n, int) for n in man["threadpools"].values())
    assert man["versions"]["aiohttp"] == aiohttp.__version__ and man["versions"]["python"]
    launch = man["launch"]
    # D134: vLLM bench's stock connector for both sessions (no limit, keep-alive), whatever max_connections says
    assert launch["engine_session"] == {"limit": 0, "limit_per_host": 0, "force_close": False}
    assert launch["telemetry_session"] == {"limit": 0, "limit_per_host": 0, "force_close": False}
    assert (launch["torch_threads"], launch["max_connections"], launch["request_timeout_s"],
            launch["router_threads"], launch["router_lane_threads"]) == (1, 64, 5.0, 4, 1)
    assert launch["event_loop"].startswith("asyncio.") or launch["event_loop"].startswith("uvloop")
    assert launch["cpus"]["count"] >= launch["cpus"]["affinity"] >= 1
    assert set(launch["env"]) >= {"OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS"}


def test_the_manifest_follows_a_config_swap(tmp_path):
    async def scenario():
        async with Harness(tmp_path, fleet_states()) as h:
            await h.start(RB)
            _, before = await h.get("/v1/manifest")
            await h.post("/v1/config", pipeline_cfg("least_outstanding"))
            _, after = await h.get("/v1/manifest")
            return before, after
    before, after = asyncio.run(scenario())
    assert after["config_digest"] != before["config_digest"] and after["system"]["mode"] == "pipeline"
    assert after["system"]["dispatch"] == {"type": "least_outstanding", "seed": 1, "latency_window": None,
                                           "count": "engine"}
    assert after["loaded"] == {"estimator": None, "latency_source": None, "latency_envelope": None,
                               "router": "PassthroughRouter", "dispatcher": "LeastOutstanding", "round_loop": False,
                               "telemetry_pollers": True}
    assert all(after[k] == before[k] for k in ("fleet_digest", "fleet", "budget", "code", "versions"))
    # dead reckoning is RouteBalance's own; the pipeline system reads the engines' counts (owner 09-25)
    assert before["telemetry"]["dead_reckoning"] is True and after["telemetry"]["dead_reckoning"] is False
    assert {k: v for k, v in after["telemetry"].items() if k != "dead_reckoning"} == \
        {k: v for k, v in before["telemetry"].items() if k != "dead_reckoning"}


def test_routebalance_lines_carry_the_round_position_limits_and_lengths(tmp_path):
    async def scenario():
        async with Harness(tmp_path, fleet_states()) as h:
            await h.start(RB)
            await asyncio.gather(*(h.post("/v1/completions", body(i)) for i in range(10)))
            await h.service.decisions.flush()
    asyncio.run(scenario())
    rounds = {}
    for d in read_jsonl(tmp_path / "logs/decisions.jsonl"):
        assert (d["round_max_batch"], d["round_timeout_ms"]) == (4, 20.0)
        assert d["lengths"] == {"S": 20.0, "L": 30.0} and d["feature_tokens"] == 30
        rounds.setdefault(d["round"], []).append(d)
    for members in rounds.values():
        assert sorted(d["round_pos"] for d in members) == list(range(members[0]["round_size"]))


def test_the_sweep_lines_carry_the_lengths_and_baseline_lines_do_not(tmp_path):
    sweep = pipeline_cfg("random")
    sweep["pending_decode"] = "own_length"
    sweep["dead_reckoning"] = True          # the sweep logs RouteBalance's view

    async def scenario():
        async with Harness(tmp_path, fleet_states()) as h:
            await h.start(sweep)
            await h.post("/v1/completions", body(0))
            await h.post("/v1/config", pipeline_cfg())
            await h.post("/v1/completions", body(1))
            await h.service.decisions.flush()
    asyncio.run(scenario())
    sweep_line, baseline_line = read_jsonl(tmp_path / "logs/decisions.jsonl")
    assert sweep_line["lengths"] == {"S": 20.0, "L": 30.0} and "lengths" not in baseline_line


def test_stats_count_the_decision_lines_not_yet_on_disk(tmp_path):
    async def scenario():
        async with Harness(tmp_path, fleet_states()) as h:
            await h.start(pipeline_cfg())
            for i in range(3):
                await h.post("/v1/completions", body(i))
            _, before = await h.get("/v1/stats")
            await h.service.decisions.flush()
            _, after = await h.get("/v1/stats")
            return before["logged"], after["logged"]
    before, after = asyncio.run(scenario())
    assert before["decisions"] + before["decisions_buffered"] == 3
    assert after == {"decisions": 3, "decisions_buffered": 0, "latency": None}


def test_a_code_change_on_disk_after_start_shows_in_the_manifest(tmp_path):
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    (upstream / "simple_cluster_router.py").write_text("X = 1\n")

    async def scenario():
        async with Harness(tmp_path, fleet_states()) as h:
            h.service = Service(h.fleet, artifact_root=tmp_path, avengers_repo=upstream, request_timeout_s=5.0,
                                max_connections=64, torch_threads=1, log_dir=tmp_path / "logs", factory=factory)
            await h.start(RB)
            _, before = await h.get("/v1/manifest")
            (upstream / "simple_cluster_router.py").write_text("X = 2\n")     # e.g. an rsync during the campaign
            _, after = await h.get("/v1/manifest")
            return before["code"], after["code"]
    before, after = asyncio.run(scenario())
    assert before["avengers_pro"] == before["avengers_pro_now"] == after["avengers_pro"]
    assert after["avengers_pro_now"] != after["avengers_pro"]
