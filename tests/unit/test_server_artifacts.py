"""What the execution manifest records about a process: the artifact and model md5s a configuration loads,
the installed distributions, the native thread pools, and the decision log's on-disk accounting."""
import asyncio
import copy
import hashlib
import json
import sys
import types
from concurrent.futures import ThreadPoolExecutor

import aiohttp
import pytest

import routebalance.server as server
from routebalance.baselines.routers import PassthroughRouter
from routebalance.config import parse_scheduler
from routebalance.decision_log import JsonlLog
from routebalance.estimate.artifacts import ArtifactError, md5_tree
from routebalance.types import Instance, Tier

SMALL = Tier("small_a30", "S", "a30", 1e-7, 1e-7, 8)
FLEET = [Instance("S_0", SMALL, "n0", "127.0.0.1", 9000)]
ENCODER = "sentence-transformers/all-MiniLM-L6-v2"

BASE = {
    "mode": "routebalance", "weights": {"w_quality": 0.5, "w_latency": 0.3, "w_cost": 0.2},
    "rounds": {"adaptive": True}, "lpt": "max", "pending_decode": "own_length", "dead_reckoning": True,
    "telemetry_read": "per_round",
    "latency": {"source": "constant", "envelope": "own_length", "normalization": "max_divide",
                "table": "calib/t.json", "table_md5": "0" * 32},
    "estimator": {"index_dir": "knn/x", "index_md5": "1" * 32, "encoder_revision": "e" * 40, "k": 50,
                  "score_type": "deepeval", "prior": "knn"},
    "budget": {"admission_filter": False, "confidence": 0.5},
    "telemetry": {"poll_ms": 50.0, "max_age_ms": 500.0, "timeout_s": 2.0},
    "pipeline": None,
}


def md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


@pytest.fixture
def fake_knn(monkeypatch):
    """load_components imports the KNN module, which imports faiss (absent from the test host); its loader is
    replaced here (the md5 check it performs is tested with the real module). Model files are recorded by a
    stand-in that shows which model, revision and role were asked for."""
    mod = types.ModuleType("routebalance.estimate.knn")
    mod.INDEX_FILE = "knn_estimator.pkl"
    mod.load_knn = lambda cfg, root, models: types.SimpleNamespace(index=types.SimpleNamespace(encoder_name=ENCODER))
    monkeypatch.setitem(sys.modules, "routebalance.estimate.knn", mod)
    monkeypatch.setattr(server, "model_files", lambda role, name, rev: {f"{role}@{name}@{rev}": "9" * 32})


def write_table(root) -> str:
    data = json.dumps({"unit": "ms_per_token", "constant": {"small_a30": 10.0}, "provenance": {}}).encode()
    (root / "calib").mkdir()
    (root / "calib/t.json").write_bytes(data)
    return md5(data)


def load(cfg: dict, root):
    with ThreadPoolExecutor(1) as ex:
        return server.load_components(parse_scheduler(cfg), FLEET, artifact_root=root, avengers_repo=None, router_lane=ex)


def pipeline(router: dict, **kw) -> dict:
    cfg = copy.deepcopy(BASE)
    cfg.update(mode="pipeline", weights=None, pending_decode=None, dead_reckoning=False, telemetry_read="poll",
               pipeline={"router": router, "dispatch": {"type": "least_outstanding", "seed": 1}}, **kw)
    return cfg


def test_routebalance_records_the_index_its_encoder_and_the_table(tmp_path, fake_knn):
    cfg = copy.deepcopy(BASE)
    cfg["latency"]["table_md5"] = write_table(tmp_path)
    c = load(cfg, tmp_path)
    assert c.artifacts == {"knn/x/knn_estimator.pkl": "1" * 32, f"estimator@{ENCODER}@{'e' * 40}": "9" * 32,
                           "calib/t.json": cfg["latency"]["table_md5"]}


def test_a_passthrough_baseline_loads_no_artifact(tmp_path, fake_knn):
    assert load(pipeline({"type": "passthrough", "kwargs": {}}), tmp_path).artifacts == {}


def test_a_router_checkpoint_is_recorded_file_by_file(tmp_path, monkeypatch, fake_knn):
    monkeypatch.setattr(server, "build_router", lambda *a, **kw: PassthroughRouter())
    (tmp_path / "br/ckpt/tok").mkdir(parents=True)
    (tmp_path / "br/ckpt/config.json").write_bytes(b"{}")
    (tmp_path / "br/ckpt/tok/spm.model").write_bytes(b"\x00\x01")
    cfg = pipeline({"type": "best_route", "kwargs": {"checkpoint": "br/ckpt", "confidence_threshold": 0.0,
                                                     "max_length": 512}})
    assert load(cfg, tmp_path).artifacts == {"br/ckpt/config.json": md5(b"{}"), "br/ckpt/tok/spm.model": md5(b"\x00\x01")}


def test_avengers_pro_records_its_artifact_dir_and_its_embedder(tmp_path, monkeypatch, fake_knn):
    monkeypatch.setattr(server, "build_router", lambda *a, **kw: PassthroughRouter())
    (tmp_path / "ap").mkdir()
    (tmp_path / "ap/kmeans.joblib").write_bytes(b"km")
    cfg = pipeline({"type": "avengers_pro", "kwargs": {"artifact_dir": "ap", "beta": 5.0, "top_k": 4,
                                                       "embedder": ENCODER, "embedder_revision": "e" * 40}})
    # the embedder is hashed at its pinned revision (D49), never the cached main
    assert load(cfg, tmp_path).artifacts == {"ap/kmeans.joblib": md5(b"km"), f"avengers_pro@{ENCODER}@{'e' * 40}": "9" * 32}


def test_the_xgboost_source_records_every_file_it_read():
    cfg = copy.deepcopy(BASE)
    cfg["latency"].update(source="xgboost", envelope="queue_aware", table="xgb", table_md5="2" * 32)
    source = types.SimpleNamespace(files={"md5s.json": "2" * 32, "feature_names.json": "3" * 32,
                                          "small_a30.xgb": "4" * 32})
    got = server.latency_artifacts(parse_scheduler(cfg), types.SimpleNamespace(source=source))
    assert got == {"xgb/md5s.json": "2" * 32, "xgb/feature_names.json": "3" * 32, "xgb/small_a30.xgb": "4" * 32}


def test_model_files_hash_a_local_model_directory(tmp_path):
    (tmp_path / "enc/1_Pooling").mkdir(parents=True)
    (tmp_path / "enc/model.safetensors").write_bytes(b"w")
    (tmp_path / "enc/1_Pooling/config.json").write_bytes(b"{}")
    assert server.model_files("estimator", str(tmp_path / "enc"), None) == {
        "estimator@enc/1_Pooling/config.json": md5(b"{}"), "estimator@enc/model.safetensors": md5(b"w")}


def test_model_files_hash_the_cached_hub_snapshot_offline(tmp_path, monkeypatch):
    snap = tmp_path / "hub/models--org--enc/snapshots/c0ffee"
    snap.mkdir(parents=True)
    (snap / "model.safetensors").write_bytes(b"w")
    calls = []

    def snapshot_download(repo_id, *, revision, local_files_only, cache_dir):
        calls.append((repo_id, revision, local_files_only, cache_dir))
        return str(snap)
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=snapshot_download))
    monkeypatch.delenv("SENTENCE_TRANSFORMERS_HOME", raising=False)
    assert server.model_files("avengers_pro", "org/enc", None) == {"avengers_pro@hf:org/enc/model.safetensors": md5(b"w")}
    monkeypatch.setenv("SENTENCE_TRANSFORMERS_HOME", str(tmp_path / "st"))   # the loader's own cache wins
    server.model_files("estimator", "org/enc", "c0ffee")
    assert calls == [("org/enc", None, True, None),                          # the cached main, never the network
                     ("org/enc", "c0ffee", True, str(tmp_path / "st"))]


def test_md5_tree_refuses_a_missing_or_empty_directory(tmp_path):
    with pytest.raises(ArtifactError):
        md5_tree(tmp_path, "absent")
    (tmp_path / "empty").mkdir()
    with pytest.raises(ArtifactError):
        md5_tree(tmp_path, "empty")


def test_library_versions_list_every_installed_distribution():
    got = server.library_versions()
    assert got["python"] == sys.version and got["aiohttp"] == aiohttp.__version__
    assert {"pydantic", "aiohappyeyeballs", "yarl", "multidict"} <= set(got)       # dependencies of dependencies


def test_native_threadpools_name_libraries_without_host_paths():
    import numpy  # noqa: F401  (loads a BLAS)
    pools = server.native_threadpools()
    assert pools and all(isinstance(n, int) and n >= 1 for n in pools.values())
    assert not any(k.count("/") != 1 or k.startswith("/") for k in pools)       # "<api>:<dir>/<file>" only


def test_a_failed_append_stays_buffered(tmp_path):
    log = JsonlLog(tmp_path / "decisions.jsonl")
    log.path.mkdir()                                      # appending to a directory fails, as a full disk would

    async def scenario():
        log.write({"a": 1})
        with pytest.raises(OSError):
            await log.flush()
        return log.buffered, log.written
    assert asyncio.run(scenario()) == (1, 0)              # never 0 buffered while a line is missing from disk


def test_load_components_hands_every_pipeline_router_the_lane(tmp_path, monkeypatch, fake_knn):
    """build_router gets the one-at-a-time lane, not RouteBalance's shared pool (owner 09-25)."""
    got = []
    monkeypatch.setattr(server, "build_router", lambda *a, **kw: got.append(kw["executor"]) or PassthroughRouter())
    (tmp_path / "br/ckpt").mkdir(parents=True)
    (tmp_path / "br/ckpt/config.json").write_bytes(b"{}")
    cfg = pipeline({"type": "best_route", "kwargs": {"checkpoint": "br/ckpt", "confidence_threshold": 0.0,
                                                     "max_length": 512}})
    with ThreadPoolExecutor(1) as lane:
        server.load_components(parse_scheduler(cfg), FLEET, artifact_root=tmp_path, avengers_repo=None,
                               router_lane=lane)
    assert got == [lane]
