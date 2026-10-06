import json
from pathlib import Path

import pytest
import yaml

from routebalance.config import ConfigError
from vsrbench import config as vc
from vsrbench import constants as C
from vsrbench import testprofile

REPO = Path(__file__).resolve().parents[2]
FIX = Path(__file__).resolve().parent / "fixtures"


def qwen_models():
    return vc.lease_models(REPO / "configs/fleet_qwen.json", FIX / "hosts_lease_like.json",
                           REPO / "configs/engine_args_qwen.json")


def test_lease_models_follow_the_fleet():
    models = {m.name: m for m in qwen_models()}
    assert sorted(models) == sorted(vc.ladder_models("qwen"))
    m3 = models["Qwen/Qwen2.5-3B"]
    assert (m3.prompt_per_1m, m3.completion_per_1m, m3.context) == (pytest.approx(0.06), pytest.approx(0.06), 8192)
    assert [(b.name, b.host, b.port) for b in m3.backends] == [
        ("Qwen-2.5-3B_0", "192.0.2.2", 8000), ("Qwen-2.5-3B_1", "192.0.2.4", 8000), ("Qwen-2.5-3B_2", "192.0.2.6", 8000)]
    assert len(models["Qwen/Qwen2.5-7B"].backends) == 5 and len(models["Qwen/Qwen2.5-72B"].backends) == 2
    m72 = models["Qwen/Qwen2.5-72B"]
    assert (m72.prompt_per_1m, m72.completion_per_1m) == (pytest.approx(0.38), pytest.approx(0.40))


def test_ladders_are_neighbouring_pairs_smallest_first():
    assert vc.ladder_models("qwen") == ["Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-14B", "Qwen/Qwen2.5-72B"]
    assert vc.ladder_models("gemma") == ["google/gemma-3-1b-it", "google/gemma-3-4b-it", "google/gemma-3-12b-it",
                                         "google/gemma-3-27b-it"]
    b = vc.bindings("qwen")
    assert list(b) == list(C.POOLS)
    assert b["medium"] == [{"model": "Qwen/Qwen2.5-7B"}, {"model": "Qwen/Qwen2.5-14B"}]


def test_gemma_fleet_matches_its_ladder():
    fleet = json.loads((REPO / "configs/fleet_gemma.json").read_text())
    assert sorted({t["model"] for t in fleet["tiers"].values()}) == sorted(vc.ladder_models("gemma"))


def test_provider_document_carries_the_fleet_and_the_means():
    models = qwen_models()
    means = {m.name: 0.5 + 0.01 * i for i, m in enumerate(models)}
    doc = vc.provider_document(models, means, test=False)
    assert doc["listeners"] == [{"name": "http-8899", "address": "0.0.0.0", "port": 8899, "timeout": "21600s"}]
    assert doc["providers"]["defaults"]["model"] == models[0].name
    by = {m["name"]: m for m in doc["providers"]["models"]}
    assert by["Qwen/Qwen2.5-3B"]["pricing"] == {"prompt_per_1m": 0.06, "completion_per_1m": 0.06, "currency": "USD"}
    assert by["Qwen/Qwen2.5-3B"]["backend_refs"][0] == {"name": "Qwen-2.5-3B_0", "provider": "vllm",
                                                        "endpoint": "192.0.2.2:8000", "protocol": "http"}
    assert all(c["max_output_tokens"] == c["context_window_size"] == 8192 for c in doc["routing"]["modelCards"])
    recs = {r["model"]: r["metrics"]["score"] for r in doc["evaluation"]["records"]}
    assert recs == pytest.approx(means)
    text = yaml.safe_dump(doc)
    assert vc.test_tokens_in(text) == []
    with pytest.raises(ConfigError):
        vc.provider_document(models, {"Qwen/Qwen2.5-3B": 0.5}, test=False)


def test_test_profile_is_marked_and_refused_by_the_lease_check(tmp_path):
    (tmp_path / "hosts.json").write_text(json.dumps(testprofile.hosts()))
    models = vc.lease_models(FIX.parent / "e2e/fleet_test.json", tmp_path / "hosts.json",
                             FIX.parent / "e2e/engine_args_test.json")
    assert [(m.name, [b.name for b in m.backends]) for m in models] == [("m-small", ["s0", "s1"])]
    doc = vc.provider_document(models, {"m-small": 0.33}, test=True)
    text = yaml.safe_dump(doc)
    assert C.TEST_MARK in text and C.TEST_GATEWAY in text
    with pytest.raises(ConfigError, match="test profile"):
        vc.refuse_test_tokens(text, "x")


def test_patch_indices_changes_only_the_pools_quality_index():
    shipped = (FIX / "shipped_test_profile.yaml").read_text()
    patched, paths = vc.patch_indices(shipped)
    assert len(paths) == 3
    a, b = yaml.safe_load(shipped), yaml.safe_load(patched)
    for p in paths:
        node_a, node_b = a, b
        for k in p:
            node_a, node_b = node_a[k], node_b[k]
        assert node_a in C.SHIPPED_INDICES and node_b == C.INDEX_ID
    assert shipped.count("\n") == patched.count("\n")
    assert sum(x != y for x, y in zip(shipped.splitlines(), patched.splitlines(), strict=True)) == 3


def test_patch_indices_refuses_a_policy_it_does_not_know():
    shipped = (FIX / "shipped_test_profile.yaml").read_text()
    with pytest.raises(ConfigError):
        vc.patch_indices(shipped.replace("index: vllm-sr/general@1.0.0", "index: other/x@1", 1))


def test_dockers_utc_created_at_parses_on_python_310():
    from vsrbench.run_cell import _unix
    assert _unix("2026-09-30T13:56:48Z") == _unix("2026-09-30T14:56:48+01:00")
