import copy
import json
from pathlib import Path

from vsrbench import constants as C
from vsrbench.config import Backend, Model
from vsrbench.live import (cluster_name, config_problems, container_problems, envoy_problems, replay_problems,
                           status_problems)

FIX = Path(__file__).resolve().parent / "fixtures"
LIVE = json.loads((FIX / "live_test_profile.json").read_text())
MODELS = [Model("m-small", 0.06, 0.06, 4096, (Backend("s0", "host.docker.internal", 8611),)),
          Model("m-large", 0.4, 0.4, 4096, (Backend("l0", "host.docker.internal", 8611),))]
MEANS = {"m-small": 0.33, "m-large": 0.43}
BINDS = {"simple": [{"model": "m-small"}], "medium": [{"model": "m-small"}, {"model": "m-large"}],
         "reasoning": [{"model": "m-large"}]}


def test_cluster_names_are_the_clis():
    assert cluster_name("m-small") == "model_m_2dsmall_cluster"
    assert cluster_name("Qwen/Qwen2.5-3B") == "model_Qwen_2fQwen2_2e5_2d3B_cluster"


def test_the_recorded_test_stack_passes_its_checks():
    assert status_problems(LIVE["status"]) == []
    assert config_problems(LIVE["config"], MODELS, MEANS, BINDS, test=True) == []


def test_a_lease_check_refuses_the_test_profile():
    assert any("test profile" in p for p in config_problems(LIVE["config"], MODELS, MEANS, BINDS, test=False))


def test_changed_prices_means_or_pools_are_found():
    live = copy.deepcopy(LIVE["config"])
    live["providers"]["models"][0]["pricing"]["prompt_per_1m"] = 1.0
    live["evaluation"]["records"][0]["metrics"]["score"] = 0.9
    got = config_problems(live, MODELS, MEANS, {"simple": [{"model": "m-large"}]}, test=True)
    assert any("prices" in p for p in got) and any("quality records" in p for p in got) and any("pools" in p for p in got)


def test_envoy_timeouts_and_breakers_are_read_from_the_dump():
    models = [Model("m-small", 0.06, 0.06, 4096, (Backend("s0", "192.0.2.1", 8611),)),
              Model("m-large", 0.4, 0.4, 4096, (Backend("l0", "192.0.2.1", 8611),))]
    assert envoy_problems(LIVE["clusters"], LIVE["config_dump"], models) == []
    dump = json.loads(json.dumps(LIVE["config_dump"]).replace('"3600s"', '"1200s"').replace("200000", "1024"))
    got = envoy_problems(LIVE["clusters"], dump, models)
    assert any("message timeouts" in p for p in got) and any("circuit breakers" in p for p in got)
    wrong = [Model("m-small", 0.06, 0.06, 4096, (Backend("s0", "203.0.113.9", 8611),)), models[1]]
    assert any("endpoints" in p for p in envoy_problems(LIVE["clusters"], LIVE["config_dump"], wrong))


def test_containers_must_be_the_pinned_four():
    images = [C.IMAGES[k] for k in ("envoy", "router", "postgres", "redis")]
    good = [{"name": n, "image": ref, "repo_digests": [f"{ref.rsplit(':', 1)[0]}@{d}"], "running": True,
             "restart_count": 0} for n, (ref, d) in zip(C.CONTAINERS, images, strict=True)]
    assert container_problems(good) == []
    hub = copy.deepcopy(good)
    hub[2]["image"] = "docker.io/library/" + hub[2]["image"]
    assert container_problems(hub) == []
    bad = copy.deepcopy(good)
    bad[1]["repo_digests"] = ["ghcr.io/vllm-project/semantic-router/vllm-sr@sha256:" + "0" * 64]
    assert any("not a pinned image" in p for p in container_problems(bad))
    assert any("four" in p for p in container_problems(good[:3]))


def test_the_replay_store_must_hold_exactly_the_expected_records():
    empty = {"object": "router_replay.list", "count": 0, "total": 0, "data": []}
    assert replay_problems(empty, 0) == []
    assert replay_problems({**empty, "total": 3}, 0) and replay_problems({**empty, "total": 3}, 40)
    assert replay_problems({"error": "x"}, 0)


def test_the_whole_generated_configuration_must_be_live():
    import yaml
    from vsrbench.config import patch_indices
    from vsrbench.live import generated_problems
    text, _ = patch_indices((FIX / "shipped_test_profile.yaml").read_text())
    gen = yaml.safe_load(text)
    assert generated_problems(LIVE["config"], gen) == []
    moved = copy.deepcopy(gen)
    objective = moved["recipes"][0]["routing"]["decisions"][0]["algorithm"]["multi_factor"]["objective"]
    objective["priorities"][0]["tolerance"] = 0.5
    assert any("differ" in p for p in generated_problems(LIVE["config"], moved))
    extra = copy.deepcopy(LIVE["config"])
    extra["providers"]["models"].append(extra["providers"]["models"][0])
    assert any("another length" in p for p in generated_problems(extra, gen))


def test_cards_follow_the_engines_context():
    from vsrbench.live import card_problems
    assert card_problems(LIVE["config"], {"m-small": 4096, "m-large": 4096}) == []
    assert card_problems(LIVE["config"], {"m-small": 8192, "m-large": 4096})


def test_a_fresh_store_is_empty_and_ends_with_the_runs_records():
    from vsrbench.live import store_problems
    fresh = {"postgres_rows": {C.REPLAY_TABLE: 0}, "redis_keys": 0}
    assert store_problems(fresh, 0, at_start=True) == []
    assert store_problems({"postgres_rows": {C.REPLAY_TABLE: 2}, "redis_keys": 0}, 0, at_start=True)
    assert store_problems({"postgres_rows": {C.REPLAY_TABLE: 0}, "redis_keys": 5}, 0, at_start=True)
    assert store_problems({"postgres_rows": {C.REPLAY_TABLE: 40}, "redis_keys": 5}, 40, at_start=False) == []
    assert store_problems({"postgres_rows": {C.REPLAY_TABLE: 40, "public.other": 0}, "redis_keys": 0}, 40,
                          at_start=False)


def test_a_container_given_a_gpu_is_found():
    images = [C.IMAGES[k] for k in ("envoy", "router", "postgres", "redis")]
    good = [{"name": n, "image": ref, "repo_digests": [f"{ref.rsplit(':', 1)[0]}@{d}"], "running": True,
             "restart_count": 0, "resources": {"DeviceRequests": None}}
            for n, (ref, d) in zip(C.CONTAINERS, images, strict=True)]
    assert container_problems(good) == []
    good[1]["resources"]["DeviceRequests"] = [{"Driver": "nvidia", "Count": -1}]
    assert any("devices" in p for p in container_problems(good))
    good[1]["resources"]["DeviceRequests"] = None
    good[1]["env"] = {"PATH": "/bin", "SR_LOG_LEVEL": "debug"}
    assert any("host variables" in p for p in container_problems(good))
