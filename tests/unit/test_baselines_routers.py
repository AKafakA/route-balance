"""Routers with injected fakes: v3 decision semantics, raise-instead-of-fallback, off-loop execution."""
import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from routebalance.baselines.routers import (ALL_INSTANCES, BEST_ROUTE_MODULES, AvengersProRouter, BestRouteRouter,
                                            BestRouteUpstreamRouter, DecoupledInputs, Estimate, PassthroughRouter,
                                            RBDecoupledRouter, RouterRequest, _check_recorded, _import_upstream,
                                            _names_the_model, _rankings, best_route_digest, build_router,
                                            load_best_route_upstream, price_order, upstream_digest, validate_options)
from routebalance.config import ConfigError, Router, Weights
from tests.fleet_s2 import POOL, TIERS

EX = ThreadPoolExecutor(max_workers=2, thread_name_prefix="router")
LABELS = {0: "Qwen/Qwen2.5-3B", 1: "Qwen/Qwen2.5-7B", 2: "Qwen/Qwen2.5-14B", 3: "Qwen/Qwen2.5-72B"}
REQ = RouterRequest("what is 2+2?", "r1")


def run(coro):
    return asyncio.run(coro)


def choose(router, pool=POOL, req=REQ):
    return run(router.choose_model(req, pool))


class FakeClassifier:
    def __init__(self, label, prob):
        self.result, self.threads, self.prompts = (label, prob), [], []

    def __call__(self, prompt):
        self.threads.append(threading.current_thread().name)
        self.prompts.append(prompt)
        return self.result


def best_route(label, prob, threshold=0.0, fallback="Qwen/Qwen2.5-3B", labels=LABELS):
    return BestRouteRouter(FakeClassifier(label, prob), labels, threshold=threshold, fallback=fallback, executor=EX)


# ---------------------------------------------------------------- passthrough

def test_passthrough_returns_the_all_instances_sentinel():
    d = choose(PassthroughRouter())
    assert (d.model, d.score, d.reason) == (ALL_INSTANCES, 0.25, "passthrough:no_routing")
    with pytest.raises(ValueError):
        choose(PassthroughRouter(), pool=[])


# ---------------------------------------------------------------- BEST-Route

def test_best_route_argmax():
    d = choose(best_route(2, 0.61))
    assert (d.model, d.score, d.reason) == ("Qwen/Qwen2.5-14B", 0.61, "best_route_4way:argmax_label=2:p=0.610")


def test_best_route_threshold_is_strict():
    assert choose(best_route(3, 0.5, threshold=0.5)).model == "Qwen/Qwen2.5-72B"      # p == t keeps the argmax
    d = choose(best_route(3, 0.4999, threshold=0.5))
    assert d.model == "Qwen/Qwen2.5-3B"
    assert d.reason == "best_route_4way:low_conf:p=0.500<0.5:fallback=Qwen/Qwen2.5-3B" and d.score == 0.4999


def test_best_route_threshold_zero_never_falls_back():
    assert choose(best_route(1, 0.26, threshold=0.0, fallback=None)).model == "Qwen/Qwen2.5-7B"


def test_best_route_pool_miss_precedes_the_threshold():
    pool = [m for m in POOL if m != "Qwen/Qwen2.5-72B"]
    d = choose(best_route(3, 0.99, threshold=0.7), pool=pool)
    assert d.model == "Qwen/Qwen2.5-3B" and d.reason == "best_route_4way:pool_miss:label=3:fallback=Qwen/Qwen2.5-3B"
    d = choose(best_route(3, 0.3, threshold=0.0), pool=pool)                        # pool miss even at t = 0
    assert d.reason.startswith("best_route_4way:pool_miss:label=3")


def test_best_route_unmapped_label_is_a_pool_miss():
    assert choose(best_route(7, 0.9)).reason == "best_route_4way:pool_miss:label=7:fallback=Qwen/Qwen2.5-3B"


def test_best_route_never_infers_a_fallback():
    pool = [m for m in POOL if m != "Qwen/Qwen2.5-72B"]
    with pytest.raises(RuntimeError, match="explicit fallback_model"):
        choose(best_route(3, 0.9, fallback=None), pool=pool)
    with pytest.raises(RuntimeError, match="not in the pool"):
        choose(best_route(1, 0.3, threshold=0.5, fallback="Qwen/Qwen2.5-1B"))
    with pytest.raises(ConfigError):
        best_route(1, 0.3, threshold=0.5, fallback=None)


def test_best_route_scores_off_the_event_loop():
    router = best_route(0, 0.9)
    choose(router)
    assert router._classify.prompts == [REQ.prompt]
    assert [t.startswith("router") for t in router._classify.threads] == [True]
    with pytest.raises(ValueError):
        choose(router, pool=[])


# ---------------------------------------------------------------- AvengersPro

def ap(result):
    calls = []

    def route_batch(prompts):
        calls.append((threading.current_thread().name, prompts))
        if isinstance(result, Exception):
            raise result
        return result
    return AvengersProRouter(route_batch, executor=EX), calls


def test_avengers_pro_takes_upstreams_first_pool_model_off_the_loop():
    router, calls = ap([["Qwen/Qwen2.5-72B"]])
    d = choose(router)
    assert (d.model, d.score, d.reason) == ("Qwen/Qwen2.5-72B", 1.0, "avengers_pro:upstream:rank=1:model=Qwen/Qwen2.5-72B")
    assert len(calls) == 1 and calls[0][1] == [REQ.prompt] and calls[0][0].startswith("router")


@pytest.mark.parametrize("result,match", [
    (RuntimeError("boom"), "upstream routing failed"),
    ([], "empty ranking"),
    ([[]], "empty ranking"),
    ([["Qwen/Qwen2.5-72B"]], "none of upstream"),
])
def test_avengers_pro_raises_instead_of_falling_back(result, match):
    router, _ = ap(result)
    pool = [m for m in POOL if m != "Qwen/Qwen2.5-72B"]
    with pytest.raises(RuntimeError, match=match):
        choose(router, pool=pool)


def test_avengers_pro_artifact_checks():
    _check_recorded({"top_k": 4}, "top_k", 4)
    _check_recorded({}, "beta", 5.0)
    with pytest.raises(ConfigError, match="top_k"):
        _check_recorded({"top_k": 4}, "top_k", 3)
    assert _rankings({"0": ["a", "b"], "1": {"ranking": ["b", "a"], "total": 3}}, ["a", "b"]) == {
        0: {"ranking": ["a", "b"]}, 1: {"ranking": ["b", "a"]}}
    for bad in ({}, {"0": {"scores": {}}}, {"0": "a"}, {"0": [1, 2]}, [], {"0": []}, {"0": {"ranking": []}},
                {"0": ["a", "c"]}):
        with pytest.raises(ConfigError):
            _rankings(bad, ["a", "b"])


# ---------------------------------------------------------------- RB-decoupled

TIER_LIST = [TIERS["14B"], TIERS["3B"], TIERS["72B"], TIERS["7B"]]
W = Weights(w_quality=0.5, w_latency=0.3, w_cost=0.2)


def decoupled(estimates, latency_per_token, normalization="fixed_ref", weights=W):
    seen = {}

    def estimate(prompts):
        seen["thread"], seen["prompts"] = threading.current_thread().name, list(prompts)
        return [estimates]

    def unloaded(tier, n):
        return latency_per_token[tier.model] * n
    inputs = DecoupledInputs(TIER_LIST, weights, normalization, estimate, unloaded)
    return RBDecoupledRouter(inputs, executor=EX), seen


EST = {"Qwen/Qwen2.5-14B": Estimate(0.70, 100.0), "Qwen/Qwen2.5-3B": Estimate(0.50, 100.0),
       "Qwen/Qwen2.5-72B": Estimate(0.80, 100.0), "Qwen/Qwen2.5-7B": Estimate(0.60, 100.0)}
TPOT = {"Qwen/Qwen2.5-14B": 0.02, "Qwen/Qwen2.5-3B": 0.01, "Qwen/Qwen2.5-72B": 0.04, "Qwen/Qwen2.5-7B": 0.015}


def reference(est, tpot, w, normalization, tiers=TIER_LIST):
    q = [est[t.model].quality for t in tiers]
    c = [est[t.model].expected_len * t.cost_out for t in tiers]
    lat = [tpot[t.model] * est[t.model].expected_len for t in tiers]
    out = {}
    for t, qi, ci, li in zip(tiers, q, c, lat, strict=True):
        ln = 1 - li / (max(lat) + 1e-8) if normalization == "max_divide" else min(lat) / li
        out[t.model] = w.w_quality * (qi - min(q)) / (max(q) - min(q) + 1e-8) + w.w_cost * (1 - ci / (max(c) + 1e-8)) \
            + w.w_latency * ln
    return out


@pytest.mark.parametrize("normalization", ["fixed_ref", "max_divide"])
@pytest.mark.parametrize("weights", [W, Weights(w_quality=0.8, w_latency=0.1, w_cost=0.1),
                                     Weights(w_quality=0.1, w_latency=0.1, w_cost=0.8),
                                     Weights(w_quality=0.1, w_latency=0.8, w_cost=0.1)])
def test_decoupled_is_the_argmax_of_the_objective_with_unloaded_latency(normalization, weights):
    router, seen = decoupled(EST, TPOT, normalization, weights)
    d = choose(router, req=RouterRequest("hello", "r"))
    ref = reference(EST, TPOT, weights, normalization)
    best = max(ref, key=ref.get)
    assert d.model == best and d.score == pytest.approx(ref[best], abs=1e-12)
    assert seen["thread"].startswith("router") and seen["prompts"] == ["hello"]


def test_decoupled_ties_go_to_the_first_tier_in_fleet_order():
    same = {m: Estimate(0.5, 10.0) for m in EST}
    flat = {m: 0.01 for m in TPOT}
    tiers = [TIERS["7B"], TIERS["3B"]]
    router = RBDecoupledRouter(DecoupledInputs(tiers, Weights(w_quality=1.0, w_latency=0.0, w_cost=0.0), "fixed_ref",
                                               lambda ps: [same], lambda t, n: flat[t.model] * n), executor=EX)
    assert choose(router).model == "Qwen/Qwen2.5-7B"


def test_decoupled_scores_only_pool_models():
    router, _ = decoupled(EST, TPOT, weights=Weights(w_quality=1.0, w_latency=0.0, w_cost=0.0))
    assert choose(router).model == "Qwen/Qwen2.5-72B"
    assert choose(router, pool=["Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-7B"]).model == "Qwen/Qwen2.5-7B"


def test_decoupled_raises_on_missing_or_bad_estimates():
    partial = {k: v for k, v in EST.items() if k != "Qwen/Qwen2.5-7B"}
    with pytest.raises(RuntimeError, match="nothing for"):
        choose(decoupled(partial, TPOT)[0])
    with pytest.raises(RuntimeError, match="finite"):
        choose(decoupled(EST | {"Qwen/Qwen2.5-7B": Estimate(float("nan"), 1.0)}, TPOT)[0])
    with pytest.raises(RuntimeError, match="latency > 0"):
        choose(decoupled(EST, TPOT | {"Qwen/Qwen2.5-3B": 0.0})[0])
    with pytest.raises(ValueError, match="no tier"):
        choose(decoupled(EST, TPOT)[0], pool=["Qwen/Qwen2.5-1B"])
    with pytest.raises(ConfigError):
        RBDecoupledRouter(DecoupledInputs([TIERS["3B"], TIERS["3B"]], W, "fixed_ref", None, None), executor=EX)


# ---------------------------------------------------------------- construction from config

def build(rtype, kwargs, **over):
    args = dict(artifact_root=Path("/nonexistent"), avengers_repo=None, executor=EX,
                decoupled=None) | over
    return build_router(Router(type=rtype, kwargs=kwargs), **args)


def test_build_passthrough_takes_no_kwargs():
    assert isinstance(build("passthrough", {}), PassthroughRouter)
    with pytest.raises(ConfigError):
        build("passthrough", {"seed": 1})


@pytest.mark.parametrize("kwargs,match", [
    ({"checkpoint": "br", "confidence_threshold": 0.5, "max_length": 512}, "fallback_model"),   # before any load
    ({"checkpoint": "br", "confidence_threshold": 0.0}, "max_length"),
    ({"checkpoint": "/abs/br", "confidence_threshold": 0.0, "max_length": 512}, "relative"),
    ({"checkpoint": "br", "confidence_threshold": 1.0, "max_length": 512}, "confidence_threshold"),
    ({"checkpoint": "br", "confidence_threshold": 0.0, "max_length": 512, "batch_scoring": True}, "batch_scoring"),
    ({"checkpoint": "br", "confidence_threshold": 0.0, "max_length": 512}, "no checkpoint"),
])
def test_build_best_route_validates_before_loading(kwargs, match):
    with pytest.raises(ConfigError, match=match):
        build("best_route", kwargs)


AP = {"artifact_dir": "ap", "beta": 5.0, "top_k": 4, "embedder": "m", "embedder_revision": "a" * 40,
      "upstream_sha256": "b" * 64}


@pytest.mark.parametrize("kwargs,match", [
    ({k: v for k, v in AP.items() if k != "embedder"}, "embedder"),
    ({k: v for k, v in AP.items() if k != "embedder_revision"}, "embedder_revision"),
    ({**AP, "embedder_revision": "main"}, "embedder_revision"),
    ({k: v for k, v in AP.items() if k != "upstream_sha256"}, "upstream_sha256"),
    ({**AP, "fallback_model": "x"}, "fallback_model"),
    ({**AP, "beta": 0.0}, "beta"),
    (AP, "upstream AvengersPro checkout"),
])
def test_build_avengers_pro_validates_before_loading(kwargs, match):
    with pytest.raises(ConfigError, match=match):
        build("avengers_pro", kwargs)


def test_build_avengers_pro_refuses_missing_artifacts(tmp_path):
    with pytest.raises(ConfigError, match="artifact files missing"):
        build("avengers_pro", AP, artifact_root=tmp_path, avengers_repo=tmp_path)


def test_the_upstream_checkout_must_be_the_pinned_one(tmp_path):
    for name in ("simple_cluster_router", "config", "embedding_cache"):
        (tmp_path / f"{name}.py").write_text(f"# {name}\n")
    digest = upstream_digest(tmp_path)
    with pytest.raises(ConfigError, match="the config pins"):
        _import_upstream(tmp_path, "0" * 64)
    (tmp_path / "config.py").write_text("# patched\n")
    assert upstream_digest(tmp_path) != digest
    (tmp_path / "embedding_cache.py").unlink()
    with pytest.raises(ConfigError, match="cannot read upstream module"):
        upstream_digest(tmp_path)


def test_router_options_validate_without_loading():
    validate_options(Router(type="avengers_pro", kwargs=AP))
    validate_options(Router(type="passthrough", kwargs={}))
    with pytest.raises(ConfigError, match="embedder_revision"):
        validate_options(Router(type="avengers_pro", kwargs={**AP, "embedder_revision": "x"}))
    with pytest.raises(ConfigError, match="max_length"):
        validate_options(Router(type="best_route", kwargs={"checkpoint": "br", "confidence_threshold": 0.0}))


def test_build_rb_decoupled_needs_its_inputs():
    with pytest.raises(ConfigError, match="estimator"):
        build("rb_decoupled", {})
    inputs = DecoupledInputs(TIER_LIST, W, "max_divide", lambda ps: [EST], lambda t, n: TPOT[t.model] * n)
    assert isinstance(build("rb_decoupled", {}, decoupled=inputs), RBDecoupledRouter)
    with pytest.raises(ConfigError):
        build("rb_decoupled", {"k": 1}, decoupled=inputs)


# ---------------------------------------------------------------- the router lane (owner 09-25)

def test_the_router_lane_runs_a_baselines_model_calls_one_at_a_time():
    """As shipped routers do (RouteLLM, NVIDIA's LLM Router): concurrent requests queue at the router."""
    from routebalance.baselines.routers import ROUTER_LANE_THREADS, router_lane
    state = {"now": 0, "peak": 0}
    lock = threading.Lock()

    def slow(prompt):
        with lock:
            state["now"] += 1
            state["peak"] = max(state["peak"], state["now"])
        time.sleep(0.02)
        with lock:
            state["now"] -= 1
        return 1, 0.9

    router = BestRouteRouter(slow, LABELS, threshold=0.0, fallback="Qwen/Qwen2.5-3B", executor=router_lane())

    async def burst():
        t0 = time.perf_counter()
        await asyncio.gather(*(router.choose_model(RouterRequest(f"q{i}", f"r{i}"), POOL) for i in range(6)))
        return time.perf_counter() - t0
    elapsed = asyncio.run(burst())
    assert ROUTER_LANE_THREADS == 1 and state["peak"] == 1 and elapsed >= 6 * 0.02


# ---------------------------------------------------------------- BEST-Route as its authors ship it (D79)

QWEN = ["Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-14B", "Qwen/Qwen2.5-72B"]


class FakeScores:
    def __init__(self, scores):
        self.scores, self.threads = scores, []

    def __call__(self, prompt):
        self.threads.append(threading.current_thread().name)
        return list(self.scores)


def upstream(scores, threshold, candidates=QWEN, by_price=QWEN):
    return BestRouteUpstreamRouter(FakeScores(scores), candidates, by_price, threshold=threshold, executor=EX)


@pytest.mark.parametrize("threshold,model", [(0.0, "Qwen/Qwen2.5-3B"), (0.3, "Qwen/Qwen2.5-14B"),
                                             (0.6, "Qwen/Qwen2.5-14B"), (0.7, "Qwen/Qwen2.5-72B"),
                                             (1.0, "Qwen/Qwen2.5-72B")])
def test_upstream_best_route_takes_the_cheapest_candidate_at_the_threshold(threshold, model):
    """upstream results_interpretation.ipynb: the lowest-cost candidate whose score clears t; the reference's is 1."""
    d = choose(upstream([0.2, -0.1, 0.6, 1.0], threshold))
    assert d.model == model and d.reason.startswith("best_route_upstream:") and d.reason.endswith(f":model={model}")


def test_the_cheapest_first_order_is_the_fleets_price_order_not_the_rankers():
    gemma = ["g-1b", "g-4b", "g-12b", "g-27b"]                    # the ranker's order: ascending size, reference last
    prices = {"g-1b": 1e-7, "g-4b": 6e-8, "g-12b": 1e-7, "g-27b": 1.2e-7}
    order = price_order(gemma, prices)
    assert order == ["g-4b", "g-1b", "g-12b", "g-27b"]            # D71: 1B and 12B tie, the smaller first
    r = upstream([0.5, 0.1, 0.4, 1.0], 0.3, candidates=gemma, by_price=order)
    assert choose(r, pool=gemma).model == "g-1b"                  # 4B (0.1) is cheaper but misses t = 0.3
    assert choose(upstream([0.5, 0.1, 0.4, 1.0], 0.0, candidates=gemma, by_price=order), pool=gemma).model == "g-4b"


def test_upstream_best_route_scores_off_the_loop_and_fails_loudly():
    r = upstream([0.2, -0.1, 0.6, 1.0], 0.0)
    choose(r)
    assert r._scores.threads and all(t.startswith("router") for t in r._scores.threads)
    with pytest.raises(RuntimeError, match="not in the pool"):
        choose(r, pool=QWEN[:3])
    with pytest.raises(RuntimeError, match="3 scores for 4 candidates"):
        choose(upstream([0.2, -0.1, 1.0], 0.0))
    with pytest.raises(RuntimeError, match="the reference's last"):
        choose(upstream([0.2, -0.1, 0.6, 0.9], 0.0))
    with pytest.raises(ConfigError, match="reordered"):
        upstream([0.0] * 4, 0.0, by_price=QWEN[:3] + ["Qwen/Qwen2.5-1B"])
    with pytest.raises(ConfigError, match="outside"):
        upstream([0.0] * 4, 1.5)
    with pytest.raises(ConfigError, match="no price"):
        price_order(QWEN, {m: 1.0 for m in QWEN[:3]})


BU = {"checkpoint": "br", "threshold": 0.3, "base_model": "microsoft/deberta-v3-small", "base_revision": "a" * 40,
      "upstream_sha256": "b" * 64}


@pytest.mark.parametrize("kwargs,match", [
    ({**BU, "threshold": 1.5}, "threshold"),
    ({**BU, "checkpoint": "/abs/br"}, "relative"),
    ({**BU, "base_revision": "main"}, "base_revision"),
    ({k: v for k, v in BU.items() if k != "upstream_sha256"}, "upstream_sha256"),
    ({**BU, "fallback_model": "x"}, "fallback_model"),
    (BU, "best-route-llm checkout"),
])
def test_build_upstream_best_route_validates_before_loading(kwargs, match):
    with pytest.raises(ConfigError, match=match):
        build("best_route_upstream", kwargs)


def test_the_best_route_checkout_must_be_the_pinned_one(tmp_path):
    for rel in BEST_ROUTE_MODULES:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(f"# {rel}\n")
    digest = best_route_digest(tmp_path)
    with pytest.raises(ConfigError, match="the config pins"):
        load_best_route_upstream(tmp_path / "ck", tmp_path, base_model="m", base_revision="a" * 40, upstream_sha256="c" * 64)
    with pytest.raises(ConfigError, match="has no config.bin"):
        load_best_route_upstream(tmp_path / "ck", tmp_path, base_model="m", base_revision="a" * 40, upstream_sha256=digest)
    (tmp_path / "hybrid_llm/pair_ranker/ranker.py").write_text("# changed\n")
    assert best_route_digest(tmp_path) != digest


@pytest.mark.parametrize("head,model,ok", [
    ("Qwen2.5-3B_bo1", "Qwen/Qwen2.5-3B", True), ("gemma-3-1b_bo1", "google/gemma-3-1b-it", True),
    ("Qwen2.5-7B_bo1", "Qwen/Qwen2.5-72B", False), ("Qwen2.5-3B_bo1", "Qwen/Qwen2.5-7B", False),
    ("gemma-3-1b_bo1", "google/gemma-3-12b-it", False), ("Qwen2.5-3B", "Qwen/Qwen2.5-3B", False),
])
def test_each_head_must_name_its_fleet_model(head, model, ok):
    """A labels file that maps a head to another model (e.g. 3B and 7B swapped) fails the load, never routes."""
    assert _names_the_model(head, model) is ok

