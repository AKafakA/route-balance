"""BEST-Route replay: for t in {0.0, 0.3, 0.5, 0.7} with fallback 3B, the rebuild's choice == the v3
router's == the banked lambda-6 cell, for all 3534 prompts; scores within 1e-6 and reasons identical.

The classifier is the expensive part and does not depend on t, so each side runs its own forward pass
once per prompt (a memo) and every threshold's decision logic runs in full on top of it.
"""
import asyncio
import collections
import functools

import pytest

from routebalance.baselines.pipeline import unwrap_chat_prompt
from routebalance.baselines.routers import BestRouteRouter, RouterRequest, build_router, load_best_route, router_executor
from routebalance.config import ConfigError, Router
from tests.equivalence.router_oracle import BANKED_N, md5
from tests.fleet_s2 import POOL

THRESHOLDS = {0.0: "br4_t00", 0.3: "br4_t03", 0.5: "br4_t05", 0.7: "br4_t07"}
FALLBACK = "Qwen/Qwen2.5-3B"
CHECKPOINT_MD5 = "01164d29c2db06b569897da2ee58b775"      # model.safetensors, May 2 build
BANKED_MIX = {0.0: (50.3, 21.9, 15.5, 12.3), 0.3: (53.8, 20.9, 13.3, 12.0), 0.5: (91.1, 0.2, 0.6, 8.1),
              0.7: (95.2, 0.0, 0.1, 4.7)}     # % 3B / 7B / 14B / 72B


class MemoModel:
    """v3 side: one forward per distinct token sequence, shared by the four v3 routers."""

    def __init__(self, model):
        self._model, self._cache = model, {}

    def __call__(self, **enc):
        key = enc["input_ids"].numpy().tobytes()
        if key not in self._cache:
            self._cache[key] = self._model(**enc)
        return self._cache[key]


@pytest.fixture(scope="module")
def banked(v3):
    rows = {t: v3.banked(cell) for t, cell in THRESHOLDS.items()}
    prompts = [r["prompt"] for r in rows[0.0]]
    for t, rs in rows.items():
        assert collections.Counter(r["prompt"] for r in rs) == collections.Counter(prompts), t
    return prompts, {t: {r["prompt"]: r["model"] for r in rs} for t, rs in rows.items()}


@pytest.fixture(scope="module")
def checkpoint(v3):
    ckpt = v3.artifacts / "best_route_4way_qwen"
    assert md5(ckpt / "model.safetensors") == CHECKPOINT_MD5
    return ckpt


def decide_all(routers, prompts, make_request):
    async def go():
        out = {}
        for t, router in routers.items():
            out[t] = [await router.choose_model(make_request(p), POOL) for p in prompts]
        return out
    return asyncio.run(go())


def test_unwrap_matches_v3_on_every_banked_prompt(v3, banked):
    v3_unwrap = v3.v3_unwrap()
    prompts, _ = banked
    assert [unwrap_chat_prompt(p) for p in prompts] == [v3_unwrap(p) for p in prompts]
    assert all(p.startswith("<|im_start|>system") for p in prompts)      # the banked prompts are templated


def test_best_route_replay(v3, banked, checkpoint):
    prompts, cells = banked
    v3_unwrap = v3.v3_unwrap()
    v3mod = v3.v3_package("routers")
    v3_routers = {t: v3mod("best_route_4way").BestRoute4WayRouter(checkpoint_path=str(checkpoint), confidence_threshold=t,
                                                                  fallback_model=FALLBACK) for t in THRESHOLDS}
    memo = MemoModel(v3_routers[0.0]._model)
    for r in v3_routers.values():
        r._model = memo
    v3_request = v3mod("base").RouterRequest
    theirs = decide_all(v3_routers, prompts, lambda p: v3_request(prompt=v3_unwrap(p)))

    classify, labels = load_best_route(checkpoint, 512)
    shared = functools.lru_cache(maxsize=None)(classify)
    ex = router_executor()
    ours = {t: BestRouteRouter(shared, labels, threshold=t, fallback=FALLBACK, executor=ex) for t in THRESHOLDS}
    got = decide_all(ours, prompts, lambda p: RouterRequest(unwrap_chat_prompt(p), ""))

    for t in THRESHOLDS:
        pairs = list(zip(prompts, theirs[t], got[t], strict=True))
        agree_v3 = sum(a.model_name == b.model for _, a, b in pairs)
        agree_banked = sum(cells[t][p] == b.model for p, _, b in pairs)
        reasons = sum(a.reason == b.reason for _, a, b in pairs)
        worst = max(abs(a.score - b.score) for _, a, b in pairs)
        mix = collections.Counter(b.model for _, _, b in pairs)
        pct = tuple(round(100 * mix[m] / BANKED_N, 1) for m in ("Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-7B",
                                                                  "Qwen/Qwen2.5-14B", "Qwen/Qwen2.5-72B"))
        print(f"BR t={t}: rebuild==v3 {agree_v3}/{BANKED_N}, rebuild==banked {agree_banked}/{BANKED_N}, "
              f"reasons {reasons}/{BANKED_N}, max |score diff| {worst:.2e}, mix {pct}")
        assert (agree_v3, agree_banked, reasons) == (BANKED_N, BANKED_N, BANKED_N) and worst <= 1e-6
        assert pct == BANKED_MIX[t]


def test_best_route_built_from_config_replays_the_banked_cell(v3, banked):
    prompts, cells = banked
    cfg = Router(type="best_route", kwargs={"checkpoint": "best_route_4way_qwen", "confidence_threshold": 0.5,
                                            "fallback_model": FALLBACK, "max_length": 512}, concurrency=1)
    router = build_router(cfg, artifact_root=v3.artifacts, avengers_repo=None, executor=router_executor(),
                          decoupled=None)
    sample = prompts[::18]
    got = decide_all({0.5: router}, sample, lambda p: RouterRequest(unwrap_chat_prompt(p), ""))[0.5]
    assert [d.model for d in got] == [cells[0.5][p] for p in sample]


def test_threshold_without_fallback_fails_at_construction_in_both(v3, checkpoint):
    v3mod = v3.v3_package("routers")
    with pytest.raises(RuntimeError):
        v3mod("best_route_4way").BestRoute4WayRouter(checkpoint_path=str(checkpoint), confidence_threshold=0.5)
    with pytest.raises(ConfigError):
        build_router(Router(type="best_route", kwargs={"checkpoint": "best_route_4way_qwen", "confidence_threshold": 0.5,
                                                       "max_length": 512}, concurrency=1),
                     artifact_root=v3.artifacts, avengers_repo=None, executor=router_executor(), decoupled=None)
