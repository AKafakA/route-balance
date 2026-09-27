"""AvengersPro replay: for each Qwen pw artifact, the rebuild's choice == the v3 wrapper's == the banked
lambda-6 cell for all 3534 prompts, with upstream routing at beta 5.0, top_k from the artifact, one model.

The embedding does not depend on pw, so each side embeds each prompt once with its own embedder (a memo
shared by that side's five routers); every route still runs upstream's full routing code.
"""
import asyncio
import collections
import json

import pytest

from routebalance.baselines.pipeline import unwrap_chat_prompt
from routebalance.baselines.routers import AvengersProRouter, RouterRequest, build_router, load_avengers_pro, router_executor
from routebalance.config import ConfigError, Router
from tests.equivalence.oracle_files import ENCODER_REVISION, UPSTREAM_SHA256
from tests.equivalence.router_oracle import BANKED_N, md5
from tests.fleet_s2 import POOL

EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"
PINS = {"embedder_revision": ENCODER_REVISION, "upstream_sha256": UPSTREAM_SHA256,   # the campaign presets' pins
        "max_tokens": 7500, "max_workers": 4}
KMEANS_MD5 = "ea9dc3b9f597e63fd09f4976c17f09b3"
ARTIFACTS = {   # pw -> (banked cell, cluster_rankings.json md5)
    "0.25": ("ap_pw025", "aa4c467259b86d408c4dded03f85c44e"),
    "0.39": ("ap_pw039", "44e9fa43d9668ea75686d4ee98d0d099"),
    "0.53": ("ap_pw053", "e5e4b9879a726ab82be4da1e07ee9caf"),
    "0.7": ("ap_pw07", "2b86ce2aab1c26383382caefaec84a27"),
    "0.8": ("ap_pw08", "56638f312805597052b28ca348af265d"),
}


def artifact(v3, pw):
    return v3.artifacts / "avengers_4model" / f"avengers_pro_qwen_pw{pw}"


class MemoEmbedder:
    def __init__(self, embedder):
        self._embedder, self._cache = embedder, {}

    def get(self, text):
        if text not in self._cache:
            self._cache[text] = self._embedder.get(text)
        return self._cache[text]


@pytest.fixture(scope="module")
def memos():
    return {}


def shared_embedder(memos, side, embedder):
    if side not in memos:
        memos[side] = MemoEmbedder(embedder)
    return memos[side]


def decide_all(router, prompts, make_request):
    async def go():
        return [await router.choose_model(make_request(p), POOL) for p in prompts]
    return asyncio.run(go())


@pytest.fixture(scope="module")
def v3_ap(v3):
    mod = v3.v3_package("routers")("avengers_pro")
    mod._AVENGERS_REPO = str(v3.upstream)     # v3 reads the checkout path from this module global
    return mod


@pytest.mark.parametrize("pw", list(ARTIFACTS))
def test_avengers_pro_replay(v3, v3_ap, memos, pw):
    cell, rankings_md5 = ARTIFACTS[pw]
    art = artifact(v3, pw)
    assert (md5(art / "kmeans.joblib"), md5(art / "cluster_rankings.json")) == (KMEANS_MD5, rankings_md5)
    rows = v3.banked(cell)
    prompts, banked = [r["prompt"] for r in rows], [r["model"] for r in rows]

    v3_router = v3_ap.AvengersProRouter(artifact_dir=str(art), upstream_repo=str(v3.upstream))
    assert v3_router._loaded
    v3_router._upstream.embedder = shared_embedder(memos, "v3", v3_router._upstream.embedder)
    v3_request = v3.v3_package("routers")("base").RouterRequest
    v3_unwrap = v3.v3_unwrap()
    theirs = decide_all(v3_router, prompts, lambda p: v3_request(prompt=v3_unwrap(p)))

    recorded_top_k = json.loads((art / "config.json").read_text())["top_k"]
    up = load_avengers_pro(art, v3.upstream, beta=5.0, top_k=recorded_top_k, embedder=EMBEDDER, **PINS)
    up.embedder = shared_embedder(memos, "rebuild", up.embedder)
    ours_cfg, v3_cfg = up.config, v3_router._upstream.config
    assert (ours_cfg.beta, ours_cfg.top_k, ours_cfg.max_router) == (v3_cfg.beta, v3_cfg.top_k, v3_cfg.max_router) \
        == (5.0, 4, 1)
    assert (ours_cfg.max_workers, ours_cfg.max_tokens) == (v3_cfg.max_workers, v3_cfg.max_tokens)
    assert up.available_models == v3_router._upstream.available_models
    got = decide_all(AvengersProRouter(up.route_queries_batch, executor=router_executor()), prompts,
                     lambda p: RouterRequest(unwrap_chat_prompt(p), ""))

    agree_v3 = sum(a.model_name == b.model for a, b in zip(theirs, got, strict=True))
    agree_banked = sum(m == b.model for m, b in zip(banked, got, strict=True))
    reasons = sum(a.reason == b.reason for a, b in zip(theirs, got, strict=True))
    mix = collections.Counter(b.model for b in got)
    print(f"AP pw{pw}: rebuild==v3 {agree_v3}/{BANKED_N}, rebuild==banked {agree_banked}/{BANKED_N}, "
          f"reasons {reasons}/{BANKED_N}, mix {dict(sorted(mix.items()))}")
    assert (agree_v3, agree_banked, reasons) == (BANKED_N, BANKED_N, BANKED_N)


def test_avengers_pro_built_from_config(v3):
    rows = v3.banked("ap_pw07")[::25]
    cfg = Router(type="avengers_pro", kwargs={"artifact_dir": "avengers_4model/avengers_pro_qwen_pw0.7", "beta": 5.0,
                                              "top_k": 4, "embedder": EMBEDDER, **PINS})
    router = build_router(cfg, artifact_root=v3.artifacts, avengers_repo=v3.upstream,
                          executor=router_executor(), decoupled=None)
    got = decide_all(router, [r["prompt"] for r in rows], lambda p: RouterRequest(unwrap_chat_prompt(p), ""))
    assert [d.model for d in got] == [r["model"] for r in rows]


@pytest.mark.parametrize("over,match", [({"top_k": 3}, "top_k"), ({"embedder": "other/model"}, "embedding_model")])
def test_config_that_contradicts_the_artifact_is_refused(v3, over, match):
    kw = {"beta": 5.0, "top_k": 4, "embedder": EMBEDDER, **PINS} | over
    with pytest.raises(ConfigError, match=match):
        load_avengers_pro(artifact(v3, "0.7"), v3.upstream, **kw)
