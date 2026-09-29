"""D79: BEST-Route as its authors ship it. Our loader and router (routebalance/baselines/routers.py
load_best_route_upstream, BestRouteUpstreamRouter), on the scheduler's pinned stack, against upstream's own inference
run in the venv the ranker was trained with (the reference file under the artifact root: 300 validation prompts per
family, each head's softmax). The checkout (best-route-llm under the oracle root) and the backbone are the ones the
presets pin, and the ranker checkpoints are the lease bundle's (best_route/ under the artifact root).
"""
import asyncio
import json
import math
from pathlib import Path

import pytest

from routebalance.baselines.routers import (BestRouteUpstreamRouter, RouterRequest, _names_the_model,
                                            load_best_route_upstream, price_order, router_lane)

REPO = Path(__file__).resolve().parents[2]
TOLERANCE = 1e-5          # fp32 CPU against the training venv's fp32 CPU inference; measured 09-26: <= 9.2e-7
THRESHOLDS = (0.0, 0.3, 0.5, 0.7)
MARGIN = 10 * TOLERANCE   # no reference score this close to a threshold, so equal decisions are required
N_PROMPTS = 300


def _pinned() -> dict:
    return json.loads((REPO / "configs" / "presets.json").read_text())["parts"]["br.router"]["kwargs"]


@pytest.mark.parametrize("family", ["qwen", "gemma"])
def test_scores_equal_upstreams_own_inference_and_route_by_its_rule(oracle, artifacts, family):
    ref = json.loads((artifacts / "best_route_reference" / f"{family}.json").read_text())
    assert len(ref["prompts"]) == len(ref["probs"]) == N_PROMPTS
    assert not Path(ref["checkpoint"]).is_absolute() and ".." not in Path(ref["checkpoint"]).parts
    kw = _pinned()
    scores, candidates = load_best_route_upstream(artifacts / ref["checkpoint"], oracle.root / "best-route-llm",
                                                  base_model=kw["base_model"], base_revision=kw["base_revision"],
                                                  upstream_sha256=kw["upstream_sha256"])
    # the heads' models from the reference's own head names and the fleet, not from the loader's labels file
    fleet = json.loads((REPO / "configs" / f"fleet_{family}.json").read_text())["tiers"].values()
    models = sorted({t["model"] for t in fleet})
    expected = [next(m for m in models if _names_the_model(h, m)) for h in ref["candidate_models"]]
    assert candidates == expected
    worst, want_all = 0.0, []
    for prompt, heads in zip(ref["prompts"], ref["probs"], strict=True):
        want = [h[0] - h[1] for h in heads] + [1.0]
        got = scores(prompt)
        assert all(math.isfinite(x) for x in want + got), (want, got)
        worst = max(worst, max(abs(a - b) for a, b in zip(want, got, strict=True)))
        want_all.append(want)
    assert worst < TOLERANCE, worst
    closest = min(abs(x - t) for w in want_all for x in w[:-1] for t in THRESHOLDS)
    assert closest > MARGIN, f"a reference score lies within {closest} of a threshold: regenerate the reference"
    order = price_order(expected, {t["model"]: (t["cost_in"] + t["cost_out"]) / 2 for t in fleet})
    lane = router_lane(1)
    for t in THRESHOLDS:
        router = BestRouteUpstreamRouter(scores, candidates, order, threshold=t, executor=lane)

        async def route(router=router):
            return [await router.choose_model(RouterRequest(p, str(i)), candidates) for i, p in enumerate(ref["prompts"])]
        got = [d.model for d in asyncio.run(route())]
        # upstream's rule on the reference scores: the cheapest candidate whose score clears t
        by_model = [dict(zip(expected, w, strict=True)) for w in want_all]
        assert got == [next(m for m in order if s[m] >= t) for s in by_model], t
