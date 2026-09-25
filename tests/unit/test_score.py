import math

import pytest

from routebalance.schedule.score import EPS, Candidate, pick, score


def cand(iid, quality, cost, lat, unloaded=None, queue=0, last=0.0, model="M"):
    return Candidate(iid, model, quality, cost, lat, unloaded, queue, last)


W = dict(w_quality=0.5, w_cost=0.2, w_latency=0.3)


def test_scores_are_v3_bitwise_in_v3_summation_order():
    q, cost, lat = [0.8, 0.43, 0.45], [3.3e-05, 2.9e-05, 5.7e-05], [1.93, 2.8, 1.22]   # order-sensitive values
    out = score([cand(n, qi, ci, li) for n, qi, ci, li in zip("abc", q, cost, lat, strict=True)],
                normalization="max_divide", **W)
    reordered = 0
    for s, qi, ci, li in zip(out, q, cost, lat, strict=True):
        tq = 0.5 * ((qi - min(q)) / (max(q) - min(q) + EPS))
        tc = 0.2 * (1.0 - (ci / (max(cost) + EPS)))
        tl = 0.3 * (1.0 - (li / (max(lat) + EPS)))
        assert s.score == tq + tc + tl + 0.0 * 0.123          # v3 order, incl. its zero-weight fourth term; exact
        reordered += (tq + tl + tc != s.score) and (tl + tc + tq != s.score)
    assert reordered >= 2                                        # the fixture tells the orders apart


def test_max_divide_pins_the_slowest_candidate_and_fixed_ref_keeps_its_load():
    cs = [cand("fast", 0.5, 1e-5, 1.0, unloaded=1.0), cand("slow", 0.5, 1e-5, 4.0, unloaded=2.0)]
    md = score(cs, normalization="max_divide", **W)
    assert md[1].latency_term == pytest.approx(0.0, abs=1e-8)
    fr = score(cs, normalization="fixed_ref", **W)
    assert fr[0].latency_term == 1.0 and fr[1].latency_term == 0.25       # best unloaded 1.0 over 4.0


def test_equal_qualities_give_a_zero_quality_term():
    out = score([cand("a", 0.6, 1e-5, 1.0), cand("b", 0.6, 2e-5, 1.0)], normalization="max_divide", **W)
    assert all(s.quality_term == 0.0 for s in out)


def test_ties_go_to_the_shorter_queue_then_least_recent_then_fleet_order():
    same = dict(quality=0.5, cost=1e-5, lat=1.0)
    out = score([cand("a", queue=3, **same), cand("b", queue=1, last=9.0, **same), cand("c", queue=1, last=2.0, **same),
                 cand("d", queue=1, last=2.0, **same)], normalization="max_divide", **W)
    assert pick(out).candidate.instance_id == "c"


@pytest.mark.parametrize("bad", [
    dict(quality=math.nan), dict(cost=-1.0), dict(lat=0.0), dict(lat=math.inf),
])
def test_invalid_candidates_raise(bad):
    args = dict(quality=0.5, cost=1e-5, lat=1.0) | bad
    with pytest.raises(ValueError):
        score([cand("a", **args)], normalization="max_divide", **W)


def test_fixed_ref_needs_every_unloaded_latency():
    with pytest.raises(ValueError):
        score([cand("a", 0.5, 1e-5, 1.0, unloaded=1.0), cand("b", 0.5, 1e-5, 1.0)], normalization="fixed_ref", **W)


def test_no_candidates_or_unknown_normalization_raise():
    with pytest.raises(ValueError):
        score([], normalization="max_divide", **W)
    with pytest.raises(ValueError):
        score([cand("a", 0.5, 1e-5, 1.0)], normalization="min_max", **W)
