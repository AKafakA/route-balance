"""Equal-quality deltas (D80, D81): one treatment configuration on the quality-latency frontier for every
metric, the tolerance, the range over the seeds, and the request bootstrap within the paired seeds."""
import numpy as np
import pytest

from analysis.equal_quality import equal_quality, value_at
from analysis.frontier import CellPoint
from analysis.metrics import RequestValues
from analysis.rules import EqualQuality
from routebalance.config import ConfigError
from tests.unit.test_frontier import pt

RULES = EqualQuality(tolerance=0.03, bootstrap_resamples=500, confidence=0.9, bootstrap_seed=0)
FRONTIER = [(0.4, 1000.0), (0.5, 2000.0)]


@pytest.mark.parametrize("q,expected", [
    (0.425, 1250.0), (0.4, 1000.0), (0.5, 2000.0),        # inside, and on the end points
    (0.38, 1000.0), (0.52, 2000.0),                        # past an end, within the tolerance
    (0.36, None), (0.54, None),                            # past an end, beyond it: never extrapolated
])
def test_value_at(q, expected):
    got = value_at(q, FRONTIER, 0.03)
    assert got is None if expected is None else got == pytest.approx(expected)


def test_value_at_degenerate_frontiers():
    assert value_at(0.4, [], 0.03) is None
    assert value_at(0.41, [(0.4, 7.0)], 0.03) == 7.0 and value_at(0.5, [(0.4, 7.0)], 0.03) is None


def _grid(seeds=(1,), shift=lambda s: 0.0):
    """Treatment presets rb-a (0.40, 1000 ms), rb-b (0.50, 2000 ms), rb-c (0.45, 3000 ms, dominated) and a
    baseline at (0.425, 2500 ms); shift(seed) is added to every configuration's latency on that seed."""
    out = []
    for s in seeds:
        d = shift(s)
        out += [pt("rb-a", "RB", s, 0.40, 1000 + d), pt("rb-b", "RB", s, 0.50, 2000 + d),
                pt("rb-c", "RB", s, 0.45, 3000 + d), pt("ap-1", "AP", s, 0.425, 2500 + d)]
    return out


def test_one_seed_gives_the_interpolated_delta_without_a_ci():
    [row] = equal_quality(_grid(), "RB", RULES)
    mean = row.deltas[0]
    assert (row.group, row.trace_id, row.config_id, row.system, row.seeds) == ("g1", "l30", "ap-1", "AP", (1,))
    assert mean.metric == "e2e_mean_ms" and mean.treatment == pytest.approx(1250.0)
    assert mean.delta == pytest.approx(-1250.0) and mean.relative == pytest.approx(-0.5)
    assert mean.ci is None and mean.undefined_resamples == 0 and mean.range is None
    assert row.frontier_presets == ("rb-a", "rb-b") and row.weight == pytest.approx(0.25)
    assert row.matched_presets == ("rb-a", "rb-c") and row.excluded_presets == ()
    assert [d.metric for d in row.deltas] == ["e2e_mean_ms", "e2e_p99_ms", "cost_decode_per_1k", "slo_miss_tight",
                                             "slo_miss_loose"]
    assert row.deltas[2].delta == pytest.approx(-1.25)


def test_a_baseline_outside_the_treatment_range_has_no_delta():
    pts = _grid()[:3] + [pt("ap-1", "AP", 1, 0.7, 2500)]
    [row] = equal_quality(pts, "RB", RULES)
    assert all(d.treatment is None and d.delta is None and d.relative is None for d in row.deltas)


def test_presets_without_the_baselines_seeds_are_excluded():
    pts = _grid(seeds=(1, 2)) + [pt("rb-x", "RB", 1, 0.42, 10.0)]
    [row] = equal_quality(pts, "RB", RULES)
    assert row.excluded_presets == ("rb-x",) and row.deltas[0].treatment == pytest.approx(1250.0)


def test_a_baseline_is_compared_only_with_its_own_groups_presets():
    pts = _grid() + [pt("ap-1", "AP", 1, 0.425, 2500, group="g2"), pt("rb-a", "RB", 1, 0.40, 8000, group="g2"),
                     pt("rb-b", "RB", 1, 0.50, 10000, group="g2")]
    rows = {r.group: r.deltas[0] for r in equal_quality(pts, "RB", RULES)}
    assert rows["g1"].treatment == pytest.approx(1250.0) and rows["g2"].treatment == pytest.approx(8500.0)
    with pytest.raises(ConfigError, match="no RB preset at g3/l30"):
        equal_quality(_grid()[:3] + [pt("ap-1", "AP", 1, 0.425, 2500, group="g3")], "RB", RULES)


def test_no_eligible_preset_is_an_error():
    pts = [pt("rb-a", "RB", 1, 0.4, 1000.0), pt("ap-1", "AP", 2, 0.4, 1000.0)]
    with pytest.raises(ConfigError, match="no RB preset"):
        equal_quality(pts, "RB", RULES)


def test_every_metric_is_read_at_the_one_configuration_on_the_latency_frontier():
    # rb-c is off the latency frontier but has the best p99: the p99 is still read between rb-a and rb-b (D80)
    pts = [CellPoint(group="g1", config_id=c, system=sys_, knob=c, dispatch="lor", trace_id="l30", seed=1, quality=q,
                     e2e_mean_ms=m, e2e_p99_ms=p99, cost_decode_per_1k=cost, slo_miss_tight=0.1, slo_miss_loose=0.05,
                     throughput_rps=20.0)
           for c, sys_, q, m, p99, cost in (("rb-a", "RB", 0.40, 1000, 5000, 1.0), ("rb-b", "RB", 0.50, 2000, 3000, 3.0),
                                            ("rb-c", "RB", 0.45, 3000, 1000, 0.1), ("ap-1", "AP", 0.425, 2500, 4000, 2.0))]
    [row] = equal_quality(pts, "RB", RULES)
    by = {d.metric: d for d in row.deltas}
    assert row.frontier_presets == ("rb-a", "rb-b") and row.weight == pytest.approx(0.25)
    assert by["e2e_p99_ms"].treatment == pytest.approx(4500.0)                    # 5000 + 0.25 * (3000 - 5000)
    assert by["cost_decode_per_1k"].treatment == pytest.approx(1.5)                # not rb-c's 0.1
    assert by["slo_miss_tight"].treatment == pytest.approx(0.1) and by["slo_miss_tight"].relative == pytest.approx(0)


def test_the_range_is_over_the_one_seed_comparisons():
    # the baseline alone is slower on seed 3: seeds 1-2 give -1250 ms (-50%), seed 3 gives 1250 - 3100
    pts = _grid(seeds=(1, 2, 3))
    pts[-1] = pt("ap-1", "AP", 3, 0.425, 3100)
    mean = equal_quality(pts, "RB", RULES)[0].deltas[0]
    assert [(v.seed, v.delta) for v in mean.per_seed] == [(1, pytest.approx(-1250.0)), (2, pytest.approx(-1250.0)),
                                                          (3, pytest.approx(-1850.0))]
    assert mean.range == pytest.approx((-1850.0, -1250.0))
    assert mean.range_relative == pytest.approx((1250 / 3100 - 1, -0.5))
    assert mean.delta == pytest.approx(1250.0 - 2700.0) and not mean.outside_range
    assert mean.ci is None                                            # no per-request values given


def test_a_value_at_the_means_outside_its_range_is_flagged():
    # a kinked frontier: each seed's baseline sits on a preset (0 change), their mean sits by the kink
    pts = [p for s in (1, 2) for p in (pt("rb-a", "RB", s, 0.40, 1000), pt("rb-b", "RB", s, 0.45, 1100),
                                       pt("rb-c", "RB", s, 0.50, 3000))]
    pts += [pt("ap-1", "AP", 1, 0.40, 1000), pt("ap-1", "AP", 2, 0.50, 3000)]
    mean = equal_quality(pts, "RB", RULES)[0].deltas[0]
    assert mean.range_relative == pytest.approx((0.0, 0.0)) and mean.relative == pytest.approx(1100 / 2000 - 1)
    assert mean.outside_range


def test_a_seed_without_an_equal_quality_point_leaves_no_range():
    pts = _grid(seeds=(1, 2))
    pts[-1] = pt("ap-1", "AP", 2, 0.9, 2500)
    mean = equal_quality(pts, "RB", RULES)[0].deltas[0]
    assert mean.per_seed[1].delta is None and mean.range is None and mean.range_relative is None


# ---------------------------------------------------------------- the request bootstrap (D80, the appendix)

N = 40


def _cell(config, system, seed, scores, e2e_s, cost=None, prompts=None):
    """A cell point and its per-request values, consistent with each other (quality, mean and p99 E2E, cost)."""
    scores, e2e_s = np.asarray(scores, dtype=float), np.asarray(e2e_s, dtype=float)
    cost = np.full(len(e2e_s), 1e-5) if cost is None else np.asarray(cost, dtype=float)
    met = np.ones(len(e2e_s))
    v = RequestValues(index=tuple(range(len(e2e_s))),
                      prompt_ids=tuple(prompts or (f"p{seed}-{i}" for i in range(len(e2e_s)))),
                      score=scores, e2e_s=e2e_s, cost=cost, met_tight=met, met_loose=met)
    point = {k: float(x[0]) for k, x in v.at(np.arange(len(e2e_s))[None, :]).items()}
    return (CellPoint(group="g1", config_id=config, system=system, knob=config, dispatch="lor", trace_id="l30",
                      seed=seed, throughput_rps=20.0, **point), (("g1", "l30", config, seed), v))


def _run(cells, rules=RULES, reading="one_config"):
    pts = [c[0] for c in cells]
    return equal_quality(pts, "RB", rules, dict(c[1] for c in cells), reading=reading)


def _latency(seed):
    return np.random.default_rng(seed).uniform(1.0, 3.0, N)


def test_the_request_bootstrap_draws_the_same_requests_for_every_system():
    # the treatment is 0.5 s faster on every request: any shared draw gives exactly -500 ms
    cells = [c for s in (1, 2) for c in (_cell("rb-a", "RB", s, [0.5] * N, _latency(s) - 0.5),
                                         _cell("ap-1", "AP", s, [0.5] * N, _latency(s)))]
    mean = _run(cells)[0].deltas[0]
    assert mean.delta == pytest.approx(-500.0) and mean.ci == pytest.approx((-500.0, -500.0))
    assert mean.undefined_resamples == 0 and mean.range == pytest.approx((-500.0, -500.0))


def test_the_request_bootstrap_is_deterministic_and_covers_request_noise():
    rng = np.random.default_rng(7)
    cells = [_cell("rb-a", "RB", 1, [0.5] * N, _latency(1) * 0.8 + rng.uniform(-0.3, 0.3, N)),
             _cell("ap-1", "AP", 1, [0.5] * N, _latency(1))]
    rows = [_run(cells)[0].deltas[0] for _ in range(2)]
    assert rows[0] == rows[1]
    lo, hi = rows[0].ci
    assert lo < rows[0].delta < hi and hi - lo > 0 and rows[0].range is None          # one seed: no range


def test_the_relative_ci_resamples_the_ratio_not_the_delta():
    # 20% faster on every request while requests vary 1-3 s: the ratio never moves, the delta does
    cells = [_cell("rb-a", "RB", 1, [0.5] * N, 0.8 * _latency(1)), _cell("ap-1", "AP", 1, [0.5] * N, _latency(1))]
    mean = _run(cells)[0].deltas[0]
    assert mean.relative == pytest.approx(-0.2) and mean.ci_relative == pytest.approx((-0.2, -0.2))
    assert mean.ci[1] - mean.ci[0] > 50


def test_resamples_without_an_equal_quality_point_void_the_ci():
    # the baseline's scores spread 0.3-0.7 around the preset's 0.5: resamples far from 0.5 have no point
    scores = [0.3, 0.7] * (N // 2)
    cells = [_cell("rb-a", "RB", 1, [0.5] * N, _latency(1)), _cell("ap-1", "AP", 1, scores, _latency(1))]
    mean = _run(cells)[0].deltas[0]
    assert mean.delta is not None and mean.ci is None and mean.undefined_resamples > 0


def test_cells_that_do_not_hold_the_same_requests_are_refused():
    cells = [_cell("rb-a", "RB", 1, [0.5] * N, _latency(1)),
             _cell("ap-1", "AP", 1, [0.5] * N, _latency(1), prompts=[f"q{i}" for i in range(N)])]
    with pytest.raises(ConfigError, match="do not hold the same requests"):
        _run(cells)
    cells = [_cell("rb-a", "RB", 1, [0.5] * N, _latency(1)), _cell("ap-1", "AP", 1, [0.5] * (N - 1), _latency(1)[1:])]
    with pytest.raises(ConfigError, match="do not hold the same requests"):
        _run(cells)


def test_a_compared_cell_without_request_values_is_an_error():
    cells = [_cell("rb-a", "RB", 1, [0.5] * N, _latency(1)), _cell("ap-1", "AP", 1, [0.5] * N, _latency(1))]
    with pytest.raises(ConfigError, match="no per-request values for"):
        equal_quality([c[0] for c in cells], "RB", RULES, dict([cells[0][1]]))


def test_corpus_gaps_are_left_out_of_the_resampled_quality():
    scores = [0.5, np.nan] * (N // 2)
    cells = [_cell("rb-a", "RB", 1, [0.5] * N, 0.9 * _latency(1)), _cell("ap-1", "AP", 1, scores, _latency(1))]
    [row] = _run(cells)
    assert row.quality == pytest.approx(0.5) and row.deltas[0].ci is not None


def test_a_resample_where_a_preset_drew_only_corpus_gaps_has_no_point():
    # reviewers' case: rb-m (1 scored request of 8) sits between rb-a and rb-b; a draw of only its gaps must void
    # that resample, not drop rb-m from the frontier and interpolate rb-a to rb-b
    n, nan = 8, float("nan")
    cells = [_cell("rb-a", "RB", 1, [0.4] * n, [1.0] * n), _cell("rb-m", "RB", 1, [0.5] + [nan] * (n - 1), [1.2] * n),
             _cell("rb-b", "RB", 1, [0.6] * n, [3.0] * n), _cell("ap-1", "AP", 1, [0.5] * n, [3.0] * n)]
    mean = _run(cells)[0].deltas[0]
    assert mean.delta == pytest.approx(-1800.0)                       # at the means: rb-m itself
    assert mean.undefined_resamples > 0 and mean.ci is None and mean.ci_relative is None


def test_a_resample_where_the_baseline_drew_only_corpus_gaps_has_no_point():
    n, nan = 8, float("nan")
    cells = [_cell("rb-a", "RB", 1, [0.5] * n, [1.0] * n), _cell("ap-1", "AP", 1, [0.5] + [nan] * (n - 1), [2.0] * n)]
    mean = _run(cells)[0].deltas[0]
    assert mean.delta == pytest.approx(-1000.0) and mean.undefined_resamples > 0 and mean.ci is None


def test_a_quality_that_is_not_a_number_is_refused():
    from analysis.equal_quality import position_at
    with pytest.raises(ConfigError, match="not a number"):
        position_at(float("nan"), [0.4, 0.5], 0.03)


def _kinked():
    """rb-c is off the latency frontier but has the best p99 and cost: the readings differ exactly there."""
    return [CellPoint(group="g1", config_id=c, system=sys_, knob=c, dispatch="lor", trace_id="l30", seed=1, quality=q,
                      e2e_mean_ms=m, e2e_p99_ms=p99, cost_decode_per_1k=cost, slo_miss_tight=0.1, slo_miss_loose=0.05,
                      throughput_rps=20.0)
            for c, sys_, q, m, p99, cost in (("rb-a", "RB", 0.40, 1000, 5000, 1.0), ("rb-b", "RB", 0.50, 2000, 3000, 3.0),
                                             ("rb-c", "RB", 0.45, 3000, 1000, 0.1), ("ap-1", "AP", 0.425, 2500, 4000, 2.0))]


def test_the_per_metric_reading_reads_each_metric_on_its_own_frontier():
    # v2/v3's reading (D86): p99's frontier is rb-c, rb-b; the baseline's 0.425 is within the tolerance of rb-c's 0.45
    [one] = equal_quality(_kinked(), "RB", RULES, reading="one_config")
    [per] = equal_quality(_kinked(), "RB", RULES, reading="per_metric")
    one_by, per_by = ({d.metric: d for d in r.deltas} for r in (one, per))
    assert per_by["e2e_mean_ms"].treatment == pytest.approx(1250.0)
    assert one_by["e2e_mean_ms"].treatment == pytest.approx(1250.0)
    assert per_by["e2e_p99_ms"].treatment == pytest.approx(1000.0) and one_by["e2e_p99_ms"].treatment == pytest.approx(4500)
    assert per_by["cost_decode_per_1k"].treatment == pytest.approx(0.1)
    assert one_by["cost_decode_per_1k"].treatment == pytest.approx(1.5)
    assert per_by["e2e_p99_ms"].frontier_presets == ("rb-c", "rb-c") and per_by["e2e_p99_ms"].weight == 0.0
    assert one_by["e2e_p99_ms"].frontier_presets == ("rb-a", "rb-b") and one_by["e2e_p99_ms"].weight == pytest.approx(0.25)
    assert per.frontier_presets == one.frontier_presets == ("rb-a", "rb-b")      # the row: the latency frontier


def test_the_per_metric_reading_bootstraps_each_metric_on_its_own_frontier():
    # the treatment is 0.5 s faster on every request: both readings give the same mean-latency CI, from one draw
    cells = [c for s in (1, 2) for c in (_cell("rb-a", "RB", s, [0.5] * N, _latency(s) - 0.5),
                                         _cell("ap-1", "AP", s, [0.5] * N, _latency(s)))]
    one, per = (_run(cells, reading=r)[0].deltas[0] for r in ("one_config", "per_metric"))
    assert one.ci == pytest.approx((-500.0, -500.0)) and per.ci == pytest.approx((-500.0, -500.0))


def test_an_unknown_reading_is_refused():
    with pytest.raises(ConfigError, match="unknown equal-quality reading"):
        equal_quality(_grid(), "RB", RULES, reading="best_of")
