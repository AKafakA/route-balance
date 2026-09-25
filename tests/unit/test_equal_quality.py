"""Equal-quality deltas: interpolation on the treatment frontier, tolerance, paired seed bootstrap."""
import pytest

from analysis.equal_quality import equal_quality, value_at
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
    assert mean.ci is None and mean.undefined_resamples == 0
    assert row.matched_presets == ("rb-a", "rb-c") and row.excluded_presets == ()
    assert [d.metric for d in row.deltas] == ["e2e_mean_ms", "e2e_p99_ms", "cost_decode_per_1k", "slo_miss_tight",
                                             "slo_miss_loose"]
    assert row.deltas[2].delta == pytest.approx(-1.25)


def test_a_baseline_outside_the_treatment_range_has_no_delta():
    pts = _grid()[:3] + [pt("ap-1", "AP", 1, 0.7, 2500)]
    [row] = equal_quality(pts, "RB", RULES)
    assert all(d.treatment is None and d.delta is None and d.relative is None for d in row.deltas)


def test_the_bootstrap_is_paired_over_seeds():
    # a seed effect shared by every configuration cancels in every resample: the CI collapses on the delta
    [row] = equal_quality(_grid(seeds=(1, 2, 3), shift=lambda s: 400.0 * s), "RB", RULES)
    mean = row.deltas[0]
    assert mean.delta == pytest.approx(-1250.0)
    assert mean.ci == pytest.approx((-1250.0, -1250.0)) and mean.undefined_resamples == 0


def test_the_bootstrap_is_deterministic_and_covers_seed_noise():
    pts = _grid(seeds=(1, 2, 3))
    pts[-1] = pt("ap-1", "AP", 3, 0.425, 3100)            # the baseline alone is slower on seed 3
    rows = [equal_quality(pts, "RB", RULES)[0].deltas[0] for _ in range(2)]
    assert rows[0] == rows[1]
    lo, hi = rows[0].ci
    assert lo < rows[0].delta < hi and hi - lo > 0


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


def test_resamples_without_an_equal_quality_point_void_the_ci():
    # on seed 2 the baseline's quality is far above every preset, so resamples heavy in seed 2 have no point
    pts = _grid(seeds=(1, 2))
    pts[-1] = pt("ap-1", "AP", 2, 0.9, 2500)
    mean = equal_quality(pts, "RB", RULES)[0].deltas[0]
    assert mean.ci is None and mean.undefined_resamples > 0
