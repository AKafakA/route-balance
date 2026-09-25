"""Frontier points: seed means, Pareto efficiency, goodput above the floor, refusal of invalid cells."""
import pytest

from analysis.frontier import CellPoint, best_goodput, by_config, cell_point, config_points, frontiers, pareto
from analysis.manifest import Coverage
from analysis.metrics import CellSummary, Spread
from bench.arrivals import constant_trace
from bench.scope import ConfigEntry, DatasetEntry, ExpectedCell
from routebalance.config import ConfigError


def pt(config, system, seed, q, mean, trace="l30", goodput=1.0, knob=None, group="g1"):
    return CellPoint(group=group, config_id=config, system=system, knob=config if knob is None else knob, dispatch="lor",
                     trace_id=trace, seed=seed, quality=q, e2e_mean_ms=mean, e2e_p99_ms=3 * mean,
                     cost_decode_per_1k=mean / 1000, goodput_rps=goodput, throughput_rps=20.0)


def test_pareto_keeps_the_efficient_points_in_quality_order():
    pairs = [(0.4, 100), (0.5, 200), (0.45, 300), (0.5, 150), (0.3, 50), (0.3, 50)]
    assert pareto(pairs) == [4, 5, 0, 3]


def test_config_points_are_seed_means():
    pts = config_points([pt("a", "S", 1, 0.4, 100), pt("a", "S", 2, 0.5, 300), pt("b", "S", 1, 0.3, 50)])
    a = next(p for p in pts if p.config_id == "a")
    assert a.seeds == (1, 2) and a.quality == pytest.approx(0.45) and a.e2e_mean_ms == 200.0
    assert a.e2e_p99_ms == 600.0 and a.cost_decode_per_1k == pytest.approx(0.2)


def test_repeated_or_relabelled_cells_are_errors():
    with pytest.raises(ConfigError, match="two cells"):
        by_config([pt("a", "S", 1, 0.4, 100), pt("a", "S", 1, 0.4, 100)])
    with pytest.raises(ConfigError, match="two different labels"):
        by_config([pt("a", "S", 1, 0.4, 100), pt("a", "T", 2, 0.4, 100)])


def test_frontiers_are_per_system_and_trace():
    pts = config_points([pt("a", "S", 1, 0.4, 100), pt("b", "S", 1, 0.5, 90), pt("c", "T", 1, 0.3, 500),
                         pt("a", "S", 1, 0.4, 10, trace="l12")])
    f = frontiers(pts, "e2e_mean_ms")
    assert [p.config_id for p in f[("g1", "l30", "S")]] == ["b"]
    assert [p.config_id for p in f[("g1", "l30", "T")]] == ["c"] and [p.config_id for p in f[("g1", "l12", "S")]] == ["a"]
    with pytest.raises(ValueError):
        frontiers(pts, "quality")


def test_best_goodput_is_taken_above_the_quality_floor():
    pts = config_points([pt("a", "S", 1, 0.30, 100, goodput=9.0), pt("b", "S", 1, 0.40, 100, goodput=5.0),
                         pt("c", "S", 1, 0.45, 100, goodput=4.0), pt("d", "T", 1, 0.2, 100, goodput=9.0)])
    best = best_goodput(pts, quality_floor=0.35)
    assert best[("g1", "l30", "S")].config_id == "b" and best[("g1", "l30", "T")] is None


def test_groups_are_never_pooled():
    # the same config id and trace in two groups (e.g. two topologies) are two configurations
    pts = config_points([pt("a", "S", 1, 0.4, 100), pt("a", "S", 1, 0.5, 900, group="g2"),
                         pt("b", "S", 1, 0.45, 50, group="g2")])
    assert [(p.group, p.config_id, p.e2e_mean_ms) for p in pts] == [("g1", "a", 100), ("g2", "a", 900),
                                                                     ("g2", "b", 50)]
    f = frontiers(pts, "e2e_mean_ms")
    assert [p.config_id for p in f[("g1", "l30", "S")]] == ["a"]
    assert [p.config_id for p in f[("g2", "l30", "S")]] == ["b", "a"]


def _expected():
    return ExpectedCell("a__l30__s1", "main", "g1", "a", ConfigEntry(system="S", knob="k", dispatch="lor",
                                                                config_digest="c" * 64),
                        "l30", constant_trace(10, 30.0, 1.0), 1, "fixed",
                        DatasetEntry(path="d.jsonl", sha256="0" * 64, rows=10, tokenizer_id="t"))


def _summary(problems):
    return CellSummary(cell_id="a__l30__s1", n_requests=10, n_ok=10, n_failed=0, quality=0.4, matched=10,
                       corpus_gap=0, unmatched=0, n_zero=1, coverage=1.0, quality_zeros_excluded=0.44,
                       cost_decode_per_1k=0.02, throughput_rps=9.0, e2e_ms=Spread(100.0, 90.0, 300.0), slo_met=8,
                       goodput_rps=8.0, overhead_ms={}, off_instance_ms=None, prompt_token_mismatches=0,
                       problems=problems)


def _coverage(ok, fleet_problems=None):
    return Coverage(expected=1, ok=ok, missing=(), invalid={}, accounted={}, unexpected=(), unused_accounting=(),
                    fleet_problems={} if fleet_problems is None else fleet_problems, pending={}, aborted=())


def test_cell_points_come_only_from_cells_ok_in_the_manifest_and_valid():
    p = cell_point(_expected(), _summary(()), _coverage(("a__l30__s1",)))
    assert (p.group, p.system, p.knob, p.quality, p.e2e_mean_ms, p.e2e_p99_ms, p.goodput_rps) == (
        "g1", "S", "k", 0.4, 100.0, 300.0, 8.0)
    with pytest.raises(ConfigError, match="invalid"):
        cell_point(_expected(), _summary(("1/10 requests failed",)), _coverage(("a__l30__s1",)))
    with pytest.raises(ConfigError, match="not ok in the manifest"):
        cell_point(_expected(), _summary(()), _coverage(()))
    with pytest.raises(ConfigError, match="cannot be compared"):
        cell_point(_expected(), _summary(()), _coverage(("a__l30__s1",), {"g1": "cells ran on 2 different fleets"}))
