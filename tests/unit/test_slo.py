"""The SLO of D57: TTFT and TPOT against k times the tier's single-request values, attainment, and goodput."""
import json

import pytest

from analysis.slo import CellSlo, Goodput, SloBase, goodput, goodputs
from routebalance.config import ConfigError
from routebalance.types import Instance, Tier

SMALL = Tier("s_a30", "S", "a30", 1e-7, 1e-7, 8, ())
LARGE = Tier("l_a100", "L", "a100", 4e-7, 4e-7, 8, ())
FLEET = [Instance("S_0", SMALL, "n0", "h0", 8000), Instance("L_1", LARGE, "n1", "h1", 8001)]
K = {"tight": {"S": 8.0, "L": 4.0}, "loose": {"S": 12.0, "L": 8.0}}          # values exact in binary


def _base(tmp_path, **over):
    d = {"unit": "s", "ttft_s": {"s_a30": 0.125, "l_a100": 0.5}, "tpot_s": {"s_a30": 0.0625, "l_a100": 0.25},
         "provenance": {}} | over
    (tmp_path / "base.json").write_text(json.dumps(d))
    return SloBase.load(tmp_path / "base.json")


def test_limits_are_k_times_the_tiers_single_request_latency(tmp_path):
    slo = CellSlo.of(K, _base(tmp_path), FLEET)
    assert (slo.limits["tight"]["S_0"].ttft_s, slo.limits["tight"]["S_0"].tpot_s) == (1.0, 0.5)
    assert (slo.limits["tight"]["L_1"].ttft_s, slo.limits["tight"]["L_1"].tpot_s) == (2.0, 1.0)
    assert (slo.limits["loose"]["L_1"].ttft_s, slo.limits["loose"]["L_1"].tpot_s) == (4.0, 2.0)


def test_ttft_counts_every_wait_before_the_first_token_and_tpot_the_decode(tmp_path):
    slo = CellSlo.of(K, _base(tmp_path), FLEET)
    # the scheduler's first token at 1.0 s; the engine took 2.5 s, 0.5 s to its first token: TPOT 2.0 / 4 = 0.5 s
    assert slo.met("tight", "S_0", 1.0, 2.5, 0.5, 5)                    # both limits are inclusive
    assert not slo.met("tight", "S_0", 1.25, 2.5, 0.5, 5)               # 0.25 s more before the first token: TTFT 1.25
    assert slo.met("loose", "S_0", 1.25, 2.5, 0.5, 5)
    assert not slo.met("tight", "S_0", 1.0, 3.0, 0.5, 5)                # TTFT 1.0, TPOT 2.5 / 4 = 0.625
    assert slo.met("loose", "S_0", 1.0, 3.0, 0.5, 5)
    assert slo.met("tight", "S_0", 1.0, 0.5, 0.5, 1)                    # one token: TTFT alone
    # the large tier's own limits (tight 2.0 s / 1.0 s, loose 4.0 s / 2.0 s), not the small tier's
    assert slo.met("tight", "L_1", 1.0, 3.0, 0.5, 5)                    # TTFT 1.0, TPOT 0.625: S_0 misses it
    assert not slo.met("tight", "L_1", 2.5, 2.5, 0.5, 5)                # TTFT 2.5
    assert slo.met("loose", "L_1", 2.5, 2.5, 0.5, 5) and not slo.met("loose", "S_0", 2.5, 2.5, 0.5, 5)


def test_every_tier_and_model_of_the_fleet_needs_its_values(tmp_path):
    with pytest.raises(ConfigError, match="tier l_a100"):
        CellSlo.of(K, _base(tmp_path, ttft_s={"s_a30": 0.02}, tpot_s={"s_a30": 0.01}), FLEET)
    with pytest.raises(ConfigError, match="model L"):
        CellSlo.of({"tight": {"S": 10.0}, "loose": {"S": 15.0, "L": 10.0}}, _base(tmp_path), FLEET)
    with pytest.raises(ConfigError, match="not in the cell's fleet"):
        CellSlo.of(K, _base(tmp_path), FLEET).met("tight", "X_9", 0.6, 0.5, 0.1, 3)


@pytest.mark.parametrize("over", [{"unit": "ms"}, {"tpot_s": {"s_a30": 0.01}}, {"ttft_s": {"s_a30": 0.0, "l_a100": 0.1}}])
def test_malformed_bases_fail(tmp_path, over):
    with pytest.raises(ConfigError):
        _base(tmp_path, **over)


def test_goodput_is_the_rate_before_attainment_first_falls_below_the_level():
    grid = [(6.0, 1.0), (12.0, 0.98), (18.0, 0.86), (24.0, 0.95), (30.0, 0.4)]
    assert goodput(grid, 0.90) == Goodput(12.0 + 6.0 * (0.98 - 0.90) / (0.98 - 0.86), False)
    assert goodput(grid, 0.99) == Goodput(6.0 + 6.0 * (1.0 - 0.99) / (1.0 - 0.98), False)
    assert goodput([(6.0, 0.95), (12.0, 0.9)], 0.99) == Goodput(None, False)
    assert goodput([(6.0, 1.0), (12.0, 0.995)], 0.99) == Goodput(12.0, True)
    with pytest.raises(ConfigError):
        goodput([(6.0, 1.0), (6.0, 0.9)], 0.9)


def test_goodputs_read_each_configuration_over_its_stationary_traces():
    from analysis.frontier import config_points
    from tests.unit.test_frontier import pt
    rates = {"l6": 6.0, "l12": 12.0, "l30": 30.0}
    pts = config_points([pt("a", "S", 1, 0.4, 100, trace="l6", miss=0.0), pt("a", "S", 1, 0.4, 100, trace="l12", miss=0.05),
                         pt("a", "S", 1, 0.4, 100, trace="l30", miss=0.5), pt("a", "S", 1, 0.4, 100, trace="step", miss=1.0),
                         pt("b", "S", 1, 0.4, 100, trace="l6", miss=0.2)])
    g = goodputs(pts, rates, 0.9, "tight")
    assert g[("g1", "a")] == Goodput(12.0 + 18.0 * (0.95 - 0.9) / (0.95 - 0.5), False)   # the step trace is left out
    assert g[("g1", "b")] == Goodput(None, False)
    assert goodputs(pts, rates, 0.9, "loose")[("g1", "a")].rate == pytest.approx(12.0 + 18.0 * (0.975 - 0.9) / 0.225)
    with pytest.raises(ConfigError, match="unknown SLO level"):
        goodputs(pts, rates, 0.9, "medium")


def test_the_ttft_is_the_schedulers_first_token_time_not_e2e_minus_the_decode(tmp_path):
    """Codex audit 09-29: E2E minus the decode also counted the reply's delivery after the last token; the SLO now takes
    the scheduler's arrival-to-first-token time (v2's client_ttft), so a delay after generation cannot flip it."""
    slo = CellSlo.of(K, _base(tmp_path), FLEET)
    assert slo.met("tight", "S_0", 1.0, 2.5, 0.5, 5) and not slo.met("tight", "S_0", 1.01, 2.5, 0.5, 5)
    with pytest.raises(ConfigError, match="without the scheduler's first-token time"):
        slo.met("tight", "S_0", None, 2.5, 0.5, 5)
