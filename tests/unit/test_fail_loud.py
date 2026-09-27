"""A malformed stream or an early loop stall is never hidden (review of 09-27)."""
import pytest

from routebalance.backend import _choices, _completion_tokens
from routebalance.decision_log import LagProbe


@pytest.mark.parametrize("event", [{}, {"choices": None}, {"choices": "x"}])
def test_an_event_without_choices_is_malformed(event):
    with pytest.raises(ValueError, match="malformed choices"):
        _choices(event)


@pytest.mark.parametrize("n", [-1, True, 1.0])
def test_completion_counts_are_non_negative_integers(n):
    with pytest.raises(ValueError, match="malformed usage"):
        _completion_tokens({"usage": {"completion_tokens": n}})
    assert _completion_tokens({"usage": {"completion_tokens": 0}}) == 0 and _completion_tokens({}) is None


def test_the_loop_lag_summary_keeps_an_early_stall():
    probe = LagProbe()
    probe._samples = [100.0] + [0.1] * 50_000
    assert probe.summary()["max_ms"] == 100.0 and probe.summary()["samples"] == 50_001
