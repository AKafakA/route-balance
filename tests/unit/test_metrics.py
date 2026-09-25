"""Per-cell summary: every metric by hand on a small cell, and every problem that invalidates a cell."""
import asyncio
import math

import numpy as np
import pytest

from analysis.metrics import Outcome, decode_prices, spread, summarize, summarize_cell
from analysis.quality import ScoreStore, V3PairCorpus, text_sha256
from analysis.rules import GoodputSlo
from bench.arrivals import constant_trace
from bench.dataset import load_prompts
from bench.records import Sampling
from bench.runner import CellSpec, run_cell
from routebalance.config import ConfigError, fleet_digest
from routebalance.types import Instance, Tier
from tests.unit.fakes import CONFIG_DIGEST, SAMPLING, FakeScheduler, FakeTokenizer, identity, write_dataset

CORPUS = V3PairCorpus({"p0": {"small": 0.0, "large": 1.0}, "p1": {"small": 0.5, "large": 0.7}, "p2": {"large": 0.9}})
PRICES = {"small": 1e-7, "large": 4e-7}
SLO = GoodputSlo(kind="e2e_affine", base_ms=1000.0, per_output_token_ms=10.0)


def _ok(prompt, model, out, e2e, server=0.25, timings=None, sent=5, served=5, text=None):
    return Outcome(prompt=prompt, ok=True, model=model, text=f"{prompt} by {model}" if text is None else text,
                   output_tokens=out, e2e_s=e2e, server_latency_s=server,
                   timings_ms={"queue_ms": e2e, "decide_ms": 1.0} if timings is None else timings,
                   prompt_tokens_sent=sent, prompt_tokens_served=served)


FAILED = Outcome(prompt="p0", ok=False, model=None, text=None, output_tokens=None, e2e_s=30.0, server_latency_s=None,
                 timings_ms=None, prompt_tokens_sent=5, prompt_tokens_served=None)
CELL = [_ok("p0", "small", 10, 0.5), _ok("p1", "large", 20, 1.5), _ok("p2", "small", 30, 2.0),
        _ok("p1", "small", 40, 1.2), FAILED]


def _summary(outcomes=CELL, duration=4.0, corpus=CORPUS, prices=PRICES, problems=()):
    return summarize("c1", outcomes, duration, corpus, prices, SLO, problems)


def test_every_metric_on_a_small_cell():
    s = _summary()
    assert (s.n_requests, s.n_ok, s.n_failed) == (5, 4, 1)
    assert (s.matched, s.corpus_gap, s.unmatched, s.n_zero) == (3, 1, 0, 1)
    assert s.quality == pytest.approx((0.0 + 0.7 + 0.5) / 3) and s.coverage == 0.75
    assert s.quality_zeros_excluded == pytest.approx(0.6)
    assert s.cost_decode_per_1k == pytest.approx((10 * 1e-7 + 20 * 4e-7 + 30 * 1e-7 + 40 * 1e-7) / 4 * 1000)
    assert s.throughput_rps == 1.0
    e2e = [0.5, 1.5, 2.0, 1.2]
    assert s.e2e_ms.mean == float(np.mean(e2e) * 1000) and s.e2e_ms.p99 == float(np.percentile(e2e, 99) * 1000)
    assert s.e2e_ms.p50 == pytest.approx(1350.0)
    assert (s.slo_met, s.goodput_rps) == (2, 0.5)                    # deadlines 1100, 1200, 1300, 1400 ms
    assert set(s.overhead_ms) == {"queue_ms", "decide_ms"} and s.overhead_ms["decide_ms"].mean == 1.0
    assert s.overhead_ms["queue_ms"] == spread(e2e, 1.0)
    assert s.off_instance_ms == spread([x - 0.25 for x in e2e], 1000.0)
    assert s.prompt_token_mismatches == 0
    assert s.problems == ("1/5 requests failed",) and not s.valid


def test_a_clean_cell_is_valid():
    s = _summary(outcomes=CELL[:4])
    assert s.valid and s.n_failed == 0


def test_an_unmatched_prompt_invalidates_the_cell():
    s = _summary(outcomes=CELL[:4] + [_ok("p9", "small", 1, 0.1)])
    assert s.unmatched == 1 and any("absent from the scored corpora" in p for p in s.problems)


def test_a_prompt_token_mismatch_invalidates_the_cell():
    # the backend prefilled more tokens than were sent: the prompt was templated again somewhere
    s = _summary(outcomes=CELL[:3] + [_ok("p1", "small", 40, 1.2, served=24)])
    assert s.prompt_token_mismatches == 1
    assert s.problems == ("1/4 requests: the backend prefilled a different number of prompt tokens than the client "
                          "sent",)


def test_a_served_model_the_corpora_never_score_invalidates_the_cell():
    s = _summary(outcomes=CELL[:4] + [_ok("p1", "small-renamed", 1, 0.1)], prices={**PRICES, "small-renamed": 1e-7})
    assert s.corpus_gap == 2 and s.problems == ("served models that the scores cover for no text: "
                                                "['small-renamed']",)


def test_an_unpriced_model_invalidates_the_cell_and_has_no_cost():
    s = _summary(outcomes=CELL[:4] + [_ok("p0", "huge", 1, 0.1)])
    assert s.cost_decode_per_1k is None and any("without a price" in p for p in s.problems)


def test_mixed_scheduler_timings_invalidate_the_cell():
    s = _summary(outcomes=CELL[:3] + [_ok("p1", "small", 40, 1.2, timings={"router_ms": 1.0})])
    assert s.overhead_ms == {} and "scheduler timing keys differ between requests" in s.problems


def test_a_cell_of_only_corpus_gaps_has_no_quality():
    s = _summary(outcomes=[_ok("p2", "small", 3, 0.2)])
    assert s.quality is None and "no successful request has a judged score" in s.problems


def test_a_cell_with_no_success_is_summarised_not_dropped():
    s = _summary(outcomes=[FAILED, FAILED])
    assert s.n_failed == 2 and s.quality is None and s.e2e_ms is None and s.cost_decode_per_1k is None
    assert "no successful requests" in s.problems and s.throughput_rps == 0.0


def test_sums_are_exactly_rounded():
    # a sequential sum of ten 0.1s is 0.9999999999999999; the v3 analysis's (compensated) sum was not
    corpus = V3PairCorpus({f"q{i}": {"small": 0.1} for i in range(10)})
    s = _summary(outcomes=[_ok(f"q{i}", "small", 1, 0.1) for i in range(10)], corpus=corpus,
                 prices={"small": 0.1})
    assert sum([0.1] * 10) != 1.0
    assert s.quality == 0.1 and s.cost_decode_per_1k == 100.0


def test_duration_must_be_positive():
    with pytest.raises(ConfigError):
        _summary(duration=0.0)


def test_prices_come_from_the_fleet_tiers():
    tiers = [Tier("s_a30", "small", "a30", 1e-7, 2e-7, 256), Tier("l_a100", "large", "a100", 3e-7, 4e-7, 256),
             Tier("s_v100", "small", "v100", 1e-7, 2e-7, 256)]
    assert decode_prices(tiers) == {"small": 2e-7, "large": 4e-7}
    with pytest.raises(ConfigError, match="two output prices"):
        decode_prices(tiers + [Tier("l_a30", "large", "a30", 3e-7, 5e-7, 256)])


FLEET = [Instance("m-small_0", Tier("m-small_a30", "m-small", "a30", 1e-7, 2e-7, 256), "node1", "h1", 8000)]


def _record(tmp_path, **fake):
    prompts = load_prompts(write_dataset(tmp_path), 12, "fixed", 1, FakeTokenizer(), "tok-id")
    spec = CellSpec(cell_id="c1", labels={}, trace=constant_trace(12, 200.0, 1.0), seed=1,
                    sampling=Sampling(**SAMPLING), expected_config_digest=CONFIG_DIGEST)
    ident = identity().model_copy(update={"fleet_digest": fleet_digest(FLEET)})
    rec = asyncio.run(run_cell(spec, prompts, FakeScheduler(identities=(ident, ident), **fake), "r1"))
    # the fake scheduler serves the text "ok" for every prompt
    return rec, ScoreStore({(p.text, "m-small", text_sha256("ok")): 0.5 for p in prompts.prompts})


def test_the_served_text_is_what_is_scored(tmp_path):
    rec, store = _record(tmp_path)
    assert summarize_cell(rec, store, FLEET, SLO).quality == 0.5
    # the same prompts and model, scored on another text: nothing stands in for the served one (D39)
    other = ScoreStore({(p, m, text_sha256("ok ")): s for (p, m, _), s in store._scores.items()})
    s = summarize_cell(rec, other, FLEET, SLO)
    assert (s.quality, s.matched, s.unscored) == (None, 0, 12)
    assert "12/12 served texts have no judged score: the cell's quality is unavailable until they are judged (D39)" \
        in s.problems


def test_one_unscored_text_makes_the_quality_unavailable(tmp_path):
    rec, store = _record(tmp_path)
    first = rec.requests[0].prompt
    partial = ScoreStore({k: v for k, v in store._scores.items() if k[0] != first})
    s = summarize_cell(rec, partial, FLEET, SLO)
    assert s.quality is None and s.quality_zeros_excluded is None and (s.matched, s.unscored) == (11, 1)
    assert s.coverage == 11 / 12 and not s.valid


def test_a_cell_is_never_scored_by_prompt_and_model(tmp_path):
    rec, _ = _record(tmp_path)
    corpus = V3PairCorpus({r.prompt: {"m-small": 0.5} for r in rec.requests})
    with pytest.raises(ConfigError, match="score store"):
        summarize_cell(rec, corpus, FLEET, SLO)


def test_the_v3_pair_lookup_ignores_the_text():
    s = _summary(outcomes=[_ok("p1", "small", 3, 0.2, text="anything"), _ok("p1", "small", 3, 0.2, text="else")])
    assert (s.matched, s.unscored, s.quality) == (2, 0, 0.5)


def test_a_cell_record_is_summarised_against_its_fleet(tmp_path):
    rec, corpus = _record(tmp_path, fail=lambda i: i == 4)
    s = summarize_cell(rec, corpus, FLEET, SLO)
    assert s.problems == ("1/12 requests failed",)
    assert (s.n_ok, s.matched, s.quality, s.coverage) == (11, 11, 0.5, 1.0)
    assert math.isclose(s.cost_decode_per_1k, 10 * 2e-7 * 1000) and s.prompt_token_mismatches == 0


def test_another_fleet_cannot_price_a_cell(tmp_path):
    rec, corpus = _record(tmp_path)
    other = [Instance("m-small_0", Tier("m-small_a30", "m-small", "a30", 1e-7, 9e-7, 256), "node1", "h1", 8000)]
    assert summarize_cell(rec, corpus, FLEET, SLO).valid
    assert summarize_cell(rec, corpus, other, SLO).problems == (
        "the fleet given to the analysis is not the fleet the cell ran on",)


def test_a_second_template_on_the_serving_side_is_caught(tmp_path):
    rec, corpus = _record(tmp_path, extra_prompt_tokens=19)
    s = summarize_cell(rec, corpus, FLEET, SLO)
    assert s.prompt_token_mismatches == 12 and not s.valid
