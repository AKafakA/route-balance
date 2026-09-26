"""Per-cell summary: every metric by hand on a small cell, and every problem that invalidates a cell."""
import asyncio
import math

import numpy as np
import pytest

from analysis.metrics import Outcome, decode_prices, spread, summarize, summarize_cell
from analysis.quality import ScoredCorpus, ScoreStore
from analysis.slo import CellSlo, Limits, SloBase
from bench.arrivals import constant_trace
from bench.dataset import load_prompts
from bench.records import Sampling
from bench.runner import CellSpec, run_cell
from routebalance.config import ConfigError, fleet_digest
from routebalance.types import Instance, Tier
from tests.unit.fakes import CONFIG_DIGEST, SAMPLING, FakeScheduler, FakeTokenizer, identity, write_dataset

CORPUS = ScoredCorpus({"p0": {"small": 0.0, "large": 1.0}, "p1": {"small": 0.5, "large": 0.7}, "p2": {"large": 0.9}})
PRICES = {"small": 1e-7, "large": 4e-7}
SLO = CellSlo({"tight": {"i0": Limits(1.0, 0.01)}, "loose": {"i0": Limits(2.0, 0.02)}})


def _ok(prompt, model, out, e2e, server=0.25, timings=None, sent=5, served=5, ttft=0.0):
    return Outcome(prompt=prompt, ok=True, model=model, instance_id="i0", text=f"{prompt} by {model}", output_tokens=out,
                   e2e_s=e2e, ttft_s=ttft,
                   server_latency_s=server,
                   timings_ms={"queue_ms": e2e, "decide_ms": 1.0} if timings is None else timings,
                   prompt_tokens_sent=sent, prompt_tokens_served=served)


FAILED = Outcome(prompt="p0", ok=False, model=None, instance_id=None, text=None, output_tokens=None, e2e_s=30.0,
                 ttft_s=None,
                 server_latency_s=None, timings_ms=None, prompt_tokens_sent=5, prompt_tokens_served=None)
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
    # decode 0.25 s each: TTFT 0.25 / 1.25 / 1.75 / 0.95 s, TPOT 0.25 / (10, 20, 30, 40 - 1) s; the failed one misses
    assert (s.slo_attainment_tight, s.slo_attainment_loose) == (1 / 5, 3 / 5)
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
    assert s.corpus_gap == 2 and s.problems == ("served models that the corpus scores for no prompt: "
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
    corpus = ScoredCorpus({f"q{i}": {"small": 0.1} for i in range(10)})
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
# the fake scheduler replies with TTFT 1 ms and 4 ms on the engine; each request's e2e adds the client's own time
CELL_SLO = CellSlo.of({"tight": {"m-small": 10.0}, "loose": {"m-small": 15.0}},
                      SloBase(unit="s", ttft_s={"m-small_a30": 1.0}, tpot_s={"m-small_a30": 1.0}, provenance={}), FLEET)


def _record(tmp_path, **fake):
    prompts = load_prompts(write_dataset(tmp_path), 12, "fixed", 1, FakeTokenizer(), "tok-id")
    spec = CellSpec(cell_id="c1", labels={}, trace=constant_trace(12, 200.0, 1.0), seed=1,
                    sampling=Sampling(**SAMPLING), expected_config_digest=CONFIG_DIGEST)
    ident = identity().model_copy(update={"fleet_digest": fleet_digest(FLEET)})
    rec = asyncio.run(run_cell(spec, prompts, FakeScheduler(identities=(ident, ident), **fake), "r1"))
    return rec, ScoredCorpus({p.text: {"m-small": 0.5} for p in prompts.prompts})


def test_a_cell_is_scored_by_prompt_and_served_model(tmp_path):
    rec, corpus = _record(tmp_path)
    s = summarize_cell(rec, corpus, FLEET, CELL_SLO)
    assert (s.quality, s.matched, s.corpus_gap, s.coverage) == (0.5, 12, 0, 1.0) and s.valid
    first = rec.requests[0].prompt
    other_model = ScoredCorpus({p: ({"m-large": 0.9} if p == first else m) for p, m in corpus._scores.items()})
    s = summarize_cell(rec, other_model, FLEET, CELL_SLO)
    assert (s.quality, s.matched, s.corpus_gap, s.coverage) == (0.5, 11, 1, 11 / 12) and s.valid


def test_a_cell_record_is_summarised_against_its_fleet(tmp_path):
    rec, corpus = _record(tmp_path, fail=lambda i: i == 4)
    s = summarize_cell(rec, corpus, FLEET, CELL_SLO)
    assert s.problems == ("1/12 requests failed",)
    assert (s.n_ok, s.matched, s.quality, s.coverage) == (11, 11, 0.5, 1.0)
    assert (s.slo_attainment_tight, s.slo_attainment_loose) == (11 / 12, 11 / 12)      # the failed request misses
    assert math.isclose(s.cost_decode_per_1k, 10 * 2e-7 * 1000) and s.prompt_token_mismatches == 0


def test_a_cell_served_outside_the_slos_fleet_is_flagged_not_crashed(tmp_path):
    rec, corpus = _record(tmp_path)
    other = [Instance("m-small_1", FLEET[0].tier, "node1", "h1", 8000)]
    other_slo = CellSlo.of({"tight": {"m-small": 10.0}, "loose": {"m-small": 15.0}},
                           SloBase(unit="s", ttft_s={"m-small_a30": 1.0}, tpot_s={"m-small_a30": 1.0}, provenance={}),
                           other)
    s = summarize_cell(rec, corpus, other, other_slo)
    assert s.slo_attainment_tight is None and s.slo_attainment_loose is None
    assert s.problems == ("the fleet given to the analysis is not the fleet the cell ran on",
                          "served instances outside the fleet the SLO was built for: ['m-small_0']")


def test_another_fleet_cannot_price_a_cell(tmp_path):
    rec, corpus = _record(tmp_path)
    other = [Instance("m-small_0", Tier("m-small_a30", "m-small", "a30", 1e-7, 9e-7, 256), "node1", "h1", 8000)]
    assert summarize_cell(rec, corpus, FLEET, CELL_SLO).valid
    assert summarize_cell(rec, corpus, other, CELL_SLO).problems == (
        "the fleet given to the analysis is not the fleet the cell ran on",)


def test_a_second_template_on_the_serving_side_is_caught(tmp_path):
    rec, corpus = _record(tmp_path, extra_prompt_tokens=19)
    s = summarize_cell(rec, corpus, FLEET, CELL_SLO)
    assert s.prompt_token_mismatches == 12 and not s.valid


# ---------------------------------------------------------------- the budget sweep's cells (D67)

def _budget_record(tmp_path, budgets):
    """The fake cell as a budget cell: request i carries budgets[i] (fake replies: 10 output tokens, "ok")."""
    rec, _ = _record(tmp_path)
    return rec.model_copy(update={"budgets_sha256": "b" * 64, "requests": [
        r.model_copy(update={"budget_cost": budgets(r)}) for r in rec.requests]})


def test_a_budget_cell_is_scored_on_its_served_text_and_counts_exhaustion_as_v3(tmp_path):
    from analysis.quality import text_sha256
    tier = FLEET[0].tier

    def budgets(r):              # 0: the prompt alone is over budget; 1: the reply spends it all; others: plenty
        prompt = r.prompt_tokens * tier.cost_in
        return {0: prompt / 2, 1: prompt + 10 * tier.cost_out}.get(r.index, 1.0)
    rec = _budget_record(tmp_path, budgets)
    store = ScoreStore({(r.prompt, "m-small", text_sha256("ok")): 0.25 for r in rec.requests})
    s = summarize_cell(rec, store, FLEET, CELL_SLO)
    assert (s.quality, s.matched, s.unscored, s.budget_exhausted) == (0.25, 12, 0, 2) and s.valid
    partial = ScoreStore({k: v for k, v in store._scores.items() if k[0] != rec.requests[5].prompt})
    s = summarize_cell(rec, partial, FLEET, CELL_SLO)
    assert s.quality is None and s.unscored == 1 and not s.valid
    assert "1/12 served texts have no judged score: the cell's quality is unavailable until they are judged (D67)" \
        in s.problems


def test_a_budget_cell_takes_the_store_and_every_other_cell_the_corpus(tmp_path):
    rec, corpus = _record(tmp_path)
    with pytest.raises(ConfigError, match="served-text score store"):
        summarize_cell(rec, ScoreStore({}), FLEET, CELL_SLO)
    (tmp_path / "b").mkdir()
    budget_rec = _budget_record(tmp_path / "b", lambda r: 1.0)
    with pytest.raises(ConfigError, match="served-text score store"):
        summarize_cell(budget_rec, corpus, FLEET, CELL_SLO)
    assert summarize_cell(rec, corpus, FLEET, CELL_SLO).budget_exhausted is None


def test_a_budget_cell_on_the_wrong_fleet_is_flagged_not_crashed(tmp_path):
    from analysis.quality import text_sha256
    rec = _budget_record(tmp_path, lambda r: 1.0)
    other = [Instance("m-small_1", FLEET[0].tier, "node1", "h1", 8000)]
    slo = CellSlo.of({"tight": {"m-small": 10.0}, "loose": {"m-small": 15.0}},
                     SloBase(unit="s", ttft_s={"m-small_a30": 1.0}, tpot_s={"m-small_a30": 1.0}, provenance={}), other)
    store = ScoreStore({(r.prompt, "m-small", text_sha256("ok")): 0.25 for r in rec.requests})
    s = summarize_cell(rec, store, other, slo)
    assert s.budget_exhausted is None and not s.valid
    assert "request 0 was served outside the fleet given: exhaustion cannot be counted" in s.problems
