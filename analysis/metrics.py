"""Per-cell summary: quality, cost, throughput, latency, SLO attainment and scheduler overhead.

Every metric is over the cell's successful requests. A cell with a failed request is still summarised and
counted, but it carries a problem and is invalid (analysis/validity.py adds the execution checks).
  quality     zero-safe mean judged score of the successful requests, looked up by (prompt, served model) in
              the scored corpus (analysis/quality.py, as v2 and v3); a 0.0 counts. A budget cell's quality is
              the judged score of each text actually served, from the score store (D67; an empty text scores
              0): one text without a score makes the cell's quality unavailable (None, and a problem), never a
              mean over the rest. `quality_zeros_excluded` (mean over the non-zero scores) is a labelled
              secondary scale kept only to compare with older numbers; it is never used for a claim
  coverage    matched / successful. Corpus gaps are counted and not problems; unmatched prompts are problems,
              and so is a served model that the corpus scores for no prompt (a model-name mismatch)
  budget_exhausted   budget cells only: the successful requests whose budget ran out, as v3 counted them (its
              budget could not pay for the prompt on the served tier, or the served reply's cost, prompt
              included, reached the budget to within 1e-9 USD); None for every other cell
  cost_total  the report's Cost (D117): sum(prompt_tokens * input price + output_tokens * output price of the served
              model) / successful * 1000, USD per 1k successful requests (as v2 priced it; the objective prices the
              prompt the same way)
  cost        DecodeCost, beside it: sum(output_tokens * output price of the served model) / successful * 1000, as
              v3 counted it; prices from the fleet config (USD per token)
  throughput  successful requests / duration_s
  e2e         client end-to-end latency, mean / p50 / p99 in ms (numpy, linear-interpolated percentiles)
  slo         the share of the cell's requests that met the SLO at each level, tight and loose (analysis/slo.py:
              TTFT and TPOT within k times the served tier's single-request values); a failed request never
              does. None only for the re-aggregated v3 cells, whose records carry no scheduler-side TTFT
  overhead    for each scheduler timing (timings_ms): mean / p50 / p99 in ms; and the off-instance residual
              e2e - server_latency (client, network, scheduler queueing and decision time)
  prompt tokens   a request whose backend-reported prompt_tokens differs from the count the client sent is a
              problem: the prompt was not templated once as bench/dataset.py documents, and every system's
              estimates were fed a wrong prefill length
A cell record is summarised against the fleet it ran on: the fleet's digest must be the cell's, and the prices
are that fleet's.
Sums of scores and costs are math.fsum (exactly rounded, whatever the order or the Python version). The v3
analysis ran on a Python whose sum() is compensated, and its quality column equals the fsum mean on every
banked row; a plain sequential sum differs from it by ~1e-14.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

from analysis.quality import ScoredCorpus, ScoreStore
from analysis.slo import LEVELS, CellSlo
from analysis.validity import integrity_problems
from bench.records import CellRecord
from routebalance.config import ConfigError, fleet_digest
from routebalance.types import Instance, Tier


@dataclass(frozen=True)
class Outcome:
    """One request as the metrics see it. Success-only fields are None for a failed request."""
    prompt: str
    ok: bool
    model: Optional[str]
    instance_id: Optional[str]
    text: Optional[str]                   # the served response (generated_text)
    output_tokens: Optional[int]
    e2e_s: float
    ttft_s: Optional[float]               # the scheduler's backend clock: its call to the instance -> first token
    server_latency_s: Optional[float]
    first_token_s: Optional[float]        # the scheduler's clock: the request's arrival -> first token (the SLO's TTFT)
    timings_ms: Optional[Mapping[str, float]]
    prompt_tokens_sent: Optional[int]
    prompt_tokens_served: Optional[int]


@dataclass(frozen=True)
class Spread:
    mean: float
    p50: float
    p99: float


@dataclass(frozen=True)
class CellSummary:
    cell_id: str
    n_requests: int
    n_ok: int
    n_failed: int
    quality: Optional[float]
    matched: int
    corpus_gap: int
    unmatched: int
    unscored: int                             # budget cells: served texts without a score (quality unavailable)
    budget_exhausted: Optional[int]           # budget cells only
    n_zero: int
    coverage: Optional[float]
    quality_zeros_excluded: Optional[float]   # secondary scale, never for claims
    cost_decode_per_1k: Optional[float]
    throughput_rps: float
    e2e_ms: Optional[Spread]
    slo_attainment_tight: Optional[float]
    slo_attainment_loose: Optional[float]
    overhead_ms: dict[str, Spread]
    off_instance_ms: Optional[Spread]
    prompt_token_mismatches: Optional[int]    # sent count != vLLM's count; None when not recorded
    problems: tuple[str, ...]
    cost_total_per_1k: Optional[float] = None  # D117: the prompt at the input price + the output at the output price
    max_send_lag_ms: Optional[float] = None    # D134: the latest start after a scheduled arrival (counted, never a gate)

    @property
    def valid(self) -> bool:
        return not self.problems


@dataclass(frozen=True, eq=False)
class RequestValues:
    """A valid cell's per-request values by send position, for the request bootstrap (analysis/equal_quality.py):
    each request's judged score (nan for a corpus gap), E2E latency, decode cost (its output tokens at the
    served model's output price, USD) and whether it met each SLO level. at() recomputes the cell's metrics on a
    draw of the requests as summarize() computes them on all of them, to rounding (numpy means in send order
    where summarize() takes exactly rounded sums in record order); a draw of only corpus gaps has quality nan."""
    index: tuple[int, ...]                 # send positions, increasing
    prompt_ids: tuple[str, ...]
    score: np.ndarray
    e2e_s: np.ndarray
    cost: np.ndarray
    met_tight: np.ndarray
    met_loose: np.ndarray
    cost_total: Optional[np.ndarray] = None   # D117: prompt + output cost (USD); None without input prices

    def at(self, draws: np.ndarray) -> dict[str, np.ndarray]:
        """Each metric of analysis/frontier.py, and quality, on each row of `draws` (resamples x positions)."""
        e2e = self.e2e_s[draws]
        scores = self.score[draws]
        scored = (~np.isnan(scores)).sum(axis=1)
        quality = np.divide(np.nansum(scores, axis=1), scored, out=np.full(len(scored), np.nan), where=scored > 0)
        return {"quality": quality,
                "e2e_mean_ms": e2e.mean(axis=1) * 1000.0,
                "e2e_p99_ms": np.percentile(e2e, 99, axis=1) * 1000.0,
                "cost_decode_per_1k": self.cost[draws].mean(axis=1) * 1000,
                "cost_total_per_1k": (np.full(len(draws), np.nan) if self.cost_total is None
                                      else self.cost_total[draws].mean(axis=1) * 1000),
                "slo_miss_tight": 1.0 - self.met_tight[draws].mean(axis=1),
                "slo_miss_loose": 1.0 - self.met_loose[draws].mean(axis=1)}


def request_values(cell: CellRecord, quality: ScoredCorpus | ScoreStore, fleet: Sequence[Instance],
                   slo: CellSlo) -> RequestValues:
    """The per-request values of a cell that summarize_cell() found valid (every request succeeded, was scored
    or is a corpus gap, and was served by a priced instance of the fleet)."""
    prices, in_prices = decode_prices(i.tier for i in fleet), input_prices(i.tier for i in fleet)
    rows = []
    for r, o in zip(cell.requests, outcomes_of(cell), strict=True):
        if not o.ok:
            raise ConfigError(f"cell {cell.cell_id}: request {r.index} failed; only a valid cell has request values")
        outcome, score = quality.lookup(o.prompt, o.model, o.text)
        if outcome not in ("matched", "corpus_gap") or o.model not in prices:
            raise ConfigError(f"cell {cell.cell_id}: request {r.index} is {outcome} or unpriced; only a valid cell "
                              "has request values")
        if o.prompt_tokens_served is None:
            raise ConfigError(f"cell {cell.cell_id}: request {r.index} has no served prompt token count (D117)")
        rows.append((r.index, r.prompt_id, score if outcome == "matched" else math.nan, o.e2e_s,
                     o.output_tokens * prices[o.model],
                     *(slo.met(lvl, o.instance_id, o.first_token_s, o.server_latency_s, o.ttft_s, o.output_tokens)
                       for lvl in LEVELS),
                     o.prompt_tokens_served * in_prices[o.model] + o.output_tokens * prices[o.model]))
    rows.sort(key=lambda row: row[0])
    if len({row[0] for row in rows}) != len(rows):
        raise ConfigError(f"cell {cell.cell_id}: two requests share a send position")
    cols = list(zip(*rows, strict=True)) if rows else [()] * 8
    tight, loose = (cols[5 + LEVELS.index(lvl)] for lvl in ("tight", "loose"))
    return RequestValues(index=tuple(cols[0]), prompt_ids=tuple(cols[1]), score=np.array(cols[2], dtype=float),
                         e2e_s=np.array(cols[3], dtype=float), cost=np.array(cols[4], dtype=float),
                         met_tight=np.array(tight, dtype=float), met_loose=np.array(loose, dtype=float),
                         cost_total=np.array(cols[7], dtype=float))


def input_prices(tiers: Iterable[Tier]) -> dict[str, float]:
    """Input price (USD per prompt token) of each served model name, from the fleet's tiers (D117)."""
    prices: dict[str, float] = {}
    for t in tiers:
        if t.model in prices and prices[t.model] != t.cost_in:
            raise ConfigError(f"model {t.model!r} has two input prices in the fleet; cost would be ambiguous")
        prices[t.model] = t.cost_in
    return prices


def decode_prices(tiers: Iterable[Tier]) -> dict[str, float]:
    """Output price (USD per token) of each served model name, from the fleet's tiers."""
    prices: dict[str, float] = {}
    for t in tiers:
        if t.model in prices and prices[t.model] != t.cost_out:
            raise ConfigError(f"model {t.model!r} has two output prices in the fleet; cost would be ambiguous")
        prices[t.model] = t.cost_out
    return prices


def spread(values: Sequence[float], scale: float) -> Spread:
    """mean / p50 / p99 of values * scale, in the order v3 computed them (numpy, then scaled)."""
    return Spread(float(np.mean(values) * scale), float(np.percentile(values, 50) * scale),
                  float(np.percentile(values, 99) * scale))


def outcomes_of(cell: CellRecord) -> list[Outcome]:
    out = []
    for r in cell.requests:
        rep = r.reply
        if rep is None:
            out.append(Outcome(prompt=r.prompt, ok=False, model=None, instance_id=None, text=None,
                               output_tokens=None, e2e_s=r.e2e_s, ttft_s=None, server_latency_s=None, timings_ms=None,
                               prompt_tokens_sent=r.prompt_tokens, prompt_tokens_served=None, first_token_s=None))
        else:
            out.append(Outcome(prompt=r.prompt, ok=True, model=rep.model, instance_id=rep.instance_id,
                               text=rep.generated_text, output_tokens=rep.output_tokens, e2e_s=r.e2e_s, ttft_s=rep.ttft_s,
                               server_latency_s=rep.server_latency_s, timings_ms=rep.timings_ms,
                               prompt_tokens_sent=r.prompt_tokens, prompt_tokens_served=rep.prompt_tokens,
                               first_token_s=rep.first_token_s))
    return out


def summarize_cell(cell: CellRecord, quality: ScoredCorpus | ScoreStore, fleet: Sequence[Instance],
                   slo: CellSlo) -> CellSummary:
    """A campaign cell, summarised against the fleet it ran on: a budget cell's quality from the served-text
    score store (D67), every other cell's from the scored corpus (D62)."""
    budgeted = cell.budgets_sha256 is not None
    if budgeted != isinstance(quality, ScoreStore) or not isinstance(quality, (ScoredCorpus, ScoreStore)):
        raise ConfigError(f"cell {cell.cell_id}: a budget cell's quality comes from the served-text score store, "
                          "every other cell's from the scored corpus")
    problems = integrity_problems(cell)
    if fleet_digest(list(fleet)) != cell.identity_start.fleet_digest:
        problems.append("the fleet given to the analysis is not the fleet the cell ran on")
    summary = summarize(cell.cell_id, outcomes_of(cell), cell.duration_s, quality,
                        decode_prices(i.tier for i in fleet), slo, problems, input_prices(i.tier for i in fleet))
    lags = [r.sent_s - r.arrival_s for r in cell.requests]            # D134: a late send is counted, never a gate
    summary = replace(summary, max_send_lag_ms=max(lags) * 1000 if lags else None)
    if not budgeted:
        return summary
    exhausted, why = budget_exhausted(cell, fleet)
    return replace(summary, budget_exhausted=exhausted, problems=summary.problems + ((why,) if why else ()))


def budget_exhausted(cell: CellRecord, fleet: Sequence[Instance]) -> tuple[Optional[int], Optional[str]]:
    """v3's count: successful requests whose budget could not pay for the prompt on the served tier, or whose
    served cost (the prompt at the input price plus the output at the output price, as sent and as served)
    reached the budget to within 1e-9 USD. (None, the reason) when a request carries no budget or was served
    by an instance outside the fleet given (a wrong fleet is a problem of its own, never a crash)."""
    tier_of = {i.instance_id: i.tier for i in fleet}
    n = 0
    for r in cell.requests:
        if r.reply is None:
            continue
        if r.budget_cost is None:
            return None, f"request {r.index} of a budget cell carries no budget: exhaustion cannot be counted"
        if r.reply.instance_id not in tier_of:
            return None, f"request {r.index} was served outside the fleet given: exhaustion cannot be counted"
        tier = tier_of[r.reply.instance_id]
        prompt_cost = r.prompt_tokens * tier.cost_in
        unaffordable = r.budget_cost - prompt_cost <= 0
        spent = prompt_cost + r.reply.output_tokens * tier.cost_out >= r.budget_cost - 1e-9
        if unaffordable or spent:
            n += 1
    return n, None


# the problems of a cell whose prompts have no scored answers (a latency-sweep run's training prompts, by design)
NO_QUALITY = ("successful requests have a prompt absent from the scored corpora",
              "no successful request has a judged score")

def summarize(cell_id: str, outcomes: Sequence[Outcome], duration_s: float, quality: ScoredCorpus | ScoreStore,
              prices: Mapping[str, float], slo: Optional[CellSlo], problems: Sequence[str],
              input_prices: Optional[Mapping[str, float]] = None) -> CellSummary:
    """`problems` are the caller's execution findings (integrity_problems for a cell record); the metric
    problems (failed requests, unmatched prompts, models the corpus never scores, unpriced models, mixed timings) are added
    here."""
    if not duration_s > 0:
        raise ConfigError(f"cell {cell_id}: duration {duration_s} is not positive")
    problems = list(problems)
    ok = [o for o in outcomes if o.ok]
    n_failed = len(outcomes) - len(ok)
    if n_failed:
        problems.append(f"{n_failed}/{len(outcomes)} requests failed")
    if not ok:
        problems.append("no successful requests")

    scores: list[float] = []
    found = {"matched": 0, "corpus_gap": 0, "unmatched": 0, "unscored": 0}
    for o in ok:
        outcome, score = quality.lookup(o.prompt, o.model, o.text)
        found[outcome] += 1
        if outcome == "matched":
            scores.append(score)
    if found["unscored"]:
        problems.append(f"{found['unscored']}/{len(ok)} served texts have no judged score: the cell's quality is "
                        "unavailable until they are judged (D67)")
        scores = []                        # never a mean over the scored rest
    if found["unmatched"]:
        problems.append(f"{found['unmatched']}/{len(ok)} {NO_QUALITY[0]}")
    if ok and not found["matched"]:
        problems.append(NO_QUALITY[1])
    # a model name the corpus never scores (a naming mismatch); the store's texts are checked one by one instead
    unknown = sorted({o.model for o in ok} - quality.models) if isinstance(quality, ScoredCorpus) else []
    if unknown:
        problems.append(f"served models that the corpus scores for no prompt: {unknown}")
    nonzero = [s for s in scores if s != 0.0]

    unpriced = sorted({o.model for o in ok if o.model not in prices})
    if unpriced:
        problems.append(f"served models without a price in the fleet: {unpriced}")
        cost = None
    else:
        cost = math.fsum(o.output_tokens * prices[o.model] for o in ok) / len(ok) * 1000 if ok else None
    # D117: the prompt at its input price as well (vLLM's served count), when the caller gives input prices
    total = None
    if (input_prices is not None and ok and not unpriced and all(o.model in input_prices for o in ok)
            and all(o.prompt_tokens_served is not None for o in ok)):
        total = math.fsum(o.prompt_tokens_served * input_prices[o.model] + o.output_tokens * prices[o.model]
                          for o in ok) / len(ok) * 1000

    overhead, overhead_problem = _overhead(ok)
    if overhead_problem:
        problems.append(overhead_problem)
    outside = [] if slo is None else sorted({o.instance_id for o in ok} - slo.instances())
    if outside:
        problems.append(f"served instances outside the fleet the SLO was built for: {outside}")
    untimed = sum(o.first_token_s is None for o in ok)
    if slo is not None and untimed:
        problems.append(f"{untimed}/{len(ok)} replies carry no scheduler first-token time: the SLO cannot be judged")
    attainment = {level: None if slo is None or outside or untimed or not outcomes else
                  sum(slo.met(level, o.instance_id, o.first_token_s, o.server_latency_s, o.ttft_s, o.output_tokens)
                      for o in ok) / len(outcomes)
                  for level in LEVELS}
    residual = [o.e2e_s - o.server_latency_s for o in ok if o.server_latency_s is not None]
    counted = [o for o in ok if o.prompt_tokens_sent is not None and o.prompt_tokens_served is not None]
    mismatched = sum(o.prompt_tokens_sent != o.prompt_tokens_served for o in counted)
    if mismatched:
        problems.append(f"{mismatched}/{len(counted)} requests: the backend prefilled a different number of prompt "
                        "tokens than the client sent")
    return CellSummary(
        cell_id=cell_id, n_requests=len(outcomes), n_ok=len(ok), n_failed=n_failed,
        quality=math.fsum(scores) / len(scores) if scores else None,
        matched=found["matched"], corpus_gap=found["corpus_gap"], unscored=found["unscored"], budget_exhausted=None,
        unmatched=found["unmatched"], n_zero=len(scores) - len(nonzero),
        coverage=found["matched"] / len(ok) if ok else None,
        quality_zeros_excluded=math.fsum(nonzero) / len(nonzero) if nonzero else None,
        cost_decode_per_1k=cost, throughput_rps=len(ok) / duration_s,
        e2e_ms=spread([o.e2e_s for o in ok], 1000.0) if ok else None,
        slo_attainment_tight=attainment["tight"], slo_attainment_loose=attainment["loose"], overhead_ms=overhead,
        off_instance_ms=spread(residual, 1000.0) if ok and len(residual) == len(ok) else None,
        prompt_token_mismatches=mismatched if ok and len(counted) == len(ok) else None,
        problems=tuple(problems), cost_total_per_1k=total)


def _overhead(ok: Sequence[Outcome]) -> tuple[dict[str, Spread], Optional[str]]:
    timed = [o.timings_ms for o in ok if o.timings_ms is not None]
    if not timed:
        return {}, None
    keys = set(timed[0])
    if len(timed) != len(ok) or any(set(t) != keys for t in timed):
        return {}, "scheduler timing keys differ between requests"
    return {k: spread([t[k] for t in timed], 1.0) for k in sorted(keys)}, None
