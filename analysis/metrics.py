"""Per-cell summary: quality, cost, throughput, latency, SLO goodput and scheduler overhead.

Every metric is over the cell's successful requests. A cell with a failed request is still summarised and
counted, but it carries a problem and is invalid (analysis/validity.py adds the execution checks).
  quality     zero-safe mean judged score of the served responses (analysis/quality.py); a 0.0 counts. It is
              the score of the text actually served (D39): one successful request whose text has no score in
              the store makes the cell's quality unavailable (None, and a problem), never a mean over the rest.
              `quality_zeros_excluded` (mean over the non-zero scores) is a labelled secondary scale kept only
              to compare with older numbers; it is never used for a claim
  coverage    matched / successful. A served model that the scores cover for no text is a problem (a model-name
              mismatch). The v3 re-aggregation (V3PairCorpus) also has corpus gaps, counted and not problems,
              and unmatched prompts, which are problems
  cost        decode only: sum(output_tokens * output price of the served model) / successful * 1000, i.e.
              USD per 1k successful requests, with prices from the fleet config (USD per token)
  throughput  successful requests / duration_s
  e2e         client end-to-end latency, mean / p50 / p99 in ms (numpy, linear-interpolated percentiles)
  goodput     successful requests that met the preregistered SLO (analysis/rules.py), per second of duration_s
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
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

from analysis.quality import QualitySource, ScoreStore
from analysis.rules import GoodputSlo
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
    text: Optional[str]                   # the served response (generated_text)
    output_tokens: Optional[int]
    e2e_s: float
    server_latency_s: Optional[float]
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
    unscored: int                             # served texts without a score: quality unavailable (D39)
    corpus_gap: int                           # V3PairCorpus only
    unmatched: int                            # V3PairCorpus only
    n_zero: int
    coverage: Optional[float]
    quality_zeros_excluded: Optional[float]   # secondary scale, never for claims
    cost_decode_per_1k: Optional[float]
    throughput_rps: float
    e2e_ms: Optional[Spread]
    slo_met: int
    goodput_rps: float
    overhead_ms: dict[str, Spread]
    off_instance_ms: Optional[Spread]
    prompt_token_mismatches: Optional[int]    # sent count != vLLM's count; None when not recorded
    problems: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.problems


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
            out.append(Outcome(prompt=r.prompt, ok=False, model=None, text=None, output_tokens=None, e2e_s=r.e2e_s,
                               server_latency_s=None, timings_ms=None, prompt_tokens_sent=r.prompt_tokens,
                               prompt_tokens_served=None))
        else:
            out.append(Outcome(prompt=r.prompt, ok=True, model=rep.model, text=rep.generated_text,
                               output_tokens=rep.output_tokens,
                               e2e_s=r.e2e_s, server_latency_s=rep.server_latency_s, timings_ms=rep.timings_ms,
                               prompt_tokens_sent=r.prompt_tokens, prompt_tokens_served=rep.prompt_tokens))
    return out


def summarize_cell(cell: CellRecord, store: ScoreStore, fleet: Sequence[Instance], slo: GoodputSlo) -> CellSummary:
    """A campaign cell: its quality is the score of each text actually served, from the score store (D39)."""
    if not isinstance(store, ScoreStore):
        raise ConfigError("a cell's quality comes from the served-response score store (D39), never from a "
                          "(prompt, model) corpus")
    problems = integrity_problems(cell)
    if fleet_digest(list(fleet)) != cell.identity_start.fleet_digest:
        problems.append("the fleet given to the analysis is not the fleet the cell ran on")
    return summarize(cell.cell_id, outcomes_of(cell), cell.duration_s, store,
                     decode_prices(i.tier for i in fleet), slo, problems)


def summarize(cell_id: str, outcomes: Sequence[Outcome], duration_s: float, quality: QualitySource,
              prices: Mapping[str, float], slo: GoodputSlo, problems: Sequence[str]) -> CellSummary:
    """`problems` are the caller's execution findings (integrity_problems for a cell record); the metric
    problems (failed requests, unscored texts, unmatched prompts, unpriced models, mixed timings) are added
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
    found = {"matched": 0, "unscored": 0, "corpus_gap": 0, "unmatched": 0}
    for o in ok:
        outcome, score = quality.lookup(o.prompt, o.model, o.text)
        found[outcome] += 1
        if outcome == "matched":
            scores.append(score)
    if found["unscored"]:
        problems.append(f"{found['unscored']}/{len(ok)} served texts have no judged score: the cell's quality is "
                        "unavailable until they are judged (D39)")
        scores = []                        # never a mean over the scored rest
    if found["unmatched"]:
        problems.append(f"{found['unmatched']}/{len(ok)} successful requests have a prompt absent from the scored "
                        "corpora")
    if ok and not found["matched"]:
        problems.append("no successful request has a judged score")
    unknown = sorted({o.model for o in ok} - quality.models)
    if unknown:
        problems.append(f"served models that the scores cover for no text: {unknown}")
    nonzero = [s for s in scores if s != 0.0]

    unpriced = sorted({o.model for o in ok if o.model not in prices})
    if unpriced:
        problems.append(f"served models without a price in the fleet: {unpriced}")
        cost = None
    else:
        cost = math.fsum(o.output_tokens * prices[o.model] for o in ok) / len(ok) * 1000 if ok else None

    overhead, overhead_problem = _overhead(ok)
    if overhead_problem:
        problems.append(overhead_problem)
    slo_met = sum(slo.met(o.e2e_s, o.output_tokens) for o in ok)
    residual = [o.e2e_s - o.server_latency_s for o in ok if o.server_latency_s is not None]
    counted = [o for o in ok if o.prompt_tokens_sent is not None and o.prompt_tokens_served is not None]
    mismatched = sum(o.prompt_tokens_sent != o.prompt_tokens_served for o in counted)
    if mismatched:
        problems.append(f"{mismatched}/{len(counted)} requests: the backend prefilled a different number of prompt "
                        "tokens than the client sent")
    return CellSummary(
        cell_id=cell_id, n_requests=len(outcomes), n_ok=len(ok), n_failed=n_failed,
        quality=math.fsum(scores) / len(scores) if scores else None,
        matched=found["matched"], unscored=found["unscored"], corpus_gap=found["corpus_gap"],
        unmatched=found["unmatched"], n_zero=len(scores) - len(nonzero),
        coverage=found["matched"] / len(ok) if ok else None,
        quality_zeros_excluded=math.fsum(nonzero) / len(nonzero) if nonzero else None,
        cost_decode_per_1k=cost, throughput_rps=len(ok) / duration_s,
        e2e_ms=spread([o.e2e_s for o in ok], 1000.0) if ok else None,
        slo_met=slo_met, goodput_rps=slo_met / duration_s, overhead_ms=overhead,
        off_instance_ms=spread(residual, 1000.0) if ok and len(residual) == len(ok) else None,
        prompt_token_mismatches=mismatched if ok and len(counted) == len(ok) else None,
        problems=tuple(problems))


def _overhead(ok: Sequence[Outcome]) -> tuple[dict[str, Spread], Optional[str]]:
    timed = [o.timings_ms for o in ok if o.timings_ms is not None]
    if not timed:
        return {}, None
    keys = set(timed[0])
    if len(timed) != len(ok) or any(set(t) != keys for t in timed):
        return {}, "scheduler timing keys differ between requests"
    return {k: spread([t[k] for t in timed], 1.0) for k in sorted(keys)}, None
