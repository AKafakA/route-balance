"""RouteBalance serving: requests are collected into rounds (schedule/rounds.py); each round's prompts are estimated
in one batch, every instance's load is read once, and the round is scheduled (schedule/assign.py), each decision
dispatched as soon as it is made, while the rest of the round is still being scored."""
from __future__ import annotations

import asyncio
import functools
import time
from concurrent.futures import Executor
from dataclasses import asdict, dataclass
from typing import Optional

from routebalance.components import P_UNDER_BUDGET_TOKENS, Components, longest
from routebalance.config import SchedulerConfig
from routebalance.request import Assignment, Completion
from routebalance.schedule.assign import Admission, Decision, RoundRequest, schedule_round
from routebalance.schedule.rounds import RoundCollector
from routebalance.telemetry import TelemetryView
from routebalance.types import Instance


@dataclass
class _Round:
    """One scheduling round as its dispatches see it: its number, size and limits, its stage times, and the
    waiting futures by request object. schedule_ms of a request runs from the load read to its own decision."""
    number: int
    size: int
    limits: tuple
    backlog: int       # requests already waiting in the scheduler when the round opened
    closed: float
    estimated: float
    fetched: float
    waiting: dict
    dispatched: int = 0   # default-ok: counts the round's dispatches as they happen
    ages: Optional[dict] = None   # default-ok: instance -> age (ms) of the snapshot the round read (counted)
    errors: Optional[dict] = None   # default-ok: instance -> its last scrape's failure, when the round used an older one


class RouteBalance:
    def __init__(self, cfg: SchedulerConfig, fleet: list[Instance], view: TelemetryView, comps: Components,
                 executor: Executor):
        self._cfg, self._fleet, self._view, self._c, self._executor = cfg, fleet, view, comps, executor
        self.collector: RoundCollector = RoundCollector(cfg.rounds)
        self.last_assigned: dict[str, float] = {}      # for this configuration's lifetime (the LRU tie-break)
        self.rounds = 0
        self._dispatch_error: Optional[BaseException] = None

    def submit(self, c: Completion) -> asyncio.Future:
        fut = asyncio.get_running_loop().create_future()
        self.collector.submit((c, fut, time.perf_counter()))
        return fut

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            batch = await self.collector.next_round()
            closed = time.perf_counter()
            limits, backlog = self.collector.limits, self.collector.backlog   # this round's; adapt() sets the next
            try:
                ests = await loop.run_in_executor(self._executor, self._c.estimator.estimate_batch,
                                                  [c.prompt for c, _, _ in batch], P_UNDER_BUDGET_TOKENS)
                estimated = time.perf_counter()
                if self._cfg.telemetry_read == "per_round":   # the per-batch load fetch; raises if any scrape failed
                    loads = await self._view.fetch_round()
                else:                                          # the cache control: the pollers' cache
                    loads = self._view.loads([i.instance_id for i in self._fleet])
                fetched = time.perf_counter()
                self.collector.adapt(loads)
                requests = [RoundRequest(c.request_id, c.prompt_tokens, e,
                                         0.0 if c.budget_cost is None else c.budget_cost, c.max_tokens)
                            for (c, _, _), e in zip(batch, ests, strict=True)]
                cfg = self._cfg
                admission = Admission() if cfg.budget.admission_filter else None
                self.rounds += 1                  # numbered before any dispatch; a round that fails leaves a gap
                rnd = _Round(self.rounds, len(batch), limits, backlog, closed, estimated, fetched,
                             {id(r): (fut, arrived) for r, (_, fut, arrived) in zip(requests, batch, strict=True)},
                             ages={i: round(ld.age_ms, 1) for i, ld in loads.items()},
                             errors={i: ld.error for i, ld in loads.items() if ld.error is not None})

                def decided(d: Decision, rnd: _Round = rnd) -> None:
                    """On the pool's thread, right after each decision: dispatch it on the loop now (v3's
                    per-pick dispatch), while the rest of the round is still being scored."""
                    n_feature = longest(d.request.estimates)   # may raise: this request is then not dispatched
                    loop.call_soon_threadsafe(self._dispatch, rnd, d, n_feature, time.perf_counter())

                await loop.run_in_executor(self._executor, functools.partial(
                    schedule_round, requests, self._fleet, loads, self._c.latency, weights=cfg.weights,
                    normalization=cfg.latency.normalization, normalize_over=cfg.normalize_over, lpt=cfg.lpt,
                    last_assigned=self.last_assigned, clock=time.monotonic, admission=admission,
                    on_decision=decided))
            except Exception as e:     # the round fails loudly: every request not yet dispatched gets the error
                for _, fut, _ in batch:
                    if not fut.done():
                        fut.set_exception(e)
            # the dispatch callbacks were queued before the round's result, so they have all run by now
            if self._dispatch_error is not None:
                raise self._dispatch_error

    def _dispatch(self, rnd: "_Round", d: Decision, n_feature: int, decided: float) -> None:
        """On the loop: record the dispatch in the view and hand the request its instance. A failure here fails
        the request and then the round loop (run() re-raises it), as the dispatch loop did before per-pick dispatch."""
        fut, arrived = rnd.waiting[id(d.request)]          # by object: duplicate request ids cannot collide
        try:
            pos = rnd.dispatched
            rnd.dispatched += 1
            token = self._view.record_dispatch(d.instance.instance_id, d.decode_tokens)
            best = next(s for s in d.scored if s.candidate.instance_id == d.instance.instance_id)
            evidence = {"round": rnd.number, "round_size": rnd.size, "round_pos": pos,
                        "round_max_batch": rnd.limits[0], "round_timeout_ms": rnd.limits[1], "round_backlog": rnd.backlog,
                        "telemetry_age_ms": rnd.ages[d.instance.instance_id],
                        "telemetry_error": rnd.errors.get(d.instance.instance_id),
                        "decode_tokens": d.decode_tokens, "lpt_key": d.lpt_key, "feature_tokens": n_feature,
                        "lengths": {m: e.length for m, e in d.request.estimates.items()},
                        "admitted_all": d.admitted_all, "score": best.score,
                        "candidates": [{"inst": s.candidate.instance_id, "score": s.score,
                                        "q": s.quality_term, "c": s.cost_term, "l": s.latency_term,
                                        "lat_s": s.candidate.latency_s} for s in d.scored],
                        "scales": None if d.scales is None else asdict(d.scales)}    # the round's scales
            ms = 1000.0
            timings = {"round_wait_ms": (rnd.closed - arrived) * ms, "estimator_ms": (rnd.estimated - rnd.closed) * ms,
                       "stats_fetch_ms": (rnd.fetched - rnd.estimated) * ms, "schedule_ms": (decided - rnd.fetched) * ms}
            if not fut.done():
                fut.set_result(Assignment(d.instance, token, d.state, evidence, timings))
            else:              # the client went away while the round was scheduled
                self._view.record_failure(token)
        except Exception as e:
            self._dispatch_error = e
            if not fut.done():
                fut.set_exception(e)

    def fail_pending(self, exc: BaseException) -> None:
        """The round loop is gone: every request still waiting for a round gets the error."""
        for _, fut, _ in self.collector.drain():
            if not fut.done():
                fut.set_exception(exc)
