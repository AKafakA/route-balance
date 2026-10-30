"""One F-MRLB scheduling round (paper Algorithm 1).

Input: the round's requests with their per-model estimates, the fleet, and one telemetry read (loads) taken
when the round is scheduled. Every instance of the fleet is a candidate.

  1. LPT: requests are taken longest first by their LPT key, a statistic of their expected lengths over the
     models: lpt="max" (the default), "min" or "mean" (the LPT-key ablations); stable, so equal keys
     keep arrival order; lpt="off" keeps arrival order.
  2. For each request, in that order: estimate its latency on every instance from the instances' current
     state, score (score.py), and take the winner.
  3. Before the next request, the winner's state counts the assignment: one more waiting request and the
     request's predicted decode tokens. So later requests in the round see earlier ones; later rounds see
     them through the telemetry view (record_dispatch).
  4. on_decision, if given, is called with each decision as soon as it is made, before the next request is
     scored: the server dispatches the request then, so no request waits for the rest of its
     round to be scored. The decisions are the same with or without it.

The decode tokens recorded for an assignment are, with pending_decode="own_length", the expected length
of the request on the chosen model; with "lpt_key", its LPT key (the configured statistic; "max" when
LPT is off), truncated to whole tokens and at least 1. The
same number goes to the telemetry view at dispatch.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable, Literal, Mapping, MutableMapping, Optional, Protocol, Sequence

from fmrlb.config import Weights
from fmrlb.schedule import budget
from fmrlb.schedule.score import Candidate, Normalization, Scales, Scored, pick, score
from fmrlb.telemetry import InstanceStats, Load
from fmrlb.types import Instance

PendingDecode = Literal["own_length", "lpt_key"]
NormalizeOver = Literal["request", "round"]   # config.SchedulerConfig.normalize_over
NEVER = 0.0   # the last-assigned time of an instance that has not won a request yet, before any clock value


class Estimate(Protocol):
    quality: float
    length: float
    bucket_probs: tuple[float, ...]


class LatencyModel(Protocol):
    def latencies(self, instances: Sequence[Instance], states: Sequence[InstanceStats],
                  lengths: Mapping[str, float], num_prompt_tokens: int) -> list[float]: ...


@dataclass(frozen=True)
class RoundRequest:
    request_id: str
    num_prompt_tokens: int
    estimates: Mapping[str, Estimate]   # every fleet model -> its estimate for this prompt
    budget_cost: float                  # dollars; 0.0 = no budget
    max_tokens: int                     # the request's output cap

    def __post_init__(self):
        if not (math.isfinite(self.budget_cost) and self.budget_cost >= 0):
            raise ValueError(f"{self.request_id}: budget_cost must be >= 0 dollars, got {self.budget_cost!r}")
        if self.max_tokens < 1:
            raise ValueError(f"{self.request_id}: max_tokens must be >= 1, got {self.max_tokens!r}")


@dataclass(frozen=True)
class Admission:
    """The budget filter: a candidate is admitted when its expected cost is within the request's budget, the
    expected cost being the prompt at the tier's input price plus the model's expected length, capped at
    max_tokens, at its output price. When none is admitted, only the cheapest candidates by worst-case cost (the
    prompt plus max_tokens) remain. Each model's own expected length is used, every cheapest instance is kept,
    and an expected length that truncates to 0 stays 0."""


@dataclass(frozen=True)
class Decision:
    request: RoundRequest
    instance: Instance
    decode_tokens: int                  # predicted decode length recorded for this assignment
    state: InstanceStats                # the chosen instance's state when the request was scored (before it)
    lpt_key: float                      # the request's LPT key under the configured statistic
    scored: tuple[Scored, ...]          # every scored candidate (the decision log's evidence)
    admitted_all: bool                  # False only if a budget filter ran and rejected every candidate
    scales: Optional[Scales] = None     # the round's scales when normalizing over the round, else None


Lpt = Literal["max", "min", "mean", "off"]


def lpt_key(req: RoundRequest, lpt: Lpt) -> float:
    lengths = [e.length for e in req.estimates.values()]
    if lpt == "min":
        return min(lengths)
    if lpt == "mean":
        return sum(lengths) / len(lengths)
    return max(lengths)


def _idle(s: InstanceStats) -> InstanceStats:
    return replace(s, num_running=0, num_waiting=0, kv_cache_utilization=0.0, pending_decode_tokens=0)


def _decode_tokens(length: float) -> int:
    """Whole tokens, at least 1 (the telemetry view needs a positive prediction)."""
    if not (math.isfinite(length) and length >= 0):
        raise ValueError(f"expected length must be finite and >= 0, got {length!r}")
    return max(1, int(length))


def _admitted(inst: Instance, req: RoundRequest) -> bool:
    expected = min(int(req.estimates[inst.model].length), req.max_tokens)
    return budget.request_cost(inst.tier, req.num_prompt_tokens, expected) <= req.budget_cost


def _worst_case(inst: Instance, req: RoundRequest) -> float:
    return budget.request_cost(inst.tier, req.num_prompt_tokens, req.max_tokens)


def schedule_round(requests: Sequence[RoundRequest], instances: Sequence[Instance], loads: Mapping[str, Load],
                   latency: LatencyModel, *, weights: Weights, normalization: Normalization,
                   normalize_over: NormalizeOver, lpt: Lpt, pending_decode: PendingDecode,
                   last_assigned: MutableMapping[str, float], clock: Callable[[], float],
                   admission: Optional[Admission],
                   on_decision: Optional[Callable[[Decision], None]]) -> list[Decision]:
    if not instances:
        raise ValueError("no instances")
    if lpt not in ("max", "min", "mean", "off") or pending_decode not in ("own_length", "lpt_key"):
        raise ValueError(f"unknown lpt {lpt!r} or pending_decode {pending_decode!r}")
    if normalize_over not in ("request", "round"):
        raise ValueError(f"unknown normalize_over {normalize_over!r}")
    fleet_models = {inst.model for inst in instances}
    for req in requests:
        absent = fleet_models - set(req.estimates)
        if absent:
            raise ValueError(f"{req.request_id}: no estimate for models {sorted(absent)}")
    order = sorted(requests, key=lambda r: lpt_key(r, lpt), reverse=True) if lpt != "off" else list(requests)
    states = {inst.instance_id: loads[inst.instance_id].stats for inst in instances}
    decisions = []
    if not requests:
        return decisions
    round_q = round_c = None
    if normalize_over == "round":     # the round's quality range and cost maximum, every request x instance
        qs = [req.estimates[inst.model].quality for req in requests for inst in instances]
        round_q = (min(qs), max(qs))
        round_c = max(req.num_prompt_tokens * inst.tier.cost_in + req.estimates[inst.model].length * inst.tier.cost_out
                      for req in requests for inst in instances)
    l_run = 0.0                       # the largest load-aware latency scored so far in this round
    for req in order:
        lengths = {m: e.length for m, e in req.estimates.items()}
        now = [states[inst.instance_id] for inst in instances]
        lat = latency.latencies(instances, now, lengths, req.num_prompt_tokens)
        base = (latency.latencies(instances, [_idle(s) for s in now], lengths, req.num_prompt_tokens)
                if normalization == "fixed_ref" else [None] * len(instances))
        cands = [Candidate(inst.instance_id, inst.model, req.estimates[inst.model].quality,
                           # the prompt at the input price + the expected output at the output price
                           req.num_prompt_tokens * inst.tier.cost_in + lengths[inst.model] * inst.tier.cost_out,
                           lt, l0, s.num_running + s.num_waiting,
                           last_assigned[inst.instance_id] if inst.instance_id in last_assigned else NEVER)
                 for inst, s, lt, l0 in zip(instances, now, lat, base, strict=True)]
        admitted_all = True
        if admission is not None and req.budget_cost > 0:
            kept = [c for c, inst in zip(cands, instances, strict=True) if _admitted(inst, req)]
            admitted_all = bool(kept)
            if not kept:
                cheapest = min(_worst_case(inst, req) for inst in instances)
                kept = [c for c, inst in zip(cands, instances, strict=True) if _worst_case(inst, req) == cheapest]
            cands = kept
        scales = None
        if normalize_over == "round":
            l_run = max(l_run, max(c.latency_s for c in cands))
            scales = Scales(round_q[0], round_q[1], round_c, l_run)
        scored = score(cands, w_quality=weights.w_quality, w_cost=weights.w_cost, w_latency=weights.w_latency,
                       normalization=normalization, scales=scales)
        best = pick(scored).candidate
        last_assigned[best.instance_id] = clock()
        inst = next(i for i in instances if i.instance_id == best.instance_id)
        key = lpt_key(req, lpt)
        tokens = _decode_tokens(lengths[inst.model] if pending_decode == "own_length" else key)
        st = states[inst.instance_id]
        states[inst.instance_id] = replace(st, num_waiting=st.num_waiting + 1,
                                           pending_decode_tokens=st.pending_decode_tokens + tokens)
        decisions.append(Decision(req, inst, tokens, st, key, tuple(scored), admitted_all, scales))
        if on_decision is not None:
            on_decision(decisions[-1])
    return decisions
