"""Pipeline mode: router, then candidates, then shared telemetry, then dispatcher.

Every baseline (and rb_decoupled) selects an instance this way:
  1. the router picks a model from the fleet's models (unique, fleet order), given the user's text;
  2. the candidates are that model's instances in fleet order (every instance for passthrough);
  3. their loads come from the shared TelemetryView, which raises rather than report an idle instance: from
     the pollers' cache, or (probe, v3's per-request arm) from a fetch made now of the candidates the
     dispatcher compares (every candidate; power of two's drawn pair);
  4. the dispatcher picks one candidate on its count (the engines' own gauges in the main grid; the in-flight
     count the view keeps, as a production balancer counts at the proxy, in the appendix row W12; dead reckoning is
     RouteBalance's own), and the dispatch is recorded in the view at once (no await in between): its pending
     decode tokens and in-flight count are tracked for every system, so on the engine gauges concurrent
     selections see each other only through the least-recently-assigned tie-break, as a balancer reading scraped
     engine metrics does, and an in-flight count includes every earlier selection.
The caller owns the returned token: record_progress as tokens stream back, then exactly one of
record_completion or record_failure; it also reports completions and timeouts to the dispatcher.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Mapping, Sequence

from routebalance.estimate.prompt import unwrap_chat_prompt   # the one unwrap, shared with the estimator
from routebalance.baselines.dispatch import Dispatcher
from routebalance.baselines.routers import ALL_INSTANCES, Router, RouterRequest
from routebalance.telemetry import DispatchToken, Load, TelemetryView
from routebalance.types import Instance

def fleet_models(instances: Sequence[Instance]) -> list[str]:
    """The model pool routers choose from: unique models in fleet order."""
    return list(dict.fromkeys(i.model for i in instances))


@dataclass(frozen=True)
class Overhead:
    router_ms: float       # routing, including the wait for a router thread
    telemetry_ms: float    # reading the candidates' loads from the view (with probe: fetching them)
    dispatch_ms: float
    total_ms: float


@dataclass(frozen=True)
class Selection:
    instance: Instance
    model: str             # the model the instance serves
    reason: str            # the router's reason
    score: float           # the router's score
    token: DispatchToken   # the dispatch as recorded in the view
    load: Load             # the chosen instance's load as the dispatcher read it, before this dispatch
    overhead: Overhead


def _candidates(instances: Sequence[Instance], model: str) -> list[Instance]:
    if model == ALL_INSTANCES:
        return list(instances)
    chosen = [i for i in instances if i.model == model]
    if not chosen:
        raise ValueError(f"router chose {model!r}, which no instance serves")
    return chosen


async def select_instance(router: Router, dispatcher: Dispatcher, instances: Sequence[Instance],
                          view: TelemetryView, request: RouterRequest, *,
                          decode_tokens: Mapping[str, int], probe: bool) -> Selection:
    """`request.prompt` is the prompt as the client sent it; the router gets the unwrapped text.
    `decode_tokens` maps every fleet model to the decode length recorded with the dispatch: the request's
    max_tokens for a baseline, as in v3, since baselines predict no length (the latency sweep records the
    estimator's length on each model instead)."""
    if not instances:
        raise ValueError("no instances")
    models = fleet_models(instances)
    bad = {m: decode_tokens.get(m) for m in models
           if not isinstance(decode_tokens.get(m), int) or decode_tokens.get(m) < 1}   # checked before any state moves
    if bad:
        raise ValueError(f"decode_tokens must give every fleet model an int >= 1, got {bad}")
    t0 = time.perf_counter()
    decision = await router.choose_model(RouterRequest(unwrap_chat_prompt(request.prompt), request.request_id,
                                                       request.prompt_tokens),
                                         models)
    t1 = time.perf_counter()
    candidates = _candidates(instances, decision.model)
    if probe:
        probed = dispatcher.probe_set(candidates)
        loads = await view.fetch([c.instance_id for c in probed])       # raises InstanceStateUnavailable
        t2 = time.perf_counter()
        chosen = dispatcher.choose_probed(probed, loads)
    else:
        loads = view.loads([c.instance_id for c in candidates])        # raises InstanceStateUnavailable
        t2 = time.perf_counter()
        chosen = dispatcher.choose(candidates, loads)
    token = view.record_dispatch(chosen.instance_id, decode_tokens[chosen.model])
    t3 = time.perf_counter()
    ms = 1000.0
    return Selection(chosen, chosen.model, decision.reason, decision.score, token, loads[chosen.instance_id],
                     Overhead((t1 - t0) * ms, (t2 - t1) * ms, (t3 - t2) * ms, (t3 - t0) * ms))
