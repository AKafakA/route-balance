"""The interfaces of the code that ran (submission 984eb1a), used exactly as the server uses them.

Every result is plain data with each float written exactly (float.hex), so two trees' results compare byte for byte.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Mapping, MutableMapping, Sequence

from routebalance.baselines.dispatch import build_dispatcher
from routebalance.baselines.pipeline import _candidates
from routebalance.baselines.routers import ALL_INSTANCES, RouterRequest
from routebalance.config import load_fleet, load_scheduler
from routebalance.estimate.knn import Estimate
from routebalance.estimate.prompt import unwrap_chat_prompt
from routebalance.schedule.assign import Admission, RoundRequest, schedule_round
from routebalance.server import P_UNDER_BUDGET_TOKENS, load_components
from routebalance.telemetry import InstanceStats, Load

from proofs.archive import hexf


class Code:
    """One configuration's components, built as the server builds them (load_components)."""

    ALL = ALL_INSTANCES                       # the passthrough router's decision: every instance is a candidate

    def __init__(self, cfg_path: Path, fleet_path: Path, hosts_path: Path, artifacts: Path, avengers_repo: Path,
                 best_route_repo: Path):
        self.cfg = load_scheduler(Path(cfg_path))
        self.fleet = load_fleet(Path(fleet_path), Path(hosts_path))
        self._lane = ThreadPoolExecutor(max_workers=1)
        self.comps = load_components(self.cfg, self.fleet, artifact_root=Path(artifacts), avengers_repo=avengers_repo,
                                     router_lane=self._lane, best_route_repo=best_route_repo)
        self.models = list(dict.fromkeys(i.model for i in self.fleet))       # the routers' pool, in fleet order
        self.probe = self.cfg.telemetry_read == "per_request"

    # ------------------------------------------------------------------ the estimator

    def estimate(self, prompts: Sequence[str]) -> tuple[list, list[dict]]:
        """(the estimator's own estimates, the same as plain data) for one batch, as one estimate_batch call.
        Plain: {model: [length, quality, [bucket probabilities], P(length <= budget)]}, every float exact."""
        raw = self.comps.estimator.estimate_batch(list(prompts), P_UNDER_BUDGET_TOKENS)
        plain = [{m: [hexf(e.length), hexf(e.quality), [hexf(p) for p in e.bucket_probs], hexf(e.p_under_budget)]
                  for m, e in d.items()} for d in raw]
        return raw, plain

    @staticmethod
    def estimates_from_plain(plain: Mapping[str, list]) -> dict:
        """One request's estimates, from the plain form `estimate` writes."""
        return {m: Estimate(quality=float.fromhex(v[1]), length=float.fromhex(v[0]),
                            bucket_probs=tuple(float.fromhex(x) for x in v[2]), p_under_budget=float.fromhex(v[3]))
                for m, v in plain.items()}

    # ------------------------------------------------------------------ RouteBalance's round

    def schedule(self, batch: Sequence[dict], estimates: Sequence, max_tokens: int, loads: Mapping[str, dict],
                 last_assigned: MutableMapping[str, float], clock: Callable[[], float]) -> list[dict]:
        """One round: `batch` the record's requests in the order the round collected them, `estimates` the
        estimator's, `loads` {instance: plain load}. Returns every decision, in decision order."""
        cfg = self.cfg
        requests = [RoundRequest(r["request_id"], r["prompt_tokens"], e,
                                 0.0 if r["budget_cost"] is None else r["budget_cost"], max_tokens)
                    for r, e in zip(batch, estimates, strict=True)]
        decisions = schedule_round(requests, self.fleet, {i: _load(v) for i, v in loads.items()}, self.comps.latency,
                                   weights=cfg.weights, normalization=cfg.latency.normalization,
                                   normalize_over=cfg.normalize_over, lpt=cfg.lpt, pending_decode=cfg.pending_decode,
                                   last_assigned=last_assigned, clock=clock,
                                   admission=Admission() if cfg.budget.admission_filter else None, on_decision=None)
        return [{"req": d.request.request_id, "inst": d.instance.instance_id, "decode": d.decode_tokens,
                 "lpt": hexf(d.lpt_key), "admitted_all": d.admitted_all,
                 "scales": None if d.scales is None else [hexf(d.scales.q_min), hexf(d.scales.q_max),
                                                          hexf(d.scales.c_max), hexf(d.scales.l_max)],
                 "state": _plain_stats(d.state),
                 "scored": [[s.candidate.instance_id, hexf(s.score), hexf(s.quality_term), hexf(s.cost_term),
                             hexf(s.latency_term), hexf(s.candidate.latency_s)] for s in d.scored]}
                for d in decisions]

    # ------------------------------------------------------------------ the pipeline

    def route(self, prompt: str, request_id: str, prompt_tokens: int) -> list:
        """The router's decision for one request, as the pipeline asks for it."""
        d = asyncio.run(self.comps.router.choose_model(
            RouterRequest(unwrap_chat_prompt(prompt), request_id, prompt_tokens), self.models))
        return [d.model, hexf(d.score), d.reason]

    def dispatcher(self):
        """A new dispatcher, as each run's new scheduler process built it."""
        return build_dispatcher(self.cfg.pipeline.dispatch)

    def dispatch(self, dispatcher, model: str, loads: Mapping[str, dict]) -> str:
        """The dispatcher's choice among `model`'s instances (ALL: every instance), as the pipeline makes it."""
        candidates = _candidates(self.fleet, model)
        if self.probe:
            probed = dispatcher.probe_set(candidates)
            return dispatcher.choose_probed(probed, {c.instance_id: _load(loads[c.instance_id]) for c in probed}).instance_id
        return dispatcher.choose(candidates, {c.instance_id: _load(loads[c.instance_id]) for c in candidates}).instance_id

    def observe(self, dispatcher, instance_id: str, line: Mapping) -> None:
        """The request's outcome, as the server tells the dispatcher once the request ends."""
        if line["success"]:
            dispatcher.observe_completion(instance_id, line["latency_s"], line["output_tokens"])
        elif line["timed_out"]:
            dispatcher.observe_timeout(instance_id)


def _load(v: Mapping) -> Load:
    s = v["stats"]
    return Load(InstanceStats(s[0], s[1], float.fromhex(s[2]), s[3], s[4], s[5]), float.fromhex(v["age_ms"]),
                v["in_flight"], v["engine_waiting"], None)


def _plain_stats(s: InstanceStats) -> list:
    return [s.num_running, s.num_waiting, hexf(s.kv_cache_utilization), s.num_preemptions, s.max_num_seqs,
            s.pending_decode_tokens]


# ---------------------------------------------------------------- the run evidence (verdict checks, gates)

_SCOPES: dict = {}


def verdict_problems(render: Path, run: Path) -> list[str]:
    """The record and engine checks of the run's verdict, recomputed (tools/cell_driver.py verdict, without its
    writing): the record parsed strictly and checked against the scope, the evidence parsed strictly, the engines,
    the cache reset proof and the traffic check."""
    from tools import cell_driver as cd
    if render not in _SCOPES:
        _SCOPES[render] = cd.load(render)[0]
    scope = _SCOPES[render]
    exp = cd.expected(scope, run.name)
    return cd.record_problems(scope, exp, run) + cd.engine_problems(run)


def gate(name: str, run: Path) -> tuple[int, str]:
    """A run gate on the run's directory, as scripts/run_cell.sh runs it: (exit status, its output)."""
    import contextlib
    import importlib
    import io
    mod = importlib.import_module(f"gates.{name}")
    argv = ["--cell", str(run)] + (["--config", str(run / "config.json")] if name == "arm" else [])
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        rc = mod.main(argv)
    return rc, out.getvalue()
