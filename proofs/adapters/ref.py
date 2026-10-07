"""The ref branch's interfaces, where they differ from the code that ran (adapters/submission.py): RouteBalance's
round no longer takes the pending-decode mode (own_length is the only one), and the run's verdict checks live in the
runner (runner/store.py) instead of tools/cell_driver.py. Every result is the same plain data."""
from __future__ import annotations

from pathlib import Path

from routebalance.schedule.assign import Admission, schedule_round

from proofs.adapters import submission
from proofs.adapters.submission import gate   # noqa: F401  the run gates are called as before


class Code(submission.Code):
    def _round(self, requests, loads, last_assigned, clock) -> list:
        cfg = self.cfg
        return schedule_round(requests, self.fleet, loads, self.comps.latency, weights=cfg.weights,
                              normalization=cfg.latency.normalization, normalize_over=cfg.normalize_over, lpt=cfg.lpt,
                              last_assigned=last_assigned, clock=clock,
                              admission=Admission() if cfg.budget.admission_filter else None, on_decision=None)


_SCOPES: dict = {}


def verdict_problems(render: Path, run: Path) -> list[str]:
    """The record and engine checks of the run's verdict, recomputed (runner/store.py verdict, without its writing)."""
    from runner import render as rd, store
    if render not in _SCOPES:
        _SCOPES[render] = rd.load(render)[0]
    scope = _SCOPES[render]
    return store.record_problems(scope, rd.expected(scope, run.name), run) + store.engine_problems(run)
