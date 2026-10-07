"""The ref branch's interfaces, where they differ from the code that ran (adapters/submission.py): RouteBalance's
round no longer takes the pending-decode mode, own_length being the only one. Every result is the same plain data."""
from __future__ import annotations

from routebalance.schedule.assign import Admission, schedule_round

from proofs.adapters import submission
from proofs.adapters.submission import gate, verdict_problems   # noqa: F401  the run evidence reads the same


class Code(submission.Code):
    def _round(self, requests, loads, last_assigned, clock) -> list:
        cfg = self.cfg
        return schedule_round(requests, self.fleet, loads, self.comps.latency, weights=cfg.weights,
                              normalization=cfg.latency.normalization, normalize_over=cfg.normalize_over, lpt=cfg.lpt,
                              last_assigned=last_assigned, clock=clock,
                              admission=Admission() if cfg.budget.admission_filter else None, on_decision=None)
