"""Integrity of one cell record, checked on what executed rather than on what was asked for.

A cell is invalid (kept and counted, never dropped) when any request failed (counted by analysis/metrics.py
and analysis/manifest.py), or when integrity_problems finds that: the request list is not the trace's, the
scheduler's configuration changed or could not be read at the end, the connection pool made a request wait
(the load was then not open loop), or the recorded arrivals or prompt order do not match the digests they
claim (the arrivals are also regenerated from the cell's own trace and seed).
"""
from __future__ import annotations

from bench.arrivals import arrival_times, schedule_digest
from bench.dataset import order_digest
from bench.records import CellRecord


def failed_requests(cell: CellRecord) -> int:
    return sum(r.reply is None for r in cell.requests)


def integrity_problems(cell: CellRecord) -> list[str]:
    """Every execution problem except failed requests."""
    reqs = cell.requests
    problems: list[str] = []
    if len(reqs) != cell.trace.n_requests:
        problems.append(f"{len(reqs)} requests recorded, the trace has {cell.trace.n_requests}")
    if [r.index for r in reqs] != list(range(len(reqs))):
        problems.append("request indices are not 0..n-1 in order")
    if cell.identity_end is None:
        problems.append(f"the scheduler identity could not be read after the run: {cell.identity_end_error}")
    elif cell.identity_end != cell.identity_start:
        problems.append("the scheduler configuration or fleet changed during the cell")
    if cell.connection_waits:
        problems.append(f"{cell.connection_waits} requests waited for a pooled connection: not open loop")
    if schedule_digest([r.arrival_s for r in reqs]) != cell.arrival_digest:
        problems.append("recorded arrival times do not match the arrival digest")
    if schedule_digest(arrival_times(cell.trace, cell.seed)) != cell.arrival_digest:
        problems.append("the arrival digest is not the schedule of the cell's trace and seed")
    if order_digest([r.prompt_row for r in reqs]) != cell.prompt_order_digest:
        problems.append("recorded prompt rows do not match the prompt-order digest")
    mismatched = sum(r.reply is not None and r.reply.prompt_tokens != r.prompt_tokens for r in reqs)
    if mismatched:   # D34 (1): checked when the cell runs, not only in the analysis
        problems.append(f"{mismatched} requests: the backend prefilled a different number of prompt tokens than the "
                        "client sent (D34)")
    return problems
