"""The TPOT-table records of banked v3 benchmark cells (the former --v3-cells mode of tools/calibrate_tpot_curve.py,
used only to rebuild the pinned v3-cells table in the equivalence tests).

The tier is <model name after "/", lower-cased>_<gpu>, the GPU from the host's hardware type (routebalance.hw).
b is rebuilt as in the O8 replay: arrivals are regenerated (gamma inter-arrivals with the cell's rate and
burstiness, seed N, rescaled to span num_prompts / rate), a request runs on its instance from arrival +
scheduling_overhead to arrival + e2el (the O8 replay's spans), and b counts the requests running on the
instance at the midpoint of that span. Errored requests are left out. (O8 sampled at start + e2el / 2, which
falls after the request's own end whenever scheduling_overhead > e2el / 2, 10-44% of requests in the v3
lambda=30 RB cells, filing them one batch size too low.)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from routebalance.hw import gpu_of


def v3_arrivals(n: int, rate: float, burstiness: float, seed: int) -> np.ndarray:
    """The v3 benchmark's request times, as the O8 replay regenerates them."""
    np.random.seed(seed)
    theta = 1.0 / (rate * burstiness)
    gaps = [np.random.gamma(shape=burstiness, scale=theta) for _ in range(n)]
    for i in range(1, n):
        gaps[i] += gaps[i - 1]
    scale = (n / rate) / gaps[-1]
    return np.array([t * scale for t in gaps])


def v3_tier(model: str, host: str) -> str:
    return f"{model.split('/')[-1].lower()}_{gpu_of(host.split('-')[0])}"


def v3_method(seed: int) -> str:
    """The provenance method of a table built from v3 cells."""
    return f"v3 cells: O8 span timeline, b at each request's span midpoint, arrival seed {seed}"


def v3_cell_records(path: Path, seed: int) -> list[dict]:
    """(tier, itl, b) records of one banked cell, with b rebuilt from the cell's own timeline."""
    cell = json.loads(path.read_text())
    details = cell["response_details"]
    arrival = v3_arrivals(len(details), float(cell["request_rate"]), float(cell["burstiness"]), seed)
    spans: dict[str, list[tuple[float, float]]] = {}
    ok = [(i, r) for i, r in enumerate(details) if not r["error"]]
    for i, r in ok:
        if r["instance_id"] not in spans:
            spans[r["instance_id"]] = []
        spans[r["instance_id"]].append((arrival[i] + float(r["scheduling_overhead"]), arrival[i] + float(r["e2el"])))
    starts = {k: np.array([s for s, _ in v]) for k, v in spans.items()}
    ends = {k: np.array([e for _, e in v]) for k, v in spans.items()}
    out = []
    for i, r in ok:
        at = arrival[i] + (float(r["scheduling_overhead"]) + float(r["e2el"])) / 2   # the span's midpoint
        inst = r["instance_id"]
        b = int(np.sum((starts[inst] <= at) & (ends[inst] > at)))
        out.append({"tier": v3_tier(r["model"], r["host"]), "itl": r["itl"], "b": b})
    return out
