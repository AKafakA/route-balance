"""TPOT calibration tables from measured decode latencies.

Input, one of:
  --records FILE.jsonl       one request per line, all lines of one shape; b is the number of requests running on
                             the instance during this request's decode, itself included:
                               {"tier": <tier_id>, "b": <int>, "itl": [s, ...]}   the inter-token gaps
                               {"tier": <tier_id>, "b": <int>, "tpot_s": s, "output_tokens": <int>}
                                                  the request's own per-token decode time, (e2e - ttft) /
                                                  (output_tokens - 1), robust to vLLM merging tokens into one
                                                  streamed delta (tools/calibrate_driver.py writes these, W1)
  --v3-cells CELL.json ... --seed N
                             banked v3 benchmark cells. The tier is <model name after "/", lower-cased>_<gpu>, the
                             GPU from the host's hardware type (routebalance.hw). b is rebuilt as in the O8 replay:
                             arrivals are regenerated (gamma inter-arrivals with the cell's rate and burstiness,
                             seed N, rescaled to span num_prompts / rate), a request runs on its instance from
                             arrival + scheduling_overhead to arrival + e2el (the O8 replay's spans), and b counts
                             the requests running on the instance at the midpoint of that span. Errored requests
                             are left out. (O8 sampled at start + e2el / 2, which falls after the request's own end
                             whenever scheduling_overhead > e2el / 2, 10-44% of requests in the v3 lambda=30 RB
                             cells, filing them one batch size too low.)

One sample per request: the mean of its inter-token latencies after the first (requests with fewer than 4
are skipped), or its tpot_s (every record: defined from 2 output tokens on). The
constant is the median over ALL of a tier's samples, every batch size included (D37); for W1 that is all six
concurrency levels with a fixed number of requests each. Output (--out), the table
routebalance.estimate.latency reads:
  {"unit": "ms_per_token",
   "constant": {tier: median of all the tier's samples},
   "curve": {tier: [[b, running max over b of the medians at b], ...]},   b >= 1 with at least MIN_SAMPLES samples
   "provenance": {"tool", "method", "min_samples", "sources": [{"file", "md5"}], "constant_samples": {tier: n},
                  "curve_samples": {tier: [[b, n], ...]}, "b0_samples": {tier: n},
                  "curve_measured": {tier: [[b, median of the samples at b], ...]}, "curve_rule"}}
A TPOT curve never decreases in b (D47): the median at a batch size can dip below a smaller batch's by sampling
noise, and the scheduler rejects such a curve, so the curve holds the running max of the medians and the
measured medians stay in the provenance.
A request counts itself (its span contains its midpoint), so b = 0 only arises from an empty span or, with
--records, a malformed record; such samples enter the constant but not the curve, and are counted. The table
is checked with the scheduler's own validation before it is written: a value below 1 ms/token is not a TPOT
(client-side ITLs collapse to ~0 when a stalled relay delivers a stream in one burst) and makes the tool fail.

Usage: python -m tools.calibrate_tpot_curve --v3-cells CELLS... --seed 1 --out TABLE.json
       python -m tools.calibrate_tpot_curve --records RECORDS.jsonl --out TABLE.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

from routebalance.estimate.latency import MIN_TPOT_MS, ConstantTPOT, LatencyError, ProfiledCurveTPOT
from routebalance.hw import gpu_of

MIN_SAMPLES = 30
CURVE_RULE = "running max over b of the per-b medians (D47): a TPOT curve never decreases in b"
MIN_ITLS = 4


ITL_KEYS = frozenset({"tier", "b", "itl"})
TPOT_KEYS = frozenset({"tier", "b", "tpot_s", "output_tokens"})


def sample_s(itl: list) -> float | None:
    """A request's TPOT sample (s/token) from its inter-token gaps, or None when it decoded too few tokens."""
    if len(itl) < MIN_ITLS:
        return None
    return float(np.mean(itl[1:]))


def record_sample(rec: dict) -> float | None:
    """The sample of one record of either shape. A tpot_s record is defined from 2 output tokens on (the reader
    refuses fewer), so every one is a sample (D37: the median over all samples)."""
    if set(rec) == TPOT_KEYS:
        return float(rec["tpot_s"])
    return sample_s(rec["itl"])


def read_records(path: Path) -> tuple[list[dict], str]:
    """The records of one file and the method they measure; a malformed or mixed file is refused."""
    records, shapes = [], set()
    with open(path) as f:
        for n, line in enumerate(f, start=1):
            if not line.strip():
                continue
            rec = json.loads(line)
            keys = frozenset(rec) if isinstance(rec, dict) else frozenset()
            if keys not in (ITL_KEYS, TPOT_KEYS):
                raise SystemExit(f"{path}:{n}: a record has exactly the keys {sorted(ITL_KEYS)} or {sorted(TPOT_KEYS)}")
            if not isinstance(rec["tier"], str) or type(rec["b"]) is not int:
                raise SystemExit(f"{path}:{n}: tier must be a string and b an integer")
            if keys == TPOT_KEYS and (type(rec["output_tokens"]) is not int or rec["output_tokens"] < 2
                                      or isinstance(rec["tpot_s"], bool) or not isinstance(rec["tpot_s"], (int, float))
                                      or not np.isfinite(rec["tpot_s"]) or rec["tpot_s"] <= 0):
                raise SystemExit(f"{path}:{n}: tpot_s must be a positive number from at least 2 output tokens")
            shapes.add(keys)
            records.append(rec)
    if len(shapes) > 1:
        raise SystemExit(f"{path}: mixes itl and tpot_s records")
    method = ("measured records: per-request TPOT (e2e - ttft) / (output_tokens - 1)" if shapes == {TPOT_KEYS}
              else "measured records: mean inter-token latency after the first")
    return records, method


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


def non_decreasing(points: list[list]) -> list[list]:
    """[[b, ms], ...] by increasing b, each ms raised to the largest ms at a smaller or equal b (D47)."""
    out, top = [], -math.inf
    for b, ms in points:
        top = max(top, ms)
        out.append([b, top])
    return out


def build_table(records: list[dict], sources: list[Path], method: str) -> dict:
    by_tier: dict[str, list[float]] = {}
    by_point: dict[tuple[str, int], list[float]] = {}
    b0: dict[str, int] = {}
    for rec in records:
        s = record_sample(rec)
        if s is None:
            continue
        tier, b = rec["tier"], int(rec["b"])
        if tier not in by_tier:
            by_tier[tier], b0[tier] = [], 0
        by_tier[tier].append(s)
        if b < 1:
            b0[tier] += 1
            continue
        if (tier, b) not in by_point:
            by_point[(tier, b)] = []
        by_point[(tier, b)].append(s)
    thin = sorted(t for t, v in by_tier.items() if len(v) < MIN_SAMPLES)
    if thin:
        raise SystemExit(f"tiers with fewer than {MIN_SAMPLES} samples: {thin}")
    curve, curve_n, measured = {}, {}, {}
    for tier in sorted(by_tier):
        points = sorted((b, v) for (t, b), v in by_point.items() if t == tier and len(v) >= MIN_SAMPLES)
        if not points:
            raise SystemExit(f"{tier}: no batch size has {MIN_SAMPLES} samples")
        measured[tier] = [[b, 1000.0 * float(np.median(v))] for b, v in points]
        low = [[b, ms] for b, ms in measured[tier] if ms < MIN_TPOT_MS]
        if low:   # checked before the running max, which would lift it above an earlier point and hide it
            raise SystemExit(f"{tier}: measured medians {low} are not a TPOT of at least {MIN_TPOT_MS} ms/token "
                             "(client-side ITLs of a stalled relay collapse to ~0)")
        curve[tier] = non_decreasing(measured[tier])
        curve_n[tier] = [[b, len(v)] for b, v in points]
    return {
        "unit": "ms_per_token",
        "constant": {t: 1000.0 * float(np.median(v)) for t, v in sorted(by_tier.items())},
        "curve": curve,
        "provenance": {"tool": "tools/calibrate_tpot_curve.py", "method": method, "min_samples": MIN_SAMPLES,
                       "sources": [{"file": p.name, "md5": hashlib.md5(p.read_bytes()).hexdigest()}   # input order
                                   for p in sorted(sources, key=lambda p: p.name)],                  # is irrelevant
                       "constant_samples": {t: len(v) for t, v in sorted(by_tier.items())},
                       "curve_samples": curve_n, "b0_samples": dict(sorted(b0.items())),
                       "curve_measured": measured, "curve_rule": CURVE_RULE},
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--records", type=Path, help="JSONL of {tier, b, itl} or {tier, b, tpot_s, output_tokens} per request")
    src.add_argument("--v3-cells", type=Path, nargs="+", help="banked v3 benchmark cells")
    ap.add_argument("--seed", type=int, help="arrival seed of the v3 cells (required with --v3-cells)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; write a new file")
    if args.v3_cells:
        if args.seed is None:
            raise SystemExit("--v3-cells needs --seed")
        records = [r for cell in args.v3_cells for r in v3_cell_records(cell, args.seed)]
        sources, method = args.v3_cells, v3_method(args.seed)
    else:
        records, method = read_records(args.records)
        sources = [args.records]
    table = build_table(records, sources, method)
    try:   # the scheduler's own validation: a point below 1 ms/token is a measurement artifact, not a TPOT
        ConstantTPOT(table["constant"]), ProfiledCurveTPOT(table["curve"])
    except LatencyError as e:
        raise SystemExit(f"refusing to write a table the scheduler would reject: {e}") from e
    args.out.write_text(json.dumps(table, indent=1))
    for tier in table["curve"]:
        pts = "  ".join(f"b{b}:{ms:.1f}" for b, ms in table["curve"][tier])
        print(f"{tier}: constant {table['constant'][tier]:.2f} ms (n={table['provenance']['constant_samples'][tier]})"
              f"; b=0 samples {table['provenance']['b0_samples'][tier]}\n  {pts}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
