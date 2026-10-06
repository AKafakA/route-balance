"""Each tier's single-request TTFT and TPOT on the lease stack, from W1: the bases of the SLO.

    python -m tools.slo_base --requests W1_OUT/requests.jsonl --table TPOT_TABLE.json --out SLO_BASE.json

TTFT: the median ttft_s of the tier's measured, successful requests at b = 1 from held instance-levels (one
request on the engine at a time; tools/calibrate_driver.py writes requests.jsonl), on the backend client's
clock. TPOT: the calibration table's curve at b = 1 (tools/calibrate_tpot_curve.py: the median per-request TPOT
of the same level). A tier without either is an error. Output (a new file):
  {"unit": "s", "ttft_s": {tier: s}, "tpot_s": {tier: s},
   "provenance": {"tool", "requests": {file, md5}, "table": {file, md5}, "ttft_samples": {tier: n}}}
analysis/slo.py reads it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

from bench.schema import read_jsonl
from routebalance.config import ConfigError, loads_strict


def md5(path: Path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def _is_count(b) -> bool:
    return isinstance(b, int) and not isinstance(b, bool)


def single_request_ttft(lines) -> tuple[dict[str, float], dict[str, int]]:
    """{tier: median ttft_s} over the measured, successful b = 1 requests of held instance-levels."""
    samples: dict[str, list[float]] = {}
    for x in lines:
        if not _is_count(x["b"]):
            raise ConfigError(f"a request of {x['tier']} has concurrency b {x['b']!r}, not a whole number")
        for k in ("filler", "held", "success"):
            if not isinstance(x[k], bool):
                raise ConfigError(f"a b={x['b']} request of {x['tier']} has {k} {x[k]!r}, not a boolean")
        if x["b"] == 1 and not x["filler"] and x["held"] and x["success"]:
            ttft = x["ttft_s"]
            if isinstance(ttft, bool) or not isinstance(ttft, (int, float)) or not (math.isfinite(ttft) and ttft > 0):
                raise ConfigError(f"a measured b=1 request of {x['tier']} has ttft_s {x['ttft_s']!r}")
            samples.setdefault(x["tier"], []).append(float(ttft))   # default-ok: grouping into a new list
    return {t: statistics.median(v) for t, v in sorted(samples.items())}, {t: len(v) for t, v in sorted(samples.items())}


def single_request_tpot(table: dict) -> dict[str, float]:
    """{tier: s/token} from the table's curve at b = 1."""
    if table.get("unit") != "ms_per_token":
        raise ConfigError(f"the TPOT table's unit is {table.get('unit')!r}, not ms_per_token")
    out = {}
    for tier, curve in sorted(table["curve"].items()):
        if not all(_is_count(b) for b, _ in curve):
            raise ConfigError(f"tier {tier}: a TPOT curve point's concurrency is not a whole number")
        at1 = [ms for b, ms in curve if b == 1]
        if len(at1) != 1:
            raise ConfigError(f"tier {tier}: the TPOT curve has no single point at b = 1")
        ms = at1[0]
        if isinstance(ms, bool) or not isinstance(ms, (int, float)) or not (math.isfinite(ms) and ms > 0):
            raise ConfigError(f"tier {tier}: the TPOT curve's b = 1 point {ms!r} is not a positive number")
        out[tier] = ms / 1000.0
    return out


def slo_base(requests: Path, table_path: Path) -> dict:
    ttft, n = single_request_ttft(rec for _, rec in read_jsonl(requests))
    tpot = single_request_tpot(loads_strict(Path(table_path).read_text()))
    if set(ttft) != set(tpot):
        raise ConfigError(f"tiers with a b = 1 TTFT {sorted(ttft)} differ from the table's {sorted(tpot)}")
    return {"unit": "s", "ttft_s": ttft, "tpot_s": tpot,
            "provenance": {"tool": "tools/slo_base.py",
                           "requests": {"file": Path(requests).name, "md5": md5(requests)},
                           "table": {"file": Path(table_path).name, "md5": md5(table_path)},
                           "ttft_samples": n}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--requests", type=Path, required=True, help="W1's requests.jsonl")
    ap.add_argument("--table", type=Path, required=True, help="the TPOT table built from W1's records")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.out.exists():
        print(f"{a.out} exists; the SLO bases are written to a new file", file=sys.stderr)
        return 1
    try:
        base = slo_base(a.requests, a.table)
    except ConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    a.out.write_text(json.dumps(base, indent=1) + "\n")
    print(f"{a.out}: " + ", ".join(f"{t} TTFT {base['ttft_s'][t] * 1000:.1f} ms, TPOT {base['tpot_s'][t] * 1000:.2f} ms"
                                   for t in base["ttft_s"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
