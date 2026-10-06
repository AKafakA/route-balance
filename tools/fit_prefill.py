"""The prefill table: each tier's time to first token as a line in the prompt's token count, fitted on the
single-request level of the latency calibration (tools/calibrate_driver.py's requests.jsonl, level 1: one request in
flight on the instance), where the time to first token is the prompt's prefill and the fixed cost of a request. Only
requests of a held instance-level are used, as for the TPOT table (calibrate_driver.calibration_records): an
instance-level that did not hold its concurrency may have queued, and its TTFT would include the wait.

    python -m tools.fit_prefill --requests W1/requests.jsonl --out calib/prefill_<family>.json [--level 1]

Per tier: ttft_ms = intercept_ms + per_token_ms x prompt_tokens, by least squares over the level's successful measured
requests of held instance-levels (fillers excluded; a measured request without its held flag is refused).
prompt_tokens is vLLM's count of the templated prompt. The engines run with vLLM's prefix cache on, so the line carries
the calibration's average caching; which tokens of a request are cached depends on what its engine served recently,
which the scheduler cannot see, so no cached share is subtracted. A tier with fewer than 20
requests, a non-positive slope or intercept, or a fit explaining less than half the variance is refused.

Output: {"unit": "ms", "prefill": {tier: {"intercept_ms", "per_token_ms", "n", "r2"}}, "provenance": {...}}.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

from routebalance.config import ConfigError

MIN_REQUESTS = 20
MIN_R2 = 0.5


def fit(pairs: list[tuple[int, float]]) -> dict:
    """{intercept_ms, per_token_ms, n, r2} of ttft_ms on prompt_tokens (least squares)."""
    n = len(pairs)
    xs = [float(p) for p, _ in pairs]
    ys = [t for _, t in pairs]
    mx, my = math.fsum(xs) / n, math.fsum(ys) / n
    sxx = math.fsum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        raise ConfigError("every request has the same prompt length: the slope is undefined")
    b = math.fsum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / sxx
    a = my - b * mx
    ss = math.fsum((y - my) ** 2 for y in ys)
    r2 = 1.0 - math.fsum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys, strict=True)) / ss if ss > 0 else 1.0
    return {"intercept_ms": a, "per_token_ms": b, "n": n, "r2": r2}


def build(requests: Path, level: int) -> dict:
    by_tier: dict[str, list[tuple[int, float]]] = defaultdict(list)   # default-ok: grouping
    for lineno, line in enumerate(open(requests, "rb").read().split(b"\n"), 1):
        if not line:
            continue
        r = json.loads(line)
        if r.get("level") != level or r.get("filler") or not r.get("success"):
            continue
        if not isinstance(r.get("held"), bool):
            raise ConfigError(f"{requests}:{lineno}: a measured request needs its instance-level's held flag")
        if not r["held"]:
            continue
        if not isinstance(r.get("prompt_tokens"), int) or not isinstance(r.get("ttft_s"), (int, float)):
            raise ConfigError(f"{requests}:{lineno}: a measured request needs prompt_tokens and ttft_s")
        by_tier[r["tier"]].append((r["prompt_tokens"], 1000.0 * r["ttft_s"]))
    if not by_tier:
        raise ConfigError(f"{requests}: no successful measured request of a held instance-level at level {level}")
    out = {}
    for tier, pairs in sorted(by_tier.items()):
        if len(pairs) < MIN_REQUESTS:
            raise ConfigError(f"tier {tier}: {len(pairs)} requests at level {level}, at least {MIN_REQUESTS} needed")
        f = fit(pairs)
        if f["per_token_ms"] <= 0 or f["intercept_ms"] <= 0 or f["r2"] < MIN_R2:
            raise ConfigError(f"tier {tier}: the fit {f} is not a usable prefill line")
        out[tier] = f
    sha = hashlib.sha256(requests.read_bytes()).hexdigest()
    return {"unit": "ms", "prefill": out,
            "provenance": {"requests": requests.name, "requests_sha256": sha, "level": level,
                           "fit": "least squares, ttft_ms = intercept_ms + per_token_ms x prompt_tokens (D118)"}}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--requests", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--level", type=int, default=1)
    a = ap.parse_args(argv)
    if a.out.exists():
        raise ConfigError(f"{a.out} exists; the table is always a new file")
    doc = build(a.requests, a.level)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(doc, indent=1) + "\n")
    for tier, f in doc["prefill"].items():
        print(f"{tier}: {f['intercept_ms']:.1f} ms + {f['per_token_ms'] * 1000:.1f} us/token (n={f['n']}, r2={f['r2']:.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
