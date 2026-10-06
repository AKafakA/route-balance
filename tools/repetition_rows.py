"""A family's repetition row (seeds 2-3) from its complete seed-1 grid.

    python -m tools.repetition_rows --report REPORT.json --scope configs/scope.json --family qwen|gemma --out SCOPE.json

At each of the family's main rates the row repeats:
  (b) every configuration that no configuration of any system dominates on (quality, mean E2E, total cost per 1,000
      requests): at least its quality, at most its latency and cost, and better in one;
  (c) each system's knee: its configuration on its own frontier closest to its best corner, each axis scaled to [0, 1]
      over that system's configurations at that rate (Euclidean distance to the best quality, the lowest mean E2E and
      the lowest cost); a system with one configuration is its own knee;
  RouteBalance's variants at all eight presets (in-batch on both families, per request on Qwen and RB-decoupled
  on both), and the thesis points (RB-decoupled at 50/30/20 and uniform at
  12, 30 and 42 req/s, Qwen). A system is a router on a dispatcher, a dispatcher alone, RB-decoupled, RouteBalance
  in-batch, or RouteBalance per request. RouteBalance's configurations outside the eight presets and the as-shipped
  copies are neither candidates nor dominators.

The report is analysis.report's report.json (its "configs": one point per configuration and rate, the seed-1 values).
No seed-1 cell of the family's main group at a main rate may be missing, invalid, failed or left out, no block of
that grid pending (a pending "row/block" is resolved against the scope's blocks), and every point must be seed 1 alone
(a grid not yet complete, or a report that averages seeds, is refused, not read). The row (FAMILIES[family]["row"],
disabled until now) gets one block per rate with the chosen configuration ids, seeds 2 and 3, and is enabled; nothing
else in the scope changes. The chosen points are printed per rate, by system, for review.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

EIGHT = ("uniform", "q60l20c20", "q80l10c10", "q90l05c05", "q05l25c70", "q22l18c60", "q25l05c70", "q50l30c20")  # 8 presets
THESIS = ("q50l30c20", "uniform")                                                                                # thesis
FAMILIES = {
    "qwen": {"row": "W9", "group": "qwen", "dataset": "eval_qwen",
             "traces": ("l12", "l18", "l24", "l30", "l36", "l42"),
             "all_eight": ("rb", "rb-reqnorm", "rb-dec"), "thesis_traces": ("l12", "l30", "l42")},
    "gemma": {"row": "W8r", "group": "gemma", "dataset": "eval_gemma",
              "traces": ("gl12", "gl30", "gl42"),
              "all_eight": ("rb", "rb-dec"), "thesis_traces": ()},
}
SEEDS = (2, 3)


class RowError(Exception):
    pass


def system(family: str, cid: str) -> str:
    """The system of a configuration id: rb, rb-reqnorm, rb-dec, disp.<d>, or <router>.<dispatcher>."""
    c = cid[len(family) + 1:] if cid.startswith(family + ".") else None
    if c is None:
        raise RowError(f"{cid}: not a {family} configuration")
    if c.startswith("rb-static-reqnorm."):
        return "rb-reqnorm"
    if c.startswith("rb-static-dec."):
        return "rb-dec"
    if c.startswith("rb-static."):
        return "rb"
    if c.startswith("disp."):
        return c
    parts = c.split(".")
    if len(parts) != 3 or parts[0] not in ("ap", "br"):
        raise RowError(f"{cid}: not a configuration D171 knows (RouteBalance, a router on a dispatcher, a dispatcher)")
    return f"{parts[0]}.{parts[2]}"


def preset(cid: str) -> str:
    return cid.rsplit(".", 1)[-1]


def candidate(family: str, cid: str) -> bool:
    if "shipped" in cid:
        return False
    return preset(cid) in EIGHT if system(family, cid).startswith("rb") else True


def dominates(a: tuple, b: tuple) -> bool:
    """a = (quality, e2e, cost) dominates b: at least b's quality, at most its latency and cost, better in one."""
    no_worse = a[0] >= b[0] and a[1] <= b[1] and a[2] <= b[2]
    better = any((a[0] > b[0], a[1] < b[1], a[2] < b[2]))
    return no_worse and better


def frontier(points: dict) -> set:
    return {n for n, v in points.items() if not any(dominates(o, v) for m, o in points.items() if m != n)}


def knee(points: dict) -> str:
    """The configuration on these points' own frontier closest to their best corner, each axis scaled to [0, 1]."""
    if len(points) == 1:
        return next(iter(points))
    own = sorted(frontier(points))
    lo = [min(v[i] for v in points.values()) for i in range(3)]
    hi = [max(v[i] for v in points.values()) for i in range(3)]

    def scaled(v, i):
        return 0.0 if hi[i] == lo[i] else (v[i] - lo[i]) / (hi[i] - lo[i])
    return min(own, key=lambda n: (math.dist((1 - scaled(points[n], 0), scaled(points[n], 1), scaled(points[n], 2)),
                                             (0.0, 0.0, 0.0)), n))


def choose(family: str, points: dict, trace: str) -> dict[str, set]:
    """{reason: configuration ids} at one rate; points: {cid: (quality, e2e, cost)} of the candidates."""
    rules = FAMILIES[family]
    by_system = defaultdict(dict)                # default-ok: grouping
    for n, v in points.items():
        by_system[system(family, n)][n] = v
    out = {"frontier": frontier(points), "knee": {knee(p) for p in by_system.values()},
           "all eight": {n for n in points if system(family, n) in rules["all_eight"]},
           "thesis": {n for n in points if trace in rules["thesis_traces"] and system(family, n) == "rb-dec"
                      and preset(n) in THESIS}}
    return out


def build(report: dict, scope: dict, family: str) -> tuple[dict, dict]:
    """(the new scope, {trace: {reason: ids}}); refuses an incomplete grid or a row that is not the placeholder."""
    rules = FAMILIES[family]
    cov = report["coverage"]

    def main_seed1(cid: str) -> bool:        # a seed-1 cell of the family's main group at a main rate (no alias prefix)
        parts = cid.split("__")
        return len(parts) == 3 and parts[0].startswith(family + ".") and parts[1] in rules["traces"] and parts[2] == "s1"
    gaps = sorted(c for c in [*cov["missing"], *cov["invalid"], *report["left_out"], *report["failures"],
                              *report["gate_failures"]] if main_seed1(c))
    if gaps:
        raise RowError(f"{len(gaps)} seed-1 cells of the {family} grid are missing, invalid, failed or left out (e.g. "
                       f"{gaps[:3]}): the grid is not complete")
    blocks_by_key = {f"{r['id']}/{b['name']}": b for r in scope["rows"]
                     for b in r.get("blocks", [])}      # default-ok: a calibration row has no blocks
    for key, why in cov["pending"].items():      # "row/block" -> reason: refused when it is the family's main grid
        b = blocks_by_key.get(key)
        if b is None:
            raise RowError(f"the report has a pending block {key} the scope does not hold: {why}")
        seeds = b["seeds"]
        if not isinstance(seeds, list):            # an owner placeholder not decided yet: the grid is not complete
            raise RowError(f"block {key} has seeds {seeds!r}: not decided")
        if b["group"] == rules["group"] and 1 in seeds and set(b["traces"]) & set(rules["traces"]):
            raise RowError(f"the {family} grid has a pending block {key}: {why}")
    pts = defaultdict(dict)                      # default-ok: grouping
    for c in report["configs"]:
        if c["group"] == rules["group"] and c["trace_id"] in rules["traces"]:
            if c.get("seeds") != [1]:                # a configuration's point averages its seeds (analysis/frontier.py)
                raise RowError(f"{c['config_id']} at {c['trace_id']} has seeds {c.get('seeds')}: the rule reads seed 1 "
                               "alone (a report of the seed-1 grid)")
            pts[c["trace_id"]][c["config_id"]] = (c["quality"], c["e2e_mean_ms"], c["cost_total_per_1k"])
    missing = [t for t in rules["traces"] if t not in pts]
    if missing:
        raise RowError(f"the report has no {family} point at {missing}: the seed-1 grid is not complete")
    for t in rules["traces"]:                  # each variant repeated at all eight needs all eight at seed 1
        for s_ in rules["all_eight"]:
            have = {preset(n) for n in pts[t] if system(family, n) == s_}
            if set(EIGHT) - have:
                raise RowError(f"{t}: {s_} has no seed-1 point at the presets {sorted(set(EIGHT) - have)}")
    rows = [r for r in scope["rows"] if r["id"] == rules["row"]]
    if len(rows) != 1:
        raise RowError(f"the scope has {len(rows)} rows {rules['row']}")
    row = rows[0]
    if row.get("enabled"):
        raise RowError(f"{rules['row']} is already enabled: this tool fills the disabled placeholder once")
    order = []                     # the placeholder's rate order (Qwen: 30, 12, 24, 18, 36, 42, as the main grid)
    for b in row["blocks"]:
        for t in b["traces"]:
            if t in rules["traces"] and t not in order:
                order.append(t)
    order += [t for t in rules["traces"] if t not in order]
    chosen, blocks, n_cells = {}, [], 0
    for t in order:
        cand = {n: v for n, v in pts[t].items() if candidate(family, n)}
        reasons = choose(family, cand, t)
        ids = sorted(set().union(*reasons.values()))
        chosen[t] = reasons
        n_cells += len(ids) * len(SEEDS)
        blocks.append({"name": f"{rules['row']}-{t}", "group": rules["group"], "dataset": rules["dataset"], "axes": {},
                       "configs": ids, "traces": [t], "seeds": list(SEEDS), "prompt_order": "per_seed", "fill": False,
                       "alias": None, "planned": {}, "enabled": True, "why": None, "hold": None})
    new = json.loads(json.dumps(scope))
    for r in new["rows"]:
        if r["id"] == rules["row"]:
            r["enabled"], r["why"], r["blocks"] = True, None, blocks
            r["about"] = (f"the repetitions (seeds 2-3) of {family}'s main grid at its points by D171 (the overall "
                          f"frontier and one knee per system per rate, on the grid whose RouteBalance configurations are "
                          f"the eight presets of D192), RouteBalance's variants at all eight presets (D192, D203)"
                          + (" and D178's thesis points" if rules["thesis_traces"] else "")
                          + "; written by tools/repetition_rows.py from the complete seed-1 report")
            r["plan"] = f"{n_cells} cells"
    return new, chosen


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--scope", type=Path, required=True)
    ap.add_argument("--family", choices=sorted(FAMILIES), required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    report = json.loads(a.report.read_text())
    try:
        new, chosen = build(report, json.loads(a.scope.read_text()), a.family)
    except (RowError, KeyError, ValueError, TypeError) as e:
        print(f"refused: {e}", file=sys.stderr)
        return 1
    group = FAMILIES[a.family]["group"]       # a failed parity leaves the group's cells out (refused above)
    state = report["parity"][group] if group in report["parity"] else "not in the report"
    print(f"parity of group {group}: {state}")
    total = 0
    for t, reasons in chosen.items():
        ids = sorted(set().union(*reasons.values()))
        total += len(ids)
        print(f"{t}: {len(ids)} configurations (" + ", ".join(f"{k} {len(v)}" for k, v in reasons.items()) + ")")
        by = defaultdict(list)                   # default-ok: grouping
        for n in ids:
            by[system(a.family, n)].append(n[len(a.family) + 1:])
        for s in sorted(by):
            print(f"   {s}: {' '.join(by[s])}")
    print(f"total: {total} points, {total * len(SEEDS)} runs (seeds {', '.join(map(str, SEEDS))})")
    a.out.write_text(json.dumps(new, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
