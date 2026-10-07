"""Arm gate: the values a cell actually used, read from its decision lines (routebalance/service.py).

Usage: python -m gates.arm --cell DIR --config CONFIG.json   (exit 0 on pass, 1 on fail; one reason per line)

CONFIG.json is the scheduler document the driver POSTed for the cell. Its digest must be the cell's: the
record's, the manifest's and every decision line's. The fleet is the one the scheduler executed
(manifest_before.json).

Coverage
  Every request of the cell record has exactly one decision line, and no line belongs to another request.
  The scheduler wrote exactly those lines in the cell's window: stats_after minus stats_before, both read
  with no line buffered and no request in service. So nothing else was served during the cell.
Every line
  Carries the cell's config digest. A dispatched line names a fleet instance and that instance's tier.
No connection bound was hit
  The scheduler's engine and telemetry pools never made a request wait for a connection during the cell
  (stats_after minus stats_before): the bound is set far above any in-flight count, and a cell in which it was
  reached is invalid, to be rerun with a larger max_connections.
RouteBalance (mode routebalance)
  - Every line is a dispatch with its scored candidates: every fleet instance, in fleet order. With the budget
    admission filter (v3's, routebalance/schedule/assign.py Admission) the gate recomputes it from the line's
    request (prompt tokens, max_tokens, budget), its expected lengths and the manifest's prices: the candidates
    are the admitted instances, or, when none is admitted, the instances at the cheapest worst-case cost, in
    fleet order, with admitted_all saying which.
  - The chosen instance is a candidate with the highest score.
  - Every score is w_quality*q + w_cost*c + w_latency*l, bitwise, which checks the weights actually used.
  - fixed_ref: every latency term lies in (0, 1], and term * latency gives the same reference for every
    candidate.
  - max_divide: every latency term is 1 - latency / (max latency + EPS), bitwise, so the slowest is ~0; normalizing
    over the round, the max is the round's running maximum logged with the line (scales.l_max), which covers
    the line's latencies and runs over the round in decision order, and the round's lines share one quality range
    and cost maximum. The scales are logged exactly when the configuration normalizes over the round.
  - Rounds: the lines of a round agree on its size and limits. There are at most `size` of them, and their
    positions are distinct and below the size. The size is at most the round's max_batch. Fixed rounds use the
    configured limits; an adaptive round's window is one of the adaptive pairs' and its max_batch is that pair's
    size, or the requests already waiting when it opened (round_backlog) up to the largest size.
  - LPT: every line's lpt_key is the configured statistic (max, min or mean; off writes the max) of the
    request's expected lengths over every model the estimator returned (the line's lengths). Unless lpt is
    off, within a round in position order the lpt_keys do not increase.
  - feature_tokens (the learned heads' length feature) is the largest expected length, in whole tokens.
  - decode_tokens: the expected length on the chosen model (pending_decode own_length), in whole tokens and at
    least 1.
Pipeline (mode pipeline)
  - Every line is a dispatch carrying the router's reason, prefixed by the configured router type, and a finite
    router score. Where the reason names a tier or model (rb_decoupled, avengers_pro, best_route_upstream), that
    is the one served.
  - decode_tokens and feature_tokens: the request's max_tokens, with no lengths logged. With pending_decode
    own_length (the latency sweep) they are the expected length on the chosen model and the LPT key.
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from collections import Counter
from pathlib import Path

from gates.cell import Cell, FleetEntry, Problems, open_cell, run
from routebalance.config import SchedulerConfig, config_digest, load_scheduler
from routebalance.schedule.score import EPS

# The adaptive round limits (max_batch, timeout_ms) of the paper, written out here so that a changed constant
# in schedule/rounds.py fails this gate.
ADAPTIVE_LIMITS = frozenset({(4, 50.0), (16, 100.0), (32, 200.0)})
BACKLOG_CAP = 32            # an adaptive round takes the scheduler's backlog up to the largest adaptive size
REASON_PREFIX = {"passthrough": "passthrough:", "avengers_pro": "avengers_pro:", "rb_decoupled": "rb_decoupled:",
                 "best_route_upstream": "best_route_upstream:"}
# The router's reason names a model, or (rb_decoupled) the tier whose model it returned: the dispatcher then
# places the request on any instance of that model, whatever its tier.
NAMED = {"rb_decoupled": (re.compile(r"^rb_decoupled:tier=([^:]+):"), "tier"),
         "avengers_pro": (re.compile(r":model=(.+)$"), "model"),
         "best_route_upstream": (re.compile(r":model=(.+)$"), "model")}
REF_RTOL = 1e-9            # fixed_ref: term * latency = the reference, up to float rounding
MEAN_RTOL = 1e-9           # lpt mean: the scheduler's mean of the lengths, up to the order of the float sum
# max_divide: the slowest term is EPS / (max latency + EPS). Every latency is at least 1 ms/token x its length,
# so the term is below 1e-5; a larger value means another normalization ran.
SLOWEST_TERM_MAX = 1e-4


class _Malformed(Exception):
    pass


def whole(tokens: float) -> int:
    """Expected length in whole decode tokens, as the scheduler records it (at least 1)."""
    return max(1, int(tokens))


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _get(line: dict, key: str):
    if key not in line:
        raise _Malformed(f"no {key!r}")
    return line[key]


def arm_problems(cell: Cell, cfg: SchedulerConfig) -> tuple[list[str], str]:
    p = Problems()
    digest = config_digest(cfg)
    rec, man = cell.record, cell.manifest("before")
    if rec.identity_start.config_digest != digest:
        p.add("the cell record's scheduler ran another configuration",
              f"{rec.identity_start.config_digest} != {digest} (the given config)")
    if man.config_digest != digest:
        p.add("manifest_before.json is for another configuration", f"{man.config_digest} != {digest}")
    want = None if cfg.pipeline is None else cfg.pipeline.router.concurrency
    if man.loaded.router_lane_threads != want:
        p.add("the router's calls ran on another number of threads than the configuration's concurrency",
              f"{man.loaded.router_lane_threads} != {want}")
    lines = cell.decisions
    _coverage(p, cell, [r.request_id for r in rec.requests], lines)
    before, after = cell.stats("before").connection_waits, cell.stats("after").connection_waits
    for pool in ("engine", "telemetry"):
        waited = getattr(after, pool) - getattr(before, pool)
        if waited:
            p.add(f"the scheduler's {pool} connection pool made requests wait ({waited} waits): rerun the cell with a "
                  "larger max_connections")
    fleet = man.fleet
    by_id = {f.instance_id: f for f in fleet}
    for line in lines:
        rid = line["request_id"] if "request_id" in line else "<no request_id>"
        if "config_digest" not in line or line["config_digest"] != digest:
            p.add("a decision line carries another config digest", rid)
        if "instance_id" in line:
            inst = by_id[line["instance_id"]] if line["instance_id"] in by_id else None
            if inst is None:
                p.add("a line names an instance outside the fleet", f"{rid}: {line['instance_id']!r}")
            elif "tier" not in line or line["tier"] != inst.tier:
                p.add("a line's tier is not its instance's", rid)
    check = _routebalance if cfg.mode == "routebalance" else _pipeline
    check(p, cfg, fleet, lines, rec.sampling.max_tokens)
    return p.lines(), f"{cell.cell_id}: {len(lines)} {cfg.mode} decision lines use config {digest[:12]} as configured"


def _coverage(p: Problems, cell: Cell, request_ids: list[str], lines: list[dict]) -> None:
    before, after = cell.stats("before"), cell.stats("after")
    for when, s in (("before", before), ("after", after)):
        if s.logged.decisions_buffered:
            p.add(f"stats_{when}.json was read with {s.logged.decisions_buffered} decision lines not yet on disk")
        if s.active:           # a request still in service writes its line later, outside the slice
            p.add(f"stats_{when}.json was read with {s.active} requests in service")
    written = after.logged.decisions - before.logged.decisions
    if written != len(lines):
        p.add("the scheduler's decision count over the cell's window is not the lines in decisions.jsonl",
              f"{written} written, {len(lines)} in the file")
    expected = Counter(request_ids)
    got = Counter(line["request_id"] for line in lines if "request_id" in line)
    if sum(got.values()) != len(lines):
        p.add("a decision line has no request_id")
    for rid in sorted(r for r, n in expected.items() if n > 1):
        p.add("the cell record repeats a request id", rid)
    for rid in sorted(set(expected) - set(got)):
        p.add("a request of the cell has no decision line", rid)
    for rid in sorted(set(got) - set(expected)):
        p.add("a decision line belongs to no request of the cell", rid)
    for rid in sorted(r for r, n in got.items() if n > 1 and r in expected):
        p.add("a request has more than one decision line", rid)


def _lengths(line: dict, models: set[str]) -> dict:
    lengths = _get(line, "lengths")
    if not (isinstance(lengths, dict) and models <= set(lengths) and all(_num(v) and v >= 0 for v in lengths.values())):
        raise _Malformed(f"lengths {lengths!r} do not give every fleet model a length >= 0")
    return lengths


def _lpt_key(p: Problems, rid: str, line: dict, lpt: str, lengths: dict) -> float:
    key = _get(line, "lpt_key")
    if not (_num(key) and key >= 0):
        raise _Malformed(f"lpt_key {key!r} is not a number >= 0")
    values = list(lengths.values())
    if lpt == "mean":
        want = sum(values) / len(values)
        ok = abs(key - want) <= MEAN_RTOL * want
    else:
        want = min(values) if lpt == "min" else max(values)    # "off" writes the max (v3)
        ok = key == want
    if not ok:
        name = "max (lpt off)" if lpt == "off" else lpt
        p.add(f"lpt_key is not the {name} of the request's expected lengths", f"{rid}: {key} != {want}")
    return key


def _decode(p: Problems, rid: str, line: dict, fleet_model: str, lengths: dict) -> None:
    """A line under pending_decode own_length: its feature and decode tokens, from its expected lengths."""
    longest = whole(max(lengths.values()))
    if _get(line, "feature_tokens") != longest:
        p.add("feature_tokens is not the largest expected length in whole tokens (the heads' feature)",
              f"{rid}: {line['feature_tokens']} != {longest}")
    want = whole(lengths[fleet_model])
    if _get(line, "decode_tokens") != want:
        p.add("decode_tokens are not the expected length on the chosen model in whole tokens",
              f"{rid}: {line['decode_tokens']} != {want}")


def _admission(line: dict, fleet: list[FleetEntry], lengths: dict) -> tuple[list[str], bool]:
    """(the candidates v3's budget filter keeps, in fleet order; whether any was admitted), from the line's
    request as sent and the fleet's prices, with the scheduler's arithmetic (routebalance/schedule/budget.py)."""
    req = _get(line, "request")
    if not isinstance(req, dict) or set(req) != {"prompt_tokens", "max_tokens", "budget_cost"} \
            or not _num(req["budget_cost"]) or type(req["prompt_tokens"]) is not int or type(req["max_tokens"]) is not int:
        raise _Malformed(f"request {req!r} does not give the prompt tokens, max_tokens and budget sent")
    p_tok, cap, budget_cost = req["prompt_tokens"], req["max_tokens"], req["budget_cost"]
    admitted = [f.instance_id for f in fleet
                if p_tok * f.cost_in + min(int(lengths[f.model]), cap) * f.cost_out <= budget_cost]
    if admitted:
        return admitted, True
    worst = {f.instance_id: p_tok * f.cost_in + cap * f.cost_out for f in fleet}
    cheapest = min(worst.values())
    return [f.instance_id for f in fleet if worst[f.instance_id] == cheapest], False


def _routebalance(p: Problems, cfg: SchedulerConfig, fleet: list[FleetEntry], lines: list[dict], max_tokens: int) -> None:
    ids = [f.instance_id for f in fleet]
    by_id = {f.instance_id: f for f in fleet}
    model_of = {f.instance_id: f.model for f in fleet}
    models = set(model_of.values())
    w = cfg.weights
    rounds: dict[int, list[dict]] = {}
    for line in lines:
        rid = line["request_id"] if "request_id" in line else "<no request_id>"
        if "candidates" not in line or "instance_id" not in line:
            p.add("a RouteBalance request was never dispatched", f"{rid} ({line['error'] if 'error' in line else ''})")
            continue
        try:
            if "reason" in line:
                p.add("a RouteBalance line carries a router reason", rid)
            cands = _get(line, "candidates")
            insts = [c["inst"] for c in cands]
            filter_ran = cfg.budget.admission_filter and _get(line, "budget") is not None
            admitted_all = _get(line, "admitted_all")
            if not filter_ran:
                if admitted_all is not True:
                    p.add("admitted_all is false but no budget filter ran", rid)
                if insts != ids:
                    p.add("the RouteBalance candidates are not every fleet instance in fleet order", f"{rid}: {insts}")
            else:
                want, fits = _admission(line, fleet, _lengths(line, models))
                if insts != want or admitted_all is not fits:
                    p.add("the candidates are not v3's budget filter's (admitted, else the cheapest by worst-case "
                          "cost)", f"{rid}: {insts} (admitted_all {admitted_all}) != {want} ({fits})")
            _scores(p, rid, line, cands, w)
            scales = _get(line, "scales")
            if (cfg.normalize_over == "round") != (scales is not None):
                p.add("the round's scales are logged exactly when the score normalizes over the round",
                      f"{rid}: normalize_over {cfg.normalize_over}, scales {scales}")
            if scales is not None and not _valid_scales(scales):
                raise _Malformed(f"scales {scales!r} are not four finite numbers q_min <= q_max, c_max, l_max > 0")
            _latency_terms(p, rid, cands, cfg.latency.normalization, scales)
            lengths = _lengths(line, models)
            _cost_terms(p, rid, line, cands, by_id, lengths, scales)
            _lpt_key(p, rid, line, cfg.lpt, lengths)
            if line["instance_id"] in model_of:
                _decode(p, rid, line, model_of[line["instance_id"]], lengths)
            for key in ("round", "round_size", "round_pos", "round_max_batch", "round_backlog"):
                if not isinstance(_get(line, key), int) or isinstance(line[key], bool):
                    raise _Malformed(f"{key} {line[key]!r} is not an integer")
            if not _num(_get(line, "round_timeout_ms")):
                raise _Malformed("round_timeout_ms is not a number")
            rounds.setdefault(line["round"], []).append(line)   # default-ok: grouping
        except (_Malformed, KeyError, TypeError, ValueError) as e:
            p.add("a RouteBalance decision line is malformed", f"{rid}: {e}")
    for r, members in sorted(rounds.items()):
        _round(p, cfg, r, members)
        complete = len(members) == members[0]["round_size"]
        if cfg.normalize_over == "round" and complete and all(_valid_scales(m["scales"]) for m in members):
            # the cost maximum is over every request of the round x every fleet instance (before any filter)
            c_max = max(_cost(m, f, _lengths(m, models)) for m in members for f in fleet)
            if any(m["scales"]["c_max"] != c_max for m in members):
                p.add("a round's cost maximum is not the maximum over its requests x the fleet", f"round {r}")


def _cost(line: dict, f: FleetEntry, lengths: dict) -> float:
    """A candidate's predicted cost as the scheduler computes it (schedule/assign.py)."""
    return line["request"]["prompt_tokens"] * f.cost_in + lengths[f.model] * f.cost_out


def _cost_terms(p: Problems, rid: str, line: dict, cands: list[dict], by_id: dict, lengths: dict, scales) -> None:
    """Every cost term is 1 - cost / (cost maximum + EPS), bitwise: the maximum over this request's candidates, or
    over the round (scales.c_max) when normalizing over the round."""
    costs = [_cost(line, by_id[c["inst"]], lengths) for c in cands]
    c_ref = max(costs) if scales is None else scales["c_max"]
    if scales is not None and not c_ref >= max(costs):
        p.add("the round's cost maximum is below this request's costs", f"{rid}: {c_ref}")
    if any(c["c"] != 1.0 - x / (c_ref + EPS) for c, x in zip(cands, costs, strict=True)):
        p.add("a cost term is not 1 - cost / (cost maximum + EPS)", rid)


def _scores(p: Problems, rid: str, line: dict, cands: list[dict], w) -> None:
    if not cands:
        raise _Malformed("no candidates")
    for c in cands:
        if not all(_num(c[k]) for k in ("score", "q", "c", "l", "lat_s")):
            raise _Malformed(f"candidate {c['inst']}: a term is not a finite number")
        if w.w_quality * c["q"] + w.w_cost * c["c"] + w.w_latency * c["l"] != c["score"]:
            p.add("a score is not the configured weights applied to its terms", f"{rid}/{c['inst']}")
    top = max(c["score"] for c in cands)
    chosen = [c for c in cands if c["inst"] == line["instance_id"]]
    if len(chosen) != 1:
        p.add("the chosen instance is not exactly one candidate", rid)
    elif chosen[0]["score"] != top or _get(line, "score") != top:
        p.add("the chosen instance does not have the highest score", f"{rid}: {chosen[0]['score']} < {top}")


def _valid_scales(scales) -> bool:
    return (isinstance(scales, dict) and set(scales) == {"q_min", "q_max", "c_max", "l_max"}
            and all(_num(scales[k]) for k in scales) and scales["q_min"] <= scales["q_max"]
            and scales["c_max"] >= 0 and scales["l_max"] > 0)


def _latency_terms(p: Problems, rid: str, cands: list[dict], normalization: str, scales) -> None:
    """scales: the round's (normalize_over round): max_divide divides by the round's running maximum, which
    covers this request's latencies; None: by this request's own maximum, so its slowest term is ~0."""
    lats, terms = [c["lat_s"] for c in cands], [c["l"] for c in cands]
    if not all(x > 0 for x in lats):
        raise _Malformed(f"latencies {lats} are not all positive")
    if normalization == "fixed_ref":
        if not all(0 < t <= 1 for t in terms):
            p.add("fixed_ref: a latency term lies outside (0, 1]", f"{rid}: {terms}")
        refs = [t * x for t, x in zip(terms, lats, strict=True)]
        if max(refs) - min(refs) > REF_RTOL * max(refs):
            p.add("fixed_ref: the latency terms do not share one reference latency", f"{rid}: {refs}")
        return
    lmax = max(lats) if scales is None else scales["l_max"]
    if scales is not None and not lmax >= max(lats):
        p.add("max_divide: the round's running latency maximum is below this request's latencies", f"{rid}: {lmax}")
    if any(t != 1.0 - x / (lmax + EPS) for t, x in zip(terms, lats, strict=True)):
        p.add("max_divide: a latency term is not 1 - latency / (max latency + EPS)", f"{rid}: {terms}")
    slowest = terms[lats.index(max(lats))]
    if scales is None and not 0 <= slowest <= SLOWEST_TERM_MAX:
        p.add("max_divide: the slowest candidate's latency term is not ~0", f"{rid}: {slowest}")


def _round(p: Problems, cfg: SchedulerConfig, r: int, members: list[dict]) -> None:
    sizes = {m["round_size"] for m in members}
    limits = {(m["round_max_batch"], m["round_timeout_ms"], m["round_backlog"]) for m in members}
    if len(sizes) != 1 or len(limits) != 1:
        p.add("a round's lines disagree on its size or limits", f"round {r}: sizes {sorted(sizes)}, limits {sorted(limits)}")
        return
    (size,), ((max_batch, timeout_ms, backlog),) = sizes, limits
    if cfg.normalize_over == "round" and all(_valid_scales(m.get("scales")) for m in members):   # else malformed
        # one quality range and cost maximum per round; the latency maximum runs over the round in decision
        # order, each request's candidates included
        if len({(m["scales"]["q_min"], m["scales"]["q_max"], m["scales"]["c_max"]) for m in members}) != 1:
            p.add("a round's lines disagree on its quality range or cost maximum", f"round {r}")
        l_run = 0.0
        for m in sorted(members, key=lambda m: m["round_pos"]):
            l_run = max(l_run, max(c["lat_s"] for c in m["candidates"]))
            if m["scales"]["l_max"] != l_run:
                p.add("a line's latency maximum is not the round's running maximum", f"round {r}: {m['request_id']}")
    positions = [m["round_pos"] for m in members]
    if len(members) > size or len(set(positions)) != len(positions) or not all(0 <= x < size for x in positions):
        p.add("a round's positions are not distinct positions below its size",
              f"round {r}: size {size}, {sorted(positions)}")
    if not 1 <= size <= max_batch:
        p.add("a round holds more requests than its max_batch", f"round {r}: {size} > {max_batch}")
    if cfg.rounds.adaptive:
        pair = [s for s, t in ADAPTIVE_LIMITS if t == timeout_ms]
        if len(pair) != 1 or backlog < 0 or max_batch != max(pair[0], min(backlog, BACKLOG_CAP)):
            p.add("adaptive round limits are not an adaptive pair widened to the round's backlog",
                  f"round {r}: {(max_batch, timeout_ms)}, backlog {backlog}")
    elif (max_batch, timeout_ms) != (cfg.rounds.max_batch, cfg.rounds.timeout_ms):
        p.add("fixed round limits are not the configured ones",
              f"round {r}: {(max_batch, timeout_ms)} != {(cfg.rounds.max_batch, cfg.rounds.timeout_ms)}")
    if cfg.lpt != "off":
        keys = [m["lpt_key"] for m in sorted(members, key=lambda m: m["round_pos"])]
        if any(a < b for a, b in zip(keys, keys[1:], strict=False)):
            p.add(f"lpt {cfg.lpt}: a round's decisions are not in non-increasing lpt_key order", f"round {r}: {keys}")


def _pipeline(p: Problems, cfg: SchedulerConfig, fleet: list[FleetEntry], lines: list[dict], max_tokens: int) -> None:
    rtype = cfg.pipeline.router.type
    by_id = {f.instance_id: f for f in fleet}
    models = {f.model for f in fleet}
    tier_model = {f.tier: f.model for f in fleet}
    named = NAMED[rtype] if rtype in NAMED else None
    for line in lines:
        rid = line["request_id"] if "request_id" in line else "<no request_id>"
        if "reason" not in line or "instance_id" not in line:
            p.add("a pipeline request was never dispatched", f"{rid} ({line['error'] if 'error' in line else ''})")
            continue
        try:
            if "candidates" in line or "round" in line:
                p.add("a pipeline line carries RouteBalance round evidence", rid)
            reason = line["reason"]
            if not (isinstance(reason, str) and reason.startswith(REASON_PREFIX[rtype])):
                p.add(f"the router reason is not the {rtype} router's", f"{rid}: {reason!r}")
            if not _num(_get(line, "router_score")):
                p.add("the router score is not a finite number", f"{rid}: {line['router_score']!r}")
            served = by_id[line["instance_id"]] if line["instance_id"] in by_id else None
            if served is None:
                continue                          # reported with the common checks
            if named is not None and isinstance(reason, str):
                m = named[0].search(reason)
                if m is not None:
                    name = m.group(1)
                    if named[1] == "tier" and name not in tier_model:
                        p.add("the router named a tier outside the fleet", f"{rid}: {name!r}")
                    elif (tier_model[name] if named[1] == "tier" else name) != served.model:
                        p.add("the router chose a model other than the one served",
                              f"{rid}: {named[1]} {name!r} vs served model {served.model!r}")
            if cfg.pending_decode == "own_length":
                _decode(p, rid, line, served.model, _lengths(line, models))
            else:
                if "lengths" in line:
                    p.add("a baseline line carries estimator lengths (no estimator should run)", rid)
                for key in ("decode_tokens", "feature_tokens"):
                    if _get(line, key) != max_tokens:
                        p.add(f"a baseline's {key} is not the request's max_tokens",
                              f"{rid}: {line[key]} != {max_tokens}")
        except (_Malformed, KeyError, TypeError, ValueError) as e:
            p.add("a pipeline decision line is malformed", f"{rid}: {e}")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Check the values a cell actually used, from its decision lines.")
    ap.add_argument("--cell", type=Path, required=True, help="the cell's evidence directory")
    ap.add_argument("--config", type=Path, required=True, help="the scheduler document POSTed for the cell")
    a = ap.parse_args(argv)
    return run("arm", lambda: arm_problems(open_cell(a.cell), load_scheduler(a.config)))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
