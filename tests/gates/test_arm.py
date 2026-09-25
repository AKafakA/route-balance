"""The arm gate passes every recorded cell and fails each seeded violation of each of its rules."""
import json

import pytest

from gates.arm import ADAPTIVE_LIMITS, _is_subsequence, arm_problems, main
from gates.cell import open_cell
from routebalance.config import load_scheduler
from routebalance.schedule.rounds import ADAPTIVE_LIMITS as ROUNDS_LIMITS
from tests.gates.cellfix import (CELLS, DECOUPLED, FIXTURES, LOR, RB_FIXEDREF, RB_MAXDIV, RB_MEAN, RB_MIN, RB_OFF,
                                 SWEEP, copy_cell, edit_json, edit_lines, record)

TIER = {"S_0": "small_a30", "S_1": "small_a30", "L_2": "large_a100"}


def problems(cell, config=None):
    return arm_problems(open_cell(cell), load_scheduler(config or cell / "config.json"))[0]


@pytest.mark.parametrize("name", CELLS)
def test_every_recorded_cell_passes(name):
    cell = FIXTURES / "cells" / name
    assert problems(cell) == []


def test_the_adaptive_limits_are_the_schedulers():
    assert ADAPTIVE_LIMITS == {(size, timeout) for _, size, timeout in ROUNDS_LIMITS}


def test_subsequence():
    assert _is_subsequence(["a", "c"], ["a", "b", "c"]) and not _is_subsequence(["c", "a"], ["a", "b", "c"])
    assert not _is_subsequence(["a", "a"], ["a", "b"])


def lines(fn):
    return lambda cell: edit_lines(cell, fn)


def first(fn):
    return lines(lambda ls: fn(ls[0]))


def full_round(ls):
    """The lines of the recorded round with the most requests whose lpt_keys are not all equal (round sizes
    depend on arrival timing when the fixtures are recorded, so no round id is assumed)."""
    rounds = {}
    for x in ls:
        rounds.setdefault(x["round"], []).append(x)
    usable = [m for m in rounds.values() if len(m) >= 2 and len({x["lpt_key"] for x in m}) >= 2]
    assert usable, "the fixture has no round of two or more distinct LPT keys"
    return max(usable, key=len)


def round_one(fn):
    return lines(lambda ls: fn(full_round(ls)))


def stats_after(fn):
    return lambda cell: edit_json(cell / "stats_after.json", fn)


def set_(key, value):
    def f(d):
        d[key] = value
    return f


def whole_max(line):
    return max(1, int(max(line["lengths"].values())))


def worst_candidate(line):
    worst = min(line["candidates"], key=lambda c: c["score"])
    line["instance_id"], line["tier"] = worst["inst"], TIER[worst["inst"]]


def slowest_term(line, value):
    slow = max(line["candidates"], key=lambda c: c["lat_s"])
    slow["l"] = value


def swap_first_two_positions(members):
    """Swap the positions of the round's largest and smallest LPT keys, which puts the round out of order."""
    hi, lo = max(members, key=lambda m: m["lpt_key"]), min(members, key=lambda m: m["lpt_key"])
    hi["round_pos"], lo["round_pos"] = lo["round_pos"], hi["round_pos"]


def other_tier_in_reason(line):
    other = "small_a30" if line["tier"] == "large_a100" else "large_a100"
    line["reason"] = line["reason"].replace(f"tier={line['tier']}", f"tier={other}")


def two_rounds_sizes(members):
    members[0]["round_size"] -= 1


MUST_FAIL = [
    # coverage and identity
    (RB_MAXDIV, first(set_("config_digest", "0" * 64)), "another config digest"),
    (RB_MAXDIV, lines(lambda ls: ls.pop(0)), "has no decision line"),
    (RB_MAXDIV, lines(lambda ls: ls.pop(0)), "decision count over the cell's window"),
    (LOR, lines(lambda ls: ls.append(dict(ls[0]))), "more than one decision line"),
    (LOR, first(set_("request_id", "another-cell.r1.0")), "belongs to no request of the cell"),
    (LOR, stats_after(lambda s: s["logged"].__setitem__("decisions_buffered", 2)), "not yet on disk"),
    (LOR, stats_after(set_("active", 1)), "requests in service"),
    (LOR, first(set_("instance_id", "X_9")), "outside the fleet"),
    (LOR, first(set_("tier", "no-such-tier")), "tier is not its instance's"),
    # RouteBalance: candidates, scores, normalization
    (RB_MAXDIV, first(lambda ln: ln.pop("candidates")), "never dispatched"),
    (RB_MAXDIV, first(set_("reason", "passthrough:no_routing")), "carries a router reason"),
    (RB_MAXDIV, first(lambda ln: ln["candidates"].pop(1)), "not every fleet instance in fleet order"),
    (RB_MAXDIV, first(lambda ln: ln["candidates"].reverse()), "not every fleet instance in fleet order"),
    (RB_MAXDIV, first(worst_candidate), "does not have the highest score"),
    (RB_MAXDIV, first(lambda ln: ln["candidates"][0].__setitem__("q", ln["candidates"][0]["q"] + 0.25)),
     "configured weights"),
    (RB_MAXDIV, first(lambda ln: slowest_term(ln, 0.2)), "max_divide: the slowest candidate's latency term is not ~0"),
    (RB_MAXDIV, first(lambda ln: slowest_term(ln, 0.2)), "max_divide: a latency term is not 1 - latency"),
    (RB_FIXEDREF, first(lambda ln: ln["candidates"][0].__setitem__("l", 1.5)), "fixed_ref: a latency term lies outside"),
    (RB_FIXEDREF, first(lambda ln: slowest_term(ln, 0.01)), "do not share one reference latency"),
    (RB_MAXDIV, first(set_("candidates", "every instance")), "decision line is malformed"),
    # rounds and LPT
    (RB_MAXDIV, round_one(lambda ms: [m.__setitem__("round_size", 5) for m in ms]), "more requests than its max_batch"),
    (RB_MAXDIV, round_one(lambda ms: [m.__setitem__("round_timeout_ms", 99.0) for m in ms]),
     "fixed round limits are not the configured ones"),
    (RB_FIXEDREF, round_one(lambda ms: [m.__setitem__("round_max_batch", 8) for m in ms]),
     "not an adaptive limit pair"),
    (RB_MAXDIV, round_one(lambda ms: ms[1].__setitem__("round_pos", ms[0]["round_pos"])), "distinct positions"),
    (RB_MAXDIV, round_one(two_rounds_sizes), "disagree on its size or limits"),
    (RB_MAXDIV, round_one(swap_first_two_positions), "lpt max: a round's decisions are not in non-increasing"),
    (RB_FIXEDREF, round_one(swap_first_two_positions), "lpt max: a round's decisions are not in non-increasing"),
    (RB_MIN, round_one(swap_first_two_positions), "lpt min: a round's decisions are not in non-increasing"),
    (RB_MEAN, round_one(swap_first_two_positions), "lpt mean: a round's decisions are not in non-increasing"),
    (RB_MAXDIV, first(lambda ln: ln.__setitem__("lpt_key", min(ln["lengths"].values()))),
     "lpt_key is not the max of the request's expected lengths"),
    (RB_MIN, first(lambda ln: ln.__setitem__("lpt_key", max(ln["lengths"].values()))),
     "lpt_key is not the min of the request's expected lengths"),
    (RB_MEAN, first(lambda ln: ln.__setitem__("lpt_key", ln["lpt_key"] * 1.01)),
     "lpt_key is not the mean of the request's expected lengths"),
    (RB_MIN, first(lambda ln: ln.__setitem__("decode_tokens", whole_max(ln))),
     "decode_tokens do not follow pending_decode lpt_key"),
    (RB_MAXDIV, first(lambda ln: ln.pop("lpt_key")), "decision line is malformed"),
    (RB_OFF, first(lambda ln: ln.__setitem__("lpt_key", min(ln["lengths"].values()))),
     "lpt_key is not the max (lpt off) of the request's expected lengths"),
    (RB_OFF, first(lambda ln: ln.__setitem__("decode_tokens", max(1, int(min(ln["lengths"].values()))))),
     "decode_tokens do not follow pending_decode lpt_key"),
    # decode and feature tokens
    (RB_MAXDIV, first(lambda ln: ln.__setitem__("feature_tokens", ln["feature_tokens"] + 1)),
     "feature_tokens is not the largest expected length"),
    (RB_MAXDIV, first(lambda ln: ln.__setitem__("decode_tokens", ln["feature_tokens"])),
     "decode_tokens do not follow pending_decode own_length"),
    (RB_FIXEDREF, first(lambda ln: ln.__setitem__("decode_tokens", ln["decode_tokens"] - 1)),
     "decode_tokens do not follow pending_decode lpt_key"),
    (RB_MAXDIV, first(lambda ln: ln["lengths"].pop("S")), "decision line is malformed"),
    # pipeline
    (LOR, first(lambda ln: ln.pop("reason")), "never dispatched"),
    (LOR, first(set_("reason", "best_route_4way:argmax_label=1:p=0.900")), "not the passthrough router's"),
    (DECOUPLED, first(other_tier_in_reason), "chose a model other than the one served"),
    (DECOUPLED, first(lambda ln: ln.__setitem__("reason", "rb_decoupled:tier=no_tier:score=0.5")),
     "named a tier outside the fleet"),
    (LOR, first(set_("router_score", "high")), "router score is not a finite number"),
    (LOR, first(set_("decode_tokens", 10)), "decode_tokens is not the request's max_tokens"),
    (LOR, first(set_("feature_tokens", 10)), "feature_tokens is not the request's max_tokens"),
    (LOR, first(set_("lengths", {"S": 1.0, "L": 2.0})), "carries estimator lengths"),
    (LOR, first(set_("candidates", [])), "carries RouteBalance round evidence"),
    (SWEEP, first(lambda ln: ln.__setitem__("decode_tokens", ln["feature_tokens"] + 1)),
     "decode_tokens do not follow pending_decode own_length"),
    (SWEEP, first(lambda ln: ln.pop("lengths")), "decision line is malformed"),
]


@pytest.mark.parametrize("name,mutate,expected", MUST_FAIL, ids=[f"{n}:{e[:40]}" for n, _, e in MUST_FAIL])
def test_each_seeded_violation_fails(tmp_path, name, mutate, expected):
    cell = copy_cell(tmp_path, name)
    mutate(cell)
    found = problems(cell)
    assert any(expected in p for p in found), found


def test_another_cells_configuration_fails(tmp_path):
    cell = copy_cell(tmp_path, RB_MAXDIV)
    found = problems(cell, FIXTURES / "cells" / RB_FIXEDREF / "config.json")
    assert any("ran another configuration" in p for p in found)
    assert any("manifest_before.json is for another configuration" in p for p in found)


def test_the_cli_passes_and_fails_with_one_reason_per_line(tmp_path, capsys):
    cell = copy_cell(tmp_path, RB_MAXDIV)
    assert main(["--cell", str(cell), "--config", str(cell / "config.json")]) == 0
    assert capsys.readouterr().out.startswith("PASS arm: ")
    edit_lines(cell, lambda ls: ls.pop(0))
    assert main(["--cell", str(cell), "--config", str(cell / "config.json")]) == 1
    out = capsys.readouterr().out.splitlines()
    assert len(out) >= 2 and all(line.startswith("FAIL arm: ") for line in out)


@pytest.mark.parametrize("breakage", ["no stats", "two records", "bad json", "wrong dir name"])
def test_an_unreadable_cell_fails(tmp_path, capsys, breakage):
    cell = copy_cell(tmp_path, LOR)
    if breakage == "no stats":
        (cell / "stats_after.json").unlink()
    elif breakage == "two records":
        (cell / "other.cell.json").write_text(record(cell).read_text())
    elif breakage == "bad json":
        (cell / "decisions.jsonl").write_text('{"request_id": 1, "request_id": 2}\n')      # repeated key
    else:
        cell = cell.rename(cell.with_name("renamed"))
    assert main(["--cell", str(cell), "--config", str(cell / "config.json")]) == 1
    assert "FAIL arm: cannot check" in capsys.readouterr().out


def _admitted(tmp_path, keep, admitted_all=True):
    """rb-maxdiv's cell as if the budget filter had admitted only `keep` on its first line. That line's latency
    terms and scores are recomputed over the admitted candidates; the quality and cost terms are left as recorded,
    since arm does not check their normalization. The line is then valid except for the rule under test. The cell
    is re-stamped with the filter-on configuration's digest."""
    from routebalance.config import config_digest, parse_scheduler
    from routebalance.schedule.score import EPS
    cell = copy_cell(tmp_path, RB_MAXDIV)
    doc = json.loads((cell / "config.json").read_text())
    doc["budget"]["admission_filter"] = True
    (cell / "config.json").write_text(json.dumps(doc))
    digest = config_digest(parse_scheduler(doc))
    edit_json(record(cell), lambda r: [r[k].__setitem__("config_digest", digest)
                                       for k in ("identity_start", "identity_end")])
    for when in ("before", "after"):
        edit_json(cell / f"manifest_{when}.json", lambda m: m.__setitem__("config_digest", digest))
    w = doc["weights"]

    def budgeted(ls):
        for ln in ls:
            ln["budget"] = {"max_tokens": 64, "affordable_tokens": 900, "exhausted": False}
            ln["config_digest"] = digest
        ln0 = ls[0]
        by_inst = {c["inst"]: c for c in ln0["candidates"]}
        cands = [by_inst[i] for i in keep]
        lmax = max(c["lat_s"] for c in cands)
        for c in cands:
            c["l"] = 1.0 - c["lat_s"] / (lmax + EPS)
            c["score"] = w["w_quality"] * c["q"] + w["w_cost"] * c["c"] + w["w_latency"] * c["l"]
        ln0["candidates"], ln0["admitted_all"] = cands, admitted_all
        best = max(cands, key=lambda c: c["score"])
        ln0["instance_id"], ln0["tier"], ln0["score"] = best["inst"], TIER[best["inst"]], best["score"]
        ln0["decode_tokens"] = max(1, int(ln0["lengths"]["S" if best["inst"] != "L_2" else "L"]))
    edit_lines(cell, budgeted)
    return problems(cell)


def test_the_admission_filter_admits_an_ordered_subset(tmp_path):
    """With the budget admission filter, a budgeted line may score only the admitted instances, in fleet order."""
    assert _admitted(tmp_path, ["S_0", "S_1"]) == []


def test_admitted_candidates_out_of_fleet_order_fail(tmp_path):
    found = _admitted(tmp_path, ["S_1", "S_0"])
    assert any("not a subset of the fleet in fleet order" in p for p in found), found


def test_a_filter_that_admitted_none_scores_every_instance(tmp_path):
    found = _admitted(tmp_path, ["S_0", "S_1"], admitted_all=False)
    assert any("not every fleet instance in fleet order" in p for p in found), found


def test_rb_decoupled_may_be_served_by_another_tier_of_its_model(tmp_path):
    """The router names the best TIER but returns its model; the dispatcher may place the request on any instance
    of that model. Here model S also runs on an A100 tier, and S_1 is that tier."""
    cell = copy_cell(tmp_path, DECOUPLED)
    for when in ("before", "after"):
        edit_json(cell / f"manifest_{when}.json",
                  lambda m: m["fleet"][1].update(tier="small_a100", gpu="a100"))

    def reroute(ls):
        assert not any(x["instance_id"] == "S_1" for x in ls)      # the recorded router always chose L
        ls[0].update(instance_id="S_1", tier="small_a100", reason="rb_decoupled:tier=small_a30:score=0.500000")
    edit_lines(cell, reroute)
    assert problems(cell) == []


def test_lpt_off_keeps_no_order_but_writes_the_max(tmp_path):
    """With lpt off the round keeps arrival order: a round whose keys increase passes."""
    cell = copy_cell(tmp_path, RB_OFF)

    def increasing(ls):
        members = full_round(ls)
        hi, lo = max(members, key=lambda m: m["lpt_key"]), min(members, key=lambda m: m["lpt_key"])
        order = sorted(members, key=lambda m: m["round_pos"])
        if order.index(hi) < order.index(lo):           # make the keys increase along the positions
            hi["round_pos"], lo["round_pos"] = lo["round_pos"], hi["round_pos"]
    edit_lines(cell, increasing)
    assert problems(cell) == []


def test_a_zero_expected_length_is_a_valid_estimate(tmp_path):
    """S5b: an estimate may be 0 tokens (lpt min over a model predicted empty); the decode count is then 1."""
    cell = copy_cell(tmp_path, RB_MIN)

    def empty(ls):
        ln = max((x for x in ls if x["round"] == 1), key=lambda x: x["round_pos"])   # last in its round: key 0 keeps order
        other = next(m for m in ln["lengths"] if m != "S")
        ln["lengths"] = {"S": 0.0, other: ln["lengths"][other]}
        ln["lpt_key"] = 0.0
        ln["decode_tokens"] = 1
        ln["feature_tokens"] = max(1, int(ln["lengths"][other]))
    edit_lines(cell, empty)
    found = problems(cell)
    assert not any("malformed" in p or "lpt_key" in p or "decode_tokens" in p or "feature_tokens" in p
                   for p in found), found
