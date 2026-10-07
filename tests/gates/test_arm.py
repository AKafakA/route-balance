"""The arm gate passes every recorded cell and fails each seeded violation of each of its rules."""
import json

import pytest

from gates.arm import ADAPTIVE_LIMITS, arm_problems, main
from gates.cell import open_cell
from routebalance.config import load_scheduler
from routebalance.schedule.rounds import ADAPTIVE_LIMITS as ROUNDS_LIMITS
from tests.gates.cellfix import (CELLS, DECOUPLED, FIXTURES, LOR, RB_FIXEDREF, RB_MAXDIV, RB_MEAN, RB_MIN, RB_OFF,
                                 RB_ROUND, SWEEP, copy_cell, edit_json, edit_lines, edit_manifests, record)

TIER = {"S_0": "small_a30", "S_1": "small_a30", "L_2": "large_a100"}


def problems(cell, config=None):
    return arm_problems(open_cell(cell), load_scheduler(config or cell / "config.json"))[0]


@pytest.mark.parametrize("name", CELLS)
def test_every_recorded_cell_passes(name):
    cell = FIXTURES / "cells" / name
    assert problems(cell) == []


def test_the_adaptive_limits_are_the_schedulers():
    assert ADAPTIVE_LIMITS == {(size, timeout) for _, size, timeout in ROUNDS_LIMITS}
    from gates.arm import BACKLOG_CAP
    from routebalance.schedule.rounds import BACKLOG_CAP as ROUNDS_CAP
    assert BACKLOG_CAP == ROUNDS_CAP                                # the largest round


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


def final_l_max_everywhere(ls):
    """Every line of a round given the round's final latency maximum (a scheduler that fixed it once per round)."""
    final = {}
    for ln in ls:
        final[ln["round"]] = max(final.get(ln["round"], 0.0), ln["scales"]["l_max"])
    for ln in ls:
        ln["scales"]["l_max"] = final[ln["round"]]


def double_c_max_consistently(ls):
    """Every line's cost maximum doubled, with its cost terms and scores rebuilt on it: consistent in itself, but not
    the maximum over the round's requests x the fleet."""
    from routebalance.schedule.score import EPS
    price, model = {"S_0": 1e-7, "S_1": 1e-7, "L_2": 4e-7}, {"S_0": "S", "S_1": "S", "L_2": "L"}
    w = {"w_quality": 0.2, "w_latency": 0.4, "w_cost": 0.4}          # rb-round's weights (the recorder's maxdiv)
    for ln in ls:
        ln["scales"]["c_max"] *= 2
        for c in ln["candidates"]:
            cost = ln["request"]["prompt_tokens"] * price[c["inst"]] + ln["lengths"][model[c["inst"]]] * price[c["inst"]]
            c["c"] = 1.0 - cost / (ln["scales"]["c_max"] + EPS)
            c["score"] = w["w_quality"] * c["q"] + w["w_cost"] * c["c"] + w["w_latency"] * c["l"]
        ln["score"] = next(c["score"] for c in ln["candidates"] if c["inst"] == ln["instance_id"])


def two_rounds_sizes(members):
    members[0]["round_size"] -= 1


def more_engine_waits(d):
    d["connection_waits"]["engine"] += 1


MUST_FAIL = [
    # a connection bound that was reached invalidates the cell: rerun with a larger bound
    (LOR, stats_after(more_engine_waits), "engine connection pool made requests wait (1 waits)"),
    (RB_MAXDIV, stats_after(lambda d: d["connection_waits"].__setitem__("telemetry", 5)),
     "telemetry connection pool made"),
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
    (RB_FIXEDREF, round_one(lambda ms: [m.__setitem__("round_max_batch", 33) for m in ms]),   # above the cap: never
     "not an adaptive pair widened to the round's backlog"),                                  # a valid limit
    (RB_FIXEDREF, round_one(lambda ms: [m.__setitem__("round_backlog", 20) for m in ms]),        # 20 waiting,
     "not an adaptive pair widened to the round's backlog"),                                   # max_batch not 20
    (RB_FIXEDREF, round_one(lambda ms: [m.pop("round_backlog") for m in ms]), "decision line is malformed"),
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
    (RB_MAXDIV, first(lambda ln: ln.pop("lpt_key")), "decision line is malformed"),
    (RB_OFF, first(lambda ln: ln.__setitem__("lpt_key", min(ln["lengths"].values()))),
     "lpt_key is not the max (lpt off) of the request's expected lengths"),
    # decode and feature tokens
    (RB_MAXDIV, first(lambda ln: ln.__setitem__("feature_tokens", ln["feature_tokens"] + 1)),
     "feature_tokens is not the largest expected length"),
    (RB_MAXDIV, first(lambda ln: ln.__setitem__("decode_tokens", ln["feature_tokens"])),
     "decode_tokens are not the expected length on the chosen model"),
    (RB_FIXEDREF, first(lambda ln: ln.__setitem__("decode_tokens", ln["decode_tokens"] - 1)),
     "decode_tokens are not the expected length on the chosen model"),
    (RB_MAXDIV, first(lambda ln: ln["lengths"].pop("S")), "decision line is malformed"),
    # pipeline
    (LOR, first(lambda ln: ln.pop("reason")), "never dispatched"),
    (LOR, first(set_("reason", "avengers_pro:upstream:rank=1:model=L")), "not the passthrough router's"),
    (DECOUPLED, first(other_tier_in_reason), "chose a model other than the one served"),
    (DECOUPLED, first(lambda ln: ln.__setitem__("reason", "rb_decoupled:tier=no_tier:score=0.5")),
     "named a tier outside the fleet"),
    (LOR, first(set_("router_score", "high")), "router score is not a finite number"),
    (LOR, first(set_("decode_tokens", 10)), "decode_tokens is not the request's max_tokens"),
    (LOR, first(set_("feature_tokens", 10)), "feature_tokens is not the request's max_tokens"),
    (LOR, first(set_("lengths", {"S": 1.0, "L": 2.0})), "carries estimator lengths"),
    (LOR, first(set_("candidates", [])), "carries RouteBalance round evidence"),
    (SWEEP, first(lambda ln: ln.__setitem__("decode_tokens", ln["feature_tokens"] + 1)),
     "decode_tokens are not the expected length on the chosen model"),
    (SWEEP, first(lambda ln: ln.pop("lengths")), "decision line is malformed"),
    # the router lane ran with the configuration's concurrency
    (DECOUPLED, lambda c: edit_manifests(c, lambda m: m["loaded"].__setitem__("router_lane_threads", 1)),
     "ran on another number of threads than the configuration's concurrency"),
    (RB_MAXDIV, lambda c: edit_manifests(c, lambda m: m["loaded"].__setitem__("router_lane_threads", 32)),
     "ran on another number of threads than the configuration's concurrency"),
    # the round's scales
    (RB_ROUND, first(set_("scales", None)), "scales are logged exactly when the score normalizes over the round"),
    (RB_MAXDIV, first(set_("scales", {"q_min": 0.0, "q_max": 1.0, "c_max": 1.0, "l_max": 9.0})),
     "scales are logged exactly when the score normalizes over the round"),
    (RB_ROUND, first(lambda ln: ln["scales"].__setitem__("l_max", ln["scales"]["l_max"] * 2)),
     "max_divide: a latency term is not 1 - latency"),
    (RB_ROUND, first(lambda ln: ln["scales"].__setitem__("l_max", ln["scales"]["l_max"] * 2)),
     "not the round's running maximum"),
    (RB_ROUND, first(lambda ln: ln["scales"].__setitem__("l_max", min(c["lat_s"] for c in ln["candidates"]) / 2)),
     "running latency maximum is below this request's latencies"),
    (RB_ROUND, round_one(lambda ms: ms[1]["scales"].__setitem__("c_max", ms[1]["scales"]["c_max"] * 2)),
     "disagree on its quality range or cost maximum"),
    (RB_ROUND, lines(final_l_max_everywhere), "not the round's running maximum"),   # the recorded rounds grow
    (RB_ROUND, first(lambda ln: ln["candidates"][0].__setitem__("c", ln["candidates"][0]["c"] + 1e-12)),
     "a cost term is not 1 - cost / (cost maximum + EPS)"),
    (RB_MAXDIV, first(lambda ln: ln["candidates"][0].__setitem__("c", ln["candidates"][0]["c"] + 1e-12)),
     "a cost term is not 1 - cost / (cost maximum + EPS)"),
    (RB_ROUND, first(lambda ln: ln["scales"].__setitem__("c_max", 0.0)), "the round's cost maximum is below"),
    (RB_ROUND, lines(double_c_max_consistently), "cost maximum is not the maximum over its requests x the fleet"),
    (RB_ROUND, first(set_("scales", [])), "decision line is malformed"),
    (RB_ROUND, first(lambda ln: ln["scales"].pop("l_max")), "decision line is malformed"),
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


def test_waits_from_before_the_cell_do_not_count(tmp_path):
    """Only waits during the cell's window invalidate it (a scheduler that waited in an earlier cell may run on)."""
    cell = copy_cell(tmp_path, LOR)
    for when in ("before", "after"):
        edit_json(cell / f"stats_{when}.json", lambda d: d["connection_waits"].update(engine=7, telemetry=3))
    assert arm_problems(open_cell(cell), load_scheduler(cell / "config.json"))[0] == []


@pytest.mark.parametrize("pool", ["engine", "telemetry"])
def test_a_stats_file_without_a_pools_waits_is_refused(tmp_path, pool):
    cell = copy_cell(tmp_path, LOR)
    edit_json(cell / "stats_after.json", lambda d: d["connection_waits"].pop(pool))
    with pytest.raises(Exception, match=pool):
        arm_problems(open_cell(cell), load_scheduler(cell / "config.json"))


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


def _admitted(tmp_path, keep, admitted_all=True, budget="fits_s"):
    """rb-maxdiv's cell as if the budget filter had admitted only `keep` on its first line. That line's cost and
    latency terms and scores are recomputed over the admitted candidates (the request's own maxima: rb-maxdiv
    normalizes per request); the quality terms are left as recorded, since arm cannot see the qualities. The line is
    then valid except for the rule under test. The cell is re-stamped with the filter-on configuration's digest."""
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
            ln["request"] = {"prompt_tokens": ln["request"]["prompt_tokens"], "max_tokens": 64,
                             "budget_cost": 1.0}                                          # admits every instance
        ln0 = ls[0]
        p_tok = ln0["request"]["prompt_tokens"]
        s_cost = p_tok * 1e-7 + min(int(ln0["lengths"]["S"]), 64) * 1e-7                  # S: 1e-7 in and out
        ln0["request"]["budget_cost"] = s_cost if budget == "fits_s" else s_cost / 10    # L costs 4x S per token
        by_inst = {c["inst"]: c for c in ln0["candidates"]}
        cands = [by_inst[i] for i in keep]
        price = {"S_0": 1e-7, "S_1": 1e-7, "L_2": 4e-7}
        cost = {c["inst"]: p_tok * price[c["inst"]] + ln0["lengths"]["S" if c["inst"] != "L_2" else "L"] * price[c["inst"]]
                for c in cands}
        lmax, cmax = max(c["lat_s"] for c in cands), max(cost.values())
        for c in cands:
            c["c"] = 1.0 - cost[c["inst"]] / (cmax + EPS)
            c["l"] = 1.0 - c["lat_s"] / (lmax + EPS)
            c["score"] = w["w_quality"] * c["q"] + w["w_cost"] * c["c"] + w["w_latency"] * c["l"]
        ln0["candidates"], ln0["admitted_all"] = cands, admitted_all
        best = max(cands, key=lambda c: c["score"])
        ln0["instance_id"], ln0["tier"], ln0["score"] = best["inst"], TIER[best["inst"]], best["score"]
        ln0["decode_tokens"] = max(1, int(ln0["lengths"]["S" if best["inst"] != "L_2" else "L"]))
    edit_lines(cell, budgeted)
    return problems(cell)


def test_the_admission_filter_keeps_what_v3s_rule_admits(tmp_path):
    """v3's budget filter, recomputed by the gate: the budget fits S's expected cost only, so S_0 and S_1 remain."""
    assert _admitted(tmp_path, ["S_0", "S_1"]) == []


@pytest.mark.parametrize("keep,admitted_all,budget", [
    (["S_1", "S_0"], True, "fits_s"),                   # out of fleet order
    (["S_0"], True, "fits_s"),                          # an admitted instance left out
    (["S_0", "S_1", "L_2"], True, "fits_s"),            # L's expected cost is over the budget
    (["S_0", "S_1"], False, "fits_s"),                  # admitted, but the line says none was
])
def test_candidates_that_are_not_v3s_filters_fail(tmp_path, keep, admitted_all, budget):
    found = _admitted(tmp_path, keep, admitted_all=admitted_all, budget=budget)
    assert any("not v3's budget filter's" in p for p in found), found


def test_when_nothing_fits_the_cheapest_by_worst_case_cost_remain(tmp_path):
    assert _admitted(tmp_path, ["S_0", "S_1"], admitted_all=False, budget="none") == []
    found = _admitted(tmp_path / "all", ["S_0", "S_1", "L_2"], admitted_all=False, budget="none")
    assert any("not v3's budget filter's" in p for p in found), found


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
    """An estimate may be 0 tokens (a model predicted to answer empty); its decode count is then 1."""
    cell = copy_cell(tmp_path, RB_MIN)

    def empty(ls):
        ln = max((x for x in ls if x["round"] == 1), key=lambda x: x["round_pos"])   # last in its round: key 0 keeps order
        other = next(m for m in ln["lengths"] if m != "S")
        ln["lengths"] = {"S": 0.0, other: ln["lengths"][other]}
        ln["lpt_key"] = 0.0
        ln["decode_tokens"] = max(1, int(ln["lengths"]["S" if ln["instance_id"] != "L_2" else other]))
        ln["feature_tokens"] = max(1, int(ln["lengths"][other]))
    edit_lines(cell, empty)
    found = problems(cell)
    assert not any("malformed" in p or "lpt_key" in p or "decode_tokens" in p or "feature_tokens" in p
                   for p in found), found
