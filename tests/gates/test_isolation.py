"""a cell counts only when every engine's warm-up proved the prefix-cache reset and, between the two engine
readings, only the cell's own requests reached the engines (gates/cell.py isolation_problems)."""
import json
import shutil
from pathlib import Path

import pytest

from bench.records import read_cell
from gates.cell import isolation_problems

FIXTURE = Path(__file__).parent / "fixtures" / "cells" / "rb-maxdiv__l400__s1"


def _warmup(instances):
    lines = []
    for i in instances:
        for step, hit in (("prime", 4), ("after_reset", 0), ("control", 16)):
            lines.append({"instance_id": i, "purpose": "warmup", "cache_step": step, "cache_queried": 27,
                          "cache_hit": hit, "success": True, "prompt_tokens": 27, "output_tokens": 3})
    return {"requests": lines, "drain_s": 0.0}


@pytest.fixture
def cell(tmp_path):
    d = tmp_path / FIXTURE.name
    shutil.copytree(FIXTURE, d)
    rec = read_cell(d / f"{FIXTURE.name}.cell.json")
    per = {}
    for r in rec.requests:
        if r.reply is not None:
            t = per.setdefault(r.reply.instance_id, [0, 0, 0])
            t[0] += 1
            t[1] += r.reply.prompt_tokens
            t[2] += r.reply.output_tokens
    for i in json.loads((d / "engines_before.json").read_text())["instances"]:     # the warm-up's three requests
        t = per.setdefault(i, [0, 0, 0])
        t[0], t[1], t[2] = t[0] + 3, t[1] + 3 * 27, t[2] + 3 * 3
    for when in ("before", "after"):
        e = json.loads((d / f"engines_{when}.json").read_text())
        for i, v in e["instances"].items():
            n, p, g = per.get(i, [0, 0, 0]) if when == "after" else (0, 0, 0)
            v["served"] = {"requests": 1000 + n, "prompt_tokens": 90000 + p, "generation_tokens": 70000 + g,
                           "prefix_queries": 90000 + p, "prefix_hits": 5000}
        (d / f"engines_{when}.json").write_text(json.dumps(e))
    (d / "warmup.json").write_text(json.dumps(_warmup(sorted(e["instances"]))))
    _edit(d / "stats_before.json", lambda st: st["logged"].update(decisions=0))    # a fresh scheduler
    return d


def _edit(path, fn):
    data = json.loads(path.read_text())
    fn(data)
    path.write_text(json.dumps(data))


def test_a_proven_reset_and_only_the_cells_own_traffic_pass(cell):
    assert isolation_problems(cell) == []


def test_a_cell_from_before_the_reset_has_no_proof_and_no_counters():
    problems = isolation_problems(FIXTURE)
    assert any("cannot prove the cache reset" in p for p in problems)


def test_a_cached_token_after_the_reset_or_a_blind_control_is_caught(cell):
    def hit(d):
        next(x for x in d["requests"] if x["cache_step"] == "after_reset")["cache_hit"] = 16
    _edit(cell / "warmup.json", hit)
    assert any("found 16 cached tokens" in p for p in isolation_problems(cell))

    def blind(d):
        for x in d["requests"]:
            if x["cache_step"] == "control":
                x["cache_hit"] = 0
    _edit(cell / "warmup.json", blind)
    assert any("control request found no cached token" in p for p in isolation_problems(cell))


def test_an_engine_without_its_reset_step_is_caught(cell):
    first = sorted(json.loads((cell / "engines_before.json").read_text())["instances"])[0]
    _edit(cell / "warmup.json", lambda d: d.update(requests=[x for x in d["requests"]
                                                             if not (x["instance_id"] == first
                                                                     and x["cache_step"] == "after_reset")]))
    assert any(p.startswith(f"{first}: the warm-up has no ['after_reset']") for p in isolation_problems(cell))


def test_one_request_from_elsewhere_is_caught(cell):
    first = sorted(json.loads((cell / "engines_after.json").read_text())["instances"])[0]

    def extra(d):
        s = d["instances"][first]["served"]
        s.update(requests=s["requests"] + 1, prompt_tokens=s["prompt_tokens"] + 30,
                 generation_tokens=s["generation_tokens"] + 8)
    _edit(cell / "engines_after.json", extra)
    assert any(p.startswith(f"{first}: the engine served") and "other traffic" in p for p in isolation_problems(cell))


def test_readings_without_serving_counters_are_refused_by_the_schema(cell):
    _edit(cell / "engines_before.json", lambda d: [v.pop("served") for v in d["instances"].values()])
    problems = isolation_problems(cell)
    assert any("cannot show the traffic" in p and "served" in p and "Field required" in p for p in problems)
    assert any("cannot prove the cache reset" in p and "served" in p for p in problems)


def test_a_reused_scheduler_is_caught(cell):
    _edit(cell / "stats_before.json", lambda st: st["logged"].update(decisions=40))
    assert any("not a fresh process" in p for p in isolation_problems(cell))


def test_a_failed_evidence_writer_is_caught(cell):
    _edit(cell / "stats_after.json", lambda st: st.update(evidence_failure="JsonlLog._append: OSError(28)"))
    assert any("an evidence writer failed" in p for p in isolation_problems(cell))


def test_resume_rechecks_the_isolation_of_a_published_valid_cell(tmp_path, monkeypatch):
    """A saved valid verdict is not enough; status(), which resume and group_ready use, rechecks
    the isolation evidence (the record checks are stubbed here: they have their own tests)."""
    from runner import store
    root = tmp_path / "cells"
    shutil.copytree(FIXTURE, root / FIXTURE.name)
    (root / FIXTURE.name / "gates").mkdir(exist_ok=True)
    (root / FIXTURE.name / "gates" / "verdict.json").write_text(json.dumps({"valid": True}))
    monkeypatch.setattr(store, "record_problems", lambda scope, exp, d: [])

    class Exp:
        cell_id = FIXTURE.name
    assert store.gated_valid(root, Exp()) is True                # the fast gate reads the verdict only
    assert store.status(None, Exp(), root) == "invalid"           # no warm-up proof, no counters
