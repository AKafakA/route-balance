"""tools/repetition_rows.py: D171's points (frontier, knees), D192/D203's all-eight sets, D178's thesis points, refusals."""
import json

import pytest

from tools import repetition_rows as rr

# D192's eight presets, written out here so that a wrong preset in the tool is caught
EIGHT = ("uniform", "q60l20c20", "q80l10c10", "q90l05c05", "q05l25c70", "q22l18c60", "q25l05c70", "q50l30c20")
QT = ("l12", "l18", "l24", "l30", "l36", "l42")


def cfg(cid, t, q, e, c):
    return {"group": "qwen", "config_id": cid, "trace_id": t, "seeds": [1], "quality": q, "e2e_mean_ms": e,
            "cost_total_per_1k": c}


def report(extra=None):
    configs = []
    for t in QT:
        for i, w in enumerate(EIGHT):                   # RouteBalance in-batch: quality and cost up with i, e2e flat
            configs.append(cfg(f"qwen.rb-static.{w}", t, 0.40 + 0.001 * i, 1000.0, 0.05 + 0.001 * i))
            configs.append(cfg(f"qwen.rb-static-dec.{w}", t, 0.40 + 0.001 * i, 1500.0, 0.06 + 0.001 * i))
            configs.append(cfg(f"qwen.rb-static-reqnorm.{w}", t, 0.40 + 0.001 * i, 1200.0, 0.05 + 0.001 * i))
        configs.append(cfg("qwen.rb-static.q10l80c10", t, 0.50, 100.0, 0.01))   # outside the eight: dominates all
        configs.append(cfg("qwen.ap.pw025.lor", t, 0.39, 3000.0, 0.03))         # cheapest: on the frontier
        configs.append(cfg("qwen.ap.pw080.lor", t, 0.38, 3500.0, 0.08))         # dominated, not its system's knee
        configs.append(cfg("qwen.br.t00.lor", t, 0.37, 4000.0, 0.09))           # its system's only configuration
        configs.append(cfg("qwen.disp.rr", t, 0.36, 2000.0, 0.07))
        # AvengersPro on another dispatcher, its own system: both points on its own frontier, both dominated overall
        # (by ap.pw025.lor); its knee (scaled over the two) is pw053, so pw025.po2 is not repeated
        configs.append(cfg("qwen.ap.pw025.po2", t, 0.386, 3100.0, 0.031))
        configs.append(cfg("qwen.ap.pw053.po2", t, 0.388, 3200.0, 0.0305))
        configs.append(cfg("qwen.ap.pw025.lor-shipped", t, 0.60, 10.0, 0.001))  # as shipped: never counts
    configs += extra or []
    return {"configs": configs, "coverage": {"missing": [], "invalid": {}, "pending": {}}, "left_out": {},
            "failures": [], "gate_failures": [], "parity": {"qwen": "not run"}}


def scope():
    return {"rows": [{"id": "W9", "kind": "cells", "enabled": False, "why": "placeholder",
                      "blocks": [{"name": "old", "group": "qwen", "traces": list(QT), "seeds": [2, 3]}]},
                     {"id": "W8r", "kind": "cells", "enabled": False, "why": "placeholder", "blocks": []},
                     {"id": "W2", "kind": "cells", "enabled": True,
                      "blocks": [{"name": "W2", "group": "qwen", "traces": list(QT), "seeds": [1]}]},
                     {"id": "W4l", "kind": "cells", "enabled": True,
                      "blocks": [{"name": "W4-learned", "group": "qwen-ablation", "traces": ["l12"], "seeds": [1]}]},
                     {"id": "W8p", "kind": "cells", "enabled": True,
                      "blocks": [{"name": "W8p", "group": "gemma", "traces": ["gl12"], "seeds": [1]}]}]}


def test_the_eight_presets_and_families():
    assert set(rr.EIGHT) == set(EIGHT) and len(rr.EIGHT) == 8
    assert rr.FAMILIES["qwen"]["traces"] == QT and rr.FAMILIES["gemma"]["traces"] == ("gl12", "gl30", "gl42")
    assert set(rr.FAMILIES["qwen"]["all_eight"]) == {"rb", "rb-reqnorm", "rb-dec"}
    assert set(rr.FAMILIES["gemma"]["all_eight"]) == {"rb", "rb-dec"}


def test_dominance_frontier_and_knee():
    assert rr.dominates((0.5, 1, 1), (0.4, 1, 1)) and not rr.dominates((0.5, 1, 1), (0.5, 1, 1))
    pts = {"a": (0.5, 10.0, 1.0), "b": (0.4, 5.0, 0.5), "c": (0.3, 20.0, 2.0)}
    assert rr.frontier(pts) == {"a", "b"}
    assert rr.knee({"only": (0.1, 1.0, 1.0)}) == "only"


def test_knee_scaling_and_ties():
    # the knee is neither the lowest latency (c) nor the best quality (a): b is closest to the scaled best corner
    pts = {"a": (0.50, 1000.0, 0.01), "b": (0.45, 600.0, 0.05), "c": (0.30, 400.0, 0.10)}
    assert rr.frontier(pts) == {"a", "b", "c"} and rr.knee(pts) == "b"
    # two frontier points equally far from the corner: the smaller id
    tie = {"x": (0.50, 1000.0, 0.05), "y": (0.40, 500.0, 0.05)}
    assert rr.knee(tie) == "x"
    # a dominated point widens the scale (the axes span all the system's configurations, not only its frontier): y
    assert rr.knee({**tie, "z": (0.00, 1100.0, 0.05)}) == "y"


def test_build_qwen_points_and_row():
    new, chosen = rr.build(report(), scope(), "qwen")
    row = next(r for r in new["rows"] if r["id"] == "W9")
    assert row["enabled"] and row["why"] is None and [b["name"] for b in row["blocks"]] == [f"W9-{t}" for t in QT]
    l12 = next(b for b in row["blocks"] if b["traces"] == ["l12"])
    assert l12["traces"] == ["l12"] and l12["seeds"] == [2, 3] and l12["axes"] == {} and not l12["fill"]
    ids = set(l12["configs"])
    assert {f"qwen.rb-static.{w}" for w in EIGHT} <= ids                  # D192 (1)
    assert {f"qwen.rb-static-dec.{w}" for w in EIGHT} <= ids              # D203
    assert {f"qwen.rb-static-reqnorm.{w}" for w in EIGHT} <= ids          # D203
    assert "qwen.rb-static.q10l80c10" not in ids                          # outside the eight (D192)
    assert not any("shipped" in i for i in ids)                           # D169
    assert "qwen.ap.pw025.lor" in ids and "qwen.ap.pw080.lor" not in ids   # frontier vs dominated, not the knee
    assert "qwen.br.t00.lor" in ids and "qwen.disp.rr" in ids             # one configuration: its own knee
    # the overall frontier, exactly: the eight in-batch presets (nothing beats their latency at their cost) and
    # AvengersPro 25 on least-outstanding (the cheapest); the outside-eight preset and the shipped copy dominate no one
    assert chosen["l12"]["frontier"] == {f"qwen.rb-static.{w}" for w in EIGHT} | {"qwen.ap.pw025.lor"}
    # on its own system's frontier but dominated overall, and not that system's knee: not repeated
    assert rr.frontier({"a": (0.386, 3100.0, 0.031), "b": (0.388, 3200.0, 0.0305)}) == {"a", "b"}
    assert "qwen.ap.pw025.po2" not in ids
    # each dispatcher is its own system: AvengersPro on power of two has its own knee
    assert "qwen.ap.pw053.po2" in ids and "qwen.ap.pw053.po2" in chosen["l12"]["knee"]
    assert chosen["l12"]["thesis"] == {"qwen.rb-static-dec.q50l30c20", "qwen.rb-static-dec.uniform"}
    assert chosen["l18"]["thesis"] == set()                               # D178: 12, 30 and 42 only
    assert row["plan"] == f"{sum(2 * len(b['configs']) for b in row['blocks'])} cells"
    assert [r for r in new["rows"] if r["id"] != "W9"] == [r for r in scope()["rows"] if r["id"] != "W9"]


def test_build_gemma_has_no_reqnorm_or_thesis_rule():
    rep = report()
    for c in rep["configs"]:
        c["group"] = "gemma"
        c["config_id"] = c["config_id"].replace("qwen.", "gemma.")
        c["trace_id"] = {"l12": "gl12", "l30": "gl30", "l42": "gl42"}.get(c["trace_id"], c["trace_id"])
    new, chosen = rr.build(rep, scope(), "gemma")
    row = next(r for r in new["rows"] if r["id"] == "W8r")
    assert [b["traces"] for b in row["blocks"]] == [["gl12"], ["gl30"], ["gl42"]]
    assert chosen["gl12"]["thesis"] == set()
    want = {f"gemma.rb-static.{w}" for w in EIGHT} | {f"gemma.rb-static-dec.{w}" for w in EIGHT}
    assert chosen["gl12"]["all eight"] == want


@pytest.mark.parametrize("mutate, msg", [
    (lambda r, s: r["coverage"]["missing"].append("qwen.ap.pw025.lor__l12__s1"), "not complete"),
    (lambda r, s: r["coverage"]["invalid"].update({"qwen.disp.rr__l42__s1": "x"}), "not complete"),
    (lambda r, s: r["coverage"]["pending"].update({"W2/W2": "pins not given"}), "pending"),
    (lambda r, s: r["coverage"]["pending"].update({"W99/x": "?"}), "does not hold"),
    (lambda r, s: r["left_out"].update({"qwen.ap.pw025.lor__l30__s1": "gate"}), "not complete"),
    (lambda r, s: r["failures"].append("qwen.disp.rr__l18__s1"), "not complete"),
    (lambda r, s: r["configs"][0].update({"seeds": [1, 2]}), "seed 1"),
    (lambda r, s: r["configs"].__setitem__(slice(None), [c for c in r["configs"] if (c["config_id"], c["trace_id"])
                                                          != ("qwen.rb-static.uniform", "l24")]), "no seed-1 point"),
    (lambda r, s: r["configs"].__setitem__(slice(None), [c for c in r["configs"] if (c["config_id"], c["trace_id"])
                                                          != ("qwen.rb-static-dec.q80l10c10", "l42")]), "no seed-1 point"),
    (lambda r, s: s["rows"][0].update({"enabled": True}), "already enabled"),
])
def test_refusals(mutate, msg):
    r, s = report(), scope()
    mutate(r, s)
    with pytest.raises(rr.RowError, match=msg):
        rr.build(r, s, "qwen")


def test_the_placeholder_rate_order_is_kept():
    sc = scope()
    sc["rows"][0]["blocks"] = [{"name": f"W9-{t}", "traces": [t]} for t in ("l30", "l12", "l24", "l18", "l36", "l42")]
    new, _ = rr.build(report(), sc, "qwen")
    assert [b["traces"][0] for b in next(r for r in new["rows"] if r["id"] == "W9")["blocks"]] == \
        ["l30", "l12", "l24", "l18", "l36", "l42"]


def test_placeholder_seeds_are_refused():
    r, sc = report(), scope()
    sc["rows"][2]["blocks"][0]["seeds"] = "$owner:seeds"
    r["coverage"]["pending"]["W2/W2"] = "an owner value"
    with pytest.raises(rr.RowError, match="not decided"):
        rr.build(r, sc, "qwen")


def test_other_groups_do_not_block():
    r = report()
    r["coverage"]["pending"]["W4l/W4-learned"] = "pins not given: xgb-heads-retrained.qwen"   # another group's block
    r["left_out"]["qwen.sweep.random__sweep6-36__s1"] = "unscored training prompts"           # not a main rate
    r["coverage"]["missing"].append("abl.qwen.rb-static.abl-lpt-off.q80l10c10__l12__s1")       # an aliased group
    r["gate_failures"].append("b30.qwen.rb-static.uniform__l16__s1")
    rr.build(r, scope(), "qwen")


def test_gemma_pending_router_block_is_refused():
    rep = report()
    for c in rep["configs"]:
        c["group"], c["config_id"] = "gemma", c["config_id"].replace("qwen.", "gemma.")
        c["trace_id"] = {"l12": "gl12", "l30": "gl30", "l42": "gl42"}.get(c["trace_id"], c["trace_id"])
    rep["coverage"]["pending"]["W8p/W8p"] = "held"
    with pytest.raises(rr.RowError, match="pending"):
        rr.build(rep, scope(), "gemma")


def test_unknown_configuration_is_refused():
    with pytest.raises(rr.RowError):
        rr.system("qwen", "qwen.vsr.mom-balance")
    with pytest.raises(rr.RowError):
        rr.system("qwen", "gemma.ap.pw025.lor")


def test_main_writes_the_scope(tmp_path, capsys):
    (tmp_path / "r.json").write_text(json.dumps(report()))
    (tmp_path / "s.json").write_text(json.dumps(scope()))
    assert rr.main(["--report", str(tmp_path / "r.json"), "--scope", str(tmp_path / "s.json"), "--family", "qwen",
                    "--out", str(tmp_path / "o.json")]) == 0
    out = json.loads((tmp_path / "o.json").read_text())
    assert next(r for r in out["rows"] if r["id"] == "W9")["enabled"]
    assert "total:" in capsys.readouterr().out
