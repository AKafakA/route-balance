"""The campaign render: every configuration renders to a valid scheduler document, every lease row to the
scope's cells, and nothing the owner or the lease has not supplied yet is invented."""
import copy
import json
import typing
from pathlib import Path

import pytest

from analysis.rules import PLACEHOLDER
from bench.schema import parse
from bench.scope import Scope, expected_cells, group_record
from routebalance.config import SchedulerConfig, parse_scheduler
from tools import campaign
from tools.campaign import PresetsFile, ScopeFile

REPO = Path(__file__).resolve().parents[2]
PRESETS_JSON = json.loads((REPO / "configs" / "presets.json").read_text())
SCOPE_JSON = json.loads((REPO / "configs" / "scope.json").read_text())
LPT_KEYS = typing.get_args(SchedulerConfig.model_fields["lpt"].annotation)

# Stand-ins for every owner value, used only to prove the files render once the owner has set them.
OWNER_PRESETS = {"parts.telemetry.poll_ms": 50.0, "parts.telemetry.max_age_ms": 500.0, "parts.telemetry.timeout_s": 2.0,
                 "parts.dispatch.seed": 1, "parts.rb.pending_decode": "own_length",
                 "parts.rounds.fixed32.timeout_ms": 200.0,
                 "parts.estimator.sweep": {"index_dir": "$pin:knn.{family}:path", "index_md5": "$pin:knn.{family}:md5",
                                           "encoder_revision": "$part:encoder_revision", "k": 50,
                                           "score_type": "deepeval", "prior": "knn"}}
OWNER_SCOPE = {
    "server.torch_threads": 4, "server.max_connections": 4096, "server.request_timeout_s": 600.0,
    "bench.max_connections": 4096, "bench.request_timeout_s": 600.0, "bench.max_send_lag_ms": 100.0,
    "datasets.eval_gemma.tokenizer_id": "tok-gemma", "datasets.train_gemma.tokenizer_id": "tok-gemma",
    "owner.g1_weights": ["q50l30c20", "uniform"],
    "owner.decoupled_weights": ["q50l30c20", "uniform", "q80l10c10", "q20l60c20"],
    "rows.W0.concurrency_per_instance": 16, "rows.W0.divergence_prompts": 200,
    "rows.W0.divergence_concurrency_per_instance": 64, "rows.W8-ref.concurrency_per_instance": 16,
    "rows.W8-ref.divergence_prompts": 200, "rows.W8-ref.divergence_concurrency_per_instance": 64,
    "rows.W1.levels": [1, 2, 4, 8, 16, 32], "rows.W1.requests_per_level": 64,
    "rows.W8-cal.levels": [1, 2, 4, 8, 16, 32], "rows.W8-cal.requests_per_level": 64,
    "traces.step6-30-6.segments": [{"n_requests": 360, "rate": 6.0}, {"n_requests": 1800, "rate": 30.0},
                                   {"n_requests": 360, "rate": 6.0}],
    "traces.gamma-cv4.segments": [{"n_requests": 3534, "rate": 18.0}],
    "traces.sweep6-36.segments": [{"n_requests": 2000, "rate": r} for r in (6.0, 12.0, 18.0, 24.0, 30.0, 36.0)],
    "blocks.W1b.seeds": [1, 2],
}
DECISIONS = {"rb_default": "rb", "headline": "q50l30c20", "w3_near_frontier": ["qwen.rb.q50l30c20", "qwen.ap.pw070.lor"],
             "w3_headline": ["qwen.rb.q50l30c20"], "w5_systems": ["qwen.rb.q50l30c20", "qwen.disp.lor"],
             "w7_systems": ["qwen.rb.q50l30c20", "qwen.rb-dec.uniform"]}
HELD = {"W4/W4-arm3", "W4/W4-const-batch", "W6/W6", "W7b/W7b"}     # must run, cannot yet: always pending
FLEETS = {"qwen_std": ("a" * 64, "1" * 64), "qwen_topo53": ("b" * 64, "2" * 64), "gemma": ("c" * 64, "3" * 64)}


def _set(doc: dict, dotted: str, value) -> None:
    head, _, rest = dotted.partition(".")
    if head == "rows":
        row_id, _, rest = rest.partition(".")
        (target,) = [r for r in doc["rows"] if r["id"] == row_id]
        target[rest] = value
    elif head == "blocks":
        name, _, key = rest.partition(".")
        (target,) = [b for r in doc["rows"] if r["kind"] == "cells" for b in r["blocks"] if b["name"] == name]
        target[key] = value
    elif head == "parts":
        name, _, key = rest.rpartition(".") if rest.count(".") and rest.rsplit(".", 1)[0] in doc["parts"] else (rest, "", "")
        if key:
            doc["parts"][name][key] = value
        else:
            doc["parts"][name] = value
    else:
        node = doc
        keys = dotted.split(".")
        for k in keys[:-1]:
            node = node[k]
        node[keys[-1]] = value


def filled(presets=True, scope=True, leave=(), unset=(), unset_scope=()):
    """The tracked files with the owner values filled (presets / scope), except `leave` (scope paths), and with
    `unset` (presets paths) and `unset_scope` (scope paths) forced back to OWNER-MUST-SET, whatever the tracked
    files hold by now."""
    p, s = copy.deepcopy(PRESETS_JSON), copy.deepcopy(SCOPE_JSON)
    for path, v in (OWNER_PRESETS.items() if presets else ()):
        _set(p, path, v)
    for path in unset:
        _set(p, path, "OWNER-MUST-SET")
    for path, v in (OWNER_SCOPE.items() if scope else ()):
        if path not in leave:
            _set(s, path, v)
    for path in unset_scope:
        _set(s, path, "OWNER-MUST-SET")
    if "min" not in LPT_KEYS:          # W11 needs the lpt key min / mean (S5b); this tree predates it
        (w11,) = [r for r in s["rows"] if r["id"] == "W11"]
        w11.update(enabled=False, why="this tree's configuration has no lpt min / mean")
    return parse(PresetsFile, p, "presets"), parse(ScopeFile, s, "scope")


def pins(presets: PresetsFile) -> dict:
    names = campaign.referenced_pins(presets, campaign.expand(presets))
    return {n: {"path": f"art/{n}", "md5": "0" * 32} for n in sorted(names)}


def full_render(**over):
    presets, scope = filled()
    kw = {"pins": pins(presets), "decisions": DECISIONS, "fleets": FLEETS} | over
    return campaign.render(presets, scope, **kw)


# ---------------------------------------------------------------- the tracked files

def test_the_tracked_files_dry_run_every_row():
    summary = campaign.dry_run(campaign.load_presets(), campaign.load_scope_file())
    rows = {r["id"]: r for r in summary["rows"]}
    assert list(rows) == ["W0", "W1", "G1", "W1b", "W2", "W2b", "W4", "W3", "W5", "W6", "W7", "W7b", "W8-ref",
                          "W8-cal", "W8", "W9", "W11", "W10"]
    cells = {k: r["cells"] for k, r in rows.items() if r["kind"] == "cells" and r["enabled"]}
    assert cells == {"G1": 32, "W1b": 0, "W2": 130, "W2b": 20, "W4": 96, "W3": 100, "W5": 30, "W6": 2, "W7": 36,
                     "W7b": 1, "W8": 108, "W9": 260, "W11": 12}
    held = {f"{r['id']}/{b['name']}" for r in rows.values() if r["kind"] == "cells" for b in r["blocks"] if b.get("hold")}
    assert held == {"W4/W4-arm3", "W4/W4-const-batch", "W6/W6", "W7b/W7b"}
    assert summary["decisions_needed"] == ["headline", "rb_default", "w3_headline", "w3_near_frontier", "w5_systems",
                                           "w7_systems"]
    unset = summary["owner_unset"]
    assert not any("parts.telemetry" in u or "parts.rb.pending_decode" in u or "parts.dispatch.seed" in u
                   or u.startswith(("server.", "bench.")) for u in unset)   # owner 09-25
    for must in (
                 "presets parts.estimator.sweep", "owner.g1_weights[0]",
                 "W1.requests_per_level", "W0.concurrency_per_instance"):
        assert must in unset
    assert "tpot-table.qwen" in summary["pins_needed"] and "xgb-heads.qwen" in summary["pins_needed"]
    assert "W4" in campaign.dry_run_text(summary)


def test_the_owner_stand_ins_cover_every_placeholder():
    presets, scope = filled()
    assert campaign.placeholders(presets.model_dump(mode="json")) == []
    enabled_rows = [r for r in scope.model_dump(mode="json")["rows"] if r["enabled"]]
    assert campaign.placeholders(scope.model_dump(mode="json", exclude={"rows"})) == []
    assert campaign.placeholders(enabled_rows) == []


def test_every_configuration_renders_to_a_valid_scheduler_document():
    presets, _ = filled()
    resolver = campaign.Resolver(presets, pins(presets))
    expanded = campaign.expand(presets)
    assert len(expanded) == 416        # 358 + the G1 read-mode controls (D54, D55): 3 x 16 RB-weight variants + 10 AP
    for p in expanded.values():
        if p.config["lpt"] == "{k}" and "min" not in LPT_KEYS:
            continue
        got, needs = campaign.render_config(p, resolver)
        assert needs.empty() and got is not None, p.preset_id
        assert campaign.config_digest(parse_scheduler(got.document)) == got.digest


def test_uniform_is_one_third_each_and_the_drops_renormalize():
    presets, _ = filled()
    resolver = campaign.Resolver(presets, pins(presets))
    expanded = campaign.expand(presets)
    doc = campaign.render_config(expanded["qwen.rb.uniform"], resolver)[0].document
    assert doc["weights"] == {"w_quality": 1 / 3, "w_latency": 1 / 3, "w_cost": 1 / 3}
    doc = campaign.render_config(expanded["qwen.abl-wlat0.q50l30c20"], resolver)[0].document
    assert doc["weights"] == {"w_quality": 0.5 / 0.7, "w_latency": 0.0, "w_cost": 0.2 / 0.7}
    doc = campaign.render_config(expanded["qwen.abl-wq0.q80l10c10"], resolver)[0].document
    assert doc["weights"]["w_quality"] == 0.0 and doc["weights"]["w_latency"] == pytest.approx(0.5)


def test_the_baseline_knobs_are_the_plans():
    presets, _ = filled()
    resolver = campaign.Resolver(presets, pins(presets))
    expanded = campaign.expand(presets)

    def doc(pid):
        return campaign.render_config(expanded[pid], resolver)[0].document
    ap_q = doc("qwen.ap.pw070.lor")["pipeline"]["router"]["kwargs"]
    ap_g = doc("gemma.ap.pw070.lor")["pipeline"]["router"]["kwargs"]
    assert (ap_q["beta"], ap_q["top_k"], ap_g["top_k"]) == (5.0, 4, 3)
    assert ap_q["embedder_revision"] == "1110a243fdf4706b3f48f1d95db1a4f5529b4d41" and len(ap_q["upstream_sha256"]) == 64
    br = doc("qwen.br.t03.po2")["pipeline"]
    assert br["router"]["kwargs"]["max_length"] == 512 and br["router"]["kwargs"]["confidence_threshold"] == 0.3
    assert br["dispatch"]["type"] == "po2"
    assert doc("qwen.disp.lowest-latency")["pipeline"]["dispatch"]["latency_window"] == 10
    sweep = doc("qwen.sweep.random")
    assert (sweep["mode"], sweep["pending_decode"], sweep["pipeline"]["dispatch"]["type"]) == ("pipeline", "own_length",
                                                                                                "random")
    assert not any(pid.startswith("gemma.br.") for pid in expanded)
    assert not any("shortest" in json.dumps(doc(pid)) for pid in ("qwen.disp.lor", "qwen.disp.rr"))


def test_the_tracked_files_render_with_every_lease_input():
    result = full_render()
    scope = parse(Scope, result.scope, "rendered")
    cells = expected_cells(scope)
    by_block = {}
    for c in cells:
        by_block[c.block] = by_block.get(c.block, 0) + 1
    assert by_block["G1"] == 32 and by_block["W2b"] == 20 and by_block["W4-ablations"] == 72
    assert all(cid in result.configs for cid in scope.configs)
    for cid, doc in result.configs.items():
        assert campaign.config_digest(parse_scheduler(doc)) == scope.configs[cid].config_digest
    # the same configuration under another alias is the same document and digest, with its own cells
    assert result.configs["abl.qwen.rb.q50l30c20"] == result.configs["qwen.rb.q50l30c20"]
    assert scope.configs["abl.qwen.rb.q50l30c20"].config_digest == scope.configs["qwen.rb.q50l30c20"].config_digest
    assert "abl.qwen.rb.q50l30c20__l12__s1" in {c.cell_id for c in cells}
    assert set(result.pending) == HELD and result.blocked_rows == {} and scope.pending == result.pending


def test_groups_declare_what_each_cell_may_differ_in():
    scope = parse(Scope, full_render().scope, "rendered")
    cells = {c.cell_id: c for c in expected_cells(scope)}
    ref = group_record(scope, cells["abl.qwen.rb.q50l30c20__l12__s1"])
    assert ref == {"group": "qwen-ablation", "reference": "abl.qwen.rb.q50l30c20", "intended_differences": []}
    arm = group_record(scope, cells["abl.qwen.abl-dr-off.q50l30c20__l30__s3"])
    assert arm["reference"] == "abl.qwen.rb.q50l30c20" and arm["intended_differences"] == ["telemetry.dead_reckoning"]
    assert group_record(scope, cells["abl.qwen.rb-static.q50l30c20__l12__s2"])["intended_differences"] == [
        "system.latency.source", "system.latency.envelope", "system.latency.normalization"]
    main = group_record(scope, cells["qwen.ap.pw070.lor__l30__s1"])
    assert main == {"group": "qwen", "reference": None, "intended_differences": ["system.*"]}
    assert group_record(scope, cells["qwen.sweep.random__sweep6-36__s1"])["intended_differences"] == []


def test_fill_blocks_do_not_repeat_cells_and_plain_blocks_refuse_to():
    scope = parse(Scope, full_render().scope, "rendered")
    cells = {c.cell_id: c for c in expected_cells(scope)}
    assert cells["qwen.rb.q50l30c20__l12__s2"].block == "W3-near-frontier"     # W9 lists it too, as fill
    assert cells["qwen.rb.q20l20c60__l12__s2"].block == "W9-l12"
    presets, scope_file = filled()
    raw = scope_file.model_dump(mode="json")
    (w9,) = [r for r in raw["rows"] if r["id"] == "W9"]
    w9["blocks"][1]["fill"] = False
    with pytest.raises(campaign.ConfigError, match="listed by an earlier block"):
        campaign.render(presets, parse(ScopeFile, raw, "scope"), pins=pins(presets), decisions=DECISIONS, fleets=FLEETS)


# ---------------------------------------------------------------- staging: nothing missing is invented

def test_a_missing_pin_leaves_its_blocks_pending():
    presets, scope = filled()
    p = pins(presets)
    del p["xgb-heads.qwen"]
    result = campaign.render(presets, scope, pins=p, decisions=DECISIONS, fleets=FLEETS)
    assert "W4/W4-xgb" in result.pending and "xgb-heads.qwen" in result.pending["W4/W4-xgb"]
    assert "W4/W4-ablations" not in result.pending
    assert result.scope["pending"] == result.pending


def test_missing_decisions_and_fleets_leave_blocks_pending():
    presets, scope = filled()
    result = campaign.render(presets, scope, pins=pins(presets), decisions={}, fleets={"qwen_std": FLEETS["qwen_std"]})
    assert "rb_default" in result.pending["W2/W2-rb-l30"] and "headline" in result.pending["W4/W4-reference"]
    assert "fleet 'gemma' not given" in result.pending["W8/W8"]
    assert "W2/W2-base-l30" not in result.pending and "G1/G1" not in result.pending
    assert result.blocked_rows["W8-ref"] == "fleet 'gemma' not given"


def test_owner_values_are_never_invented():
    presets, scope = filled(unset={"parts.telemetry.poll_ms"})   # a value every configuration reads
    result = campaign.render(presets, scope, pins=pins(presets), decisions=DECISIONS, fleets=FLEETS)
    assert result.scope is None and result.configs == {}
    assert all("parts.telemetry.poll_ms" in r for b, r in result.pending.items() if b not in HELD)
    presets, scope = filled(unset_scope={"server.torch_threads"})   # the launch values gate every cell (D44)
    result = campaign.render(presets, scope, pins=pins(presets), decisions=DECISIONS, fleets=FLEETS)
    assert result.scope is None and all("server.torch_threads" in r for r in result.pending.values())
    assert all(result.pending[b].startswith("OWNER-MUST-SET") and "held:" in result.pending[b] for b in HELD)
    assert result.blocked_rows == {}                           # the direct rows do not need them
    presets, scope = filled(unset_scope={"bench.request_timeout_s"})   # a bench value unset: no scope at all
    with pytest.raises(campaign.ConfigError, match="owner values not set yet"):
        campaign.render(presets, scope, pins=pins(presets), decisions=DECISIONS, fleets=FLEETS)
    presets, scope = filled(leave={"owner.g1_weights"})     # one owner list unset: only its blocks wait
    result = campaign.render(presets, scope, pins=pins(presets), decisions=DECISIONS, fleets=FLEETS)
    assert set(result.pending) == {"G1/G1"} | HELD and "owner.g1_weights[0]" in result.pending["G1/G1"]


def test_the_first_render_before_w1_gives_the_direct_rows(tmp_path):
    presets, scope = filled()
    early = {k: v for k, v in pins(presets).items() if not k.startswith(("tpot-table", "xgb"))}
    result = campaign.render(presets, scope, pins=early, decisions={}, fleets=FLEETS)
    assert result.scope is None and result.pending and result.blocked_rows == {}
    campaign.write_render(result, tmp_path / "r0", {})
    assert not (tmp_path / "r0" / "scope.json").exists()
    plan = campaign.load_plan(tmp_path / "r0")
    assert campaign.plan_row(plan, "W1", "calibration")["params"]["requests_per_level"] == 64
    assert plan["bench"]["sampling"]["frequency_penalty"] == 1.2
    later = campaign.render(presets, scope, pins=pins(presets), decisions={}, fleets=FLEETS)
    campaign.check_previous(later, tmp_path / "r0")


def test_rb_decoupled_follows_the_default_arm():
    presets, scope = filled()
    rb = campaign.render(presets, scope, pins=pins(presets), decisions=DECISIONS, fleets=FLEETS)
    static = campaign.render(presets, scope, pins=pins(presets), decisions={**DECISIONS, "rb_default": "rb-static"},
                             fleets=FLEETS)
    assert "qwen.rb-dec.uniform" in rb.configs and "qwen.rb-static-dec.uniform" not in rb.configs
    assert "qwen.rb-static-dec.uniform" in static.configs and "qwen.rb-dec.uniform" not in static.configs
    assert static.configs["qwen.rb-static-dec.uniform"]["latency"] == static.configs["qwen.rb-static.uniform"]["latency"]
    assert "g1.qwen.rb-dec.uniform" in static.configs                  # G1 pairs RB-new with its own control
    pending = campaign.render(presets, scope, pins=pins(presets), decisions={}, fleets=FLEETS).pending
    assert "rb_default" in pending["W2/W2-dec-l30"] and "W2/W2-base-l30" not in pending


def test_a_fill_block_waits_for_the_earlier_blocks_that_could_list_its_cells(tmp_path):
    presets, scope = filled()
    p = pins(presets)
    first = campaign.render(presets, scope, pins=p, decisions={"rb_default": "rb"}, fleets=FLEETS)
    assert "waits for W3/W3-near-frontier" in first.pending["W9/W9-l12"]
    campaign.write_render(first, tmp_path / "r1", {})
    later = campaign.render(presets, scope, pins=p, decisions=DECISIONS, fleets=FLEETS)
    campaign.check_previous(later, tmp_path / "r1")                     # nothing moved between blocks


def test_the_pins_and_decisions_files_are_checked():
    presets, scope = filled()
    with pytest.raises(campaign.ConfigError, match="referenced by no configuration"):
        campaign.render(presets, scope, pins={**pins(presets), "knn.qwem": {"path": "x", "md5": "0" * 32}},
                        decisions=DECISIONS, fleets=FLEETS)
    with pytest.raises(campaign.ConfigError, match="32 hex"):
        campaign.render(presets, scope, pins={**pins(presets), "knn.qwen": {"path": "x", "md5": "zz"}},
                        decisions=DECISIONS, fleets=FLEETS)
    with pytest.raises(campaign.ConfigError, match="not declared"):
        campaign.render(presets, scope, pins=pins(presets), decisions={**DECISIONS, "typo": "x"}, fleets=FLEETS)
    with pytest.raises(campaign.ConfigError, match="no template defines"):
        campaign.render(presets, scope, pins=pins(presets), decisions={**DECISIONS, "w5_systems": ["qwen.nope"]},
                        fleets=FLEETS)
    with pytest.raises(campaign.ConfigError, match="another family"):
        campaign.render(presets, scope, pins=pins(presets), decisions={**DECISIONS, "w5_systems": ["gemma.disp.lor"]},
                        fleets=FLEETS)


def test_a_later_stage_keeps_every_earlier_configuration_and_cell(tmp_path):
    presets, scope = filled()
    p = pins(presets)
    early = campaign.render(presets, scope, pins={k: v for k, v in p.items() if not k.startswith("xgb")},
                            decisions=DECISIONS, fleets=FLEETS)
    campaign.write_render(early, tmp_path / "r1", {})
    later = campaign.render(presets, scope, pins=p, decisions=DECISIONS, fleets=FLEETS)
    campaign.check_previous(later, tmp_path / "r1")
    changed = campaign.render(presets, scope, pins={**p, "knn.qwen": {"path": "art/other", "md5": "1" * 32}},
                              decisions=DECISIONS, fleets=FLEETS)
    with pytest.raises(campaign.ConfigError, match="changed since"):
        campaign.check_previous(changed, tmp_path / "r1")
    moved = campaign.render(presets, scope, pins=p, decisions={**DECISIONS, "w3_near_frontier": ["qwen.ap.pw070.lor"]},
                            fleets=FLEETS)
    with pytest.raises(campaign.ConfigError, match="changed block|no longer in the scope"):
        campaign.check_previous(moved, tmp_path / "r1")
    with pytest.raises(campaign.ConfigError, match="exists"):
        campaign.write_render(later, tmp_path / "r1", {})


def test_a_render_directory_holds_what_the_drivers_read(tmp_path):
    result = full_render()
    campaign.write_render(result, tmp_path / "r", {"presets_sha256": "x"})
    plan = campaign.load_plan(tmp_path / "r")
    assert plan["warmup"]["requests_per_instance"] == 2 and plan["drain"]["timeout_s"] == 300.0
    w1 = campaign.plan_row(plan, "W1", "calibration")
    assert w1["params"]["levels"] == [1, 2, 4, 8, 16, 32] and w1["params"]["dataset"] == "train_qwen"
    with pytest.raises(campaign.ConfigError, match="not calibration"):
        campaign.plan_row(plan, "W0", "calibration")
    parse(Scope, json.loads((tmp_path / "r" / "scope.json").read_text()), "scope")
    assert json.loads((tmp_path / "r" / "configs" / "qwen.disp.lor.json").read_text()) == result.configs["qwen.disp.lor"]


# ---------------------------------------------------------------- the template language

def _mini(templates, parts=None, lists=None):
    return parse(PresetsFile, {"notes": {}, "lists": lists or {}, "parts": parts or {}, "templates": templates},
                 "presets")


BASE = copy.deepcopy(next(t for t in PRESETS_JSON["templates"] if t["id"] == "{family}.disp.{d}"))


def test_templates_expand_over_their_axes_and_ids_are_unique():
    t = {**BASE, "axes": {"family": ["qwen"], "d": ["rr", "lor"]}}
    got = campaign.expand(_mini([t], PRESETS_JSON["parts"]))
    assert list(got) == ["qwen.disp.rr", "qwen.disp.lor"] and got["qwen.disp.rr"].dispatch == "rr"
    with pytest.raises(campaign.ConfigError, match="defined twice"):
        campaign.expand(_mini([t, t], PRESETS_JSON["parts"]))
    with pytest.raises(campaign.ConfigError, match="family axis"):
        campaign.expand(_mini([{**t, "axes": {"d": ["rr"]}, "id": "x.{d}"}]))
    with pytest.raises(campaign.ConfigError, match="states every key"):
        campaign.expand(_mini([{**t, "config": {k: v for k, v in t["config"].items() if k != "budget"}}]))
    with pytest.raises(campaign.ConfigError, match="unknown variable"):
        campaign.expand(_mini([{**t, "knob": "{nope}"}]))


def test_references_resolve_or_fail_loudly():
    parts = {"a": "$part:b", "b": "$part:a", "w": {"w_quality": 1.0, "w_latency": 0.0, "w_cost": 0.0}}
    r = campaign.Resolver(_mini([BASE], parts), {"p": {"path": "x"}})
    needs = campaign.Needs()
    with pytest.raises(campaign.ConfigError, match="cycle"):
        r.resolve("$part:a", "t", needs, {})
    with pytest.raises(campaign.ConfigError, match="unknown reference"):
        r.resolve("$weird:x", "t", needs, {})
    with pytest.raises(campaign.ConfigError, match="no 'md5'"):
        r.resolve("$pin:p:md5", "t", needs, {})
    assert r.resolve("$pin:q:path", "t", needs, {}) is campaign.MISSING and needs.pins == {"q"}
    with pytest.raises(campaign.ConfigError, match="leaves no weight"):
        r.resolve("$without:w_quality:w", "t", campaign.Needs(), {})
    with pytest.raises(campaign.ConfigError, match="drops one of"):
        r.resolve("$without:w_balance:w", "t", campaign.Needs(), {})
    assert r.resolve(PLACEHOLDER, "t", needs, {}) == PLACEHOLDER and needs.owner == {"presets t"}


def test_a_held_block_keeps_the_campaign_incomplete(tmp_path):
    result = full_render()
    assert result.pending["W4/W4-arm3"].startswith("held: v2's four-arm isolation arm 3")
    scope = parse(Scope, result.scope, "rendered")
    assert "W4/W4-arm3" in scope.pending                     # analysis.manifest never reports COMPLETE with it
    assert not any(c.block == "W4-arm3" for c in expected_cells(scope))
