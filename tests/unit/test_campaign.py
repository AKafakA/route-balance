"""The campaign render: every configuration renders to a valid scheduler document, every lease row to the
scope's cells, and nothing the owner or the lease has not supplied yet is invented."""
import copy
import hashlib
import json
import typing
from pathlib import Path

import pytest

from analysis.rules import PLACEHOLDER
from bench.schema import parse
from bench.scope import Scope, expected_cells, group_record
from routebalance.config import ConfigError, SchedulerConfig, parse_scheduler
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
    "rows.W1.levels": [1, 2, 4, 8, 16, 32], "rows.W1.requests_per_level": 64,
    "rows.W8-cal.levels": [1, 2, 4, 8, 16, 32], "rows.W8-cal.requests_per_level": 64,
    "traces.sweep6-36.segments": [{"n_requests": 2000, "rate": r} for r in (6.0, 12.0, 18.0, 24.0, 30.0, 36.0)],
    "blocks.W1b.seeds": [1, 2],
}
LATENCY_PATHS = ["system.latency.*"]
DECISIONS = {"rb_default": "rb", "decoupled_arm": "rb", "headline": "q50l30c20",
             "w3_near_frontier": ["qwen.rb.q50l30c20", "qwen.ap.pw070.lor"],
             "w3_headline": ["qwen.rb.q50l30c20"], "w5_systems": ["qwen.rb.q50l30c20", "qwen.disp.lor"],
             "w7_systems": ["qwen.rb.q50l30c20", "qwen.rb-dec.uniform"]}
W3_DECISIONS = {k: DECISIONS[k] for k in ("w3_near_frontier", "w3_headline")}   # W3 is disabled since D121
HELD: set[str] = set()     # blocks that must run but cannot yet (none since 09-25): always pending
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
    if "min" not in LPT_KEYS:          # the LPT-key blocks need the lpt key min / mean (S5b); this tree predates it
        for r in s["rows"]:
            for b in r.get("blocks", []):
                if b["name"] in ("W4-lpt-key", "W4r-lpt-key"):
                    b.update(enabled=False, why="this tree's configuration has no lpt min / mean")
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
    summary = campaign.dry_run(campaign.load_presets(campaign.PRESETS), campaign.load_scope_file(campaign.SCOPE))
    rows = {r["id"]: r for r in summary["rows"]}
    assert list(rows) == ["W1", "G1", "W1b", "W1c", "G1b", "G1c", "W2", "W2b", "W4", "W3", "W5", "W6", "W7", "W8-cal", "W8",
                          "W8r", "W7r", "W9", "W4r", "W5r", "W7b", "W11", "W12", "W13", "W10"]   # D121: seed 1 first
    cells = {k: r["cells"] for k, r in rows.items() if r["kind"] == "cells" and r["enabled"]}
    assert cells == {"G1": 66, "W1b": 24, "W1c": 6, "G1b": 6, "G1c": 16, "W2": 156, "W2b": 20, "W4": 51, "W5": 27, "W7": 18,
                     "W8": 78, "W8r": 156, "W7r": 36, "W9": 312, "W4r": 102, "W5r": 54, "W7b": 9, "W12": 12, "W13": 7}
    # W8: the Qwen grid's baselines (D70); W12: the in-flight dispatchers (D84); W13: leave one source out (D93);
    # D121: seed 1 of every main row first (W4 with the LPT-key variants, W5, W7, W8), their seeds 2-3 after (W8r, W7r,
    # W4r, W5r) with the full grid's (W9, W3 merged in); W3 and W11 disabled
    held = {f"{r['id']}/{b['name']}" for r in rows.values() if r["kind"] == "cells" for b in r["blocks"] if b.get("hold")}
    assert held == HELD
    assert summary["decisions_needed"] == ["decoupled_arm", "headline", "rb_default", "w5_systems",
                                           "w7_systems"]                  # D121: no seed picks
    assert summary["owner_unset"] == []                                    # every owner value is set (09-25)
    assert {"tpot-table.qwen", "prefill-table.qwen", "xgb-heads.qwen"} <= set(summary["pins_needed"])
    assert "W4" in campaign.dry_run_text(summary)


def test_the_burst_sweep_and_the_three_rates():
    """D125: the burst sweep's traces (vLLM bench's burstiness: the gamma shape, the mean rate fixed) and the three rates
    of the ablations, the topology and Gemma rows."""
    traces = SCOPE_JSON["traces"]
    bursts = {f"ga{a}-l{r}": (b, float(r)) for a, b in (("025", 0.25), ("05", 0.5), ("2", 2.0)) for r in (12, 30, 42)}
    for name, (b, rate) in bursts.items():
        assert traces[name] == {"burstiness": b, "segments": [{"n_requests": 3532, "rate": rate}]}, name
    rows = {r["id"]: r for r in SCOPE_JSON["rows"]}
    for rid in ("W5", "W5r"):          # alpha 1 is the main grid's own Poisson run at each rate and seed
        assert {t for b in rows[rid]["blocks"] for t in b["traces"]} == set(bursts)
    for rid, want in (("W4", {"l12", "l30", "l42"}), ("W4r", {"l12", "l30", "l42"}), ("W7", {"l12", "l30", "l42"}),
                      ("W7r", {"l12", "l30", "l42"}), ("W8", {"gl12", "gl30", "gl42"}), ("W8r", {"gl12", "gl30", "gl42"})):
        assert {t for b in rows[rid]["blocks"] if b["enabled"] for t in b["traces"]} == want, rid
    assert [traces[g]["segments"][0]["n_requests"] for g in ("gl12", "gl30", "gl42")] == [1979] * 3


def test_the_latency_sweep_sends_every_training_prompt_once_a_run():
    for trace in ("sweep6-36", "sweep38-42"):                  # W1b and its 38-42 req/s extension W1c (D119, D120)
        sweep = sum(s["n_requests"] for s in SCOPE_JSON["traces"][trace]["segments"])
        assert sweep == SCOPE_JSON["datasets"]["train_qwen"]["rows"] == 14919, trace
    assert [s["rate"] for s in SCOPE_JSON["traces"]["sweep38-42"]["segments"]] == [38.0, 40.0, 42.0]


def test_the_time_varying_load_is_v2s():
    """Owner 09-25: v2's square wave (900 requests at 30 req/s, then 180 at 6: 30 s each) and gamma bursts with
    burstiness 0.3 at 18 req/s, each over the whole evaluation set."""
    square, gamma = SCOPE_JSON["traces"]["square30-6"], SCOPE_JSON["traces"]["gamma-b03"]
    phases = [(g["n_requests"], g["rate"]) for g in square["segments"]]
    assert phases == [(900, 30.0), (180, 6.0)] * 3 + [(292, 30.0)] and square["burstiness"] == 1.0
    assert all(n / r == 30.0 for n, r in phases[:-1])
    assert gamma == {"burstiness": 0.3, "segments": [{"n_requests": 3532, "rate": 18.0}]}
    assert sum(n for n, _ in phases) == SCOPE_JSON["datasets"]["eval_qwen"]["rows"] == 3532


def test_every_fleet_family_has_its_scored_corpus(tmp_path):
    scope = copy.deepcopy(SCOPE_JSON)
    assert set(scope["scored_corpora"]) == {"qwen", "gemma"}
    del scope["scored_corpora"]["gemma"]
    (tmp_path / "scope.json").write_text(json.dumps(scope))
    with pytest.raises(campaign.ConfigError, match="one corpus per fleet family"):
        campaign.load_scope_file(tmp_path / "scope.json")


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
    assert len(expanded) == 961        # 358; the G1 read-mode controls (3 x 16 + 10 AP); 8 rb-learned (Qwen only);
    #                                    the budget sweep's filter arms (rb, rb-static x 2 families; rb-learned, Qwen);
    #                                    Gemma BEST-Route, 4 t x 3 dispatchers (D70); the power-of-two probe
    #                                    controls (10 AP + 2 dispatcher-only, D59); the mechanism ablations once per
    #                                    RouteBalance latency variant (11 x 2 families x 8 weights x 2 more), and
    #                                    abl-xgb gone (-16: rb-learned is the learned variant in W4, D78); the
    #                                    dispatch-only least-outstanding and power of two counting in flight
    #                                    (2 x 2 families, the appendix row W12, D84); the leave-one-source-out
    #                                    indexes (7 sources x 3 variants x 8 weights, Qwen only, the optional W13, D93)
    for p in expanded.values():
        if p.config["lpt"] == "{k}" and "min" not in LPT_KEYS:
            continue
        got, needs = campaign.render_config(p, resolver)
        assert needs.empty() and got is not None, p.preset_id
        assert campaign.config_digest(parse_scheduler(got.document)) == got.digest
        table = got.document["latency"]["prefill"]["table"]            # D118: in every variant, the family's table
        assert table.startswith("art/prefill-table.") and table.split(".")[-1] in p.preset_id.split("."), p.preset_id


def test_uniform_is_one_third_each_and_the_drops_renormalize():
    presets, _ = filled()
    resolver = campaign.Resolver(presets, pins(presets))
    expanded = campaign.expand(presets)
    doc = campaign.render_config(expanded["qwen.rb.uniform"], resolver)[0].document
    assert doc["weights"] == {"w_quality": 1 / 3, "w_latency": 1 / 3, "w_cost": 1 / 3}
    doc = campaign.render_config(expanded["qwen.rb.abl-wlat0.q50l30c20"], resolver)[0].document
    assert doc["weights"] == {"w_quality": 0.5 / 0.7, "w_latency": 0.0, "w_cost": 0.2 / 0.7}
    doc = campaign.render_config(expanded["qwen.rb.abl-wq0.q80l10c10"], resolver)[0].document
    assert doc["weights"]["w_quality"] == 0.0 and doc["weights"]["w_latency"] == pytest.approx(0.5)


def test_the_baseline_knobs_are_the_plans():
    presets, _ = filled()
    resolver = campaign.Resolver(presets, pins(presets))
    expanded = campaign.expand(presets)

    def doc(pid):
        return campaign.render_config(expanded[pid], resolver)[0].document
    ap_q = doc("qwen.ap.pw070.lor")["pipeline"]["router"]["kwargs"]
    ap_g = doc("gemma.ap.pw070.lor")["pipeline"]["router"]["kwargs"]
    assert (ap_q["beta"], ap_q["top_k"], ap_g["beta"], ap_g["top_k"]) == (9.0, 1, 9.0, 1)   # upstream's defaults (D89)
    assert ap_q["embedder_revision"] == "1110a243fdf4706b3f48f1d95db1a4f5529b4d41" and len(ap_q["upstream_sha256"]) == 64
    br = doc("qwen.br.t03.po2")["pipeline"]
    kw = br["router"]["kwargs"]      # BEST-Route as its authors ship it (D79): upstream's ranker and routing rule
    assert br["router"]["type"] == "best_route_upstream" and kw["threshold"] == 0.3
    assert (kw["base_model"], kw["base_revision"]) == ("microsoft/deberta-v3-small",
                                                       "a36c739020e01763fe789b4b85e2df55d6180012")
    assert kw["upstream_sha256"] == "85279e1f6080d87db95dcbded46d7713fcf2728b82339c3b6a30ef3b56579005"
    assert br["dispatch"]["type"] == "po2"
    assert doc("qwen.disp.lowest-latency")["pipeline"]["dispatch"]["latency_window"] == 10
    # D84: the main grid's load-reading dispatchers read the engines' cached gauges, alone, behind the routers and
    # behind RB-decoupled; the probes fetch the gauges per request (v3); the appendix row W12 counts in flight
    def read(pid):
        d = doc(pid)
        return d["telemetry_read"], d["pipeline"]["dispatch"]["count"]
    for pid in ("qwen.disp.lor", "qwen.disp.po2", "qwen.ap.pw070.lor", "qwen.ap.pw080.po2", "qwen.br.t03.lor",
                "qwen.br.t00.po2", "qwen.rb-dec.uniform", "qwen.rb-static-dec.uniform", "gemma.disp.lor",
                "gemma.ap.pw070.lor"):
        assert read(pid) == ("poll", "engine"), pid
    for pid in ("qwen.ap.pw070.lor-probe", "qwen.ap.pw080.po2-probe", "qwen.rb-dec-probe.uniform", "qwen.disp.po2-probe"):
        assert read(pid) == ("per_request", "engine"), pid
    for pid in ("qwen.disp.lor-inflight", "qwen.disp.po2-inflight"):
        assert read(pid) == ("poll", "in_flight"), pid
    assert doc("qwen.disp.rr")["pipeline"]["dispatch"]["count"] is None
    sweep = doc("qwen.sweep.random")
    assert (sweep["mode"], sweep["pending_decode"], sweep["pipeline"]["dispatch"]["type"]) == ("pipeline", "own_length",
                                                                                                "random")
    br_g = doc("gemma.br.t05.lor")["pipeline"]["router"]                   # D70: Gemma gets BEST-Route too
    assert br_g["type"] == "best_route_upstream" and br_g["kwargs"]["checkpoint"] == "art/br.gemma"
    # the owner's backup (D79): v3's 4-way recipe, with D71's fallbacks (4B for Gemma, 3B for Qwen)
    v3 = {f: resolver.resolve("$part:br.router.v3", "test", campaign.Needs(), {"family": f, "t": "t05"})["kwargs"]
          for f in ("gemma", "qwen")}
    assert (v3["gemma"]["fallback_model"], v3["qwen"]["fallback_model"]) == ("google/gemma-3-4b-it", "Qwen/Qwen2.5-3B")
    assert not any("shortest" in json.dumps(doc(pid)) for pid in ("qwen.disp.lor", "qwen.disp.rr"))


def test_the_tracked_files_render_with_every_lease_input():
    result = full_render()
    scope = parse(Scope, result.scope, "rendered")
    cells = expected_cells(scope)
    by_block = {}
    for c in cells:
        by_block[c.block] = by_block.get(c.block, 0) + 1
    assert by_block["G1"] == 66 and by_block["W2b"] == 20 and by_block["W4-mechanisms"] == 27   # seed 1, 3 rates (D125)
    assert (by_block["W4-latency"], by_block["W4-learned"], by_block["W4-const-batch"]) == (12, 3, 3)
    assert (by_block["W4r-mechanisms"], by_block["W4-lpt-key"], by_block["W4r-lpt-key"]) == (54, 6, 12)
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
    arm = group_record(scope, cells["abl.qwen.rb.abl-dr-off.q50l30c20__l30__s3"])
    assert arm["reference"] == "abl.qwen.rb.q50l30c20" and arm["intended_differences"] == ["telemetry.dead_reckoning"]
    assert group_record(scope, cells["abl.qwen.rb-static.q50l30c20__l12__s2"])["intended_differences"] == LATENCY_PATHS
    main = group_record(scope, cells["qwen.ap.pw070.lor__l30__s1"])
    assert main == {"group": "qwen", "reference": None, "intended_differences": ["system.*", "telemetry.dead_reckoning"]}
    assert group_record(scope, cells["qwen.sweep.random__sweep6-36__s1"])["intended_differences"] == []


def _executed_sections(doc: dict) -> dict:
    """The configuration-derived part of the scheduler's manifest (routebalance/server.py), flattened as parity
    compares it: system.*, budget.*, and the dead-reckoning and timeout keys of telemetry."""
    from gates.cell import flatten
    cfg, dump = parse_scheduler(doc), (lambda s: None if s is None else s.model_dump(mode="json"))
    pipe = cfg.pipeline
    return flatten({"system": {"mode": cfg.mode, "telemetry_read": cfg.telemetry_read, "weights": dump(cfg.weights),
                               "rounds": dump(cfg.rounds), "lpt": cfg.lpt, "pending_decode": cfg.pending_decode,
                               "latency": dump(cfg.latency), "estimator": dump(cfg.estimator),
                               "router": None if pipe is None else dump(pipe.router),
                               "dispatch": None if pipe is None else dump(pipe.dispatch)},
                    "budget": dump(cfg.budget),
                    "telemetry": {"timeout_s": cfg.telemetry.timeout_s, "dead_reckoning": cfg.dead_reckoning}})


def test_every_main_groups_configurations_differ_only_as_declared():
    """What the parity gate will find on the lease, checked on the rendered documents: in a main comparison
    group, two configurations differ only in keys the group's patterns match (found 09-25: the budget groups
    declared budget.admission_filter only, while the filter arm also changed another budget key)."""
    from gates.parity import matches
    result = full_render()
    scope = parse(Scope, result.scope, "rendered")
    by_group: dict[str, set[str]] = {}
    for c in expected_cells(scope):
        by_group.setdefault(c.group, set()).add(c.config_id)
    checked = 0
    for g, ids in sorted(by_group.items()):
        spec = scope.groups[g]
        if spec.reference is not None:
            continue
        patterns = next(iter(spec.intended_differences.values()))
        flats = {cid: _executed_sections(result.configs[cid]) for cid in sorted(ids)}
        first = next(iter(flats))
        for cid, flat in flats.items():
            bad = [k for k in set(flat) | set(flats[first])
                   if flat.get(k) != flats[first].get(k) and not any(matches(p, k) for p in patterns)]
            assert bad == [], f"group {g}: {first} vs {cid} differ in {bad}, not declared ({patterns})"
            checked += 1
    assert checked >= 80 and {"qwen-budget-75", "g1", "qwen", "gemma"} <= set(by_group)


def _with_w3(scope_file) -> dict:
    """The scope with W3 enabled again (disabled since D121): the tracked scope's one fill overlap, W9 over W3."""
    raw = scope_file.model_dump(mode="json")
    for r in raw["rows"]:
        if r["id"] == "W3":
            r.update(enabled=True, why=None)
            for b in r["blocks"]:
                b.update(enabled=True, why=None)
    return raw


def test_fill_blocks_do_not_repeat_cells_and_plain_blocks_refuse_to():
    scope = parse(Scope, full_render().scope, "rendered")
    cells = {c.cell_id: c for c in expected_cells(scope)}
    assert cells["qwen.rb.q50l30c20__l12__s2"].block == "W9-l12"               # D121: W3 merged into W9
    presets, scope_file = filled()
    raw = _with_w3(scope_file)
    w3 = campaign.render(presets, parse(ScopeFile, raw, "scope"), pins=pins(presets),
                         decisions={**DECISIONS, **W3_DECISIONS}, fleets=FLEETS)
    cells = {c.cell_id: c for c in expected_cells(parse(Scope, w3.scope, "rendered"))}
    assert cells["qwen.rb.q50l30c20__l12__s2"].block == "W3-near-frontier"     # W9 lists it too, as fill
    assert cells["qwen.rb.q20l20c60__l12__s2"].block == "W9-l12"
    (w9,) = [r for r in raw["rows"] if r["id"] == "W9"]
    w9["blocks"][1]["fill"] = False
    with pytest.raises(campaign.ConfigError, match="listed by an earlier block"):
        campaign.render(presets, parse(ScopeFile, raw, "scope"), pins=pins(presets),
                        decisions={**DECISIONS, **W3_DECISIONS}, fleets=FLEETS)


# ---------------------------------------------------------------- staging: nothing missing is invented

def test_a_missing_pin_leaves_its_blocks_pending():
    presets, scope = filled()
    p = pins(presets)
    del p["xgb-heads.qwen"]
    result = campaign.render(presets, scope, pins=p, decisions=DECISIONS, fleets=FLEETS)
    assert "W4/W4-learned" in result.pending and "xgb-heads.qwen" in result.pending["W4/W4-learned"]
    assert "W4/W4-mechanisms" not in result.pending
    assert result.scope["pending"] == result.pending


def test_missing_decisions_and_fleets_leave_blocks_pending():
    presets, scope = filled()
    result = campaign.render(presets, scope, pins=pins(presets), decisions={}, fleets={"qwen_std": FLEETS["qwen_std"]})
    assert "rb_default" in result.pending["W2/W2-rb-l30"] and "headline" in result.pending["W4/W4-latency"]
    assert "fleet 'gemma' not given" in result.pending["W8/W8"]
    assert "W2/W2-base-l30" not in result.pending and "G1/G1" not in result.pending
    assert result.blocked_rows["W8-cal"] == "fleet 'gemma' not given"


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
    presets, scope = filled()                                     # one owner list unset: only its blocks wait
    scope = scope.model_copy(update={"owner": {**scope.owner, "g1_weights": [PLACEHOLDER, PLACEHOLDER]}})
    result = campaign.render(presets, scope, pins=pins(presets), decisions=DECISIONS, fleets=FLEETS)
    assert set(result.pending) == {"G1/G1", "G1b/G1b", "G1c/G1c", "G1c/G1c-l30"} | HELD
    assert "owner.g1_weights[0]" in result.pending["G1/G1"]


def test_the_first_render_before_w1_gives_the_direct_rows(tmp_path):
    presets, scope = filled()
    early = {k: v for k, v in pins(presets).items() if not k.startswith(("tpot-table", "prefill-table", "xgb"))}
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
    static = campaign.render(presets, scope, pins=pins(presets),
                             decisions={**DECISIONS, "rb_default": "rb-static", "decoupled_arm": "rb-static"},
                             fleets=FLEETS)
    learned = campaign.render(presets, scope, pins=pins(presets),
                              decisions={**DECISIONS, "rb_default": "rb-learned", "decoupled_arm": "rb"}, fleets=FLEETS)
    # with the learned heads as RouteBalance's default, the decoupled control keeps the curve (it scores idle
    # instances, so it cannot use the heads), and the Gemma rows (no heads) follow decoupled_arm too
    assert "qwen.rb-learned.uniform" in learned.configs and "qwen.rb-dec.uniform" in learned.configs
    assert learned.configs["qwen.rb-learned.uniform"]["latency"]["source"] == "xgboost"
    assert "gemma.rb.uniform" in learned.configs and "gemma.rb-learned.uniform" not in learned.configs
    assert "qwen.rb-dec.uniform" in rb.configs and "qwen.rb-static-dec.uniform" not in rb.configs
    assert "qwen.rb-static-dec.uniform" in static.configs and "qwen.rb-dec.uniform" not in static.configs
    assert static.configs["qwen.rb-static-dec.uniform"]["latency"] == static.configs["qwen.rb-static.uniform"]["latency"]
    assert "g1.qwen.rb-dec.uniform" in static.configs                  # G1 pairs RB-new with its own control
    pending = campaign.render(presets, scope, pins=pins(presets), decisions={}, fleets=FLEETS).pending
    assert "decoupled_arm" in pending["W2/W2-dec-l30"] and "rb_default" in pending["W2/W2-rb-l30"]
    assert "W2/W2-base-l30" not in pending


def test_a_fill_block_waits_for_the_earlier_blocks_that_could_list_its_cells(tmp_path):
    presets, scope = filled()
    p = pins(presets)
    tracked = campaign.render(presets, scope, pins=p, decisions={"rb_default": "rb"}, fleets=FLEETS)
    assert "waits for W3" not in tracked.pending["W9/W9-l12"]           # D121: W3 disabled, W9 waits for no pick
    scope = parse(ScopeFile, _with_w3(scope), "scope")
    first = campaign.render(presets, scope, pins=p, decisions={"rb_default": "rb"}, fleets=FLEETS)
    assert "waits for W3/W3-near-frontier" in first.pending["W9/W9-l12"]
    campaign.write_render(first, tmp_path / "r1", {})
    later = campaign.render(presets, scope, pins=p, decisions={**DECISIONS, **W3_DECISIONS}, fleets=FLEETS)
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
    moved = campaign.render(presets, scope, pins=p, decisions={**DECISIONS, "w5_systems": ["qwen.disp.lor"]},
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
        campaign.plan_row(plan, "G1", "calibration")
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
    presets, scope = filled()
    rows = [r.model_copy(update={"blocks": [b.model_copy(update={"hold": "waits for its support"})
                                            for b in r.blocks]}) if r.id == "W13" else r for r in scope.rows]
    held = campaign.render(presets, scope.model_copy(update={"rows": rows}), pins=pins(presets), decisions=DECISIONS,
                           fleets=FLEETS)
    assert held.pending["W13/W13"] == "held: waits for its support"
    assert "W13/W13" in parse(Scope, held.scope, "rendered").pending   # analysis.manifest never reports COMPLETE
    result = full_render()
    scope = parse(Scope, result.scope, "rendered")
    assert result.pending == {} and sum(c.block.startswith("W7b-") for c in expected_cells(scope)) == 9
    assert "W6/W6" not in result.pending and not any(c.block == "W6" for c in expected_cells(scope))   # dropped
    assert "W4/W4-arm3" not in result.pending and not any(c.block == "W4-arm3" for c in expected_cells(scope))
    assert sum(c.block in ("W4-const-batch", "W4r-const-batch") for c in expected_cells(scope)) == 9   # the D45 control


@pytest.mark.parametrize("arm", ["rb", "rb-static", "rb-learned"])
def test_the_ablations_take_apart_the_decided_variant(arm):
    """D78: W4's reference is rb_default's variant and its nine mechanism ablations are of that variant; the six
    latency variants all run, one of them the reference; 102 cells whichever variant is decided (90, and the 12 LPT-key
    cells since D121) at two rates, 153 at three since D125, seed 1 in W4 and seeds 2-3 in W4r."""
    presets, scope = filled()
    distinct = {n: {"path": p["path"], "md5": hashlib.md5(n.encode()).hexdigest()} for n, p in pins(presets).items()}
    result = campaign.render(presets, scope, pins=distinct, decisions={**DECISIONS, "rb_default": arm}, fleets=FLEETS)
    rendered = parse(Scope, result.scope, "rendered")
    w4 = [c for c in expected_cells(rendered) if c.block.startswith(("W4-", "W4r-"))]
    assert len(w4) == 153 and rendered.groups["qwen-ablation"].reference == f"abl.qwen.{arm}.q50l30c20"
    lat = {"rb": "latency.curve", "rb-static": "latency.static", "rb-learned": "latency.xgb"}[arm]
    want = campaign.Resolver(presets, distinct).resolve(f"$part:{lat}", "test", campaign.Needs(), {"family": "qwen"})
    mechanisms = [c for c in w4 if c.block == "W4-mechanisms"]
    assert len({c.config_id for c in mechanisms}) == 9
    assert all(result.configs[c.config_id]["latency"] == want for c in mechanisms)
    ref = group_record(rendered, next(c for c in w4 if c.config_id == f"abl.qwen.{arm}.q50l30c20"))
    assert ref["intended_differences"] == []
    others = {c.config_id for c in w4 if c.block in ("W4-latency", "W4-learned", "W4-const-batch")} - {ref["reference"]}
    assert len(others) == 5 and all(
        group_record(rendered, next(c for c in w4 if c.config_id == cid))["intended_differences"] == LATENCY_PATHS
        for cid in others)
    # both parity rules for an ablation group (gates/parity.py), on the rendered documents, for W4 and W11: a
    # variant differs from the reference only in its declared patterns, and every declared pattern does differ
    # (found 09-26: five declared leaves failed the second rule wherever a variant changed only some of them)
    from gates.parity import matches
    group = [c for c in expected_cells(rendered) if c.group == "qwen-ablation"]
    base = _executed_sections(result.configs[ref["reference"]])
    for cid in sorted({c.config_id for c in group} - {ref["reference"]}):
        declared = group_record(rendered, next(c for c in group if c.config_id == cid))["intended_differences"]
        flat = _executed_sections(result.configs[cid])
        differing = [k for k in set(base) | set(flat) if base.get(k) != flat.get(k)]
        assert declared and all(any(matches(p, k) for p in declared) for k in differing), (cid, differing, declared)
        assert all(any(matches(p, k) for k in differing) for p in declared), (cid, "declared but equal", declared)


def test_the_learned_variant_scores_latency_by_max_divide():
    """D78: v3's learned variant used max-divide; fixed-reference stays the measured curve's."""
    presets, _ = filled()
    resolver = campaign.Resolver(presets, pins(presets))
    expanded = campaign.expand(presets)
    def normalization(pid):
        return campaign.render_config(expanded[pid], resolver)[0].document["latency"]["normalization"]
    for pid in ("qwen.rb-learned.uniform", "qwen.rb-learned-budget.uniform", "qwen.rb-learned.abl-dr-off.uniform"):
        assert normalization(pid) == "max_divide"
    assert normalization("qwen.rb.uniform") == "fixed_ref"



def test_an_enabled_block_needs_seeds_and_previous_must_be_a_render(tmp_path):
    s = copy.deepcopy(SCOPE_JSON)
    for path, v in OWNER_SCOPE.items():
        _set(s, path, v)
    _set(s, "blocks.W1b.seeds", [])
    with pytest.raises(ConfigError, match="seeds"):
        parse(ScopeFile, s, "scope")
    presets, scope = filled()
    result = campaign.render(presets, scope, pins=pins(presets), decisions={}, fleets=FLEETS)
    with pytest.raises(ConfigError, match="not a render"):
        campaign.check_previous(result, tmp_path / "nowhere")
