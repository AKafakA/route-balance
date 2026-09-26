"""The report (plan Part B): the lease's cells to the paper's numbers, with every input checked."""
import asyncio
import hashlib
import json
import shutil

import pytest

from analysis import report as rp
from analysis.rules import load_rules
from bench.dataset import load_prompts
from bench.records import write_cell
from bench.runner import CellSpec, run_cell
from bench.schema import parse
from bench.scope import GroupSpec, Scope, cell_id, expected_cells
from routebalance.config import ConfigError, fleet_digest, load_fleet
from tests.unit.test_frontier import pt
from tests.unit.fakes import CONFIG_DIGEST, FakeScheduler, FakeTokenizer, identity, scope_dict, write_dataset

CONFIGS = {"rb-a": ("RouteBalance", CONFIG_DIGEST), "ap-x": ("AvengersPro", "a" * 64)}
KEY = "deepeval-llama3.1-8b-it_reference"


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def env(tmp_path):
    dataset = write_dataset(tmp_path)
    scope_raw = scope_dict(dataset, CONFIGS, seeds=(1, 2), prompt_order="per_seed")
    render = tmp_path / "render"
    render.mkdir()
    (render / "scope.json").write_text(json.dumps(scope_raw))
    fleet = tmp_path / "fleet.json"
    fleet.write_text(json.dumps({"tiers": {"t-small": {"model": "m-small", "gpu": "a30", "cost_in": 1e-7,
                                                       "cost_out": 2e-7, "max_num_seqs": 256}},
                                 "instances": [{"instance_id": "m-small_0", "tier": "t-small", "node": "n0",
                                                "port": 8000}]}))
    hosts = tmp_path / "hosts.json"
    hosts.write_text(json.dumps({"n0": {"address": "n0.invalid", "hw": "d7525"}}))
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("".join(json.dumps({"prompt": f"question {i}?", "models": {
        "m-small": {"llm_judge_scores": {KEY: 0.5}}}}) + "\n" for i in range(12)))
    (render / "plan.json").write_text(json.dumps({
        "format": "rb-plan/1", "rows": [], "groups": {"g1": {"fleet": "f1"}},
        "fleets": {"f1": {"family": "fam", "fleet_sha256": _sha(fleet)}},
        "scored_corpora": {"fam": {"path": corpus.name, "sha256": _sha(corpus)}}}))
    base = tmp_path / "slo_base.json"
    base.write_text(json.dumps({"unit": "s", "ttft_s": {"t-small": 0.01}, "tpot_s": {"t-small": 0.01},
                                "provenance": {}}))
    rules = json.loads(open("configs/prereg_rules.json").read())
    for level in ("tight", "loose"):
        rules["slo"][level]["m-small"] = 10.0 if level == "tight" else 15.0
    rules_path = tmp_path / "rules.json"
    rules_path.write_text(json.dumps(rules))
    digest = fleet_digest(load_fleet(fleet, hosts))
    scope = parse(Scope, scope_raw, "scope")
    cells = tmp_path / "cells"
    cells.mkdir()
    for cell in expected_cells(scope):
        prompts = load_prompts(dataset, cell.trace.n_requests, cell.prompt_order, cell.seed, FakeTokenizer(),
                               cell.dataset.tokenizer_id)
        spec = CellSpec(cell_id=cell.cell_id, labels=cell.labels, trace=cell.trace, seed=cell.seed,
                        sampling=scope.bench.sampling, expected_config_digest=cell.config.config_digest)
        ident = identity(cell.config.config_digest).model_copy(update={"fleet_digest": digest})
        slow = cell.config_id == "ap-x"
        rec = asyncio.run(run_cell(spec, prompts, FakeScheduler(identities=(ident, ident),
                                                                latency=lambda i, s=slow: 0.02 if s else 0.002), "r1"))
        (cells / cell.cell_id).mkdir()
        write_cell(rec, cells / cell.cell_id)
    return {"render": render, "cells": cells, "data_root": tmp_path, "fleet": fleet, "hosts": hosts, "base": base,
            "rules": rules_path, "tmp": tmp_path}


def _argv(env, out, **over):
    a = {"--render": env["render"], "--cells": env["cells"], "--data-root": env["data_root"],
         "--fleet": f"f1={env['fleet']}", "--hosts": env["hosts"], "--slo-base": f"fam={env['base']}",
         "--rules": env["rules"], "--treatment": "RouteBalance", "--out": out} | over
    return [x for k, v in a.items() for x in (k, *(map(str, v) if isinstance(v, list) else [str(v)]))]


def test_the_report_compares_at_equal_quality_and_writes_the_papers_files(env):
    out = env["tmp"] / "report"
    assert rp.main(_argv(env, out)) == 0
    report = json.loads((out / "report.json").read_text())
    assert report["coverage"]["ok"] == 4 and not report["left_out"] and len(report["configs"]) == 2
    assert all(c["model_mix"] == {"m-small": 1.0} for c in report["cells"].values())
    rows = report["equal_quality"]["RouteBalance"]
    assert [r["config_id"] for r in rows] == ["ap-x"] and rows[0]["seeds"] == [1, 2]
    mean = next(d for d in rows[0]["deltas"] if d["metric"] == "e2e_mean_ms")
    assert mean["relative"] < 0 and mean["ci"] is not None           # RouteBalance faster at the same quality
    assert (out / "tables" / "systems.g1.l200.tex").read_text().count(r"\\") == 3      # header + 2 configs
    assert "ap-x" in (out / "tables" / "equal.RouteBalance.g1.l200.tex").read_text()
    macros = (out / "macros.tex").read_text()
    assert r"\eqRouteBalanceGOneLTwoZeroZeroMeanMin" in macros and r"\eqRouteBalanceGOneLTwoZeroZeroCostMax" in macros
    assert report["goodput"]["tight@0.9"]["g1|rb-a"] == {"rate": 200.0, "censored": True, "rates": [200.0]}
    assert report["not_compared"] == {"RouteBalance": {}}
    assert "[" in (out / "tables" / "equal.RouteBalance.g1.l200.tex").read_text()      # the range over the seeds
    assert mean["range_relative"] is not None and [v["seed"] for v in mean["per_seed"]] == [1, 2]
    assert rows[0]["frontier_presets"] == ["rb-a", "rb-a"]                        # one preset: its own point
    assert "[" in (out / "tables" / "equal_ci.RouteBalance.g1.l200.tex").read_text()
    assert (out / "tables" / "equal_seeds.RouteBalance.g1.l200.tex").read_text().count(r"\\") == 3   # header + 2 seeds
    with pytest.raises(ConfigError, match="exists"):
        rp.main(_argv(env, out))


def test_inputs_other_than_the_pinned_ones_are_refused(env):
    other = env["tmp"] / "fleet2.json"
    other.write_text(env["fleet"].read_text().replace("2e-07", "3e-07"))
    with pytest.raises(ConfigError, match="not the fleet f1"):
        rp.main(_argv(env, env["tmp"] / "r1", **{"--fleet": f"f1={other}"}))
    (env["data_root"] / "corpus.jsonl").write_text("")
    with pytest.raises(ConfigError, match="not the scored corpus"):
        rp.main(_argv(env, env["tmp"] / "r2"))


def test_a_cell_without_its_inputs_fails_loudly(env):
    with pytest.raises(ConfigError, match="needs --slo-base fam"):
        rp.main(_argv(env, env["tmp"] / "r3", **{"--slo-base": f"other={env['base']}"}))


def test_macro_names_are_letters_only():
    assert rp.macro_name("RouteBalance-static", "qwen", "l12", "Mean") == "eqRouteBalancestaticQwenLOneTwoMean"


def test_only_one_segment_poisson_traces_have_a_rate(env):
    scope = parse(Scope, json.loads((env["render"] / "scope.json").read_text()), "scope")
    assert rp.stationary_rates(scope) == {"l200": 200.0}
    assert load_rules(env["rules"]).judge_key == KEY


def test_a_missing_treatment_seed_or_system_is_listed_and_the_report_still_writes(env):
    shutil.rmtree(env["cells"] / cell_id("rb-a", "l200", 2))
    out = env["tmp"] / "r4"
    assert rp.main(_argv(env, out, **{"--treatment": ["RouteBalance", "BEST-Route"]})) == 0
    report = json.loads((out / "report.json").read_text())
    assert report["equal_quality"] == {"RouteBalance": [], "BEST-Route": []}
    assert report["not_compared"] == {
        "RouteBalance": {"g1|l200|ap-x": "no RouteBalance preset has cells for every seed [1, 2]"},
        "BEST-Route": {"g1|l200|ap-x": "no BEST-Route configuration at g1/l200",
                       "g1|l200|rb-a": "no BEST-Route configuration at g1/l200"}}


def test_an_ablation_group_is_not_compared_at_equal_quality():
    groups = {"g1": GroupSpec(reference=None, intended_differences={}),
              "abl": GroupSpec(reference="rb-a", intended_differences={})}
    pts = [pt("rb-a", "RB", 1, 0.4, 1000.0), pt("ap-1", "AP", 1, 0.4, 2000.0),
           pt("rb-a", "RB", 1, 0.4, 1000.0, group="abl"), pt("rb-v", "RB-v", 1, 0.4, 900.0, group="abl")]
    these, skipped = rp.comparable(pts, "RB", groups)
    assert {(p.group, p.config_id) for p in these} == {("g1", "rb-a"), ("g1", "ap-1")}
    assert skipped == {"abl|l30|rb-v": "ablation group: read against its reference rb-a"}



def test_request_values_that_do_not_reproduce_the_summary_are_refused():
    import numpy as np
    from analysis.metrics import CellSummary, RequestValues, Spread
    one = np.ones(2)
    v = RequestValues(index=(0, 1), prompt_ids=("a", "b"), score=np.array([0.5, 0.5]), e2e_s=np.array([1.0, 1.0]),
                      cost=np.array([1e-6, 1e-6]), met_tight=one, met_loose=one)
    s = CellSummary(cell_id="c", n_requests=2, n_ok=2, n_failed=0, quality=0.5, matched=2, corpus_gap=0, unmatched=0,
                    unscored=0, budget_exhausted=None, n_zero=0, coverage=1.0, quality_zeros_excluded=0.5,
                    cost_decode_per_1k=1e-3, throughput_rps=1.0, e2e_ms=Spread(1000.0, 1000.0, 1000.0),
                    slo_attainment_tight=1.0, slo_attainment_loose=1.0, overhead_ms={}, off_instance_ms=None,
                    prompt_token_mismatches=0, problems=())
    rp.check_values("c", s, v)
    with pytest.raises(ConfigError, match="do not reproduce the summary"):
        rp.check_values("c", s.__class__(**{**s.__dict__, "quality": 0.6}), v)


def test_attainment_changes_are_shown_in_points_with_the_interval_flipped():
    # the treatment misses 5 points fewer requests: attainment +5.0, its range [+2.0, +8.0] from miss deltas [-8, -2]
    assert rp._change("slo_miss_tight", None, -0.05, None, (-0.08, -0.02)) == "+5.0 [+2.0, +8.0]"
    assert rp._change("e2e_mean_ms", -0.2, -100.0, (-0.25, -0.15)) == r"-20.0\% [-25.0, -15.0]"
    assert rp._change("e2e_mean_ms", None, None) == "--"
