"""The expected-cells manifest: the scope decides what a cell is; a directory listing never does."""
import asyncio
import copy
import json

import pytest

from analysis import manifest
from bench.dataset import load_prompts
from bench.records import CELL_SUFFIX, write_cell
from bench.runner import CellSpec, run_cell
from bench.schema import parse
from bench.scope import Scope, expected_cells
from routebalance.config import ConfigError
from tests.unit.fakes import CONFIG_DIGEST, FakeScheduler, FakeTokenizer, identity, scope_dict, write_dataset

CONFIGS = {"rb-a": ("RouteBalance", CONFIG_DIGEST), "ap-x": ("AvengersPro", "a" * 64)}


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    """These tests are about coverage; the isolation evidence (D128/D132) has its own tests (tests/gates)."""
    monkeypatch.setattr(manifest, "isolation_problems", lambda d: [])


@pytest.fixture
def env(tmp_path):
    dataset = write_dataset(tmp_path)
    out = tmp_path / "cells"
    out.mkdir()
    return dataset, out


def _scope(dataset, **kw) -> Scope:
    return parse(Scope, scope_dict(dataset, CONFIGS, seeds=(1, 2), prompt_order="per_seed", **kw), "scope")


def _run(scope, dataset, out, cell_id, fleet=None, **fake):
    cell = {c.cell_id: c for c in expected_cells(scope)}[cell_id]
    prompts = load_prompts(dataset, cell.trace.n_requests, cell.prompt_order, cell.seed, FakeTokenizer(),
                           cell.dataset.tokenizer_id)
    spec = CellSpec(cell_id=cell.cell_id, labels=cell.labels, trace=cell.trace, seed=cell.seed,
                    sampling=scope.bench.sampling, expected_config_digest=cell.config.config_digest)
    ident = identity(cell.config.config_digest) if fleet is None else \
        identity(cell.config.config_digest).model_copy(update={"fleet_digest": fleet})
    rec = asyncio.run(run_cell(spec, prompts, FakeScheduler(identities=(ident, ident), **fake), "r1"))
    (out / cell_id).mkdir(exist_ok=True)
    return write_cell(rec, out / cell_id), rec


ALL = ["rb-a__l200__s1", "rb-a__l200__s2", "ap-x__l200__s1", "ap-x__l200__s2"]


def test_the_scope_enumerates_its_cells(env):
    cells = expected_cells(_scope(env[0]))
    assert [c.cell_id for c in cells] == ALL
    assert cells[0].labels == {"block": "main", "group": "g1", "config_id": "rb-a", "trace_id": "l200",
                               "system": "RouteBalance", "knob": "rb-a", "dispatch": "lor"}


def test_a_full_directory_is_complete(env):
    dataset, out = env
    scope = _scope(dataset)
    for c in ALL:
        _run(scope, dataset, out, c)
    cov = manifest.check(scope, out)
    assert cov.complete and cov.ok == tuple(ALL) and cov.expected == 4


def test_missing_invalid_and_unexpected_are_kept_apart(env):
    dataset, out = env
    scope = _scope(dataset)
    _run(scope, dataset, out, ALL[0])
    _run(scope, dataset, out, ALL[1], fail=lambda i: i == 0)
    (out / f"flat__l200__s1{CELL_SUFFIX}").write_text("{}")         # a record outside a cell directory
    (out / "stray__l200__s1").mkdir()
    (out / "stray__l200__s1" / f"stray__l200__s1{CELL_SUFFIX}").write_text("{}")
    (out / ALL[2]).mkdir()                                            # an attempt that never wrote its record
    for sidecar in ("ap-x__l200__s1.json", "ap-x__l200__s1.provenance.json", "x.bench.log", "decisions.jsonl"):
        (out / ALL[2] / sidecar).write_text("{}")         # never cells, whatever their names say
    (out / "_groups").mkdir()
    cov = manifest.check(scope, out)
    assert cov.ok == (ALL[0],) and cov.missing == tuple(ALL[2:]) and cov.aborted == (ALL[2],)
    assert list(cov.invalid) == [ALL[1]] and cov.invalid[ALL[1]][0] == "1/12 requests failed"
    assert cov.unexpected == (f"flat__l200__s1{CELL_SUFFIX}", "stray__l200__s1/") and not cov.complete
    assert "an aborted attempt" in manifest.report(cov)


def test_a_cell_directory_holds_exactly_its_own_record(env):
    dataset, out = env
    scope = _scope(dataset)
    path, rec = _run(scope, dataset, out, ALL[0])
    (path.parent / f"{ALL[1]}{CELL_SUFFIX}").write_text(path.read_text())
    assert manifest.check(scope, out).invalid[ALL[0]][0].startswith("the cell directory must hold exactly")


def test_a_scope_with_pending_blocks_is_never_complete(env):
    dataset, out = env
    scope = parse(Scope, {**scope_dict(dataset, CONFIGS, seeds=(1, 2), prompt_order="per_seed"),
                          "pending": {"W4-xgb": "needs pin xgb.qwen"}}, "scope")
    for c in ALL:
        _run(scope, dataset, out, c)
    cov = manifest.check(scope, out)
    assert cov.ok == tuple(ALL) and cov.pending == {"W4-xgb": "needs pin xgb.qwen"} and not cov.complete
    assert "PENDING    block W4-xgb: needs pin xgb.qwen" in manifest.report(cov)


def test_accounted_cells_do_not_block_completion(env):
    dataset, out = env
    reason = "BR sq/po2 overloaded the scheduler; kept as a finding"
    scope = _scope(dataset, accounted={ALL[3]: reason, ALL[0]: "stale"})
    for c in ALL[:3]:
        _run(scope, dataset, out, c)
    cov = manifest.check(scope, out)
    assert cov.complete and cov.accounted == {ALL[3]: reason} and cov.unused_accounting == (ALL[0],)


def test_a_cell_that_is_not_its_expectation_is_invalid(env):
    dataset, out = env
    scope = _scope(dataset)
    path, rec = _run(scope, dataset, out, ALL[0])
    path.unlink()
    write_cell(rec.model_copy(update={"labels": {**rec.labels, "knob": "other"}, "seed": 5}), out / ALL[0])
    problems = manifest.check(scope, out).invalid[ALL[0]]
    assert any(p.startswith("labels is") for p in problems) and any(p.startswith("seed is 5") for p in problems)
    assert "the arrival digest is not the schedule of the cell's trace and seed" in problems


def test_a_cell_whose_budgets_are_not_the_scopes_is_invalid(env):
    """W7b: a record's budget table and per-request budgets must be what its dataset entry says."""
    dataset, out = env
    scope = _scope(dataset)
    path, rec = _run(scope, dataset, out, ALL[0])
    path.unlink()
    budgeted = [r.model_copy(update={"budget_cost": 1e-5}) for r in rec.requests]
    write_cell(rec.model_copy(update={"budgets_sha256": "c" * 64, "requests": budgeted}), out / ALL[0])
    problems = manifest.check(scope, out).invalid[ALL[0]]
    assert any(p.startswith("budget table sha256 is 'ccc") for p in problems)
    assert f"{len(budgeted)}/{len(budgeted)} requests carried a budget, the scope expects none" in problems


def test_a_budget_cell_without_every_budget_is_invalid(env):
    """The other direction: the scope's dataset has a budget table, the record has none, or only some budgets."""
    dataset, out = env
    doc = scope_dict(dataset, {k: v for k, v in CONFIGS.items()}, seeds=(1, 2), prompt_order="per_seed")
    doc["datasets"]["evalset"]["budgets"] = {"path": "b.jsonl", "sha256": "d" * 64}
    scope = parse(Scope, doc, "scope")
    path, rec = _run(_scope(dataset), dataset, out, ALL[0])
    path.unlink()
    some = [r.model_copy(update={"budget_cost": 1e-5}) if r.index % 2 else r for r in rec.requests]
    write_cell(rec.model_copy(update={"requests": some}), out / ALL[0])
    problems = manifest.check(scope, out).invalid[ALL[0]]
    assert "budget table sha256 is None, the scope expects 'ddd" + "d" * 61 + "'" in problems
    assert f"{sum(r.budget_cost is not None for r in some)}/{len(some)} requests carried a budget, the scope expects " \
        "all" in problems


def test_a_cell_run_under_another_configuration_is_invalid(env):
    dataset, out = env
    scope = _scope(dataset)
    other = parse(Scope, {**scope_dict(dataset, {"rb-a": ("RouteBalance", "b" * 64), "ap-x": CONFIGS["ap-x"]},
                                       seeds=(1, 2), prompt_order="per_seed")}, "scope")
    _run(other, dataset, out, ALL[0])
    problems = manifest.check(scope, out).invalid[ALL[0]]
    assert any(p.startswith("scheduler configuration digest") for p in problems)


def test_an_unreadable_cell_is_invalid(env):
    dataset, out = env
    (out / ALL[0]).mkdir()
    (out / ALL[0] / f"{ALL[0]}{CELL_SUFFIX}").write_text('{"format": "rb-cell/1"}')
    assert manifest.check(_scope(dataset), out).invalid[ALL[0]][0].startswith("unreadable")


def test_one_group_runs_on_one_fleet(env):
    dataset, out = env
    scope = _scope(dataset)
    for c in ALL[:3]:
        _run(scope, dataset, out, c)
    _run(scope, dataset, out, ALL[3], fleet="9" * 64)
    cov = manifest.check(scope, out)
    assert list(cov.fleet_problems) == ["g1"] and "cells ran on 2 different fleets" in cov.fleet_problems["g1"]
    assert not cov.complete


def test_a_cell_accounted_in_advance_that_ran_ok_still_counts_in_the_fleet_check(env):
    dataset, out = env
    scope = _scope(dataset, accounted={ALL[3]: "may fail at this rate"})
    for c in ALL[:3]:
        _run(scope, dataset, out, c)
    _run(scope, dataset, out, ALL[3], fleet="9" * 64)
    cov = manifest.check(scope, out)
    assert ALL[3] in cov.ok and cov.unused_accounting == (ALL[3],)
    assert list(cov.fleet_problems) == ["g1"] and not cov.complete


def test_an_accounted_cell_from_another_fleet_does_not_block_its_group(env):
    dataset, out = env
    scope = _scope(dataset, accounted={ALL[3]: "ran before the node swap; kept as evidence"})
    for c in ALL[:3]:
        _run(scope, dataset, out, c)
    _run(scope, dataset, out, ALL[3], fleet="9" * 64, fail=lambda i: i == 0)
    cov = manifest.check(scope, out)
    assert cov.complete and cov.fleet_problems == {} and list(cov.accounted) == [ALL[3]]


def test_a_late_send_is_counted_never_a_gate(env):
    """D134 (owner): a late send no longer invalidates a cell; the report counts it (CellSummary.max_send_lag_ms)."""
    dataset, out = env
    scope = _scope(dataset, max_send_lag_ms=1e-6)            # a scope from before D134: its bound is read by nothing
    _run(scope, dataset, out, ALL[0])
    assert ALL[0] not in manifest.check(scope, out).invalid


def test_the_command_line_exit_code(env, tmp_path, capsys):
    dataset, out = env
    path = tmp_path / "scope.json"
    path.write_text(json.dumps(scope_dict(dataset, CONFIGS, seeds=(1, 2), prompt_order="per_seed")))
    assert manifest.main(["--scope", str(path), "--cells", str(out)]) == 1
    assert "MISSING    rb-a__l200__s1" in capsys.readouterr().out
    scope = _scope(dataset)
    for c in ALL:
        _run(scope, dataset, out, c)
    assert manifest.main(["--scope", str(path), "--cells", str(out)]) == 0
    assert capsys.readouterr().out.strip().endswith("COMPLETE")


@pytest.mark.parametrize("mutate,fragment", [
    (lambda d: d["blocks"][0]["configs"].append("nope"), "unknown config"),
    (lambda d: d["blocks"][0]["traces"].append("nope"), "unknown trace"),
    (lambda d: d["blocks"].append(copy.deepcopy(d["blocks"][0])), "listed twice"),
    (lambda d: d["blocks"][0]["seeds"].append(1), "listed twice"),
    (lambda d: d["accounted"].update({"x__y__s1": "why"}), "does not contain"),
    (lambda d: d["accounted"].update({"rb-a__l200__s1": " "}), "need a reason"),
    (lambda d: d["pending"].update({"W4-xgb": ""}), "pending entries need a reason"),
    (lambda d: d["groups"].clear(), "not declared under groups"),
    (lambda d: d["configs"].update({"bad__id": d["configs"]["rb-a"]}), "ids must match"),
    (lambda d: d["traces"]["l200"]["segments"][0].update(n_requests=13), "needs 13 prompts"),
    (lambda d: d["blocks"][0].update(seeds=[0]), "not valid for per_seed"),
    (lambda d: d["blocks"].append({**copy.deepcopy(d["blocks"][0]), "name": "repeats", "seeds": [3],
                                   "prompt_order": "fixed"}), "group g1: block repeats"),
])
def test_an_inconsistent_scope_fails(env, mutate, fragment):
    d = scope_dict(env[0], CONFIGS, seeds=(1, 2), prompt_order="per_seed")
    mutate(d)
    with pytest.raises(ConfigError, match=fragment):
        expected_cells(parse(Scope, d, "scope"))


@pytest.mark.parametrize("path", ["/abs/eval.jsonl", "../eval.jsonl", ""])
def test_dataset_paths_stay_inside_the_data_root(env, path):
    d = scope_dict(env[0], CONFIGS)
    d["datasets"]["evalset"]["path"] = path
    with pytest.raises(ConfigError):
        parse(Scope, d, "scope")
