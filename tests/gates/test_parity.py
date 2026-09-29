"""The parity gate passes the recorded groups (a main group of five systems, an ablation group) and fails every
seeded execution difference, every undeclared or unrealized intended difference, and every broken group.json."""
import json
import shutil

import pytest

from gates.cell import open_cell
from gates.parity import flatten, main, matches, parity_problems
from tests.gates.cellfix import (FIXTURES, G1, RB_FIXEDREF, RB_MAXDIV, SWEEP, W4, W4_NODR, W4_REF, copy_cell,
                                 edit_json, edit_manifests, record)


def problems(dirs):
    return parity_problems([open_cell(d) for d in dirs])


def group(tmp_path, names):
    return [copy_cell(tmp_path, n) for n in names]


def spec(cell, fn):
    edit_json(cell / "group.json", fn)


@pytest.mark.parametrize("names,kind", [(G1, "main group g1"), (W4, "ablation group against w4-ref w4-dr")])
def test_the_recorded_groups_pass(names, kind):
    found, summary = problems([FIXTURES / "cells" / c for c in names])
    assert found == [] and kind in summary


def test_the_main_group_is_a_real_test():
    """Its systems do differ, in the system-defining keys only."""
    mans = [json.loads((FIXTURES / "cells" / c / "manifest_before.json").read_text()) for c in G1]
    assert len({json.dumps(m["system"], sort_keys=True) for m in mans}) == len(G1)
    assert len({json.dumps(m["loaded"], sort_keys=True) for m in mans}) >= 3


def test_flatten_and_patterns():
    assert flatten({"a": {"b": 1, "c": [2, {"d": 3}]}, "e": {}, "f": []}) == {"a.b": 1, "a.c[0]": 2, "a.c[1].d": 3,
                                                                          "e": {}, "f": []}
    assert matches("system.*", "system.rounds.adaptive") and matches("fleet.*", "fleet[0].tier")
    assert matches("system.lpt", "system.lpt") and not matches("system.lpt", "system.lpt_x")
    assert matches("system.router.*", "system.router")          # the root itself, when it is null, {} or []
    assert not matches("system.rounds.*", "system.roundsx") and not matches("system.r*", "system.rounds.x")


def setter(*path, value):
    def f(m):
        d = m
        for k in path[:-1]:
            d = d[k]
        d[path[-1]] = value
    return f


SHARED = [
    (("launch", "torch_threads"), 7, "launch.torch_threads"),
    (("launch", "torch_threads_effective"), 7, "launch.torch_threads_effective"),
    (("launch", "max_connections"), 65, "launch.max_connections"),
    (("launch", "request_timeout_s"), 60.0, "launch.request_timeout_s"),
    (("launch", "router_threads"), 8, "launch.router_threads"),
    (("launch", "engine_session", "force_close"), True, "launch.engine_session.force_close"),
    (("launch", "engine_session", "limit"), 32, "launch.engine_session.limit"),
    (("launch", "telemetry_session", "limit"), 1, "launch.telemetry_session.limit"),
    (("launch", "event_loop"), "uvloop.Loop", "launch.event_loop"),
    (("launch", "env", "OMP_NUM_THREADS"), "1", "launch.env.OMP_NUM_THREADS"),
    (("launch", "cpus", "affinity"), 1, "launch.cpus.affinity"),
    (("telemetry", "poll_ms"), 100.0, "telemetry.poll_ms"),
    (("telemetry", "max_age_ms"), 900.0, "telemetry.max_age_ms"),
    (("telemetry", "dead_reckoning"), False, "telemetry.dead_reckoning"),
    (("budget", "admission_filter"), True, "budget.admission_filter"),
    (("fleet_digest",), "f" * 64, "fleet_digest"),
    (("versions", "aiohttp"), "3.9.0", "versions.aiohttp"),
    (("versions", "python"), "3.11.0", "versions.python"),
    (("code", "routebalance"), "d" * 64, "code.routebalance"),
]


@pytest.mark.parametrize("path,value,key", SHARED, ids=[k for _, _, k in SHARED])
def test_an_undeclared_difference_fails_in_a_main_group(tmp_path, path, value, key):
    cells = group(tmp_path, G1)
    for c in cells:   # g1 declares dead reckoning (v3's probe arm, D55); a group that does not must fail on it
        spec(c, lambda s: s.__setitem__("intended_differences", ["system.*"]))
    edit_manifests(cells[0], setter(*path, value=value))
    found, _ = problems(cells)
    assert any(f"an undeclared execution difference: {key}:" in p for p in found), found


@pytest.mark.parametrize("path,value,key", SHARED[:6], ids=[k for _, _, k in SHARED[:6]])
def test_an_undeclared_difference_fails_in_an_ablation_group(tmp_path, path, value, key):
    cells = group(tmp_path, W4)
    edit_manifests(cells[1], setter(*path, value=value))
    found, _ = problems(cells)
    assert any(f"an undeclared execution difference: {key}:" in p for p in found), found


def test_the_report_shows_both_values(tmp_path):
    cells = group(tmp_path, W4)
    edit_manifests(cells[1], setter("launch", "torch_threads", value=7))
    found, _ = problems(cells)
    (line,) = [p for p in found if "launch.torch_threads:" in p]
    assert f"{W4_REF} (reference): 1 vs {W4_NODR}: 7" in line


def test_a_fleet_entry_difference_fails(tmp_path):
    cells = group(tmp_path, G1)
    edit_manifests(cells[1], lambda m: m["fleet"][2].__setitem__("max_num_seqs", 16))
    assert any("fleet[2].max_num_seqs" in p for p in problems(cells)[0])


def test_a_missing_version_fails(tmp_path):
    cells = group(tmp_path, G1)
    edit_manifests(cells[2], lambda m: m["versions"].pop("aiohttp"))
    assert any("versions.aiohttp" in p and "<absent>" in p for p in problems(cells)[0])


def test_one_artifact_path_with_two_contents_fails(tmp_path):
    cells = group(tmp_path, [RB_MAXDIV, RB_FIXEDREF])
    edit_manifests(cells[1], lambda m: m["artifacts"].__setitem__("calib/tpot.json", "0" * 32))
    assert any("artifacts.calib/tpot.json differs between the cells that loaded it" in p for p in problems(cells)[0])


def test_loaded_components_must_match_when_the_systems_do(tmp_path):
    cells = group(tmp_path, W4)
    edit_manifests(cells[1], lambda m: m["loaded"].__setitem__("router", "PassthroughRouter"))
    assert any("undeclared execution difference: loaded.router" in p for p in problems(cells)[0])


def test_an_extra_artifact_without_a_system_difference_fails(tmp_path):
    cells = group(tmp_path, W4)
    edit_manifests(cells[1], lambda m: m["artifacts"].__setitem__("extra/file.bin", "1" * 32))
    assert any("undeclared execution difference: artifacts.extra/file.bin" in p for p in problems(cells)[0])


def test_one_threadpool_with_two_sizes_fails(tmp_path):
    cells = group(tmp_path, W4)
    key = next(iter(json.loads((cells[0] / "manifest_before.json").read_text())["threadpools"]))
    edit_manifests(cells[1], lambda m: m["threadpools"].__setitem__(key, 97))
    assert any(f"threadpools.{key} differs between the cells that loaded it" in p for p in problems(cells)[0])


def test_a_declared_difference_that_did_not_happen_fails(tmp_path):
    cells = group(tmp_path, W4)
    spec(cells[1], lambda s: s["intended_differences"].append("system.lpt"))
    found, _ = problems(cells)
    assert any("a declared difference did not happen (the ablation did not run)" in p and "system.lpt" in p
               for p in found)


def test_a_declared_difference_passes_as_a_subtree(tmp_path):
    cells = group(tmp_path, W4)
    spec(cells[1], lambda s: s.__setitem__("intended_differences", ["telemetry.*"]))
    assert problems(cells)[0] == []


def test_an_ablation_group_needs_its_reference_cell(tmp_path):
    cells = group(tmp_path, W4)
    edit_json(record(cells[0]), lambda r: r["labels"].__setitem__("config_id", "renamed"))
    assert any("no reference cell (config w4-ref)" in p for p in problems(cells)[0])


def test_a_reference_cell_that_declares_differences_fails(tmp_path):
    cells = group(tmp_path, W4)
    spec(cells[0], lambda s: s.__setitem__("intended_differences", ["telemetry.dead_reckoning"]))
    assert any("a reference cell declares intended differences" in p for p in problems(cells)[0])


def test_reference_cells_must_be_identical(tmp_path):
    cells = group(tmp_path, W4)
    twin = tmp_path / W4_REF.replace("__s1", "__s2")
    shutil.copytree(cells[0], twin)
    (twin / f"{W4_REF}.cell.json").rename(twin / f"{twin.name}.cell.json")
    edit_json(twin / f"{twin.name}.cell.json", lambda r: r.__setitem__("cell_id", twin.name))
    edit_manifests(twin, setter("launch", "torch_threads", value=3))
    found, _ = problems([*cells, twin])
    assert any("the reference cells differ: launch.torch_threads" in p for p in found), found


GROUP_FAIL = [
    (lambda s: s.__setitem__("group", "other"), "group.json names another group than the cell record"),
    (lambda s: s.__setitem__("reference", "lor"), "the cells name different reference configurations"),
    (lambda s: s.__setitem__("intended_differences", ["system.*", "budget.*"]),
     "the cells of a main group declare different intended differences"),
    (lambda s: s.__setitem__("intended_differences", ["sytem.*"]), "is not a manifest path"),
    (lambda s: s.__setitem__("intended_differences", ["config_digest"]), "is not a manifest path"),
    (lambda s: s.__setitem__("intended_differences", ["system.r*"]), "is not a manifest path"),
    (lambda s: s.pop("reference"), "group.json"),
    (lambda s: s.__setitem__("extra", 1), "group.json"),
]


@pytest.mark.parametrize("mutate,expected", GROUP_FAIL, ids=[e[:40] for _, e in GROUP_FAIL])
def test_a_broken_group_spec_fails(tmp_path, capsys, mutate, expected):
    cells = group(tmp_path, G1)
    spec(cells[1], mutate)
    assert main(["--cells", *map(str, cells)]) == 1
    out = capsys.readouterr().out
    assert expected in out, out


def test_a_missing_group_spec_fails(tmp_path, capsys):
    cells = group(tmp_path, G1)
    (cells[0] / "group.json").unlink()
    assert main(["--cells", *map(str, cells)]) == 1
    assert "group.json is missing" in capsys.readouterr().out


def test_a_manifest_that_changed_during_a_cell_fails(tmp_path):
    cells = group(tmp_path, G1)
    edit_json(cells[0] / "manifest_after.json", lambda m: m["launch"].__setitem__("torch_threads", 7))
    assert any("changed during a cell" in p and "launch.torch_threads: 1 -> 7" in p for p in problems(cells)[0])


def test_code_changed_on_disk_fails_even_when_every_cell_agrees(tmp_path):
    cells = group(tmp_path, G1)
    for c in cells:
        edit_manifests(c, lambda m: m["code"].__setitem__("routebalance_now", "e" * 64))
    assert any("code changed on disk after it started" in p and f"-> {'e' * 64}" in p for p in problems(cells)[0])


def test_a_record_without_a_group_label_fails(tmp_path):
    cells = group(tmp_path, G1)
    edit_json(record(cells[0]), lambda r: r["labels"].pop("group"))
    assert any("group.json names another group than the cell record" in p for p in problems(cells)[0])


@pytest.mark.parametrize("field,value,key", [("sampling", {"temperature": 0.7}, "sampling.temperature"),
                                             ("max_connections", 32, "max_connections"),
                                             ("tokenizer_id", "other-tok", "tokenizer_id")])
def test_a_bench_difference_fails(tmp_path, field, value, key):
    cells = group(tmp_path, G1)

    def change(r):
        if isinstance(value, dict):
            r[field].update(value)
        else:
            r[field] = value
    edit_json(record(cells[3]), change)
    assert any(f"the bench ran differently: {key}" in p for p in problems(cells)[0])


def test_fewer_than_two_or_repeated_cells_fail(tmp_path):
    cell = copy_cell(tmp_path, W4_NODR)
    assert any("at least two cells" in p for p in problems([cell])[0])
    assert any("listed twice" in p for p in problems([cell, cell])[0])
    ref = copy_cell(tmp_path / "ref", W4_REF)                   # an ablation reference declares []: two needed
    assert any("at least two cells" in p for p in problems([ref])[0])
    main = copy_cell(tmp_path / "main", SWEEP)                  # a main group declaring differences: two needed
    assert any("at least two cells" in p for p in problems([main])[0])
    assert any("at least two cells" in p for p in problems([])[0])


def test_a_group_that_declares_no_difference_may_be_one_cell(tmp_path):
    """The W1b sweep: one configuration at one seed (found in the 09-25 rehearsal, where it was refused)."""
    cell = copy_cell(tmp_path, SWEEP)
    spec(cell, lambda s: s.__setitem__("intended_differences", []))
    assert problems([cell])[0] == []
    edit_manifests_after = lambda d: d["launch"].__setitem__("max_connections", 1)   # noqa: E731
    edit_json(cell / "manifest_after.json", edit_manifests_after)
    assert any("the execution manifest changed during a cell" in p for p in problems([cell])[0])   # still checked


def test_the_cli(tmp_path, capsys):
    cells = group(tmp_path, G1)
    assert main(["--cells", *map(str, cells)]) == 0 and capsys.readouterr().out.startswith("PASS parity: ")
    edit_manifests(cells[0], setter("launch", "torch_threads", value=3))
    assert main(["--cells", *map(str, cells)]) == 1
    out = capsys.readouterr().out.splitlines()
    assert out and all(line.startswith("FAIL parity: ") for line in out)


def _treatment(tmp_path, cells, name):
    """A second treatment of the ablation group: a copy of w4-nodr under another cell id."""
    twin = tmp_path / name
    shutil.copytree(cells[1], twin)
    (twin / f"{cells[1].name}.cell.json").rename(twin / f"{name}.cell.json")
    edit_json(twin / f"{name}.cell.json", lambda r: r.__setitem__("cell_id", name))
    return twin


def test_two_treatments_loading_one_path_with_two_contents_fail(tmp_path):
    """Treatments are compared with the reference, and also with each other on what they loaded."""
    cells = group(tmp_path, W4)
    twin = _treatment(tmp_path, cells, W4_NODR.replace("__s1", "__s2"))
    for c, md5 in ((cells[1], "1" * 32), (twin, "2" * 32)):
        edit_manifests(c, lambda m, md5=md5: m["artifacts"].__setitem__("xgb/small_a30.xgb", md5))
        spec(c, lambda s: s["intended_differences"].append("artifacts.*"))
    found, _ = problems([*cells, twin])
    assert any("artifacts.xgb/small_a30.xgb differs between the cells that loaded it" in p for p in found), found


def test_an_indexed_pattern_declares_one_fleet_entry(tmp_path):
    cells = group(tmp_path, W4)
    edit_manifests(cells[1], lambda m: (m["fleet"][2].__setitem__("max_num_seqs", 16),
                                        m.__setitem__("fleet_digest", "9" * 64)))
    spec(cells[1], lambda s: s.__setitem__("intended_differences",
                                           ["telemetry.dead_reckoning", "fleet[2].max_num_seqs", "fleet_digest"]))
    assert problems(cells)[0] == []


def test_threadpool_presence_follows_the_process_history(tmp_path):
    """One scheduler serves many cells and loads libraries lazily: a pool listed in one cell and not in another
    of the same system is not an execution difference (a pool both list must still agree)."""
    cells = group(tmp_path, W4)
    edit_manifests(cells[1], lambda m: m["threadpools"].__setitem__("openmp:xgboost.libs/libgomp.so.1", 2))
    assert problems(cells)[0] == []


def test_main_group_declarations_compare_as_sets(tmp_path):
    cells = group(tmp_path, G1)
    for c in cells:
        spec(c, lambda s: s.__setitem__("intended_differences", ["system.*", "telemetry.dead_reckoning",
                                                                 "budget.admission_filter"]))
    spec(cells[0], lambda s: s.__setitem__("intended_differences", ["budget.admission_filter", "system.*",
                                                                    "telemetry.dead_reckoning"]))
    assert problems(cells)[0] == []


def test_a_treatment_that_declares_nothing_fails(tmp_path):
    cells = group(tmp_path, W4)
    spec(cells[1], lambda s: s.__setitem__("intended_differences", []))
    found, _ = problems(cells)
    assert any("declares no difference" in p for p in found)


def test_a_record_without_a_config_id_fails(tmp_path):
    cells = group(tmp_path, W4)
    edit_json(record(cells[1]), lambda r: r["labels"].pop("config_id"))
    assert any("has no config_id label" in p for p in problems(cells)[0])


def _engines(cell, fn, whens=("before", "after")):
    for when in whens:
        edit_json(cell / f"engines_{when}.json", fn)


def test_an_engine_launched_differently_in_one_cell_fails_the_group(tmp_path):
    """D48: every cell of a group ran on the same engine launches, whatever the group declares."""
    cells = group(tmp_path, G1)
    _engines(cells[1], lambda d: d["instances"]["L_2"]["cache_config"].update(num_gpu_blocks="7000"))
    found, _ = problems(cells)
    assert any("the engines were launched differently: L_2.cache_config" in f for f in found)
    cells = group(tmp_path / "argv", G1)
    _engines(cells[2], lambda d: d["instances"]["S_0"].update(argv_sha256="0" * 64))
    assert any("L_2" not in f and "S_0.argv_sha256" in f for f in problems(cells)[0])
    cells = group(tmp_path / "gpus", G1)                                # the same argv on other GPUs of the node
    _engines(cells[3], lambda d: d["instances"]["S_1"].update(gpus="1"))
    assert any("the engines were launched differently: S_1.gpus" in f for f in problems(cells)[0])


def test_a_restart_during_a_cell_fails_and_one_between_cells_does_not(tmp_path):
    cells = group(tmp_path, G1)
    _engines(cells[0], lambda d: d["instances"]["S_1"].update(start_time_s=1790000060.0), whens=("after",))
    found, _ = problems(cells)
    assert any("an engine restarted or changed during a cell" in f and "S_1.start_time_s" in f for f in found)
    cells = group(tmp_path / "between", G1)
    _engines(cells[0], lambda d: d["instances"]["S_1"].update(start_time_s=1790000060.0))   # the same launch
    assert problems(cells)[0] == []


def test_a_cell_without_its_engine_record_fails(tmp_path, capsys):
    cells = group(tmp_path, G1)
    (cells[0] / "engines_after.json").unlink()
    assert main(["--cells", *map(str, cells)]) == 1
    assert "engines_after.json is missing" in capsys.readouterr().out


def test_an_engine_missing_from_every_cells_record_fails(tmp_path):
    """D48: each cell's record reads every engine of its fleet, even when all the cells miss the same one."""
    cells = group(tmp_path, G1)
    for c in cells:
        _engines(c, lambda d: d["instances"].pop("L_2"))
    found, _ = problems(cells)
    (line,) = [f for f in found if f.startswith("the engine record does not cover the cell's fleet")]
    assert line.endswith(f"and {len(cells) - 5} more") and "L_2" in line        # one line, every cell counted
