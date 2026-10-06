"""The zero-call gate passes every recorded cell and fails each seeded violation."""
import pytest

from gates.cell import open_cell
from gates.zero_call import main, zero_call_problems
from tests.gates.cellfix import CELLS, FIXTURES, LOR, RB_FIXEDREF, RB_MAXDIV, copy_cell, edit_json, edit_manifests


def problems(cell):
    return zero_call_problems(open_cell(cell))


@pytest.mark.parametrize("name", CELLS)
def test_every_recorded_cell_passes(name):
    found, summary = problems(FIXTURES / "cells" / name)
    assert found == [] and ("0 booster calls" in summary or "no latency model loaded" in summary)


def calls(when, n):
    return lambda cell: edit_json(cell / f"stats_{when}.json", lambda s: s.__setitem__("booster_calls", n))


def loaded_source(source):
    return lambda cell: edit_manifests(cell, lambda m: m["loaded"].__setitem__("latency_source", source))


MUST_FAIL = [
    (RB_MAXDIV, calls("after", 3), "constant source: stats_after.json reports 3 booster calls"),
    (RB_FIXEDREF, calls("before", 1), "profiled_curve source: stats_before.json reports 1 booster calls"),
    (RB_FIXEDREF, calls("after", None), "reports None booster calls, not 0"),
    (LOR, calls("after", 0), "no latency model is loaded"),
    (RB_MAXDIV, loaded_source("xgboost"), "but the scheduler loaded 'xgboost'"),
    (RB_MAXDIV, lambda cell: edit_json(cell / "manifest_after.json",
                                       lambda m: m["launch"].__setitem__("torch_threads", 7)), "changed during the cell"),
]


@pytest.mark.parametrize("name,mutate,expected", MUST_FAIL, ids=[e[:40] for _, _, e in MUST_FAIL])
def test_each_seeded_violation_fails(tmp_path, name, mutate, expected):
    cell = copy_cell(tmp_path, name)
    mutate(cell)
    found, _ = problems(cell)
    assert any(expected in p for p in found), found


def test_the_xgboost_arm_is_not_applicable(tmp_path):
    cell = copy_cell(tmp_path, RB_MAXDIV)
    edit_manifests(cell, lambda m: (m["system"]["latency"].__setitem__("source", "xgboost"),
                                    m["loaded"].__setitem__("latency_source", "xgboost")))
    for when in ("before", "after"):
        edit_json(cell / f"stats_{when}.json", lambda s: s.__setitem__("booster_calls", 40))
    found, summary = problems(cell)
    assert found == [] and "not applicable" in summary


def test_the_cli(tmp_path, capsys):
    cell = copy_cell(tmp_path, RB_MAXDIV)
    assert main(["--cell", str(cell)]) == 0 and capsys.readouterr().out.startswith("PASS zero_call: ")
    calls("after", 2)(cell)
    assert main(["--cell", str(cell)]) == 1 and capsys.readouterr().out.startswith("FAIL zero_call: ")
    (cell / "stats_before.json").unlink()
    assert main(["--cell", str(cell)]) == 1 and "cannot check" in capsys.readouterr().out


def test_a_multiline_reason_prints_as_one_line(capsys):
    from gates.cell import GateError, run

    def check():
        raise GateError("stats_after.json: 1 validation error\nactive\n  Field required")
    assert run("demo", check) == 1
    assert capsys.readouterr().out.splitlines() == [
        "FAIL demo: cannot check: stats_after.json: 1 validation error | active | Field required"]
