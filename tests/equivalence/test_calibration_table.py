"""The pinned TPOT table is exactly what tools/calibrate_tpot_curve.py builds from the banked v3 cells.

Inputs: every lambda=6 and lambda=30 cell except BEST-Route's (whose relayed streams arrived in bursts, so its
client-side ITLs are not TPOTs; the tool refuses a table built with them), arrival seed 1.
"""
import json

import pytest

from tests.equivalence.oracle_files import TPOT_TABLE, TPOT_TABLE_MD5, md5_of
from tools.calibrate_tpot_curve import build_table, v3_cell_records, v3_method


def _cells(oracle):
    cells = [p for d in ("s2_p1_l6", "s2_p1_l30") for p in sorted(oracle.cells_dir(d).glob("*_n3534.json"))]
    return [p for p in cells if not p.name.startswith("br4_")]


def test_pinned_table_is_rebuilt_from_the_v3_cells(oracle, artifacts):
    table_path = artifacts / TPOT_TABLE
    assert md5_of(table_path) == TPOT_TABLE_MD5
    cells = _cells(oracle)
    assert len(cells) == 54
    records = [r for cell in cells for r in v3_cell_records(cell, 1)]
    rebuilt = build_table(records, cells, v3_method(1))
    assert json.loads(json.dumps(rebuilt)) == json.loads(table_path.read_text())
    assert all(n == 0 for n in rebuilt["provenance"]["b0_samples"].values())   # every request counts itself


def test_best_route_cells_make_a_table_the_scheduler_rejects(oracle):
    from routebalance.estimate.latency import LatencyError, ProfiledCurveTPOT

    cells = [p for d in ("s2_p1_l6", "s2_p1_l30") for p in sorted(oracle.cells_dir(d).glob("*_n3534.json"))]
    table = build_table([r for cell in cells for r in v3_cell_records(cell, 1)], cells, "all cells")
    with pytest.raises(LatencyError, match="not a TPOT"):
        ProfiledCurveTPOT(table["curve"])
