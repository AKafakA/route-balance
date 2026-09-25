"""The SLO bases: each tier's median TTFT at b = 1 from W1's requests, its TPOT from the table's curve at b = 1."""
import json

import pytest

from analysis.slo import SloBase
from tools import slo_base

LINE = {"tier": "s_a30", "instance": "S_0", "level": 1, "b": 1, "held": True, "filler": False, "success": True,
        "ttft_s": 0.02, "tpot_s": 0.01}
TABLE = {"unit": "ms_per_token", "constant": {"s_a30": 12.0, "l_a100": 30.0},
         "curve": {"s_a30": [[1, 10.0], [4, 11.0]], "l_a100": [[1, 25.0], [4, 28.0]]}, "provenance": {}}


def _files(tmp_path, lines, table=TABLE):
    (tmp_path / "requests.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    (tmp_path / "table.json").write_text(json.dumps(table))
    return tmp_path / "requests.jsonl", tmp_path / "table.json"


def _lines():
    ok = [{**LINE, "ttft_s": t} for t in (0.02, 0.03, 0.05)] + [{**LINE, "tier": "l_a100", "ttft_s": 0.1}]
    ignored = [{**LINE, "ttft_s": 9.0, **over} for over in ({"b": 4, "level": 4}, {"filler": True}, {"held": False},
                                                             {"success": False})]
    return ok + ignored


def test_bases_come_from_the_single_request_level(tmp_path):
    base = slo_base.slo_base(*_files(tmp_path, _lines()))
    assert base["ttft_s"] == {"l_a100": 0.1, "s_a30": 0.03} and base["tpot_s"] == {"l_a100": 0.025, "s_a30": 0.01}
    assert base["provenance"]["ttft_samples"] == {"l_a100": 1, "s_a30": 3}


def test_the_tool_writes_what_the_analysis_reads(tmp_path):
    requests, table = _files(tmp_path, _lines())
    assert slo_base.main(["--requests", str(requests), "--table", str(table), "--out", str(tmp_path / "b.json")]) == 0
    assert SloBase.load(tmp_path / "b.json").tpot_s["s_a30"] == 0.01
    assert slo_base.main(["--requests", str(requests), "--table", str(table), "--out", str(tmp_path / "b.json")]) == 1


@pytest.mark.parametrize("lines,table", [
    ([LINE], TABLE),                                                            # l_a100 has no b = 1 request
    (_lines(), {**TABLE, "curve": {"s_a30": [[4, 11.0]], "l_a100": [[1, 25.0]]}}),   # s_a30's curve has no b = 1
    (_lines(), {**TABLE, "unit": "s_per_token"}),
])
def test_a_tier_without_both_bases_fails(tmp_path, lines, table):
    requests, table_path = _files(tmp_path, lines, table)
    assert slo_base.main(["--requests", str(requests), "--table", str(table_path), "--out", str(tmp_path / "b.json")]) == 1
    assert not (tmp_path / "b.json").exists()
