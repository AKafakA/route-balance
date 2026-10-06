"""The TPOT table from W1's per-request records: the curve per batch size, the constant over every sample."""
import json

import numpy as np
import pytest

from tools import calibrate_tpot_curve as cal


def _write(tmp_path, rows, name="records.jsonl"):
    path = tmp_path / name
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def _w1(levels=(1, 2, 4, 8, 16, 32), n=40):
    """n records per level, TPOT growing with b, and one short request per level (3 decode steps)."""
    rows = []
    for b in levels:
        rows += [{"tier": "t_a30", "b": b, "tpot_s": 0.010 + 0.0005 * b + 1e-5 * i, "output_tokens": 50}
                 for i in range(n)]
        rows.append({"tier": "t_a30", "b": b, "tpot_s": 0.5, "output_tokens": 4})
    return rows


def test_tpot_records_give_the_curve_and_the_constant_over_all_samples(tmp_path):
    rows = _w1()
    records, method = cal.read_records(_write(tmp_path, rows))
    assert "(e2e - ttft) / (output_tokens - 1)" in method
    table = cal.build_table(records, [tmp_path / "records.jsonl"], method)
    every = [r["tpot_s"] for r in rows]                     # every sample, short requests included
    assert table["constant"]["t_a30"] == pytest.approx(1000.0 * float(np.median(every)))
    assert [b for b, _ in table["curve"]["t_a30"]] == [1, 2, 4, 8, 16, 32]
    b32 = [r["tpot_s"] for r in rows if r["b"] == 32]
    assert table["curve"]["t_a30"][-1][1] == pytest.approx(1000.0 * float(np.median(b32)))
    assert table["provenance"]["curve_samples"]["t_a30"] == [[b, 41] for b in (1, 2, 4, 8, 16, 32)]
    assert table["provenance"]["constant_samples"]["t_a30"] == 246


def test_short_requests_count_in_the_constant(tmp_path):
    # 31 short requests at 10 ms and 30 long ones at 20 ms per level: the median of all samples is 10 ms
    rows = [{"tier": "t", "b": b, "tpot_s": 0.010, "output_tokens": 4} for b in (1, 2) for _ in range(31)]
    rows += [{"tier": "t", "b": b, "tpot_s": 0.020, "output_tokens": 200} for b in (1, 2) for _ in range(30)]
    records, method = cal.read_records(_write(tmp_path, rows))
    assert cal.build_table(records, [tmp_path / "records.jsonl"], method)["constant"]["t"] == pytest.approx(10.0)


def test_itl_records_still_build_as_before(tmp_path):
    rows = [{"tier": "t_a30", "b": 1, "itl": [0.9, 0.01, 0.01, 0.01]} for _ in range(30)]
    records, method = cal.read_records(_write(tmp_path, rows))
    table = cal.build_table(records, [tmp_path / "records.jsonl"], method)
    assert table["constant"]["t_a30"] == pytest.approx(10.0) and "inter-token" in method


@pytest.mark.parametrize("rows,match", [
    ([{"tier": "t", "b": 1, "tpot_s": 0.01, "output_tokens": 9, "extra": 1}], "exactly the keys"),
    ([{"tier": "t", "b": 1, "tpot_s": 0.01, "output_tokens": 9}, {"tier": "t", "b": 1, "itl": [0.1] * 5}], "mixes"),
    ([{"tier": "t", "b": 1.0, "tpot_s": 0.01, "output_tokens": 9}], "b an integer"),
    ([{"tier": "t", "b": 1, "tpot_s": -0.01, "output_tokens": 9}], "positive"),
    ([{"tier": "t", "b": 1, "tpot_s": 0.01, "output_tokens": 1}], "at least 2"),
    ([{"tier": "t", "b": 1, "tpot_s": None, "output_tokens": 9}], "positive"),
    ([{"tier": "t", "b": 0, "tpot_s": 0.01, "output_tokens": 9}], "b an integer >= 1"),
    ([{"tier": "t", "b": 1, "itl": [0.1, -0.1, 0.1, 0.1]}], "itl must be"),
    ([{"tier": "t", "b": 1, "itl": {}}], "itl must be"),
])
def test_malformed_records_are_refused(tmp_path, rows, match):
    with pytest.raises(SystemExit, match=match):
        cal.read_records(_write(tmp_path, rows))


def test_the_command_line_writes_a_table_the_scheduler_accepts(tmp_path):
    src = _write(tmp_path, _w1())
    out = tmp_path / "table.json"
    assert cal.main(["--records", str(src), "--out", str(out)]) == 0
    table = json.loads(out.read_text())
    assert table["unit"] == "ms_per_token" and set(table["curve"]) == {"t_a30"}
    with pytest.raises(SystemExit, match="exists"):
        cal.main(["--records", str(src), "--out", str(out)])


def test_a_dip_in_the_measured_medians_is_held_at_the_running_max(tmp_path):
    """the curve never decreases in b; the measured medians stay in the provenance."""
    medians = {1: 0.012, 2: 0.011, 4: 0.013, 8: 0.0125, 16: 0.02}
    rows = [{"tier": "t_a30", "b": b, "tpot_s": s, "output_tokens": 50} for b, s in medians.items() for _ in range(31)]
    records, method = cal.read_records(_write(tmp_path, rows))
    table = cal.build_table(records, [tmp_path / "records.jsonl"], method)
    assert table["curve"]["t_a30"] == [[1, 12.0], [2, 12.0], [4, 13.0], [8, 13.0], [16, 20.0]]
    assert table["provenance"]["curve_measured"]["t_a30"] == [[1, 12.0], [2, 11.0], [4, 13.0], [8, 12.5], [16, 20.0]]
    assert table["provenance"]["curve_rule"] == cal.CURVE_RULE
    cal.ProfiledCurveTPOT(table["curve"])                                    # the scheduler accepts it
    assert cal.non_decreasing([]) == [] and cal.non_decreasing([[1, 5.0]]) == [[1, 5.0]]


def test_a_sub_ms_median_is_refused_before_the_running_max_could_hide_it(tmp_path):
    rows = [{"tier": "t_a30", "b": b, "tpot_s": s, "output_tokens": 50} for b, s in ((1, 0.012), (2, 0.0004))
            for _ in range(31)]
    records, method = cal.read_records(_write(tmp_path, rows))
    with pytest.raises(SystemExit, match=r"measured medians \[\[2, 0.4"):
        cal.build_table(records, [tmp_path / "records.jsonl"], method)
