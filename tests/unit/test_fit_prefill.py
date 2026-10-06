"""The prefill table's fit (tools/fit_prefill.py): a least-squares line per tier on the calibration's
single-request level, refused when it is not a usable prefill line."""
import json

import pytest

from routebalance.config import ConfigError
from routebalance.estimate.latency import Prefill
from tools import fit_prefill


def _requests(tmp_path, rows):
    p = tmp_path / "requests.jsonl"
    p.write_bytes(b"".join(json.dumps(r).encode() + b"\n" for r in rows))
    return p


def _row(tier, prompt_tokens, ttft_ms, level=1, **kw):
    return {"tier": tier, "level": level, "success": True, "filler": False, "held": True, "prompt_tokens": prompt_tokens,
            "ttft_s": ttft_ms / 1000.0} | kw


def test_the_fit_recovers_each_tiers_line_and_reads_only_the_level(tmp_path):
    rows = [_row("a", p, 20.0 + 0.05 * p) for p in range(10, 410, 10)]
    rows += [_row("b", p, 30.0 + 0.1 * p) for p in range(10, 410, 10)]
    rows += [_row("a", 100, 900.0, level=16), _row("a", 100, 900.0, filler=True), _row("a", 100, 900.0, success=False),
             _row("a", 100, 900.0, held=False)]                     # an instance-level that did not hold
    doc = fit_prefill.build(_requests(tmp_path, rows), 1)
    a, b = doc["prefill"]["a"], doc["prefill"]["b"]
    assert a["n"] == b["n"] == 40 and a["r2"] == pytest.approx(1.0) and b["r2"] == pytest.approx(1.0)
    assert (a["intercept_ms"], a["per_token_ms"]) == (pytest.approx(20.0), pytest.approx(0.05))
    assert (b["intercept_ms"], b["per_token_ms"]) == (pytest.approx(30.0), pytest.approx(0.1))
    assert Prefill(doc["prefill"]).seconds("a", 200) == pytest.approx(0.030)     # the scheduler reads the table


@pytest.mark.parametrize("rows,match", [
    ([_row("a", p, 20.0 + 0.05 * p) for p in range(10, 200, 10)], "at least 20"),            # 19 requests
    ([_row("a", p, 50.0 - 0.05 * p) for p in range(10, 410, 10)], "not a usable"),           # negative slope
    ([_row("a", p, 20.0 + 0.2 * p + (400.0 if (p // 10) % 2 else 0.0)) for p in range(10, 410, 10)],
     "not a usable"),                                                                            # r2 < 0.5, slope > 0
    ([_row("a", 100, 25.0) for _ in range(40)], "same prompt length"),
    ([_row("a", 100, 25.0, level=4)], "no successful measured request"),
    ([_row("a", p, 20.0 + 0.05 * p, held=False) for p in range(10, 410, 10)], "no successful measured request"),
    ([{"tier": "a", "level": 1, "success": True, "held": True, "prompt_tokens": 10}], "prompt_tokens and ttft_s"),
    ([{k: v for k, v in _row("a", 10, 20.0).items() if k != "held"}], "held flag"),
])
def test_the_fit_refuses_what_is_not_a_prefill_line(tmp_path, rows, match):
    with pytest.raises(ConfigError, match=match):
        fit_prefill.build(_requests(tmp_path, rows), 1)


def test_the_table_is_always_a_new_file(tmp_path):
    req = _requests(tmp_path, [_row("a", p, 20.0 + 0.05 * p) for p in range(10, 410, 10)])
    out = tmp_path / "prefill.json"
    assert fit_prefill.main(["--requests", str(req), "--out", str(out)]) == 0
    assert json.loads(out.read_text())["provenance"]["level"] == 1
    with pytest.raises(ConfigError, match="exists"):
        fit_prefill.main(["--requests", str(req), "--out", str(out)])
