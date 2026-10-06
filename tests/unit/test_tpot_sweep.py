"""The scheduler's TPOT sweep records: exact shape, v3 drop rules, file order (tools/tpot_sweep.py)."""
import json

import pytest

from tools.tpot_sweep import read_sweep

NAMES = ("num_running", "num_waiting", "pending_decode_tokens", "kv_cache_utilization", "num_prompt_tokens",
         "num_predicted_output_tokens", "req_decode_ctx_avg", "req_decode_ctx_max")


def _rec(tier="qwen2.5-3b_a30", tpot_s=0.011, success=True, output_tokens=120, **feature_overrides):
    features = {"num_running": 3.0, "num_waiting": 0.0, "pending_decode_tokens": 410.0, "kv_cache_utilization": 0.25,
                "num_prompt_tokens": 88.0, "num_predicted_output_tokens": 150.0, "req_decode_ctx_avg": 162.5,
                "req_decode_ctx_max": 237.0}
    features.update(feature_overrides)
    return {"tier": tier, "features": features, "tpot_s": tpot_s, "output_tokens": output_tokens, "success": success}


def _write(tmp_path, records, name="sweep.jsonl"):
    path = tmp_path / name
    path.write_text("".join(json.dumps(r) + "\n" for r in records) + "\n")
    return path


def test_rows_follow_the_feature_order_and_the_file_order(tmp_path):
    a = _write(tmp_path, [_rec(num_running=1.0), _rec(tier="qwen2.5-14b_v100", tpot_s=0.02)], "a.jsonl")
    b = _write(tmp_path, [_rec(num_running=2.0)], "b.jsonl")
    sweep = read_sweep([a, b], NAMES)
    rows = sweep.tiers["qwen2.5-3b_a30"]
    assert [r[0] for r in rows.features] == [1.0, 2.0] and rows.tpot_s == [0.011, 0.011]
    assert rows.features[0] == [1.0, 0.0, 410.0, 0.25, 88.0, 150.0, 162.5, 237.0]
    assert sweep.tiers["qwen2.5-14b_v100"].tpot_s == [0.02] and rows.output_tokens == [120, 120]


def test_failed_requests_and_missing_targets_are_dropped_and_counted(tmp_path):
    sweep = read_sweep([_write(tmp_path, [_rec(success=False), _rec(tpot_s=None), _rec(tpot_s=0.0),
                                          _rec(tpot_s=-1.0), _rec()])], NAMES)
    assert (sweep.failed, sweep.no_target, len(sweep.tiers["qwen2.5-3b_a30"].tpot_s)) == (1, 3, 1)


@pytest.mark.parametrize("mutate", [
    lambda r: r["features"].pop("num_waiting"),                     # a feature missing
    lambda r: r["features"].update(decode_ctx_p95=911.0),           # a fork-only field
    lambda r: r["features"].update(num_running=float("nan")),       # not finite
    lambda r: r["features"].update(num_running=True),               # not a number
    lambda r: r["features"].update(num_running="3"),
    lambda r: r.pop("success"),
    lambda r: r.update(success=1),                                  # a boolean, not a truthy value
    lambda r: r.update(extra=1),
    lambda r: r.update(tier=""),
    lambda r: r.update(output_tokens=12.0),
    lambda r: r.update(output_tokens=-1),
    lambda r: r.update(tpot_s="0.01"),
])
def test_malformed_records_fail(tmp_path, mutate):
    rec = _rec()
    mutate(rec)
    with pytest.raises(ValueError):
        read_sweep([_write(tmp_path, [_rec(), rec])], NAMES)


def test_a_malformed_record_fails_even_when_it_would_be_dropped(tmp_path):
    rec = _rec(success=False)
    rec["features"].pop("num_waiting")
    with pytest.raises(ValueError, match="features must be exactly"):
        read_sweep([_write(tmp_path, [rec])], NAMES)
