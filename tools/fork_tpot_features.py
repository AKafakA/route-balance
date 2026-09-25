"""The feature vector the v3 TPOT heads were served with, bit for bit. For comparisons with v3 only.

v3 fed each head 25 float32 features (V3_FEATURES, in this order) built from the vLLM fork's
/instance_stats body. Five were never in that body (UNSERVED) and read as 0.0, and because the two
token-rate EMAs were missing, the three features divided by them took v3's 1.0 tok/s divide-by-zero floor.
The scheduler cannot serve these heads (official vLLM reports none of the fork's context or backlog
fields); tools/retrain_xgb_served.py uses this vector to score the v3 heads as they actually ran, and
tests/equivalence checks it against v3's own build_feature_vector.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np

V3_STATE_FEATURES = ("ema_decode_tok_per_s", "ema_prefill_tok_per_s", "ema_decode_iter_ms", "decode_ctx_p50",
                     "decode_ctx_p95", "decode_ctx_max", "num_running", "num_active_decode_seqs", "num_waiting",
                     "pending_prefill_tokens", "pending_decode_tokens", "token_budget_per_iter", "prefill_chunk_size",
                     "max_num_seqs", "kv_cache_utilization", "kv_free_blocks", "kv_evictions_per_s")
V3_FEATURES = V3_STATE_FEATURES + ("num_prompt_tokens", "num_predicted_output_tokens", "req_decode_ctx_avg",
                                   "req_decode_ctx_max", "backlog_seconds", "request_prefill_time_est",
                                   "request_decode_time_est", "ctx_relative_to_p95")
UNSERVED = frozenset({"ema_decode_tok_per_s", "ema_prefill_tok_per_s", "ema_decode_iter_ms", "prefill_chunk_size",
                      "kv_evictions_per_s"})
FORK_FIELDS = tuple(f for f in V3_STATE_FEATURES if f not in UNSERVED)   # what the fork's body carried
_RATE_FLOOR = 1.0


def v3_serving_vector(fork_stats: Mapping[str, float], num_prompt_tokens: int, output_tokens: int) -> np.ndarray:
    p, o = num_prompt_tokens, output_tokens
    v = {f: float(fork_stats[f]) for f in FORK_FIELDS}
    v.update({f: 0.0 for f in UNSERVED})
    p95 = v["decode_ctx_p95"]
    p95_divisor = p95 if p95 != 0.0 else _RATE_FLOOR
    ctx_max = p + o - 1
    v["num_prompt_tokens"], v["num_predicted_output_tokens"] = float(p), float(o)
    v["req_decode_ctx_avg"] = float(p + (o - 1) / 2)
    v["req_decode_ctx_max"] = float(ctx_max)
    v["backlog_seconds"] = float(fork_stats["pending_prefill_tokens"] / _RATE_FLOOR
                                 + fork_stats["pending_decode_tokens"] / _RATE_FLOOR)
    v["request_prefill_time_est"] = float(p / _RATE_FLOOR)
    v["request_decode_time_est"] = float(o / _RATE_FLOOR)
    v["ctx_relative_to_p95"] = float(ctx_max / p95_divisor) if p95_divisor > 0 else 1.0
    return np.array([v[n] for n in V3_FEATURES], dtype=np.float32)
