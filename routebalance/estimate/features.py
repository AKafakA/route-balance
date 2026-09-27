"""The learned TPOT heads' input features, built without numpy or xgboost.

The server logs these for every request (the D38 sweep records the heads train on) and xgboost_tpot.py
serves them, both through feature_values(), so training and serving vectors are identical by construction;
the constant and curve latency sources never import xgboost.
"""
from __future__ import annotations

from routebalance.estimate.latency import InstanceState

FEATURES = ("num_running", "num_waiting", "pending_decode_tokens", "kv_cache_utilization", "num_prompt_tokens",
            "num_predicted_output_tokens", "req_decode_ctx_avg", "req_decode_ctx_max")


def feature_values(state: InstanceState, num_prompt_tokens: int, output_tokens: int) -> dict[str, float]:
    p, o = num_prompt_tokens, output_tokens
    return {"num_running": float(state.num_running), "num_waiting": float(state.num_waiting),
            "pending_decode_tokens": float(state.pending_decode_tokens),
            "kv_cache_utilization": float(state.kv_cache_utilization),
            "num_prompt_tokens": float(p), "num_predicted_output_tokens": float(o),
            "req_decode_ctx_avg": float(p + (o - 1) / 2),      # the v3 definitions: mean and last decode context
            "req_decode_ctx_max": float(p + o - 1)}
