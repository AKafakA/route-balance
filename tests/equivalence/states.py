"""Instance states for the latency tests: a synthetic grid and recorded states from the latency sweep.

The v3 comparisons take the vLLM fork's /instance_stats fields (what v3 read); the scheduler's own
latency models take routebalance.estimate.latency.InstanceState (what official vLLM and the scheduler know).
"""
import json
import random
from types import SimpleNamespace

from tests.equivalence.oracle_files import LATENCY_TEST
from tools.fork_tpot_features import FORK_FIELDS

STATE_FIELDS = ("num_running", "num_waiting", "kv_cache_utilization", "max_num_seqs", "pending_decode_tokens")


def synthetic() -> list:
    """Fork /instance_stats bodies: all zeros, idle, busy, full and queued, p95 = 0, KV cache full."""
    base = {f: 0 for f in FORK_FIELDS}
    busy = dict(base, decode_ctx_p50=310.0, decode_ctx_p95=911.0, decode_ctx_max=1400.0, num_running=37,
                num_active_decode_seqs=35, num_waiting=0, pending_prefill_tokens=812, pending_decode_tokens=9021,
                token_budget_per_iter=8192, max_num_seqs=256, kv_cache_utilization=0.43, kv_free_blocks=5012)
    return [base, dict(base, max_num_seqs=256, token_budget_per_iter=8192, kv_free_blocks=6118), busy,
            dict(busy, num_running=256, num_waiting=19), dict(busy, decode_ctx_p95=0.0),
            dict(busy, kv_cache_utilization=1.0, kv_free_blocks=0)]


def recorded(oracle, n: int) -> list:
    """n recorded schedule states (all fields, including the ones the fork never served), seeded sample."""
    with open(oracle.data_file(LATENCY_TEST)) as f:
        rows = [json.loads(line)["schedule_state"] for line in f if line.strip()]
    return random.Random(0).sample(rows, n)


def fork_body(state: dict) -> dict:
    """What the fork's /instance_stats would have carried for this state."""
    return {f: state[f] for f in FORK_FIELDS}


def instance_state(state: dict) -> SimpleNamespace:
    """The same state as the scheduler sees it (latency.InstanceState)."""
    return SimpleNamespace(**{f: state[f] for f in STATE_FIELDS})
