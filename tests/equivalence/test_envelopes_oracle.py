"""T5: the queue-aware and own-length envelopes equal the v3 code they replace, bit for bit.

The v3 code is taken from the md5-pinned v3 server module itself: the queue-aware envelope is its lines
1389-1400 (the in-process predictor branch), the static prior its _static_tpot_map/_static_tpot_for. The
banked v3 static-arm decision log is replayed: every candidate's scored latency must be reproduced.
"""
import ast
import json
import textwrap
from types import SimpleNamespace

import pytest

from routebalance.estimate.latency import ConstantTPOT, own_length, queue_aware
from tests.equivalence.oracle_files import q1_static_decisions, read_jsonl, v3_source
from tests.equivalence.states import fork_body, recorded, synthetic

Q1_PRESET_MS = {"3b": 10.7, "7b": 19.9, "14b": 16.5, "72b": 42.6}      # the v3 Q1 static arm, in v3's short keys
Q1_BY_TIER = {"qwen2.5-3b_a30": 10.7, "qwen2.5-7b_a30": 19.9, "qwen2.5-14b_v100": 16.5, "qwen2.5-72b_a100": 42.6}
LENGTHS = (0.0, 0.4, 1.0, 92.9, 311.5, 1023.7)
TPOTS = (0.001, 0.0107, 0.0165, 0.0426, 0.0873)


@pytest.fixture(scope="module")
def v3_serve_source(oracle):
    return v3_source(oracle, "serve").read_text()


@pytest.fixture(scope="module")
def v3_envelope(v3_serve_source):
    lines = v3_serve_source.splitlines()[1388:1400]
    assert lines[0].strip() == "ss = stats or {}" and lines[-1].strip() == "sidecar_e2e = (queue_iters + own_out) * tpot"
    body = textwrap.indent(textwrap.dedent("\n".join(lines)), "    ")
    namespace = {}
    exec("def envelope(stats, expected_length, tpot):\n" + body + "\n    return sidecar_e2e\n", namespace)
    return namespace["envelope"]


@pytest.fixture(scope="module")
def v3_static(v3_serve_source):
    tree = ast.parse(v3_serve_source)
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("_static_tpot_map", "_static_tpot_for")]
    assert len(funcs) == 2
    namespace = {"slo_defaults": {"static_tpot_ms_per_tok": Q1_PRESET_MS}}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), "v3_static", "exec"), namespace)
    return namespace


def test_queue_aware_is_the_v3_envelope(oracle, v3_envelope):
    states = synthetic() + [fork_body(s) for s in recorded(oracle, 2000)]
    n = 0
    for s in states:
        for waiting in (0, 1, 7):
            for cap in (1, 8, 37, 256):
                body = dict(s, num_waiting=waiting, max_num_seqs=cap)
                for length in LENGTHS:
                    for tpot in TPOTS:
                        assert queue_aware(tpot, SimpleNamespace(**body), length) == v3_envelope(body, length, tpot)
                        n += 1
    print(f"T5: {n} queue-aware envelopes bitwise equal to v3")


def test_static_prior_matches_v3_by_tier(v3_static):
    smap = v3_static["_static_tpot_map"]()
    ours = ConstantTPOT(Q1_BY_TIER)
    for tier in Q1_BY_TIER:
        v3_tpot = v3_static["_static_tpot_for"](tier, smap)
        assert ours.tpot_s(tier) == v3_tpot == max(0.001, v3_tpot)    # >= 1 ms: v3's predict-boundary clamp is a no-op
        for length in LENGTHS:
            assert own_length(ours.tpot_s(tier), length) == v3_tpot * length


def test_banked_static_arm_decisions_replay(oracle):
    ours = ConstantTPOT(Q1_BY_TIER)
    records = read_jsonl(q1_static_decisions(oracle))
    n = 0
    for rec in records:
        assert rec["static_active"] and rec["static_cfg_ms"] == Q1_PRESET_MS
        for c in rec["cand"]:
            assert own_length(ours.tpot_s(c["itype"]), c["exp_len"]) == c["lat_s"], json.dumps(c)
            n += 1
    print(f"T5: {n} candidate latencies from {len(records)} banked static-arm decisions reproduced bitwise")
    assert n > 10000
