"""Direct engine runs: the closed loop holds its level to the end with fillers, keeps measured results in order,
and stops at once when an engine is unreachable."""
import asyncio

import pytest

from bench.records import Sampling
from routebalance.backend import BackendUnreachable, GenerationResult
from tests.fleet_s2 import FLEET
from tools import direct
from tools.calibrate_driver import check_levels, records, slices, tpot_s

SAMPLING = Sampling(temperature=0.0, repetition_penalty=1.0, frequency_penalty=1.2, max_tokens=64)
INST = FLEET[0]


class FakeBackend:
    """generate() takes `delay(request)` seconds and records how many requests were in flight when each
    measured request finished."""

    def __init__(self, delay=lambda req: 0.01, unreachable=lambda req: False):
        self.in_flight = 0
        self.peak = 0
        self.at_finish: dict[str, int] = {}
        self.sent: list[str] = []
        self._delay, self._unreachable = delay, unreachable

    async def generate(self, inst, req, on_progress=None):
        self.sent.append(req.request_id)
        if self._unreachable(req):
            raise BackendUnreachable(f"{inst.instance_id}: refused")
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        try:
            await asyncio.sleep(self._delay(req))
            self.at_finish[req.request_id] = self.in_flight
        finally:
            self.in_flight -= 1
        n = 5
        return GenerationResult(req.request_id, inst.instance_id, inst.model, True, "", False, f"text {req.prompt}",
                                7, n, 0.05, (), 0.05 + 0.01 * (n - 1))


def prompts(n, tag="p"):
    return [direct.Prompt(f"{tag}{i}", f"{tag} prompt {i}") for i in range(n)]


def test_every_measured_request_finishes_with_the_level_in_flight():
    backend = FakeBackend(delay=lambda req: 0.05 if ".m" in req.request_id else 0.01)
    measured = prompts(10)
    run = asyncio.run(direct.closed_loop(backend, INST, measured, 4, SAMPLING, request_prefix="t",
                                         fillers=prompts(40, "f")))
    assert [r.text for r in run.measured] == [f"text p prompt {i}" for i in range(10)]
    assert backend.peak == 4 and not run.fillers_ran_out
    assert all(backend.at_finish[f"t.m{i}"] == 4 for i in range(10))      # the tail is held at the level
    assert run.fillers and all(".f" in f.request_id for f in run.fillers)


def test_a_filler_is_sent_at_most_once_and_running_out_is_reported():
    backend = FakeBackend(delay=lambda req: 0.2 if ".m" in req.request_id else 0.005)
    run = asyncio.run(direct.closed_loop(backend, INST, prompts(2), 4, SAMPLING, request_prefix="t",
                                         fillers=prompts(3, "f")))
    sent = [r for r in backend.sent if ".f" in r]
    assert sorted(sent) == ["t.f0", "t.f1", "t.f2"] and run.fillers_ran_out


def test_without_fillers_the_tail_drains_below_the_level():
    backend = FakeBackend(delay=lambda req: 0.02)
    run = asyncio.run(direct.closed_loop(backend, INST, prompts(5), 4, SAMPLING, request_prefix="t"))
    assert len(run.measured) == 5 and run.fillers == [] and min(backend.at_finish.values()) < 4
    assert not run.fillers_ran_out                                    # no fillers were asked for


def test_an_unreachable_engine_ends_the_run_at_once():
    backend = FakeBackend(delay=lambda req: 0.01, unreachable=lambda req: req.request_id == "t.m3")
    with pytest.raises(BackendUnreachable):
        asyncio.run(direct.closed_loop(backend, INST, prompts(8), 2, SAMPLING, request_prefix="t",
                                       fillers=prompts(2, "f")))


def test_request_ids_are_unique_and_fillers_are_cancelled_at_the_end():
    backend = FakeBackend(delay=lambda req: 0.2 if ".f" in req.request_id else 0.01)
    asyncio.run(direct.closed_loop(backend, INST, prompts(3), 3, SAMPLING, request_prefix="t", fillers=prompts(4, "f")))
    assert len(set(backend.sent)) == len(backend.sent) and backend.in_flight == 0


def test_the_per_token_decode_time_is_exact_for_merged_chunks():
    assert tpot_s(1.05, 0.05, 101) == pytest.approx(0.01)
    assert tpot_s(0.5, 0.5, 1) is None


def test_every_prompt_goes_to_an_engine_at_most_once():
    measured, fillers = slices(FLEET, 10, 6, 1000)          # the largest tier (7B) has 5 instances: K = 5
    firsts = {i.instance_id: measured[0][i.instance_id] for i in FLEET if i.instance_id.endswith("_0")}
    assert set(firsts.values()) == {(0, 10)}                 # instance k of every tier: the same prompts
    assert measured[0]["Qwen-2.5-7B_4"] == (40, 50) and measured[1]["Qwen-2.5-7B_0"] == (50, 60)
    for inst in FLEET:                                       # per engine: levels and fillers never overlap
        spans = [measured[j][inst.instance_id] for j in range(6)] + fillers
        taken = [i for a, b in spans for i in range(a, b)]
        assert len(taken) == len(set(taken))
    assert fillers[0] == (300, 416) and fillers[-1][1] <= 1000
    with pytest.raises(direct.ConfigError, match="none for fillers"):
        slices(FLEET, 10, 6, 300)


def test_levels_are_six_distinct_and_within_the_engine_batch():
    assert check_levels([1, 2, 4, 8, 16, 32], FLEET) == [1, 2, 4, 8, 16, 32]
    for bad in ([1, 2, 4, 8, 16], [1, 1, 2, 4, 8, 16], [0, 1, 2, 3, 4, 5], [1, 2, 4, 8, 16, 512]):
        with pytest.raises(direct.ConfigError):
            check_levels(bad, FLEET)


def test_calibration_records_are_measured_successful_requests_of_held_levels_only():
    base = {"tier": "t", "b": 4, "tpot_s": 0.01, "output_tokens": 9, "filler": False, "success": True, "held": True}
    lines = [base, {**base, "filler": True}, {**base, "success": False, "tpot_s": None},
             {**base, "tpot_s": None, "output_tokens": 1}, {**base, "held": False, "tpot_s": 0.03}]
    assert records(lines) == [{"tier": "t", "b": 4, "tpot_s": 0.01, "output_tokens": 9}]


def test_a_divergence_pass_whose_fillers_fail_does_not_hold_its_load(tmp_path, monkeypatch):
    from tools import reference_generation as rg

    async def no_wait(*a, **kw):
        return 0.0

    async def fake_loop(backend, inst, measured, concurrency, sampling, *, request_prefix, fillers=(), on_measured=None):
        ok = [GenerationResult(p.prompt_id, inst.instance_id, inst.model, True, "", False, "same   text", 5, 3,
                               0.1, (), 0.3) for p in measured]
        bad = [GenerationResult("f", inst.instance_id, inst.model, False, "HTTP 400", False, "", 0, 0, 0.0, (), 0.01)]
        for i, res in enumerate(ok):
            on_measured(i, res)
        return direct.LoopResult(ok, bad if fillers else [], False)
    monkeypatch.setattr(direct, "drain", no_wait)
    monkeypatch.setattr(direct, "warm_up", no_wait)
    monkeypatch.setattr(direct, "closed_loop", fake_loop)
    plan = {"bench": {"sampling": SAMPLING.model_dump(), "request_timeout_s": 10.0},
            "warmup": {"requests_per_instance": 1, "max_tokens": 4, "prompt": "x"}, "drain": {"timeout_s": 1.0}}
    row = {"id": "W0", "params": {"concurrency_per_instance": 1, "divergence_prompts": 2,
                                  "divergence_concurrency_per_instance": 4}}
    manifest = {}
    texts = asyncio.run(rg.generate(plan, row, [INST], prompts(6), tmp_path, manifest))
    assert len(texts) == 8 and list(manifest["load_not_held"]) == [f"divergence.{INST.instance_id}"]
    assert "1 fillers failed" in manifest["load_not_held"][f"divergence.{INST.instance_id}"]
    assert [rec for _, rec in rg.read_jsonl(tmp_path / "texts.jsonl")] == texts     # U+2028 inside a text
