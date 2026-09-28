"""D102: every model answers every prompt once, directly on the engines, with the serving payload."""
import asyncio
import dataclasses
import json

import pytest

from tests.contract.test_direct_engines import LARGE, SMALL, Engines, plan
from tools import collect_answers, direct

PROMPTS = [direct.Prompt(f"q{i}", f"question {i}?") for i in range(7)]


def test_every_model_answers_every_prompt_once_with_its_tiers_stop_tokens(tmp_path):
    stopping = dataclasses.replace(SMALL, stop_token_ids=(151645,))

    async def scenario():
        async with Engines([stopping, stopping, LARGE]) as e:
            p, _ = plan({})
            out = tmp_path / "answers.jsonl"
            written, failed = await collect_answers.collect(p, e.fleet, PROMPTS, 2, out)
            return e, written, failed, [json.loads(x) for x in out.read_text().splitlines()]
    e, written, failed, lines = asyncio.run(scenario())
    assert (written, failed) == (14, 0)
    assert sorted((x["model"], x["prompt_id"]) for x in lines) == sorted(
        (m, p.prompt_id) for m in ("S", "L") for p in PROMPTS)
    # the model's prompts are split across its instances in fleet order
    assert {x["prompt_id"] for x in lines if x["instance_id"] == "S_0"} == {"q0", "q2", "q4", "q6"}
    measured = [r for st in e.states for r in st.chat_requests if r["request_id"].startswith("collect.")]
    assert len(measured) == 14
    assert {tuple(r["stop_token_ids"]) for r in e.states[0].chat_requests} == {(151645,)}
    assert {tuple(r["stop_token_ids"]) for r in e.states[2].chat_requests} == {()}


def test_a_failing_engine_fails_the_run(tmp_path):
    async def scenario():
        async with Engines([SMALL, LARGE]) as e:
            e.states[1].chat_mode = "http500"
            p, _ = plan({})
            await collect_answers.collect(p, e.fleet, PROMPTS, 2, tmp_path / "answers.jsonl")
    with pytest.raises(direct.NotIdle):
        asyncio.run(scenario())
