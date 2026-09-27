"""Contract with the vLLM fork's streaming chat endpoint, over real HTTP against the fake server."""
import asyncio

import aiohttp
import pytest

from routebalance.backend import BackendClient, BackendUnreachable, GenerationRequest
from tests.contract.fake_vllm import FakeState
from tests.contract.test_metrics import TIER, _free_port, _serve
from routebalance.types import Instance

REQ = GenerationRequest("r1", "What is 2+2?", max_tokens=64, temperature=0.0, repetition_penalty=1.0,
                        frequency_penalty=1.2)


def _generate(state: FakeState, req: GenerationRequest = REQ, timeout_s: float = 10.0):
    async def scenario():
        runner, inst = await _serve(state)
        try:
            async with aiohttp.ClientSession() as session:
                return await BackendClient(session, timeout_s).generate(inst, req)
        finally:
            await runner.cleanup()
    return asyncio.run(scenario())


def test_success_carries_text_tokens_and_timings():
    state = FakeState(chat_tokens=("Hello", " world", "!"), chat_delay_s=0.01)
    r = _generate(state)
    assert r.success and r.error == "" and r.text == "Hello world!" and r.output_tokens == 3
    assert 0 < r.ttft_s <= r.latency_s and len(r.itl_s) == 3          # role delta, then 3 token messages
    assert all(g >= 0 for g in r.itl_s)


def test_payload_sends_the_raw_prompt_once_and_the_fork_fields():
    state = FakeState()
    _generate(state)
    p = state.chat_requests[0]
    assert p["messages"] == [{"role": "user", "content": "What is 2+2?"}]          # single templating: raw prompt
    assert "predicted_decode_tokens" not in p and p["max_completion_tokens"] == 64      # official vLLM fields only
    assert p["stream"] is True and p["request_id"] == "r1"
    assert p["stream_options"] == {"include_usage": True, "continuous_usage_stats": True}
    assert p["model"] == "M"


@pytest.mark.parametrize("mode,match", [
    ("http500", "HTTP 500"),
    ("no_choices", "no streamed message with choices"),
    ("no_usage", "a streamed chunk without its usage"),
    ("no_final_usage", "no final usage chunk"),
    ("engine_error", "server error: 'engine died'"),
    ("malformed", "malformed stream"),
    ("null_event", "malformed stream"),
    ("null_choice", "malformed stream"),
    ("int_content", "malformed stream"),
    ("no_delta", "malformed stream"),
    ("unterminated", "malformed stream"),
])
def test_bad_streams_are_failures_not_results(mode, match):
    r = _generate(FakeState(chat_mode=mode))
    assert not r.success and match in r.error


def test_connection_cut_mid_stream_is_a_failure():
    r = _generate(FakeState(chat_mode="cut", chat_tokens=("a", "b", "c")))
    assert not r.success and "stream interrupted" in r.error


def test_unreachable_instance_raises():
    async def scenario():
        inst = Instance("gone", TIER, "n0", "127.0.0.1", _free_port())
        async with aiohttp.ClientSession() as session:
            await BackendClient(session, 2.0).generate(inst, REQ)
    with pytest.raises(BackendUnreachable):
        asyncio.run(scenario())


@pytest.mark.parametrize("kwargs", [{"max_tokens": 0}, {"max_tokens": -1}])
def test_requests_need_positive_lengths(kwargs):
    base = dict(request_id="r", prompt="p", max_tokens=8, temperature=0.0, repetition_penalty=1.0, frequency_penalty=1.2)
    base.update(kwargs)
    with pytest.raises(ValueError):
        GenerationRequest(**base)


def test_many_concurrent_requests_share_one_bounded_pool():
    async def scenario():
        state = FakeState(chat_delay_s=0.005)
        runner, inst = await _serve(state)
        try:
            connector = aiohttp.TCPConnector(limit=32)
            async with aiohttp.ClientSession(connector=connector) as session:
                client = BackendClient(session, 30.0)
                reqs = [GenerationRequest(f"r{i}", "p", 8, 0.0, 1.0, 1.2) for i in range(300)]
                results = await asyncio.gather(*(client.generate(inst, r) for r in reqs))
        finally:
            await runner.cleanup()
        return results
    results = asyncio.run(scenario())
    assert len(results) == 300 and all(r.success for r in results)


def test_crlf_framing_is_parsed():
    r = _generate(FakeState(chat_mode="crlf", chat_tokens=("a", "b")))
    assert r.success and r.text == "ab" and r.output_tokens == 2 and len(r.itl_s) == 2


def test_multiline_data_event_is_one_event():
    r = _generate(FakeState(chat_mode="multiline"))
    assert r.success and r.text == "x" and r.output_tokens == 1


def test_unicode_line_separators_inside_content_survive():
    tokens = ("a\u2028b", "c\u2029d", "e\x85f", "中")
    r = _generate(FakeState(chat_tokens=tokens))
    assert r.success and r.text == "".join(tokens)


@pytest.mark.parametrize("mode", ["slow_headers", "close_after_body"])
def test_failures_after_the_server_received_the_request_are_results_not_unreachable(mode):
    r = _generate(FakeState(chat_mode=mode), timeout_s=0.3)
    assert not r.success and r.error.startswith("no response")


def test_sse_parser_handles_crlf_split_across_reads():
    from routebalance.backend import _SSEParser
    p = _SSEParser()
    first = p.feed(b'data: {"choices": []}\r')
    rest = p.feed(b'\n\r\ndata: {"usage": {"completion_tokens": 3}}\r\n\r\n')
    assert first == [] and rest == [{"choices": []}, {"usage": {"completion_tokens": 3}}]
    assert p.finish() == []


def _progress(state):
    seen = []

    async def scenario():
        runner, inst = await _serve(state)
        try:
            async with aiohttp.ClientSession() as session:
                return await BackendClient(session, 10.0).generate(inst, REQ, on_progress=seen.append)
        finally:
            await runner.cleanup()
    return asyncio.run(scenario()), seen


def test_progress_reports_cumulative_generated_tokens():
    r, seen = _progress(FakeState(chat_tokens=("a", "b", "c")))
    assert r.success and seen == [1, 2, 3] and r.output_tokens == 3 and r.prompt_tokens == 5   # vLLM's own count


def test_progress_counts_tokens_even_when_vllm_merges_them_into_one_delta():
    r, seen = _progress(FakeState(chat_mode="merged", chat_tokens=("Hel", "lo", " wor", "ld", "!")))
    assert r.success and r.text == "Hello world!" and seen == [4, 5] and r.output_tokens == 5
