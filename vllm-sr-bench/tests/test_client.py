import asyncio
import json

from aiohttp import web

from bench.client import parse_reply, request_body
from bench.dataset import Prompt
from bench.records import Sampling, SchedulerIdentity
from vsrbench import constants as C
from vsrbench.client import VsrClient, chat_body

SAMPLING = Sampling(temperature=0.0, repetition_penalty=1.0, frequency_penalty=1.2, max_tokens=16)


def chunk(obj):
    return f"data: {json.dumps(obj)}\n\n".encode()


async def serve(handler):
    app = web.Application()
    app.router.add_post("/v1/chat/completions", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    return runner, site._server.sockets[0].getsockname()[1]


def body(rid="c.r.0"):
    return request_body(rid, Prompt(row=0, prompt_id="p0", text="hello", prompt_tokens=9, budget_cost=None), SAMPLING)


def test_chat_body_sends_only_the_accepted_fields_and_no_stop():
    got = chat_body(json.loads(body()))
    assert tuple(got) == C.REQUEST_FIELDS
    assert got["messages"] == [{"role": "user", "content": "hello"}] and got["max_completion_tokens"] == 16
    assert got["stream_options"] == {"include_usage": True}
    assert "stop" not in got and "stop_token_ids" not in got


def run(handler, rid="c.r.0"):
    seen = {}

    async def wrapped(request):
        seen["body"] = await request.json()
        seen["rid"] = request.headers.get("x-request-id")
        return await handler(request)

    async def go():
        runner, port = await serve(wrapped)
        ident = SchedulerIdentity(config_digest="a" * 64, fleet_digest="f")

        async def identity():
            return ident
        try:
            async with VsrClient(f"http://127.0.0.1:{port}", 16, 30.0, identity) as c:
                status, raw = await c.complete(body(rid))
                return status, raw, c.headers
        finally:
            await runner.cleanup()
    status, raw, headers = asyncio.run(go())
    return status, raw, headers, seen


async def streaming(request, model="m-small", header="m-small", usage=True):
    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream", "x-vsr-selected-model": header,
                                       "x-vsr-selected-decision": "simple"})
    await resp.prepare(request)
    await resp.write(chunk({"model": model, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]}))
    await asyncio.sleep(0.05)
    for t in ("Hel", "lo"):
        await resp.write(chunk({"model": model, "choices": [{"index": 0, "delta": {"content": t}}]}))
    if usage:
        await resp.write(chunk({"model": model, "choices": [], "usage": {"prompt_tokens": 9, "completion_tokens": 2}}))
    await resp.write(b"data: [DONE]\n\n")
    return resp


def test_a_streamed_reply_meets_the_scheduler_contract():
    status, raw, headers, seen = run(streaming)
    reply, error = parse_reply(status, raw, "c.r.0")
    assert error is None
    assert (reply.model, reply.generated_text, reply.prompt_tokens, reply.output_tokens) == ("m-small", "Hello", 9, 2)
    assert reply.instance_id == "" and reply.timings_ms == {}
    assert 0 < reply.first_token_s == reply.ttft_s < 0.05 <= reply.server_latency_s
    assert headers["c.r.0"]["x-vsr-selected-decision"] == "simple"
    assert seen["rid"] == "c.r.0" and seen["body"]["model"] == C.ENTRYPOINT


def test_a_stream_without_usage_or_on_another_model_is_a_failed_request():
    for handler, why in ((lambda r: streaming(r, usage=False), "no final usage"),
                         (lambda r: streaming(r, header="m-large"), "the router chose")):
        status, raw, _, _ = run(handler)
        reply, error = parse_reply(status, raw, "c.r.0")
        assert reply is None and why in error


def test_an_http_error_is_a_failed_request():
    async def refuse(request):
        return web.Response(status=503, text="upstream_unavailable")
    status, raw, _, _ = run(refuse)
    reply, error = parse_reply(status, raw, "c.r.0")
    assert reply is None and error.startswith("HTTP 503")


def test_a_malformed_stream_event_is_a_failed_request():
    async def bad_delta(request):
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream", "x-vsr-selected-model": "m-small"})
        await resp.prepare(request)
        role = {"index": 0, "delta": {"role": "assistant", "content": ""}}
        await resp.write(chunk({"model": "m-small", "choices": [role]}))
        await resp.write(chunk({"model": "m-small", "choices": [{"index": 0}]}))             # a choice without its delta
        usage = {"prompt_tokens": 9, "completion_tokens": 2}
        await resp.write(chunk({"model": "m-small", "choices": [], "usage": usage}))
        await resp.write(b"data: [DONE]\n\n")
        return resp
    try:
        status, raw, _, _ = run(bad_delta)
    except KeyError:            # the stream's own error, raised to bench.runner, which records the request as failed
        return
    reply, error = parse_reply(status, raw, "c.r.0")
    assert reply is None
