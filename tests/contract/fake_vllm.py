"""A fake vLLM (0.18.1) server for contract tests.

It serves /health, /version and /v1/models (shaped like vLLM's replies), the Prometheus /metrics exposition
(with the gauges the telemetry view reads, among histogram noise like the real one) and a streaming
/v1/chat/completions.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field

from aiohttp import web


@dataclass
class FakeState:
    running: int = 0  # default-ok: an idle engine
    waiting: int = 0  # default-ok: an idle engine
    kv_usage: float = 0.05  # default-ok: an idle engine's cache
    preemptions: int = 0  # default-ok: none yet
    status: int = 200  # default-ok: healthy unless a test says otherwise
    omit: tuple[str, ...] = ()
    duplicate: tuple[str, ...] = ()
    raw_body: bytes | None = None  # when set, served verbatim as /metrics
    metrics_drop: bool = False  # default-ok: /metrics drops the connection unanswered (a network failure)
    # default-ok: ok | http500 | no_choices | no_usage | malformed | cut | crlf | multiline | null_event |
    # null_choice | int_content | unterminated | slow_headers | close_after_body | merged | engine_error
    # | no_final_usage (continuous usage on every chunk, but the stream stops before the final usage chunk)
    # | no_delta (a choice without its delta, then a complete finish)
    chat_mode: str = "ok"  # default-ok: see the list above
    chat_tokens: tuple[str, ...] = ("Hello", " world")
    chat_delay_s: float = 0.0  # default-ok: no pacing unless a test asks for it
    chat_requests: list[dict] = field(default_factory=list)
    scrapes: int = 0  # default-ok: counts /metrics reads
    version: str = "0.18.1"  # default-ok: the campaign's vLLM
    served: tuple[str, ...] = ()  # default-ok: the model ids /v1/models lists (vLLM's served model names)
    live_running: bool = False  # default-ok: report the chat streams in flight as the running gauge
    streams: int = 0  # default-ok: chat streams in flight
    start_time_s: float = 1790000000.0  # default-ok: process_start_time_seconds (a restart changes it)
    # default-ok: the labels of vllm:cache_config_info, as vLLM 0.18.1 reports them (a subset)
    cache_config: tuple[tuple[str, str], ...] = (("block_size", "16"), ("cache_dtype", "auto"),
                                                 ("enable_prefix_caching", "True"), ("gpu_memory_utilization", "0.9"),
                                                 ("num_gpu_blocks", "8000"))
    # the prefix cache, as vLLM counts it (tokens): every chat request queries its prompt tokens (5); a prompt already
    # cached hits 4 of them. /reset_prefix_cache (served only in vLLM's developer mode) empties it.
    dev_mode: bool = True  # default-ok: VLLM_SERVER_DEV_MODE=1, as the lease engines run
    reset_ignored: bool = False  # default-ok: a reset that answers 200 and empties nothing (the proof must catch it)
    cached: set = field(default_factory=set)
    prefix_queries: int = 0  # default-ok: a fresh engine
    prefix_hits: int = 0  # default-ok: a fresh engine
    resets: int = 0  # default-ok: counts /reset_prefix_cache calls
    # vLLM's serving counters (a cell's engine readings must change by exactly its own requests)
    served_requests: int = 0  # default-ok: a fresh engine
    served_prompt_tokens: int = 0  # default-ok: a fresh engine
    served_generation_tokens: int = 0  # default-ok: a fresh engine

    def metrics_text(self) -> str:
        self.scrapes += 1
        labels = '{engine="0",model_name="M"}'
        samples = {
            "vllm:num_requests_running": self.streams if self.live_running else self.running,
            "vllm:num_requests_waiting": self.waiting,
            "vllm:kv_cache_usage_perc": self.kv_usage,
            "vllm:num_preemptions_total": self.preemptions,
            "vllm:prefix_cache_queries_total": self.prefix_queries,
            "vllm:prefix_cache_hits_total": self.prefix_hits,
            "vllm:request_success_total": self.served_requests,
            "vllm:prompt_tokens_total": self.served_prompt_tokens,
            "vllm:generation_tokens_total": self.served_generation_tokens,
        }
        lines = ["# HELP vllm:num_requests_running Number of requests in model execution batches.",
                 "# TYPE vllm:num_requests_running gauge"]
        for name, value in samples.items():
            if name in self.omit:
                continue
            lines.append(f"{name}{labels} {float(value)}")
            if name in self.duplicate:
                lines.append(f"{name}{labels} {float(value)}")
        lines += [f'vllm:e2e_request_latency_seconds_bucket{{engine="0",le="{b}",model_name="M"}} 3.0'
                  for b in (0.3, 0.5, 1.0, "+Inf")]
        lines.append('vllm:num_preemptions_created{engine="0",model_name="M"} 1.7e9')
        lines += ["# HELP process_start_time_seconds Start time of the process since unix epoch in seconds.",
                  "# TYPE process_start_time_seconds gauge", f"process_start_time_seconds {self.start_time_s}"]
        info = ",".join(f'{k}="{v}"' for k, v in self.cache_config)
        lines += ["# TYPE vllm:cache_config_info gauge", f'vllm:cache_config_info{{{info},engine="0"}} 1.0']
        return "\n".join(lines) + "\n"


def make_app(state: FakeState) -> web.Application:
    async def health(_request):
        return web.Response(status=200 if state.status == 200 else 503)

    async def version(_request):
        return web.json_response({"version": state.version})

    async def models(_request):
        return web.json_response({"object": "list", "data": [
            {"id": m, "object": "model", "created": 1759000000, "owned_by": "vllm", "root": m, "parent": None,
             "max_model_len": 4096, "permission": []} for m in state.served]})

    async def metrics(request):
        if state.metrics_drop:
            request.transport.close()
            return web.Response(status=200)
        if state.status != 200:
            return web.Response(status=state.status)
        if state.raw_body is not None:
            return web.Response(status=200, body=state.raw_body, content_type="text/plain")
        return web.Response(status=200, text=state.metrics_text(), content_type="text/plain")

    async def chat(request):
        state.streams += 1
        try:
            return await stream(request)
        finally:
            state.streams -= 1

    async def stream(request):
        payload = await request.json()
        state.chat_requests.append(payload)
        mode = state.chat_mode
        if mode == "http500":
            return web.Response(status=500, reason="Internal Server Error")
        text = payload["messages"][0]["content"]
        state.prefix_queries += 5
        state.prefix_hits += 4 if text in state.cached else 0
        state.cached.add(text)
        if mode == "close_after_body":
            request.transport.close()              # received the request, then drop the connection
            return web.Response(status=200)
        if mode == "slow_headers":
            await asyncio.sleep(1.0)               # received, but headers come after the client's timeout
        resp = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        end = "\r\n\r\n" if mode == "crlf" else "\n\n"

        continuous = bool((payload.get("stream_options") or {}).get("continuous_usage_stats")) and mode != "no_usage"
        produced = [0]

        async def send(obj, tokens=0):
            # like vLLM (pydantic model_dump_json): non-ASCII characters, incl. U+2028, are left raw
            produced[0] += tokens
            if continuous and "usage" not in obj:
                obj = dict(obj, usage={"prompt_tokens": 5, "completion_tokens": produced[0]})
            await resp.write(f"data: {json.dumps(obj, ensure_ascii=False)}{end}".encode())

        special = {
            "malformed": b"data: {not json}\n\n",
            "null_event": b"data: null\n\n",
            "null_choice": b'data: {"choices": [null]}\n\n',
            "int_content": b'data: {"choices": [{"delta": {"content": 5}}]}\n\n',
            "no_delta": b'data: {"choices": [{"index": 0}], "usage": {"prompt_tokens": 5, "completion_tokens": 1}}\n\n'
                        b'data: {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 1}}\n\n'
                        b'data: [DONE]\n\n',
        }
        if mode in special:
            await resp.write(special[mode])
            return resp
        if mode == "multiline":                    # one SSE event whose data spans two lines
            await resp.write(b'data: {"choices":\ndata: [{"delta": {"role": "assistant"}}], '
                             b'"usage": {"prompt_tokens": 5, "completion_tokens": 0}}\n\n')
            await resp.write(b'data: {"choices": [{"delta": {"content": "x"}}], '
                             b'"usage": {"prompt_tokens": 5, "completion_tokens": 1}}\n\n')
            await resp.write(b'data: {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 1}}\n\n'
                             b'data: [DONE]\n\n')
            return resp
        if mode == "unterminated":
            await send({"choices": [{"delta": {"role": "assistant"}}]})
            await resp.write(b'data: {"choices": [{"delta": {"content": "b"}}]}')     # no blank line, then EOF
            return resp
        if mode == "engine_error":                 # vLLM's streaming error path: an error event, then [DONE]
            await send({"choices": [{"delta": {"role": "assistant"}}]})
            for tok in state.chat_tokens[:2]:
                await send({"choices": [{"delta": {"content": tok}}]}, tokens=1)
            await send({"error": {"message": "engine died", "type": "InternalServerError", "code": 500}})
            await resp.write(f"data: [DONE]{end}".encode())
            return resp
        if mode == "merged":                       # vLLM merges tokens when the engine runs ahead of the stream
            await send({"choices": [{"delta": {"role": "assistant"}}]})
            await send({"choices": [{"delta": {"content": "".join(state.chat_tokens[:-1])}}]},
                       tokens=len(state.chat_tokens) - 1)
            await send({"choices": [{"delta": {"content": state.chat_tokens[-1]}}]}, tokens=1)
            await send({"choices": [{"delta": {}, "finish_reason": "stop"}]})
            await send({"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": len(state.chat_tokens)}})
            await resp.write(f"data: [DONE]{end}".encode())
            return resp
        if mode != "no_choices":
            await send({"choices": [{"delta": {"role": "assistant"}}]})   # role-only delta, like vLLM
            for i, tok in enumerate(state.chat_tokens):
                await asyncio.sleep(state.chat_delay_s)
                await send({"choices": [{"delta": {"content": tok}}]}, tokens=1)
                if mode == "cut" and i == 0:
                    request.transport.close()                              # drop the connection mid-stream
                    return resp
        if mode not in ("no_usage", "no_final_usage"):
            await send({"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": len(state.chat_tokens)}})
        if mode == "ok":                           # vLLM counts a request when it finishes
            state.served_requests += 1
            state.served_prompt_tokens += 5
            state.served_generation_tokens += len(state.chat_tokens)
        await resp.write(f": keep-alive{end}data: [DONE]{end}".encode())
        return resp

    async def reset_prefix_cache(_request):
        if not state.dev_mode:
            return web.Response(status=404, text='{"detail":"Not Found"}')
        state.resets += 1
        if not state.reset_ignored:
            state.cached.clear()
        return web.Response(status=200)

    app = web.Application()
    app.router.add_post("/reset_prefix_cache", reset_prefix_cache)
    app.router.add_get("/health", health)
    app.router.add_get("/version", version)
    app.router.add_get("/v1/models", models)
    app.router.add_get("/metrics", metrics)
    app.router.add_post("/v1/chat/completions", chat)
    return app
