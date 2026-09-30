"""The benchmark client's side of vLLM-SR: bench.client's SchedulerClient, speaking OpenAI chat to Envoy's listener.

bench.runner hands complete() the scheduler request body it sends every system (bench.client.request_body); this
client sends the same prompt and sampling as one streaming chat request to the router's entrypoint, with only the
fields vLLM-SR v0.4.0 accepts (constants.REQUEST_FIELDS; D151: no stop), the request id in the x-request-id header
(Envoy keeps it, logs it and vLLM uses it), and returns the reply in the scheduler contract's shape:
  model             the router's choice (x-vsr-selected-model), which must be the model the stream reports
  instance_id       joined after the run from Envoy's access log (join.py); "" until then
  ttft_s                   the client's clock from the send to the first streamed token
  first_token_s            the SLO's TTFT (D57, D161): set by the join, from Envoy's own request start (vLLM-SR's front
                           door) to this first token, as our scheduler's runs from its handler; the client's send-to-first
                           value until then, and the first token's wall-clock time is kept (self.first_wall)
The stream is parsed chunk by chunk as it arrives, as routebalance/backend.py parses vLLM's and as vLLM bench's own client
does (a burst of parsing at a stream's end would delay the other requests' timestamps on the same loop). The
scheduler's loop-lag probe runs on the client's loop (self.lag) as the record of the client's own work.
  server_latency_s  the client's clock from the send to the end of the stream
  prompt_tokens, output_tokens    vLLM's usage counts (the stream's final usage chunk)
  timings_ms        the router's own timings, joined after the run from its log (join.py); {} until then
The routing headers of every reply are kept (self.headers) for the join.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Optional

import aiohttp

from bench.client import check_fd_limit
from bench.records import SchedulerIdentity
from routebalance.backend import _SSEParser, stock_connector
from routebalance.config import ConfigError, loads_strict
from routebalance.decision_log import LagProbe
from vsrbench import constants as C

ROUTING_HEADERS = ("x-vsr-selected-model", "x-vsr-selected-decision", "x-vsr-selected-recipe",
                   "x-vsr-selected-confidence", "x-vsr-selected-algorithm", "x-vsr-replay-id")


def chat_body(request: dict) -> dict:
    """The chat request for one scheduler request (bench.client.request_body)."""
    if request.get("budget_cost") is not None:
        raise ConfigError("vLLM-SR takes no per-request budget: budget cells are not run through it")
    body = {"model": C.ENTRYPOINT, "messages": [{"role": "user", "content": request["prompt"]}],
            "temperature": request["temperature"], "repetition_penalty": request["repetition_penalty"],
            "frequency_penalty": request["frequency_penalty"], "max_completion_tokens": request["max_tokens"],
            "stream": True, "stream_options": {"include_usage": True}}
    assert tuple(body) == C.REQUEST_FIELDS
    return body


class VsrClient:
    """One aiohttp session on vLLM bench's stock connector (D134), as HttpSchedulerClient; a request that waits for a
    pooled connection is counted (a cell with any wait is not open loop)."""

    def __init__(self, base_url: str, max_connections: int, timeout_s: float, identity):
        if max_connections < 1 or not timeout_s > 0:
            raise ConfigError("max_connections must be >= 1 and timeout_s > 0")
        check_fd_limit(max_connections)
        self._url = base_url.rstrip("/") + "/v1/chat/completions"
        self.max_connections, self.timeout_s = max_connections, timeout_s
        self.connection_waits = 0
        self._identity = identity            # async () -> SchedulerIdentity, read from the live stack
        self._session: Optional[aiohttp.ClientSession] = None
        self.headers: dict[str, dict[str, str]] = {}
        self.first_wall: dict[str, float] = {}     # the first token's wall-clock time per request
        self.lag = LagProbe()                      # the scheduler's event-loop lag probe, on this client's loop
        self._lag_task: Optional[asyncio.Task] = None

    async def __aenter__(self) -> "VsrClient":
        trace = aiohttp.TraceConfig()
        trace.on_connection_queued_start.append(self._on_queued)
        self._session = aiohttp.ClientSession(connector=stock_connector(), trace_configs=[trace],
                                              timeout=aiohttp.ClientTimeout(total=self.timeout_s))
        self._lag_task = asyncio.create_task(self.lag.run())
        return self

    async def __aexit__(self, *exc) -> None:
        self._lag_task.cancel()
        await self._session.close()

    async def _on_queued(self, session, ctx, params) -> None:
        self.connection_waits += 1

    async def identity(self) -> SchedulerIdentity:
        return await self._identity()

    async def complete(self, body: bytes) -> tuple[int, bytes]:
        request = loads_strict(body.decode("utf-8"))
        rid = request["request_id"]
        start = time.perf_counter()
        async with self._session.post(self._url, json=chat_body(request),
                                      headers={"x-request-id": rid}) as resp:
            if resp.status != 200:
                return resp.status, await resp.read()
            self.headers[rid] = {h: resp.headers[h] for h in ROUTING_HEADERS if h in resp.headers}
            state = _Stream(start)
            async for chunk in resp.content.iter_any():      # each chunk as it arrives, as backend.py and vLLM bench
                events = state.parser.feed(chunk)
                now, wall = time.perf_counter(), time.time()   # one reading: the first token's clock and wall time
                had_first = state.first is not None
                for event in events:
                    state.event(event, now)
                if not had_first and state.first is not None:
                    self.first_wall[rid] = wall             # for the Envoy join (D161)
            for event in state.parser.finish():
                state.event(event, time.perf_counter())
            end = time.perf_counter() - start
        chosen = self.headers[rid].get("x-vsr-selected-model")
        if state.first is None:
            return 200, _fail("no streamed message with choices")
        if state.usage is None:
            return 200, _fail("no final usage chunk: the stream did not finish")
        if chosen is None or state.model != chosen:
            return 200, _fail(f"the router chose {chosen!r}, the stream reports {state.model!r}")
        reply = {"success": True, "request_id": rid, "model": chosen, "instance_id": "",
                 "generated_text": "".join(state.pieces), "prompt_tokens": state.usage["prompt_tokens"],
                 "output_tokens": state.usage["completion_tokens"], "ttft_s": state.first, "server_latency_s": end,
                 "timings_ms": {}, "first_token_s": state.first}
        return 200, json.dumps(reply).encode()


def _fail(error: str) -> bytes:
    return json.dumps({"success": False, "error": error}).encode()


class _Stream:
    """One reply's stream, read as routebalance/backend.py reads vLLM's: the first event with choices (vLLM's role
    chunk) is the first token, timed when its bytes were read; the usage-only event (choices == []) ends it; an error
    event is a failed request."""

    def __init__(self, start: float):
        self.parser, self.start = _SSEParser(), start
        self.pieces: list[str] = []
        self.first: Optional[float] = None
        self.usage: Optional[dict] = None
        self.model: Optional[str] = None

    def event(self, event: dict, now: float) -> None:
        if "error" in event:
            raise RuntimeError(f"server error: {str(event['error'])[:300]}")
        self.model = event.get("model", self.model)
        choices = event.get("choices")     # as routebalance/backend.py: a list of objects, [] in the usage-only event
        if not isinstance(choices, list) or not all(isinstance(c, dict) for c in choices):
            raise ValueError(f"malformed choices {choices!r:.80}")
        if not choices:
            if event.get("usage"):
                self.usage = event["usage"]
            return
        if self.first is None:
            self.first = now - self.start
        delta = choices[0]["delta"]                     # a choice without one is malformed
        if not isinstance(delta, dict):
            raise ValueError(f"malformed delta {delta!r:.80}")
        text = delta.get("content") if delta else None  # the first delta carries only the role
        if text is not None and not isinstance(text, str):
            raise ValueError(f"malformed content {text!r:.80}")
        if text:
            self.pieces.append(text)
