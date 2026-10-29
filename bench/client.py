"""The benchmark client's side of the scheduler API, and its pooled HTTP implementation.

Contract (the scheduler server implements the other side):

POST /v1/completions   one request, JSON body, non-streaming
  request_id           str    unique within the scheduler session: "<cell_id>.<run_id>.<index>"
  prompt               str    the raw user prompt. The scheduler sends it to vLLM's chat endpoint as the single
                              user message; vLLM applies the chat template, once
  prompt_tokens        int    tokens of that templated message (bench/dataset.py); every system reads this
  max_tokens           int    the output cap
  temperature, repetition_penalty, frequency_penalty    float, passed to vLLM unchanged
  budget_cost          float  the request's cost budget in USD, only in budget cells (the scheduler clamps the
                              output to it, routebalance/schedule/budget.py); null otherwise
Reply: HTTP 200 with {"success": true} plus exactly the fields of bench.records.Reply:
  request_id           the echo of the request's id
  model                the served model (HF id, as priced in the fleet config and scored in the corpora)
  instance_id          the instance that served it
  generated_text       the completion
  prompt_tokens, output_tokens      vLLM's usage counts
  ttft_s, server_latency_s          the scheduler's backend-client clock: from its call to the instance until
                                    the first token, and until the end of the stream
  first_token_s        the scheduler's clock from the request's arrival to the first backend token (the SLO's TTFT)
  timings_ms           the scheduler's own time for this request by stage, in ms (finite, >= 0)
Anything else is a failed request: a non-200 status, success false, a missing, extra or mistyped field, a
different request_id, a transport error or a timeout. It is recorded with its error text and never retried.

GET /v1/config returns the live configuration including config_digest and fleet_digest
(routebalance.config.config_digest / fleet_digest). The runner reads only those two.
"""
from __future__ import annotations

import json
import resource
from typing import Optional, Protocol

import aiohttp
from pydantic import ValidationError

from bench.dataset import Prompt
from bench.records import Reply, Sampling, SchedulerIdentity
from routebalance.backend import stock_connector
from bench.schema import parse
from routebalance.config import ConfigError, loads_strict

_JSON = {"Content-Type": "application/json"}
_FD_HEADROOM = 256            # descriptors kept free for everything that is not a scheduler connection
_ERROR_CHARS = 2000           # a failure keeps this much of the reply body


class SchedulerClient(Protocol):
    max_connections: int      # the connection pool bound
    timeout_s: float          # per-request timeout
    connection_waits: int     # requests that had to wait for a free pooled connection

    async def identity(self) -> SchedulerIdentity: ...

    async def complete(self, body: bytes) -> tuple[int, bytes]: ...


def request_body(request_id: str, prompt: Prompt, sampling: Sampling) -> bytes:
    body = {
        "request_id": request_id,
        "prompt": prompt.text,
        "prompt_tokens": prompt.prompt_tokens,
        "max_tokens": sampling.max_tokens,
        "temperature": sampling.temperature,
        "repetition_penalty": sampling.repetition_penalty,
        "frequency_penalty": sampling.frequency_penalty,
        "budget_cost": prompt.budget_cost,
    }
    return json.dumps(body, allow_nan=False).encode()


def parse_reply(status: int, body: bytes, request_id: str) -> tuple[Optional[Reply], Optional[str]]:
    """(reply, None) for a successful request, (None, error text) for a failed one."""
    shown = body.decode("utf-8", errors="replace")[:_ERROR_CHARS]      # for error messages only
    if status != 200:
        return None, f"HTTP {status}: {shown}"
    try:
        data = loads_strict(body.decode("utf-8"))
    except (UnicodeDecodeError, ConfigError) as e:
        return None, f"unreadable reply: {e}"
    if not isinstance(data, dict) or not isinstance(data.get("success"), bool):
        return None, f"malformed reply (no boolean 'success'): {shown}"
    if not data["success"]:
        return None, f"scheduler error: {shown}"
    fields = {k: v for k, v in data.items() if k != "success"}
    try:
        reply = Reply.model_validate(fields)
    except ValidationError as e:
        return None, f"malformed reply: {e}"
    if reply.request_id != request_id:
        return None, f"reply is for request {reply.request_id!r}, not {request_id!r}"
    return reply, None


def check_fd_limit(max_connections: int) -> None:
    soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft != resource.RLIM_INFINITY and soft < max_connections + _FD_HEADROOM:
        raise ConfigError(f"open-file limit {soft} is below max_connections {max_connections} + {_FD_HEADROOM}; "
                          "raise it (ulimit -n) before running the cell")


class HttpSchedulerClient:
    """One aiohttp session and one bounded connection pool for the whole cell (a session per request churns
    file descriptors under load and can fail whole cells). A request that finds the pool
    full waits for a connection; each wait is counted, and a cell with any wait is invalid, because the
    client then held back sends and the load was no longer open loop."""

    def __init__(self, base_url: str, max_connections: int, timeout_s: float):
        if max_connections < 1 or not timeout_s > 0:
            raise ConfigError("max_connections must be >= 1 and timeout_s > 0")
        check_fd_limit(max_connections)
        self._base_url = base_url.rstrip("/")
        self.max_connections = max_connections
        self.timeout_s = timeout_s
        self._session: Optional[aiohttp.ClientSession] = None
        self.connection_waits = 0

    async def __aenter__(self) -> "HttpSchedulerClient":
        trace = aiohttp.TraceConfig()
        trace.on_connection_queued_start.append(self._on_queued)
        # vLLM bench's stock connector (no connection limit, keep-alive); max_connections sizes the open-file check
        self._session = aiohttp.ClientSession(connector=stock_connector(), trace_configs=[trace],
                                              timeout=aiohttp.ClientTimeout(total=self.timeout_s))
        return self

    async def __aexit__(self, *exc) -> None:
        await self._session.close()

    async def _on_queued(self, session, ctx, params) -> None:
        self.connection_waits += 1

    async def complete(self, body: bytes) -> tuple[int, bytes]:
        async with self._session.post(f"{self._base_url}/v1/completions", data=body, headers=_JSON) as resp:
            return resp.status, await resp.read()

    async def identity(self) -> SchedulerIdentity:
        async with self._session.get(f"{self._base_url}/v1/config") as resp:
            text = await resp.text()
            if resp.status != 200:
                raise ConfigError(f"GET /v1/config returned HTTP {resp.status}: {text[:_ERROR_CHARS]}")
        data = loads_strict(text)
        if not isinstance(data, dict):
            raise ConfigError("GET /v1/config did not return an object")
        return parse(SchedulerIdentity, {k: data.get(k) for k in ("config_digest", "fleet_digest")},
                     "scheduler identity")
