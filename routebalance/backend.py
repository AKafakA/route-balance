"""Client for official vLLM's streaming chat endpoint (D24).

Every system (RouteBalance and each baseline) sends requests through this one client, over one
pooled, bounded aiohttp session owned by the server. The raw user prompt goes in a single chat
message, so the server applies the chat template exactly once. Requests use only the standard
OpenAI-compatible fields of official vLLM (0.18.1); the optional `on_progress` callback reports the
cumulative number of generated tokens (from vLLM's continuous usage stats) as it grows, which the
telemetry view uses for pending decode tokens.

Timing is on the scheduler host's clock (time.perf_counter), taken from the moment generate() is
called: ttft = first streamed message with choices; itl = gaps between later such messages.
"""
from __future__ import annotations

import asyncio
import codecs
import json
import time
from dataclasses import dataclass
from typing import Callable

import aiohttp

from routebalance.types import Instance


class BackendUnreachable(RuntimeError):
    """The request provably never reached the instance: the connection could not be established.

    Any later failure (timeout, disconnect) returns GenerationResult(success=False) instead, because the
    server may have received the request; the caller then closes the telemetry token as completed."""


@dataclass(frozen=True)
class GenerationRequest:
    request_id: str
    prompt: str                    # the raw user prompt, never pre-templated
    max_tokens: int
    temperature: float
    repetition_penalty: float
    frequency_penalty: float

    def __post_init__(self):
        if self.max_tokens < 1:
            raise ValueError(f"{self.request_id}: max_tokens must be >= 1")


@dataclass(frozen=True)
class GenerationResult:
    request_id: str
    instance_id: str
    model: str
    success: bool
    error: str                     # empty when success
    timed_out: bool                # failed because the request timed out (no response, or mid-stream)
    text: str
    prompt_tokens: int             # vLLM's usage count of the templated prompt (final usage chunk); 0 on failure
    output_tokens: int
    ttft_s: float
    itl_s: tuple[float, ...]
    latency_s: float


def _chat_payload(inst: Instance, req: GenerationRequest) -> dict:
    return {
        "model": inst.model,
        "messages": [{"role": "user", "content": req.prompt}],
        "temperature": req.temperature,
        "repetition_penalty": req.repetition_penalty,
        "frequency_penalty": req.frequency_penalty,
        "max_completion_tokens": req.max_tokens,
        "stop_token_ids": list(inst.tier.stop_token_ids),      # the tier's end-of-turn stops (D102)
        "stream": True,
        # continuous usage stats: every chunk carries the cumulative completion_tokens, because vLLM may merge
        # several tokens into one content delta when the engine runs ahead of the API server
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
        "request_id": req.request_id,
    }


class _SSEParser:
    """Splits a byte stream into server-sent events and decodes each event's data as one JSON object.

    Handles LF and CRLF framing (even when a CRLF is split across reads), UTF-8 split across reads,
    comment lines, several data: lines per event (joined with newlines), and the [DONE] sentinel.
    """

    def __init__(self):
        self._decoder = codecs.getincrementaldecoder("utf-8")()
        self._buffer = ""

    def _normalized(self, final: bool) -> str:
        buf = self._buffer.replace("\r\n", "\n")
        keep = "" if final or not buf.endswith("\r") else "\r"   # a trailing CR may be half of a CRLF
        body = buf[:-1] if keep else buf
        self._buffer = keep
        return body.replace("\r", "\n")

    def _events(self, text: str) -> list[dict]:
        events = []
        for raw in text.split("\n\n"):
            data = [line[len("data:"):].lstrip(" ") for line in raw.split("\n") if line.startswith("data:")]
            other = [line for line in raw.split("\n") if line and not line.startswith(("data:", ":"))]
            if other:
                raise ValueError(f"unexpected SSE line {other[0][:80]!r}")
            if not data:
                continue                            # comment-only or empty event
            payload = "\n".join(data)
            if payload.strip() == "[DONE]":
                continue
            event = json.loads(payload)
            if not isinstance(event, dict):
                raise ValueError(f"SSE event is not an object: {payload[:80]!r}")
            events.append(event)
        return events

    def feed(self, chunk: bytes) -> list[dict]:
        self._buffer += self._decoder.decode(chunk)
        text = self._normalized(final=False)
        complete, sep, rest = text.rpartition("\n\n")
        self._buffer = rest + self._buffer
        return self._events(complete) if sep else []

    def finish(self) -> list[dict]:
        """Everything left, including a final event that was not followed by a blank line."""
        self._buffer += self._decoder.decode(b"", final=True)
        text = self._normalized(final=True)
        self._buffer = ""
        return self._events(text) if text.strip() else []


def _choices(event: dict) -> list:
    choices = event.get("choices")     # vLLM sends choices in every event, [] in the final usage chunk
    if not isinstance(choices, list) or not all(isinstance(c, dict) for c in choices):
        raise ValueError(f"malformed choices {choices!r:.80}")
    return choices


def _raise_if_error(event: dict) -> None:
    # vLLM ends a stream that fails mid-generation with {"error": {...}} and then [DONE], with no final usage
    if "error" in event:
        err = event["error"]
        msg = err.get("message") if isinstance(err, dict) else err
        raise _ServerError(f"{msg!r:.200}")


class _ServerError(Exception):
    pass


def _completion_tokens(event: dict) -> int | None:
    usage = event.get("usage")
    if usage is None:
        return None
    n = usage.get("completion_tokens") if isinstance(usage, dict) else None
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise ValueError(f"malformed usage {usage!r:.80}")
    return n


def _prompt_tokens(event: dict) -> int:
    n = event["usage"].get("prompt_tokens")
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise ValueError(f"malformed usage.prompt_tokens {n!r:.40}")
    return n


class BackendClient:
    def __init__(self, session: aiohttp.ClientSession, timeout_s: float):
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self._session = session
        self._timeout = aiohttp.ClientTimeout(total=timeout_s)

    async def generate(self, inst: Instance, req: GenerationRequest,
                       on_progress: Callable[[int], None] | None = None) -> GenerationResult:
        start = time.perf_counter()
        url = f"{inst.base_url}/v1/chat/completions"
        pieces: list[str] = []
        itl: list[float] = []
        ttft = last = None
        output_tokens = None
        prompt_tokens = 0
        final_usage = False     # the usage-only chunk (choices == []) vLLM sends after a stream that finished
        def result(success: bool, error: str, timed_out: bool = False) -> GenerationResult:   # default-ok: most outcomes
            return GenerationResult(req.request_id, inst.instance_id, inst.model, success, error, timed_out, "".join(pieces),
                                    prompt_tokens, output_tokens if output_tokens is not None else 0,
                                    ttft if ttft is not None else 0.0, tuple(itl), time.perf_counter() - start)

        try:
            response = await self._session.post(url, json=_chat_payload(inst, req),
                                                headers={"X-Request-Id": req.request_id}, timeout=self._timeout)
        except aiohttp.ClientConnectorError as e:
            # the connection was never established, so the request cannot have reached the server
            raise BackendUnreachable(f"{inst.instance_id}: {type(e).__name__}: {e}") from e
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:     # 3.10: not the builtin TimeoutError
            # sent (or possibly sent) but no response: the server may be working on it
            return result(False, f"no response: {type(e).__name__}: {e}", isinstance(e, asyncio.TimeoutError))

        async with response:
            if response.status != 200:
                return result(False, f"HTTP {response.status} {response.reason}")
            parser = _SSEParser()
            try:
                async for chunk in response.content.iter_any():
                    events = parser.feed(chunk)
                    now = time.perf_counter()
                    for ev in events:
                        _raise_if_error(ev)
                        tokens = _completion_tokens(ev)
                        if tokens is not None:
                            if output_tokens is not None and tokens < output_tokens:
                                raise ValueError(f"completion_tokens went backwards ({output_tokens} -> {tokens})")
                            reported = 0 if output_tokens is None else output_tokens
                            if on_progress is not None and tokens > reported:    # the role chunk carries 0
                                on_progress(tokens)
                            output_tokens = tokens
                        choices = _choices(ev)
                        if choices and tokens is None:      # continuous usage was requested for every chunk
                            raise ValueError("a streamed chunk without its usage")
                        if not choices:
                            if tokens is not None and ev.get("choices") == []:
                                final_usage = True
                                prompt_tokens = _prompt_tokens(ev)
                            continue
                        if ttft is None:
                            ttft = now - start
                        else:
                            itl.append(now - last)
                        last = now
                        delta = choices[0]["delta"]                 # a choice without one is malformed
                        if not isinstance(delta, dict):
                            raise ValueError(f"malformed delta {delta!r:.80}")
                        text = delta.get("content") if delta else None    # the first delta carries only the role
                        if text is not None and not isinstance(text, str):
                            raise ValueError(f"malformed content {text!r:.80}")
                        if text:
                            pieces.append(text)
                for ev in parser.finish():
                    _raise_if_error(ev)
                    if _choices(ev):
                        raise ValueError("stream ended with an unterminated event carrying text")
                    tokens = _completion_tokens(ev)
                    if tokens is not None:
                        if output_tokens is not None and tokens < output_tokens:
                            raise ValueError(f"completion_tokens went backwards ({output_tokens} -> {tokens})")
                        output_tokens = tokens
                        if ev.get("choices") == []:
                            final_usage = True
                            prompt_tokens = _prompt_tokens(ev)
            except _ServerError as e:
                return result(False, f"server error: {e}")
            except (ValueError, KeyError, TypeError) as e:
                return result(False, f"malformed stream: {e}")
            except (aiohttp.ClientPayloadError, aiohttp.ClientConnectionError, asyncio.TimeoutError) as e:
                return result(False, f"stream interrupted: {type(e).__name__}: {e}", isinstance(e, asyncio.TimeoutError))
        if ttft is None:
            return result(False, "no streamed message with choices")
        if not final_usage:
            return result(False, "no final usage chunk: the stream did not finish")
        return result(True, "")
