"""Test doubles for the benchmark: a scripted scheduler client, a tokenizer, a local HTTP scheduler."""
from __future__ import annotations

import asyncio
import hashlib
import json
import threading
import time

from aiohttp import web

from bench.records import SchedulerIdentity

CONFIG_DIGEST = "c" * 64
FLEET_DIGEST = "f" * 64
IDENTITY = SchedulerIdentity(config_digest=CONFIG_DIGEST, fleet_digest=FLEET_DIGEST)


SAMPLING = {"temperature": 0.0, "repetition_penalty": 1.0, "frequency_penalty": 1.2, "max_tokens": 64}


def identity(config_digest: str = CONFIG_DIGEST) -> SchedulerIdentity:
    return SchedulerIdentity(config_digest=config_digest, fleet_digest=FLEET_DIGEST)


def write_dataset(directory, n_rows: int = 12):
    path = directory / "eval.jsonl"
    path.write_text("".join(json.dumps({"id": f"p{i}", "prompt": f"question {i}?"}) + "\n" for i in range(n_rows)))
    return path


def scope_dict(dataset_path, configs: dict, seeds=(1,), prompt_order="fixed", n_requests: int = 12,
               rate: float = 200.0, accounted=None, max_send_lag_ms: float = 1000.0) -> dict:
    """A one-block scope (group "g1") over `configs` = {config_id: (system, config_digest)}."""
    rows = len(dataset_path.read_text().splitlines())
    return {
        "bench": {"sampling": SAMPLING, "max_connections": 64, "request_timeout_s": 10.0,
                  "max_send_lag_ms": max_send_lag_ms},
        "datasets": {"evalset": {"path": dataset_path.name, "sha256": hashlib.sha256(dataset_path.read_bytes())
                                 .hexdigest(), "rows": rows, "tokenizer_id": "tok-id", "budgets": None}},
        "traces": {"l200": {"burstiness": 1.0, "segments": [{"n_requests": n_requests, "rate": rate}]}},
        "configs": {cid: {"system": system, "knob": cid, "dispatch": "lor", "config_digest": digest}
                    for cid, (system, digest) in configs.items()},
        "blocks": [{"name": "main", "group": "g1", "dataset": "evalset", "configs": list(configs), "traces": ["l200"],
                    "seeds": list(seeds), "prompt_order": prompt_order}],
        "groups": {"g1": {"reference": None, "intended_differences": {cid: ["system.*"] for cid in configs}}},
        "accounted": {} if accounted is None else accounted,
        "pending": {},
    }


class FakeTokenizer:
    """One token per character of '<u>' + message + '</u><a>'; records every call."""

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, conversation, *, add_generation_prompt, tokenize):
        self.calls.append((conversation, add_generation_prompt, tokenize))
        return list(range(len("<u>" + conversation[0]["content"] + "</u><a>")))


def reply_body(request_id: str, model: str = "m-small", output_tokens: int = 10, **overrides) -> dict:
    body = {"success": True, "request_id": request_id, "model": model, "instance_id": f"{model}_0",
            "generated_text": "ok", "prompt_tokens": 12, "output_tokens": output_tokens, "ttft_s": 0.001,
            "server_latency_s": 0.004, "timings_ms": {"queue_ms": 0.5, "decide_ms": 0.25}}
    body.update(overrides)
    return body


class FakeScheduler:
    """A SchedulerClient answering from a script. latency(index) and fail(index) pick each request's delay and
    whether it fails (HTTP 500); raise_on(index) makes complete() raise. identities are returned in order.
    Replies echo the sent prompt_tokens plus extra_prompt_tokens (19 imitates a second chat template)."""

    def __init__(self, latency=lambda i: 0.002, fail=lambda i: False, raise_on=lambda i: False,
                 identities=(IDENTITY, IDENTITY), model=lambda i: "m-small", extra_prompt_tokens: int = 0):
        self.max_connections = 64
        self.timeout_s = 10.0
        self.connection_waits = 0
        self.calls: list[tuple[int, float]] = []          # (request index, perf_counter at the call)
        self._latency, self._fail, self._raise_on, self._model = latency, fail, raise_on, model
        self._extra = extra_prompt_tokens
        self._identities = list(identities)

    async def identity(self) -> SchedulerIdentity:
        ident = self._identities.pop(0)
        if isinstance(ident, Exception):
            raise ident
        return ident

    async def complete(self, body: bytes) -> tuple[int, bytes]:
        req = json.loads(body)
        index = int(req["request_id"].rsplit(".", 1)[1])
        self.calls.append((index, time.perf_counter()))
        await asyncio.sleep(self._latency(index))
        if self._raise_on(index):
            raise RuntimeError(f"connection reset on {index}")
        if self._fail(index):
            return 500, b"boom"
        return 200, json.dumps(reply_body(req["request_id"], model=self._model(index),
                                          prompt_tokens=req["prompt_tokens"] + self._extra)).encode()


class LocalScheduler:
    """A real HTTP scheduler stub on a local port, served from its own thread and event loop.
    It records the peer port of every completion (one per TCP connection) and the peak concurrency."""

    def __init__(self, delay_s: float = 0.0, identities: list | None = None):
        """identities: the GET /v1/config bodies, in order; the last one repeats."""
        self.delay_s = delay_s
        self.identities = [{"config_digest": CONFIG_DIGEST, "fleet_digest": FLEET_DIGEST, "mode": "test"}] \
            if identities is None else list(identities)
        self.peers: list[int] = []
        self.bodies: list[dict] = []           # every POST /v1/completions body, as received
        self.active = 0
        self.peak = 0
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)

    async def _complete(self, request: web.Request) -> web.Response:
        self.peers.append(request.transport.get_extra_info("peername")[1])
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            req = await request.json()
            self.bodies.append(req)
            await asyncio.sleep(self.delay_s)
            return web.json_response(reply_body(req["request_id"], prompt_tokens=req["prompt_tokens"]))
        finally:
            self.active -= 1

    async def _config(self, request: web.Request) -> web.Response:
        return web.json_response(self.identities.pop(0) if len(self.identities) > 1 else self.identities[0])

    def _serve(self) -> None:
        asyncio.set_event_loop(self._loop)
        app = web.Application()
        app.add_routes([web.post("/v1/completions", self._complete), web.get("/v1/config", self._config)])
        self._runner = web.AppRunner(app)
        self._loop.run_until_complete(self._runner.setup())
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        self._loop.run_until_complete(site.start())
        port = self._runner.addresses[0][1]
        self.url = f"http://127.0.0.1:{port}"
        self._ready.set()
        self._loop.run_forever()

    def __enter__(self) -> "LocalScheduler":
        self._thread.start()
        self._ready.wait(10)
        return self

    def __exit__(self, *exc) -> None:
        asyncio.run_coroutine_threadsafe(self._runner.cleanup(), self._loop).result(10)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(10)
