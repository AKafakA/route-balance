"""The scheduler API contract on the client side, and the pooled, bounded HTTP client."""
import asyncio
import json
import resource

import pytest

from bench.client import HttpSchedulerClient, check_fd_limit, parse_reply, request_body
from bench.dataset import Prompt
from bench.records import Sampling
from routebalance.config import ConfigError
from tests.unit.fakes import CONFIG_DIGEST, FLEET_DIGEST, LocalScheduler, reply_body

SAMPLING = Sampling(temperature=0.0, repetition_penalty=1.0, frequency_penalty=1.2, max_tokens=1024)
PROMPT = Prompt(row=4, prompt_id="p4", text="raw prompt", prompt_tokens=31, budget_cost=None)


def _reply(request_id="cell.run.0", **overrides):
    return json.dumps(reply_body(request_id, **overrides)).encode()


def test_request_body_is_the_contract():
    assert json.loads(request_body("cell.run.0", PROMPT, SAMPLING)) == {
        "request_id": "cell.run.0", "prompt": "raw prompt", "prompt_tokens": 31, "max_tokens": 1024,
        "temperature": 0.0, "repetition_penalty": 1.0, "frequency_penalty": 1.2, "budget_cost": None}


def test_a_complete_reply_parses():
    reply, error = parse_reply(200, _reply(), "cell.run.0")
    assert error is None and reply.model == "m-small" and reply.output_tokens == 10
    assert reply.timings_ms == {"queue_ms": 0.5, "decide_ms": 0.25}


@pytest.mark.parametrize("status,body,fragment", [
    (500, b"internal", "HTTP 500: internal"),
    (200, b"not json", "unreadable reply"),
    (200, b"[1, 2]", "no boolean 'success'"),
    (200, json.dumps({"success": 1}).encode(), "no boolean 'success'"),
    (200, json.dumps({"success": False, "error": "no instance"}).encode(), "scheduler error"),
    (200, _reply(output_tokens="10"), "malformed reply"),
    (200, _reply(output_tokens=True), "malformed reply"),
    (200, _reply(ttft_s=-0.1), "malformed reply"),
    (200, _reply(timings_ms={"queue_ms": -1.0}), "malformed reply"),
    (200, _reply(extra_field=1), "malformed reply"),
    (200, _reply(request_id="cell.run.9"), "is for request"),
    (200, json.dumps({k: v for k, v in reply_body("cell.run.0").items() if k != "model"}).encode(),
     "malformed reply"),
    (200, _reply().replace(b'"ok"', b'NaN'), "unreadable reply"),
    (200, _reply().replace(b'"ok"', b'"\xff"'), "unreadable reply"),          # not UTF-8
])
def test_anything_else_is_a_failure(status, body, fragment):
    reply, error = parse_reply(status, body, "cell.run.0")
    assert reply is None and fragment in error


def test_identity_and_completion_over_http():
    async def go(url):
        async with HttpSchedulerClient(url, max_connections=8, timeout_s=10.0) as client:
            ident = await client.identity()
            status, body = await client.complete(request_body("cell.run.0", PROMPT, SAMPLING))
            return ident, parse_reply(status, body, "cell.run.0"), client.connection_waits
    with LocalScheduler() as server:
        ident, (reply, error), waits = asyncio.run(go(server.url))
    assert (ident.config_digest, ident.fleet_digest) == (CONFIG_DIGEST, FLEET_DIGEST)
    assert error is None and reply.request_id == "cell.run.0" and waits == 0


def test_one_pooled_session_reuses_connections():
    async def go(url):
        async with HttpSchedulerClient(url, max_connections=8, timeout_s=10.0) as client:
            for i in range(20):
                await client.complete(request_body(f"cell.run.{i}", PROMPT, SAMPLING))
            return client.connection_waits
    with LocalScheduler() as server:
        waits = asyncio.run(go(server.url))
        assert len(server.peers) == 20 and len(set(server.peers)) == 1 and waits == 0


def test_the_client_uses_vllm_benchs_connector_so_no_request_waits_for_a_connection():
    """As vLLM bench: no connection limit and keep-alive, as vLLM bench's connector; the
    max_connections value only sizes the open-file check, so six concurrent requests never queue."""
    async def go(url):
        async with HttpSchedulerClient(url, max_connections=2, timeout_s=10.0) as client:
            conn = client._session.connector
            settings = (conn.limit, conn.limit_per_host, conn.force_close)
            await asyncio.gather(*(client.complete(request_body(f"cell.run.{i}", PROMPT, SAMPLING))
                                   for i in range(6)))
            return client.connection_waits, settings
    with LocalScheduler(delay_s=0.2) as server:
        waits, settings = asyncio.run(go(server.url))
        assert server.peak == 6
    assert waits == 0 and settings == (0, 0, False)


@pytest.mark.parametrize("identity", [{"config_digest": CONFIG_DIGEST}, {"config_digest": 1, "fleet_digest": "f"}])
def test_an_incomplete_identity_fails(identity):
    async def go(url):
        async with HttpSchedulerClient(url, max_connections=2, timeout_s=10.0) as client:
            await client.identity()
    with LocalScheduler(identities=[identity]) as server:
        with pytest.raises(ConfigError):
            asyncio.run(go(server.url))


def test_the_open_file_limit_must_cover_the_pool(monkeypatch):
    monkeypatch.setattr(resource, "getrlimit", lambda what: (1024, 4096))
    check_fd_limit(512)
    with pytest.raises(ConfigError, match="ulimit"):
        check_fd_limit(1000)
    with pytest.raises(ConfigError):
        HttpSchedulerClient("http://unused", max_connections=4096, timeout_s=1.0)
