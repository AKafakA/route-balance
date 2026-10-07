"""The scheduler: one HTTP server for RouteBalance and every baseline.

  POST /v1/completions   one request; the contract is documented in bench/client.py
  GET  /v1/config        the live configuration with its config_digest and the fleet_digest
  POST /v1/config        replace the configuration whole: parsed strictly, built off the event loop, and
                         swapped in only while no request is being served (409 otherwise)
  GET  /v1/stats         parity evidence: event-loop lag, telemetry age, rounds, booster calls, logged lines
  GET  /v1/manifest      the execution manifest: everything this process executes for the live configuration
                         (code, library versions, launch parameters, pools, telemetry, loaded components and
                         artifact md5s); gates/parity.py diffs it across the systems of a comparison group
  GET  /health

python -m routebalance.server starts it (scripts/node_ctl.sh): the service (service.py) behind uvicorn, as v3 and
vLLM serve. Its evidence goes under --log-dir (service.py describes it).
"""
from __future__ import annotations

import argparse
import asyncio
import email.message
import logging
import math
import signal
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from pydantic import ValidationError

from routebalance.backend import BackendUnreachable
from routebalance.config import ConfigError, config_digest, load_fleet, load_scheduler, loads_strict, parse_scheduler
from routebalance.execution import set_ulimit
from routebalance.request import Busy, Completion, ServeError
from routebalance.service import Service

log = logging.getLogger("routebalance.server")     # its name when run as python -m routebalance.server too


# ---------------------------------------------------------------- HTTP
# v3's scheduler served FastAPI through uvicorn (its run_server -> serve_http, a
# copy of vLLM's launcher), as vLLM's own server does: uvicorn.Config with the host, the port and v3's
# keep-alive; every other setting is uvicorn's stock (a 2048 listen backlog, httptools, the access log).
TIMEOUT_KEEP_ALIVE_S = 600    # v3's (its TIMEOUT_KEEP_ALIVE: "Was 5 -- caused localhost RST race vs bench keepalive=60s")


def http_server(app: FastAPI, host: str, port: int) -> uvicorn.Server:
    return uvicorn.Server(uvicorn.Config(app, host=host, port=port, timeout_keep_alive=TIMEOUT_KEEP_ALIVE_S))


async def serve_http(server: uvicorn.Server) -> None:
    """v3's launcher (its serve_http, a copy of vLLM's): SIGINT and SIGTERM cancel the serving
    task, and the server then shuts down gracefully, so the caller's cleanup runs (v3's debug-only report of the
    process holding the port is left out)."""
    loop = asyncio.get_running_loop()
    task = loop.create_task(server.serve())
    loop.add_signal_handler(signal.SIGINT, task.cancel)
    loop.add_signal_handler(signal.SIGTERM, task.cancel)
    try:
        await task
    except asyncio.CancelledError:
        log.info("Shutting down FastAPI HTTP server.")
        await server.shutdown()


async def _body(request: Request) -> str:
    """The body in its declared charset, UTF-8 when none (or an empty one) is declared, as aiohttp's request.text()
    read it before. A body that does not decode is a ValueError: a bad request; an unknown charset
    name raises LookupError: a server error, as before."""
    header, content_type = email.message.Message(), request.headers.get("content-type")
    if content_type is not None:
        header["content-type"] = content_type
    charset = header.get_content_charset()           # None: no charset; "": an empty one
    return (await request.body()).decode("utf-8" if not charset else charset)


def build_app(service: Service) -> FastAPI:
    app = FastAPI()

    @app.post("/v1/completions")
    async def completions(request: Request) -> Response:
        try:
            c = Completion.model_validate(loads_strict(await _body(request)))
        except (ValidationError, ConfigError, ValueError) as e:
            return JSONResponse({"success": False, "error": f"bad request: {e}"}, status_code=400)
        try:
            return JSONResponse(await service.complete(c))
        except Busy as e:
            return JSONResponse({"success": False, "error": str(e)}, status_code=503)
        except (ServeError, BackendUnreachable) as e:
            return JSONResponse({"success": False, "error": str(e)}, status_code=502)
        except Exception as e:         # the scheduler failed this request: loud, never a fabricated reply
            log.exception("request %s failed", c.request_id)
            return JSONResponse({"success": False, "error": f"{type(e).__name__}: {e}"}, status_code=500)

    @app.get("/v1/config")
    async def get_config() -> Response:
        cfg = service.cfg
        return JSONResponse({"config": None if cfg is None else cfg.model_dump(mode="json"),
                             "config_digest": None if cfg is None else config_digest(cfg),
                             "fleet_digest": service.fleet_digest})

    @app.post("/v1/config")
    async def post_config(request: Request) -> Response:
        try:
            cfg = parse_scheduler(loads_strict(await _body(request)))
        except (ConfigError, ValueError) as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        try:
            await service.apply(cfg)
        except Busy as e:
            return JSONResponse({"error": str(e)}, status_code=409)
        except Exception as e:         # nothing was swapped: the old configuration is still live
            return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=400)
        return await get_config()

    @app.get("/v1/stats")
    async def stats() -> Response:
        return JSONResponse(service.stats())

    @app.get("/v1/manifest")
    async def manifest() -> Response:
        try:
            return JSONResponse(await service.manifest())
        except Busy as e:
            return JSONResponse({"error": str(e)}, status_code=503)

    @app.get("/health")
    async def health() -> Response:
        ok = service.failure is None and service.cfg is not None
        return PlainTextResponse("ok" if ok else "failed", status_code=200 if ok else 503)
    return app


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--fleet", type=Path, required=True)
    p.add_argument("--hosts", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True, help="the scheduler configuration served at start")
    p.add_argument("--artifacts", type=Path, required=True)
    p.add_argument("--avengers-repo", type=Path, default=None, help="upstream AvengersPro checkout (its router)")
    p.add_argument("--best-route-repo", type=Path, default=None, help="upstream best-route-llm checkout (its ranker)")
    p.add_argument("--host", required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--request-timeout-s", type=float, required=True)
    p.add_argument("--max-connections", type=int, required=True, help="engine connection bound, above peak in flight")
    p.add_argument("--torch-threads", type=int, default=None,
                   help="intra-op threads for every model call; unset: torch's default (D134, as v3)")
    p.add_argument("--log-dir", type=Path, required=True)
    a = p.parse_args(argv)
    if not (a.request_timeout_s > 0 and math.isfinite(a.request_timeout_s)) or a.max_connections < 1 \
            or (a.torch_threads is not None and a.torch_threads < 1):
        p.error("timeouts, connection bound and thread count must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    set_ulimit()                      # v3's, before anything opens a file
    if a.torch_threads is not None:   # unset: torch's default, as v3; the manifest records the effective count
        import torch
        torch.set_num_threads(a.torch_threads)
    import uvloop
    uvloop.install()
    fleet, cfg = load_fleet(a.fleet, a.hosts), load_scheduler(a.config)

    async def serve():
        service = Service(fleet, artifact_root=a.artifacts, avengers_repo=a.avengers_repo,
                          request_timeout_s=a.request_timeout_s, max_connections=a.max_connections,
                          torch_threads=a.torch_threads, log_dir=a.log_dir, best_route_repo=a.best_route_repo)
        await service.start(cfg)
        server = service.http = http_server(build_app(service), a.host, a.port)
        log.info("starting uvicorn for %s on %s:%d (config %s, fleet %s)", cfg.mode, a.host, a.port,
                 config_digest(cfg), service.fleet_digest)
        try:
            await serve_http(server)
        finally:
            await service.stop()
    asyncio.run(serve())      # on uvloop, installed above (v3: uvloop's event-loop policy, then asyncio.run)


if __name__ == "__main__":
    main()
