"""The scheduler served in-process for the contract tests, through the production launcher (server.http_server)."""
import asyncio

from routebalance.server import build_app, http_server
from routebalance.service import Service

STOP_S = 10.0    # the tests' bound on a graceful stop: a handler that never returns fails the test, never hangs it


class ServedScheduler:
    def __init__(self, service: Service, port: int):
        self.server = service.http = http_server(build_app(service), "127.0.0.1", port)
        self.task = None

    async def start(self) -> "ServedScheduler":
        self.task = asyncio.create_task(self.server.serve())
        while not self.server.started:
            if self.task.done():                 # a failed startup (uvicorn exits): its error, never a silent hang
                self.task.result()
                raise RuntimeError("the scheduler's server stopped before it started")
            await asyncio.sleep(0.01)
        return self

    async def stop(self) -> None:
        self.server.should_exit = True
        try:
            await asyncio.wait_for(asyncio.shield(self.task), STOP_S)
        except asyncio.TimeoutError:
            self.server.force_exit = True        # uvicorn's own way out of a graceful stop
            await self.task
            raise
