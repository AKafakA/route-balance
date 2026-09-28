"""Evidence written while serving, off the event loop, and the event-loop lag probe.

JsonlLog buffers lines in memory; a background task appends them to the file every flush_s on a worker
thread, so a request never waits on the disk. close() writes what is left. A write error ends the task,
and the server then refuses requests: the logs are evidence, not optional.

LagProbe measures how late a 10 ms sleep wakes up on the serving loop, the parity evidence that no system
blocks it (router models, the estimator and the scheduler round all run on the thread pool).
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path


FLUSH_S = 1.0     # the server's decision and latency logs write to disk once a second


class JsonlLog:
    def __init__(self, path: Path, flush_s: float):
        if flush_s <= 0:
            raise ValueError("flush_s must be positive")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._flush_s = flush_s
        self._lines: list[str] = []
        self._writing = 0             # lines handed to the worker thread and not yet on disk
        self.written = 0              # lines on disk

    @property
    def buffered(self) -> int:
        """Lines written by the server and not yet on disk (0: the file holds every line so far)."""
        return len(self._lines) + self._writing

    def write(self, record: dict) -> None:
        self._lines.append(json.dumps(record, separators=(",", ":"), allow_nan=False))

    async def run(self) -> None:
        while True:
            await asyncio.sleep(self._flush_s)
            await self.flush()

    async def flush(self) -> None:
        lines, self._lines = self._lines, []
        if lines:
            self._writing += len(lines)
            # a failed append leaves its lines counted as buffered for good: the file never looks complete
            await asyncio.get_running_loop().run_in_executor(None, self._append, lines)
            self._writing -= len(lines)
            self.written += len(lines)

    def _append(self, lines: list[str]) -> None:
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")


class LagProbe:
    PERIOD_S = 0.010

    def __init__(self):
        self._samples: list[float] = []
        self._since = time.perf_counter()

    async def run(self) -> None:
        while True:
            t0 = time.perf_counter()
            await asyncio.sleep(self.PERIOD_S)
            if t0 >= self._since:          # a sleep begun before the last reset timed the previous configuration's apply
                self._samples.append(max(0.0, (time.perf_counter() - t0 - self.PERIOD_S) * 1000.0))

    def reset(self) -> None:
        self._samples = []
        self._since = time.perf_counter()

    def summary(self) -> dict:
        s = sorted(self._samples)          # every sample since the configuration was applied (reset)
        if not s:
            return {"samples": 0}

        def q(p: float) -> float:        # nearest-rank quantile
            return s[min(len(s) - 1, int(round(p * (len(s) - 1))))]
        return {"samples": len(s), "p50_ms": q(0.5), "p99_ms": q(0.99), "p999_ms": q(0.999), "max_ms": s[-1]}
