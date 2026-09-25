"""Instance telemetry shared by every system.

Load comes from two places, exactly as production balancers get it:
  * the engine: vLLM's Prometheus /metrics (running and waiting requests, KV-cache usage, preemptions),
    scraped by one poller per instance every poll_ms (so a slow or dead instance never delays the others)
    or fetched when a decision is made (the three ways to read, below);
  * the scheduler itself: every dispatch is tracked until it completes. Its predicted output length
    and the tokens streamed so far give the instance's pending decode tokens.

Dead reckoning: every request an engine serves was sent by this scheduler, so the scheduler's own
in-flight count is the instance's load, exact at read time. The engine's gauge only splits it:
running = min(engine running, in_flight), waiting = in_flight - running. The gauges lag (vLLM
publishes them with completed engine steps and they are scraped every poll_ms), so right after a
dispatch, or during a long first prefill, a scrape can still show an idle engine, and right after a
completion it can still show the finished request; the in-flight count covers both without any
timing assumption. RouteBalance and every baseline dispatcher read the same view (the parity rule).
With dead_reckoning off, only the engine's gauges are used.

Three ways to read (config telemetry_read; D53, D55):
  poll        the snapshot the background pollers keep, every poll_ms, as production balancers read a scraped
              cache: the pipeline systems (router + dispatcher, deciding per request), and RouteBalance's D54 control;
  per_round   RouteBalance's own: every instance fetched once per batch, at each round's start (fetch_round), and
              the whole round scheduled from that fetch;
  per_request v3's baselines: each request fetches its candidates (fetch) and reads the engine's own counts.
A fetch uses only scrapes sent after it began; an instance without one fails the round or the request.

Freshness is measured from when a scrape was SENT, so a slow reply cannot pass for fresh data. A
malformed reply makes the instance unavailable until a valid one arrives; a failed fetch leaves the
last snapshot usable only while it is younger than max_age_ms. Nothing falls back to an idle default.
"""
from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Iterable, Sequence

import aiohttp

from routebalance.types import Instance

# vLLM (0.18.1) Prometheus metrics the view reads; each must appear exactly once per engine.
METRICS = {
    "vllm:num_requests_running": "num_running",
    "vllm:num_requests_waiting": "num_waiting",
    "vllm:kv_cache_usage_perc": "kv_cache_utilization",
    "vllm:num_preemptions_total": "num_preemptions",
}
AUTO_EXTEND_TOKENS = 10   # a request reaching its predicted length is extended by this many tokens (v3 fork rule)


class InstanceStateUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class EngineSnapshot:
    """What one scrape of vLLM's /metrics says about an instance."""
    num_running: int
    num_waiting: int
    kv_cache_utilization: float
    num_preemptions: int


def _find_sample(instance_id: str, text: str, name: str) -> float:
    """The value of the one sample of `name` at a line start; C-speed substring search, not a line-by-line scan."""
    hits = []
    pos = 0 if text.startswith(name) else text.find("\n" + name)
    while pos != -1:
        start = pos if pos == 0 and text.startswith(name) else pos + 1
        after = start + len(name)
        if after < len(text) and text[after] in "{ \t":        # exactly this name, not a longer one
            hits.append(start)
        pos = text.find("\n" + name, after)
    if not hits:
        raise InstanceStateUnavailable(f"{instance_id}: /metrics lacks {name}")
    if len(hits) > 1:
        raise InstanceStateUnavailable(f"{instance_id}: {name} appears more than once (one engine per instance)")
    end = text.find("\n", hits[0])
    line = text[hits[0]:] if end == -1 else text[hits[0]:end]
    try:
        value = float(line.rsplit(None, 1)[1])
    except (IndexError, ValueError):
        raise InstanceStateUnavailable(f"{instance_id}: malformed sample {line[:80]!r}") from None
    if not math.isfinite(value) or value < 0:
        raise InstanceStateUnavailable(f"{instance_id}: {name}={value} is not a finite non-negative number")
    return value


def parse_metrics(instance_id: str, text: str) -> EngineSnapshot:
    """Read the needed samples from a Prometheus text exposition; anything missing, repeated or malformed raises."""
    vals = {field: _find_sample(instance_id, text, name) for name, field in METRICS.items()}
    for k in ("num_running", "num_waiting", "num_preemptions"):
        if vals[k] != int(vals[k]):
            raise InstanceStateUnavailable(f"{instance_id}: {k}={vals[k]} is not an integer count")
    if vals["kv_cache_utilization"] > 1.0:
        raise InstanceStateUnavailable(f"{instance_id}: kv_cache_utilization {vals['kv_cache_utilization']} > 1")
    return EngineSnapshot(int(vals["num_running"]), int(vals["num_waiting"]), vals["kv_cache_utilization"],
                          int(vals["num_preemptions"]))


@dataclass(frozen=True)
class InstanceStats:
    """The load an instance presents to a scheduler at decision time."""
    num_running: int              # engine gauge (dead reckoning: at most the in-flight count)
    num_waiting: int              # engine gauge (dead reckoning: the in-flight requests not running)
    kv_cache_utilization: float   # engine gauge, 0..1
    num_preemptions: int          # engine counter (cumulative)
    max_num_seqs: int             # the instance's launch limit (fleet config)
    pending_decode_tokens: int    # scheduler-side: remaining predicted decode tokens of in-flight requests


@dataclass(frozen=True)
class Load:
    stats: InstanceStats
    age_ms: float                 # time since the scrape behind this snapshot was sent
    in_flight: int                # requests dispatched and not yet completed or failed (scheduler-side)
    engine_waiting: int           # the engine's own waiting gauge, whatever dead reckoning says: requests
                                  # the engine has received but cannot run yet (the round-size busy signal)


@dataclass(eq=False)
class DispatchToken:
    """Returned by record_dispatch; pass it to record_progress, then to record_completion or record_failure once."""
    instance_id: str
    predicted_decode_tokens: int
    produced_tokens: int = 0  # default-ok: nothing streamed at dispatch

    def remaining(self) -> int:
        return self.predicted_decode_tokens - self.produced_tokens   # > 0: record_progress extends before it hits 0


@dataclass
class _Entry:
    max_num_seqs: int
    snapshot: EngineSnapshot | None = None
    requested_at: float = -math.inf
    outcome_seq: int = -1  # default-ok: issue sequence of the newest accepted scrape outcome; -1 before any
    error: str | None = None
    open_tokens: dict[int, DispatchToken] = field(default_factory=dict)


Fetch = Callable[[Instance], Awaitable[str]]


class TelemetryView:
    def __init__(self, instances: Sequence[Instance], fetch: Fetch, *, poll_ms: float, max_age_ms: float,
                 dead_reckoning: bool, clock: Callable[[], float] = time.monotonic):
        ids = [i.instance_id for i in instances]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate instance ids")
        if not 0 < 3 * poll_ms <= max_age_ms:
            raise ValueError("need 0 < 3 * poll_ms <= max_age_ms")
        self._instances = list(instances)
        self._fetch = fetch
        self._settings = (poll_ms, max_age_ms, dead_reckoning)    # as given, for the execution manifest
        self._poll_s = poll_ms / 1000.0
        self._max_age_s = max_age_ms / 1000.0
        self._dead_reckoning = dead_reckoning
        self._clock = clock
        self._entries: dict[str, _Entry] = {i.instance_id: _Entry(i.tier.max_num_seqs) for i in instances}
        self._seq = 0                      # scrapes are ordered by issue sequence, never by timestamp
        self._state = "idle"               # idle -> starting -> running -> stopped (no restart)
        self._tasks: list[asyncio.Task] = []

    @property
    def poll_ms(self) -> float:
        return self._settings[0]

    @property
    def max_age_ms(self) -> float:
        return self._settings[1]

    @property
    def dead_reckoning(self) -> bool:
        return self._settings[2]

    @property
    def polling(self) -> bool:
        """The background pollers are running (what executes, for the manifest)."""
        return self._state == "running"

    # ---------------------------------------------------------------- polling

    async def _poll_one(self, inst: Instance) -> None:
        self._seq += 1
        seq, requested = self._seq, self._clock()
        entry = self._entries[inst.instance_id]
        try:
            snapshot = parse_metrics(inst.instance_id, await self._fetch(inst))
        except InstanceStateUnavailable as e:     # the server answered with something malformed: stop using it
            if seq > entry.outcome_seq:
                entry.outcome_seq, entry.snapshot, entry.error = seq, None, str(e)
            return
        except Exception as e:                    # network failure: the last snapshot ages out on its own
            if seq > entry.outcome_seq:
                entry.outcome_seq, entry.error = seq, f"{type(e).__name__}: {e}"
            return
        if seq < entry.outcome_seq:
            return  # an older scrape finishing late must not replace a newer outcome
        entry.outcome_seq, entry.snapshot, entry.requested_at, entry.error = seq, snapshot, requested, None

    async def refresh(self) -> None:
        """Scrape every instance once, concurrently. Not while the pollers run."""
        if self._state == "running":
            raise RuntimeError("refresh() while the pollers are running would double-poll every instance")
        await asyncio.gather(*(self._poll_one(i) for i in self._instances))

    async def fetch(self, instance_ids: Sequence[str]) -> dict[str, Load]:
        """Scrape these instances now, concurrently, and return their loads from scrapes sent after this call
        began: this one's, or a concurrent fetch's newer one. An instance without such a scrape fails the whole
        fetch; an older snapshot is never used, however young. Only without background pollers."""
        if self._state == "running":
            raise RuntimeError("fetch() with background pollers running: a view either polls or fetches")
        by_id = {i.instance_id: i for i in self._instances}
        started = self._clock()
        await asyncio.gather(*(self._poll_one(by_id[i]) for i in instance_ids))
        failed = {i: self._entries[i].error for i in instance_ids
                  if self._entries[i].snapshot is None or self._entries[i].error is not None
                  or self._entries[i].requested_at < started}
        if failed:
            raise InstanceStateUnavailable(f"the load fetch failed for {failed}")
        return self.loads(instance_ids)

    async def fetch_round(self) -> dict[str, Load]:
        """RouteBalance's per-batch load fetch (D53): every instance, at a round's start."""
        return await self.fetch([i.instance_id for i in self._instances])

    async def _run_one(self, inst: Instance) -> None:
        while True:
            started = self._clock()
            await self._poll_one(inst)
            await asyncio.sleep(max(0.0, self._poll_s - (self._clock() - started)))

    async def start(self) -> None:
        if self._state != "idle":
            raise RuntimeError(f"telemetry view cannot start from state {self._state!r}")
        self._state = "starting"
        await asyncio.gather(*(self._poll_one(i) for i in self._instances))
        if self._state != "starting":      # stop() was called during the first scrape
            return
        missing = {i: e.error for i, e in self._entries.items() if e.snapshot is None}
        if missing:
            self._state = "stopped"
            raise InstanceStateUnavailable(f"no first snapshot from {missing}")
        self._tasks = [asyncio.create_task(self._run_one(i), name=f"telemetry-{i.instance_id}")
                       for i in self._instances]
        self._state = "running"

    async def stop(self) -> None:
        self._state = "stopped"
        tasks, self._tasks = self._tasks, []
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    def _check_pollers(self) -> None:
        for t in self._tasks:              # only running pollers are listed; stop() clears the list first
            if t.cancelled():
                raise InstanceStateUnavailable(f"telemetry poller {t.get_name()} was cancelled")
            if t.done():
                raise InstanceStateUnavailable(f"telemetry poller {t.get_name()} died: {t.exception()!r}")

    # ---------------------------------------------------------------- reads and updates

    def loads(self, instance_ids: Iterable[str]) -> dict[str, Load]:
        self._check_pollers()
        now = self._clock()
        out: dict[str, Load] = {}
        for iid in instance_ids:
            entry = self._entries[iid]
            snap = entry.snapshot
            if snap is None:
                raise InstanceStateUnavailable(f"{iid}: no usable telemetry ({entry.error})")
            age = now - entry.requested_at
            if age > self._max_age_s:
                raise InstanceStateUnavailable(f"{iid}: telemetry is {age * 1000:.0f} ms old "
                                               f"(max {self._max_age_s * 1000:.0f} ms; last error: {entry.error})")
            in_flight = len(entry.open_tokens)
            if self._dead_reckoning:
                running = min(snap.num_running, in_flight)
                waiting = in_flight - running
            else:
                running, waiting = snap.num_running, snap.num_waiting
            stats = InstanceStats(running, waiting, snap.kv_cache_utilization, snap.num_preemptions,
                                  entry.max_num_seqs, sum(t.remaining() for t in entry.open_tokens.values()))
            out[iid] = Load(stats=stats, age_ms=age * 1000.0, in_flight=in_flight, engine_waiting=snap.num_waiting)
        return out

    def record_dispatch(self, instance_id: str, predicted_decode_tokens: int) -> DispatchToken:
        if predicted_decode_tokens < 1:
            raise ValueError("predicted_decode_tokens must be >= 1")
        entry = self._entries[instance_id]
        token = DispatchToken(instance_id, predicted_decode_tokens)
        entry.open_tokens[id(token)] = token
        return token

    def record_progress(self, token: DispatchToken, produced_tokens: int) -> None:
        """Called as tokens stream back; keeps the scheduler-side pending decode count exact."""
        if id(token) not in self._entries[token.instance_id].open_tokens:
            raise RuntimeError(f"{token.instance_id}: progress on a closed or unknown token")
        if produced_tokens < token.produced_tokens:
            raise ValueError("produced token count cannot decrease")
        token.produced_tokens = produced_tokens
        if produced_tokens >= token.predicted_decode_tokens:       # ran past the prediction: extend it (fork rule)
            token.predicted_decode_tokens = produced_tokens + AUTO_EXTEND_TOKENS

    def _close(self, token: DispatchToken) -> _Entry:
        entry = self._entries[token.instance_id]
        if entry.open_tokens.pop(id(token), None) is None:
            raise RuntimeError(f"{token.instance_id}: token already closed or never issued")
        return entry

    def record_completion(self, token: DispatchToken) -> None:
        self._close(token)

    def record_failure(self, token: DispatchToken) -> None:
        """The request never reached the engine."""
        self._close(token)


async def fetch_metrics(session: aiohttp.ClientSession, inst: Instance, timeout_s: float) -> str:
    """GET /metrics from one instance. A non-200 status or a body that is not UTF-8 is a malformed reply."""
    url = f"{inst.base_url}/metrics"
    async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout_s)) as resp:
        if resp.status != 200:
            raise InstanceStateUnavailable(f"{inst.instance_id}: HTTP {resp.status} from {url}")
        raw = await resp.read()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as e:
        raise InstanceStateUnavailable(f"{inst.instance_id}: /metrics body is not UTF-8: {e}") from e
