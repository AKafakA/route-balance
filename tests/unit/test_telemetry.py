"""TelemetryView semantics: metrics parsing, fail-loud reads, staleness, dead reckoning, scheduler-side
pending decode, ordering, lifecycle, in-flight counts."""
import asyncio

import pytest

from routebalance.telemetry import AUTO_EXTEND_TOKENS, InstanceStateUnavailable, TelemetryView, parse_metrics
from routebalance.types import Instance, Tier

TIER = Tier("t", "M", "a30", 1e-7, 1e-7, 256, ())
A = Instance("a", TIER, "n0", "h0", 8000)
B = Instance("b", TIER, "n1", "h1", 8000)


def metrics(running=2, waiting=0, kv=0.1, preempt=0, extra=""):
    lab = '{engine="0",model_name="M"}'
    return (f"# TYPE vllm:num_requests_running gauge\nvllm:num_requests_running{lab} {float(running)}\n"
            f"vllm:num_requests_waiting{lab} {float(waiting)}\nvllm:kv_cache_usage_perc{lab} {kv}\n"
            f"vllm:num_preemptions_total{lab} {float(preempt)}\nvllm:num_preemptions_created{lab} 1.7e9\n{extra}")


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def view_with(fetch, instances=(A, B), clock=None, dead_reckoning=True, max_age_ms=500.0):
    return TelemetryView(list(instances), fetch, poll_ms=50.0, max_age_ms=max_age_ms,
                         dead_reckoning=dead_reckoning, clock=clock or Clock())


def static_fetch(bodies):
    async def fetch(inst):
        v = bodies[inst.instance_id]
        if isinstance(v, Exception):
            raise v
        return v
    return fetch


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------- parsing

def test_parse_reads_the_four_gauges_and_ignores_the_rest():
    snap = parse_metrics("a", metrics(running=7, waiting=3, kv=0.42, preempt=5,
                                      extra='vllm:e2e_request_latency_seconds_bucket{le="1.0"} 9\n'))
    assert (snap.num_running, snap.num_waiting, snap.kv_cache_utilization, snap.num_preemptions) == (7, 3, 0.42, 5)


@pytest.mark.parametrize("text,match", [
    (metrics().replace("vllm:kv_cache_usage_perc", "vllm:other"), "lacks"),
    (metrics() + 'vllm:num_requests_running{engine="1"} 1.0\n', "more than once"),
    (metrics(running=1.5), "integer"),
    (metrics(kv=1.5), "> 1"),
    (metrics().replace("0.1\n", "NaN\n", 1), "finite"),
    (metrics().replace(" 2.0\n", " two\n", 1), "malformed"),
    (metrics().replace(" 2.0\n", " 2.0 1700000000000\n", 1), "malformed"),   # a timestamp is never read as the value
    (metrics().replace(" 2.0\n", "\n", 1), "malformed"),
    ("", "lacks"),
    (metrics().replace("vllm:num_requests_waiting", "vllm:num_requests_waiting_by_reason"), "lacks"),
    ("vllm:num_requests_running 1.0\n" + metrics(), "more than once"),        # duplicate on the very first line
])
def test_malformed_metrics_are_unavailable(text, match):
    with pytest.raises(InstanceStateUnavailable, match=match):
        parse_metrics("a", text)


# ---------------------------------------------------------------- reads

def test_snapshot_is_read_back_with_config_limit():
    view = view_with(static_fetch({"a": metrics(), "b": metrics(running=5)}), dead_reckoning=False)   # raw gauges
    run(view.refresh())
    loads = view.loads(["a", "b"])
    assert loads["a"].stats.num_running == 2 and loads["b"].stats.num_running == 5
    assert loads["a"].stats.max_num_seqs == 256 and loads["a"].in_flight == 0
    assert loads["a"].stats.pending_decode_tokens == 0


def test_no_snapshot_raises_with_the_error():
    view = view_with(static_fetch({"a": RuntimeError("connection refused"), "b": metrics()}), dead_reckoning=False)
    run(view.refresh())
    with pytest.raises(InstanceStateUnavailable, match="connection refused"):
        view.loads(["a"])
    assert view.loads(["b"])["b"].stats.num_running == 2


def test_malformed_reply_invalidates_the_cached_snapshot():
    bodies = {"a": metrics(), "b": metrics()}
    view = view_with(static_fetch(bodies))
    run(view.refresh())
    bodies["a"] = "garbage\n"
    run(view.refresh())
    with pytest.raises(InstanceStateUnavailable, match="lacks"):
        view.loads(["a"])


def test_network_failure_keeps_last_snapshot_until_it_is_stale():
    bodies = {"a": metrics(), "b": metrics()}
    clock = Clock()
    view = view_with(static_fetch(bodies), clock=clock, dead_reckoning=False)
    run(view.refresh())
    bodies["a"] = RuntimeError("timeout")
    clock.t += 0.1
    run(view.refresh())
    assert view.loads(["a"])["a"].stats.num_running == 2
    clock.t += 0.45
    with pytest.raises(InstanceStateUnavailable, match="timeout"):
        view.loads(["a"])


def test_age_is_measured_from_when_the_scrape_was_sent():
    clock = Clock()

    async def slow_fetch(inst):
        clock.t += 2.0
        return metrics()

    view = view_with(slow_fetch, instances=(A,), clock=clock, max_age_ms=500.0)
    run(view.refresh())
    with pytest.raises(InstanceStateUnavailable, match="old"):
        view.loads(["a"])


def test_a_fetch_is_bounded_by_the_scrape_timeout_not_by_max_age():
    """D30: max_age_ms bounds a cached snapshot; a fetch's own scrape may take longer (its timeout bounds it)."""
    clock = Clock()

    async def slow_fetch(inst):
        clock.t += 0.6                                         # longer than max_age_ms, within timeout_s
        return metrics(running=2)

    view = view_with(slow_fetch, instances=(A,), clock=clock, max_age_ms=500.0, dead_reckoning=False)
    loads = run(view.fetch(["a"]))
    assert loads["a"].stats.num_running == 2 and loads["a"].age_ms == pytest.approx(600.0)
    with pytest.raises(InstanceStateUnavailable, match="old"):   # the same snapshot, read as a cache: too old
        view.loads(["a"])


# ---------------------------------------------------------------- dispatch bookkeeping

def test_in_flight_requests_the_engine_does_not_show_are_counted_as_waiting():
    bodies = {"a": metrics(running=0, waiting=0), "b": metrics()}
    view = view_with(static_fetch(bodies))
    run(view.refresh())
    view.record_dispatch("a", predicted_decode_tokens=120)
    assert view.loads(["a"])["a"].stats.num_waiting == 1
    for _ in range(3):                          # a long first prefill: the engine keeps publishing 0/0
        run(view.refresh())
        assert view.loads(["a"])["a"].stats.num_waiting == 1
    bodies["a"] = metrics(running=1, waiting=0)  # now the engine shows it running
    run(view.refresh())
    load = view.loads(["a"])["a"]
    assert load.stats.num_waiting == 0 and load.stats.num_running == 1 and load.in_flight == 1


def test_a_stale_gauge_never_hides_a_completion_or_a_dispatch():
    bodies = {"a": metrics(running=2, waiting=1), "b": metrics()}
    view = view_with(static_fetch(bodies))
    run(view.refresh())
    tokens = [view.record_dispatch("a", 10) for _ in range(3)]      # the engine shows all three
    view.record_completion(tokens[0])                              # finished; the gauge still shows 2 + 1
    load = view.loads(["a"])["a"]
    assert (load.stats.num_running, load.stats.num_waiting, load.in_flight) == (2, 0, 2)
    assert load.engine_waiting == 1                                # the raw gauge stays available
    view.record_dispatch("a", 10)                                  # the next dispatch is visible at once
    load = view.loads(["a"])["a"]
    assert (load.stats.num_running, load.stats.num_waiting, load.in_flight) == (2, 1, 3)


def test_gauges_above_the_in_flight_count_are_clamped_to_it():
    view = view_with(static_fetch({"a": metrics(running=4, waiting=7), "b": metrics()}))
    run(view.refresh())
    view.record_dispatch("a", 10)
    stats = view.loads(["a"])["a"].stats
    assert (stats.num_running, stats.num_waiting) == (1, 0)


def test_pending_decode_tracks_streamed_progress_and_auto_extends():
    view = view_with(static_fetch({"a": metrics(), "b": metrics()}))
    run(view.refresh())
    t1 = view.record_dispatch("a", 100)
    t2 = view.record_dispatch("a", 10)
    view.record_progress(t1, 30)
    view.record_progress(t2, 10)              # reached its prediction: extended to 20
    assert view.loads(["a"])["a"].stats.pending_decode_tokens == 70 + AUTO_EXTEND_TOKENS
    view.record_progress(t2, 15)              # the extension counts down
    assert view.loads(["a"])["a"].stats.pending_decode_tokens == 70 + 5
    view.record_progress(t2, 20)              # reached the extension: extended again
    assert view.loads(["a"])["a"].stats.pending_decode_tokens == 70 + AUTO_EXTEND_TOKENS
    view.record_completion(t2)
    assert view.loads(["a"])["a"].stats.pending_decode_tokens == 70


def test_progress_cannot_go_backwards_or_follow_closure():
    view = view_with(static_fetch({"a": metrics(), "b": metrics()}))
    run(view.refresh())
    t = view.record_dispatch("a", 50)
    view.record_progress(t, 5)
    with pytest.raises(ValueError):
        view.record_progress(t, 4)
    view.record_completion(t)
    with pytest.raises(RuntimeError, match="closed"):
        view.record_progress(t, 6)


def test_failed_dispatch_retracts_its_delta():
    view = view_with(static_fetch({"a": metrics(), "b": metrics()}))
    run(view.refresh())
    token = view.record_dispatch("a", 120)
    view.record_failure(token)
    load = view.loads(["a"])["a"]
    assert load.stats.num_waiting == 0 and load.in_flight == 0 and load.stats.pending_decode_tokens == 0


def test_dead_reckoning_off_uses_engine_gauges_only_but_keeps_scheduler_counts():
    view = view_with(static_fetch({"a": metrics(running=0), "b": metrics()}), dead_reckoning=False)
    run(view.refresh())
    token = view.record_dispatch("a", 120)
    load = view.loads(["a"])["a"]
    assert load.stats.num_waiting == 0 and load.in_flight == 1 and load.stats.pending_decode_tokens == 120
    view.record_completion(token)
    assert view.loads(["a"])["a"].in_flight == 0


def test_tokens_close_exactly_once_and_need_a_prediction():
    view = view_with(static_fetch({"a": metrics(), "b": metrics()}))
    run(view.refresh())
    token = view.record_dispatch("a", 1)
    view.record_completion(token)
    with pytest.raises(RuntimeError, match="already closed"):
        view.record_completion(token)
    with pytest.raises(RuntimeError, match="already closed"):
        view.record_failure(token)
    with pytest.raises(ValueError):
        view.record_dispatch("a", 0)


# ---------------------------------------------------------------- ordering

def _race(first_result, second_result):
    started, gate, calls = asyncio.Event(), asyncio.Event(), []

    async def fetch(inst):
        calls.append(1)
        result = first_result if len(calls) == 1 else second_result
        if len(calls) == 1:
            started.set()
            await gate.wait()
        if isinstance(result, Exception):
            raise result
        return result

    view = view_with(fetch, instances=(A,), dead_reckoning=False)   # the clock never moves: one timestamp

    async def scenario():
        old = asyncio.create_task(view.refresh())    # issued first, answers last
        await started.wait()
        await view.refresh()                          # issued second, answers first
        gate.set()
        await old
        return view
    return run(scenario())


def test_older_success_does_not_replace_newer_snapshot():
    assert _race(metrics(running=1), metrics(running=9)).loads(["a"])["a"].stats.num_running == 9


def test_older_failure_does_not_mask_a_newer_snapshot():
    assert _race(RuntimeError("late timeout"), metrics(running=9)).loads(["a"])["a"].stats.num_running == 9


def test_older_success_does_not_undo_a_newer_malformed_reply():
    view = _race(metrics(running=1), "garbage\n")
    with pytest.raises(InstanceStateUnavailable):
        view.loads(["a"])


# ---------------------------------------------------------------- lifecycle and isolation

def test_one_slow_instance_does_not_starve_the_others():
    async def scenario():
        async def fetch(inst):
            if inst.instance_id == "b" and started_b.is_set():
                await asyncio.sleep(10)
            if inst.instance_id == "b":
                started_b.set()
            return metrics()

        started_b = asyncio.Event()
        view = TelemetryView([A, B], fetch, poll_ms=10.0, max_age_ms=100.0,
                             dead_reckoning=True)
        await view.start()
        await asyncio.sleep(0.3)
        try:
            a_ok = view.loads(["a"])["a"].age_ms < 100.0
            with pytest.raises(InstanceStateUnavailable):
                view.loads(["b"])
        finally:
            await view.stop()
        return a_ok

    assert run(scenario()) is True


def test_start_requires_a_first_snapshot_from_every_instance():
    view = view_with(static_fetch({"a": metrics(), "b": RuntimeError("down")}))
    with pytest.raises(InstanceStateUnavailable, match="no first snapshot"):
        run(view.start())


def test_stop_during_start_leaves_no_poller():
    async def scenario():
        gate = asyncio.Event()
        reads = []

        async def fetch(inst):
            reads.append(1)
            await gate.wait()
            return metrics()

        view = TelemetryView([A], fetch, poll_ms=10.0, max_age_ms=100.0, dead_reckoning=True)
        starting = asyncio.create_task(view.start())
        await asyncio.sleep(0.01)
        await view.stop()
        gate.set()
        await starting
        n = len(reads)
        await asyncio.sleep(0.05)
        return n, len(reads), view._tasks

    before, after, tasks = run(scenario())
    assert before == after and tasks == []


def test_second_start_and_refresh_while_running_are_refused():
    async def scenario():
        view = view_with(static_fetch({"a": metrics(), "b": metrics()}))
        await view.start()
        try:
            with pytest.raises(RuntimeError, match="cannot start"):
                await view.start()
            with pytest.raises(RuntimeError, match="double-poll"):
                await view.refresh()
        finally:
            await view.stop()
    run(scenario())


class _Boom(BaseException):
    """Escapes `except Exception` like a real bug would, without the loop-level special cases of SystemExit."""


def test_dead_or_cancelled_poller_is_reported_on_read():
    async def scenario():
        calls = []

        async def fetch(inst):
            calls.append(1)
            if len(calls) > 1:
                raise _Boom("boom")
            return metrics()

        view = TelemetryView([A], fetch, poll_ms=1.0, max_age_ms=10000.0, dead_reckoning=True)
        await view.start()
        await asyncio.sleep(0.05)
        try:
            with pytest.raises(InstanceStateUnavailable, match="died"):
                view.loads(["a"])
        finally:
            await view.stop()

        view2 = view_with(static_fetch({"a": metrics(), "b": metrics()}), max_age_ms=10000.0)
        await view2.start()
        view2._tasks[0].cancel()
        await asyncio.sleep(0)
        try:
            with pytest.raises(InstanceStateUnavailable, match="cancelled"):
                view2.loads(["a"])
        finally:
            await view2.stop()
    run(scenario())


@pytest.mark.parametrize("poll,age", [(20.0, 20.0), (50.0, 100.0), (0.0, 100.0)])
def test_view_rejects_configs_that_break_freshness(poll, age):
    with pytest.raises(ValueError):
        TelemetryView([A], static_fetch({"a": metrics()}), poll_ms=poll, max_age_ms=age, dead_reckoning=True)


def test_parse_accepts_a_sample_on_the_first_line_and_without_labels():
    text = "vllm:num_requests_running 4\n" + "\n".join(metrics().splitlines()[2:]) + "\n"
    assert parse_metrics("a", text).num_running == 4


def test_fetch_round_returns_this_scrapes_loads_only():
    bodies = {"a": metrics(running=3), "b": metrics(running=1)}
    view = view_with(static_fetch(bodies), dead_reckoning=False)
    loads = run(view.fetch_round())
    assert loads["a"].stats.num_running == 3 and loads["b"].stats.num_running == 1
    bodies["a"] = RuntimeError("connection refused")          # the previous snapshot is still young ...
    with pytest.raises(InstanceStateUnavailable, match="load fetch failed.*connection refused"):
        run(view.fetch_round())                               # ... but a round never uses it (D53)


def test_fetch_scrapes_only_the_instances_asked_for():
    """D55's per-request probe: a request fetches its candidates, not the fleet."""
    scraped = []

    async def fetch(inst):
        scraped.append(inst.instance_id)
        return metrics(running=4, waiting=2)
    view = view_with(fetch, dead_reckoning=False)
    loads = run(view.fetch(["b"]))
    assert scraped == ["b"] and list(loads) == ["b"]
    assert (loads["b"].stats.num_running, loads["b"].stats.num_waiting) == (4, 2)   # the engine's own counts


def test_a_concurrent_fetchs_newer_scrape_counts_but_an_older_one_does_not():
    """Two requests probing one instance at once: the later scrape's outcome serves both (it was sent after
    both fetches began); a snapshot from before a fetch began never does."""
    clock = Clock()
    slow = asyncio.Event()
    replies = ["slow failure", metrics(running=5), RuntimeError("refused")]

    async def fetch(inst):
        reply = replies.pop(0)
        if reply == "slow failure":      # the first fetch's scrape answers last, and fails
            await slow.wait()
            raise RuntimeError("reset")
        if isinstance(reply, Exception):
            raise reply
        return reply

    async def scenario():
        view = view_with(fetch, instances=(A,), clock=clock, dead_reckoning=False)
        first = asyncio.create_task(view.fetch(["a"]))
        await asyncio.sleep(0)
        clock.t += 0.01
        second = await view.fetch(["a"])            # the newer scrape succeeds
        slow.set()
        first_loads = await first                    # its own scrape failed late; the newer one was sent after it began
        assert second["a"].stats.num_running == 5 and first_loads["a"].stats.num_running == 5
        clock.t += 0.01
        with pytest.raises(InstanceStateUnavailable, match="refused"):
            await view.fetch(["a"])                  # the young snapshot from before this fetch is not used
    run(scenario())


def test_fetch_round_refuses_a_polling_view():
    view = view_with(static_fetch({"a": metrics(), "b": metrics()}))

    async def scenario():
        await view.start()
        try:
            with pytest.raises(RuntimeError, match="either polls or fetches"):
                await view.fetch_round()
        finally:
            await view.stop()
    run(scenario())
