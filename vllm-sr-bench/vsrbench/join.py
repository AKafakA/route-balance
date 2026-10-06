"""After a run: each reply's instance from Envoy's access log, and the router's own timings from its log, joined by the
request id (the x-request-id header the client sent).

Envoy's access log (the CLI template's format) ends every line with "<x-request-id>" "<authority>" "<upstream host>";
the upstream host (ip:port) names the replica, and so the fleet instance. The router logs one JSON object per event
(its stdout, as `docker logs` returns it); per request it logs signal_phase_timing, decision_phase_timing and
routing_decision, whose timings are its own. Each joined reply's first_token_s (the SLO's TTFT) becomes the time from
Envoy's own request start to the client's first token: it starts at vLLM-SR's front door, as our scheduler's starts
at its handler (a wait before either front door is outside both), and it ends one hop later than ours, at the client
after Envoy forwards the first chunk, where ours ends at the scheduler's receipt of it (Envoy's stock access log has no
first-token time; %START_TIME% is at millisecond resolution). The client's send-to-first value is kept in the evidence,
and per request the stream's end at the client after Envoy's own end (%START_TIME% + %DURATION%), a measure of that
last hop and the client's loop (recorded, never judged).
The router's timings:
  signals   signal_phase_ms      computing the policy's signals (classifiers, embeddings)
  decision  decision_ms          matching the pools' rules
  routing   routing_latency_ms   from the request's arrival at the router to its choice
A reply that cannot be joined keeps instance_id "" (the traffic check then fails the run) and the reason is recorded.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from bench.records import CellRecord
from vsrbench.config import Model

_ACCESS = re.compile(r'^\[(?P<start>[^\]]+)\] "(?P<method>\S+) (?P<path>\S+) (?P<proto>[^"]+)" (?P<code>\d+) '
                     r'(?P<flags>\S+) (?P<rx>\d+) (?P<tx>\d+) (?P<duration>\d+|-) (?P<upstream_ms>\d+|-) '
                     r'"(?P<xff>[^"]*)" "(?P<ua>[^"]*)" "(?P<rid>[^"]*)" "(?P<authority>[^"]*)" "(?P<upstream>[^"]*)"$')
TIMINGS = {"signal_phase_timing": ("signal_phase_ms", "signals"), "decision_phase_timing": ("decision_ms", "decision"),
           "routing_decision": ("routing_latency_ms", "routing")}


def access_lines(path: Path) -> tuple[dict[str, list[dict]], list[str]]:
    by_id: dict[str, list[dict]] = {}
    bad = []
    for line in Path(path).read_text(errors="replace").split("\n"):   # newlines only (U+2028 is not a line end)
        if not line.strip():
            continue
        m = _ACCESS.match(line)
        if m is None:
            bad.append(line[:200])
            continue
        by_id.setdefault(m["rid"], []).append(m.groupdict())    # default-ok: grouping
    return by_id, bad


def router_events(path: Path) -> dict[str, dict[str, dict]]:
    """{request_id: {event name: the event}} for the per-request events (the log's other lines are skipped)."""
    out: dict[str, dict[str, dict]] = {}
    for line in Path(path).read_text(errors="replace").split("\n"):   # newlines only (U+2028 is not a line end)
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        name, rid = ev.get("event") or ev.get("msg"), ev.get("request_id")
        if rid and (name in TIMINGS or name == "llm_usage"):
            out.setdefault(rid, {})[name] = ev                       # default-ok: grouping
    return out


def envoy_start_unix(start: str) -> float:
    """Envoy's %START_TIME% (UTC, milliseconds) as a Unix time."""
    return datetime.strptime(start, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc).timestamp()


def join(record: CellRecord, models: list[Model], access_log: Path, router_log: Path,
         headers: dict[str, dict[str, str]], first_wall: dict[str, float]) -> tuple[CellRecord, list[dict], list[str],
                                                                                   list[str]]:
    """(the record with instance ids and router timings, one evidence line per request, problems, notes). A problem
    means the run cannot show where a request ran or that other traffic went through the stack (the run is invalid);
    a note is missing evidence of the router's own time (the run stands, the overhead is not reported)."""
    problems: list[str] = []
    try:
        by_id, unparsed = access_lines(access_log)
    except OSError as e:           # the evidence could not be copied out: every reply stays unjoined
        by_id, unparsed = {}, []
        problems.append(f"Envoy's access log is missing: {e}")
    try:
        events = router_events(router_log)
    except OSError as e:
        events = {}
        problems.append(f"the router's log is missing: {e}")
    replica = {f"{b.host}:{b.port}": (b.name, m.name) for m in models for b in m.backends}
    ids = {r.request_id for r in record.requests}
    problems += [f"{len(unparsed)} access-log lines unreadable: {unparsed[:2]}"] if unparsed else []
    stray = sorted(set(by_id) - ids)
    if stray:
        problems.append(f"{len(stray)} requests through Envoy that are not the run's: {stray[:3]}")
    requests, evidence, unjoined, untimed = [], [], [], []
    for r in record.requests:
        lines, ev = by_id.get(r.request_id, []), events.get(r.request_id, {})
        line = {"request_id": r.request_id, "headers": headers.get(r.request_id, {}),
                "access": lines, "router": {k: ev.get(k) for k in TIMINGS} | {"llm_usage": ev.get("llm_usage")}}
        if r.reply is None:
            requests.append(r)
            evidence.append(line)
            continue
        why = None
        if len(lines) != 1:
            why = f"{len(lines)} access-log lines"
        elif lines[0]["code"] != "200" or lines[0]["upstream"] not in replica:
            why = f"access-log line {lines[0]['code']} to {lines[0]['upstream']!r}"
        elif replica[lines[0]["upstream"]][1] != r.reply.model:
            why = f"served by {replica[lines[0]['upstream']]}, the reply names {r.reply.model}"
        instance = "" if why else replica[lines[0]["upstream"]][0]
        from_envoy = None
        if not why and r.request_id in first_wall:   # Envoy's own request start to the client's first token (ms clock)
            from_envoy = first_wall[r.request_id] - envoy_start_unix(lines[0]["start"])
            line["first_token_from_envoy_s"] = from_envoy
            if from_envoy < -0.001:          # beyond Envoy's millisecond truncation: the clocks disagree
                unjoined.append(f"{r.request_id}: its first token precedes Envoy's start by {-from_envoy:.3f} s")
                from_envoy = None
            else:
                from_envoy = max(0.0, from_envoy)
        line["first_token_from_send_s"] = r.reply.first_token_s
        if not why and r.request_id in first_wall and lines[0]["duration"] != "-":   # recorded only (the hop at the end)
            client_end = first_wall[r.request_id] + (r.reply.server_latency_s - r.reply.ttft_s)
            envoy_end = envoy_start_unix(lines[0]["start"]) + int(lines[0]["duration"]) / 1000
            line["end_after_envoy_s"] = client_end - envoy_end
        if why:
            unjoined.append(f"{r.request_id}: {why}")
        timings = {}
        for name, (field, key) in TIMINGS.items():
            v = (ev.get(name) or {}).get(field)
            if isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 0:
                timings[key] = float(v)
        line["router_timings_ms"] = dict(timings)
        if len(timings) != len(TIMINGS):
            untimed.append(r.request_id)
        decided = (ev.get("routing_decision") or {}).get("selected_model")
        if decided is not None and decided != r.reply.model:
            unjoined.append(f"{r.request_id}: the router logged {decided}, the reply names {r.reply.model}")
        line["instance_id"] = instance
        evidence.append(line)
        if from_envoy is None and not why:
            unjoined.append(f"{r.request_id}: no first-token time from Envoy's start")
        update = {"instance_id": instance, "timings_ms": timings}
        if from_envoy is not None:
            update["first_token_s"] = from_envoy       # the SLO's TTFT from vLLM-SR's front door, as ours
        requests.append(r.model_copy(update={"reply": r.reply.model_copy(update=update)}))
    if unjoined:
        problems.append(f"{len(unjoined)} replies not joined to one replica of their model: {unjoined[:3]}")
    notes = []
    if untimed:
        # the overhead is reported for every reply or for none (analysis/metrics.py compares the keys across replies);
        # what the router did log stays in the evidence lines
        requests = [r if r.reply is None else r.model_copy(update={"reply": r.reply.model_copy(update={"timings_ms": {}})})
                    for r in requests]
        notes.append(f"{len(untimed)} replies without the router's three timings ({untimed[:3]}): no reply carries timings")
    return record.model_copy(update={"requests": requests}), evidence, problems, notes
