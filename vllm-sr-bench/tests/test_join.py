import json

from bench.arrivals import Segment, Trace
from bench.records import FORMAT, CellRecord, Reply, RequestRecord, Sampling, SchedulerIdentity
from vsrbench.config import Backend, Model
from vsrbench.join import join

MODELS = [Model("m-small", 0.06, 0.06, 4096, (Backend("s0", "192.0.2.1", 8611), Backend("s1", "192.0.2.1", 8612))),
          Model("m-large", 0.4, 0.4, 4096, (Backend("l0", "192.0.2.1", 8613),))]


def record(models):
    reqs = []
    for i, m in enumerate(models):
        reply = None if m is None else Reply(request_id=f"c.r.{i}", model=m, instance_id="", generated_text="x",
                                            prompt_tokens=5, output_tokens=3, ttft_s=0.1, server_latency_s=0.5,
                                            timings_ms={}, first_token_s=0.1)
        reqs.append(RequestRecord(index=i, request_id=f"c.r.{i}", prompt_row=i, prompt_id=str(i), prompt="p",
                                  prompt_tokens=5, budget_cost=None, arrival_s=i, sent_s=i, e2e_s=0.5, reply=reply,
                                  error=None if m else "HTTP 503: x"))
    ident = SchedulerIdentity(config_digest="a" * 64, fleet_digest="b")
    return CellRecord(format=FORMAT, cell_id="c", run_id="r", labels={}, seed=1,
                      trace=Trace(burstiness=1.0, segments=[Segment(n_requests=len(models), rate=1.0)]),
                      arrival_digest="d", dataset_sha256="s", budgets_sha256=None, tokenizer_id="t",
                      prompt_order="per_seed", prompt_order_digest="o",
                      sampling=Sampling(temperature=0, repetition_penalty=1, frequency_penalty=1.2, max_tokens=8),
                      max_connections=10, request_timeout_s=10, identity_start=ident, identity_end=ident,
                      identity_end_error=None, connection_waits=0, started_unix=0, ended_unix=1, duration_s=1,
                      requests=reqs)


def access(rid, upstream, code=200):
    return (f'[2026-09-30T14:56:48.409Z] "POST /v1/chat/completions HTTP/1.1" {code} - 300 20620 1492 5 "-" '
            f'"Python/3.12 aiohttp/3.9" "{rid}" "127.0.0.1:8899" "{upstream}"\n')


def router(rid, model, routing=186):
    return "".join(json.dumps(e) + "\n" for e in (
        {"event": "signal_phase_timing", "request_id": rid, "signal_phase_ms": 185},
        {"event": "decision_phase_timing", "request_id": rid, "decision_ms": 0},
        {"event": "routing_decision", "request_id": rid, "routing_latency_ms": routing, "selected_model": model}))


def test_every_reply_gets_its_replica_and_the_routers_timings(tmp_path):
    (tmp_path / "a.log").write_text(access("c.r.0", "192.0.2.1:8611") + access("c.r.1", "192.0.2.1:8612")
                                    + access("c.r.2", "192.0.2.1:8613"))
    (tmp_path / "r.log").write_text("Starting router...\n" + router("c.r.0", "m-small") + router("c.r.1", "m-small")
                                    + router("c.r.2", "m-large"))
    out, lines, problems, notes = join(record(["m-small", "m-small", "m-large"]), MODELS, tmp_path / "a.log",
                                       tmp_path / "r.log", {}, {"c.r.0": 1790780208.509, "c.r.1": 1790780208.6,
                                                                       "c.r.2": 1790780208.7})
    assert problems == [] and notes == []
    assert [r.reply.instance_id for r in out.requests] == ["s0", "s1", "l0"]
    assert out.requests[0].reply.timings_ms == {"signals": 185.0, "decision": 0.0, "routing": 186.0}
    assert [ln["instance_id"] for ln in lines] == ["s0", "s1", "l0"]
    assert abs(lines[0]["first_token_from_envoy_s"] - 0.1) < 1e-6
    assert abs(out.requests[0].reply.first_token_s - 0.1) < 1e-6          # the SLO's TTFT from Envoy's start
    assert lines[0]["first_token_from_send_s"] == 0.1                      # the client's send-to-first kept


def test_a_reply_on_the_wrong_model_or_without_a_line_is_not_joined(tmp_path):
    (tmp_path / "a.log").write_text(access("c.r.0", "192.0.2.1:8613") + access("c.r.9", "192.0.2.1:8611"))
    (tmp_path / "r.log").write_text(router("c.r.0", "m-small"))
    out, _, problems, notes = join(record(["m-small", "m-small", None]), MODELS, tmp_path / "a.log",
                                   tmp_path / "r.log", {}, {})
    assert [r.reply and r.reply.instance_id for r in out.requests] == ["", "", None]
    assert any("not the run's" in p for p in problems) and any("not joined" in p for p in problems)
    assert notes and "timings" in notes[0]


def test_timings_are_kept_for_every_reply_or_none(tmp_path):
    (tmp_path / "a.log").write_text(access("c.r.0", "192.0.2.1:8611") + access("c.r.1", "192.0.2.1:8612"))
    (tmp_path / "r.log").write_text(router("c.r.0", "m-small"))
    out, lines, problems, notes = join(record(["m-small", "m-small"]), MODELS, tmp_path / "a.log", tmp_path / "r.log",
                                       {}, {"c.r.0": 1790780208.6, "c.r.1": 1790780208.6})
    assert problems == [] and [r.reply.timings_ms for r in out.requests] == [{}, {}]
    assert lines[0]["router_timings_ms"] == {"signals": 185.0, "decision": 0.0, "routing": 186.0} and notes


def test_missing_evidence_is_a_problem_not_a_crash(tmp_path):
    out, _, problems, _ = join(record(["m-small"]), MODELS, tmp_path / "none.log", tmp_path / "none2.log", {}, {})
    assert out.requests[0].reply.instance_id == "" and any("access log is missing" in p for p in problems)


def test_a_reply_without_its_first_token_time_is_not_joined(tmp_path):
    (tmp_path / "a.log").write_text(access("c.r.0", "192.0.2.1:8611"))
    (tmp_path / "r.log").write_text(router("c.r.0", "m-small"))
    out, _, problems, _ = join(record(["m-small"]), MODELS, tmp_path / "a.log", tmp_path / "r.log", {}, {})
    assert any("no first-token time" in p for p in problems)
