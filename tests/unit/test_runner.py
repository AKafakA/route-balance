"""The open-loop runner: sends on schedule, records everything, never retries, refuses a wrong configuration."""
import asyncio
import hashlib
import json

import pytest

import bench.runner as runner
from analysis.validity import integrity_problems
from bench.arrivals import arrival_times, schedule_digest
from tests.traces import constant_trace
from bench.dataset import load_prompts, order_digest
from bench.records import Sampling, read_cell, write_cell
from bench.runner import CellSpec, WrongConfiguration, run_cell
from routebalance.config import ConfigError
from tests.unit.fakes import (CONFIG_DIGEST, IDENTITY, SAMPLING, FakeScheduler, FakeTokenizer, LocalScheduler,
                              identity, scope_dict, write_dataset)


def _spec(n=12, rate=200.0, seed=1, digest=CONFIG_DIGEST):
    return CellSpec(cell_id="cfg__l200__s1", labels={"system": "RouteBalance"}, trace=constant_trace(n, rate, 1.0),
                    seed=seed, sampling=Sampling(**SAMPLING), expected_config_digest=digest)


def _prompts(tmp_path, n=12, seed=1, rows=12):
    return load_prompts(write_dataset(tmp_path, rows), n, "fixed", seed, FakeTokenizer(), "tok-id")


def _run(spec, prompts, fake):
    return asyncio.run(run_cell(spec, prompts, fake, run_id="r1"))


def test_open_loop_sends_on_schedule_whatever_is_in_flight(tmp_path):
    spec = _spec(n=40, rate=200.0)
    fake = FakeScheduler(latency=lambda i: 0.5 if i == 0 else 0.01)
    rec = _run(spec, _prompts(tmp_path, n=40, rows=40), fake)
    lags = [r.sent_s - r.arrival_s for r in rec.requests]
    assert min(lags) >= 0 and max(lags) < 0.05
    first = rec.requests[0]
    assert first.e2e_s >= 0.5
    assert all(r.sent_s < first.sent_s + first.e2e_s for r in rec.requests[1:])   # nothing waited for request 0
    assert rec.duration_s >= first.sent_s + first.e2e_s


def test_the_record_holds_what_ran(tmp_path):
    spec, prompts = _spec(), _prompts(tmp_path)
    rec = _run(spec, prompts, FakeScheduler())
    arrivals = arrival_times(spec.trace, spec.seed)
    assert [r.index for r in rec.requests] == list(range(12))
    assert [r.request_id for r in rec.requests] == [f"cfg__l200__s1.r1.{i}" for i in range(12)]
    assert [r.prompt_row for r in rec.requests] == [p.row for p in prompts.prompts]
    assert [r.prompt for r in rec.requests] == [p.text for p in prompts.prompts]
    assert [r.prompt_tokens for r in rec.requests] == [p.prompt_tokens for p in prompts.prompts]
    assert [r.arrival_s for r in rec.requests] == arrivals
    assert rec.arrival_digest == schedule_digest(arrivals)
    assert rec.prompt_order_digest == order_digest([p.row for p in prompts.prompts])
    assert rec.identity_start == rec.identity_end == IDENTITY and rec.connection_waits == 0
    assert all(r.reply is not None and r.error is None for r in rec.requests)
    assert rec.labels == {"system": "RouteBalance"} and rec.dataset_sha256 == prompts.dataset_sha256
    assert integrity_problems(rec) == []


def test_failures_are_recorded_once_and_never_retried(tmp_path):
    fake = FakeScheduler(fail=lambda i: i == 3, raise_on=lambda i: i == 5)
    rec = _run(_spec(), _prompts(tmp_path), fake)
    assert sorted(i for i, _ in fake.calls) == list(range(12))                  # one call per request
    assert rec.requests[3].reply is None and rec.requests[3].error == "HTTP 500: boom"
    assert rec.requests[5].reply is None and rec.requests[5].error == "RuntimeError: connection reset on 5"
    assert rec.requests[3].e2e_s > 0
    assert sum(r.reply is None for r in rec.requests) == 2


def test_a_wrong_configuration_sends_nothing(tmp_path):
    fake = FakeScheduler()
    with pytest.raises(WrongConfiguration):
        _run(_spec(digest="d" * 64), _prompts(tmp_path), fake)
    assert fake.calls == []


def test_a_configuration_change_during_the_cell_is_recorded(tmp_path):
    rec = _run(_spec(), _prompts(tmp_path), FakeScheduler(identities=(IDENTITY, identity("e" * 64))))
    assert rec.identity_end.config_digest == "e" * 64
    assert "the scheduler configuration or fleet changed during the cell" in integrity_problems(rec)


def test_a_prompt_token_count_that_differs_from_the_backends_invalidates_the_cell(tmp_path):
    rec = _run(_spec(), _prompts(tmp_path), FakeScheduler())
    bad = rec.requests[4].model_copy(update={"reply": rec.requests[4].reply.model_copy(
        update={"prompt_tokens": rec.requests[4].prompt_tokens + 1})})
    rec = rec.model_copy(update={"requests": [*rec.requests[:4], bad, *rec.requests[5:]]})
    assert integrity_problems(rec) == ["1 requests: the backend prefilled a different number of prompt tokens than the "
                                       "client sent (D34)"]


def test_a_scheduler_lost_at_the_end_still_leaves_a_record(tmp_path):
    rec = _run(_spec(), _prompts(tmp_path), FakeScheduler(identities=(IDENTITY, RuntimeError("down"))))
    assert rec.identity_end is None and rec.identity_end_error == "RuntimeError: down"
    assert any("could not be read" in p for p in integrity_problems(rec))


@pytest.mark.parametrize("spec", [_spec(seed=2), _spec(n=11)])
def test_the_prompt_set_must_be_the_cells(tmp_path, spec):
    with pytest.raises(ConfigError):
        _run(spec, _prompts(tmp_path), FakeScheduler())


def test_a_cell_file_round_trips_and_is_never_overwritten(tmp_path):
    rec = _run(_spec(), _prompts(tmp_path), FakeScheduler(fail=lambda i: i == 2))
    out = tmp_path / "cells"
    out.mkdir()
    path = write_cell(rec, out)
    assert path.name == "cfg__l200__s1.cell.json"
    assert read_cell(path) == rec
    with pytest.raises(ConfigError, match="never overwritten"):
        write_cell(rec, out)
    assert [p.name for p in out.iterdir()] == [path.name]                      # no temporary left behind


def test_a_tampered_cell_file_does_not_load(tmp_path):
    rec = _run(_spec(), _prompts(tmp_path), FakeScheduler())
    out = tmp_path / "cells"
    out.mkdir()
    path = write_cell(rec, out)
    data = json.loads(path.read_text())
    data["requests"][0]["e2e_s"] = "0.1"
    path.write_text(json.dumps(data))
    with pytest.raises(ConfigError):
        read_cell(path)


def test_the_command_line_runs_a_scope_cell_end_to_end(tmp_path, monkeypatch):
    dataset = write_dataset(tmp_path)
    scope = tmp_path / "scope.json"
    scope.write_text(json.dumps(scope_dict(dataset, {"rb-a": ("RouteBalance", CONFIG_DIGEST)})))
    monkeypatch.setattr(runner, "load_hf_tokenizer", lambda tokenizer_id: FakeTokenizer())
    out = tmp_path / "cells" / "rb-a__l200__s1"
    out.mkdir(parents=True)
    with LocalScheduler() as server:
        argv = ["--scope", str(scope), "--cell", "rb-a__l200__s1", "--scheduler-url", server.url,
                "--data-root", str(tmp_path), "--out-dir", str(out)]
        assert runner.main(argv) == 0
        rec = read_cell(out / "rb-a__l200__s1.cell.json")
        assert len(server.peers) == 12 and integrity_problems(rec) == []
        with pytest.raises(ConfigError, match="never overwritten"):
            runner.main(argv)
    assert rec.labels["system"] == "RouteBalance" and rec.tokenizer_id == "tok-id"


def test_a_cell_is_written_only_to_its_own_empty_directory(tmp_path, monkeypatch):
    argv = _cli(tmp_path, monkeypatch)
    other = tmp_path / "cells" / "rb-a__l200__s2"
    other.mkdir()
    with LocalScheduler() as server:
        with pytest.raises(ConfigError, match="its own directory"):
            runner.main(argv(server.url)[:-1] + [str(other)])
        (tmp_path / "cells" / "rb-a__l200__s1" / "stray.cell.json").write_text("{}")
        with pytest.raises(ConfigError, match="already holds a cell record"):
            runner.main(argv(server.url))
        assert server.peers == []


def _cli(tmp_path, monkeypatch, rows=None, make_out=True):
    dataset = write_dataset(tmp_path)
    d = scope_dict(dataset, {"rb-a": ("RouteBalance", CONFIG_DIGEST)})
    if rows is not None:
        d["datasets"]["evalset"]["rows"] = rows
    scope = tmp_path / "scope.json"
    scope.write_text(json.dumps(d))
    monkeypatch.setattr(runner, "load_hf_tokenizer", lambda tokenizer_id: FakeTokenizer())
    out = tmp_path / "cells" / "rb-a__l200__s1"
    if make_out:
        out.mkdir(parents=True)
    return lambda url: ["--scope", str(scope), "--cell", "rb-a__l200__s1", "--scheduler-url", url,
                        "--data-root", str(tmp_path), "--out-dir", str(out)]


def test_nothing_is_sent_when_the_output_directory_is_missing(tmp_path, monkeypatch):
    argv = _cli(tmp_path, monkeypatch, make_out=False)
    with LocalScheduler() as server:
        with pytest.raises(ConfigError, match="not a directory"):
            runner.main(argv(server.url))
        assert server.peers == []


def test_nothing_is_sent_when_the_prompt_order_is_not_the_scopes(tmp_path, monkeypatch):
    argv = _cli(tmp_path, monkeypatch, rows=13)                   # the file has 12 rows; the digest differs
    with LocalScheduler() as server:
        with pytest.raises(ConfigError, match="prompt order"):
            runner.main(argv(server.url))
        assert server.peers == []


def test_the_exit_code_reports_a_configuration_change(tmp_path, monkeypatch, capsys):
    argv = _cli(tmp_path, monkeypatch)
    start = {"config_digest": CONFIG_DIGEST, "fleet_digest": "f" * 64}
    with LocalScheduler(identities=[start, {**start, "fleet_digest": "e" * 64}]) as server:
        assert runner.main(argv(server.url)) == 1
    assert "INVALID: the scheduler configuration or fleet changed during the cell" in capsys.readouterr().out


def _budget_table(tmp_path, n_rows=12, fn=lambda i: 1e-5 * (i + 1)):
    path = tmp_path / "budgets.jsonl"
    path.write_text("".join(json.dumps({"id": f"p{i}", "budget_cost": fn(i), "category": "W"}) + "\n"
                            for i in range(n_rows)))
    return path


def test_a_budget_dataset_sends_each_prompts_budget_and_records_it(tmp_path, monkeypatch):
    """W7b: the dataset's budget table gives each request its budget_cost; the record keeps it and the table's
    sha256, and the manifest checks both against the scope."""
    dataset, table = write_dataset(tmp_path), _budget_table(tmp_path)
    doc = scope_dict(dataset, {"rb-a": ("RouteBalance", CONFIG_DIGEST)})
    doc["datasets"]["evalset"]["budgets"] = {"path": table.name,
                                             "sha256": hashlib.sha256(table.read_bytes()).hexdigest()}
    (tmp_path / "scope.json").write_text(json.dumps(doc))
    monkeypatch.setattr(runner, "load_hf_tokenizer", lambda tokenizer_id: FakeTokenizer())
    out = tmp_path / "cells" / "rb-a__l200__s1"
    out.mkdir(parents=True)
    with LocalScheduler() as server:
        assert runner.main(["--scope", str(tmp_path / "scope.json"), "--cell", "rb-a__l200__s1", "--scheduler-url",
                            server.url, "--data-root", str(tmp_path), "--out-dir", str(out)]) == 0
        sent = {b["request_id"].rsplit(".", 1)[1]: b["budget_cost"] for b in server.bodies}
    rec = read_cell(out / "rb-a__l200__s1.cell.json")
    assert rec.budgets_sha256 == doc["datasets"]["evalset"]["budgets"]["sha256"]
    assert all(r.budget_cost == 1e-5 * (int(r.prompt_id[1:]) + 1) == sent[str(r.index)] for r in rec.requests)


@pytest.mark.parametrize("rows,match", [
    (lambda t: t.write_text(t.read_text().replace('"p3"', '"p99"')), "exactly the ids"),
    (lambda t: t.write_text("\n".join(reversed(t.read_text().splitlines())) + "\n"), "in its order"),
    (lambda t: t.write_text(t.read_text().replace('"budget_cost": 4e-05', '"budget_cost": 0')), "positive"),
    (lambda t: t.write_text(t.read_text().replace('"budget_cost": 4e-05', '"budget_cost": true')), "positive"),
])
def test_a_budget_table_that_does_not_fit_the_dataset_is_refused(tmp_path, rows, match):
    from bench.dataset import attach_budgets
    dataset, table = write_dataset(tmp_path), _budget_table(tmp_path)
    rows(table)
    prompts = load_prompts(dataset, 12, "fixed", 1, FakeTokenizer(), "tok-id")
    with pytest.raises(ConfigError, match=match):
        attach_budgets(prompts, table, dataset)


def test_every_request_states_its_budget_null_or_the_prompts():
    from bench.client import request_body
    from bench.dataset import Prompt
    p = Prompt(row=0, prompt_id="p0", text="q", prompt_tokens=3, budget_cost=None)
    s = Sampling(**SAMPLING)
    assert json.loads(request_body("r", p, s))["budget_cost"] is None
    assert json.loads(request_body("r", p.model_copy(update={"budget_cost": 2e-5}), s))["budget_cost"] == 2e-5


def test_the_collector_is_off_while_sending_and_back_on_after(tmp_path):
    """no cyclic garbage collection between the first send and the last reply (a full collection had delayed
    the last send by up to 204 ms); the collector's state is restored afterwards, also when the cell fails."""
    import gc

    class Recording(FakeScheduler):
        async def complete(self, body):
            self.gc_during.append(gc.isenabled())
            return await super().complete(body)

    fake = Recording()
    fake.gc_during = []
    assert gc.isenabled()
    _run(_spec(), _prompts(tmp_path), fake)
    assert fake.gc_during and not any(fake.gc_during) and gc.isenabled()
    failing = Recording(raise_on=lambda i: True)
    failing.gc_during = []
    _run(_spec(), _prompts(tmp_path), failing)
    assert gc.isenabled()


def test_the_last_request_starts_before_the_final_wait(tmp_path, monkeypatch):
    """the last request's task runs before asyncio.gather over every request is built."""
    order = []
    real_gather = runner.asyncio.gather

    def gather(*aws, **kw):
        order.append("gather")
        return real_gather(*aws, **kw)

    class Recording(FakeScheduler):
        async def complete(self, body):
            order.append(json.loads(body)["request_id"].rsplit(".", 1)[1])
            return await super().complete(body)

    monkeypatch.setattr(runner.asyncio, "gather", gather)
    _run(_spec(), _prompts(tmp_path), Recording())
    assert order.index("11") < order.index("gather")
