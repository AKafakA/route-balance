"""Preflight's checks over recorded replies (no network): the clean recording passes, each seeded violation fails.
The HTTP side (probing a live fleet and scheduler) is tested in tests/contract/test_gates_live.py."""
import json

import pytest

from gates.preflight import Reply, engine_problems, main, scheduler_problems
from routebalance.config import load_fleet
from tests.gates.cellfix import FIXTURES

PRE = FIXTURES / "preflight"


def fleet():
    return load_fleet(PRE / "fleet.json", PRE / "hosts.json")


def expect():
    e = json.loads((PRE / "expect.json").read_text())
    return e["config_digest"], e["fleet_digest"]


def engine_replies(**override) -> dict:
    replies = {"/health": Reply(200, "", None)}
    for path, name in (("/version", "version.json"), ("/v1/models", "models.json"), ("/metrics", "metrics.txt")):
        replies[path] = Reply(200, (PRE / f"engine_{name}").read_text(), None)
    return replies | override


def scheduler_replies(**override) -> dict:
    replies = {"/health": Reply(200, "ok", None)}
    for path, name in (("/v1/config", "config.json"), ("/v1/manifest", "manifest.json"), ("/v1/stats", "stats.json")):
        replies[path] = Reply(200, (PRE / f"scheduler_{name}").read_text(), None)
    return replies | override


def edited(name: str, fn) -> Reply:
    data = json.loads((PRE / name).read_text())
    fn(data)
    return Reply(200, json.dumps(data), None)


def test_the_recorded_replies_pass():
    config, fleet_d = expect()
    assert engine_problems(fleet()[0], engine_replies()) == []
    assert scheduler_problems(scheduler_replies(), fleet(), config, fleet_d) == []


def metrics(**gauges) -> Reply:
    text = (PRE / "engine_metrics.txt").read_text()
    for name, value in gauges.items():
        sample = f'vllm:{name}{{engine="0",model_name="M"}} '
        assert sample + "0.0" in text
        text = text.replace(sample + "0.0", f"{sample}{value}")
    return Reply(200, text, None)


ENGINE_FAIL = [
    ({"/health": Reply(503, "", None)}, "/health is 503"),
    ({"/health": Reply(None, "", "ClientConnectorError: refused")}, "/health is ClientConnectorError"),
    ({"/version": Reply(200, '{"version": "0.18.0"}', None)}, "runs vLLM '0.18.0', the campaign runs 0.18.1"),
    ({"/version": Reply(200, '{"vllm": "0.18.1"}', None)}, "runs vLLM None"),
    ({"/version": Reply(404, "not found", None)}, "/version: HTTP 404"),
    ({"/v1/models": edited("engine_models.json", lambda d: d["data"].append(dict(d["data"][0], id="extra")))},
     "serves ['S', 'extra']"),
    ({"/v1/models": edited("engine_models.json", lambda d: d["data"][0].__setitem__("id", "L"))}, "serves ['L']"),
    ({"/v1/models": edited("engine_models.json", lambda d: d.__setitem__("data", []))}, "serves []"),
    ({"/v1/models": Reply(200, '{"object": "list"}', None)}, "has no list of models"),
    ({"/version": Reply(200, "null", None)}, "/version: the reply is not a JSON object"),
    ({"/metrics": metrics(num_requests_running=1.0)}, "not drained (1 running, 0 waiting)"),
    ({"/metrics": metrics(num_requests_waiting=3.0)}, "not drained (0 running, 3 waiting)"),
    ({"/metrics": Reply(200, "# nothing\n", None)}, "/metrics lacks vllm:num_requests_running"),
    ({"/metrics": Reply(500, "", None)}, "/metrics is 500"),
]


@pytest.mark.parametrize("override,expected", ENGINE_FAIL, ids=[e[:40] for _, e in ENGINE_FAIL])
def test_each_engine_violation_fails(override, expected):
    found = engine_problems(fleet()[0], engine_replies(**override))
    assert any(expected in p for p in found), found


def stats(fn):
    return {"/v1/stats": edited("scheduler_stats.json", fn)}


def manifest(fn):
    return {"/v1/manifest": edited("scheduler_manifest.json", fn)}


def first_age(value):
    def f(s):
        s["telemetry_age_ms"][next(iter(s["telemetry_age_ms"]))] = value
    return f


SCHEDULER_FAIL = [
    ({"/health": Reply(503, "failed", None)}, "/health is 503"),
    ({"/v1/config": edited("scheduler_config.json", lambda d: d.__setitem__("config_digest", "0" * 64))},
     "/v1/config config_digest is"),
    ({"/v1/config": edited("scheduler_config.json", lambda d: d.__setitem__("fleet_digest", "0" * 64))},
     "/v1/config fleet_digest is"),
    (manifest(lambda m: m.__setitem__("config_digest", "0" * 64)), "the manifest is for config"),
    (manifest(lambda m: m["fleet"].reverse()), "executes fleet"),
    (manifest(lambda m: m["code"].__setitem__("routebalance_now", "e" * 64)),
     "its code changed on disk after it started (routebalance: "),
    (manifest(lambda m: m.pop("threadpools")), "scheduler: /v1/manifest"),
    ({"/v1/manifest": Reply(503, '{"error": "a configuration is being applied"}', None)}, "/v1/manifest: HTTP 503"),
    (stats(lambda s: s.__setitem__("failure", "log writer: OSError")), "has failed: log writer"),
    (stats(lambda s: s.__setitem__("active", 2)), "2 requests are in service"),
    (stats(lambda s: s["logged"].__setitem__("decisions_buffered", 4)), "4 decision lines are not yet on disk"),
    (stats(lambda s: s.__setitem__("telemetry_age_ms", {"error": "S_0: telemetry is 900 ms old"})),
     "telemetry is not fresh for every instance"),
    (stats(first_age(2500.0)), "telemetry older than max_age_ms 2000.0"),
    (stats(lambda s: s["telemetry_age_ms"].popitem()), "telemetry is not fresh for every instance"),
    (stats(lambda s: s.__setitem__("telemetry_age_ms", None)), "scheduler: /v1/stats"),     # poll without ages
    ({"/v1/stats": Reply(200, '{"active": 0}', None)}, "scheduler: /v1/stats"),
    ({"/v1/stats": Reply(200, "null", None)}, "/v1/stats: the reply is not a JSON object"),
    ({"/v1/config": Reply(200, "[]", None)}, "/v1/config: the reply is not a JSON object"),
]


@pytest.mark.parametrize("override,expected", SCHEDULER_FAIL, ids=[e[:40] for _, e in SCHEDULER_FAIL])
def test_each_scheduler_violation_fails(override, expected):
    config, fleet_d = expect()
    found = scheduler_problems(scheduler_replies(**override), fleet(), config, fleet_d)
    assert any(expected in p for p in found), found


def test_ages_are_not_checked_against_a_default_when_the_manifest_is_missing():
    config, fleet_d = expect()
    found = scheduler_problems(scheduler_replies(**{"/v1/manifest": Reply(None, "", "timeout")}), fleet(), config,
                               fleet_d)
    assert any("cannot be checked without the manifest" in p for p in found)


def test_the_cli_fails_loudly_on_unreadable_fleet_files(tmp_path, capsys):
    config, fleet_d = expect()
    code = main(["--scheduler", "http://127.0.0.1:9", "--fleet", str(tmp_path / "absent.json"),
                 "--hosts", str(PRE / "hosts.json"), "--expect-config-digest", config, "--expect-fleet-digest", fleet_d])
    assert code == 1 and "FAIL preflight: cannot check" in capsys.readouterr().out


def per_round_manifest(d):
    d["system"].update(mode="routebalance", telemetry_read="per_round")
    d["loaded"]["telemetry_pollers"] = False


def test_a_per_round_system_needs_no_cache_ages():
    """D53: RouteBalance fetches every engine at each round start and fails the round on a miss; preflight does not
    ask it for cache ages (the engines are checked directly)."""
    config, fleet_d = expect()
    per_round = stats(lambda s: s.update(telemetry_read="per_round", telemetry_age_ms=None))
    assert scheduler_problems(scheduler_replies(**per_round, **manifest(per_round_manifest)), fleet(), config,
                              fleet_d) == []


def test_a_per_request_probe_needs_no_cache_ages():
    """D55: v3's probe arm fetches its candidates per request; like the per-round read it keeps no cache."""
    config, fleet_d = expect()
    probe = stats(lambda s: s.update(telemetry_read="per_request", telemetry_age_ms=None))
    probe_manifest = manifest(lambda d: (d["system"].update(telemetry_read="per_request"),
                                         d["loaded"].update(telemetry_pollers=False)))
    assert scheduler_problems(scheduler_replies(**probe, **probe_manifest), fleet(), config, fleet_d) == []


def test_the_age_exemption_needs_the_manifest_to_agree():
    """A pipeline system's stats claiming the per-round read do not skip its cache-age check."""
    config, fleet_d = expect()
    per_round = stats(lambda s: s.update(telemetry_read="per_round", telemetry_age_ms=None))
    found = scheduler_problems(scheduler_replies(**per_round), fleet(), config, fleet_d)
    assert any("reads telemetry 'per_round', the manifest's pipeline system 'poll'" in p for p in found), found


def test_pollers_must_run_exactly_for_the_cache_read():
    """What executes: a per-round manifest whose pollers still run (a leftover view) fails."""
    config, fleet_d = expect()
    leftover = manifest(lambda d: (per_round_manifest(d), d["loaded"].update(telemetry_pollers=True)))
    found = scheduler_problems(scheduler_replies(**leftover), fleet(), config, fleet_d)
    assert any("telemetry pollers run under telemetry_read 'per_round'" in p for p in found), found


def test_a_manifest_read_mode_must_follow_its_mode():
    config, fleet_d = expect()
    bad = manifest(lambda d: d["system"].update(telemetry_read="per_round"))
    found = scheduler_problems(scheduler_replies(**bad), fleet(), config, fleet_d)
    assert any("a pipeline system cannot read telemetry 'per_round'" in p for p in found), found
