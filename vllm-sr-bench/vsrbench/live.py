"""What the running stack executes, read from the stack itself (a check reads the live
system, never the file used to deploy it).

  the router     its management API: /api/v1/status (the active configuration's hash, converged), /api/v1/config
                 (the configuration it serves: models, prices, replicas, quality records, policies) and
                 /api/v1/observability/replays (its replay store, the Balance policy's only stored state: empty when a
                 run starts, exactly the run's requests when it ends)
  Envoy          its admin API: /clusters (the endpoints of every cluster) and /config_dump (timeouts, breakers,
                 load balancing)
  Docker         the containers and the images they run
Both APIs listen only inside the stack's network, so they are read from inside the router container.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable

from routebalance.config import ConfigError
from vsrbench import constants as C
from vsrbench.config import Model, test_tokens_in
from vsrbench.stack import Docker

URLS = {"status": "http://127.0.0.1:8080/api/v1/status", "config": "http://127.0.0.1:8080/api/v1/config",
        "replays": "http://127.0.0.1:8080/api/v1/observability/replays?limit=1",
        "clusters": f"http://vllm-sr-envoy-container:{C.ENVOY_ADMIN_PORT}/clusters?format=json",
        "config_dump": f"http://vllm-sr-envoy-container:{C.ENVOY_ADMIN_PORT}/config_dump"}
_READER = """
import json, sys, urllib.request
urls = json.loads(sys.argv[1])
out = {}
for k, u in urls.items():
    with urllib.request.urlopen(u, timeout=30) as r:
        out[k] = json.loads(r.read())
print(json.dumps(out))
"""


def read(docker: Docker, keys: Iterable[str] = tuple(URLS)) -> dict:
    urls = {k: URLS[k] for k in keys}
    out = docker.run(["exec", "vllm-sr-router-container", "python3", "-c", _READER, json.dumps(urls)])
    return json.loads(out)


def cluster_name(model: str) -> str:
    """The CLI's cluster name for a model: every character but a letter or digit as _<hex> (m-small ->
    model_m_2dsmall_cluster)."""
    return "model_" + "".join(c if c.isascii() and c.isalnum() else f"_{ord(c):02x}" for c in model) + "_cluster"


_ABSENT = object()


def _walk(node, path=()):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _walk(v, path + (k,))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _walk(v, path + (i,))
    else:
        yield path, node


def status_problems(status: dict) -> list[str]:
    cfg = status.get("config") or {}
    conds = {c.get("type"): c.get("status") for c in status.get("conditions", [])}
    out = []
    if not cfg.get("active_hash") or cfg.get("active_hash") != cfg.get("observed_hash"):
        out.append(f"the router's active and observed configuration differ: {cfg}")
    for t in ("Live", "StartupComplete", "ActiveConfigAvailable", "ConfigConverged"):
        if conds.get(t) != "True":
            out.append(f"the router's {t} condition is {conds.get(t)!r}")
    return out


def replay_problems(replays: dict, expected: int) -> list[str]:
    total = replays.get("total") if replays.get("object") == "router_replay.list" else None
    if total != expected:
        return [f"the router's replay store holds {total!r} records, {expected} expected"]
    return []


def replay_total(docker: Docker) -> int:
    return read(docker, ("replays",))["replays"].get("total")


def config_problems(live: dict, models: list[Model], means: dict[str, float], binds: dict, *, test: bool) -> list[str]:
    """The router serves the models, prices, replicas and quality records it was given, one entrypoint on the Balance
    policy, every pool's quality from our index and bound to its models."""
    out = []
    got = {m["name"]: m for m in live.get("providers", {}).get("models", [])}
    if sorted(got) != sorted(m.name for m in models):
        out.append(f"the router serves models {sorted(got)}, the fleet {sorted(m.name for m in models)}")
    for m in models:
        g = got.get(m.name)
        if g is None:
            continue
        p = g.get("pricing", {})
        if (p.get("prompt_per_1m"), p.get("completion_per_1m"), p.get("currency")) != \
                (float(f"{m.prompt_per_1m:.12g}"), float(f"{m.completion_per_1m:.12g}"), "USD"):
            out.append(f"{m.name}: the router prices it {p}")
        ends = [(b.get("name"), b.get("endpoint")) for b in g.get("backend_refs", [])]
        want = [(b.name, f"{b.host}:{b.port}") for b in m.backends]
        if ends != want:
            out.append(f"{m.name}: the router's replicas {ends}, the fleet's {want}")
        if g.get("provider_model_id") != m.name:
            out.append(f"{m.name}: the router sends it as {g.get('provider_model_id')!r}")
    recs = {r["model"]: r["metrics"]["score"] for r in live.get("evaluation", {}).get("records", [])
            if r.get("benchmark") == C.BENCHMARK_ID}
    if recs != {k: float(f"{v:.12g}") for k, v in means.items()}:
        out.append(f"the router's quality records {recs}, the training means {means}")
    if live.get("entrypoints") != [{"model_names": [C.ENTRYPOINT], "recipe": C.RECIPE}]:
        out.append(f"the router's entrypoints are {live.get('entrypoints')}")
    indices = [v for p, v in _walk(live.get("recipes", {})) if p[-2:] == ("quality", "index")]
    if len(indices) != len(C.POOLS) or set(indices) != {C.INDEX_ID}:
        out.append(f"the policies' quality indices are {indices}, {len(C.POOLS)} x {C.INDEX_ID} expected")
    pools = {}
    for recipe in live.get("recipes", []):
        for d in (recipe.get("routing") or {}).get("decisions", []):
            pools[d.get("name")] = sorted(r.get("model") for r in d.get("modelRefs", []))
    want_pools = {name: sorted(x["model"] for x in b) for name, b in binds.items()}
    if pools != want_pools:
        out.append(f"the pools are {pools}, the bindings {want_pools}")
    text = json.dumps(live)
    if not test and test_tokens_in(text):
        out.append(f"the live configuration carries the test profile's {test_tokens_in(text)}")
    return out


def generated_problems(live: dict, generated: dict) -> list[str]:
    """Every setting of the generated configuration (config.yaml: providers, cards, evaluation, the policies with their
    orders and margins, entrypoints, globals) is in the live configuration with the same value, and every list has the
    same length (an extra model, replica or pool shows). Settings the router adds itself (its defaults) are not judged."""
    lv = dict(_walk(live))
    changed = [f"{'.'.join(map(str, p))}: {v!r} -> {lv.get(p, '<absent>')!r}" for p, v in _walk(generated)
               if lv.get(p, _ABSENT) != v]
    lengths = []

    def lists(g, lnode, path=()):
        if isinstance(g, dict) and isinstance(lnode, dict):
            for k, v in g.items():
                lists(v, lnode.get(k), path + (k,))
        elif isinstance(g, list):
            if not isinstance(lnode, list) or len(lnode) != len(g):
                lengths.append(f"{'.'.join(map(str, path))}: {len(g)} entries generated, "
                               f"{len(lnode) if isinstance(lnode, list) else 'none'} live")
            else:
                for i, (x, y) in enumerate(zip(g, lnode, strict=True)):
                    lists(x, y, path + (i,))
    lists(generated, live)
    out = [f"{len(changed)} generated settings differ in the live configuration: {changed[:3]}"] if changed else []
    return out + ([f"lists of another length live: {lengths[:3]}"] if lengths else [])


def card_problems(live: dict, max_model_len: dict[str, int]) -> list[str]:
    """Each model's card gives its context and output limit as the context its engines report (engines_before.json)."""
    cards = {c.get("name"): c for c in live.get("routing", {}).get("modelCards", [])}
    out = []
    if sorted(cards) != sorted(max_model_len):
        out.append(f"the router's model cards are {sorted(cards)}, the engines serve {sorted(max_model_len)}")
    for m, n in max_model_len.items():
        c = cards.get(m, {})
        if (c.get("context_window_size"), c.get("max_output_tokens")) != (n, n):
            out.append(f"{m}: its card says context {c.get('context_window_size')}, output {c.get('max_output_tokens')}; "
                       f"its engines serve {n}")
    return out


def store_problems(state: dict, replays: int, *, at_start: bool) -> list[str]:
    """The stack's stored state: Postgres holds only the replay store, with exactly `replays` records (0 at the start:
    the fresh stack of every run; after it, the run's own requests only); at the start Redis holds no key. After the run
    Redis's keys are the router's own serving state: recorded, never judged."""
    rows = state.get("postgres_rows", {})
    out = []
    if sorted(rows) != [C.REPLAY_TABLE]:
        out.append(f"Postgres holds tables {sorted(rows)}, only {C.REPLAY_TABLE} expected")
    if rows.get(C.REPLAY_TABLE) != replays:
        out.append(f"{C.REPLAY_TABLE} holds {rows.get(C.REPLAY_TABLE)} rows, {replays} expected")
    if at_start and state.get("redis_keys") != 0:
        out.append(f"Redis holds {state.get('redis_keys')} keys, none expected")
    return out


def envoy_problems(clusters: dict, dump: dict, models: list[Model]) -> list[str]:
    """Envoy sends each model's requests to exactly its fleet replicas, by round robin (upstream's default), with the
    authors' timeouts and circuit breakers on every listener, route and cluster."""
    out = []
    live = {c["name"]: sorted((h["address"]["socket_address"]["address"], h["address"]["socket_address"]["port_value"])
                              for h in c.get("host_statuses", []))
            for c in clusters.get("cluster_statuses", [])}
    want = {cluster_name(m.name): m for m in models}
    names = sorted(n for n in live if n.startswith("model_"))
    if names != sorted(want):
        out.append(f"Envoy's model clusters are {names}, the fleet's {sorted(want)}")
    for name, m in want.items():
        exp = sorted((b.host, b.port) for b in m.backends)
        if name in live and live[name] != exp:
            out.append(f"{name}: Envoy's endpoints {live[name]}, the fleet's {exp}")
    if "extproc_service" not in live:
        out.append("Envoy has no extproc_service cluster (the router)")
    msg, route, idle, breakers, lb = [], [], [], {}, {}
    for p, v in _walk(dump):
        if p[-1] == "message_timeout":
            msg.append(v)
        elif p[-2:] == ("route", "timeout"):
            route.append(v)
        elif p[-1] == "stream_idle_timeout":
            idle.append(v)
    for cfg in dump.get("configs", []):
        for sc in cfg.get("static_clusters", []):
            c = sc["cluster"]
            th = [t for t in c.get("circuit_breakers", {}).get("thresholds", [])
                  if t.get("priority", "DEFAULT") == "DEFAULT"]
            breakers[c["name"]] = [(t.get("max_connections"), t.get("max_pending_requests"), t.get("max_requests"))
                                   for t in th]
            lb[c["name"]] = c.get("lb_policy", "ROUND_ROBIN")       # the proto default is omitted from a dump
    if not msg or set(msg) != {f"{C.MESSAGE_TIMEOUT_S}s"}:
        out.append(f"the ext_proc message timeouts are {msg}, {C.MESSAGE_TIMEOUT_S}s expected")
    if not route or set(route) != {f"{C.LISTENER_TIMEOUT_S}s"}:
        out.append(f"the route timeouts are {sorted(set(route))}, {C.LISTENER_TIMEOUT_S}s expected")
    if not idle or set(idle) != {f"{C.LISTENER_TIMEOUT_S}s"}:
        out.append(f"the stream idle timeouts are {sorted(set(idle))}, {C.LISTENER_TIMEOUT_S}s expected")
    b = (C.CIRCUIT_BREAKER,) * 3
    bad = {n: t for n, t in breakers.items() if t != [b]}
    if not breakers or bad:
        out.append(f"circuit breakers other than {C.CIRCUIT_BREAKER}: {bad or 'no cluster read'}")
    notrr = {n: v for n, v in lb.items() if v != "ROUND_ROBIN"}
    if notrr:
        out.append(f"clusters not on round robin: {notrr}")
    if "retry_policy" in json.dumps(dump):
        out.append("a route carries a retry policy (upstream's default has none)")
    return out


def _hub(ref: str) -> str:
    """Docker Hub's implicit prefix removed (docker.io/library/postgres:16-alpine is postgres:16-alpine)."""
    for prefix in ("docker.io/library/", "docker.io/"):
        if ref.startswith(prefix):
            return ref[len(prefix):]
    return ref


def container_problems(containers: list[dict]) -> list[str]:
    out = []
    names = sorted(c["name"] for c in containers)
    if names != sorted(C.CONTAINERS):
        out.append(f"the stack runs containers {names}, upstream's four {sorted(C.CONTAINERS)} expected")
    pinned = {ref: digest for ref, digest in C.IMAGES.values()}
    for c in containers:
        leaked = sorted(set(c.get("env") or {}) & set(C.FORWARDED_ENV))
        if leaked:
            out.append(f"{c['name']} carries host variables {leaked} (the stack runs under a clean environment)")
        res = c.get("resources") or {}
        if (res.get("DeviceRequests") or res.get("Devices") or res.get("Privileged")
                or res.get("Runtime") not in (None, "runc")):
            out.append(f"{c['name']} is given devices or privileges {res} (the stack runs on the CPU)")
        if not c["running"] or c["restart_count"]:
            out.append(f"{c['name']}: running {c['running']}, restarts {c['restart_count']}")
        image = _hub(c["image"])
        digest = pinned.get(image)
        repo = image.rsplit(":", 1)[0]
        if digest is None or f"{repo}@{digest}" not in [_hub(d) for d in c["repo_digests"]]:
            out.append(f"{c['name']} runs {c['image']} ({c['repo_digests']}), not a pinned image")
    return out


@dataclass(frozen=True)
class Expected:
    models: list[Model]
    means: dict[str, float]
    binds: dict
    generated: dict               # config.yaml as generated
    max_model_len: dict[str, int]  # per model, from engines_before.json
    test: bool


def problems(docker: Docker, containers: list[dict], state: dict, exp: Expected, *, replays: int) -> tuple[list[str], dict]:
    """Every live check, with the stores expected to hold `replays` replay records; returns (problems, what was read)."""
    got = read(docker)
    return (container_problems(containers) + status_problems(got["status"]) + replay_problems(got["replays"], replays)
            + store_problems(state, replays, at_start=replays == 0)
            + config_problems(got["config"], exp.models, exp.means, exp.binds, test=exp.test)
            + generated_problems(got["config"], exp.generated) + card_problems(got["config"], exp.max_model_len)
            + envoy_problems(got["clusters"], got["config_dump"], exp.models)), got


def active_hash(docker: Docker) -> str:
    status = read(docker, ("status",))["status"]
    bad = status_problems(status)
    if bad:
        raise ConfigError("; ".join(bad))
    return status["config"]["active_hash"]


def live_endpoints(docker: Docker) -> dict[str, list[tuple[str, int]]]:
    clusters = read(docker, ("clusters",))["clusters"]
    return {c["name"]: sorted((h["address"]["socket_address"]["address"], h["address"]["socket_address"]["port_value"])
                              for h in c.get("host_statuses", []))
            for c in clusters.get("cluster_statuses", []) if c["name"].startswith("model_")}
