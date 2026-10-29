"""One vLLM-SR run, end to end, fail-loud, in the main grid's cell layout.

The run takes its trace, seed, prompts, order, sampling and bench parameters from a main-grid cell of the same
render (the template: the round-robin dispatcher's cell at the same rate and seed), so it sends exactly what every
system of that rate and seed is sent. Order (scripts/run_cell.sh's, with the stack in place of our scheduler):
  0. our scheduler is not running (its port is closed, no process of it) and no previous stack is left
  1. the configuration: upstream's generator from the fleet and the training means, the index edit, upstream's validator
  2. drain the engines; a fresh stack (vllm-sr serve); the engines as they report themselves; the live checks (live.py),
     the stores empty: a run never starts on a stack that does not execute the configuration, or that holds state from
     before
  3. the warm-up with the proven prefix-cache reset, then drain
  4. the bench runner's open loop through Envoy (bench.runner.run_cell with vsrbench.client.VsrClient); the raw record is
     written at once
  5. the engines again; Envoy's access log flushed; the replay store holds the run's requests; the live checks again;
     the evidence copied out; the stack torn down. A step that fails here is a problem of the run, never its loss: a run
     that served is always published, valid or not (as scripts/run_cell.sh keeps such cells)
  6. the join (instances, router timings), the record, the checks of the main grid (analysis.manifest.cell_problems,
     gates.cell reset and traffic, the engines unchanged and launched as the paired main-grid run's), the verdict
     (gates/verdict.json); publish; mirror
Exit: 0 valid; 1 ran and invalid (published); 2 could not run (the attempt stays under _attempts); 3 the mirror failed.
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import json
import os
import resource
import socket
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

from analysis.manifest import cell_problems
from bench.arrivals import Segment, Trace
from bench.dataset import load_hf_tokenizer, load_prompts
from bench.records import Sampling, SchedulerIdentity, cell_dir, write_cell
from bench.runner import CellSpec, _check_before_sending, run_cell
from bench.scope import ConfigEntry, ExpectedCell, expected_cells
from gates.cell import GateError, engine_changes, open_cell, reset_problems, traffic_problems
from gates.preflight import ENGINE_PATHS, engine_problems, fetch
from routebalance.backend import BackendClient
from routebalance.config import ConfigError, fleet_digest, load_fleet
from tools import cell_driver, direct, engines
from vsrbench import constants as C
from vsrbench import config as vc
from vsrbench import join, live
from vsrbench.client import VsrClient
from vsrbench.quality import training_means
from vsrbench.stack import Docker, Stack, tree_digest

MODULE = Path(__file__).resolve().parents[1]
REPO = MODULE.parent
SCHEDULER_PORT = 8200          # our scheduler's port on the cluster (scripts/run_scope.sh --scheduler)
LEASE_ROOT = "/mydata"         # the cluster nodes' data disk: a test run never writes there
SHARED = ("scripts/mirror.sh",)   # besides every shared Python module loaded (harness_record reads sys.modules)
SHARED_PACKAGES = ("analysis", "bench", "gates", "routebalance", "tools")
# the environment scripts/run_cell.sh gives the bench runner (env -i), and the interpreter's module path
CLIENT_ENV = ("HOME", "PATH", "LANG", "HF_HOME", "HF_HUB_OFFLINE", "PYTHONPATH")


def vsr_cell(template: ExpectedCell, family: str, config_sha256: str, test: bool) -> ExpectedCell:
    config_id = ("test." if test else "") + f"{family}.vsr.mom-balance"
    return dataclasses.replace(
        template, cell_id=f"{config_id}__{template.trace_id}__s{template.seed}", block=f"vLLM-SR-{template.trace_id}",
        config_id=config_id,
        config=ConfigEntry(system="vLLM-SR", knob="mom-balance", dispatch="rr", config_digest=config_sha256))


def port_open(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(1.0)
        return s.connect_ex(("127.0.0.1", port)) == 0


def scheduler_processes() -> list[int]:
    """Our scheduler's processes on this node, from /proc (no pattern can match this process itself)."""
    out = []
    for d in Path("/proc").iterdir():
        if d.name.isdigit() and int(d.name) != os.getpid():
            try:
                cmd = (d / "cmdline").read_bytes().replace(b"\0", b" ")
            except OSError:
                continue
            if b"routebalance.server" in cmd or b"routebalance/server.py" in cmd:
                out.append(int(d.name))
    return out


def _write(path: Path, data) -> None:
    if path.exists():
        raise ConfigError(f"{path} exists")
    path.write_text(json.dumps(data, indent=1, sort_keys=True, default=str) + "\n")


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def versions(vsr: Path, docker) -> dict:
    """The versions a run executes with, for the block's consistency check: this interpreter, the vllm-sr CLI and its
    venv's packages, Docker's engine and its data root, and the images Docker holds (their repository digests)."""
    cli_python = Path(vsr).parent / "python"
    freeze = subprocess.run([str(cli_python), "-m", "pip", "freeze", "--all"], capture_output=True, text=True).stdout
    return {"python": sys.version.split()[0],
            "vllm_sr": subprocess.run([str(vsr), "--version"], capture_output=True, text=True).stdout.strip(),
            "cli_packages_sha256": hashlib.sha256(freeze.encode()).hexdigest(),
            "docker": docker.run(["version", "--format", "{{.Server.Version}}"]).decode().strip(),
            "docker_root": docker.run(["info", "--format", "{{.DockerRootDir}}"]).decode().strip(),
            "docker_storage": docker.run(["info", "--format", "{{.Driver}} {{json .DriverStatus}}"]).decode().strip(),
            "images": {k: sorted(json.loads(docker.run(["image", "inspect", "--format", "{{json .RepoDigests}}", ref])))
                       for k, (ref, _) in C.IMAGES.items()}}


def shared_tree_check() -> None:
    """The shared packages (bench, gates, tools, ...) load from this tree: harness.json records this tree's code, and the
    checks that judge a run must be the code it records (run_cell and vsrbench.block both call it)."""
    elsewhere = sorted({m.__name__.split(".")[0] for m in list(sys.modules.values())
                        if m.__name__.split(".")[0] in SHARED_PACKAGES and getattr(m, "__file__", None)
                        and not Path(m.__file__).resolve().is_relative_to(REPO)})
    if elsewhere:
        raise ConfigError(f"{elsewhere} load from outside {REPO}: set PYTHONPATH to this tree")


def harness_record() -> dict:
    """The code that ran: this module's files and the shared ones it calls, by sha256; the tree's commit if it has one;
    the Hub environment the tokenizer loaded from."""
    git = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"], capture_output=True, text=True)
    loaded = sorted({Path(m.__file__).resolve() for m in list(sys.modules.values())
                     if getattr(m, "__file__", None) and Path(m.__file__).resolve().is_relative_to(REPO)
                     and not Path(m.__file__).resolve().is_relative_to(MODULE)})
    return {"module": {str(p.relative_to(MODULE)): _sha(p) for p in sorted(MODULE.glob("vsrbench/*.py"))},
            "shared": {**{str(p.relative_to(REPO)): _sha(p) for p in loaded}, **{f: _sha(REPO / f) for f in SHARED}},
            "git_head": git.stdout.strip() if git.returncode == 0 else None,
            "git_dirty": dirty.stdout.splitlines() if dirty.returncode == 0 else None,
            "hf": {k: os.environ.get(k) for k in ("HF_HOME", "HF_HUB_OFFLINE")},
            "open_files": resource.getrlimit(resource.RLIMIT_NOFILE),
            "node": {"conntrack_loaded": Path("/proc/net/nf_conntrack").exists()
                     or Path("/proc/sys/net/netfilter/nf_conntrack_count").exists(),
                     "dockerd": subprocess.run(["systemctl", "is-active", "docker"], capture_output=True,
                                               text=True).stdout.strip()}}


class Identity:
    """The live stack's identity for the runner: the generated configuration's digest while the router still serves
    the configuration it was checked on (its active hash unchanged), and the fleet's digest while Envoy's endpoints are
    the fleet's."""

    def __init__(self, docker: Docker, config_sha256: str, active: str, fleet_sha: str, endpoints: dict):
        self.docker, self.config_sha256, self.active = docker, config_sha256, active
        self.fleet_sha, self.endpoints = fleet_sha, endpoints

    async def __call__(self) -> SchedulerIdentity:
        active = await asyncio.to_thread(live.active_hash, self.docker)
        endpoints = await asyncio.to_thread(live.live_endpoints, self.docker)
        if active != self.active:
            raise ConfigError(f"the router now serves configuration {active}, it was checked on {self.active}")
        if endpoints != self.endpoints:
            raise ConfigError(f"Envoy's endpoints changed: {endpoints}")
        return SchedulerIdentity(config_digest=self.config_sha256, fleet_digest=self.fleet_sha)


def engine_parity(ours: Path, reference: Path) -> list[str]:
    """Every engine launched as for the paired main-grid run (gates/parity.py's rule: everything but the start time and
    the counters)."""
    try:
        a = open_cell(ours).engines("before")
        b = open_cell(reference).engines("before")
    except GateError as e:
        return [f"the engines cannot be compared with the paired main-grid run {reference.name}: {str(e)[:300]}"]
    if set(a.instances) != set(b.instances):
        return [f"the run's engines {sorted(a.instances)}, the paired run's {sorted(b.instances)}"]
    out = []
    for inst in sorted(a.instances):
        la, lb = a.instances[inst].launch(), b.instances[inst].launch()
        out += [f"{inst}.{key} differs from the paired run {reference.name}" for key in sorted(set(la) | set(lb))
                if la.get(key) != lb.get(key)]
    return out


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(allow_abbrev=False, description=__doc__.splitlines()[0])
    ap.add_argument("--profile", choices=("lease", "test"), required=True)
    ap.add_argument("--render", type=Path, required=True)
    ap.add_argument("--template-cell", required=True, help="the main-grid cell whose trace, seed and prompts it sends")
    ap.add_argument("--seed", type=int, help="the seed this run sends, if not the template's: the arrivals and the prompt "
                                             "order follow from the trace and the seed (bench.scope.ExpectedCell), as "
                                             "every system of that rate and seed is sent; the main grid renders no seeds "
                                             "2-3")
    ap.add_argument("--family", choices=sorted(C.LADDER), required=True)
    ap.add_argument("--fleet", type=Path, required=True)
    ap.add_argument("--hosts", type=Path, required=True)
    ap.add_argument("--engine-args", type=Path, required=True)
    ap.add_argument("--engines-record", type=Path, required=True)
    ap.add_argument("--reference-cells", type=Path, help="the main grid's cells root (lease: the paired run's engines)")
    ap.add_argument("--reference-cell", help="the main-grid run whose engines this run's must match (default: the "
                                             "template cell; vsrbench.block picks a valid one of the group)")
    ap.add_argument("--training-rows", type=Path, help="the training split's judged rows (lease: the means)")
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--cells-root", type=Path, required=True)
    ap.add_argument("--mirror", required=True, help="the second location (rsync syntax)")
    ap.add_argument("--vsr", type=Path, required=True, help="the vllm-sr CLI of the pinned venv")
    ap.add_argument("--templates", type=Path, required=True, help="the committed Envoy template directory")
    ap.add_argument("--models-cache", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True, help="where each run's CLI state directory is created")
    ap.add_argument("--connect", choices=("published", "envoy"), default="envoy",
                    help="the client's path to Envoy: its container address on the stack's network (the default: "
                         "direct, as our client reaches our scheduler) or the CLI's published port (through docker-proxy)")
    ap.add_argument("--docker-sg", action="store_true")
    ap.add_argument("--test-trace", help="test profile only: N,RATE (a short trace in place of the template's)")
    ap.add_argument("--test-pair", action="store_true",
                    help="test profile only: two models (m-small, m-large) in the pools, the selection path")
    return ap


def main(argv: list[str]) -> int:
    a = _parser().parse_args(argv)
    test = a.profile == "test"
    if not test and a.connect != "envoy":
        raise ConfigError("a lease run connects to Envoy directly")
    if (a.test_trace or a.test_pair) and not test:
        raise ConfigError("--test-trace and --test-pair are the test profile's")
    if test and (any(str(Path(x).resolve()).startswith(LEASE_ROOT) for x in (a.cells_root, a.work))
                 or LEASE_ROOT in a.mirror):
        raise ConfigError(f"a test run never writes under {LEASE_ROOT}")
    if not test:
        extra = sorted(set(os.environ) - set(CLIENT_ENV))
        if extra or os.environ.get("HF_HUB_OFFLINE") != "1" or not os.environ.get("HF_HOME"):
            raise ConfigError(f"a lease run starts as the main grid's client does (scripts/run_cell.sh: env -i with "
                              f"{CLIENT_ENV}, HF_HUB_OFFLINE=1, HF_HOME the node's Hub cache); found also {extra}")
        shared_tree_check()
    if C.VSR_VERSION not in subprocess.run([str(a.vsr), "--version"], capture_output=True, text=True).stdout:
        raise ConfigError(f"{a.vsr} is not vllm-sr {C.VSR_VERSION}")
    if tree_digest(a.templates) != C.TEMPLATE_SHA256:
        raise ConfigError(f"{a.templates} is not the committed template tree ({C.TEMPLATE_SHA256})")

    # 0. isolation before anything runs
    if not test and (port_open(SCHEDULER_PORT) or scheduler_processes()):
        raise ConfigError(f"our scheduler runs (port {SCHEDULER_PORT} or pids {scheduler_processes()}): stop it "
                          "(a vLLM-SR run shares no node service)")
    docker = Docker(a.docker_sg)
    scope, plan = cell_driver.load(a.render)
    template = cell_driver.expected(scope, a.template_cell)
    if a.seed is not None:
        template = dataclasses.replace(template, seed=a.seed)
    if template.dataset.budgets is not None:
        raise ConfigError("budget cells are not run through vLLM-SR")
    if test:
        instances = load_fleet(a.fleet, a.hosts)
        if a.test_trace:
            n, rate = a.test_trace.split(",")
            template = dataclasses.replace(template, trace=Trace(burstiness=1.0,
                                                                 segments=[Segment(n_requests=int(n), rate=float(rate))]))
    else:
        instances = cell_driver.fleet_of(plan, template.group, a.fleet, a.hosts)
        if a.reference_cells is None:
            raise ConfigError("--reference-cells is required for a lease run (the paired run's engines)")
    models = vc.lease_models(a.fleet, a.hosts, a.engine_args)
    binds = (vc.test_pair_bindings() if a.test_pair else vc.test_bindings()) if test else vc.bindings(a.family)
    if not test:
        ladder = vc.ladder_models(a.family)
        if sorted(m.name for m in models) != sorted(ladder):
            raise ConfigError(f"the fleet serves {[m.name for m in models]}, the {a.family} ladder {ladder}")
        models.sort(key=lambda m: ladder.index(m.name))        # the provider lists the models smallest first
    names = [m.name for m in models]
    if test:
        means = {"m-small": 0.33, "m-large": 0.43} if a.test_pair else {"m-small": 0.33}
        quality = {"means": means, "rows": 0, "sha256": None, "judge_key": None, "profile": "test"}
    else:
        if a.training_rows is None:
            raise ConfigError("--training-rows is required for a lease run")
        sha = _sha(a.training_rows)
        corpora = {v.get("sha256") for v in plan.get("scored_corpora", {}).values()}
        if sha in corpora:
            raise ConfigError(f"{a.training_rows} is a scored evaluation corpus: the policy is tuned on training data only")
        if sha != C.TRAINING_ROWS_SHA256.get(a.family):
            raise ConfigError(f"{a.training_rows} (sha256 {sha}) is not the pinned {a.family} training rows "
                              f"({C.TRAINING_ROWS_SHA256.get(a.family)})")
        quality = training_means(a.training_rows, names)
        means = quality["means"]

    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    started_unix = time.time()
    gen_dir = a.work / f"config.{a.family}.{stamp}"
    gen = vc.generate(gen_dir, a.vsr, vc.provider_document(models, means, test=test), binds, test=test)
    exp = vsr_cell(template, a.family, gen["sha256"]["config.yaml"], test)
    published = cell_dir(a.cells_root, exp.cell_id)
    if published.exists():
        raise ConfigError(f"{published} exists: a cell is never overwritten (retire it first)")
    held = cell_driver.unpublished(a.cells_root, exp.cell_id) + completed_attempts(a.cells_root, exp.cell_id)
    if held:
        raise ConfigError(f"{held} hold a run of {exp.cell_id}: publish or retire it first")
    reference = None if test else cell_dir(a.reference_cells, a.reference_cell or a.template_cell)
    attempt = a.cells_root / "_attempts" / f"{exp.cell_id}.{stamp}" / exp.cell_id
    ev = attempt / "vsr"
    ev.mkdir(parents=True)
    (attempt / "gates").mkdir()
    for f in ("provider.yaml", "bindings.yaml", "shipped.yaml", "config.yaml", "config.record.json"):
        (ev / f).write_bytes((gen_dir / f).read_bytes())
    _write(ev / "quality.json", quality)
    _write(ev / "harness.json", {**harness_record(), "versions": versions(a.vsr, Docker(a.docker_sg))})
    _write(ev / "template_cell.json", {"template": a.template_cell, "seed": exp.seed, "cell": exp.cell_id,
                                       "labels": exp.labels,
                                       "profile": a.profile, "connect": a.connect,
                                       "reference": None if reference is None else str(reference)})
    fleet_sha = fleet_digest(instances)
    stack = Stack(a.work / f"stack.{exp.cell_id}.{stamp}", a.vsr, a.templates, a.models_cache, docker)
    w, sampling = plan["warmup"], Sampling.model_validate(scope.bench.sampling.model_dump())
    drain_s = plan["drain"]["timeout_s"]
    generated = yaml.safe_load((ev / "config.yaml").read_text())

    async def drain() -> float:
        async with direct.session(2 * len(instances)) as sess:
            return await direct.drain(sess, instances, timeout_s=drain_s)

    async def warm() -> None:
        async with direct.session(len(instances)) as sess:
            backend = BackendClient(sess, drain_s)
            lines = await direct.warm_up(sess, backend, instances, requests_per_instance=w["requests_per_instance"],
                                         max_tokens=w["max_tokens"], prompt=w["prompt"], sampling=sampling,
                                         run_id=attempt.parent.name)
            waited = await direct.drain(sess, instances, timeout_s=drain_s)
        _write(attempt / "warmup.json", {"requests": lines, "drain_s": waited})

    # the classifier weights the stack will load: their contents as vllm-sr-bench/models.sha256 lists them
    bad_weights = verify_weights(a.models_cache)
    if bad_weights:
        raise ConfigError(f"the classifier weights at {a.models_cache} differ from models.sha256: {bad_weights[:5]}")

    # 2-4: until the run has served, any failure means it could not run (exit 2); the stack is torn down either way
    asyncio.run(drain())
    record, client, expected, active = None, None, None, None
    try:
        _write(ev / "stack_up.json", stack.up(ev / "config.yaml"))
        weights_up = listing(a.models_cache)
        changed = verify_weights(a.models_cache)
        if changed:
            raise ConfigError(f"the router changed classifier weights at startup: {changed[:5]}")
        asyncio.run(_engines(instances, a.engines_record, attempt / "engines_before.json"))
        max_len: dict[str, int] = {}
        for inst, e in open_cell(attempt).engines("before").instances.items():
            model = next(i.model for i in instances if i.instance_id == inst)
            if max_len.setdefault(model, e.max_model_len) != e.max_model_len:   # default-ok: the first replica's
                raise ConfigError(f"{model}'s replicas serve different contexts")
        expected = live.Expected(models, means, binds, generated, max_len, test)
        containers = stack.inspect()
        state = stack.store_state()
        created = docker.volumes_created()
        bad, got = live.problems(docker, containers, state, expected, replays=0)
        old = {v: t for v, t in created.items() if _unix(t) < started_unix - 1}
        if old:
            bad.append(f"storage volumes older than this run: {old}")
        _write(ev / "live_before.json", {"containers": containers, "stores": state, "volumes": created, **got})
        if bad:
            raise ConfigError("the stack does not execute its configuration or is not fresh: " + "; ".join(bad))
        active = got["status"]["config"]["active_hash"]
        pre, engine_versions = asyncio.run(engine_preflight(instances))
        _write(ev / "engine_preflight.json", {"problems": pre, "versions": engine_versions})
        if pre:
            raise ConfigError("the engines are not the campaign's: " + "; ".join(pre))
        identity = Identity(docker, exp.config.config_digest, active, fleet_sha, live.live_endpoints(docker))
        asyncio.run(warm())
        _guard([], "the listen counters before the run", lambda: _write(ev / "listen_before.json", listen_counters(docker)))

        tokenizer = load_hf_tokenizer(exp.dataset.tokenizer_id)
        prompts = load_prompts(a.data_root / exp.dataset.path, exp.trace.n_requests, exp.prompt_order, exp.seed,
                               tokenizer, exp.dataset.tokenizer_id)
        _check_before_sending(exp, prompts)
        spec = CellSpec(cell_id=exp.cell_id, labels=exp.labels, trace=exp.trace, seed=exp.seed,
                        sampling=scope.bench.sampling, expected_config_digest=exp.config.config_digest)
        host = "127.0.0.1" if a.connect == "published" else stack.envoy_address()
        client = VsrClient(f"http://{host}:{C.LISTENER_PORT}", scope.bench.max_connections,
                           scope.bench.request_timeout_s, identity)

        async def go():
            async with client:
                return await run_cell(spec, prompts, client, uuid.uuid4().hex[:8])
        record = asyncio.run(go())
        (ev / "raw_record.json").write_text(record.model_dump_json())       # the run is never lost from here on
    finally:
        if record is None:           # it could not run: the evidence so far, then the teardown
            _guard([], "collecting the stack's evidence", lambda: stack.collect(ev / "stack"))
            stack.down()

    # 5. after the run: every step's failure is a problem of the run
    post: list[str] = []
    _guard(post, "the client's evidence", lambda: (
        _write(ev / "headers.json", client.headers), _write(ev / "first_wall.json", client.first_wall),
        _write(ev / "client_loop_lag.json", client.lag.summary())))
    _guard(post, "the listen counters after the run", lambda: _write(ev / "listen_after.json", listen_counters(docker)))
    _guard(post, "the engines after the run",
           lambda: asyncio.run(_engines(instances, a.engines_record, attempt / "engines_after.json")))
    _guard(post, "Envoy's access log", lambda: _write(ev / "access_log_flush.json",
                                                      stack.wait_access_log({r.request_id for r in record.requests})))
    _guard(post, "the replay store", lambda: _write(ev / "replay_wait.json", stack.wait_replays(
        lambda: live.replay_total(docker), len(record.requests))))

    def after() -> None:
        state = stack.store_state()
        bad, got = live.problems(docker, stack.inspect(), state, expected, replays=len(record.requests))
        _write(ev / "live_after.json", {"stores": state, **got})
        post.extend(f"after the run: {p}" for p in bad)
        if got["status"]["config"]["active_hash"] != active:
            post.append("the router's active configuration changed during the run")
    _guard(post, "the live checks after the run", after)

    def weights() -> None:
        after_run = listing(a.models_cache)
        before = json.loads((ev / "stack_up.json").read_text())["models_listing"]
        _write(ev / "models_listing.json", {"before_serve": before, "after_startup": weights_up, "after_run": after_run})
        if after_run != weights_up:
            post.append(f"the classifier weights changed during the run: {_listing_diff(weights_up, after_run)}")
        if weights_up != before:
            notes.append(f"the router wrote into the classifier weights at startup: {_listing_diff(before, weights_up)}")
    notes: list[str] = []
    _guard(post, "the classifier weights' listing", weights)
    _guard(post, "collecting the stack's evidence", lambda: stack.collect(ev / "stack"))
    teardown: list[str] = []      # after the run's checks and evidence: a note (the next run's up() refuses leftovers)
    _guard(teardown, "tearing the stack down", stack.down)

    # 6. join, record, checks, verdict, publish, mirror
    try:
        record, lines, join_problems, join_notes = join.join(record, models, ev / "stack" / "envoy_access.log",
                                                        ev / "stack" / "vllm-sr-router-container.log", client.headers,
                                                        client.first_wall)
    except Exception as e:     # the raw record is published unjoined; the traffic check then fails it too
        lines, join_notes = [], []
        join_problems = [f"the join failed: {type(e).__name__}: {str(e)[:300]}"]
    _guard(join_problems, "writing the joined evidence", lambda: (ev / "requests.jsonl").write_text(
        "".join(json.dumps(line, sort_keys=True) + "\n" for line in lines)))
    write_cell(record, attempt)
    checks: list[str] = []
    _guard(checks, "the isolation checks", lambda: checks.extend(_isolation(attempt, instances)))
    if reference is not None:
        _guard(checks, "the engines' comparison with the paired run",
               lambda: checks.extend(engine_parity(attempt, reference)))

        def inputs() -> None:
            ref = inputs_reference(scope, a.reference_cells, a.template_cell, exp.seed)
            if ref is None:     # the render's checks below prove the inputs; vsrbench.block pairs the run once one is
                notes.append(f"inputs checked against the render (the arrival and prompt-order digests, per request); "
                             f"no published main-grid run at {exp.trace_id}, seed {exp.seed} to pair them with yet")
            else:
                checks.extend(input_problems(record, open_cell(ref).record))
                notes.append(f"inputs compared with the main-grid run {ref.name} (record sha256 "
                             f"{_sha(next(ref.glob('*.cell.json')))})")
        _guard(checks, "the inputs' comparison with the main grid", inputs)
    else:
        notes.append("test profile: no paired main-grid run to compare the engines and the inputs with")
    notes += join_notes + teardown
    problems = cell_problems(record, exp, scope.bench) + checks + post + join_problems
    _write(attempt / "gates" / "verdict.json", {"cell_id": exp.cell_id, "valid": not problems, "problems": problems,
                                                "notes": notes, "profile": a.profile})
    out = cell_driver.publish(attempt, a.cells_root)
    rc = subprocess.run(["bash", "-c", f'source "{REPO}/scripts/mirror.sh" && mirror_cell "$0" "$1" "$2"',
                         str(a.cells_root), a.mirror, exp.cell_id]).returncode
    print(f"{out}: {len(record.requests)} requests in {record.duration_s:.1f} s; "
          + ("valid" if not problems else "INVALID: " + "; ".join(problems)) + (f"; notes: {notes}" if notes else ""))
    if rc != 0:
        print(f"the mirror to {a.mirror} failed ({rc})", file=sys.stderr)
        return 3
    return 0 if not problems else 1


def _isolation(attempt: Path, instances) -> list[str]:
    cell = open_cell(attempt)
    before, after = cell.engines("before"), cell.engines("after")
    fleet = sorted(i.instance_id for i in instances)
    covered = [] if sorted(before.instances) == fleet else [f"the engine record covers {sorted(before.instances)}, "
                                                           f"the fleet {fleet}"]
    changed = engine_changes(before, after)
    return (reset_problems(cell) + traffic_problems(cell) + covered
            + ([f"an engine restarted or changed during the run: {changed}"] if changed else []))


def _guard(problems: list[str], what: str, step) -> None:
    try:
        step()
    except Exception as e:  # recorded as the run's problem: the run is kept (scripts/run_cell.sh's rule)
        problems.append(f"{what} failed: {type(e).__name__}: {str(e)[:300]}")
        print(f"[vsr run_cell] {what} failed: {type(e).__name__}: {e}", file=sys.stderr)


def _unix(created: str) -> float:
    """Docker's volume CreatedAt (RFC 3339 with an offset) as a Unix time."""
    return datetime.fromisoformat(created.strip().replace("Z", "+00:00")).timestamp()   # 3.10 rejects "Z"


def listing(root: Path) -> dict[str, list]:
    """Every file under the shared classifier weights: its size and mtime (the content was checked by the setup role
    against vllm-sr-bench/models.sha256)."""
    out = {}
    for p in sorted(Path(root).rglob("*")):
        if p.is_file():
            st = p.stat()
            out[str(p.relative_to(root))] = [st.st_size, st.st_mtime_ns]
    return out


def verify_weights(root: Path) -> list[str]:
    """The model files under the weights (every manifest entry but the download metadata the router may rewrite: .cache/
    and its revision receipts) are byte for byte the ones vllm-sr-bench/models.sha256 lists; returns the paths that
    are not."""
    bad = []
    for line in (MODULE / "models.sha256").read_text().splitlines():
        digest, rel = line.split("  ", 1)
        rel = rel.removeprefix("./")
        if "/.cache/" in f"/{rel}" or rel.endswith(".router-model-revision.json"):
            continue
        f = Path(root) / rel
        if not f.is_file() or _sha(f) != digest:
            bad.append(rel)
    return bad


def _listing_diff(a: dict, b: dict) -> list[str]:
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))[:10]


async def engine_preflight(instances) -> tuple[list[str], dict]:
    """gates/preflight.py's engine check on every engine (health, vLLM's version, the one model, drained), as
    every main-grid run passes before it starts."""
    async with direct.session(len(instances) * len(ENGINE_PATHS)) as sess:
        replies = {i.instance_id: dict(zip(ENGINE_PATHS, await asyncio.gather(
            *(fetch(sess, f"{i.base_url}{path}") for path in ENGINE_PATHS)), strict=True)) for i in instances}
    problems = [p for i in instances for p in engine_problems(i, replies[i.instance_id])]
    versions = {iid: r["/version"].body[:200] for iid, r in replies.items()}
    return problems, versions


INPUT_FIELDS = ("arrival_digest", "prompt_order_digest", "dataset_sha256", "budgets_sha256", "tokenizer_id", "sampling",
                "max_connections", "request_timeout_s", "seed", "trace", "prompt_order")


def input_problems(rec, ref) -> list[str]:
    """This run sent what the main-grid run at its rate and seed was sent: the record's inputs and, per request, the
    prompt row, the arrival and the prompt's token count, compared with that published record."""
    out = [f"{f} differs from the paired main-grid run {ref.cell_id}" for f in INPUT_FIELDS
           if getattr(rec, f) != getattr(ref, f)]
    mine = [(r.prompt_row, r.arrival_s, r.prompt_tokens) for r in rec.requests]
    theirs = [(r.prompt_row, r.arrival_s, r.prompt_tokens) for r in ref.requests]
    if mine != theirs:
        n = sum(a != b for a, b in zip(mine, theirs, strict=False)) + abs(len(mine) - len(theirs))   # lengths may differ
        out.append(f"{n} requests differ in prompt, arrival or prompt tokens from {ref.cell_id}")
    return out


def inputs_reference(scope, root: Path, template: str, seed: int) -> Optional[Path]:
    """A published, done main-grid run of the template's group at its rate and the given seed (the template itself
    first): its record is the inputs every system of that rate and seed was sent."""
    exp = {c.cell_id: c for c in expected_cells(scope)}
    t = exp[template]
    same = sorted(c.cell_id for c in exp.values() if c.group == t.group and c.trace_id == t.trace_id and c.seed == seed)
    same = sorted(same, key=lambda c: c != template)
    for cid in same:
        if cell_dir(root, cid).is_dir() and cell_driver.status(scope, exp[cid], root) == "done":
            return cell_dir(root, cid)
    return None


def netstat(text: str) -> dict:
    """TcpExt ListenOverflows and ListenDrops from a /proc/net/netstat text (as the scheduler records tcp_listen)."""
    lines = [ln.split() for ln in text.splitlines() if ln.startswith("TcpExt:")]
    if len(lines) != 2:
        return {}
    got = dict(zip(lines[0][1:], lines[1][1:], strict=True))
    return {k: int(got[k]) for k in ("ListenOverflows", "ListenDrops") if k in got}


def listen_counters(docker) -> dict:
    """The kernel's listen-queue counters of the host (our client's side) and of Envoy's network namespace (vLLM-SR's
    front door): recorded, never judged."""
    host = netstat(Path("/proc/net/netstat").read_text())
    envoy = netstat(docker.exec_sh("vllm-sr-envoy-container", "cat /proc/net/netstat"))
    return {"host": host, "envoy": envoy}


def completed_attempts(root: Path, cid: str) -> list[Path]:
    """Attempts of the cell whose run served (a raw record) but that were never published: recover or retire them."""
    base = Path(root) / "_attempts"
    if not base.is_dir():
        return []
    return sorted(d / cid for d in base.iterdir() if d.name.startswith(f"{cid}.") and (d / cid / "vsr" /
                                                                                       "raw_record.json").exists())


async def _engines(instances, record: Path, out: Path) -> None:
    _write(out, await engines.engines(instances, record, direct.SCRAPE_TIMEOUT_S))


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except Exception as e:          # it could not run: no published record (exit 2, as run_cell.sh)
        print(f"[vsr run_cell] could not run: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(2)
