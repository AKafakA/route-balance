"""One cell's evidence directory, and strict readers for its files: the gates' only input format.

scripts/run_cell.sh writes, per cell, <cells_root>/<cell_id>/:
  <cell_id>.cell.json     the bench record (bench/records.py)
  config.json             the exact scheduler document POSTed for the cell
  decisions.jsonl         the scheduler's decision lines (routebalance/server.py) that carry the cell's config
                          digest, from the cell's window only
  stats_before.json       GET /v1/stats before the first request, read once no decision line is buffered
  stats_after.json        GET /v1/stats after the last reply, read once the decision log is flushed
  manifest_before.json    GET /v1/manifest before the first request
  manifest_after.json     GET /v1/manifest after the last reply
  group.json              {"group", "reference", "intended_differences"}: the cell's comparison group, written
                          from the scope (gates/parity.py)
  engines_before.json     every engine of the fleet as it reports itself before the cell, with the launch command
  engines_after.json      deploy.sh ran for it, and again after the last reply (tools/engines.py, D48)
  gates/*.log, latency.jsonl (the D38 sweep only)       not read by the gates
Every reader is strict. The gate that reads a file fails on any of these: the file is missing; a key is
missing or unknown; a value has the wrong type; a number is not finite; a JSON key is repeated. Nothing is
defaulted.
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Annotated, Any, Callable, Literal, Optional, Union

from pydantic import Field, ValidationError, field_validator, model_validator

from bench.records import CELL_SUFFIX, CellRecord, read_cell
from bench.schema import Strict, read_json
from routebalance.config import ConfigError, loads_strict

MANIFEST_FORMAT = "rb-manifest/1"
_MD5 = r"^[0-9a-f]{32}$"
_SHA256 = r"^[0-9a-f]{64}$"


class GateError(ValueError):
    """The gate cannot check the cell: an input is missing or malformed. The gate fails."""


# ---------------------------------------------------------------- GET /v1/stats

class Logged(Strict):
    decisions: int = Field(ge=0)              # decision lines on disk since the scheduler started
    decisions_buffered: int = Field(ge=0)     # written by the scheduler, not yet on disk
    latency: Optional[int]


class ConnectionWaits(Strict):
    """Cumulative waits for a pooled connection, per scheduler session (a requeued request counts again)."""
    engine: int = Field(ge=0)
    telemetry: int = Field(ge=0)


class Stats(Strict):
    loop_lag: dict[str, Union[int, float]]
    telemetry_read: Literal["per_round", "per_request", "poll"]   # D53, D55; the pollers' cache is D19
    telemetry_age_ms: Optional[dict[str, Union[float, str]]]   # poll only: instance -> age, or {"error": why stale}
    active: int = Field(ge=0)
    rounds: Optional[int]
    booster_calls: Optional[int]                     # None: no latency model is loaded
    failure: Optional[str]
    evidence_failure: Optional[str] = None           # D134: a log writer failed (the run is invalid, never refused)
    connection_waits: ConnectionWaits
    logged: Logged

    @model_validator(mode="after")
    def _ages_iff_polling(self):
        if (self.telemetry_read == "poll") != (self.telemetry_age_ms is not None):
            raise ValueError("telemetry ages are reported exactly when the system reads the pollers' cache")
        return self


# ---------------------------------------------------------------- GET /v1/manifest

class FleetEntry(Strict):
    instance_id: str
    tier: str
    model: str
    gpu: str
    cost_in: float
    cost_out: float
    max_num_seqs: int
    node: str


class SystemKeys(Strict):
    """The system-defining settings: the only ones that may differ between the systems of a group."""
    mode: Literal["routebalance", "pipeline"]
    telemetry_read: Literal["per_round", "per_request", "poll"]   # part of the system (D53, D55)
    weights: Optional[dict[str, float]]
    rounds: dict[str, Any]
    lpt: str
    pending_decode: Optional[str]
    latency: dict[str, Any]
    estimator: dict[str, Any]
    router: Optional[dict[str, Any]]
    dispatch: Optional[dict[str, Any]]

    @model_validator(mode="after")
    def _read_follows_mode(self):
        allowed = ("per_round", "poll") if self.mode == "routebalance" else ("per_request", "poll")
        if self.telemetry_read not in allowed:
            raise ValueError(f"a {self.mode} system cannot read telemetry '{self.telemetry_read}' (D53, D55)")
        return self


class TelemetrySettings(Strict):
    poll_ms: float
    max_age_ms: Optional[float] = None     # records made before D133 only
    timeout_s: float
    dead_reckoning: bool


class Loaded(Strict):
    estimator: Optional[str]
    latency_source: Optional[str]
    latency_envelope: Optional[str]
    router: Optional[str]
    dispatcher: Optional[str]
    round_loop: bool
    telemetry_pollers: bool                 # read from the live view: running exactly when telemetry_read is poll


class Code(Strict):
    """Digests of the code at the scheduler's start, and of the files on disk when the manifest was read."""
    routebalance: str = Field(pattern=_SHA256)
    routebalance_now: str = Field(pattern=_SHA256)
    avengers_pro: Optional[str]
    avengers_pro_now: Optional[str]

    def changed(self) -> list[str]:
        """'<tree>: <digest at start> -> <digest on disk now>' for every code tree that changed since start."""
        return [f"{k}: {getattr(self, k)} -> {getattr(self, f'{k}_now')}" for k in ("routebalance", "avengers_pro")
                if getattr(self, k) != getattr(self, f"{k}_now")]


class SessionSettings(Strict):
    limit: int
    limit_per_host: int
    force_close: bool


class Cpus(Strict):
    count: int
    affinity: int


class Launch(Strict):
    torch_threads: Optional[int]           # None: torch's default (D134)
    torch_threads_effective: Optional[int]
    max_connections: int
    request_timeout_s: float
    router_threads: int
    router_lane_threads: int
    engine_session: SessionSettings
    telemetry_session: SessionSettings
    event_loop: str
    cpus: Cpus
    env: dict[str, Optional[str]]


class Manifest(Strict):
    format: Literal["rb-manifest/1"]
    config_digest: str = Field(pattern=_SHA256)
    fleet_digest: str = Field(pattern=_SHA256)
    fleet: list[FleetEntry] = Field(min_length=1)
    system: SystemKeys
    budget: dict[str, Any]
    telemetry: TelemetrySettings
    loaded: Loaded
    artifacts: dict[str, Annotated[str, Field(pattern=_MD5)]]
    code: Code
    threadpools: dict[str, int]
    versions: dict[str, str]
    launch: Launch

    @model_validator(mode="after")
    def _pollers_follow_the_read(self):
        """What executes, not what the config says (the first rule): pollers run exactly for the cache read."""
        if self.loaded.telemetry_pollers != (self.system.telemetry_read == "poll"):
            raise ValueError(f"telemetry pollers {'run' if self.loaded.telemetry_pollers else 'do not run'} "
                             f"under telemetry_read '{self.system.telemetry_read}'")
        return self


# ---------------------------------------------------------------- group.json

# manifest keys a declared difference may name (config_digest is never compared: gates/parity.py)
PATTERN_ROOTS = tuple(k for k in Manifest.model_fields if k != "config_digest")
_SEGMENT = re.compile(r"[^\s.*\[\]]+(\[\d+\])*")


class GroupSpec(Strict):
    """The cell's comparison group. reference null: a main group, whose cells may differ in the declared
    patterns only. reference = a config id: an ablation group, each cell compared with the reference cells.
    A pattern is a manifest path, a leaf ("system.lpt") or a subtree ("system.rounds.*")."""
    group: str = Field(min_length=1)
    reference: Optional[str]
    intended_differences: list[str]

    @field_validator("intended_differences")
    @classmethod
    def _patterns(cls, v: list[str]) -> list[str]:
        for p in v:
            body = p[:-2] if p.endswith(".*") else p
            segments = body.split(".")
            if not (all(_SEGMENT.fullmatch(s) for s in segments) and re.split(r"[.\[]", body)[0] in PATTERN_ROOTS):
                raise ValueError(f"{p!r} is not a manifest path (a leaf, or <path>.* for a subtree) under one of "
                                 f"{list(PATTERN_ROOTS)}")
        if len(set(v)) != len(v):
            raise ValueError(f"repeated patterns in {v}")
        return v


def flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    """Leaf values by dotted path; list items by index (fleet[2].model); an empty dict or list is a leaf."""
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            out |= flatten(v, f"{prefix}.{k}" if prefix else str(k))
        return out if obj or not prefix else {prefix: {}}
    if isinstance(obj, list):
        out = {}
        for i, v in enumerate(obj):
            out |= flatten(v, f"{prefix}[{i}]")
        return out if obj else {prefix: []}
    return {prefix: obj}


def changed_keys(a: "Manifest", b: "Manifest") -> str:
    """Every key that differs between two manifests, with both values: "key: before -> after; ..."."""
    fa, fb = flatten(a.model_dump(mode="json")), flatten(b.model_dump(mode="json"))

    def show(f, k):
        return json.dumps(f[k], sort_keys=True) if k in f else "<absent>"
    return "; ".join(f"{k}: {show(fa, k)} -> {show(fb, k)}"
                     for k in sorted(set(fa) | set(fb)) if (k not in fa) or (k not in fb) or fa[k] != fb[k])


def parse_manifest(data: Any, what: str) -> Manifest:
    return _parse(Manifest, data, what)


def parse_stats(data: Any, what: str) -> Stats:
    return _parse(Stats, data, what)


def _parse(model, data: Any, what: str):
    try:
        return model.model_validate(data)
    except ValidationError as e:
        raise GateError(f"{what}: {e}") from e


# ---------------------------------------------------------------- engines_before.json / engines_after.json

class Served(Strict):
    """An engine's cumulative serving counters at one reading, from its /metrics (tools/engines.py, D132)."""
    requests: int = Field(ge=0)            # vllm:request_success_total over every finished_reason
    prompt_tokens: int = Field(ge=0)       # vllm:prompt_tokens_total
    generation_tokens: int = Field(ge=0)   # vllm:generation_tokens_total
    prefix_queries: int = Field(ge=0)      # vllm:prefix_cache_queries_total
    prefix_hits: int = Field(ge=0)         # vllm:prefix_cache_hits_total


class EngineLaunch(Strict):
    """One engine as it reports itself (tools/engines.py), with the launch command deploy.sh ran for it."""
    argv_sha256: str = Field(pattern=_SHA256)
    argv: str = Field(min_length=1)            # the planned vLLM arguments
    gpus: str = Field(min_length=1)            # CUDA_VISIBLE_DEVICES, as planned and as executed
    cmdline: str = Field(min_length=1)         # the process's command line, read on the node after deploy
    start_time_s: float = Field(gt=0)
    cache_config: dict[str, str] = Field(min_length=1)
    model: str = Field(min_length=1)
    max_model_len: int = Field(ge=1)
    served: Optional[Served] = None        # D132; records made before it have none (and fail traffic_problems)

    def launch(self) -> dict:
        """Everything but the start time and the counters: what a cell of the group must share with every other."""
        return self.model_dump(mode="json", exclude={"start_time_s", "served"})


class Engines(Strict):
    record_sha256: str = Field(pattern=_SHA256)
    instances: dict[str, EngineLaunch] = Field(min_length=1)


def engine_changes(before: Engines, after: Engines) -> list[str]:
    """What differs between a cell's two readings: a restart (start time) or another launch."""
    if set(before.instances) != set(after.instances):
        return [f"instances {sorted(before.instances)} vs {sorted(after.instances)}"]
    return [f"{i}.{k}" for i in sorted(before.instances)
            for k, v in before.instances[i].model_dump(mode="json", exclude={"served"}).items()
            if after.instances[i].model_dump(mode="json")[k] != v]


def fleet_uncovered(engines: Engines, manifest: Manifest) -> list[str]:
    """D48: the engine record reads every engine of the cell's fleet, and only those."""
    fleet = sorted(f.instance_id for f in manifest.fleet)
    return [] if sorted(engines.instances) == fleet else [f"engines {sorted(engines.instances)} vs the fleet {fleet}"]


# ---------------------------------------------------------------- the directory

@dataclass(frozen=True)
class Cell:
    dir: Path

    @property
    def cell_id(self) -> str:
        return self.dir.name

    def _file(self, name: str) -> Path:
        path = self.dir / name
        if not path.is_file():
            raise GateError(f"{path} is missing")
        return path

    def _json(self, name: str) -> Any:
        try:
            return read_json(self._file(name))
        except ConfigError as e:
            raise GateError(str(e)) from e

    @cached_property
    def record(self) -> CellRecord:
        cells = sorted(p.name for p in self.dir.glob(f"*{CELL_SUFFIX}"))
        if cells != [f"{self.cell_id}{CELL_SUFFIX}"]:
            raise GateError(f"{self.dir} must hold exactly one cell record, {self.cell_id}{CELL_SUFFIX}; found {cells}")
        try:
            rec = read_cell(self._file(cells[0]))
        except ConfigError as e:
            raise GateError(str(e)) from e
        if rec.cell_id != self.cell_id:
            raise GateError(f"{self.dir}: the record is for cell {rec.cell_id!r}")
        return rec

    @cached_property
    def decisions(self) -> list[dict]:
        path = self._file("decisions.jsonl")
        try:
            text = path.read_bytes().decode("utf-8")
        except (OSError, UnicodeDecodeError) as e:
            raise GateError(f"cannot read {path}: {e}") from e
        lines = []
        for n, raw in enumerate(text.splitlines(), 1):
            try:
                line = loads_strict(raw)
            except ConfigError as e:
                raise GateError(f"{path}:{n}: {e}") from e
            if not isinstance(line, dict):
                raise GateError(f"{path}:{n}: a decision line must be a JSON object")
            lines.append(line)
        return lines

    def stats(self, when: Literal["before", "after"]) -> Stats:
        return parse_stats(self._json(f"stats_{when}.json"), f"{self.dir}/stats_{when}.json")

    def manifest(self, when: Literal["before", "after"]) -> Manifest:
        return parse_manifest(self._json(f"manifest_{when}.json"), f"{self.dir}/manifest_{when}.json")

    @cached_property
    def group_spec(self) -> GroupSpec:
        return _parse(GroupSpec, self._json("group.json"), f"{self.dir}/group.json")

    def engines(self, when: Literal["before", "after"]) -> Engines:
        return _parse(Engines, self._json(f"engines_{when}.json"), f"{self.dir}/engines_{when}.json")


def open_cell(path: Path) -> Cell:
    path = Path(path)
    if not path.is_dir():
        raise GateError(f"{path} is not a cell directory")
    return Cell(path.resolve())


# ---------------------------------------------------------------- reporting

class Problems:
    """Collects a gate's reasons. A rule that fails on many lines shows a few examples and a count, so
    one reason stays one line and the output stays readable for a whole cell."""

    EXAMPLES = 5

    def __init__(self):
        self._by_rule: dict[str, list[str]] = {}

    def add(self, rule: str, detail: str = "") -> None:
        self._by_rule.setdefault(rule, []).append(detail)   # default-ok: grouping

    def lines(self) -> list[str]:
        out = []
        for rule, details in self._by_rule.items():
            shown = [d for d in details if d][:self.EXAMPLES]
            more = len(details) - len(shown) if shown else 0
            text = rule if not shown else f"{rule}: " + "; ".join(shown) + (f"; and {more} more" if more else "")
            out.append(text)
        return out

    def __bool__(self) -> bool:
        return bool(self._by_rule)


def run(gate: str, check: Callable[[], tuple[list[str], str]]) -> int:
    """Run one gate: print one reason per line and return 1, or print one PASS line and return 0."""
    try:
        problems, summary = check()
    except (GateError, ConfigError) as e:
        problems, summary = [f"cannot check: {e}"], ""
    except Exception as e:          # an input the gate did not anticipate: still a failure, with its reason
        problems, summary = [f"cannot check: {type(e).__name__}: {e}"], ""
    if problems:
        for p in problems:
            print(f"FAIL {gate}: " + " | ".join(part.strip() for part in str(p).splitlines() if part.strip()))
        return 1
    print(f"PASS {gate}: {summary}")
    sys.stdout.flush()
    return 0


# ---------------------------------------------------------------- isolation (D128, D132)

RESET_STEPS = ("prime", "after_reset", "control")


def reset_problems(cell: Cell) -> list[str]:
    """D128: every engine of the cell's fleet started the cell from an emptied prefix cache (but for the warm-up
    prompt's own blocks, the same in every cell), as its warm-up proved
    (tools/direct.py): the prime, then after vLLM's reset a request with no cached token, then the same prompt again
    with some. A cell without that proof (made before the reset, or on engines that could not reset) is invalid."""
    try:
        fleet = sorted(cell.engines("before").instances)
        lines = cell._json("warmup.json")["requests"]
    except (GateError, KeyError, TypeError) as e:
        return [f"the warm-up record cannot prove the cache reset: {str(e)[:300]}"]
    out = []
    for inst in fleet:
        steps = {x.get("cache_step"): x for x in lines if isinstance(x, dict) and x.get("instance_id") == inst}
        missing = [k for k in RESET_STEPS if k not in steps]
        if missing:
            out.append(f"{inst}: the warm-up has no {missing} step: the prefix-cache reset is not proven")
            continue
        after, control = steps["after_reset"], steps["control"]
        if not (after.get("cache_queried", 0) > 0 and after.get("cache_hit") == 0):
            out.append(f"{inst}: the first request after the reset found {after.get('cache_hit')} cached tokens of "
                       f"{after.get('cache_queried')}")
        if not control.get("cache_hit", 0) > 0:
            out.append(f"{inst}: the control request found no cached token: the counters do not show the cache")
    return out


def traffic_problems(cell: Cell) -> list[str]:
    """D132: between the cell's two engine readings (the first taken before the warm-up, so the reset and its proof are
    inside), only the warm-up's and the cell's own requests reached the engines. Per engine, the change in finished
    requests, prompt tokens and generated tokens must equal what warmup.json and the record say it served."""
    try:
        before, after, rec = cell.engines("before"), cell.engines("after"), cell.record
        warm = cell._json("warmup.json")["requests"]
    except (GateError, KeyError, TypeError) as e:
        return [f"the engine readings cannot show the traffic: {str(e)[:300]}"]
    served: dict[str, list[int]] = {i: [0, 0, 0] for i in before.instances}
    for w in warm:
        if not (isinstance(w, dict) and w.get("success") and w.get("instance_id") in served):
            return [f"a warm-up line is not a served request of the fleet: {str(w)[:200]}"]
        t = served[w["instance_id"]]
        t[0] += 1
        t[1] += w["prompt_tokens"]
        t[2] += w["output_tokens"]
    for r in rec.requests:
        if r.reply is not None:
            if r.reply.instance_id not in served:
                return [f"request {r.request_id} was served by {r.reply.instance_id}, which the engine record lacks"]
            t = served[r.reply.instance_id]
            t[0] += 1
            t[1] += r.reply.prompt_tokens
            t[2] += r.reply.output_tokens
    out = []
    for i, want in sorted(served.items()):
        b, a = before.instances[i].served, after.instances.get(i) and after.instances[i].served
        if b is None or a is None:
            out.append(f"{i}: no serving counters in the engine readings")
            continue
        got = [a.requests - b.requests, a.prompt_tokens - b.prompt_tokens, a.generation_tokens - b.generation_tokens]
        if got != want:
            out.append(f"{i}: the engine served (requests, prompt tokens, generated tokens) {tuple(got)} during the "
                       f"cell, its record {tuple(want)}: other traffic reached it, or the record is short")
    return out


def scheduler_problems(cell: Cell) -> list[str]:
    """D132: the cell ran on a fresh scheduler process (it had logged no decision before the cell), and D134: no
    evidence writer failed during it (requests were still served; the cell is invalid)."""
    try:
        before, after = cell.stats("before"), cell.stats("after")
    except GateError as e:
        return [f"the scheduler's stats cannot show a fresh process: {str(e)[:300]}"]
    out = []
    if before.logged.decisions != 0:
        out.append(f"the scheduler had logged {before.logged.decisions} decisions before the cell: not a fresh process")
    failed = [f for f in (before.evidence_failure, after.evidence_failure) if f is not None]
    if failed:
        out.append(f"an evidence writer failed: {failed[0]}")
    return out


def isolation_problems(path: Path) -> list[str]:
    """The isolation checks on a cell directory (the verdict, resume and the report's coverage all apply them)."""
    try:
        cell = open_cell(path)
    except GateError as e:
        return [str(e)]
    return reset_problems(cell) + traffic_problems(cell) + scheduler_problems(cell)
