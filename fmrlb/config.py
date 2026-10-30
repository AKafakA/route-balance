"""Typed, fail-loud configuration.

A run is defined by three documents:
  fleet      configs/fleet_*.json (tracked): tiers (model, GPU, prices) and instances by logical node
  hosts      local/hosts.json (untracked): node -> {address, hw, service_address?}; real host names never enter git
  scheduler  one complete document; POST /v1/config replaces it whole (validated first, no merging)

Every key is required unless stated, unknown keys are rejected, and no value is coerced from a string.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import socket
from pathlib import Path
from typing import Annotated, Any, Literal, Mapping, Optional, Union

from pydantic import (AfterValidator, BaseModel, ConfigDict, Field, ValidationError, field_serializer, field_validator,
                      model_validator)

from fmrlb.hw import UnknownHardware, gpu_of
from fmrlb.types import Instance, Tier


class ConfigError(ValueError):
    pass


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)

    def model_copy(self, *args, **kwargs):  # updates must go through parse_scheduler, never an unvalidated copy
        raise TypeError("configs are replaced whole: build a new document and call parse_scheduler()")


_ID = r"^[A-Za-z0-9._-]+$"          # ASCII only: no whitespace, no lookalike characters
_ID_RE = re.compile(r"[A-Za-z0-9._-]+")          # used with fullmatch: Python's `$` would accept a trailing newline
_TRIMMED = r"^\S(.*\S)?$"                          # non-empty, no leading/trailing whitespace
_HOST = re.compile(r"[a-z0-9]([a-z0-9.-]*[a-z0-9])?")
Scalar = Union[bool, int, float, str]


def _relative_path(v: str) -> str:
    """An artifact path relative to the artifact root: no absolute paths, no '..', no surrounding whitespace."""
    if not v or v != v.strip() or v.startswith("/") or ".." in v.split("/"):
        raise ValueError(f"{v!r} must be a relative path inside the artifact root")
    return v


ArtifactPath = Annotated[str, AfterValidator(_relative_path)]      # relative to the artifact root


# ---------------------------------------------------------------- fleet


class _TierSpec(_Strict):
    model: str = Field(pattern=_TRIMMED)
    gpu: Literal["a30", "v100", "a100"]
    cost_in: float = Field(ge=0)
    cost_out: float = Field(gt=0)
    max_num_seqs: int = Field(ge=1, le=4096)   # passed to vLLM as --max-num-seqs; the telemetry view reads it here
    # token ids every request to this tier also stops at (vLLM's stop_token_ids), besides the model's own
    # end-of-sequence tokens: the Qwen2.5 base models stop only at <|endoftext|>, so under the chat template an answer
    # could end its turn (<|im_end|>) and run on into a further turn; [] where the model's own tokens suffice
    stop_token_ids: list[Annotated[int, Field(ge=0)]]


class _InstanceSpec(_Strict):
    instance_id: str = Field(pattern=_ID)
    tier: str = Field(pattern=_ID)
    node: str = Field(pattern=_ID)
    port: int = Field(gt=0, lt=65536)


class _FleetSpec(_Strict):
    tiers: dict[str, _TierSpec] = Field(min_length=1)
    instances: list[_InstanceSpec] = Field(min_length=1)


class _HostSpec(_Strict):
    address: str
    hw: str
    service_address: Optional[str] = None   # the engines are served to the scheduler here (the experiment network)

    @field_validator("address", "service_address")
    @classmethod
    def _plain_host(cls, v: Optional[str]) -> Optional[str]:
        # a bare lower-case host name or IPv4 address: no scheme, port, whitespace or trailing dot
        if v is None:
            return v
        if not _HOST.fullmatch(v):
            raise ValueError(f"address {v!r} must be a bare lower-case host name or IPv4 address")
        try:
            packed = socket.inet_aton(v)   # accepts every numeric spelling the resolver would (127.1, 0x7f.0.0.1, ...)
        except OSError:
            return v                       # a host name
        if v != socket.inet_ntoa(packed):
            raise ValueError(f"numeric address {v!r} must be written as the canonical dotted quad "
                             f"{socket.inet_ntoa(packed)!r}")
        return v


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict:
    out: dict = {}
    for k, v in pairs:
        if k in out:
            raise ConfigError(f"duplicate JSON key {k!r}")
        out[k] = v
    return out


def loads_strict(text: str) -> Any:
    """json.loads that rejects duplicate keys and non-finite numbers. Used for files and for POST bodies."""
    def _no_constant(c: str):
        raise ConfigError(f"non-finite number {c} in JSON")
    try:
        return json.loads(text, object_pairs_hook=_no_duplicate_keys, parse_constant=_no_constant)
    except ConfigError:
        raise
    except (json.JSONDecodeError, RecursionError, ValueError) as e:
        raise ConfigError(f"invalid JSON: {e}") from e


def read_json(path: Path) -> Any:
    try:
        text = Path(path).read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError) as e:
        raise ConfigError(f"cannot read {path}: {e}") from e
    try:
        return loads_strict(text)
    except ConfigError as e:
        raise ConfigError(f"{path}: {e}") from e


def parse(model: type[BaseModel], data: Any, what: str):
    try:
        return model.model_validate(data)
    except ValidationError as e:
        raise ConfigError(f"invalid {what}: {e}") from e


def load_fleet(fleet_path: Path, hosts_path: Path) -> list[Instance]:
    """Instances in fleet-file order (order is behaviour: dispatchers break ties by it)."""
    fleet = parse(_FleetSpec, read_json(fleet_path), f"fleet {fleet_path}")
    raw_hosts = read_json(hosts_path)
    if not isinstance(raw_hosts, dict):
        raise ConfigError(f"hosts file {hosts_path} must map node -> {{address, hw, service_address?}}")
    hosts = {n: parse(_HostSpec, h, f"host {n!r}") for n, h in raw_hosts.items()}

    bad_ids = [tid for tid in fleet.tiers if not _ID_RE.fullmatch(tid)]
    if bad_ids:
        raise ConfigError(f"tier ids must match {_ID}: {bad_ids}")
    pairs = [(t.model, t.gpu) for t in fleet.tiers.values()]
    if len(set(pairs)) != len(pairs):
        raise ConfigError("two tiers share one (model, gpu); prices would be ambiguous")
    bad_nodes = [n for n in raw_hosts if not _ID_RE.fullmatch(n)]
    if bad_nodes:
        raise ConfigError(f"node names must match {_ID}: {bad_nodes}")
    hw_by_address: dict[str, str] = {}
    for h in hosts.values():
        seen = hw_by_address.get(h.address)
        if seen is not None and seen != h.hw:
            raise ConfigError(f"address {h.address!r} is listed with two hardware types")
        hw_by_address[h.address] = h.hw
    tiers = {tid: Tier(tid, t.model, t.gpu, t.cost_in, t.cost_out, t.max_num_seqs, tuple(t.stop_token_ids))
             for tid, t in fleet.tiers.items()}
    out: list[Instance] = []
    seen_ids: set[str] = set()
    seen_endpoints: set[tuple[str, int]] = set()
    for spec in fleet.instances:
        if spec.instance_id in seen_ids:
            raise ConfigError(f"duplicate instance_id {spec.instance_id!r}")
        if spec.tier not in tiers:
            raise ConfigError(f"instance {spec.instance_id!r}: unknown tier {spec.tier!r}")
        if spec.node not in hosts:
            raise ConfigError(f"instance {spec.instance_id!r}: node {spec.node!r} missing from {hosts_path}")
        host = hosts[spec.node]
        try:
            gpu = gpu_of(host.hw)
        except UnknownHardware as e:
            raise ConfigError(f"node {spec.node!r}: {e}") from e
        tier = tiers[spec.tier]
        if gpu != tier.gpu:
            raise ConfigError(f"instance {spec.instance_id!r}: node {spec.node!r} is {host.hw} ({gpu}) "
                              f"but tier {tier.tier_id!r} expects {tier.gpu}")
        served_at = host.address if host.service_address is None else host.service_address
        endpoints = {(host.address, spec.port), (served_at, spec.port)}
        if endpoints & seen_endpoints:
            raise ConfigError(f"instance {spec.instance_id!r}: endpoint {sorted(endpoints & seen_endpoints)} already used")
        seen_ids.add(spec.instance_id)
        seen_endpoints |= endpoints
        out.append(Instance(spec.instance_id, tier, spec.node, host.address, spec.port, host.service_address))
    unused = set(tiers) - {i.tier.tier_id for i in out}
    if unused:
        raise ConfigError(f"tiers with no instance: {sorted(unused)}")
    return out


# ---------------------------------------------------------------- scheduler


class Weights(_Strict):
    w_quality: float = Field(ge=0)
    w_latency: float = Field(ge=0)
    w_cost: float = Field(ge=0)

    @model_validator(mode="after")
    def _simplex(self):
        total = self.w_quality + self.w_latency + self.w_cost
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"weights must sum to 1, got {total}")
        return self


class Rounds(_Strict):
    # adaptive: round size and timeout follow the busy-instance ratio (<0.3: 4/50 ms, <0.7: 16/100 ms,
    # else 32/200 ms). Otherwise the fixed values below apply (the round-size ablations).
    adaptive: bool
    max_batch: Optional[int] = Field(default=None, ge=1, le=1024)
    timeout_ms: Optional[float] = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _fixed_needs_values(self):
        fixed = (self.max_batch, self.timeout_ms)
        if self.adaptive and any(v is not None for v in fixed):
            raise ValueError("adaptive rounds take no max_batch/timeout_ms")
        if not self.adaptive and any(v is None for v in fixed):
            raise ValueError("fixed rounds need max_batch and timeout_ms")
        return self


# The (TPOT source, envelope) pairs that exist (fmrlb/estimate/latency.py defines each):
#   constant + own_length        queue-blind: the request's own decode at a constant TPOT
#   constant + queue_aware       the queue envelope with a constant TPOT
#   xgboost + queue_aware        the learned TPOT heads in the queue envelope
#   profiled_curve + batch_aware TPOT as a function of the running batch, plus a waiting term when full
#   constant + batch_aware       the same envelope on a flat TPOT: the control that isolates the curve
LATENCY_PAIRS = frozenset({("constant", "own_length"), ("constant", "queue_aware"), ("xgboost", "queue_aware"),
                           ("profiled_curve", "batch_aware"), ("constant", "batch_aware")})


class Prefill(_Strict):
    table: ArtifactPath  # the prefill table (tools/fit_prefill.py): per tier, TTFT as a line in the prompt's tokens
    table_md5: str = Field(pattern=r"^[0-9a-f]{32}$")


class Latency(_Strict):
    source: Literal["profiled_curve", "constant", "xgboost"]
    envelope: Literal["batch_aware", "queue_aware", "own_length"]
    normalization: Literal["fixed_ref", "max_divide"]
    table: ArtifactPath  # calibration table (curve or constants) or XGBoost dir
    table_md5: str = Field(pattern=r"^[0-9a-f]{32}$")   # of the table file, or of the XGBoost dir's md5s.json
    # the prompt's prefill, added to every envelope; null: no prefill term
    prefill: Optional[Prefill]

    @model_validator(mode="after")
    def _supported_pair(self):
        if (self.source, self.envelope) not in LATENCY_PAIRS:
            raise ValueError(f"latency source {self.source!r} has no envelope {self.envelope!r}; "
                             f"supported pairs: {sorted(LATENCY_PAIRS)}")
        return self


class Estimator(_Strict):
    index_dir: ArtifactPath
    index_md5: str = Field(pattern=r"^[0-9a-f]{32}$")
    encoder_revision: str = Field(pattern=r"^[0-9a-f]{40}$")   # Hub commit of the index's sentence encoder
    k: int = Field(ge=1, le=1000)
    score_type: Literal["deepeval"]
    prior: Literal["knn", "per_model_mean"]


class Budget(_Strict):
    admission_filter: bool                # the budget filter (fmrlb/schedule/assign.py Admission)


class Telemetry(_Strict):
    poll_ms: float = Field(gt=0)
    timeout_s: float = Field(gt=0)
    # no age bound: a stale snapshot is counted, never a reason to refuse a request; the configurations of older
    # records carry one, which nothing reads
    max_age_ms: Optional[float] = Field(default=None, gt=0)


class Router(_Strict):
    type: Literal["avengers_pro", "best_route", "best_route_upstream", "passthrough", "rb_decoupled"]
    # flat, finite scalars; stored as sorted (key, value) pairs so the frozen config is truly immutable and hashable
    kwargs: tuple[tuple[str, Scalar], ...]
    # how many of the router's model calls run at once (baselines/routers.py router_lane: 1 as shipped, 32 the main
    # setting); required by every router that calls a model, taken by no other (passthrough calls none)
    concurrency: Optional[int] = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _concurrency_where_a_model_is_called(self):
        if (self.type == "passthrough") == (self.concurrency is not None):
            raise ValueError("concurrency is required by every router that calls a model and taken by passthrough")
        return self

    @field_validator("kwargs", mode="before")
    @classmethod
    def _pairs(cls, v: Any) -> Any:
        if not isinstance(v, Mapping):
            raise ValueError("router kwargs must be an object")
        for k, x in v.items():
            if not isinstance(x, (bool, int, float, str)):
                raise ValueError(f"router kwarg {k!r} must be a scalar")
            if isinstance(x, float) and not math.isfinite(x):
                raise ValueError(f"router kwarg {k!r} is not finite")
        return tuple(sorted((str(k), x) for k, x in v.items()))

    @field_serializer("kwargs")
    def _as_dict(self, v: tuple[tuple[str, Scalar], ...]) -> dict:
        return dict(v)

    def options(self) -> dict[str, Scalar]:
        """A fresh copy of the router's options."""
        return dict(self.kwargs)


LOAD_READING_DISPATCHERS = ("least_outstanding", "po2")    # the dispatchers whose pick depends on the loads


class Dispatch(_Strict):
    # no shortest_queue: the shortest queue is least_outstanding on the engines' counts
    type: Literal["round_robin", "po2", "least_outstanding", "random", "lowest_latency"]
    seed: int = Field(ge=0, lt=2**32)
    # lowest_latency only (required there, rejected elsewhere): latency samples kept per instance
    latency_window: Optional[int] = Field(default=None, ge=1)
    # the load-reading dispatchers only (required there, rejected elsewhere): "engine" or "in_flight", as
    # baselines/dispatch.py describes
    count: Optional[Literal["in_flight", "engine"]] = Field(default=None)

    @model_validator(mode="after")
    def _window_only_for_lowest_latency(self):
        if (self.type == "lowest_latency") != (self.latency_window is not None):
            raise ValueError("latency_window is required by the lowest_latency dispatcher and taken by no other")
        if (self.type in LOAD_READING_DISPATCHERS) != (self.count is not None):
            raise ValueError(f"count is required by the load-reading dispatchers {LOAD_READING_DISPATCHERS} and "
                             "taken by no other")
        return self


class Pipeline(_Strict):
    router: Router
    dispatch: Dispatch


class SchedulerConfig(_Strict):
    mode: Literal["fmrlb", "pipeline"]
    weights: Optional[Weights]
    rounds: Rounds
    lpt: Literal["max", "min", "mean", "off"]     # LPT key statistic over the models' expected lengths
    # decode tokens an assignment adds to its instance's pending decode (schedule/assign.py): required for
    # F-MRLB; a pipeline system records max_tokens (None) or, for the latency sweep, own_length
    pending_decode: Optional[Literal["own_length", "lpt_key"]]
    dead_reckoning: bool
    # how load is read (telemetry.py describes the three ways): F-MRLB per round, with dead
    # reckoning; the pipeline systems from the pollers' cache or per request
    telemetry_read: Literal["per_round", "per_request", "poll"]
    # what F-MRLB's score normalizes over. "round": quality and cost by the round's range
    # over every request x instance (known when the round opens), latency (max_divide) by the round's running
    # maximum; "request": each request's own candidates. Required in fmrlb mode, taken by no
    # pipeline system (they score one request at a time)
    normalize_over: Optional[Literal["request", "round"]] = None
    latency: Latency
    estimator: Estimator
    budget: Budget
    telemetry: Telemetry
    pipeline: Optional[Pipeline]

    @model_validator(mode="after")
    def _mode_consistent(self):
        if self.mode == "pipeline" and self.pipeline is None:
            raise ValueError("mode 'pipeline' needs a pipeline section")
        if self.mode == "fmrlb" and self.pipeline is not None:
            raise ValueError("mode 'fmrlb' takes no pipeline section")
        decoupled = self.pipeline is not None and self.pipeline.router.type == "rb_decoupled"
        uses_weights = any((self.mode == "fmrlb", decoupled))
        if uses_weights and self.weights is None:
            raise ValueError("F-MRLB scoring needs weights")
        if not uses_weights and self.weights is not None:
            raise ValueError("weights given but this configuration never scores with them")
        if self.mode == "fmrlb" and self.pending_decode is None:
            raise ValueError("pending_decode is required in fmrlb mode")
        if (self.mode == "fmrlb") != (self.normalize_over is not None):
            raise ValueError("normalize_over is required in fmrlb mode and taken by no pipeline system")
        if self.mode != "fmrlb" and self.pending_decode not in (None, "own_length"):
            raise ValueError("a pipeline system records max_tokens as pending decode, or with pending_decode "
                             "own_length (the latency sweep only) the estimator's length; lpt_key is RB-only")
        if self.mode == "fmrlb" and self.telemetry_read == "per_request":
            raise ValueError("F-MRLB reads load per round or from the pollers' cache, never per request")
        if self.mode != "fmrlb" and self.telemetry_read == "per_round":
            raise ValueError("a pipeline system decides per request: it reads the pollers' cache or probes per request")
        sweep = self.mode != "fmrlb" and self.pending_decode == "own_length"   # the latency sweep
        if self.mode != "fmrlb" and self.dead_reckoning and not sweep:
            raise ValueError("dead reckoning is F-MRLB's own mechanism: a pipeline system's dispatcher reads "
                             "the engines' own gauges or counts in-flight requests")
        if sweep and not self.dead_reckoning:
            raise ValueError("the latency sweep logs the learned heads' training features, which F-MRLB serves "
                             "on dead-reckoned counts: it runs with dead reckoning (its random dispatch reads no load)")
        if self.telemetry_read == "per_request" and self.pipeline.dispatch.type not in LOAD_READING_DISPATCHERS:
            raise ValueError(f"a per-request probe would change nothing for the {self.pipeline.dispatch.type} "
                             f"dispatcher, which reads no load")
        if self.pipeline is not None and self.dead_reckoning and self.pipeline.dispatch.count == "engine":
            raise ValueError("a dispatcher counting the engines' gauges needs a view without dead reckoning, which "
                             "would replace them by the in-flight count")
        if self.telemetry_read == "per_request" and self.pipeline.dispatch.count != "engine":
            raise ValueError("a per-request probe fetches the engines' gauges: its dispatcher counts them "
                             "(count engine); an in-flight count needs no fetch")
        if self.mode != "fmrlb" and self.budget.admission_filter:
            raise ValueError("the budget admission filter exists only in fmrlb mode (pipeline systems get the clamp)")
        return self


def parse_scheduler(data: Any) -> SchedulerConfig:
    """Validate one complete scheduler document. Used at startup and for every POST /v1/config
    (parse the body with loads_strict first). The result replaces the old configuration whole."""
    return parse(SchedulerConfig, data, "scheduler config")


def load_scheduler(path: Path) -> SchedulerConfig:
    return parse_scheduler(read_json(path))


def _canonical(obj: Any) -> Any:
    if isinstance(obj, float):
        return obj + 0.0            # -0.0 and 0.0 are the same setting
    if isinstance(obj, Mapping):
        return {k: _canonical(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_canonical(v) for v in obj]
    return obj


def _digest(obj: Any) -> str:
    canonical = json.dumps(_canonical(obj), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def config_digest(cfg: SchedulerConfig) -> str:
    """Stable identity of a scheduler configuration, recorded in every cell."""
    return _digest(cfg.model_dump(mode="python"))


def fleet_digest(instances: list[Instance]) -> str:
    """Identity of the deployment in order (order is behaviour), including tiers and prices."""
    return _digest([[i.instance_id, i.tier.tier_id, i.tier.model, i.tier.gpu, i.tier.cost_in, i.tier.cost_out,
                     i.tier.max_num_seqs, list(i.tier.stop_token_ids), i.node, i.address, i.port]
                    + ([i.service_address] if i.service_address else []) for i in instances])   # the path used
