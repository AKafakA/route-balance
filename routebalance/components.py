"""What a configuration loads: the estimator, the latency model, the router and the dispatcher, with the md5 of every
artifact file they read (the manifest's "artifacts")."""
from __future__ import annotations

import hashlib
import os
from concurrent.futures import Executor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from routebalance.baselines.dispatch import Dispatcher, build_dispatcher
from routebalance.baselines.pipeline import fleet_models
from routebalance.baselines.routers import (BEST_ROUTE_MODULES, DecoupledInputs, Estimate as DecoupledEstimate, Router,
                                            build_router)
from routebalance.config import ConfigError, SchedulerConfig
from routebalance.estimate.artifacts import md5_tree
from routebalance.telemetry import InstanceStats
from routebalance.types import Instance, Tier

P_UNDER_BUDGET_TOKENS = 256   # the estimator also reports P(length <= this); nothing here reads it (v3 passed 256)
# The router option that names the artifact directory a router loads by path (its files are md5-recorded).
ROUTER_ARTIFACT_OPTION = {"best_route_upstream": "checkpoint", "avengers_pro": "artifact_dir"}


@dataclass
class Components:
    estimator: object = None    # KnnEstimator-like: estimate_batch(prompts, budget_tokens) -> [{model: Estimate}]
    latency: object = None      # LatencyModel-like: latencies(instances, states, lengths, num_prompt_tokens)
    router: Optional[Router] = None
    dispatcher: Optional[Dispatcher] = None
    artifacts: dict[str, str] = field(default_factory=dict)   # {path under the artifact root: md5} of what was loaded


def needs(cfg: SchedulerConfig) -> tuple[bool, bool]:
    """(estimator, latency model) the configuration uses."""
    decoupled = cfg.pipeline is not None and cfg.pipeline.router.type == "rb_decoupled"
    rb = cfg.mode == "routebalance"
    return any((rb, decoupled, cfg.pending_decode == "own_length")), any((rb, decoupled))


def tiers(fleet: list[Instance]) -> list[Tier]:
    return list({i.tier.tier_id: i.tier for i in fleet}.values())


def longest(estimates) -> int:
    """The output length the latency model feeds every learned head: the largest expectation over the
    models (v3's convention, estimate/latency.py)."""
    return max(1, int(max(e.length for e in estimates.values())))


def idle_state(tier: Tier) -> InstanceStats:
    return InstanceStats(0, 0, 0.0, 0, tier.max_num_seqs, 0)


def load_components(cfg: SchedulerConfig, fleet: list[Instance], *, artifact_root: Path,
                    avengers_repo: Optional[Path], router_lane: Optional[Executor],
                    best_route_repo: Optional[Path] = None) -> Components:
    """Loads models and tables (seconds): call it off the event loop. A pipeline router makes its model calls on
    router_lane, sized by the configuration's router concurrency (baselines/routers.py router_lane: 1 as shipped,
    32 the main setting); RouteBalance's own calls use the shared pool."""
    from routebalance.estimate.knn import INDEX_FILE, load_knn
    from routebalance.estimate.latency import load_latency_model
    need_est, need_lat = needs(cfg)
    c = Components()
    if need_est:
        c.estimator = load_knn(cfg.estimator, artifact_root, fleet_models(fleet))
        c.artifacts[f"{cfg.estimator.index_dir}/{INDEX_FILE}"] = cfg.estimator.index_md5   # verified at load
        c.artifacts |= model_files("estimator", c.estimator.index.encoder_name, cfg.estimator.encoder_revision)
    if need_lat:
        c.latency = load_latency_model(cfg.latency, artifact_root, [t.tier_id for t in tiers(fleet)])
        c.artifacts |= latency_artifacts(cfg, c.latency)
    if cfg.pipeline is not None:
        decoupled = None
        if cfg.pipeline.router.type == "rb_decoupled":
            if cfg.latency.source == "xgboost":
                raise ConfigError("rb_decoupled scores idle instances; the xgboost source needs the request's "
                                  "prompt length, which the router interface does not carry")
            decoupled = decoupled_inputs(cfg, fleet, c.estimator, c.latency)
        priced: dict[str, tuple[float, float]] = {}   # (input, output) per token, by model: one pair per model
        for i in fleet:
            pair = (i.tier.cost_in, i.tier.cost_out)
            if priced.setdefault(i.model, pair) != pair:   # default-ok: the first instance of the model sets it
                raise ConfigError(f"model {i.model} has two prices in the fleet ({priced[i.model]} and {pair})")
        prices = {m: (cin + cout) / 2 for m, (cin, cout) in priced.items()}   # the routers' cheapest-first order
        c.router = build_router(cfg.pipeline.router, artifact_root=artifact_root, avengers_repo=avengers_repo,
                                executor=router_lane, decoupled=decoupled, best_route_repo=best_route_repo,
                                prices=prices)
        c.dispatcher = build_dispatcher(cfg.pipeline.dispatch)
        options = cfg.pipeline.router.options()
        option = ROUTER_ARTIFACT_OPTION.get(cfg.pipeline.router.type)
        if option is not None:          # loaded by path, not pinned by md5: hash every file of the directory
            c.artifacts |= md5_tree(artifact_root, str(options[option]))
        if cfg.pipeline.router.type == "avengers_pro":      # its embedder is loaded by name at a pinned revision
            c.artifacts |= model_files("avengers_pro", str(options["embedder"]), str(options["embedder_revision"]))
        if cfg.pipeline.router.type == "best_route_upstream":   # upstream's modules and its ranker's backbone
            c.artifacts |= {f"best_route_upstream@{rel}": hashlib.md5((best_route_repo / rel).read_bytes()).hexdigest()
                            for rel in BEST_ROUTE_MODULES}
            c.artifacts |= model_files("best_route_upstream", str(options["base_model"]), str(options["base_revision"]))
    return c


def model_files(role: str, name: str, revision: Optional[str]) -> dict[str, str]:
    """{"<role>@<model>/<file>": md5} for every file of a model that a component loaded by name
    (sentence-transformers): the local directory `name`, or else the Hugging Face cache snapshot that `name`
    resolves to at `revision` (None: the cached main), found without network access. The role keeps two
    components that load the same model apart (a parity diff compares a file with the same role only)."""
    local = Path(name)
    if local.is_dir():
        return {f"{role}@{k}": m for k, m in md5_tree(local.parent, local.name).items()}
    from huggingface_hub import snapshot_download     # installed with sentence-transformers
    # the cache SentenceTransformer itself uses when given no cache_folder (None: the default Hub cache)
    cache = os.environ.get("SENTENCE_TRANSFORMERS_HOME")   # env-ok: the loader's own cache, read to find its files
    snap = Path(snapshot_download(name, revision=revision, local_files_only=True, cache_dir=cache))
    return {f"{role}@hf:{name}/{k.split('/', 1)[1]}": m for k, m in md5_tree(snap.parent, snap.name).items()}


def latency_artifacts(cfg: SchedulerConfig, latency) -> dict[str, str]:
    """The files the latency model read, each verified against its pinned md5 at load (the xgboost source
    lists its md5s.json and every file that manifest pinned and it read; the prefill table when configured)."""
    t = cfg.latency
    if t.source != "xgboost":
        files = {t.table: t.table_md5}
    else:
        files = {f"{t.table}/{n}": m for n, m in latency.source.files.items()}
    if t.prefill is not None:
        files[t.prefill.table] = t.prefill.table_md5
    return files


def decoupled_inputs(cfg: SchedulerConfig, fleet: list[Instance], estimator, latency) -> DecoupledInputs:
    first: dict[str, Instance] = {}          # each tier's first instance in fleet order
    for inst in fleet:
        if inst.tier.tier_id not in first:
            first[inst.tier.tier_id] = inst

    def estimate(prompts):
        return [{m: DecoupledEstimate(e.quality, e.length) for m, e in d.items()}
                for d in estimator.estimate_batch(list(prompts), P_UNDER_BUDGET_TOKENS)]

    def unloaded(tier: Tier, tokens: float, prompt_tokens: int) -> float:
        inst = first[tier.tier_id]
        return latency.latencies([inst], [idle_state(tier)], {tier.model: tokens}, prompt_tokens)[0]
    return DecoupledInputs(tiers(fleet), cfg.weights, cfg.latency.normalization, estimate, unloaded)
