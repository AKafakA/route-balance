"""Baseline routers: each picks a MODEL for a request; a dispatcher then picks the instance.

  passthrough   no model choice: the dispatcher sees every instance (the v3 "__ALL__" sentinel)
  best_route    BEST-Route's 4-way DeBERTa classifier: argmax, with an optional confidence floor
  avengers_pro  the upstream AvengersPro cluster router, wrapped as v3 wrapped it
  rb_decoupled  RouteBalance's own predictors pick the model from UNLOADED latency (no telemetry),
                so a dispatcher places it afterwards: "router in front of a load balancer"

Decisions replay v3 exactly. What changed is execution: every model call (tokenizer, forward pass,
embedding, estimator) runs on one bounded thread pool shared by all routers, never on the event
loop, as RouteBalance's own estimator does. No router falls back silently: every failure raises.
"""
from __future__ import annotations

import asyncio
import dataclasses
import functools
import hashlib
import importlib
import logging
import math
import sys
from abc import ABC, abstractmethod
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Mapping, NamedTuple, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from routebalance.schedule.score import Candidate, pick, score
from routebalance.config import ConfigError, Router as RouterConfig, Weights, _relative_path, loads_strict
from routebalance.types import Tier

ALL_INSTANCES = "__ALL__"   # passthrough's decision: every instance is a candidate

# One pool for all router inference. At the highest arrival rate (30 req/s) BEST-Route needs about
# 1.5 worker-seconds per second at its measured ~50 ms per prompt, so four workers leave headroom
# while bounding CPU contention on the scheduler node, which also serves a model instance.
ROUTER_THREADS = 4


def router_executor() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=ROUTER_THREADS, thread_name_prefix="router")


@dataclass(frozen=True)
class RouterRequest:
    prompt: str          # the user's text (estimate.prompt.unwrap_chat_prompt), as v3 handed it to routers
    request_id: str


@dataclass(frozen=True)
class RouterDecision:
    model: str           # a pool member, or ALL_INSTANCES
    score: float
    reason: str


class Router(ABC):
    @abstractmethod
    async def choose_model(self, request: RouterRequest, pool: Sequence[str]) -> RouterDecision:
        """`pool` = the fleet's models, unique, in fleet order."""


def _require_pool(pool: Sequence[str]) -> None:
    if not pool:
        raise ValueError("model pool is empty")


async def _off_loop(executor: Executor, fn: Callable, *args):
    return await asyncio.get_running_loop().run_in_executor(executor, fn, *args)


class PassthroughRouter(Router):
    async def choose_model(self, request: RouterRequest, pool: Sequence[str]) -> RouterDecision:
        _require_pool(pool)
        return RouterDecision(ALL_INSTANCES, 1.0 / len(pool), "passthrough:no_routing")


# ---------------------------------------------------------------- BEST-Route

Classify = Callable[[str], tuple[int, float]]   # prompt -> (argmax label, its softmax probability)


class BestRouteRouter(Router):
    """v3 decision order: a label outside the pool goes to the fallback even at threshold 0; then a
    probability strictly below the threshold goes to the fallback. The fallback is never inferred."""

    def __init__(self, classify: Classify, label_to_model: Mapping[int, str], *, threshold: float,
                 fallback: Optional[str], executor: Executor):
        if threshold > 0 and fallback is None:
            raise ConfigError(f"best_route: confidence_threshold={threshold} needs an explicit fallback_model")
        self._classify = classify
        self._label_to_model = dict(label_to_model)
        self._threshold = threshold
        self._fallback = fallback
        self._executor = executor

    def _fallback_in(self, pool: Sequence[str], why: str) -> str:
        if self._fallback is None:
            raise RuntimeError(f"best_route: {why} needs an explicit fallback_model")
        if self._fallback not in pool:
            raise RuntimeError(f"best_route: fallback_model {self._fallback!r} is not in the pool {list(pool)}")
        return self._fallback

    async def choose_model(self, request: RouterRequest, pool: Sequence[str]) -> RouterDecision:
        _require_pool(pool)
        label, prob = await _off_loop(self._executor, self._classify, request.prompt)
        chosen = self._label_to_model.get(label)
        if chosen is None or chosen not in pool:
            fb = self._fallback_in(pool, "pool_miss")
            return RouterDecision(fb, prob, f"best_route_4way:pool_miss:label={label}:fallback={fb}")
        if self._threshold > 0 and prob < self._threshold:
            fb = self._fallback_in(pool, "low_conf")
            return RouterDecision(fb, prob, f"best_route_4way:low_conf:p={prob:.3f}<{self._threshold}:fallback={fb}")
        return RouterDecision(chosen, prob, f"best_route_4way:argmax_label={label}:p={prob:.3f}")


class _DebertaClassifier:
    """The checkpoint exactly as v3 ran it: slow SentencePiece tokenizer, one unpadded prompt truncated
    at max_length, float32 on the CPU (the scheduler node's GPU serves a model instance)."""

    def __init__(self, checkpoint: Path, max_length: int):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self._torch = torch
        self._max_length = max_length
        self._tok = AutoTokenizer.from_pretrained(str(checkpoint), use_fast=False)
        self._model = AutoModelForSequenceClassification.from_pretrained(str(checkpoint)).to("cpu").eval()

    def __call__(self, prompt: str) -> tuple[int, float]:
        enc = self._tok(prompt, max_length=self._max_length, truncation=True, padding=False, return_tensors="pt")
        with self._torch.no_grad():
            probs = self._torch.softmax(self._model(**enc).logits.squeeze(0), dim=-1)
        top = int(self._torch.argmax(probs).item())   # first maximal label on a tie
        return top, float(probs[top].item())


def load_best_route(checkpoint: Path, max_length: int) -> tuple[Classify, dict[int, str]]:
    if not (checkpoint / "config.json").is_file():
        raise ConfigError(f"best_route: no checkpoint at {checkpoint}")
    results = checkpoint / "training_results.json"
    try:
        label_to_model = loads_strict(results.read_text())["label_to_model"]
    except (OSError, KeyError, TypeError, ConfigError) as e:
        raise ConfigError(f"best_route: no label_to_model in {results}: {e!r}") from e
    if not isinstance(label_to_model, dict) or not label_to_model:
        raise ConfigError(f"best_route: label_to_model in {results} must be a non-empty object")
    return _DebertaClassifier(checkpoint, max_length), {int(k): str(v) for k, v in label_to_model.items()}


# ---------------------------------------------------------------- AvengersPro

RouteBatch = Callable[[list], list]   # upstream route_queries_batch: [prompt] -> [[ranked model, ...]]


class AvengersProRouter(Router):
    def __init__(self, route_batch: RouteBatch, *, executor: Executor):
        self._route_batch = route_batch
        self._executor = executor

    async def choose_model(self, request: RouterRequest, pool: Sequence[str]) -> RouterDecision:
        _require_pool(pool)
        try:
            results = await _off_loop(self._executor, self._route_batch, [request.prompt])
        except Exception as e:
            raise RuntimeError(f"avengers_pro: upstream routing failed ({type(e).__name__}: {e})") from e
        if not results or not results[0]:
            raise RuntimeError("avengers_pro: upstream returned an empty ranking")
        for name in results[0]:
            if name in pool:
                return RouterDecision(name, 1.0, f"avengers_pro:upstream:rank=1:model={name}")
        raise RuntimeError(f"avengers_pro: none of upstream's models {results[0]} is in the pool {list(pool)}")


class _LocalEmbedder:
    """Stands in for upstream's HTTP embedding client: a local sentence-transformers model pinned to
    the CPU and to one Hub revision, with the one method upstream calls, get(text) -> 1-D embedding (v3 D9)."""

    def __init__(self, name: str, revision: str):
        from sentence_transformers import SentenceTransformer
        self._st = SentenceTransformer(name, device="cpu", revision=revision)

    def get(self, query: str):
        return self._st.encode([query], normalize_embeddings=False)[0]


UPSTREAM_MODULES = ("simple_cluster_router", "config", "embedding_cache")


def upstream_digest(repo: Path) -> str:
    """sha256 over the upstream modules the router imports (name and sha256 of each file, in a fixed order):
    the pinned checkout, local patches included, is exactly what runs."""
    lines = []
    for name in UPSTREAM_MODULES:
        path = Path(repo) / f"{name}.py"
        try:
            lines.append(f"{name}.py {hashlib.sha256(path.read_bytes()).hexdigest()}")
        except OSError as e:
            raise ConfigError(f"avengers_pro: cannot read upstream module {path}: {e}") from e
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def _import_upstream(repo: Path, digest: str):
    """Upstream is a flat checkout: simple_cluster_router imports `config` and `embedding_cache` as
    top-level modules. Refuse to run if the checkout is not the pinned one, or if any module resolved to a
    file outside `repo`. Upstream's progress bars (one per routed request) are silenced."""
    repo = repo.resolve()
    if not (repo / "simple_cluster_router.py").is_file():
        raise ConfigError(f"avengers_pro: no upstream checkout at {repo}")
    found = upstream_digest(repo)
    if found != digest:
        raise ConfigError(f"avengers_pro: the upstream checkout at {repo} has digest {found}, the config pins {digest}")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        scr = importlib.import_module("simple_cluster_router")
    except ImportError as e:
        raise ConfigError(f"avengers_pro: cannot import upstream from {repo}: {e}") from e
    for name in UPSTREAM_MODULES:
        where = Path(sys.modules[name].__file__).resolve().parent
        if where != repo:
            raise ConfigError(f"avengers_pro: module {name!r} was imported from {where}, not {repo}")
    if not isinstance(scr.tqdm, functools.partial):
        scr.tqdm = functools.partial(scr.tqdm, disable=True)
    return scr


def _check_recorded(artifact: Mapping, key: str, ours: object) -> None:
    if key in artifact and artifact[key] != ours:
        raise ConfigError(f"avengers_pro: artifact records {key}={artifact[key]!r}, config says {ours!r}")


def _rankings(raw: object, models: Sequence[str]) -> dict[int, dict[str, list[str]]]:
    """cluster id -> {"ranking": [model, ...]}, from either layout v3 accepted (a list, or {"ranking": list}).
    Every ranking must be non-empty and name only artifact models: upstream scores an unranked model 0,
    so empty or misnamed rankings would route every request to the first model without an error."""
    if not isinstance(raw, dict) or not raw:
        raise ConfigError("avengers_pro: cluster_rankings.json must be a non-empty object")
    out = {}
    for k, v in raw.items():
        ranking = v["ranking"] if isinstance(v, dict) and "ranking" in v else v
        if not isinstance(ranking, list) or not ranking or not set(ranking) <= set(models):
            raise ConfigError(f"avengers_pro: cluster {k} needs a non-empty ranking of artifact models, got {ranking!r}")
        out[int(k)] = {"ranking": list(ranking)}
    return out


def load_avengers_pro(artifact_dir: Path, upstream_repo: Path, *, beta: float, top_k: int, embedder: str,
                      embedder_revision: str, upstream_sha256: str):
    """Build upstream's SimpleClusterRouter from an artifact directory, as v3 did.

    Upstream's __init__ opens an HTTP embedding client, so the object is built with __new__ and its
    route-time attributes are set directly. top_k and beta are routing parameters: upstream's own
    defaults (top_k 1, beta 9.0) never apply; v3 ran top_k 4 (Qwen artifacts) and beta 5.0. The embedder
    is loaded at the pinned Hub revision, and the upstream checkout must have the pinned digest.
    """
    paths = {n: artifact_dir / n for n in ("config.json", "kmeans.joblib", "cluster_rankings.json")}
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise ConfigError(f"avengers_pro: artifact files missing: {missing}")
    art = loads_strict(paths["config.json"].read_text())
    models = art.get("models") if isinstance(art, dict) else None
    if not isinstance(models, list) or not models or not all(isinstance(m, str) for m in models):
        raise ConfigError(f"avengers_pro: {paths['config.json']} needs a non-empty 'models' list")
    for key, ours in (("top_k", top_k), ("beta", beta), ("max_router", 1),
                      ("embedding_model", embedder), ("embedder", embedder)):
        _check_recorded(art, key, ours)
    scr = _import_upstream(upstream_repo, upstream_sha256)
    import joblib
    import numpy as np
    import tiktoken
    from sklearn.preprocessing import Normalizer

    fields = {f.name for f in dataclasses.fields(scr.SimpleClusterConfig)}
    kwargs = {k: v for k, v in art.items() if k in fields}
    # max_router 1: the wrapper returns upstream's single best model. data_path is a training input that
    # upstream's config validation requires to exist; nothing reads it at route time.
    kwargs.update(top_k=top_k, beta=beta, max_router=1, data_path=str(paths["config.json"]))
    up = scr.SimpleClusterRouter.__new__(scr.SimpleClusterRouter)
    up.config = scr.SimpleClusterConfig(**kwargs)
    up.logger = logging.getLogger("SimpleClusterRouter")
    up.tokenizer = tiktoken.get_encoding("cl100k_base")    # upstream truncates prompts past max_tokens with it
    up.embedder = _LocalEmbedder(embedder, embedder_revision)
    up.kmeans_model = joblib.load(paths["kmeans.joblib"])
    up.cluster_centers = up.kmeans_model.cluster_centers_
    norm = artifact_dir / "normalizer.joblib"
    if norm.is_file():
        up.normalizer = joblib.load(norm)
    else:   # L2 normalisation is stateless; fitting only satisfies sklearn (the Gemma artifact ships none)
        up.normalizer = Normalizer(norm="l2").fit(np.zeros((1, up.cluster_centers.shape[1])))
    up.cluster_rankings = _rankings(loads_strict(paths["cluster_rankings.json"].read_text()), models)
    if set(up.cluster_rankings) != set(range(len(up.cluster_centers))):   # upstream skips an unranked cluster
        raise ConfigError(f"avengers_pro: rankings cover clusters {sorted(up.cluster_rankings)}, "
                          f"k-means has {len(up.cluster_centers)}")
    up.available_models = list(models)
    # One call before any concurrent use: the embedder's fast tokenizer sets its padding/truncation
    # state on first use, which must not race; it also fails construction, not a request, if broken.
    probe = up.route_queries_batch(["construction probe"])
    if not probe or not probe[0]:
        raise ConfigError("avengers_pro: upstream returned no model for the construction probe")
    return up


# ---------------------------------------------------------------- RouteBalance, decoupled

class Estimate(NamedTuple):
    quality: float        # predicted quality of this prompt on the model
    expected_len: float   # predicted output tokens on the model


EstimateBatch = Callable[[Sequence[str]], Sequence[Mapping[str, Estimate]]]   # one {model: Estimate} per prompt
UnloadedLatency = Callable[[Tier, float], float]                            # (tier, output tokens) -> s, idle instance

@dataclass(frozen=True)
class DecoupledInputs:
    """RouteBalance's components that rb_decoupled reuses: the same estimator, latency model and weights."""
    tiers: Sequence[Tier]              # the fleet's tiers in fleet order
    weights: Weights
    normalization: Literal["fixed_ref", "max_divide"]
    estimate: EstimateBatch
    unloaded_latency: UnloadedLatency


class RBDecoupledRouter(Router):
    """Scores every tier of a pool model with RouteBalance's objective, but with the latency of an
    idle instance, and returns the best tier's model. The first tier in fleet order wins exact ties."""

    def __init__(self, inputs: DecoupledInputs, *, executor: Executor):
        if inputs.normalization not in ("fixed_ref", "max_divide"):
            raise ConfigError(f"rb_decoupled: unknown normalization {inputs.normalization!r}")
        ids = [t.tier_id for t in inputs.tiers]
        if not ids or len(set(ids)) != len(ids):
            raise ConfigError(f"rb_decoupled: tiers must be non-empty and unique, got {ids}")
        self._in = inputs
        self._executor = executor

    def _decide(self, prompt: str, pool: Sequence[str]) -> tuple[Tier, float]:
        tiers = [t for t in self._in.tiers if t.model in pool]
        if not tiers:
            raise ValueError(f"rb_decoupled: no tier serves a pool model {list(pool)}")
        (est,) = self._in.estimate([prompt])
        missing = sorted({t.model for t in tiers} - set(est))
        if missing:
            raise RuntimeError(f"rb_decoupled: the estimator returned nothing for {missing}")
        quality = [est[t.model].quality for t in tiers]
        lens = [est[t.model].expected_len for t in tiers]
        latency = [self._in.unloaded_latency(t, n) for t, n in zip(tiers, lens, strict=True)]
        if not all(math.isfinite(x) for x in (*quality, *lens, *latency)) or min(lens) < 0 or min(latency) <= 0:
            raise RuntimeError(f"rb_decoupled: need finite quality, length >= 0 and latency > 0, got "
                               f"{quality} / {lens} / {latency}")
        # RouteBalance's own objective (schedule/score.py) over tiers, every latency that of an idle instance;
        # no queue or recency: the first tier in fleet order wins exact ties
        cands = [Candidate(t.tier_id, t.model, q, n * t.cost_out, lat, lat, 0, 0.0)   # decode-only cost, as RB
                 for t, q, n, lat in zip(tiers, quality, lens, latency, strict=True)]
        w = self._in.weights
        scored = score(cands, w_quality=w.w_quality, w_cost=w.w_cost, w_latency=w.w_latency,
                       normalization=self._in.normalization)
        best = pick(scored)
        return tiers[scored.index(best)], best.score

    async def choose_model(self, request: RouterRequest, pool: Sequence[str]) -> RouterDecision:
        _require_pool(pool)
        tier, score = await _off_loop(self._executor, self._decide, request.prompt, tuple(pool))
        return RouterDecision(tier.model, score, f"rb_decoupled:tier={tier.tier_id}:score={score:.6f}")


# ---------------------------------------------------------------- construction from config

class _Options(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)


class _NoOptions(_Options):
    pass


class _BestRouteOptions(_Options):
    checkpoint: str                                      # relative to the artifact root
    confidence_threshold: float = Field(ge=0, lt=1)      # 0 = pure argmax (with 4 labels, anything <= 0.25 is)
    fallback_model: Optional[str] = None                 # absent: no fallback, so a pool miss raises
    max_length: int = Field(ge=1)                        # 512 = training length = the model's positions

    @field_validator("checkpoint")
    @classmethod
    def _checkpoint_is_relative(cls, v: str) -> str:
        return _relative_path(v)

    @model_validator(mode="after")
    def _fallback_for_threshold(self):   # checked before the checkpoint is loaded (v3 order)
        if self.confidence_threshold > 0 and self.fallback_model is None:
            raise ValueError("confidence_threshold > 0 needs an explicit fallback_model")
        return self


class _AvengersProOptions(_Options):
    artifact_dir: str                    # relative to the artifact root
    beta: float = Field(gt=0)            # v3: 5.0
    top_k: int = Field(ge=1)             # v3: 4 for the Qwen artifacts (recorded there), 3 for Gemma
    embedder: str                        # sentence-transformers model the artifact was clustered with
    embedder_revision: str = Field(pattern=r"^[0-9a-f]{40}$")   # its Hub commit
    upstream_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")     # upstream_digest() of the pinned checkout

    @field_validator("artifact_dir")
    @classmethod
    def _artifact_dir_is_relative(cls, v: str) -> str:
        return _relative_path(v)


def _options(model: type[_Options], cfg: RouterConfig) -> _Options:
    try:
        return model.model_validate(cfg.options())
    except ValidationError as e:
        raise ConfigError(f"invalid {cfg.type} router kwargs: {e}") from e


_OPTIONS = {"passthrough": _NoOptions, "best_route": _BestRouteOptions, "avengers_pro": _AvengersProOptions,
            "rb_decoupled": _NoOptions}


def validate_options(cfg: RouterConfig) -> None:
    """Check a router's kwargs without loading anything (the campaign render checks every configuration)."""
    if cfg.type not in _OPTIONS:
        raise ConfigError(f"unknown router type {cfg.type!r}")
    _options(_OPTIONS[cfg.type], cfg)


def build_router(cfg: RouterConfig, *, artifact_root: Path, avengers_repo: Optional[Path], executor: Executor,
                 decoupled: Optional[DecoupledInputs]) -> Router:
    """Build a router from its config. Loading models takes seconds: call this off the event loop."""
    if cfg.type == "passthrough":
        _options(_NoOptions, cfg)
        return PassthroughRouter()
    if cfg.type == "best_route":
        br = _options(_BestRouteOptions, cfg)
        classify, labels = load_best_route(artifact_root / br.checkpoint, br.max_length)
        return BestRouteRouter(classify, labels, threshold=br.confidence_threshold, fallback=br.fallback_model,
                               executor=executor)
    if cfg.type == "avengers_pro":
        ap = _options(_AvengersProOptions, cfg)
        if avengers_repo is None:
            raise ConfigError("avengers_pro needs the upstream AvengersPro checkout")
        up = load_avengers_pro(artifact_root / ap.artifact_dir, avengers_repo, beta=ap.beta, top_k=ap.top_k,
                               embedder=ap.embedder, embedder_revision=ap.embedder_revision,
                               upstream_sha256=ap.upstream_sha256)
        return AvengersProRouter(up.route_queries_batch, executor=executor)
    if cfg.type == "rb_decoupled":
        _options(_NoOptions, cfg)
        if decoupled is None:
            raise ConfigError("rb_decoupled needs RouteBalance's estimator and latency model")
        return RBDecoupledRouter(decoupled, executor=executor)
    raise ConfigError(f"unknown router type {cfg.type!r}")
