"""Avengers-Pro router (DAI'25 Best Paper, arXiv 2508.12631) — A3.6 / #31
(refactored April 15, 2026 / task #36 to WRAP upstream SimpleClusterRouter).

Wrapper design
--------------
We import `SimpleClusterRouter` from the cloned AvengersPro repo at
`$AVENGERS_REPO` (or `~/AvengersPro`) and delegate routing decisions to it
via its `route_queries_batch([prompt])` API.

The one necessary deviation: their `__init__` hard-depends on an
OpenAI-compatible embedding HTTP service (`EmbeddingCache`). Our A30
cluster nodes have no API access. We therefore:

    1. Bypass their __init__ (use `__new__` + manual attribute setup).
    2. Inject a local sentence-transformers embedder object that exposes
       the same method they call (`get_embeddings(list[str]) ->
       list[np.ndarray]`).
    3. Load the trained artifact (kmeans / rankings / normalizer /
       available_models) from a directory produced by upstream's
       `export_cluster_models()` OR our own retraining script.

Runtime flow: `choose_model(prompt)` calls their
`route_queries_batch([prompt])` → list of top-K models per query → take
first query's first model → map to our runtime pool.

Training / artifact generation lives in
`route_balance_paper/smoke_test_apr_13/scripts/retrain_avengers_pro_qwen.py` and
is run OUT OF BAND on node0.
"""
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

from .base import RouterBase, RouterDecision, RouterRequest


logger = logging.getLogger(__name__)

_AVENGERS_REPO = next(
    (p for p in (
        str(Path.home() / "AvengersPro"),
    ) if Path(p).exists()),
    "${HOME}/AvengersPro",
)


def _ensure_upstream_on_path() -> None:
    if _AVENGERS_REPO not in sys.path and Path(_AVENGERS_REPO).exists():
        sys.path.insert(0, _AVENGERS_REPO)


class _LocalEmbedderShim:
    """Thin shim giving the sentence-transformers encoder the interface
    upstream's SimpleClusterRouter expects from EmbeddingCache."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        # Pin to CPU. The scheduler is co-located with a GPU-resident LLM
        # (e.g. 72B TP=4 on the A100 node), so the default CUDA device is
        # saturated -> SentenceTransformer load raises CUDA OOM. That OOM is
        # swallowed by _try_load (returns False), which silently degrades the
        # router to _fallback_choice = max(pool, by size) = the LARGEST model,
        # i.e. an all-72B flood. MiniLM is tiny and sub-ms/query on CPU, and
        # the embeddings are numerically identical, so routing is unchanged.
        self._st = SentenceTransformer(model_name, device="cpu")
        self._model_name = model_name

    def get_embeddings(self, queries: List[str], **kwargs):
        # Upstream (older API) stores the result as a list of np.ndarray.
        embs = self._st.encode(queries, normalize_embeddings=False)
        return [embs[i] for i in range(len(queries))]

    def get(self, query: str, **kwargs):
        # Upstream (current API, simple_cluster_router.py:_get_embedding_batch)
        # calls embedder.get(single_query) -> 1D embedding. The EmbeddingCache
        # this shim replaces exposed BOTH .get (single) and batch helpers; we
        # mirror both so the wrapper survives upstream interface drift.
        emb = self._st.encode([query], normalize_embeddings=False)
        return emb[0]


class AvengersProRouter(RouterBase):
    """Wraps upstream SimpleClusterRouter with a local-embedder injection."""

    def __init__(
        self,
        *,
        artifact_dir: str = "models/route_balance/avengers_pro_qwen",
        embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        fallback_model: Optional[str] = None,
        upstream_repo: str = _AVENGERS_REPO,
    ):
        """
        Args:
            artifact_dir: Directory with
                - kmeans.joblib
                - cluster_rankings.json    (int cluster_id → {"ranking": [...]} or [...])
                - normalizer.joblib        (optional, defaults to L2 normalizer)
                - config.json              (has "models" list + n_clusters etc.)
            embedder_model: HF sentence-transformer name.
            fallback_model: Fallback when artifact missing / upstream fails.
            upstream_repo: Path to cloned AvengersPro repo.
        """
        self._artifact_dir = Path(artifact_dir)
        self._embedder_name = embedder_model
        self._fallback = fallback_model
        self._upstream_repo = upstream_repo
        self._upstream = None  # SimpleClusterRouter instance (lazy)
        self._loaded = self._try_load()

    def _try_load(self) -> bool:
        # Require artifact presence
        km_path = self._artifact_dir / "kmeans.joblib"
        rank_path = self._artifact_dir / "cluster_rankings.json"
        cfg_path = self._artifact_dir / "config.json"
        if not (km_path.exists() and rank_path.exists() and cfg_path.exists()):
            return False

        # Import upstream
        if not Path(self._upstream_repo).exists():
            logger.warning(
                "AvengersPro repo not at %s; wrapper disabled.",
                self._upstream_repo,
            )
            return False
        _ensure_upstream_on_path()

        try:
            import joblib
            from simple_cluster_router import SimpleClusterRouter, SimpleClusterConfig  # noqa
        except Exception as e:
            logger.warning(
                "Could not import upstream SimpleClusterRouter: %s; "
                "falling back to disabled router.", e,
            )
            return False

        # Build an upstream instance that bypasses EmbeddingCache init
        upstream = SimpleClusterRouter.__new__(SimpleClusterRouter)
        from dataclasses import fields as _fields
        # Minimal config just to satisfy upstream attribute reads.
        cfg_dict = json.loads(cfg_path.read_text())
        field_names = {f.name for f in _fields(SimpleClusterConfig)}
        cfg_kwargs = {
            k: v for k, v in cfg_dict.items() if k in field_names
        }
        # Required fields upstream reads at route-time.
        cfg_kwargs.setdefault("top_k", 3)
        cfg_kwargs.setdefault("beta", 5.0)
        cfg_kwargs.setdefault("max_router", 1)
        # Fill ALL remaining REQUIRED dataclass fields (no default) with
        # type-appropriate dummies — routing never reads training-only fields
        # like data_path / excluded_datasets, but the dataclass __init__ needs them.
        import dataclasses as _dc
        for _f in _fields(SimpleClusterConfig):
            if _f.name in cfg_kwargs:
                continue
            _has_def = (_f.default is not _dc.MISSING) or (
                _f.default_factory is not _dc.MISSING
            )
            if _has_def:
                continue
            _ann = _f.type
            cfg_kwargs[_f.name] = (
                "" if _ann in (str, "str")
                else 0 if _ann in (int, "int")
                else 0.0 if _ann in (float, "float")
                else False if _ann in (bool, "bool")
                else None
            )
        # config.__post_init__ validates data_path EXISTS — point it at an
        # existing file (never read at route-time) to pass validation.
        if not cfg_kwargs.get("data_path") or not Path(str(cfg_kwargs["data_path"])).exists():
            cfg_kwargs["data_path"] = str(cfg_path)
        cfg = SimpleClusterConfig(**cfg_kwargs)

        upstream.config = cfg
        upstream.logger = logging.getLogger("SimpleClusterRouter")

        # Inject local embedder
        try:
            upstream.embedder = _LocalEmbedderShim(self._embedder_name)
        except Exception as e:
            logger.warning("Local embedder init failed: %s", e)
            return False

        # Inject tokenizer for upstream _truncate_text (cl100k_base matches
        # upstream's own fallback at config.py / line 97). Without it the
        # router logs "no attribute 'tokenizer'" and char-truncates; with
        # max_tokens=7500 our short prompts are never truncated either way,
        # but inject it for parity + to silence the error path.
        try:
            import tiktoken
            upstream.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.warning("tiktoken init failed (%s); truncation char-fallback", e)
            upstream.tokenizer = None

        # Load artifacts
        try:
            upstream.kmeans_model = joblib.load(km_path)
            upstream.cluster_centers = upstream.kmeans_model.cluster_centers_
        except Exception as e:
            logger.warning("kmeans load failed: %s", e)
            return False

        norm_path = self._artifact_dir / "normalizer.joblib"
        if norm_path.exists():
            upstream.normalizer = joblib.load(norm_path)
        else:
            from sklearn.preprocessing import Normalizer
            upstream.normalizer = Normalizer(norm="l2")
            # Fit on a trivial vector so .transform works — L2 is stateless.
            import numpy as np
            upstream.normalizer.fit(np.zeros((1, upstream.cluster_centers.shape[1])))

        # Cluster rankings: upstream expects {int: {"ranking": [str, ...]}}
        raw = json.loads(rank_path.read_text())
        upstream.cluster_rankings = {}
        for k, v in raw.items():
            if isinstance(v, list):
                upstream.cluster_rankings[int(k)] = {"ranking": list(v)}
            elif isinstance(v, dict) and "ranking" in v:
                upstream.cluster_rankings[int(k)] = {"ranking": list(v["ranking"])}
            else:
                upstream.cluster_rankings[int(k)] = {"ranking": []}

        upstream.available_models = list(cfg_dict.get("models", []))

        self._upstream = upstream
        return True

    def _fallback_choice(self, pool: List[str]) -> str:
        if self._fallback and self._fallback in pool:
            return self._fallback
        import re

        def sz(n):
            m = re.search(r"(\d+(?:\.\d+)?)[Bb]", n)
            return float(m.group(1)) if m else 1.0

        return max(pool, key=sz)

    async def choose_model(
        self,
        req: RouterRequest,
        model_pool: List[str],
    ) -> RouterDecision:
        if not model_pool:
            raise ValueError("model_pool is empty")
        if not self._loaded:
            return RouterDecision(
                model_name=self._fallback_choice(model_pool),
                score=0.0,
                reason="avengers_pro:no_artifact:fallback",
            )

        try:
            import asyncio
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None, self._upstream.route_queries_batch, [req.prompt]
            )
        except Exception as e:
            return RouterDecision(
                model_name=self._fallback_choice(model_pool),
                score=0.0,
                reason=f"avengers_pro:upstream_err:{type(e).__name__}",
            )

        if not results or not results[0]:
            return RouterDecision(
                model_name=self._fallback_choice(model_pool),
                score=0.0,
                reason="avengers_pro:empty_result:fallback",
            )

        # Upstream returns top-K models; take first that is in our pool.
        top_models = results[0]
        for name in top_models:
            if name in model_pool:
                return RouterDecision(
                    model_name=name,
                    score=1.0,
                    reason=f"avengers_pro:upstream:rank=1:model={name}",
                )

        return RouterDecision(
            model_name=self._fallback_choice(model_pool),
            score=0.0,
            reason=f"avengers_pro:upstream_top={top_models[0]}_not_in_pool:fallback",
        )
