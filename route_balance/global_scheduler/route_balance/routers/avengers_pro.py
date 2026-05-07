"""Avengers-Pro router (DAI'25 Best Paper, arXiv:2508.12631) — wrapper
around the upstream ``SimpleClusterRouter``.

We import ``SimpleClusterRouter`` from a checkout of the AvengersPro repo
(path supplied via the ``ROUTE_BALANCE_AVENGERS_REPO`` env var, default
``$HOME/AvengersPro``) and delegate routing decisions to it via its
``route_queries_batch([prompt])`` API.

The one necessary deviation: upstream's ``__init__`` hard-depends on an
OpenAI-compatible embedding HTTP service (``EmbeddingCache``). To run
without a remote embedding API we:

    1. Bypass their ``__init__`` (use ``__new__`` + manual attribute setup).
    2. Inject a local ``sentence-transformers`` embedder shim that exposes
       the same method upstream calls (``get_embeddings(list[str]) ->
       list[np.ndarray]``).
    3. Load the trained artifact (k-means / rankings / normalizer /
       available_models) from a directory produced by upstream's
       ``export_cluster_models()`` or our own retraining script.

Runtime flow: ``choose_model(prompt)`` calls upstream's
``route_queries_batch([prompt])`` → list of top-K models per query → take
the first query's first model → map to our runtime pool.
"""
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

from .base import RouterBase, RouterDecision, RouterRequest


logger = logging.getLogger(__name__)

_AVENGERS_REPO = os.environ.get(
    "ROUTE_BALANCE_AVENGERS_REPO",
    str(Path.home() / "AvengersPro"),
)


def _ensure_upstream_on_path() -> None:
    if _AVENGERS_REPO not in sys.path and Path(_AVENGERS_REPO).exists():
        sys.path.insert(0, _AVENGERS_REPO)


class _LocalEmbedderShim:
    """Thin shim giving the sentence-transformers encoder the interface
    upstream's SimpleClusterRouter expects from EmbeddingCache."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self._st = SentenceTransformer(model_name)
        self._model_name = model_name

    def get_embeddings(self, queries: List[str], **kwargs):
        # Upstream stores the result as a list of np.ndarray. Keep shape.
        embs = self._st.encode(queries, normalize_embeddings=False)
        return [embs[i] for i in range(len(queries))]


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
        # Fill remaining with safe defaults where possible.
        cfg = SimpleClusterConfig(**cfg_kwargs)

        upstream.config = cfg
        upstream.logger = logging.getLogger("SimpleClusterRouter")

        # Inject local embedder
        try:
            upstream.embedder = _LocalEmbedderShim(self._embedder_name)
        except Exception as e:
            logger.warning("Local embedder init failed: %s", e)
            return False

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
            results = self._upstream.route_queries_batch([req.prompt])
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
