"""Learned TPOT: one XGBoost booster per tier over what the scheduler knows at serving time.

Serving runs official vLLM, so an instance's state is the latency.InstanceState protocol: num_running and
num_waiting (Prometheus /metrics), kv_cache_utilization (the KV-cache usage gauge), max_num_seqs (the
deployment config) and pending_decode_tokens (the scheduler's own sum of remaining predicted decode tokens
of its in-flight requests). A head may read only FEATURES: those state fields, the request's prompt
tokens and expected output tokens, and the two features derived from the request alone. The heads
come from tools/retrain_xgb_served.py, which builds its training vectors with feature_values() below, so
training and serving vectors are identical by construction; target TPOT in seconds per token.

Model dir: feature_names.json and <tier>.xgb, with md5s.json {file: md5} pinning every file (the config
pins md5s.json itself). A booster file is named by the fleet tier_id it serves.
"""
from __future__ import annotations

import math
import threading
from pathlib import Path
from typing import Collection, Mapping, Sequence

import numpy as np
import xgboost as xgb

from fmrlb.config import loads_strict
from fmrlb.estimate.artifacts import ArtifactError, read_verified
from fmrlb.estimate.features import FEATURES, feature_values   # noqa: F401  (re-exported)
from fmrlb.estimate.latency import InstanceState

TPOT_FLOOR_S = 0.001       # every booster output is clamped to at least 1 ms/token
MANIFEST = "md5s.json"


def feature_vector(names: Sequence[str], state: InstanceState, num_prompt_tokens: int, output_tokens: int) -> np.ndarray:
    values = feature_values(state, num_prompt_tokens, output_tokens)
    return np.array([values[n] for n in names], dtype=np.float32)


class XGBoostTPOT:
    kind = "xgboost"

    def __init__(self, boosters: Mapping[str, xgb.Booster], feature_names: Sequence[str]):
        names = tuple(feature_names)
        unknown = sorted(set(names) - set(FEATURES))
        if unknown:
            raise ArtifactError(f"TPOT features {unknown} cannot be computed at serving time")
        if not names:
            raise ArtifactError("a TPOT head needs at least one feature")
        self.feature_names = names
        self._boosters = dict(boosters)
        self._lock = threading.Lock()
        self._calls = 0
        self.files: dict[str, str] = {}      # {file: md5} of every file load() read (the execution manifest)

    @property
    def tiers(self) -> frozenset[str]:
        return frozenset(self._boosters)

    @property
    def predict_calls(self) -> int:
        return self._calls

    def booster(self, tier_id: str) -> xgb.Booster:
        return self._boosters[tier_id]

    @classmethod
    def load(cls, model_dir: Path, file_md5: Mapping[str, str], tier_ids: Collection[str]) -> "XGBoostTPOT":
        """Heads for `tier_ids`; `file_md5` pins every file that is read."""
        model_dir = Path(model_dir)

        def read(name: str) -> bytes:
            if name not in file_md5:
                raise ArtifactError(f"{model_dir}: {name} has no pinned md5")
            return read_verified(model_dir / name, file_md5[name])

        names = loads_strict(read("feature_names.json").decode("utf-8"))
        if not (isinstance(names, list) and all(isinstance(n, str) for n in names)):
            raise ArtifactError(f"{model_dir}: feature_names.json must be a list of names")
        absent = sorted(t for t in set(tier_ids) if f"{t}.xgb" not in file_md5)
        if absent:
            raise ArtifactError(f"{model_dir}: no TPOT head for fleet tiers {absent}")
        boosters = {}
        for tier in sorted(set(tier_ids)):
            booster = xgb.Booster()
            booster.load_model(bytearray(read(f"{tier}.xgb")))
            if booster.num_features() != len(names) or booster.feature_names != names:   # names fix the column order
                raise ArtifactError(f"{model_dir}/{tier}.xgb was not trained on the listed features")
            boosters[tier] = booster
        out = cls(boosters, names)
        out.files = {n: file_md5[n] for n in ("feature_names.json", *(f"{t}.xgb" for t in sorted(set(tier_ids))))}
        return out

    @classmethod
    def from_manifest(cls, model_dir: Path, manifest_md5: str, tier_ids: Collection[str]) -> "XGBoostTPOT":
        """The configured form: the pinned md5 is that of md5s.json, which pins every other file."""
        file_md5 = loads_strict(read_verified(Path(model_dir) / MANIFEST, manifest_md5).decode("utf-8"))
        if not (isinstance(file_md5, dict) and all(isinstance(v, str) for v in file_md5.values())):
            raise ArtifactError(f"{model_dir}/{MANIFEST} must map file names to md5s")
        out = cls.load(model_dir, file_md5, tier_ids)
        out.files = {MANIFEST: manifest_md5, **out.files}
        return out

    def features(self, state: InstanceState, num_prompt_tokens: int, output_tokens: int) -> np.ndarray:
        return feature_vector(self.feature_names, state, num_prompt_tokens, output_tokens)

    def tpots(self, tier_ids: Sequence[str], states: Sequence[InstanceState], num_prompt_tokens: int,
              output_tokens: int) -> list[float]:
        """TPOT (s/token) per candidate: one booster call per tier over that tier's candidates."""
        rows: dict[str, list[int]] = {}
        for i, tier in enumerate(tier_ids):
            if tier not in self._boosters:
                raise ArtifactError(f"no TPOT head for tier {tier!r}")
            if tier not in rows:
                rows[tier] = []
            rows[tier].append(i)
        out: dict[int, float] = {}
        for tier, members in rows.items():
            x = np.stack([self.features(states[i], num_prompt_tokens, output_tokens) for i in members])
            with self._lock:
                self._calls += 1
            for i, p in zip(members, self._boosters[tier].inplace_predict(x), strict=True):
                if not math.isfinite(float(p)):
                    raise ArtifactError(f"the TPOT head of tier {tier!r} predicted {float(p)!r}")
                out[i] = max(TPOT_FLOOR_S, float(p))
        return [out[i] for i in range(len(tier_ids))]
