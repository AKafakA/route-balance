"""The real estimator (MiniLM at the pinned revision) and the evaluation prompts, loaded once per session."""
import json
from functools import lru_cache
from pathlib import Path

from routebalance.config import Estimator
from routebalance.estimate.knn import KnnEstimator, load_knn
from tests.equivalence.banked_cells import EVAL_SET
from tests.equivalence.oracle_files import ENCODER_REVISION, KNN_K50, knn_dir


@lru_cache(maxsize=None)
def _estimator(index_dir: Path, md5: str, prior: str) -> KnnEstimator:
    cfg = Estimator(index_dir=index_dir.name, index_md5=md5, encoder_revision=ENCODER_REVISION, k=50,
                    score_type="deepeval", prior=prior)
    return load_knn(cfg, index_dir.parent, fleet_models=())


def real_estimator(oracle, prior: str = "knn") -> KnnEstimator:
    return _estimator(knn_dir(oracle, KNN_K50), KNN_K50, prior)


def real_encoder(oracle):
    return real_estimator(oracle)._encoder


@lru_cache(maxsize=None)
def _prompts(path: Path) -> tuple:
    with open(path) as f:
        return tuple(json.loads(line)["prompt"] for line in f if line.strip())


def eval_prompts(oracle) -> list:
    prompts = list(_prompts(oracle.data_file(EVAL_SET)))
    assert len(prompts) == 3534
    return prompts
