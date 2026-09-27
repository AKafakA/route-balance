"""The v3 source files and artifacts the estimator tests compare with.

Each is found under the oracle root by file name and pinned by md5, so a test can only pass against the exact
v3 file (the v3 tree's directory names never appear in the repository). The v3 modules are imported by path;
their own `block.*` imports resolve through <root>/Block.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from functools import lru_cache
from pathlib import Path

V3_SOURCES = {                        # key -> (file name pattern under <root>/Block/block, md5 at v3 head)
    "knn": ("knn_estimator.py", "d17e5e239ffda84fddd59acad4860846"),
    "model_estimator": ("model_estimator.py", "c702ee99b83dc88af152db2cd4974c98"),
    "xgboost": ("xgboost_predictor.py", "7a831be8b53aba0220f06cab2b89dc09"),
    "serve": ("*_serve.py", "2144a2f9ae660c6f9b2d1d51e8d6e2e1"),
}
KNN_K50 = "d2868fdc21b26b9e75dbb79d634dda19"      # main v3 index, 14,919 prompts, k=50
KNN_F50 = "0467b12e8bca35a13f213075d9e0cd68"      # 50% labels, 7,459 prompts, k=50
TRAIN_SCORED = ("train_scored_filtered.jsonl", "0171225264aa4746af255143a5054fcf")
LATENCY_TEST = ("latency_test_tagged.jsonl", "f7b9561628508df69b3c5c95e5f7b02f")
Q1_STATIC_DECISIONS = ("s2_q1_PROD_zerocall", "q1_static_s1_l12_n3534.decisions.jsonl",
                       "c7e7b478f1c237fdf259bb316f4e1195")
V3_TPOT_METADATA = "8d66b0e33f70a4eb5b788ee79d6b46d0"
V3_TPOT_MD5 = {                       # the v3 TPOT heads the retrained ones are compared with
    "qwen2.5-14b_v100.xgb": "63b1129ffe690a3b67e422e79ed24d6c",
    "qwen2.5-3b_a30.xgb": "c5caf54d5dc04d238d0126240f92390c",
    "qwen2.5-72b_a100.xgb": "da715c2cb7a3f22302c753383acf8285",
    "qwen2.5-7b_a30.xgb": "45eaaf9895608892333004cf0ff04d21",
}
V3_TIERS = ("qwen2.5-14b_v100", "qwen2.5-3b_a30", "qwen2.5-72b_a100", "qwen2.5-7b_a30")
ENCODER_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"   # all-MiniLM-L6-v2 as pinned for the v3 reruns
# routers.upstream_digest of the upstream AvengersPro checkout: commit dfb60b1 plus the local patch that makes its
# training-only `datasets` import optional; configs/presets.json pins the same value
UPSTREAM_SHA256 = "6cfaf58f525217b47c2e90361fbb3e94309f66fbbc7eadef067e01c82946493b"
# The rebuild's delivered artifacts under --artifact-root, pinned like any other (regenerating one means
# updating its pin here and in the config that uses it).
SERVED_HEADS = "xgb_served"                                     # built by the removed --v3-sweep mode (998421e)
SERVED_HEADS_MD5 = "1bfe49bb364e2c76c29fbd750a53a73f"                  # md5 of its md5s.json (the config's table_md5)
TPOT_TABLE = "tpot_table_v3cells_d47.json"                      # banked_cell_records.py + build_table, with the
TPOT_TABLE_MD5 = "3500d6467dc3cae802cc783356bc4252"             # D47 running max (the pre-D47 table: 7bf3dd4a...)


def md5_of(path: Path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


@lru_cache(maxsize=None)
def find_one(base: Path, pattern: str, md5: str) -> Path:
    found = [p for p in sorted(Path(base).rglob(pattern)) if md5_of(p) == md5]
    assert len(found) == 1, f"expected one {pattern} with md5 {md5} under {base}, found {found}"
    return found[0]


def v3_source(oracle, key: str) -> Path:
    name, md5 = V3_SOURCES[key]
    return find_one(oracle.root / "Block" / "block", name, md5)


def v3_module(oracle, key: str):
    block_root = str(oracle.root / "Block")
    if block_root not in sys.path:
        sys.path.insert(0, block_root)
    return _import(v3_source(oracle, key), f"oracle_{key}")


@lru_cache(maxsize=None)
def _import(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def knn_dir(oracle, md5: str) -> Path:
    return find_one(oracle.root, "knn_estimator.pkl", md5).parent


def tpot_dir(oracle) -> Path:
    return find_one(oracle.root / "Block" / "models", "xgboost_metadata.json", V3_TPOT_METADATA).parent


def q1_static_decisions(oracle) -> Path:
    cells, name, md5 = Q1_STATIC_DECISIONS
    path = oracle.cells_dir(cells) / name
    assert md5_of(path) == md5, f"{path} is not the banked v3 decision log"
    return path


def read_jsonl(path: Path) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
