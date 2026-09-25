"""Retrain the per-tier TPOT heads on what the scheduler knows at serving time.

Serving runs official vLLM, so a head may read only routebalance.estimate.xgboost_tpot.FEATURES (the
instance's running and waiting counts, KV-cache utilization and pending decode tokens, the request's prompt
and expected output tokens, and two features derived from the request alone). Two inputs:

  --sweep FILE...      the sweep the scheduler logs under official vLLM (tools/tpot_sweep.py): each record
                       carries the feature vector feature_values() built at dispatch, pending_decode_tokens
                       computed RouteBalance's way and the expected output length, so heads use all FEATURES.
  --v3-sweep TRAIN TEST
                       the v3 latency sweep (fork /instance_stats). The sweep sent every request with 1024 as
                       its predicted length, so num_predicted_output_tokens (and the two features derived from
                       it) is 1024 in every record and pending_decode_tokens counts 1024-token predictions: at
                       serving the scheduler puts its own expected lengths in both. Heads trained on this sweep
                       use V3_SWEEP_FEATURES, which leaves those four out. The training vectors are built by
                       feature_values() from the recorded state, so they are what the scheduler serves.
A head never trains on a feature that is constant in its training rows (it could learn nothing from it and
the value would only move at serving), nor on fewer than MIN_ROWS usable records (v3 skipped such a tier; here
a fleet tier without a head is an error): the tool fails instead.

Everything else is the v3 procedure (offline_training/train_xgboost.py --target tpot), which this tool
reproduces; with the v3 sweep in the order train split then test split, the v3 heads' recorded validation
MAE is reproduced exactly:
  records  per tier (instance_type / tier), in file order; failed requests and missing or non-positive targets
           are dropped
  target   TPOT in s/token (v3 sweep: the first of tpot, actual_tpot present; new sweep: tpot_s)
  split    per tier, RandomState(42).permutation; the last 20% is the validation set
  booster  reg:squarederror, max_depth 6, eta 0.1, subsample 0.8, colsample_bytree 0.8,
           min_child_weight 5, seed 42; 500 rounds, early stopping after 30 on validation RMSE.
           Predictions use every tree, as v3 served them. Only nthread is added (the host's CPU cap).
Tiers trained: v3 sweep, those the v3 dir has heads for; new sweep, every tier in the records.

Output (--out): <tier>.xgb (UBJSON), feature_names.json, metrics.json, and md5s.json (the md5 of every
other file; md5s.json's own md5 is what a config pins as latency.table_md5). It is written to <out>.partial
and renamed to <out> only after the heads reload, through the scheduler's own loader, to the trained ones.

metrics.json, per tier and split (the same train and validation rows for every model):
  served          the new heads
  v3_zero_filled  (v3 sweep only) the v3 heads on the recorded state with the five fields the fork never
                  served set to 0.0 (tools/fork_tpot_features.py), the zero-fill v3 applied at serving; the
                  1024-based inputs above are as recorded, so this isolates the zero-fill
  v3_trained      (v3 sweep only) the v3 heads on their training vector, made by v3's own
                  build_feature_vector (--v3-features)
Each gives n, MAE (s/token), MAPE (%, v3's definition |err| / max(y, 1e-6)), median APE (%) and the
under-prediction rate, on raw booster outputs as v3 reported them.

Usage (GPU VM): python -m tools.retrain_xgb_served --sweep SWEEP.jsonl --threads 6 --out DIR
                python -m tools.retrain_xgb_served --v3-sweep TRAIN.jsonl TEST.jsonl --v3-dir V3_TPOT_DIR
                    --v3-features V3/xgboost_predictor.py --threads 6 --out DIR
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import xgboost as xgb

from routebalance.estimate.xgboost_tpot import FEATURES, XGBoostTPOT, feature_vector
from tools.fork_tpot_features import FORK_FIELDS, V3_FEATURES, v3_serving_vector
from tools.tpot_sweep import read_sweep

V3_PARAMS = {"objective": "reg:squarederror", "eval_metric": ["mae", "rmse"], "max_depth": 6, "learning_rate": 0.1,
             "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5, "seed": 42}
ROUNDS, EARLY_STOP, VAL_SHARE, SPLIT_SEED = 500, 30, 0.2, 42
MIN_ROWS = 50                 # v3 skipped a tier with fewer usable records (train_xgboost.py)
V3_SWEEP_FEATURES = ("num_running", "num_waiting", "kv_cache_utilization", "num_prompt_tokens")
STATE_FIELDS = ("num_running", "num_waiting", "kv_cache_utilization", "max_num_seqs", "pending_decode_tokens")
_FALLBACK_KEYS = ("input_len", "max_tokens", "output_len")   # v3 read these when a request field was 0 or absent


def _v3_target(rec: dict) -> float | None:
    for key in ("tpot", "actual_tpot"):
        if rec.get(key) is not None:
            return float(rec[key])
    return None


def v3_sweep_rows(paths: list[Path], v3_training_vector) -> dict[str, dict[str, list]]:
    """Per tier: the served, v3 zero-filled and v3 training vectors and the targets, in file order."""
    rows: dict[str, dict[str, list]] = {}
    for path in paths:
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if "success" in rec and not rec["success"]:
                    continue
                y = _v3_target(rec)
                if y is None or y <= 0:
                    continue
                if any(k in rec for k in _FALLBACK_KEYS):
                    raise ValueError(f"{path}: a record carries {_FALLBACK_KEYS}; v3 may have read those instead")
                p, o = int(rec["num_prompt_tokens"]), int(rec["num_predicted_output_tokens"])
                state = rec["schedule_state"]
                tier = rec["instance_type"]
                if tier not in rows:
                    rows[tier] = {"served": [], "v3_zero_filled": [], "v3_trained": [], "y": []}
                t = rows[tier]
                served = SimpleNamespace(**{k: state[k] for k in STATE_FIELDS})
                t["served"].append(feature_vector(V3_SWEEP_FEATURES, served, p, o))
                t["v3_zero_filled"].append(v3_serving_vector({k: state[k] for k in FORK_FIELDS}, p, o))
                t["v3_trained"].append(v3_training_vector(state, p, o))
                t["y"].append(y)
    return rows


def sweep_rows(paths: list[Path]) -> tuple[dict[str, dict[str, list]], dict]:
    sweep = read_sweep(paths, FEATURES)
    rows = {tier: {"served": [np.array(r, dtype=np.float32) for r in t.features], "y": t.tpot_s}
            for tier, t in sweep.tiers.items()}
    return rows, {"dropped_failed": sweep.failed, "dropped_no_target": sweep.no_target}


def split(n: int) -> tuple[np.ndarray, np.ndarray]:
    n_val = int(n * VAL_SHARE)
    order = np.random.RandomState(SPLIT_SEED).permutation(n)
    return order[:n - n_val], order[n - n_val:]


def metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    err = np.abs(y - pred)
    ape = err / np.maximum(y, 1e-6)
    return {"n": int(len(y)), "mae": float(np.mean(err)), "mape_pct": float(np.mean(ape) * 100),
            "median_ape_pct": float(np.median(ape) * 100), "under_rate": float(np.mean(pred < y))}


def train(x_tr, y_tr, x_val, y_val, names: tuple[str, ...], threads: int) -> xgb.Booster:
    dtrain = xgb.DMatrix(x_tr, label=y_tr, feature_names=list(names))
    dval = xgb.DMatrix(x_val, label=y_val, feature_names=list(names))
    return xgb.train(dict(V3_PARAMS, nthread=threads), dtrain, num_boost_round=ROUNDS,
                     evals=[(dtrain, "train"), (dval, "val")], early_stopping_rounds=EARLY_STOP, verbose_eval=False)


def _v3_training_vector(path: Path):
    spec = importlib.util.spec_from_file_location("v3_xgboost_predictor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if tuple(module.ALL_FEATURES) != V3_FEATURES:
        raise ValueError(f"{path} is not the v3 feature builder")
    return module.build_feature_vector


def _v3_heads(v3_dir: Path) -> tuple[dict[str, xgb.Booster], dict[str, str]]:
    meta = json.loads((v3_dir / "xgboost_metadata.json").read_text())
    if tuple(meta["feature_names"]) != V3_FEATURES:
        raise ValueError(f"{v3_dir} does not hold the 25-feature v3 heads")
    heads, digests = {}, {}
    for tier in meta["instance_types"]:
        raw = (v3_dir / f"{tier}.xgb").read_bytes()
        heads[tier] = xgb.Booster()
        heads[tier].load_model(bytearray(raw))
        digests[f"{tier}.xgb"] = _md5(raw)
    return heads, digests


def _md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def fit_tier(tier: str, rows: dict[str, list], names: tuple[str, ...], v3_heads: dict, threads: int):
    """The trained head and its report; raises on a feature that is constant in the training rows."""
    if len(rows["y"]) < MIN_ROWS:
        raise SystemExit(f"{tier}: {len(rows['y'])} usable records, fewer than {MIN_ROWS}")
    t = {k: np.stack(v) if k != "y" else np.array(v, dtype=np.float32) for k, v in rows.items()}
    tr, val = split(len(t["y"]))
    constant = [f for i, f in enumerate(names) if np.unique(t["served"][tr][:, i]).size == 1]
    if constant:
        raise SystemExit(f"{tier}: {constant} are constant in the training rows; a head would learn nothing from "
                         "them and only see them move at serving")
    start = time.time()
    booster = train(t["served"][tr], t["y"][tr], t["served"][val], t["y"][val], names, threads)
    report = {"best_iteration": booster.best_iteration, "trees": booster.num_boosted_rounds(),
              "train_s": round(time.time() - start, 1),
              "top_gain": dict(sorted(booster.get_score(importance_type="gain").items(), key=lambda kv: -kv[1])[:8])}
    for name, idx in (("train", tr), ("val", val)):
        y = t["y"][idx]
        report[name] = {"served": metrics(y, booster.inplace_predict(t["served"][idx]))}
        for kind in ("v3_zero_filled", "v3_trained"):
            if kind in t:
                report[name][kind] = metrics(y, v3_heads[tier].inplace_predict(t[kind][idx]))
    return booster, t["served"][val], report


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--sweep", type=Path, nargs="+", help="sweep JSONL logged by the scheduler (tools/tpot_sweep.py)")
    src.add_argument("--v3-sweep", type=Path, nargs=2, metavar=("TRAIN", "TEST"), help="the v3 latency sweep, in v3 order")
    ap.add_argument("--v3-dir", type=Path, help="with --v3-sweep: v3 TPOT heads (xgboost_metadata.json + <tier>.xgb)")
    ap.add_argument("--v3-features", type=Path, help="with --v3-sweep: v3 xgboost_predictor.py (training vectors)")
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    partial = args.out.with_name(args.out.name + ".partial")
    if args.out.exists() or partial.exists():
        raise SystemExit(f"{args.out} or {partial} exists; write a new directory")
    report: dict = {"params": dict(V3_PARAMS), "rounds": ROUNDS, "early_stopping_rounds": EARLY_STOP,
                    "val_share": VAL_SHARE, "split_seed": SPLIT_SEED, "nthread": args.threads, "xgboost": xgb.__version__}
    if args.v3_sweep:
        if args.v3_dir is None or args.v3_features is None:
            raise SystemExit("--v3-sweep needs --v3-dir and --v3-features")
        v3_heads, v3_digests = _v3_heads(args.v3_dir)
        tiers, names, data = sorted(v3_heads), V3_SWEEP_FEATURES, list(args.v3_sweep)
        rows = v3_sweep_rows(data, _v3_training_vector(args.v3_features))
        report.update(input="v3 sweep", v3_heads=v3_digests)
    else:
        if args.v3_dir is not None or args.v3_features is not None:
            raise SystemExit("--v3-dir and --v3-features go with --v3-sweep")
        v3_heads, data = {}, list(args.sweep)
        rows, dropped = sweep_rows(data)
        tiers, names = sorted(rows), FEATURES
        report.update(input="scheduler sweep", **dropped)
    if not tiers:
        raise SystemExit("no usable training records for any tier")
    absent = sorted(set(tiers) - set(rows))
    if absent:
        raise SystemExit(f"no usable training records for tiers {absent}")
    report.update(features=list(names), data=[{"file": p.name, "md5": _md5(p.read_bytes())} for p in data], tiers={})

    written: dict[str, bytes] = {"feature_names.json": json.dumps(list(names), indent=1).encode()}
    trained = {}
    for tier in tiers:
        booster, x_val, report["tiers"][tier] = fit_tier(tier, rows[tier], names, v3_heads, args.threads)
        written[f"{tier}.xgb"] = bytes(booster.save_raw("ubj"))
        trained[tier] = (booster, x_val)
        v = report["tiers"][tier]["val"]
        print(f"{tier}: val MAE {1000 * v['served']['mae']:.3f} ms, MAPE {v['served']['mape_pct']:.1f}%"
              + "".join(f" | {k} MAE {1000 * v[k]['mae']:.3f} ms, MAPE {v[k]['mape_pct']:.1f}%"
                        for k in ("v3_zero_filled", "v3_trained") if k in v), flush=True)
    written["metrics.json"] = json.dumps(report, indent=1).encode()
    partial.mkdir(parents=True)
    for name, blob in written.items():
        (partial / name).write_bytes(blob)
    manifest = json.dumps({name: _md5(blob) for name, blob in written.items()}, indent=1).encode()
    (partial / "md5s.json").write_bytes(manifest)

    reloaded = XGBoostTPOT.from_manifest(partial, _md5(manifest), tiers)   # the scheduler's own load path
    for tier, (booster, x_val) in trained.items():
        if not np.array_equal(reloaded.booster(tier).inplace_predict(x_val), booster.inplace_predict(x_val)):
            raise SystemExit(f"{tier}: the saved head does not reload to the trained one; see {partial}")
    partial.rename(args.out)
    print(f"wrote {args.out}; md5s.json md5 {_md5(manifest)}")
    for name, digest in sorted(json.loads(manifest).items()):
        print(f"  {digest}  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
