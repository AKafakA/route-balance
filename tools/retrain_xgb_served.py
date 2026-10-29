"""Retrain the per-tier TPOT heads on what the scheduler knows at serving time.

Serving runs official vLLM, so a head may read only routebalance.estimate.xgboost_tpot.FEATURES (the
instance's running and waiting counts, KV-cache utilization and pending decode tokens, the request's prompt
and expected output tokens, and two features derived from the request alone). The input is the sweep the
scheduler logs under official vLLM (--sweep, tools/tpot_sweep.py): each record carries the feature vector
feature_values() built at dispatch, pending_decode_tokens computed RouteBalance's way and the expected output
length, so heads use all FEATURES. A head never trains on a feature that is constant in its training rows (it
could learn nothing from it and the value would only move at serving), nor on fewer than MIN_ROWS usable
records (a fleet tier without a head is an error): the tool fails instead.

The procedure:
  records  per tier, in file order; failed requests and missing or non-positive targets are dropped
  target   TPOT in s/token (tpot_s)
  split    per tier, RandomState(42).permutation; the last 20% is the validation set
  booster  reg:squarederror, max_depth 6, eta 0.1, subsample 0.8, colsample_bytree 0.8,
           min_child_weight 5, seed 42; 500 rounds, early stopping after 30 on validation RMSE.
           Predictions use every tree. nthread is the host's CPU cap.
Every tier in the records is trained.

Output (--out): <tier>.xgb (UBJSON), feature_names.json, metrics.json, and md5s.json (the md5 of every
other file; md5s.json's own md5 is what a config pins as latency.table_md5). It is written to <out>.partial
and renamed to <out> only after the heads reload, through the scheduler's own loader, to the trained ones.

metrics.json, per tier and split (train, validation): the new heads' n, MAE (s/token), MAPE (%,
|err| / max(y, 1e-6)), median APE (%) and under-prediction rate, on raw booster outputs.

Usage: python -m tools.retrain_xgb_served --sweep SWEEP.jsonl --threads 8 --out DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import xgboost as xgb

from routebalance.estimate.xgboost_tpot import FEATURES, XGBoostTPOT
from tools.tpot_sweep import read_sweep

V3_PARAMS = {"objective": "reg:squarederror", "eval_metric": ["mae", "rmse"], "max_depth": 6, "learning_rate": 0.1,
             "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5, "seed": 42}
ROUNDS, EARLY_STOP, VAL_SHARE, SPLIT_SEED = 500, 30, 0.2, 42
MIN_ROWS = 50                 # a tier needs at least this many usable records


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


def _md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def fit_tier(tier: str, rows: dict[str, list], names: tuple[str, ...], threads: int):
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
    return booster, t["served"][val], report


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sweep", type=Path, nargs="+", required=True,
                    help="sweep JSONL logged by the scheduler (tools/tpot_sweep.py)")
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    partial = args.out.with_name(args.out.name + ".partial")
    if args.out.exists() or partial.exists():
        raise SystemExit(f"{args.out} or {partial} exists; write a new directory")
    report: dict = {"params": dict(V3_PARAMS), "rounds": ROUNDS, "early_stopping_rounds": EARLY_STOP,
                    "val_share": VAL_SHARE, "split_seed": SPLIT_SEED, "nthread": args.threads, "xgboost": xgb.__version__}
    data = list(args.sweep)
    rows, dropped = sweep_rows(data)
    tiers, names = sorted(rows), FEATURES
    report.update(input="scheduler sweep", **dropped)
    if not tiers:
        raise SystemExit("no usable training records for any tier")
    report.update(features=list(names), data=[{"file": p.name, "md5": _md5(p.read_bytes())} for p in data], tiers={})

    written: dict[str, bytes] = {"feature_names.json": json.dumps(list(names), indent=1).encode()}
    trained = {}
    for tier in tiers:
        booster, x_val, report["tiers"][tier] = fit_tier(tier, rows[tier], names, args.threads)
        written[f"{tier}.xgb"] = bytes(booster.save_raw("ubj"))
        trained[tier] = (booster, x_val)
        v = report["tiers"][tier]["val"]
        print(f"{tier}: val MAE {1000 * v['served']['mae']:.3f} ms, MAPE {v['served']['mape_pct']:.1f}%", flush=True)
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
