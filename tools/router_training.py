"""The routers' training inputs and the AvengersPro artifacts, built fresh for both families (D68-D72).

    python -m tools.router_training filter --corpus SCORED.jsonl --fleet FLEET.json --judge-key KEY --out ROWS.jsonl
    python -m tools.router_training avengers-pro --corpus ROWS.jsonl --fleet FLEET.json --judge-key KEY --models M... \
        --pw 0.25 0.39 0.53 0.70 0.80 --embedder EMBEDDER --embedder-revision REV --n-clusters 32 --out-root DIR
    python -m tools.router_training best-route-data --corpus ROWS.jsonl --fleet FLEET.json --judge-key KEY \
        --models M... --out DIR

The corpus is a family's DeepEval-scored training split (JSONL, one prompt per line: prompt, input_len and, per
model, response, output_length and llm_judge_scores.<judge key>). --models lists the fleet's models in ascending
size; it is the order every tie below falls back to.

filter (D69). A prompt stays only if every fleet model has a judge score on it: every builder needs all the
models' scores on a row (v3's KNN builder would store a missing score as 0.0, its BEST-Route converter too). The
dropped prompts are listed. The output feeds v3's KNN builder unchanged and the two commands below.

avengers-pro (D72). The prompts are embedded with the pinned MiniLM (normalized, as the router embeds them at route
time) and clustered once per family (k-means, 32 clusters, n_init 10, random_state 0, v3's settings); every pw
shares the clustering. Each cluster's ranking is upstream AvengersPro's (balance_cluster_router.py,
compute_balance_score), with the cost it uses: each model's average per-query cost over the cluster's prompts,
  cost = prompt tokens x the model's input price + its output tokens x its output price   (the fleet's prices),
divided by the highest average cost in the cluster; accuracy is the mean judge score, min-max normalized within
the cluster (1.0 for every model when they are all equal); score = pw x accuracy + (1 - pw) x (1 - cost); models
sorted by score, ties in --models order (upstream's stable sort). Upstream's minimum-accuracy threshold is 0 by
default and changes nothing. v3's builder used one number per model instead (its size, or its mean per-token
price), the same in every cluster. Output per pw: <out-root>/avengers_pro_<family>_pw<pw>/ with kmeans.joblib,
cluster_rankings.json ({cluster id: [models, best first]}) and config.json, the files the router loads.

best-route-data (D71). v3's BEST-Route candidates format (its Qwen data converter: instruction, input,
output, candidates [text, model <base>_bo1, scores.armoRM_scores = the judge score, token_num_prompt,
token_num_responses]) for the 4-way head (v3's train_best_route_wrapper.py --num-labels 4), which labels a prompt
with its FIRST best candidate. The candidates are ordered by each model's mean per-token price (the mean of its
input and output price), ties in --models order, so a tie goes to the cheapest model: for both families' fleets
this is the size order (Qwen: v3's order; Gemma: D208's prices). labels.json records that order for --label-models. The
validation set is 10% of the training prompts (seed 42), never test prompts (v3 took half of the test split).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

from bench.dataset import file_sha256
from bench.schema import read_jsonl, write_new_jsonl
from routebalance.config import ConfigError, loads_strict

JUDGE_KEY = "deepeval-llama3.1-8b-it_reference"
N_CLUSTERS = 32          # v3's AvengersPro artifacts
KMEANS_SEED = 0          # v3's retrain_avengers_pro_qwen.py
VAL_FRACTION = 0.1
SPLIT_SEED = 42          # v3's converter shuffled with seed 42


def fleet_prices(fleet_path: Path) -> dict[str, tuple[float, float]]:
    """model -> (input price, output price) per token, from a fleet file's tiers; a model on several tiers must
    carry one price."""
    fleet = loads_strict(Path(fleet_path).read_text())
    out: dict[str, tuple[float, float]] = {}
    for tier_id, t in fleet["tiers"].items():
        price = (t["cost_in"], t["cost_out"])
        if not all(isinstance(p, (int, float)) and not isinstance(p, bool) and p > 0 for p in price):
            raise ConfigError(f"{fleet_path}: tier {tier_id} needs positive numeric prices, got {price}")
        if t["model"] in out and out[t["model"]] != price:
            raise ConfigError(f"{fleet_path}: model {t['model']} has two prices")
        out[t["model"]] = price
    return out


def check_models(models: Sequence[str], prices: Mapping[str, tuple[float, float]]) -> None:
    if len(set(models)) != len(models) or set(models) != set(prices):
        raise ConfigError(f"--models {list(models)} must list each of the fleet's models {sorted(prices)} once")


def score_of(row: dict, model: str, judge_key: str) -> Optional[float]:
    models = row.get("models")
    md = models.get(model) if isinstance(models, dict) else None
    judged = md.get("llm_judge_scores") if isinstance(md, dict) else None
    s = judged.get(judge_key) if isinstance(judged, dict) else None
    if s is None:
        return None                     # unscored: filter's escalation (D69) decides
    if isinstance(s, bool) or not isinstance(s, (int, float)):
        raise ConfigError(f"{model}: judge score {s!r} is not a number")
    v = float(s)
    if not (math.isfinite(v) and 0.0 <= v <= 1.0):
        raise ConfigError(f"{model}: judge score {s!r} is not in [0, 1]")
    return v


def row_id(row: dict, i: int) -> str:
    if row.get("request_id") is None:
        raise ConfigError(f"corpus row {i} has no request_id")
    return str(row["request_id"])


def response_of(row: dict, model: str, i: int) -> str:
    text = row["models"][model].get("response")
    if not isinstance(text, str):
        raise ConfigError(f"row {i} ({model}): no response text")
    return text


# ---------------------------------------------------------------- filter (D69)

def keep_scored(rows: Sequence[dict], models: Sequence[str], judge_key: str) -> tuple[list[dict], list[str]]:
    """(the rows every model has a score on, the ids of the rest)."""
    kept, dropped = [], []
    for i, r in enumerate(rows):
        if not isinstance(r.get("prompt"), str) or not r["prompt"]:
            raise ConfigError(f"row {i}: no prompt")
        if all(score_of(r, m, judge_key) is not None for m in models):
            kept.append(r)
        else:
            dropped.append(row_id(r, i))
    return kept, dropped


def _whole(x: object) -> bool:
    """A non-negative whole number (a bool is not one)."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return False
    return x >= 0 and x == int(x)


def _tokens(row: dict, model: str, i: int) -> tuple[int, int]:
    md = row["models"][model]
    p, o = row.get("input_len"), md.get("output_length")
    if not (_whole(p) and _whole(o)):
        raise ConfigError(f"row {i} ({model}): needs whole non-negative input_len and output_length, got {p!r}, {o!r}")
    return int(p), int(o)


# ---------------------------------------------------------------- avengers-pro (D72)

def matrices(rows: Sequence[dict], models: Sequence[str], prices: Mapping[str, tuple[float, float]],
             judge_key: str) -> tuple[np.ndarray, np.ndarray]:
    """(quality, cost), each (n prompts, n models): the judge score, and the per-query cost at the fleet's prices."""
    q = np.zeros((len(rows), len(models)))
    c = np.zeros((len(rows), len(models)))
    for i, r in enumerate(rows):
        for j, m in enumerate(models):
            s = score_of(r, m, judge_key)
            if s is None:
                raise ConfigError(f"row {i}: {m} has no {judge_key} score (run filter first)")
            p, o = _tokens(r, m, i)
            q[i, j] = s
            c[i, j] = p * prices[m][0] + o * prices[m][1]
    return q, c


def balance_ranking(quality: np.ndarray, cost: np.ndarray, models: Sequence[str], pw: float) -> list[str]:
    """One cluster's ranking, upstream AvengersPro's (compute_balance_score): quality and cost are this cluster's
    (prompts, models) matrices."""
    # Per model, from a list, as upstream does (np.mean(successes)): a column reduction sums in another order,
    # and min-max normalization would turn that last-bit difference into a full accuracy lead.
    acc = np.array([np.mean(quality[:, j].tolist()) for j in range(len(models))])
    avg_cost = np.array([np.mean(cost[:, j].tolist()) for j in range(len(models))])
    span = acc.max() - acc.min()
    norm_acc = (acc - acc.min()) / span if span > 0 else np.ones_like(acc)
    top = avg_cost.max()
    cost_score = 1.0 - avg_cost / top if top > 0 else np.ones_like(avg_cost)
    score = pw * norm_acc + (1.0 - pw) * cost_score
    order = sorted(range(len(models)), key=lambda j: -score[j])       # stable: ties keep --models order
    return [models[j] for j in order]


def cluster_rankings(labels: np.ndarray, quality: np.ndarray, cost: np.ndarray, models: Sequence[str], pw: float,
                     n_clusters: int) -> dict[str, list[str]]:
    out = {}
    for k in range(n_clusters):
        mask = labels == k
        if not mask.any():
            raise ConfigError(f"cluster {k} is empty; the router needs a ranking for every cluster")
        out[str(k)] = balance_ranking(quality[mask], cost[mask], models, pw)
    return out


def embed_and_cluster(prompts: Sequence[str], embedder: str, revision: str, n_clusters: int):
    """(the fitted k-means, its labels): MiniLM at the pinned revision, normalized, on the CPU."""
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import KMeans

    model = SentenceTransformer(embedder, revision=revision, device="cpu")
    emb = model.encode(list(prompts), normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=KMEANS_SEED)
    return km, km.fit_predict(emb)


def _pw_name(pw: float) -> str:
    return f"{pw:.2f}"


def build_avengers_pro(a: argparse.Namespace) -> None:
    prices = fleet_prices(a.fleet)
    check_models(a.models, prices)
    if not a.pw or not all(0.0 <= p <= 1.0 for p in a.pw) or len(set(a.pw)) != len(a.pw):
        raise ConfigError(f"--pw needs distinct weights in [0, 1], got {a.pw}")
    rows = [rec for _, rec in read_jsonl(a.corpus)]
    kept, dropped = keep_scored(rows, a.models, a.judge_key)
    if dropped:
        raise ConfigError(f"{len(dropped)} rows lack a score (run filter first), e.g. {dropped[:3]}")
    quality, cost = matrices(kept, a.models, prices, a.judge_key)
    family = a.family
    dirs = {pw: Path(a.out_root) / f"avengers_pro_{family}_pw{_pw_name(pw)}" for pw in a.pw}
    taken = [str(d) for d in dirs.values() if d.exists()]
    if taken:
        raise ConfigError(f"{taken} exist; write new directories")
    km, labels = embed_and_cluster([r["prompt"] for r in kept], a.embedder, a.embedder_revision, a.n_clusters)
    import joblib

    corpus_sha = file_sha256(a.corpus)
    for pw, d in dirs.items():
        d.mkdir(parents=True)
        joblib.dump(km, d / "kmeans.joblib")
        ranks = cluster_rankings(labels, quality, cost, a.models, pw, a.n_clusters)
        (d / "cluster_rankings.json").write_text(json.dumps(ranks, indent=2))
        (d / "config.json").write_text(json.dumps({
            "embedder": a.embedder, "embedder_revision": a.embedder_revision, "n_clusters": a.n_clusters,
            "rank_formula": "upstream", "cost": "per_query_per_cluster", "performance_weight": pw,
            "cost_sensitivity": round(1.0 - pw, 10), "models": list(a.models), "train_rows": len(kept),
            "prices_per_token": {m: list(prices[m]) for m in a.models}, "corpus_sha256": corpus_sha,
            "judge_key": a.judge_key}, indent=2))
        tops = {m: sum(r[0] == m for r in ranks.values()) for m in a.models}
        print(f"{d}: top model per cluster {tops}")


# ---------------------------------------------------------------- best-route-data (D71)

def price_order(models: Sequence[str], prices: Mapping[str, tuple[float, float]]) -> list[str]:
    """The models by mean per-token price, ties in the given (size) order."""
    return sorted(models, key=lambda m: (prices[m][0] + prices[m][1]) / 2.0)


def best_route_example(row: dict, i: int, order: Sequence[str], largest: str, judge_key: str) -> dict:
    cands = []
    for m in order:
        p, o = _tokens(row, m, i)
        cands.append({"text": response_of(row, m, i), "model": f"{m.split('/')[-1]}_bo1",
                      "scores": {"armoRM_scores": score_of(row, m, judge_key)},
                      "token_num_prompt": p, "token_num_responses": o, "cost": 0.0})
    # output: the largest model's response, as v3 recorded it (the 4-way trainer does not read it)
    return {"id": row_id(row, i), "instruction": row["prompt"], "input": "",
            "output": response_of(row, largest, i), "candidates": cands}


def split(rows: Sequence[dict], fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    """(train, validation): a seeded shuffle, the first `fraction` of it held out."""
    idx = list(range(len(rows)))
    random.Random(seed).shuffle(idx)
    n_val = int(round(len(rows) * fraction))
    val = sorted(idx[:n_val])
    held = set(val)
    return [rows[i] for i in range(len(rows)) if i not in held], [rows[i] for i in val]


def build_best_route_data(a: argparse.Namespace) -> None:
    prices = fleet_prices(a.fleet)
    check_models(a.models, prices)
    rows = [rec for _, rec in read_jsonl(a.corpus)]
    kept, dropped = keep_scored(rows, a.models, a.judge_key)
    if dropped:
        raise ConfigError(f"{len(dropped)} rows lack a score (run filter first), e.g. {dropped[:3]}")
    order = price_order(a.models, prices)
    examples = [best_route_example(r, i, order, a.models[-1], a.judge_key) for i, r in enumerate(kept)]
    train, val = split(examples, VAL_FRACTION, SPLIT_SEED)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)                 # the tool creates its output directory, as before
    write_new_jsonl(out / "train.jsonl", train)
    write_new_jsonl(out / "val.jsonl", val)
    labels = {"label_models": order, "judge_key": a.judge_key, "train": len(train), "val": len(val),
              "rule": "label = the first best candidate; candidates ordered by mean per-token price"}
    (out / "labels.json").write_text(json.dumps(labels, indent=2))
    print(f"{out}: {len(train)} train, {len(val)} validation prompts; label order {order}")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("filter")
    av = sub.add_parser("avengers-pro")
    br = sub.add_parser("best-route-data")
    for p in (f, av, br):
        p.add_argument("--corpus", type=Path, required=True)
        p.add_argument("--fleet", type=Path, required=True)
        p.add_argument("--judge-key", required=True, help=f"the scores' key ({JUDGE_KEY})")
    f.add_argument("--out", type=Path, required=True)
    for p in (av, br):
        p.add_argument("--models", nargs="+", required=True, help="the fleet's models, ascending size")
    av.add_argument("--family", required=True)
    av.add_argument("--pw", type=float, nargs="+", required=True)
    av.add_argument("--embedder", required=True, help="sentence-transformers/all-MiniLM-L6-v2, the KNN encoder")
    av.add_argument("--embedder-revision", required=True)
    av.add_argument("--n-clusters", type=int, required=True, help=f"k-means clusters ({N_CLUSTERS}, as v3)")
    av.add_argument("--out-root", type=Path, required=True)
    br.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "filter":
        prices = fleet_prices(a.fleet)
        rows = [rec for _, rec in read_jsonl(a.corpus)]
        kept, dropped = keep_scored(rows, sorted(prices), a.judge_key)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        write_new_jsonl(a.out, kept)
        print(f"{a.out}: kept {len(kept)} of {len(rows)} prompts; dropped {len(dropped)}: {dropped}")
    elif a.cmd == "avengers-pro":
        build_avengers_pro(a)
    else:
        build_best_route_data(a)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
