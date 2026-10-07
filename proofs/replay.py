"""Replay the archived runs through one code tree, and compare two trees' replays.

    python -m proofs.replay run TASK --adapter A --runs ROOT --artifacts DIR --fleets DIR --hosts FILE
                                     --avengers-repo DIR --best-route-repo DIR --out FILE
                                     [--estimates DIR] [--shard I/N] [--only RUN...]
    python -m proofs.replay compare A.jsonl B.jsonl

Every replay process must run with PYTHONHASHSEED=0 and fixed OMP_NUM_THREADS and MKL_NUM_THREADS (the CPU kernels
of the encoders can round differently with another thread count), the same for the two trees compared.

TASK is one of
  estimates    RouteBalance's runs: the estimator on each round's requests in the order the round collected them
               (send order), as one batch, as the server calls it (the encoder pads to the round's longest prompt, so
               a prompt's estimate depends on its round). With --estimates DIR it also writes each run's estimates to
               DIR/<run>.json.gz for the rounds task. Checked against the log: each line's lengths, exact, and the
               largest relative difference.
  rounds       RouteBalance's runs: every round (schedule_round) on the estimates in --estimates DIR (written by the
               code that ran) and on a load view rebuilt from the log (each instance's last logged state, idle before
               any), with a fresh least-recently-assigned record per run and a counting clock.
  estimator    the pipeline runs that used the estimator (RB-decoupled's router, pending decode counted with the
               expected length): the estimator on each request alone, as the server calls it. Checked against the
               log, where the run counted pending decode: the decode and feature tokens.
  routers      the runs with a router that calls a model (AvengersPro, BEST-Route, RB-decoupled): the router on each
               request. Checked against the log: the chosen model (the dispatched instance's), the score (exact, and
               the largest difference) and the reason.
  dispatchers  every pipeline run: a new dispatcher (as each run's new scheduler built it) choosing among the logged
               model's instances for each request in send order, on the rebuilt load view, told each request's logged
               outcome after its dispatch. Checked against the log, for round robin and random behind the passthrough
               router: the instance, and where the two differ, whether the request swapped places with its neighbour
               (the server took them in the other order).

A router's decision and a lone request's estimate depend only on the configuration and the request, so each process
computes each distinct call once. Each run gives one line: {run, task, units, digest, checks}; the digest is the sha256
of every unit's outputs, each float exact, so two trees give the same digests exactly when their outputs are equal.
`compare` prints every run whose digest differs and exits 1 if any does.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib
import itertools
import json
import sys
from pathlib import Path

from proofs import archive as A

TASKS = ("estimates", "rounds", "estimator", "routers", "dispatchers")
MODEL_ROUTERS = ("avengers_pro", "best_route_upstream", "rb_decoupled")
NOT_THE_ROUTER = ("telemetry", "telemetry_read", "dead_reckoning")    # configuration keys a router never reads


class LoadView:
    """A deterministic load view for the replay: each instance's last logged state (a decision line records the
    chosen instance's state as the decision read it), idle before any. Both trees see the same view."""

    def __init__(self, fleet: dict):
        tiers = fleet["tiers"]
        self.stats = {i["instance_id"]: [0, 0, A.hexf(0.0), 0, tiers[i["tier"]]["max_num_seqs"], 0]
                      for i in fleet["instances"]}

    def update(self, line: dict) -> None:
        st, inst = line.get("state"), line.get("instance_id")
        if st and inst in self.stats:
            self.stats[inst] = [st["num_running"], st["num_waiting"], A.hexf(st["kv_cache_utilization"]),
                                st["num_preemptions"], st["max_num_seqs"], st["pending_decode_tokens"]]

    def snapshot(self) -> dict:
        return {i: {"stats": list(s), "age_ms": A.hexf(0.0), "in_flight": s[0] + s[1], "engine_waiting": s[1]}
                for i, s in self.stats.items()}


class Context:
    """Each configuration's components, loaded once, and the per-process caches of calls that depend only on the
    configuration and the request."""

    def __init__(self, adapter, args):
        self.adapter, self.args = adapter, args
        self._codes: dict[str, object] = {}
        self.routed: dict[str, list] = {}
        self.estimated: dict[str, list] = {}

    def code(self, run: Path):
        key = hashlib.sha256((run / "config.json").read_bytes()).hexdigest()
        if key not in self._codes:
            a, fam = self.args, A.family(run)
            self._codes[key] = self.adapter.Code(run / "config.json", a.fleets / f"fleet_{fam}.json", a.hosts,
                                                 a.artifacts, a.avengers_repo, a.best_route_repo)
        return self._codes[key]

    def fleet(self, run: Path) -> dict:
        return A.read_json(self.args.fleets / f"fleet_{A.family(run)}.json")


def _model_of(fleet: dict, instance_id: str) -> str:
    tier = next(i["tier"] for i in fleet["instances"] if i["instance_id"] == instance_id)
    return fleet["tiers"][tier]["model"]


def _relative(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-300)


def task_estimates(ctx: Context, run: Path, digest: A.Digest, checks: dict) -> None:
    code, rec, lines = ctx.code(run), A.record(run), A.decisions(run)
    saved = []
    for number, members, batch in A.rounds(lines, rec):
        _, plain = code.estimate([r["prompt"] for r in batch])
        digest.add({"round": number, "estimates": plain})
        saved.append({"round": number, "requests": [r["request_id"] for r in batch], "estimates": plain})
        by_id = {r["request_id"]: p for r, p in zip(batch, plain, strict=True)}
        for line in members:
            got = {m: float.fromhex(v[0]) for m, v in by_id[line["request_id"]].items()}
            checks["lengths_equal" if got == line["lengths"] else "lengths_differ"] += 1
            for m, x in line["lengths"].items():
                checks["lengths_max_relative_difference"] = max(checks["lengths_max_relative_difference"],
                                                                _relative(got[m], x))
    if ctx.args.estimates is not None:
        ctx.args.estimates.mkdir(parents=True, exist_ok=True)
        data = json.dumps(saved, separators=(",", ":")).encode()
        with open(ctx.args.estimates / f"{run.name}.json.gz", "wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as f:     # mtime 0: the same estimates, the same bytes
                f.write(data)


def task_rounds(ctx: Context, run: Path, digest: A.Digest, checks: dict) -> None:
    code, rec, lines = ctx.code(run), A.record(run), A.decisions(run)
    with gzip.open(ctx.args.estimates / f"{run.name}.json.gz", "rt") as f:
        saved = {x["round"]: x for x in json.load(f)}
    view, last_assigned, ticks = LoadView(ctx.fleet(run)), {}, itertools.count(1)
    max_tokens = rec["sampling"]["max_tokens"]
    for number, members, batch in A.rounds(lines, rec):
        s = saved[number]
        if s["requests"] != [r["request_id"] for r in batch]:
            raise SystemExit(f"{run.name} round {number}: the saved estimates are for other requests")
        estimates = [code.estimates_from_plain(p) for p in s["estimates"]]
        decisions = code.schedule(batch, estimates, max_tokens, view.snapshot(), last_assigned,
                                  lambda: float(next(ticks)))
        digest.add({"round": number, "decisions": decisions})
        checks["decisions"] += len(decisions)
        for line in members:
            view.update(line)


def task_estimator(ctx: Context, run: Path, digest: A.Digest, checks: dict) -> None:
    code, rec, fleet = ctx.code(run), A.record(run), ctx.fleet(run)
    lines = {line["request_id"]: line for line in A.decisions(run)}
    own_length = code.cfg.pending_decode == "own_length"
    est_key = A.canonical({"family": A.family(run), "estimator": A.config(run)["estimator"]})
    for r in A.in_send_order(rec):
        key = hashlib.sha256((est_key + "\x00" + r["prompt"]).encode()).hexdigest()
        if key not in ctx.estimated:
            ctx.estimated[key] = code.estimate([r["prompt"]])[1]
        plain = ctx.estimated[key]
        digest.add({"req": r["request_id"], "estimates": plain})
        line = lines.get(r["request_id"])
        if own_length and line is not None and line.get("instance_id"):
            lengths = {m: float.fromhex(v[0]) for m, v in plain[0].items()}
            want = (max(1, int(lengths[_model_of(fleet, line["instance_id"])])), max(1, int(max(lengths.values()))))
            checks["tokens_equal" if want == (line["decode_tokens"], line["feature_tokens"]) else "tokens_differ"] += 1


def task_routers(ctx: Context, run: Path, digest: A.Digest, checks: dict) -> None:
    code, rec, fleet = ctx.code(run), A.record(run), ctx.fleet(run)
    lines = {line["request_id"]: line for line in A.decisions(run)}
    cfg = A.config(run)
    router_cfg = {k: v for k, v in cfg.items() if k not in NOT_THE_ROUTER and k != "pipeline"}
    router_cfg["router"] = {k: v for k, v in cfg["pipeline"]["router"].items() if k != "concurrency"}
    cfg_key = A.canonical({"family": A.family(run), "config": router_cfg})
    for r in A.in_send_order(rec):
        key = hashlib.sha256(f"{cfg_key}\x00{r['prompt_tokens']}\x00{r['prompt']}".encode()).hexdigest()
        if key not in ctx.routed:
            ctx.routed[key] = code.route(r["prompt"], r["request_id"], r["prompt_tokens"])
        model, score, reason = ctx.routed[key]
        digest.add({"req": r["request_id"], "decision": [model, score, reason]})
        line = lines.get(r["request_id"])
        if line is not None and line.get("instance_id"):
            checks["model_equal" if model == _model_of(fleet, line["instance_id"]) else "model_differ"] += 1
            checks["score_equal" if float.fromhex(score) == line["router_score"] else "score_differ"] += 1
            checks["reason_equal" if reason == line["reason"] else "reason_differ"] += 1
            checks["score_max_abs_difference"] = max(checks["score_max_abs_difference"],
                                                     abs(float.fromhex(score) - line["router_score"]))


def task_dispatchers(ctx: Context, run: Path, digest: A.Digest, checks: dict) -> None:
    code, rec, fleet = ctx.code(run), A.record(run), ctx.fleet(run)
    lines = {line["request_id"]: line for line in A.decisions(run)}
    router, dispatch = code.cfg.pipeline.router.type, code.cfg.pipeline.dispatch.type
    view, dispatcher = LoadView(fleet), code.dispatcher()
    check = router == "passthrough" and dispatch in ("round_robin", "random")
    replayed, logged = [], []
    for r in A.in_send_order(rec):
        line = lines.get(r["request_id"])
        if line is None or not line.get("instance_id"):
            continue                                  # a request that was never dispatched
        model = code.ALL if router == "passthrough" else _model_of(fleet, line["instance_id"])
        chosen = code.dispatch(dispatcher, model, view.snapshot())
        code.observe(dispatcher, chosen, line)     # the outcome, fed in send order (both trees alike)
        digest.add({"req": r["request_id"], "instance": chosen})
        replayed.append(chosen)
        logged.append(line["instance_id"])
        view.update(line)
    if check:
        i = 0
        while i < len(replayed):
            if replayed[i] == logged[i]:
                checks["instance_equal"] += 1
                i += 1
            elif i + 1 < len(replayed) and replayed[i] == logged[i + 1] and replayed[i + 1] == logged[i]:
                checks["instance_swapped_with_neighbour"] += 2
                i += 2
            else:
                checks["instance_differ"] += 1
                i += 1


FN = {"estimates": task_estimates, "rounds": task_rounds, "estimator": task_estimator, "routers": task_routers,
      "dispatchers": task_dispatchers}


def wanted(task: str, cfg: dict) -> bool:
    pipeline = cfg.get("pipeline")
    if task in ("estimates", "rounds"):
        return cfg["mode"] == "routebalance"
    if task == "estimator":
        return pipeline is not None and (pipeline["router"]["type"] == "rb_decoupled"
                                         or cfg.get("pending_decode") == "own_length")
    if task == "routers":
        return pipeline is not None and pipeline["router"]["type"] in MODEL_ROUTERS
    return pipeline is not None


def run(args) -> int:
    if (args.task == "rounds") and args.estimates is None:
        raise SystemExit("rounds needs --estimates DIR (written by the estimates task with the code that ran)")
    ctx = Context(importlib.import_module(f"proofs.adapters.{args.adapter}"), args)
    todo = [r for r in A.runs(args.runs, only=set(args.only) if args.only else None) if wanted(args.task, A.config(r))]
    i, n = (int(x) for x in args.shard.split("/"))
    with open(args.out, "a") as f:
        for r in todo[i::n]:
            digest, checks = A.Digest(), _Counter()
            FN[args.task](ctx, r, digest, checks)
            A.write_line(f, {"run": r.name, "task": args.task, "units": digest.units, "digest": digest.hexdigest(),
                             "checks": dict(sorted(checks.items()))})
    return 0


def compare(a: Path, b: Path) -> int:
    left = {(x["run"], x["task"]): x for x in A.read_lines(a)}
    right = {(x["run"], x["task"]): x for x in A.read_lines(b)}
    only = sorted(set(left) ^ set(right))
    differ = sorted(k for k in set(left) & set(right) if left[k]["digest"] != right[k]["digest"]
                    or left[k]["units"] != right[k]["units"])
    for k in only:
        print(f"ONLY IN ONE {k}")
    for k in differ:
        print(f"DIFFER {k}: {left[k]['units']} vs {right[k]['units']} units")
    tasks = sorted({k[1] for k in left})
    print(f"compared {len(set(left) & set(right))} runs over {tasks}: {len(differ)} differ, {len(only)} in one file only")
    return 1 if differ or only else 0


class _Counter(dict):
    def __missing__(self, key):
        return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("task", choices=TASKS)
    r.add_argument("--adapter", required=True)
    for name in ("runs", "artifacts", "fleets", "hosts", "avengers-repo", "best-route-repo", "out"):
        r.add_argument(f"--{name}", type=Path, required=True)
    r.add_argument("--estimates", type=Path)
    r.add_argument("--shard", default="0/1")
    r.add_argument("--only", nargs="*")
    c = sub.add_parser("compare")
    c.add_argument("a", type=Path)
    c.add_argument("b", type=Path)
    a = ap.parse_args(argv)
    return run(a) if a.cmd == "run" else compare(a.a, a.b)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
