"""The end-to-end run on one GPU: two real vLLM engines, the real scheduler and a tree's own runner, on a small render
made from the campaign's; proofs/gpu/run.sh runs every step.

    python proofs/gpu/e2e.py build  --tree REF_TREE --out OUT --artifacts DIR --data-root DIR --hf-home DIR [--hf-home DIR]
    python proofs/gpu/e2e.py engines-up   --out OUT --tree REF_TREE --engine-venv VENV
    python proofs/gpu/e2e.py run    --out OUT --tree TREE --kind ref|submission --artifacts DIR --data-root DIR
                                    --avengers-repo DIR --best-route-repo DIR --tiktoken DIR --scheduler-python PY
    python proofs/gpu/e2e.py compare --out OUT
    python proofs/gpu/e2e.py engines-down --out OUT

What stands in for the lease, and why (everything else is the tree's own code, unchanged):
  - the open-file limit: raised to this machine's hard limit before the runs, as the client asks (`ulimit -n`);
  - the fleet: Qwen2.5-3B and Qwen2.5-7B, the campaign's tiers (configs/fleet_qwen.json) on two one-GPU "nodes" of the
    A30 type at 127.0.0.1, both engines on the one GPU of this machine; the engine arguments are the campaign's
    (configs/engine_args_qwen.json) with --gpu-memory-utilization lowered so the two fit on one GPU;
  - the engines are launched as scripts/node_ctl.sh launches them (its environment, the planned arguments of
    tools/engine_plan.py, a new session each) and recorded in deploy.sh's record format; their PATH also holds the
    engine environment's bin, where vLLM's kernel builds find ninja (the lease nodes have it from the system);
  - the scheduler is launched as node_ctl.sh sched-start launches it (its environment and arguments), restarted
    before every run, as the rows runner's --fresh-scheduler does, by this harness: the node scripts need the lease
    nodes' layout and setup state;
  - each run goes through the tree's own one-run command: `python -m runner cell` (ref) or `scripts/run_cell.sh`
    (submission), in a copy of the tree whose scripts/node_layout.sh names this machine's data folder and CUDA;
  - the render: the campaign's configurations, unchanged, of RouteBalance (the main preset; one request per round;
    the learned latency heads; the pollers' cache), RB-decoupled and the five dispatchers alone (one with the
    per-request probe), each on one short trace at seed 1, on the campaign's prompt set. BEST-Route and AvengersPro
    need the campaign's four Qwen models in the fleet, which do not fit on one GPU: BEST-Route's ranker scores every
    candidate, and AvengersPro returns its one best model, often the 14B or the 72B (a fleet without it refuses the
    request, as the code that ran does). Their choices are proven by the replay of every archived request.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

CONFIGS = ["qwen.rb-static.q80l10c10", "qwen.rb-static.abl-round1.q80l10c10", "g1.qwen.rb-learned.q50l30c20",
           "g1.qwen.rb-static-poll.q50l30c20", "qwen.rb-static-dec.q80l10c10", "qwen.disp.rr", "qwen.disp.random",
           "qwen.disp.po2", "qwen.disp.lor", "qwen.disp.lowest-latency", "g1.qwen.disp.po2-probe"]
TRACE = "gpu40"
TRACE_DOC = {"burstiness": 1.0, "segments": [{"n_requests": 40, "rate": 2.0}]}
TIERS = {"qwen2.5-3b_a30": ("Qwen-2.5-3B_0", "n0", 18001, 0.42), "qwen2.5-7b_a30": ("Qwen-2.5-7B_0", "n1", 18002, 0.45)}
SCHEDULER_PORT = 18100
KEEP_ALIVE_S = "600"            # node_layout.sh ENGINE_KEEP_ALIVE_S


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1) + "\n")


def get(url: str, timeout: float = 10.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode()


# ---------------------------------------------------------------- the inputs

def build(a) -> None:
    sys.path.insert(0, str(a.tree))
    from routebalance.config import fleet_digest, load_fleet
    out, src = Path(a.out), Path(a.tree) / "configs"
    inp = out / "inputs"
    fleet_q = json.loads((src / "fleet_qwen.json").read_text())
    args_q = json.loads((src / "engine_args_qwen.json").read_text())
    fleet = {"tiers": {t: fleet_q["tiers"][t] for t in TIERS},
             "instances": [{"instance_id": iid, "tier": t, "node": node, "port": port}
                           for t, (iid, node, port, _) in TIERS.items()]}
    hosts = {node: {"address": "127.0.0.1", "hw": "d7525"} for _, node, _, _ in TIERS.values()}
    engine_args = {"vllm": args_q["vllm"],
                   "tiers": {t: {"revision": args_q["tiers"][t]["revision"],
                                 "args": {**args_q["tiers"][t]["args"], "--gpu-memory-utilization": share}}
                             for t, (_, _, _, share) in TIERS.items()},
                   "gpus": {iid: [0] for iid, _, _, _ in TIERS.values()}}
    write(inp / "fleet.json", fleet)
    write(inp / "hosts.json", hosts)
    write(inp / "engine_args.json", engine_args)
    # the render: the chosen configurations byte for byte, their groups and datasets, one short trace at seed 1
    r36 = src / "render36"
    scope36, plan36 = json.loads((r36 / "scope.json").read_text()), json.loads((r36 / "plan.json").read_text())
    render = inp / "render"
    (render / "configs").mkdir(parents=True)
    blocks, groups = {}, {}
    for cid in CONFIGS:
        shutil.copyfile(r36 / "configs" / f"{cid}.json", render / "configs" / f"{cid}.json")
        (b,) = {(x["group"], x["dataset"], x["prompt_order"]) for x in scope36["blocks"] if cid in x["configs"]}
        g, ds, order = b
        blocks.setdefault(g, {"name": f"GPU-{g}", "group": g, "dataset": ds, "configs": [], "traces": [TRACE],
                              "seeds": [1], "prompt_order": order})["configs"].append(cid)
        groups.setdefault(g, {"reference": scope36["groups"][g]["reference"], "intended_differences": {}})
        groups[g]["intended_differences"][cid] = scope36["groups"][g]["intended_differences"][cid]
    datasets = sorted({b["dataset"] for b in blocks.values()})
    scope = {"bench": scope36["bench"], "datasets": {d: scope36["datasets"][d] for d in datasets},
             "traces": {TRACE: TRACE_DOC}, "configs": {c: scope36["configs"][c] for c in CONFIGS},
             "blocks": list(blocks.values()), "groups": groups, "accounted": {}, "pending": {}}
    write(render / "scope.json", scope)
    cells = [f"{c}__{TRACE}__s1" for b in blocks.values() for c in b["configs"]]
    plan = {"format": plan36["format"],
            "rows": [{"id": "GPU", "kind": "cells", "prio": "must", "enabled": True, "why": None,
                      "about": "the end-to-end run on one GPU", "plan": f"{len(cells)} runs", "cells": cells,
                      "groups": sorted(groups), "pending": {}, "disabled_blocks": {}}],
            "groups": {g: {**plan36["groups"][g], "fleet": "gpu"} for g in groups},
            "fleets": {"gpu": {"family": "qwen", "about": "Qwen2.5-3B and 7B on one GPU",
                               "fleet_sha256": sha256(inp / "fleet.json"),
                               "fleet_digest": fleet_digest(load_fleet(inp / "fleet.json", inp / "hosts.json"))}},
            **{k: plan36[k] for k in ("server", "warmup", "drain", "scored_corpora", "datasets", "bench",
                                      "pending_groups")}}
    write(render / "plan.json", plan)
    # one Hub home for the engines, the scheduler and the client, under OUT: each model's blobs and snapshots linked
    # from the given homes, its refs copied, so nothing is written into those homes
    hub = out / "mydata" / "hf" / "hub"
    hub.mkdir(parents=True)
    for home in a.hf_home:
        for d in sorted((Path(home) / "hub").glob("models--*")):
            if (hub / d.name).exists():
                continue
            (hub / d.name).mkdir()
            for part in ("blobs", "snapshots"):
                (hub / d.name / part).symlink_to((d / part).resolve())
            if (d / "refs").is_dir():
                shutil.copytree(d / "refs", hub / d.name / "refs")
    # the client loads its tokenizer by name, offline: refs/main names the revision the engine arguments pin
    for t, spec in engine_args["tiers"].items():
        ref = hub / f"models--{fleet['tiers'][t]['model'].replace('/', '--')}" / "refs" / "main"
        ref.parent.mkdir(exist_ok=True)
        ref.write_text(spec["revision"])
    print(f"built {len(cells)} runs in {render}")


def node_tree(tree: Path, dest: Path, mydata: Path) -> None:
    """A copy of a tree whose node layout names this machine's data folder and CUDA."""
    shutil.copytree(tree, dest, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", ".venv*", ".git"))
    layout = dest / "scripts" / "node_layout.sh"
    text = layout.read_text()
    for old, new in (("MYDATA=/mydata\n", f"MYDATA={mydata}\n"),
                     ("CUDA_DIR=/usr/local/cuda-12.8\n", "CUDA_DIR=/usr/local/cuda\n")):
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    layout.write_text(text)


def layout_values(tree: Path) -> dict:
    out = subprocess.run(["bash", "-c", 'source scripts/node_layout.sh && printf "%s\\n%s\\n%s\\n" "$RB_PATH" '
                          '"$RB_HF_HOME" "$CUDA_DIR"'], cwd=tree, capture_output=True, text=True, check=True)
    path, hf, cuda = out.stdout.split("\n")[:3]
    return {"RB_PATH": path, "RB_HF_HOME": hf, "CUDA_DIR": cuda}


# ---------------------------------------------------------------- the engines

def engines_up(a) -> None:
    out = Path(a.out)
    inp = out / "inputs"
    tree = out / "layout"
    if not tree.exists():
        node_tree(Path(a.tree), tree, out / "mydata")
    lay = layout_values(tree)
    plan = subprocess.run([sys.executable, "-m", "tools.engine_plan", "engines", "--fleet", str(inp / "fleet.json"),
                           "--hosts", str(inp / "hosts.json"), "--engine-args", str(inp / "engine_args.json")],
                          cwd=tree, capture_output=True, text=True, check=True).stdout.splitlines()
    record = out / "ENGINES.record"
    lines = ["# deploy.sh engines (the end-to-end harness, node_ctl.sh's launch)"] + plan
    pids = {}
    logs = out / "logs"
    logs.mkdir(exist_ok=True)
    for line in plan:
        iid, node, address, port, gpus, *_rest = line.split("\t")
        argv = line.split("\t")[8].split(" ")
        env = {"HOME": os.environ["HOME"], "PATH": f"{Path(a.engine_venv) / 'bin'}:{lay['RB_PATH']}", "LANG": "C.UTF-8",
               "CUDA_HOME": lay["CUDA_DIR"], "CUDA_VISIBLE_DEVICES": gpus, "HF_HOME": lay["RB_HF_HOME"],
               "HF_HUB_OFFLINE": "1", "VLLM_SERVER_DEV_MODE": "1", "VLLM_HTTP_TIMEOUT_KEEP_ALIVE": KEEP_ALIVE_S}
        with open(logs / f"engine.{iid}.log", "w") as log:
            p = subprocess.Popen([str(Path(a.engine_venv) / "bin" / "python"), "-u", "-m",
                                  "vllm.entrypoints.openai.api_server", *argv], env=env, stdout=log,
                                 stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        pids[iid] = (p.pid, port, gpus, argv)
        (out / "engines.pids").write_text(json.dumps({k: v[0] for k, v in pids.items()}))   # engines-down stops it
        for _ in range(240):                       # one at a time: each takes its memory share before the next
            try:
                get(f"http://127.0.0.1:{port}/health", 5)
                break
            except OSError:
                if p.poll() is not None:
                    sys.exit(f"{iid} exited while loading; see {logs / f'engine.{iid}.log'}")
                time.sleep(5)
        else:
            sys.exit(f"{iid} is not healthy after 20 minutes")
    for iid, (pid, port, gpus, _argv) in pids.items():   # node_ctl.sh started, once all are healthy
        start = [ln.split()[1] for ln in get(f"http://127.0.0.1:{port}/metrics").splitlines()
                 if ln.startswith("process_start_time_seconds ")]
        env = dict(x.split("=", 1) for x in Path(f"/proc/{pid}/environ").read_bytes().decode().split("\0") if "=" in x)
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().decode().replace("\0", " ").rstrip()
        assert len(start) == 1 and env["CUDA_VISIBLE_DEVICES"] == gpus
        lines.append("\t".join(["started", iid, start[0], env["CUDA_VISIBLE_DEVICES"],
                                f"VLLM_HTTP_TIMEOUT_KEEP_ALIVE={env['VLLM_HTTP_TIMEOUT_KEEP_ALIVE']} {cmdline}"]))
    record.write_text("\n".join(lines) + "\n")
    print(f"engines up: {sorted(pids)}; record {record}")


def engines_down(a) -> None:
    record = Path(a.out) / "engines.pids"
    if not record.exists():
        print("no engine was started")
        return
    pids = json.loads(record.read_text())
    for pid in pids.values():
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    time.sleep(10)
    for pid in pids.values():
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    print("engines down")


# ---------------------------------------------------------------- the runs

def scheduler_start(a, tree: Path, run_dir: Path, log_dir: Path, lay: dict) -> subprocess.Popen:
    """As node_ctl.sh sched-start: its environment and arguments, a new log directory."""
    inp = Path(a.out) / "inputs"
    argv = ["--fleet", str(inp / "fleet.json"), "--hosts", str(inp / "hosts.json"), "--config",
            str(run_dir / "scheduler.json"), "--artifacts", str(a.artifacts), "--avengers-repo", str(a.avengers_repo),
            "--best-route-repo", str(a.best_route_repo), "--host", "127.0.0.1", "--port", str(SCHEDULER_PORT),
            "--request-timeout-s", "21600.0", "--max-connections", "4096", "--log-dir", str(log_dir)]
    env = {"HOME": os.environ["HOME"], "PATH": lay["RB_PATH"], "LANG": "C.UTF-8", "CUDA_VISIBLE_DEVICES": "",
           "HF_HOME": lay["RB_HF_HOME"], "HF_HUB_OFFLINE": "1", "TIKTOKEN_CACHE_DIR": str(a.tiktoken)}
    log_dir.mkdir(parents=True)
    log = open(run_dir / f"scheduler.{log_dir.name}.log", "w")
    p = subprocess.Popen([a.scheduler_python, "-u", "-m", "routebalance.server", *argv], cwd=tree, env=env,
                         stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    for _ in range(240):
        try:
            get(f"http://127.0.0.1:{SCHEDULER_PORT}/v1/config", 5)
            return p
        except OSError:
            if p.poll() is not None:
                sys.exit(f"the scheduler exited while starting; see {log.name}")
            time.sleep(2)
    sys.exit("the scheduler is not healthy after 8 minutes")


def scheduler_stop(p: subprocess.Popen) -> None:
    os.killpg(p.pid, signal.SIGTERM)
    try:
        p.wait(60)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        p.wait()


def run(a) -> None:
    # the benchmark client needs more open files than its connections (bench/client.py check_fd_limit); its message
    # says to raise the limit before the run, as `ulimit -n` would: the soft limit to the hard one, for every child
    resource.setrlimit(resource.RLIMIT_NOFILE, (resource.getrlimit(resource.RLIMIT_NOFILE)[1],) * 2)
    out = Path(a.out)
    inp = out / "inputs"
    base = out / a.kind
    tree = base / "code"
    node_tree(Path(a.tree), tree, out / "mydata")
    lay = layout_values(tree)
    render = inp / "render"
    plan = json.loads((render / "plan.json").read_text())
    cells, root, mirror = plan["rows"][0]["cells"], base / "cells", base / "mirror"
    root.mkdir()
    mirror.mkdir()
    run_dir = base / "scheduler"
    run_dir.mkdir()
    shutil.copyfile(render / "configs" / f"{cells[0].split('__')[0]}.json", run_dir / "scheduler.json")
    results = {}
    for cell in cells:
        log_dir = base / "logs" / f"{cell}.{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
        p = scheduler_start(a, tree, run_dir, log_dir, lay)
        args = ["--render", str(render), "--cell", cell, "--scheduler", f"http://127.0.0.1:{SCHEDULER_PORT}",
                "--scheduler-log-dir", str(log_dir), "--fleet", str(inp / "fleet.json"), "--hosts", str(inp / "hosts.json"),
                "--engines-record", str(out / "ENGINES.record"), "--data-root", str(a.data_root), "--cells-root",
                str(root), "--mirror", str(mirror)]
        if a.kind == "ref":
            cmd = [a.scheduler_python, "-m", "runner", "cell", *args]
            env = {**os.environ}
        else:
            cmd = ["bash", "scripts/run_cell.sh", *args]
            env = {**os.environ, "PY": a.scheduler_python}
        with open(base / f"{cell}.out", "w") as f:
            rc = subprocess.run(cmd, cwd=tree, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
        scheduler_stop(p)
        results[cell] = rc
        print(f"{a.kind} {cell}: exit {rc}", flush=True)
    write(base / "results.json", results)


def compare(a) -> None:
    """Every run's verdict, and the choices that do not depend on timing, request by request: RB-decoupled's decision
    reason (its tier and score), and round robin's and random's instances."""
    out = Path(a.out)
    report = {}
    for kind in ("ref", "submission"):
        res = json.loads((out / kind / "results.json").read_text())
        for cell, rc in res.items():
            v = out / kind / "cells" / cell / "gates" / "verdict.json"
            report.setdefault(cell, {})[kind] = {"exit": rc, "valid": json.loads(v.read_text())["valid"] if v.exists()
                                                 else None}
    same = {}
    for cell in report:
        lines = {}
        for kind in ("ref", "submission"):
            d = out / kind / "cells" / cell / "decisions.jsonl"
            if d.exists():
                rows = [json.loads(x) for x in d.read_text().splitlines()]
                lines[kind] = {r["request_id"].rsplit(".", 1)[-1]: r for r in rows}
        if len(lines) != 2:
            continue
        keys = ("reason",) if "rb-static-dec" in cell else ()     # rb_decoupled:tier=...:score=... (6 decimals)
        if any(x in cell for x in (".disp.rr", ".disp.random")):
            keys += ("instance_id",)
        if keys:
            common = sorted(set(lines["ref"]) & set(lines["submission"]))
            diff = [k for k in common if any(lines["ref"][k].get(f) != lines["submission"][k].get(f) for f in keys)]
            same[cell] = {"fields": keys, "requests": len(common), "differ": len(diff)}
    write(out / "compare.json", {"runs": report, "same_choices": same})
    print(json.dumps({"runs": report, "same_choices": same}, indent=1))


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    for f in ("tree", "out", "artifacts", "data-root"):
        b.add_argument(f"--{f}", required=True)
    b.add_argument("--hf-home", action="append", required=True)
    e = sub.add_parser("engines-up")
    for f in ("out", "tree", "engine-venv"):
        e.add_argument(f"--{f}", required=True)
    sub.add_parser("engines-down").add_argument("--out", required=True)
    r = sub.add_parser("run")
    for f in ("out", "tree", "kind", "artifacts", "data-root", "avengers-repo", "best-route-repo", "tiktoken",
              "scheduler-python"):
        r.add_argument(f"--{f}", required=True)
    sub.add_parser("compare").add_argument("--out", required=True)
    a = ap.parse_args(argv)
    for name in ("tree", "out", "artifacts", "data_root", "engine_venv", "avengers_repo", "best_route_repo", "tiktoken"):
        if getattr(a, name, None) is not None:            # the steps change directory: every path absolute
            setattr(a, name, str(Path(getattr(a, name)).resolve()))
    if getattr(a, "hf_home", None):
        a.hf_home = [str(Path(h).resolve()) for h in a.hf_home]
    if getattr(a, "scheduler_python", None):          # absolute, its links kept: a venv's python is a link
        a.scheduler_python = os.path.abspath(a.scheduler_python)
    tree = getattr(a, "tree", None)
    if tree is not None and Path(a.out).is_relative_to(tree):
        sys.exit(f"{a.out} is inside the tree {tree}, which the runs copy")
    {"build": build, "engines-up": engines_up, "engines-down": engines_down, "run": run, "compare": compare}[a.cmd](a)


if __name__ == "__main__":
    main(sys.argv[1:])
