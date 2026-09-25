"""W0 (D39): generate every (prompt, model) reference once on the lease stack, and measure how often a response
changes with the load it is generated under.

    python -m tools.reference_generation --render DIR --row W0 --fleet FLEET.json --hosts HOSTS.json
                                         --data-root DIR --out DIR

The row (plan.json) names the fleet, the dataset (the evaluation set) and the loads. Requests go directly to
the engines, never through the scheduler, with the serving settings: the raw prompt as the single user message,
templated once by vLLM, and the scope's sampling (tools/direct.py). Per model, the first instance in fleet order:
  reference   every prompt of the dataset, closed loop at concurrency_per_instance (low load)
  divergence  the first divergence_prompts prompts of the seed-1 send order (bench/dataset.py), generated again
              at divergence_concurrency_per_instance, which must differ from the reference load; fillers (the
              other prompts, discarded) keep that load until the last sampled request has finished
The engines are warmed up and drained first, and drained between the passes. Output (a new directory, written as
the run goes, so a failure keeps the passes that finished):
  texts.jsonl       one line per request: pass, prompt_id, prompt, model, instance_id, text, text_sha256, and the
                    backend's outcome (success, error, token counts, timings); each written as it finishes
  divergence.json   per model: prompts sampled, pairs where both succeeded, identical texts, differing texts
                    and the differing rate. Every distinct text is a key of the score store (analysis/quality.py);
                    the rate sizes the judge's tail during the lease
  manifest.json     what ran: the row, fleet and dataset identity, instances, sampling, loads, times, counts, and
                    the error that stopped the run, if one did
tools/judge_served.py turns texts.jsonl (and the cells' served texts) into the judge's input. Exit 0 when every
request succeeded, the divergence load held (no filler failed or ran out) and the run finished, 1 otherwise.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from pathlib import Path

from analysis.quality import read_jsonl, text_sha256
from bench.dataset import prompt_order
from bench.records import Sampling
from routebalance.backend import BackendClient
from routebalance.config import ConfigError
from routebalance.types import Instance
from tools import direct
from tools.campaign import load_plan, plan_row


def one_per_model(fleet: list[Instance]) -> list[Instance]:
    """The first instance of every model, in fleet order."""
    first: dict[str, Instance] = {}
    for inst in fleet:
        if inst.model not in first:
            first[inst.model] = inst
    return list(first.values())


def divergence(texts: list[dict]) -> dict:
    """Per model, how often the divergence pass served a different text than the reference pass."""
    ref = {(t["model"], t["prompt_id"]): t for t in texts if t["pass"] == "reference"}
    out = {}
    for model in sorted({t["model"] for t in texts}):
        again = [t for t in texts if t["pass"] == "divergence" and t["model"] == model]
        pairs = [(ref[(model, t["prompt_id"])], t) for t in again if (model, t["prompt_id"]) in ref]
        both = [(a, b) for a, b in pairs if a["success"] and b["success"]]
        differ = sum(a["text_sha256"] != b["text_sha256"] for a, b in both)
        out[model] = {"sampled": len(again), "both_ok": len(both), "identical": len(both) - differ,
                      "different": differ, "different_rate": differ / len(both) if both else None}
    return out


def passes(row: dict, prompts: list[direct.Prompt]) -> list[tuple[str, list, int, list]]:
    """(name, measured prompts, concurrency, fillers) of the two passes."""
    p = row["params"]
    k, c_ref, c_div = p["divergence_prompts"], p["concurrency_per_instance"], p["divergence_concurrency_per_instance"]
    if not all(type(v) is int and v >= 1 for v in (k, c_ref, c_div)):
        raise ConfigError(f"the W0 loads and sample size must be ints >= 1, got {k}, {c_ref}, {c_div}")
    if c_ref == c_div:
        raise ConfigError(f"the divergence sample must be generated at another load than the references ({c_ref})")
    if k >= len(prompts):
        raise ConfigError(f"divergence_prompts {k} leaves no fillers among the {len(prompts)} prompts")
    order = [prompts[i] for i in prompt_order(len(prompts), "per_seed", 1)]
    return [("reference", prompts, c_ref, []), ("divergence", order[:k], c_div, order[k:])]


async def generate(plan: dict, row: dict, fleet: list[Instance], prompts: list[direct.Prompt], out: Path,
                   manifest: dict) -> list[dict]:
    """Both passes, appending to out/texts.jsonl as each finishes; `manifest` is filled as it goes."""
    sampling = Sampling.model_validate(plan["bench"]["sampling"])
    instances = one_per_model(fleet)
    todo = passes(row, prompts)
    run_id = uuid.uuid4().hex[:8]
    drain_s = plan["drain"]["timeout_s"]
    texts: list[dict] = []
    manifest.update(run_id=run_id, instances=[i.instance_id for i in instances], started_unix=time.time(),
                    load_not_held={})
    async with direct.session(max(c for _, _, c, _ in todo) * len(instances) + len(fleet)) as sess:
        backend = BackendClient(sess, plan["bench"]["request_timeout_s"])
        await direct.drain(sess, fleet, timeout_s=drain_s)
        w = plan["warmup"]
        manifest["warmup"] = await direct.warm_up(backend, instances, requests_per_instance=w["requests_per_instance"],
                                                  max_tokens=w["max_tokens"], prompt=w["prompt"], sampling=sampling,
                                                  run_id=run_id)
        for name, batch, conc, fillers in todo:
            await direct.drain(sess, fleet, timeout_s=drain_s)
            started = time.time()
            lines: list[dict] = []
            with open(out / "texts.jsonl", "a", encoding="utf-8") as f:

                def writer(pass_name, prompts_of_pass, sink, into):
                    def on_measured(i, res):      # each text is on disk as soon as it is generated
                        line = direct.outcome(res, **{"pass": pass_name}, prompt_id=prompts_of_pass[i].prompt_id,
                                              prompt=prompts_of_pass[i].text, text_sha256=text_sha256(res.text))
                        sink.write(json.dumps(line, ensure_ascii=False) + "\n")
                        sink.flush()
                        into.append(line)
                    return on_measured
                per = await asyncio.gather(*(direct.closed_loop(backend, inst, batch, conc, sampling, fillers=fillers,
                                                                request_prefix=f"{row['id']}.{run_id}.{name}.{inst.instance_id}",
                                                                on_measured=writer(name, batch, f, lines))
                                             for inst in instances), return_exceptions=True)
            for inst, run in zip(instances, per, strict=True):
                if isinstance(run, BaseException):
                    manifest["load_not_held"][f"{name}.{inst.instance_id}"] = f"{type(run).__name__}: {run}"
                    continue
                failed_fillers = sum(not f.success for f in run.fillers)
                if failed_fillers or run.fillers_ran_out:    # the divergence load was not kept to the end
                    manifest["load_not_held"][f"{name}.{inst.instance_id}"] = (
                        f"{failed_fillers} fillers failed" + ("; the fillers ran out" if run.fillers_ran_out else ""))
            texts += lines
            manifest[f"{name}_seconds"] = time.time() - started
            failures = [r for r in per if isinstance(r, BaseException)]
            if failures:                         # the finished instances of this pass are saved; stop here
                raise failures[0]
        await direct.drain(sess, fleet, timeout_s=drain_s)
    manifest["ended_unix"] = time.time()
    return texts


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--render", type=Path, required=True, help="the campaign render (tools/campaign.py)")
    ap.add_argument("--row", required=True, help="W0 or W8-ref")
    ap.add_argument("--fleet", type=Path, required=True)
    ap.add_argument("--hosts", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.out.exists():
        raise ConfigError(f"{a.out} exists; references are written to a new directory")
    plan = load_plan(a.render)
    row = plan_row(plan, a.row, "references")
    p = row["params"]
    fleet = direct.lease_fleet(plan, p["fleet"], a.fleet, a.hosts)
    prompts = direct.lease_prompts(plan, p["dataset"], a.data_root)
    passes(row, prompts)                                   # the loads are refused before anything is sent
    a.out.mkdir(parents=True)
    manifest: dict = {"row": row, "fleet": plan["fleets"][p["fleet"]], "dataset": plan["datasets"][p["dataset"]],
                      "sampling": plan["bench"]["sampling"], "error": None, "load_not_held": {}}
    try:
        texts = asyncio.run(generate(plan, row, fleet, prompts, a.out, manifest))
    except Exception as e:     # the passes that finished are on disk; the manifest says what stopped the run
        manifest["error"] = f"{type(e).__name__}: {e}"
        texts = [rec for _, rec in read_jsonl(a.out / "texts.jsonl")] if (a.out / "texts.jsonl").exists() else []
    div = divergence(texts)
    (a.out / "divergence.json").write_text(json.dumps(div, indent=1) + "\n")
    failed = sum(not t["success"] for t in texts)
    manifest |= {"requests": len(texts), "failed": failed}
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    for model, d in div.items():
        print(f"{model}: {d['different']}/{d['both_ok']} texts changed with the load")
    print(f"{a.out}: {len(texts)} requests, {failed} failed" + (f"; STOPPED: {manifest['error']}"
                                                                if manifest["error"] else ""))
    return 0 if failed == 0 and manifest["error"] is None and not manifest["load_not_held"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
