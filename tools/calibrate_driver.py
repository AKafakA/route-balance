"""W1 (D37): measure TPOT at fixed running-batch sizes on every tier, directly on the engines.

    python -m tools.calibrate_driver --render DIR --row W1 --fleet FLEET.json --hosts HOSTS.json
                                     --data-root DIR --out DIR
    python -m tools.calibrate_tpot_curve --records DIR/records.jsonl --out TABLE.json

The row (plan.json) gives six concurrency levels, the number of measured requests per level and instance, and
the held-out prompt set (the training split, disjoint from the evaluation set). Every instance of the fleet is
measured at the same time, each on its own engine, never through the scheduler (tools/direct.py). Per level c,
in the order given: drain; each instance runs a closed loop of c requests in flight until its measured requests
have finished, with filler requests keeping c in flight to the end (tools/direct.py), so every measured request
decodes with c requests on its instance and is recorded with b = c. A level above a tier's max_num_seqs would
queue on the engine instead and is refused.

That the batch really was c is checked, per instance and level: the engines' /metrics are read every POLL_S
while the level runs, each instance's running batch only while its own closed loop runs (a faster tier finishes
its requests earlier and then idles until the level ends), and the level holds on an instance when its engine
preempted nothing (the KV cache fit c contexts), the median of its running-batch reads is c less at most floor(c
/ 32) (a closed loop's gauge dips while a finished request's slot refills: exact below 32, within 3 at 96; owner
09-25), no filler failed and the fillers did not run out before the last measured request finished. (A waiting
request alone is normal: vLLM keeps arrivals waiting while a step's token budget goes to prefill.) An instance-
level that does not hold is recorded but gives no calibration records, and the run exits 1.

Prompts: the dataset in its seed-1 send order (bench/dataset.py). Every prompt is sent at most once per engine,
so no measured decode step shares the engine with a prefix-cache hit that serving would not have: level j gives
instance k of every tier (k = its position among the tier's instances in fleet order) the slice
[(j*K + k)*N, (j*K + k + 1)*N), with K the largest tier's instance count, and each level has its own range of
fillers after all measured slices. Instances of different tiers get the same slices (they are different
engines), so the tiers see the same prompts. The engines are warmed up and drained first.

A request's TPOT is (e2e - ttft) / (output_tokens - 1) in s/token, on the backend client's clock: per token and
robust to vLLM merging several tokens into one streamed delta, which a mean chunk gap is not (a first delta that
already carries several tokens makes it slightly low). The scheduler's D38 sweep logs the same quantity.
Output (a new directory, written as the run goes, so a failure keeps what finished):
  requests.jsonl  every measured request and every finished filler: tier, instance, level, b, held, tpot_s,
                  filler, and the backend's outcome; appended level by level
  records.jsonl   the calibration records tools/calibrate_tpot_curve.py reads: {tier, b, tpot_s, output_tokens}
                  for each successful measured request of a held instance-level with at least 2 output tokens
  manifest.json   what ran: row, fleet and dataset identity, levels, requests per level, prompt slices, sampling,
                  per level and instance the engine gauges seen and whether the level held, times; and the error
                  that stopped the run, if one did
Exit 0 when every measured request succeeded and every instance-level held, 1 otherwise.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from pathlib import Path

import aiohttp

from bench.schema import read_jsonl
from bench.dataset import prompt_order
from bench.records import Sampling
from routebalance.backend import BackendClient
from routebalance.config import ConfigError
from routebalance.telemetry import fetch_metrics, parse_metrics
from routebalance.types import Instance
from tools import direct
from tools.campaign import load_plan, plan_row

POLL_S = 0.5            # how often the engines' gauges are read during a level


def tpot_s(latency_s: float, ttft_s: float, output_tokens: int):
    """Per-token decode time of one request; None when it decoded fewer than 2 tokens."""
    if output_tokens < 2:
        return None
    return (latency_s - ttft_s) / (output_tokens - 1)


def positions(fleet: list[Instance]) -> dict[str, int]:
    """instance id -> its position among its tier's instances, in fleet order."""
    seen: dict[str, int] = {}
    out = {}
    for inst in fleet:
        k = seen[inst.tier.tier_id] if inst.tier.tier_id in seen else 0
        seen[inst.tier.tier_id] = k + 1
        out[inst.instance_id] = k
    return out


def slices(fleet: list[Instance], n: int, levels: int, rows: int) -> tuple[dict, list[tuple[int, int]]]:
    """({level index: {instance id: [start, end)}}, [each level's filler range]) in the prompt order."""
    pos = positions(fleet)
    k_max = max(pos.values()) + 1
    measured = {j: {i: ((j * k_max + k) * n, (j * k_max + k + 1) * n) for i, k in pos.items()} for j in range(levels)}
    first = levels * k_max * n
    per = (rows - first) // levels
    if per < 1:
        raise ConfigError(f"{rows} prompts leave none for fillers after {first} measured ones")
    return measured, [(first + j * per, first + (j + 1) * per) for j in range(levels)]


def check_levels(levels: list, fleet: list[Instance]) -> list[int]:
    if len(levels) != 6 or len(set(levels)) != 6 or not all(type(c) is int and c >= 1 for c in levels):
        raise ConfigError(f"W1 takes six distinct concurrency levels >= 1, got {levels}")
    over = sorted({i.tier.tier_id for i in fleet for c in levels if c > i.tier.max_num_seqs})
    if over:
        raise ConfigError(f"levels above max_num_seqs would queue on the engine, not batch: tiers {over}")
    return list(levels)


class Gauges:
    """The engines' waiting and preemption gauges, read every POLL_S while a level runs."""

    def __init__(self, sess: aiohttp.ClientSession, fleet: list[Instance], level: int):
        self._sess, self._fleet, self._level = sess, fleet, level
        self.seen: dict[str, dict] = {i.instance_id: {"running": [], "max_waiting": 0} for i in fleet}
        self.active = {i.instance_id for i in fleet}   # instances whose closed loop still runs
        self.error: str | None = None       # a read that failed while the level ran: nothing is known to hold

    async def read(self, during: bool) -> None:
        """One read of every engine; `during` reads (while the level runs) give the running-batch sample."""
        texts = await asyncio.gather(*(fetch_metrics(self._sess, i, direct.SCRAPE_TIMEOUT_S) for i in self._fleet))
        for inst, text in zip(self._fleet, texts, strict=True):
            snap = parse_metrics(inst.instance_id, text)
            s = self.seen[inst.instance_id]
            if "preemptions_start" not in s:
                s["preemptions_start"] = snap.num_preemptions
            s["preemptions_end"] = snap.num_preemptions
            if during and inst.instance_id in self.active:
                s["running"].append(snap.num_running)
                s["max_waiting"] = max(s["max_waiting"], snap.num_waiting)

    async def run(self) -> None:
        try:
            while True:
                await asyncio.sleep(POLL_S)
                await self.read(during=True)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"

    def held(self, instance_id: str) -> bool:
        """No preemption, and the engine's running batch was the level: the median of the reads taken while the
        instance's closed loop ran is c, less at most floor(c / 32). (A waiting request alone does not break it:
        vLLM keeps new arrivals waiting while a step's token budget is spent on prefill, at every level.)"""
        s = self.seen[instance_id]
        runs = sorted(s["running"])
        c = self._level
        return self.error is None and bool(runs) and "preemptions_end" in s \
            and s["preemptions_end"] == s["preemptions_start"] and c - c // 32 <= runs[len(runs) // 2] <= c


async def measured_loop(backend: BackendClient, inst: Instance, prompts: list, c: int, sampling: Sampling,
                        prefix: str, fillers: list, gauges: Gauges):
    """One instance's closed loop at level c; when it ends, its batch is no longer the level (Gauges.active)."""
    try:
        return await direct.closed_loop(backend, inst, prompts, c, sampling, request_prefix=prefix, fillers=fillers)
    finally:
        gauges.active.discard(inst.instance_id)


def records(lines: list[dict]) -> list[dict]:
    """The calibration records: measured, successful, from a held instance-level, at least 2 output tokens."""
    return [{"tier": x["tier"], "b": x["b"], "tpot_s": x["tpot_s"], "output_tokens": x["output_tokens"]}
            for x in lines if not x["filler"] and x["held"] and x["success"] and x["tpot_s"] is not None]


async def calibrate(plan: dict, row: dict, fleet: list[Instance], prompts: list[direct.Prompt], out: Path,
                    manifest: dict) -> list[dict]:
    """Run the levels, appending to out/requests.jsonl as each finishes; `manifest` is filled as it goes."""
    p = row["params"]
    sampling = Sampling.model_validate(plan["bench"]["sampling"])
    levels = check_levels(p["levels"], fleet)
    n = p["requests_per_level"]
    if not (type(n) is int and n >= 1):
        raise ConfigError(f"requests_per_level must be an int >= 1, got {n!r}")
    order = [prompts[i] for i in prompt_order(len(prompts), "per_seed", 1)]
    measured, filler_ranges = slices(fleet, n, len(levels), len(order))
    run_id = uuid.uuid4().hex[:8]
    drain_s = plan["drain"]["timeout_s"]
    lines: list[dict] = []
    manifest.update(run_id=run_id, levels=levels, requests_per_level=n, per_level={}, started_unix=time.time(),
                    slices={str(c): measured[j] for j, c in enumerate(levels)},
                    fillers={str(c): filler_ranges[j] for j, c in enumerate(levels)})
    async with direct.session(max(levels) * len(fleet) + len(fleet)) as sess:
        backend = BackendClient(sess, plan["bench"]["request_timeout_s"])
        await direct.drain(sess, fleet, timeout_s=drain_s)
        w = plan["warmup"]
        warm = BackendClient(sess, drain_s)      # the warm-up's own bound: the owner's kept pre-run bound (D134 item 10)
        manifest["warmup"] = await direct.warm_up(sess, warm, fleet, requests_per_instance=w["requests_per_instance"],
                                                  max_tokens=w["max_tokens"], prompt=w["prompt"], sampling=sampling,
                                                  run_id=run_id)
        for j, c in enumerate(levels):
            await direct.drain(sess, fleet, timeout_s=drain_s)
            started = time.time()
            gauges = Gauges(sess, fleet, c)
            await gauges.read(during=False)                                 # the preemption counters before the level
            watcher = asyncio.create_task(gauges.run())
            fill = order[filler_ranges[j][0]:filler_ranges[j][1]]
            try:
                runs = await asyncio.gather(*(
                    measured_loop(backend, inst, order[measured[j][inst.instance_id][0]:measured[j][inst.instance_id][1]],
                                  c, sampling, f"{row['id']}.{run_id}.c{c}.{inst.instance_id}", fill, gauges)
                    for inst in fleet), return_exceptions=True)
            finally:
                watcher.cancel()
                await asyncio.gather(watcher, return_exceptions=True)
            try:
                await gauges.read(during=False)
            except Exception as e:           # the counters after the level are unknown: nothing held
                gauges.error = f"{type(e).__name__}: {e}"
            level_lines, per_instance = [], {}
            for inst, run in zip(fleet, runs, strict=True):
                if isinstance(run, BaseException):
                    per_instance[inst.instance_id] = {"error": f"{type(run).__name__}: {run}"}
                    continue
                held = gauges.held(inst.instance_id) and not run.fillers_ran_out \
                    and all(f.success for f in run.fillers)
                per_instance[inst.instance_id] = {**gauges.seen[inst.instance_id], "held": held,
                                                  "failed_fillers": sum(not f.success for f in run.fillers),
                                                  "fillers_ran_out": run.fillers_ran_out}
                for res in run.measured:
                    level_lines.append(direct.outcome(res, tier=inst.tier.tier_id, level=c, b=c, held=held, filler=False,
                                                      tpot_s=tpot_s(res.latency_s, res.ttft_s, res.output_tokens)
                                                      if res.success else None))
                for res in run.fillers:
                    level_lines.append(direct.outcome(res, tier=inst.tier.tier_id, level=c, b=c, held=held, filler=True,
                                                      tpot_s=None))
            with open(out / "requests.jsonl", "a", encoding="utf-8") as f:
                for line in level_lines:
                    f.write(json.dumps(line, ensure_ascii=False) + "\n")
            lines += level_lines
            manifest["per_level"][str(c)] = {"seconds": time.time() - started, "instances": per_instance,
                                             "gauge_error": gauges.error}
            failures = [r for r in runs if isinstance(r, BaseException)]
            if failures:                         # the finished instances of this level are saved; stop here
                raise failures[0]
        await direct.drain(sess, fleet, timeout_s=drain_s)
    manifest["ended_unix"] = time.time()
    return lines


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--render", type=Path, required=True, help="the campaign render (tools/campaign.py)")
    ap.add_argument("--row", required=True, help="W1 or W8-cal")
    ap.add_argument("--fleet", type=Path, required=True)
    ap.add_argument("--hosts", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.out.exists():
        raise ConfigError(f"{a.out} exists; a calibration is written to a new directory")
    plan = load_plan(a.render)
    row = plan_row(plan, a.row, "calibration")
    p = row["params"]
    fleet = direct.lease_fleet(plan, p["fleet"], a.fleet, a.hosts)
    prompts = direct.lease_prompts(plan, p["dataset"], a.data_root)
    a.out.mkdir(parents=True)
    manifest: dict = {"row": row, "fleet": plan["fleets"][p["fleet"]], "dataset": plan["datasets"][p["dataset"]],
                      "sampling": plan["bench"]["sampling"], "error": None}
    lines: list[dict] = []
    try:
        lines = asyncio.run(calibrate(plan, row, fleet, prompts, a.out, manifest))
    except Exception as e:     # what finished is on disk; the manifest says what stopped the run
        manifest["error"] = f"{type(e).__name__}: {e}"
        lines = [rec for _, rec in read_jsonl(a.out / "requests.jsonl")] if (a.out / "requests.jsonl").exists() else []
    with open(a.out / "records.jsonl", "w", encoding="utf-8") as f:
        for r in records(lines):
            f.write(json.dumps(r) + "\n")
    measured = [x for x in lines if not x["filler"]]
    failed = sum(not x["success"] for x in measured)
    not_held = sorted({f"{x['instance_id']}@{x['level']}" for x in measured if not x["held"]})
    manifest |= {"measured": len(measured), "failed": failed, "not_held": not_held, "records": len(records(lines))}
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"{a.out}: {len(measured)} measured requests, {failed} failed, {len(not_held)} instance-levels not held, "
          f"{manifest['records']} records" + (f"; STOPPED: {manifest['error']}" if manifest["error"] else ""))
    return 0 if failed == 0 and not not_held and manifest["error"] is None else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
