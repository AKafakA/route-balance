"""The engines a cell ran on, as each engine reports itself (the first rule checked on execution).

For every instance of the cell's fleet, read from the engine itself:
  start_time_s   process_start_time_seconds from its /metrics (vLLM's one API server exposes the default
                 Prometheus registry, whose process collector reports it): a restart changes it
  cache_config   the labels of vllm:cache_config_info (the KV cache the engine built: block size, dtype,
                 memory utilisation, number of GPU blocks, prefix caching, ...), without its engine index
  model, max_model_len    the one model /v1/models lists and its context length
and from the deploy record (scripts/deploy.sh engines --record), which holds each engine's planned arguments
and, written once all were healthy, what each executes (node_ctl.sh started): its process start time, its
CUDA_VISIBLE_DEVICES and its command line, read on the node from the process itself. The engine read for a cell
must be that process (the same start time): an engine relaunched any other way is refused, so the record's
command line is what runs. The executed command line must end in the planned arguments and its GPUs must be
the planned ones.
A cell is invalid when any of these changed between before and after (tools/cell_driver.py verdict); the parity
gate requires every cell of a group to have run on the same launches (everything but start_time_s).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Sequence

import aiohttp

from bench.dataset import file_sha256
from fmrlb.config import ConfigError
from fmrlb.types import Instance
from tools import direct

START = "process_start_time_seconds"
CACHE_INFO = "vllm:cache_config_info"
_LABEL = re.compile(r'\s*([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"\s*(,|$)')


SERVER = "-m vllm.entrypoints.openai.api_server"


def deploy_record(path: Path) -> dict[str, dict]:
    """{instance_id: {"address", "port", "gpus", "argv", "start_time_s", "cmdline"}} from deploy.sh's record:
    comment lines; one tab-separated line per engine, as tools/engine_plan.py prints it: instance, node, address,
    port, gpus, gpu class, GPUs per node, model, the vLLM server's arguments; and one line per engine
    "started", instance, process start time, CUDA_VISIBLE_DEVICES, command line (node_ctl.sh started)."""
    out: dict[str, dict] = {}
    started: dict[str, dict] = {}
    try:
        lines = Path(path).read_text().splitlines()
    except OSError as e:
        raise ConfigError(f"cannot read the deploy record {path}: {e}") from e
    for n, line in enumerate(lines, 1):
        if not line.strip() or line.startswith("#"):
            continue
        cols = line.split("\t")
        if cols[0] == "started":
            try:
                start = float(cols[2]) if len(cols) == 5 else math.nan
            except ValueError:
                start = math.nan
            if not math.isfinite(start) or start <= 0 or cols[1] in started:
                raise ConfigError(f"{path}:{n}: a started line is: started, instance, start time, GPUs, command "
                                  "line, once per engine")
            started[cols[1]] = {"start_time_s": start, "cuda_visible_devices": cols[3], "cmdline": cols[4]}
            continue
        if len(cols) != 9 or not cols[3].isdigit():
            raise ConfigError(f"{path}:{n}: a deploy record line has 9 tab-separated fields, the port an integer")
        if cols[0] in out:
            raise ConfigError(f"{path}:{n}: instance {cols[0]} is recorded twice")
        out[cols[0]] = {"address": cols[2], "port": int(cols[3]), "gpus": cols[4], "argv": cols[8]}
    if not out:
        raise ConfigError(f"{path}: the deploy record lists no engine")
    if set(started) != set(out):
        raise ConfigError(f"{path}: every engine needs its started line (written once all are healthy; a record "
                          f"without them is from an unfinished deploy): engines {sorted(out)}, started "
                          f"{sorted(started)}")
    for iid, rec in out.items():
        run = started[iid]
        if not run["cmdline"].endswith(f"{SERVER} {rec['argv']}"):
            raise ConfigError(f"{path}: {iid} executes {run['cmdline']!r}, not the planned arguments")
        if run["cuda_visible_devices"] != rec["gpus"]:
            raise ConfigError(f"{path}: {iid} runs on GPUs {run['cuda_visible_devices']}, planned {rec['gpus']}")
        rec.update(start_time_s=run["start_time_s"], cmdline=run["cmdline"])
    return out


def _one_line(instance_id: str, text: str, name: str) -> str:
    hits = [ln for ln in text.splitlines() if ln.startswith(name) and ln[len(name):len(name) + 1] in ("{", " ")]
    if len(hits) != 1:
        raise ConfigError(f"{instance_id}: /metrics has {len(hits)} {name} samples, not one")
    return hits[0]


def start_time(instance_id: str, text: str) -> float:
    line = _one_line(instance_id, text, START)
    rest = line[len(START):]
    if rest.startswith("{"):
        rest = rest[rest.find("}") + 1:] if "}" in rest else ""
    parts = rest.split()
    try:
        if len(parts) != 1:                            # exactly the value: no timestamp, nothing else
            raise ValueError
        value = float(parts[0])
    except ValueError:
        raise ConfigError(f"{instance_id}: malformed {START} sample {line[:80]!r}") from None
    if not math.isfinite(value) or value <= 0:
        raise ConfigError(f"{instance_id}: {START}={value} is not a time")
    return value


def cache_config(instance_id: str, text: str) -> dict[str, str]:
    line = _one_line(instance_id, text, CACHE_INFO)
    if not line.startswith(CACHE_INFO + "{") or "}" not in line:
        raise ConfigError(f"{instance_id}: {CACHE_INFO} has no labels")
    body = line[len(CACHE_INFO) + 1:line.rindex("}")]
    value = line[line.rindex("}") + 1:].split()
    try:
        if len(value) != 1 or not math.isfinite(float(value[0])):    # an info metric: one value, no timestamp
            raise ValueError
    except ValueError:
        raise ConfigError(f"{instance_id}: malformed {CACHE_INFO} sample {line[:80]!r}") from None
    labels, pos = {}, 0
    while pos < len(body):
        m = _LABEL.match(body, pos)
        if m is None:
            raise ConfigError(f"{instance_id}: malformed {CACHE_INFO} labels {body[pos:pos + 60]!r}")
        if m.group(1) in labels:
            raise ConfigError(f"{instance_id}: {CACHE_INFO} repeats label {m.group(1)}")
        labels[m.group(1)] = m.group(2)
        pos = m.end()
    labels.pop("engine", None)            # default-ok: the engine index, 0 for the one engine of an instance
    if not labels:
        raise ConfigError(f"{instance_id}: {CACHE_INFO} carries no configuration")
    return labels


async def _get(session: aiohttp.ClientSession, url: str, timeout_s: float, what: str):
    async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout_s)) as resp:
        if resp.status != 200:
            raise ConfigError(f"{what}: HTTP {resp.status} from {url}")
        return await resp.read()


async def engine(session: aiohttp.ClientSession, inst: Instance, record: dict, timeout_s: float) -> dict:
    if inst.instance_id not in record:
        raise ConfigError(f"the deploy record has no engine {inst.instance_id}")
    rec = record[inst.instance_id]
    if (rec["address"], rec["port"]) != (inst.address, inst.port):
        raise ConfigError(f"{inst.instance_id}: deployed at {rec['address']}:{rec['port']}, the fleet says "
                          f"{inst.address}:{inst.port}")
    try:
        text = (await _get(session, f"{inst.base_url}/metrics", timeout_s, inst.instance_id)).decode("utf-8")
        models = json.loads(await _get(session, f"{inst.base_url}/v1/models", timeout_s, inst.instance_id))
    except (aiohttp.ClientError, asyncio.TimeoutError, UnicodeDecodeError, ValueError) as e:
        raise ConfigError(f"{inst.instance_id}: cannot read the engine: {type(e).__name__}: {e}") from e
    data = models["data"] if isinstance(models, dict) and "data" in models else None
    if not isinstance(data, list) or len(data) != 1 or not isinstance(data[0], dict) \
            or type(data[0].get("max_model_len")) is not int or not isinstance(data[0].get("id"), str):
        raise ConfigError(f"{inst.instance_id}: /v1/models must list exactly one model with an id and an integer "
                          f"max_model_len, got {str(models)[:200]}")
    now = start_time(inst.instance_id, text)
    if now != rec["start_time_s"]:
        raise ConfigError(f"{inst.instance_id}: the engine answering started at {now}, not the process deploy.sh "
                          f"launched at {rec['start_time_s']} (relaunched outside that deploy run?)")
    return {"argv_sha256": hashlib.sha256(rec["argv"].encode()).hexdigest(), "argv": rec["argv"],
            "gpus": rec["gpus"], "cmdline": rec["cmdline"], "start_time_s": now,
            "cache_config": cache_config(inst.instance_id, text), "model": data[0]["id"],
            "max_model_len": data[0]["max_model_len"], "served": served(inst.instance_id, text)}


SERVED = {"requests": "vllm:request_success_total", "prompt_tokens": "vllm:prompt_tokens_total",
          "generation_tokens": "vllm:generation_tokens_total", "prefix_queries": "vllm:prefix_cache_queries_total",
          "prefix_hits": "vllm:prefix_cache_hits_total"}


def served(instance_id: str, text: str) -> dict:
    """the engine's cumulative serving counters, each summed over its series (request_success_total has one per
    finished_reason, abort and error included). The change over a cell must be the cell's own requests there."""
    got = {name: [] for name in SERVED.values()}
    for line in text.splitlines():
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name in got:
            got[name].append(float(line.rsplit(" ", 1)[1]))
    missing = [n for n, v in got.items() if not v]
    if missing:
        raise ConfigError(f"{instance_id}: /metrics has no {missing}")
    out = {k: sum(got[n]) for k, n in SERVED.items()}
    for v in out.values():
        if not (v == int(v) and v >= 0):
            raise ConfigError(f"{instance_id}: malformed serving counters {out}")
    return {k: int(v) for k, v in out.items()}


async def engines(fleet: Sequence[Instance], record_path: Path, timeout_s: float) -> dict:
    record = deploy_record(record_path)
    async with direct.session(2 * len(fleet)) as session:
        got = await asyncio.gather(*(engine(session, inst, record, timeout_s) for inst in fleet))
    return {"record_sha256": file_sha256(record_path),
            "instances": {inst.instance_id: e for inst, e in zip(fleet, got, strict=True)}}
