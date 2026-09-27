"""Engine deployment plan: fleet x hosts x engine args -> one vLLM launch per instance.

The scripts in scripts/ never parse JSON themselves; they ask this module, which validates every document
first (a missing or unknown key raises, nothing is defaulted).

Engine args (configs/engine_args_<family>.json), one file per fleet:
  {"vllm": "<version>",
   "tiers": {<tier_id>: {"revision": <Hub commit, 40 hex>,
                         "args": {"--flag": <number | string | true>, ...}}},     every flag the engine gets
   "gpus": {<instance_id>: [<GPU index on its node>, ...]}}                       CUDA_VISIBLE_DEVICES
The tier and instance sets must equal the fleet's. The model (from the fleet tier), --revision, --port (from
the fleet instance) and --max-num-seqs (the fleet tier's max_num_seqs, which the telemetry view also reads)
are added here and may not appear in "args"; --tokenizer-revision is left to vLLM, which takes the model's
revision (vllm/config/model.py). --dtype, --tensor-parallel-size, --gpu-memory-utilization and
--max-model-len are required; the tensor-parallel size must equal the instance's GPU count, the GPUs must
exist on the node (routebalance.hw) and no two instances of a node may share one. `true` is a switch
(--enforce-eager); there is no false: an absent switch is vLLM's default and the engine's resolved config
is the record of it.

The scheduler launch file (configs/scheduler_launch.example.json; the real one under local/) holds the D44
parameters, identical for every system: {"host", "port", "torch_threads", "max_connections",
"request_timeout_s"}. An OWNER-MUST-SET placeholder, or any other non-number, is refused.

Usage:
  python -m tools.engine_plan engines --fleet F --hosts H --engine-args E [--node N]
      one tab-separated line per instance: instance_id node address port gpus gpu_class gpus_on_node model argv
  python -m tools.engine_plan models  --fleet F --hosts H --engine-args E --node N
      the node's <model>@<revision> specs for scripts/setup.sh --model
  python -m tools.engine_plan nodes   --fleet F --hosts H
      one line per node that serves an instance: node address hw gpu_class gpus_on_node
  python -m tools.engine_plan launch  --launch L
      host port torch_threads max_connections request_timeout_s
  python -m tools.engine_plan hub     --file configs/scheduler_hub.json
      the scheduler node's setup.sh arguments: "--model<TAB>spec" and "--tokenizer<TAB>spec" lines
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Union

from pydantic import Field, StrictBool, StrictFloat, StrictInt, StrictStr, field_validator

from bench.schema import Strict, parse, read_json
from routebalance.config import ConfigError, load_fleet
from routebalance.hw import gpu_of, gpus_per_node
from routebalance.types import Instance

REQUIRED_FLAGS = ("--dtype", "--tensor-parallel-size", "--gpu-memory-utilization", "--max-model-len")
PLAN_FLAGS = frozenset({"--model", "--revision", "--port", "--max-num-seqs",           # added by this module
                        "--tokenizer-revision", "--tokenizer", "--served-model-name", "--host", "-tp"})
_FLAG = re.compile(r"--[a-z0-9][a-z0-9-]*")
_TOKEN = re.compile(r"[A-Za-z0-9._/:=,@+-]+")      # argv tokens travel space-separated through the scripts
Value = Union[StrictBool, StrictInt, StrictFloat, StrictStr]


class _TierArgs(Strict):
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    args: dict[str, Value]


class _EngineArgs(Strict):
    vllm: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    tiers: dict[str, _TierArgs] = Field(min_length=1)
    gpus: dict[str, list[int]] = Field(min_length=1)


_SPEC = r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*@[0-9a-f]{40}$"   # == setup.sh's SPEC_RE


class _Hub(Strict):
    models: list[str] = Field(min_length=1)       # full snapshots (sentence encoders)
    tokenizers: list[str]                         # tokenizer files only (the benchmark's reference tokenizer)

    @field_validator("models", "tokenizers")
    @classmethod
    def _pinned(cls, v: list[str]) -> list[str]:
        bad = [s for s in v if not re.fullmatch(_SPEC, s)]
        if bad:
            raise ValueError(f"{bad} are not ORG/NAME@<40-hex Hub commit>")
        return v


class _Launch(Strict):
    host: str = Field(min_length=1)
    port: int = Field(gt=0, lt=65536)
    torch_threads: int = Field(ge=1)
    max_connections: int = Field(ge=1)
    request_timeout_s: float = Field(gt=0)


@dataclass(frozen=True)
class EngineLaunch:
    instance: Instance
    hw: str
    gpus: tuple[int, ...]
    revision: str
    argv: tuple[str, ...]          # the vLLM server's arguments (after `-m vllm.entrypoints.openai.api_server`)

    @property
    def gpu_class(self) -> str:
        return gpu_of(self.hw)


def _render_value(flag: str, v) -> list[str]:
    if isinstance(v, bool):
        if not v:
            raise ConfigError(f"{flag}: false is not a flag; leave the switch out to keep vLLM's default")
        return [flag]
    return [flag, str(v)]


def _tier_argv(tier_id: str, spec: _TierArgs) -> list[str]:
    missing = [f for f in REQUIRED_FLAGS if f not in spec.args]
    if missing:
        raise ConfigError(f"tier {tier_id!r}: engine args lack {missing}")
    out: list[str] = []
    for flag, v in spec.args.items():
        if not _FLAG.fullmatch(flag):
            raise ConfigError(f"tier {tier_id!r}: {flag!r} is not a long option")
        if flag in PLAN_FLAGS:
            raise ConfigError(f"tier {tier_id!r}: {flag} comes from the fleet and the plan, not the engine args")
        out += _render_value(flag, v)
    return out


def plan(fleet_path: Path, hosts_path: Path, engine_args_path: Path) -> list[EngineLaunch]:
    """Every instance's launch, in fleet order."""
    fleet = load_fleet(fleet_path, hosts_path)
    hosts = read_json(hosts_path)                  # load_fleet validated it; only the hardware type is needed here
    ea = parse(_EngineArgs, read_json(engine_args_path), f"engine args {engine_args_path}")
    tiers = {i.tier.tier_id for i in fleet}
    if set(ea.tiers) != tiers:
        raise ConfigError(f"{engine_args_path}: tiers {sorted(ea.tiers)} != the fleet's {sorted(tiers)}")
    ids = [i.instance_id for i in fleet]
    if set(ea.gpus) != set(ids):
        raise ConfigError(f"{engine_args_path}: gpus cover {sorted(ea.gpus)}, the fleet has {sorted(ids)}")
    used: dict[str, set[int]] = {}
    out = []
    for inst in fleet:
        spec = ea.tiers[inst.tier.tier_id]
        hw = hosts[inst.node]["hw"]
        gpus = ea.gpus[inst.instance_id]
        n_node = gpus_per_node(hw)
        if not gpus or len(set(gpus)) != len(gpus) or not all(0 <= g < n_node for g in gpus):
            raise ConfigError(f"instance {inst.instance_id!r}: gpus {gpus} must be distinct indices below {n_node} "
                              f"({hw} has {n_node} GPU(s))")
        taken = used[inst.node] if inst.node in used else set()
        if taken & set(gpus):
            raise ConfigError(f"instance {inst.instance_id!r}: GPUs {sorted(taken & set(gpus))} of node "
                              f"{inst.node!r} are already assigned")
        used[inst.node] = taken | set(gpus)
        argv = ["--model", inst.model, "--revision", spec.revision, "--port", str(inst.port),
                "--max-num-seqs", str(inst.tier.max_num_seqs), *_tier_argv(inst.tier.tier_id, spec)]
        tp = spec.args["--tensor-parallel-size"]
        if type(tp) is not int or tp != len(gpus):
            raise ConfigError(f"instance {inst.instance_id!r}: --tensor-parallel-size {tp!r} but {len(gpus)} GPU(s)")
        bad = [t for t in argv if not _TOKEN.fullmatch(t)]
        if bad:
            raise ConfigError(f"instance {inst.instance_id!r}: argv tokens {bad} contain characters the scripts "
                              f"cannot pass through")
        out.append(EngineLaunch(inst, hw, tuple(gpus), spec.revision, tuple(argv)))
    return out


def load_launch(path: Path) -> _Launch:
    raw = read_json(path)
    if isinstance(raw, dict):
        unset = sorted(k for k, v in raw.items() if v == "OWNER-MUST-SET")
        if unset:
            raise ConfigError(f"{path}: {unset} are OWNER-MUST-SET; the owner sets them (D44), nobody else")
    return parse(_Launch, raw, f"scheduler launch file {path}")


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="python -m tools.engine_plan", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("engines", "models", "nodes"):
        s = sub.add_parser(name)
        s.add_argument("--fleet", type=Path, required=True)
        s.add_argument("--hosts", type=Path, required=True)
        if name != "nodes":
            s.add_argument("--engine-args", type=Path, required=True)
        if name == "engines":
            s.add_argument("--node")
        if name == "models":
            s.add_argument("--node", required=True)
    s = sub.add_parser("launch")
    s.add_argument("--launch", type=Path, required=True)
    s = sub.add_parser("hub")
    s.add_argument("--file", type=Path, required=True)
    a = p.parse_args(argv)
    try:
        if a.cmd == "hub":
            hub = parse(_Hub, read_json(a.file), f"scheduler hub file {a.file}")
            for spec in hub.models:
                print("--model", spec, sep="\t")
            for spec in hub.tokenizers:
                print("--tokenizer", spec, sep="\t")
            return 0
        if a.cmd == "launch":
            x = load_launch(a.launch)
            print(x.host, x.port, x.torch_threads, x.max_connections, x.request_timeout_s, sep="\t")
            return 0
        if a.cmd == "nodes":
            fleet = load_fleet(a.fleet, a.hosts)
            hosts = read_json(a.hosts)
            seen: dict[str, None] = {}
            for i in fleet:
                if i.node not in seen:
                    seen[i.node] = None
                    hw = hosts[i.node]["hw"]
                    print(i.node, i.address, hw, gpu_of(hw), gpus_per_node(hw), sep="\t")
            return 0
        launches = plan(a.fleet, a.hosts, a.engine_args)
        if a.node is not None and a.node not in {x.instance.node for x in launches}:
            raise ConfigError(f"node {a.node!r} serves no instance of {a.fleet}")
        chosen = [x for x in launches if a.node is None or x.instance.node == a.node]
        if a.cmd == "models":
            specs = list(dict.fromkeys(f"{x.instance.model}@{x.revision}" for x in chosen))
            print("\n".join(specs))
            return 0
        for x in chosen:
            i = x.instance
            print(i.instance_id, i.node, i.address, i.port, ",".join(map(str, x.gpus)), x.gpu_class,
                  gpus_per_node(x.hw), i.model, " ".join(x.argv), sep="\t")
        return 0
    except ConfigError as e:
        print(f"engine_plan: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
