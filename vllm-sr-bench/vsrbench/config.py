"""vLLM-SR's configuration for one family: the operator's provider file and pool bindings, upstream's built-in Balance
policy generated from them by upstream's CLI, and the one edit (each pool's quality index).

The provider file holds only what an operator must supply: the models, their prices, their replicas, their cards and
the quality evidence. Everything else (signals, pools, orders, margins, the router's globals) comes from
`vllm-sr recipe builtin init balance --bundle mom-v1` unchanged.

Two profiles, never mixed:
  lease  the fleet file and the experiment network, the training-split means; refused if any test token appears
  test   the test machine's small engines behind Docker's bridge gateway, marked with TEST_MARK
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml

from routebalance.config import ConfigError, load_fleet
from vsrbench import constants as C


@dataclass(frozen=True)
class Backend:
    name: str          # the replica's name (the fleet's instance id)
    host: str          # where Envoy connects (the experiment network)
    port: int


@dataclass(frozen=True)
class Model:
    name: str                  # the served model id (the HF id vLLM serves and the fleet prices)
    prompt_per_1m: float       # USD per million tokens, the fleet's prices
    completion_per_1m: float
    context: int               # the engine's --max-model-len
    backends: tuple[Backend, ...]


def lease_models(fleet_path: Path, hosts_path: Path, engine_args_path: Path) -> list[Model]:
    """The family's models from the fleet file (prices, replicas in fleet order) and the engine arguments (context)."""
    fleet = load_fleet(fleet_path, hosts_path)
    args = json.loads(Path(engine_args_path).read_text())["tiers"]
    by_model: dict[str, list] = {}
    for inst in fleet:
        by_model.setdefault(inst.model, []).append(inst)       # default-ok: grouping
    models = []
    for name, insts in by_model.items():
        tiers = {i.tier.tier_id for i in insts}
        if len(tiers) != 1:
            raise ConfigError(f"{name} is served by several tiers {sorted(tiers)}: one card per model needs one tier")
        tier = insts[0].tier
        context = args[tier.tier_id]["args"]["--max-model-len"]
        models.append(Model(
            name=name, prompt_per_1m=tier.cost_in * 1e6, completion_per_1m=tier.cost_out * 1e6, context=int(context),
            backends=tuple(Backend(i.instance_id, i.service_address or i.address, i.port) for i in insts)))
    return models


def ladder_models(family: str) -> list[str]:
    """The family's models, smallest first (the pools' neighbouring pairs)."""
    seen: list[str] = []
    for pool in C.POOLS:
        for m in C.LADDER[family][pool]:
            if m not in seen:
                seen.append(m)
    return seen


def bindings(family: str) -> dict:
    return {pool: [{"model": m} for m in C.LADDER[family][pool]] for pool in C.POOLS}


def test_bindings() -> dict:
    """The test machine's test profile: one model on two replicas (the GPU the judge leaves holds two small engines)."""
    return {pool: [{"model": "m-small"}] for pool in C.POOLS}


def test_pair_bindings() -> dict:
    """The test machine's two-model test profile: the pools' selection between two models (the lease profile's path)."""
    return {"simple": [{"model": "m-small"}], "medium": [{"model": "m-small"}, {"model": "m-large"}],
            "reasoning": [{"model": "m-large"}]}


def _num(x: float) -> float:
    """A price or score as YAML writes it back (no float noise such as 0.060000000000000005)."""
    return float(f"{x:.12g}")


def provider_document(models: list[Model], means: dict[str, float], *, test: bool) -> dict:
    names = [m.name for m in models]
    if sorted(means) != sorted(names):
        raise ConfigError(f"quality means for {sorted(means)}, models {sorted(names)}")
    profile = {"id": C.PROFILE, "display_name": "Training split",
               "description": "Mean judged score of the served model on the training prompts"
                              + (f" ({C.TEST_MARK})" if test else "")}
    return {
        "version": "v0.3",
        "listeners": [{"name": f"http-{C.LISTENER_PORT}", "address": "0.0.0.0", "port": C.LISTENER_PORT,
                       "timeout": f"{C.LISTENER_TIMEOUT_S}s"}],
        "providers": {
            # the schema requires a default; upstream's config import takes the first model (cli/config_import.py:383).
            # The Balance policy's medium pool has no rule, so every entrypoint request matches a pool and the default
            # is never the routed model.
            "defaults": {"model": names[0]},
            "models": [{
                "name": m.name, "provider_model_id": m.name, "api_format": "openai",
                "pricing": {"prompt_per_1m": _num(m.prompt_per_1m), "completion_per_1m": _num(m.completion_per_1m),
                            "currency": "USD"},
                "backend_refs": [{"name": b.name, "provider": "vllm", "endpoint": f"{b.host}:{b.port}",
                                  "protocol": "http"} for b in m.backends],
            } for m in models],
        },
        "routing": {"modelCards": [{"name": m.name, "context_window_size": m.context, "max_output_tokens": m.context,
                                    "capabilities": ["chat"]} for m in models]},
        "evaluation": {
            "benchmarks": [{"id": C.BENCHMARK_ID, "display_name": "DeepEval on the training split",
                            "domain": "general_reasoning", "source": "https://example.invalid/route-balance",
                            "default_profile": C.PROFILE, "profiles": [profile],
                            "metrics": [{"id": "score", "unit": "proportion", "direction": "higher_is_better",
                                         "range": [0, 1]}]}],
            # the index's fields as upstream's stock vllm-sr/general@1.0.0 (cli/model_assets/v0.4/catalog.yaml): weighted
            # mean on a 0-100 scale, all components required, one domain, identity normalization of a 0-1 proportion; so
            # the pools' shipped tolerances apply on the scale they were set for
            "indices": [{"id": C.INDEX_ID, "display_name": "DeepEval quality", "aggregation": "weighted_mean",
                         "scale": [0, 100], "missing": {"policy": "require_all"},
                         "domains": {"general_reasoning": 1.0},
                         "components": [{"benchmark": C.BENCHMARK_ID, "benchmark_profile": C.PROFILE,
                                         "metric": "score", "weight": 1.0, "normalization": {"type": "identity"}}]}],
            "records": [{"model": m, "benchmark": C.BENCHMARK_ID, "benchmark_profile": C.PROFILE,
                         "metrics": {"score": _num(means[m])}} for m in names],
        },
    }


def test_tokens_in(text: str) -> list[str]:
    return [t for t in C.TEST_TOKENS if t in text]


def refuse_test_tokens(text: str, what: str) -> None:
    found = test_tokens_in(text)
    if found:
        raise ConfigError(f"{what} carries the test profile's {found}: a lease run never uses the test machine's setup")


def _walk(node, path=()):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _walk(v, path + (k,))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _walk(v, path + (i,))
    else:
        yield path, node


def _quality_indices(doc) -> list[tuple]:
    return [path for path, _ in _walk(doc) if len(path) >= 2 and path[-2:] == ("quality", "index")]


def patch_indices(shipped_text: str) -> tuple[str, list[tuple]]:
    """Each pool's quality.index pointed at our index: the one edit to the shipped policy, made on upstream's text
    (a YAML round trip could rewrite values the Go router reads differently). Returns the patched text and the paths
    changed; a patch that changes anything else is refused."""
    shipped = yaml.safe_load(shipped_text)
    paths = _quality_indices(shipped)
    values = {}
    for path in paths:
        node = shipped
        for p in path:
            node = node[p]
        values[path] = node
    if len(paths) != len(C.POOLS) or any(v not in C.SHIPPED_INDICES for v in values.values()):
        raise ConfigError(f"the shipped policy's quality indices are {values}; one shipped index per pool expected")
    patched_text = shipped_text
    for index in C.SHIPPED_INDICES:
        patched_text = patched_text.replace(f"index: {index}\n", f"index: {C.INDEX_ID}\n")
    patched = yaml.safe_load(patched_text)
    before, after = dict(_walk(shipped)), dict(_walk(patched))
    other = sorted(str(p) for p in set(before) | set(after) if p not in paths and before.get(p) != after.get(p))
    if set(before) != set(after) or other or any(after[p] != C.INDEX_ID for p in paths):
        raise ConfigError(f"the patch changed more or less than the pools' quality indices: {other[:5]}")
    return patched_text, paths


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _cli(vsr: Path, args: Iterable[str], cwd: Path) -> str:
    env = {"HOME": os.environ["HOME"], "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
           "LANG": "C.UTF-8"}         # a clean environment, as every process of a run starts
    proc = subprocess.run([str(vsr), *args], cwd=cwd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        raise ConfigError(f"vllm-sr {' '.join(args)} failed ({proc.returncode}): {(proc.stderr or proc.stdout)[-1500:]}")
    return proc.stdout


def generate(out_dir: Path, vsr: Path, provider: dict, binds: dict, *, test: bool) -> dict:
    """Write provider.yaml and bindings.yaml, run upstream's generator, apply the index edit, validate with upstream's
    validator. Returns the files' digests; the served config is out_dir/config.yaml."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "provider.yaml").write_text(yaml.safe_dump(provider, sort_keys=False))
    (out_dir / "bindings.yaml").write_text(yaml.safe_dump(binds, sort_keys=False))
    _cli(vsr, ["recipe", "builtin", "init", C.RECIPE, "--bundle", C.BUNDLE, "--config", "provider.yaml",
               "--bindings", "bindings.yaml", "--model-name", C.ENTRYPOINT, "--output", "shipped.yaml"], out_dir)
    text, changed = patch_indices((out_dir / "shipped.yaml").read_text())
    (out_dir / "config.yaml").write_text(text)
    if not test:
        refuse_test_tokens(text, str(out_dir / "config.yaml"))
    elif C.TEST_MARK not in text:
        raise ConfigError("a test-profile config must carry the test mark")
    _cli(vsr, ["config", "validate", "--config", "config.yaml"], out_dir)
    record = {"profile": "test" if test else "lease", "vllm_sr": C.VSR_VERSION,
              "patched_paths": [".".join(map(str, p)) for p in changed],
              "sha256": {n: sha256_file(out_dir / n) for n in ("provider.yaml", "bindings.yaml", "shipped.yaml",
                                                                  "config.yaml")}}
    (out_dir / "config.record.json").write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    return record
