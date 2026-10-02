"""The fleet configs and engine args against what v3 deployed.

Every v3 file is found under the oracle root by name and pinned by md5 (the S2 model_deployment and
scheduler config are the md5s the v3 head snapshot recorded). The engine arguments are compared with what
v3's launchers passed, translated to official vLLM: the fork-era deploy passed --dtype from the precision,
--tensor-parallel-size = every visible GPU of the node, --trust-remote-code, --gpu-memory-utilization,
--max-model-len and, where set, --enforce-eager; its `port` and `tensor-parallel-size` config keys had no
reader (the engines listened on vLLM's default port 8000). The Gemma A100 instances were launched by the
Gemma deploy script with a CUDA mask per instance, so their flags come from that script. Gemma's prices are not
v3's (D208, owner 10-02): OpenRouter's of 2 Oct 2026, the 1B (which OpenRouter does not list) at $0.05/$0.05 per
1M tokens; its models, GPUs and placement are v3's.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from routebalance.config import load_fleet, loads_strict
from routebalance.hw import gpu_of, gpus_per_node
from tests.equivalence.oracle_files import LATENCY_TEST, find_one, md5_of

REPO = Path(__file__).resolve().parents[2]
HOSTS = REPO / "configs" / "hosts.example.json"
V3_QWEN_DEPLOY = ("model_deployment.json", "dfec4da4e1209966398b623f311a2c57")
V3_QWEN_SCHED = ("scheduler_config_s2.json", "002413dae0ae8c1534e90d31e7ecdd12")
V3_GEMMA_DEPLOY = ("model_deployment.json", "07907edec3a9b99e21bffa3fcd84f861")
V3_GEMMA_SCHED = ("scheduler_config_gemma.json", "5020c3cd30bfb1404e6e9374a4a68783")
V3_GEMMA_SCRIPT = ("gemma_deploy_v3.sh", "27bb91e49b2c53263453df98cd67527c")
V3_HUB_PINS = ("RERUN_PINS.md", "dc849c902568c3cb85acdd362f0a07bf")   # "HF model revisions actually loaded"
# D208 (owner 10-02): Gemma's prices per token (input, output), OpenRouter's on pricepertoken.com (2 Oct 2026 09:43
# BST) and the owner's 1B; v3's were 1B 1e-07/1e-07, 4B 4e-08/8e-08, 12B 5e-08/1.5e-07, 27B 8e-08/1.6e-07
GEMMA_D208_PRICES = {"gemma-3-1b_a30": (5e-08, 5e-08), "gemma-3-4b_a30": (5e-08, 1e-07),
                     "gemma-3-12b_a100": (5e-08, 1.5e-07), "gemma-3-27b_a100": (8e-08, 4.5e-07)}
V3_FLAGS = {"--dtype", "--tensor-parallel-size", "--trust-remote-code", "--gpu-memory-utilization",
            "--max-model-len", "--enforce-eager"}
_HW = re.compile(r"^(?:[^@]+@)?([a-z0-9]+)-")      # CloudLab host name -> hardware type (v3's rule)


def pinned(base: Path, name: str, md5: str) -> Path:
    """The file named `name` under `base` with this md5 (identical copies are the same file)."""
    found = [p for p in sorted(Path(base).rglob(name)) if md5_of(p) == md5]
    assert found, f"no {name} with md5 {md5} under {base}"
    return found[0]


def _json(path: Path):
    return loads_strict(Path(path).read_text())


def _ours(family: str):
    fleet = load_fleet(REPO / "configs" / f"fleet_{family}.json", HOSTS)
    ea = _json(REPO / "configs" / f"engine_args_{family}.json")
    return fleet, ea


def _hw_of(host: str) -> str:
    m = _HW.match(host)
    assert m, f"cannot read the hardware type of {host!r}"
    return m.group(1)


def _tier_ids(sched: dict) -> dict[tuple[str, str], str]:
    """(model, GPU class) -> tier id, as the v3 scheduler config records them."""
    meta = sched["instance_metadata"]
    out = {(m["model_name"], m["gpu_type"]): tid for tid, m in meta.items()}
    assert len(out) == len(meta)
    return out


def _dtype(entry: dict) -> str:
    params = entry.get("vllm_params", {})   # default-ok: the 12B/27B Gemma entries carry none
    if "dtype" in params:
        return params["dtype"]
    return "float16" if entry["precision"] == "fp16" else {"bf16": "bfloat16"}[entry["precision"]]


def _v3_instances(dep: dict, sched: dict) -> list[dict]:
    """v3's instances in registration order ("<key>_<idx>"): node_hosts first, then the embedded instances."""
    tier_of = _tier_ids(sched)
    out = []
    for key, entry in dep.items():
        hosts = [{"host": h, "port": 8000, "mask": None} for h in entry.get("node_hosts", [])]  # default-ok
        hosts += [{"host": i["hostname"], "port": i["backend_port"], "mask": i["cuda_visible_devices"]}
                  for i in entry.get("instances", [])]                                    # default-ok
        for idx, h in enumerate(hosts):
            hw = _hw_of(h["host"])
            n_gpus = gpus_per_node(hw) if h["mask"] is None else len(h["mask"].split(","))
            gpus = list(range(gpus_per_node(hw))) if h["mask"] is None else [int(g) for g in h["mask"].split(",")]
            out.append({"id": f"{key}_{idx}", "tier": tier_of[(entry["hf_model_name"], gpu_of(hw))],
                        "model": entry["hf_model_name"], "host": h["host"].split("@")[-1], "port": h["port"],
                        "gpus": gpus, "tp": n_gpus, "entry": entry})
    return out


def _hub_pins(oracle) -> dict[str, str]:
    text = pinned(oracle.root / "rb-exp" / "v3-exp", *V3_HUB_PINS).read_text()
    return dict(re.findall(r"^- (\S+) @ ([0-9a-f]{40})$", text, re.M))


def _check_instances(fleet, ea, v3):
    assert [(i.instance_id, i.tier.tier_id, i.model) for i in fleet] == [(v["id"], v["tier"], v["model"]) for v in v3]
    ours_by_node, v3_by_host = {}, {}
    for i in fleet:
        ours_by_node.setdefault(i.node, Counter())[(i.tier.tier_id, i.port, tuple(ea["gpus"][i.instance_id]))] += 1
    for v in v3:
        v3_by_host.setdefault(v["host"], Counter())[(v["tier"], v["port"], tuple(v["gpus"]))] += 1
    # the same placement: every v3 host corresponds to one of our nodes with exactly its instances
    assert sorted(map(sorted, map(Counter.elements, ours_by_node.values()))) == \
        sorted(map(sorted, map(Counter.elements, v3_by_host.values())))


def _check_prices(fleet, sched: dict, prices: dict | None = None):
    """Each tier's model and GPU are v3's; its prices are v3's, or `prices` (D208 for Gemma)."""
    meta = sched["instance_metadata"]
    tiers = {i.tier.tier_id: i.tier for i in fleet}
    assert set(tiers) == set(meta)
    for tid, t in tiers.items():
        m = meta[tid]
        want = (m["cost_per_input_token"], m["cost_per_output_token"]) if prices is None else prices[tid]
        assert (t.model, t.gpu, t.cost_in, t.cost_out) == (m["model_name"], m["gpu_type"], *want), tid


def _check_revisions(fleet, ea, pins: dict):
    for tid, spec in ea["tiers"].items():
        model = next(i.model for i in fleet if i.tier.tier_id == tid)
        assert spec["revision"] == pins[model], f"{tid}: {spec['revision']} != v3's {pins[model]}"


def test_qwen_fleet_is_the_v3_s2_deployment(oracle):
    block = oracle.root / "Block" / "block"
    dep = _json(find_one(block, *V3_QWEN_DEPLOY))
    sched = _json(find_one(block, *V3_QWEN_SCHED))
    fleet, ea = _ours("qwen")
    v3 = _v3_instances(dep, sched)
    assert len(v3) == 13 and all(v["port"] == 8000 for v in v3)
    _check_instances(fleet, ea, v3)
    _check_prices(fleet, sched)
    _check_revisions(fleet, ea, _hub_pins(oracle))
    for v in v3:
        entry, args = v["entry"], ea["tiers"][v["tier"]]["args"]
        p = entry["vllm_params"]
        want = {"--dtype": _dtype(entry), "--tensor-parallel-size": v["tp"], "--trust-remote-code": True,
                "--gpu-memory-utilization": p["gpu-memory-utilization"], "--max-model-len": p["max-model-len"]}
        if p.get("enforce-eager", False):              # default-ok: v3's deploy read it with this default
            want["--enforce-eager"] = True
        assert args == want, v["tier"]
        assert not p.get("disable-prefix-caching", False)   # default-ok: v3 kept vLLM's prefix caching


def test_qwen_max_num_seqs_is_what_the_v3_engines_reported(oracle):
    """The fork's /instance_stats carried the engine's resolved max_num_seqs (v3 passed no flag)."""
    fleet, _ = _ours("qwen")
    want = {i.tier.tier_id: i.tier.max_num_seqs for i in fleet}
    seen: dict[str, set] = {t: set() for t in want}
    with open(oracle.data_file(LATENCY_TEST)) as f:
        for line in f:
            r = json.loads(line)
            if r["instance_type"] in seen:
                seen[r["instance_type"]].add(r["schedule_state"]["max_num_seqs"])
    assert {t: sorted(v) for t, v in seen.items()} == {t: [n] for t, n in want.items()}


def test_gemma_fleet_is_the_v3_gemma_ladder(oracle):
    dep = _json(find_one(oracle.root / "Block" / "block", *V3_GEMMA_DEPLOY))
    sched = _json(pinned(oracle.root / "Block", *V3_GEMMA_SCHED))
    script = pinned(oracle.root / "Block", *V3_GEMMA_SCRIPT).read_text()
    fleet, ea = _ours("gemma")
    v3 = _v3_instances(dep, sched)
    assert Counter(v["tier"] for v in v3) == {"gemma-3-1b_a30": 3, "gemma-3-4b_a30": 5, "gemma-3-12b_a100": 4,
                                               "gemma-3-27b_a100": 2}
    _check_instances(fleet, ea, v3)
    assert set(GEMMA_D208_PRICES) == set(sched["instance_metadata"])
    _check_prices(fleet, sched, GEMMA_D208_PRICES)
    _check_revisions(fleet, ea, _hub_pins(oracle))
    # the A100 instances' launch line in the Gemma deploy script (TP = the GPUs of the instance's mask)
    (line,) = [ln for ln in script.splitlines() if "vllm.entrypoints.openai.api_server" in ln]
    flag = {f: re.search(rf"{f} (\S+)", line).group(1) for f in ("--dtype", "--gpu-memory-utilization", "--max-model-len")}
    assert "--trust-remote-code" in line and "--tensor-parallel-size $GPU_COUNT" in line
    assert "--enforce-eager" not in line and "prefix-caching" not in line
    for v in v3:
        args = ea["tiers"][v["tier"]]["args"]
        if v["entry"].get("node_hosts"):               # default-ok: the A30 tiers went through the v3 deploy
            p = v["entry"]["vllm_params"]
            src = {"--dtype": _dtype(v["entry"]), "--gpu-memory-utilization": p["gpu-memory-utilization"],
                   "--max-model-len": p["max-model-len"]}
        else:
            src = {"--dtype": flag["--dtype"], "--gpu-memory-utilization": float(flag["--gpu-memory-utilization"]),
                   "--max-model-len": int(flag["--max-model-len"])}
        want = dict(src, **{"--tensor-parallel-size": v["tp"], "--trust-remote-code": True})
        assert args == want, v["tier"]


def test_engine_args_carry_no_flag_v3_did_not_pass():
    for family in ("qwen", "gemma"):
        for tid, spec in _json(REPO / "configs" / f"engine_args_{family}.json")["tiers"].items():
            assert set(spec["args"]) <= V3_FLAGS, (family, tid)


def test_setup_reproduces_the_avengers_pro_copy_v3_served(oracle):
    """setup.sh's AvengersPro pins: its patched md5 is the oracle's served copy, and undoing the embedded
    patch on that copy gives the upstream md5 it expects at the pinned commit."""
    text = (REPO / "scripts" / "setup.sh").read_text()
    want = dict(re.findall(r"^(AP_UPSTREAM_MD5|AP_PATCHED_MD5)=([0-9a-f]{32})", text, re.M))
    patch = text.split("<<'PATCH'\n", 1)[1].split("\nPATCH\n", 1)[0].splitlines()
    hunk = patch[next(k for k, ln in enumerate(patch) if ln.startswith("@@")) + 1:]
    new = "".join(ln[1:] + "\n" for ln in hunk if ln[:1] in (" ", "+"))
    old = "".join(ln[1:] + "\n" for ln in hunk if ln[:1] in (" ", "-"))
    served = (oracle.root / "AvengersPro" / "simple_cluster_router.py").read_bytes()
    assert hashlib.md5(served).hexdigest() == want["AP_PATCHED_MD5"]
    assert served.decode().count(new) == 1
    upstream = served.decode().replace(new, old).encode()
    assert hashlib.md5(upstream).hexdigest() == want["AP_UPSTREAM_MD5"]
