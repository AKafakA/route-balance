"""The tracked fleet configs, engine args and launch files, and the plan that turns them into vLLM launches."""
import json
import re
from collections import Counter
from pathlib import Path

import pytest

from routebalance.config import ConfigError, load_fleet
from routebalance.hw import gpus_per_node
from tools import engine_plan
from tools.engine_plan import REQUIRED_FLAGS, load_launch, main, plan

REPO = Path(__file__).resolve().parents[2]
CONFIGS = REPO / "configs"
HOSTS = CONFIGS / "hosts.example.json"
FAMILIES = {
    "qwen": {"qwen2.5-14b_v100": 3, "qwen2.5-3b_a30": 3, "qwen2.5-72b_a100": 2, "qwen2.5-7b_a30": 5},
    "gemma": {"gemma-3-1b_a30": 3, "gemma-3-4b_a30": 5, "gemma-3-12b_a100": 4, "gemma-3-27b_a100": 2},
}


def _paths(family):
    return CONFIGS / f"fleet_{family}.json", HOSTS, CONFIGS / f"engine_args_{family}.json"


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_fleet_loads_against_the_example_hosts(family):
    fleet = load_fleet(CONFIGS / f"fleet_{family}.json", HOSTS)
    assert Counter(i.tier.tier_id for i in fleet) == FAMILIES[family]
    assert all(i.tier.max_num_seqs == 256 for i in fleet)


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_plan_renders_every_instance(family):
    launches = plan(*_paths(family))
    ea = json.loads(_paths(family)[2].read_text())
    assert [x.instance.instance_id for x in launches] == [i.instance_id for i in load_fleet(*_paths(family)[:2])]
    for x in launches:
        i, argv = x.instance, list(x.argv)
        assert argv[:8] == ["--model", i.model, "--revision", ea["tiers"][i.tier.tier_id]["revision"],
                            "--port", str(i.port), "--max-num-seqs", str(i.tier.max_num_seqs)]
        assert all(f in argv for f in REQUIRED_FLAGS)
        assert int(argv[argv.index("--tensor-parallel-size") + 1]) == len(x.gpus)
        assert all(0 <= g < gpus_per_node(x.hw) for g in x.gpus)
    per_node = {}
    for x in launches:
        per_node.setdefault(x.instance.node, []).extend(x.gpus)
    assert all(len(g) == len(set(g)) for g in per_node.values())


def test_vllm_pin_agrees_everywhere():
    pyproject = (REPO / "pyproject.toml").read_text()
    serve = re.search(r"^serve = \[\s*\"vllm==([0-9.]+)\",\s*\]", pyproject, re.M).group(1)
    layout = re.search(r"^VLLM_VERSION=([0-9.]+)", (REPO / "scripts" / "node_layout.sh").read_text(), re.M).group(1)
    engine = {json.loads(_paths(f)[2].read_text())["vllm"] for f in FAMILIES}
    assert {serve, layout} | engine == {"0.18.1"}


# ---------------------------------------------------------------- refusals

def _write(tmp_path, family, mutate):
    fleet, hosts, ea_path = _paths(family)
    ea = json.loads(ea_path.read_text())
    mutate(ea)
    out = tmp_path / "ea.json"
    out.write_text(json.dumps(ea))
    return fleet, hosts, out


def _tier(ea, tid):
    return ea["tiers"][tid]["args"]


@pytest.mark.parametrize("mutate, message", [
    (lambda ea: _tier(ea, "qwen2.5-3b_a30").pop("--max-model-len"), "lack"),
    (lambda ea: _tier(ea, "qwen2.5-3b_a30").update({"--port": 9000}), "comes from the fleet"),
    (lambda ea: _tier(ea, "qwen2.5-3b_a30").update({"--max-num-seqs": 64}), "comes from the fleet"),
    (lambda ea: _tier(ea, "qwen2.5-3b_a30").update({"--enforce-eager": False}), "false is not a flag"),
    (lambda ea: _tier(ea, "qwen2.5-3b_a30").update({"-O": 3}), "not a long option"),
    (lambda ea: _tier(ea, "qwen2.5-3b_a30").update({"--tensor-parallel-size": 2}), "tensor-parallel-size"),
    (lambda ea: _tier(ea, "qwen2.5-3b_a30").update({"--tensor-parallel-size": 1.0}), "tensor-parallel-size"),
    (lambda ea: _tier(ea, "qwen2.5-3b_a30").update({"--dtype": "float 16"}), "cannot pass through"),
    (lambda ea: ea["gpus"].update({"Qwen-2.5-3B_0": [1]}), "distinct indices below 1"),
    (lambda ea: ea["gpus"].update({"Qwen-2.5-72B_0": [0, 1, 2, 2]}), "distinct indices"),
    (lambda ea: ea["gpus"].pop("Qwen-2.5-7B_4"), "gpus cover"),
    (lambda ea: ea["tiers"].pop("qwen2.5-7b_a30"), "tiers"),
    (lambda ea: ea["tiers"]["qwen2.5-7b_a30"].update({"revision": "main"}), "revision"),
    (lambda ea: ea.update({"extra": 1}), "(?i)extra inputs"),
])
def test_plan_refuses(tmp_path, mutate, message):
    with pytest.raises(ConfigError, match=message):
        plan(*_write(tmp_path, "qwen", mutate))


def test_plan_refuses_two_instances_on_one_gpu(tmp_path):
    def mutate(ea):
        ea["gpus"]["Gemma-3-12B_1"] = [2]           # the node's other 12B instance holds GPU 2
    with pytest.raises(ConfigError, match="already assigned"):
        plan(*_write(tmp_path, "gemma", mutate))


def test_launch_example_is_refused_until_the_owner_sets_it(tmp_path):
    with pytest.raises(ConfigError, match="OWNER-MUST-SET"):
        load_launch(CONFIGS / "scheduler_launch.example.json")
    doc = dict(json.loads((CONFIGS / "scheduler_launch.example.json").read_text()),
               torch_threads=4, max_connections=512, request_timeout_s=600)
    ok = tmp_path / "launch.json"
    ok.write_text(json.dumps(doc))
    x = load_launch(ok)
    assert (x.torch_threads, x.max_connections, x.request_timeout_s) == (4, 512, 600.0)
    for bad in ({"torch_threads": 0}, {"max_connections": "512"}, {"request_timeout_s": -1}, {"extra": 1}):
        p = tmp_path / "bad.json"
        p.write_text(json.dumps(dict(doc, **bad)))
        with pytest.raises(ConfigError):
            load_launch(p)


def test_hub_file_is_pinned(tmp_path, capsys):
    assert main(["hub", "--file", str(CONFIGS / "scheduler_hub.json")]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines and all(re.fullmatch(r"--(model|tokenizer)\t[^@\s]+/[^@\s]+@[0-9a-f]{40}", ln) for ln in lines)
    bad = tmp_path / "hub.json"
    bad.write_text(json.dumps({"models": ["sentence-transformers/all-MiniLM-L6-v2@main"], "tokenizers": []}))
    assert main(["hub", "--file", str(bad)]) == 1


def test_cli(capsys):
    fleet, hosts, ea = map(str, _paths("qwen"))
    assert main(["models", "--fleet", fleet, "--hosts", hosts, "--engine-args", ea, "--node", "v100-0"]) == 0
    assert capsys.readouterr().out.split() == ["Qwen/Qwen2.5-14B@97e1e76335b7017d8f67c08a19d103c0504298c9"]
    assert main(["engines", "--fleet", fleet, "--hosts", hosts, "--engine-args", ea, "--node", "a100-1"]) == 0
    (line,) = capsys.readouterr().out.splitlines()
    fields = line.split("\t")
    assert fields[:8] == ["Qwen-2.5-72B_1", "a100-1", "a100-1.invalid", "8000", "0,1,2,3", "a100", "4", "Qwen/Qwen2.5-72B"]
    assert fields[8].endswith("--enforce-eager")
    assert main(["models", "--fleet", fleet, "--hosts", hosts, "--engine-args", ea, "--node", "nowhere"]) == 1
    assert "serves no instance" in capsys.readouterr().err
    assert main(["nodes", "--fleet", fleet, "--hosts", hosts]) == 0
    assert len(capsys.readouterr().out.splitlines()) == 13


def test_plan_module_hardcodes_no_hardware():
    src = Path(engine_plan.__file__).read_text()
    assert not re.search(r"d7525|c4130|d8545", src)
