"""The engine record against fake vLLM engines over real HTTP: what run_cell.sh writes before and after a cell."""
import asyncio

import pytest

from routebalance.config import ConfigError
from tests.contract.test_direct_engines import LARGE, SMALL, Engines
from tools.engines import engines


def _record(tmp_path, fleet, skip=(), port_shift=0, start=1790000000.0):
    """As deploy.sh writes it: the plan's lines, then what each engine executes (node_ctl.sh started)."""
    path = tmp_path / "engines.record"
    kept = [i for i in fleet if i.instance_id not in skip]
    path.write_text("# deploy.sh engines (test)\n" + "".join(
        f"{i.instance_id}\t{i.node}\t{i.address}\t{i.port + port_shift}\t0\t{i.tier.gpu}\t1\t{i.model}\t"
        f"--model {i.model} --port {i.port}\n" for i in kept) + "".join(
        f"started\t{i.instance_id}\t{start}\t0\t/venv/bin/python -u -m vllm.entrypoints.openai.api_server "
        f"--model {i.model} --port {i.port}\n" for i in kept))
    return path


def test_each_engine_is_read_as_it_reports_itself(tmp_path):
    async def scenario():
        async with Engines([SMALL, LARGE]) as e:
            for st, m in zip(e.states, "SL", strict=True):
                st.served = (m,)
            record = _record(tmp_path, e.fleet)
            return e, await engines(e.fleet, record, 5.0), await engines(e.fleet, record, 5.0)
    e, first, second = asyncio.run(scenario())
    s0 = first["instances"]["S_0"]
    assert (s0["model"], s0["max_model_len"], s0["start_time_s"], s0["gpus"]) == ("S", 4096, 1790000000.0, "0")
    assert s0["cache_config"]["num_gpu_blocks"] == "8000" and "engine" not in s0["cache_config"]
    assert s0["argv"] == f"--model S --port {e.fleet[0].port}" and len(s0["argv_sha256"]) == 64
    assert s0["cmdline"].endswith(f"vllm.entrypoints.openai.api_server {s0['argv']}")
    assert second == first


@pytest.mark.parametrize("case,match", [("unrecorded", "has no engine L_1"), ("moved", "deployed at"),
                                        ("two_models", "exactly one model"), ("no_start", "process_start_time"),
                                        ("relaunched", "not the process deploy.sh launched")])
def test_a_reading_that_does_not_match_the_deployment_fails(tmp_path, case, match):
    async def scenario():
        async with Engines([SMALL, LARGE]) as e:
            for st, m in zip(e.states, "SL", strict=True):
                st.served = (m,)
            record = _record(tmp_path, e.fleet, skip=("L_1",) if case == "unrecorded" else (),
                             port_shift=1 if case == "moved" else 0)
            if case == "two_models":
                e.states[0].served = ("S", "S2")
            if case == "no_start":
                e.states[0].raw_body = b'vllm:cache_config_info{block_size="16"} 1.0\n'
            if case == "relaunched":                              # started again after the deploy recorded it
                e.states[1].start_time_s += 30.0
            await engines(e.fleet, record, 5.0)
    with pytest.raises(ConfigError, match=match):
        asyncio.run(scenario())


def test_an_unreachable_engine_fails(tmp_path):
    async def scenario():
        async with Engines([SMALL]) as e:
            e.states[0].served = ("S",)
            record = _record(tmp_path, e.fleet)
            await e.runners[0].cleanup()
            await engines(e.fleet, record, 1.0)
    with pytest.raises(ConfigError, match="cannot read the engine"):
        asyncio.run(scenario())
