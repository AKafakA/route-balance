"""The engine record (D48): the deploy record's launches, and each engine's start time and KV cache as it reports them."""
import pytest

from routebalance.config import ConfigError
from tools.engines import cache_config, deploy_record, start_time

ARGV = "--model Qwen/Qwen2.5-7B --port 8000"
LINE = f"Qwen-2.5-7B_0\ta30-0\td7525.example\t8000\t0\ta30\t1\tQwen/Qwen2.5-7B\t{ARGV}"
STARTED = ("started\tQwen-2.5-7B_0\t1.79035433425e+09\t0\t"
           f"/mydata/rb/venv-engine/bin/python -u -m vllm.entrypoints.openai.api_server {ARGV}")
# the line vLLM 0.18.1 served on the V100 node on 2026-09-25 (14B, TP4)
REAL_0181 = ('vllm:cache_config_info{_block_size_resolved="True",block_size="16",cache_dtype="auto",'
             'calculate_kv_scales="False",cpu_kvcache_space_bytes="None",enable_prefix_caching="True",engine="0",'
             'gpu_memory_utilization="0.85",is_attention_free="False",kv_cache_memory_bytes="None",'
             'kv_offloading_backend="native",kv_offloading_size="None",kv_sharing_fast_prefill="False",'
             'mamba_block_size="None",mamba_cache_dtype="auto",mamba_cache_mode="none",mamba_page_size_padded="None",'
             'mamba_ssm_cache_dtype="auto",num_cpu_blocks="None",num_gpu_blocks="7515",num_gpu_blocks_override="None",'
             'prefix_caching_hash_algo="sha256",sliding_window="None",user_specified_block_size="False"} 1.0')
METRICS = "\n".join([
    "# HELP process_start_time_seconds Start time of the process since unix epoch in seconds.",
    "# TYPE process_start_time_seconds gauge",
    "process_start_time_seconds 1.79032e+09",
    "# TYPE vllm:cache_config_info gauge",
    'vllm:cache_config_info{block_size="16",cache_dtype="auto",engine="0",gpu_memory_utilization="0.9",'
    'num_gpu_blocks="8123",prefix_caching_hash_algo="sha256",note="a \\"quoted\\" value"} 1.0',
    'vllm:num_requests_running{engine="0",model_name="M"} 0.0',
])


def test_the_deploy_record_is_read_as_deploy_sh_writes_it(tmp_path):
    path = tmp_path / "engines.record"
    path.write_text("# deploy.sh engines 2026-09-25T07:15:14Z\n# fleet sha256 0\n" + LINE + "\n\n" + STARTED + "\n")
    assert deploy_record(path) == {"Qwen-2.5-7B_0": {
        "address": "d7525.example", "port": 8000, "gpus": "0", "argv": ARGV, "start_time_s": 1790354334.25,
        "cmdline": STARTED.split("\t")[4]}}


@pytest.mark.parametrize("body,match", [
    ("# only comments\n", "lists no engine"),
    (LINE + "\n" + LINE + "\n", "recorded twice"),
    (LINE.replace("\t8000\t", "\tport\t") + "\n", "9 tab-separated fields"),
    ("\t".join(LINE.split("\t")[:8]) + "\n", "9 tab-separated fields"),
    (LINE + "\n", "needs its started line"),                                     # a deploy that did not finish
    (LINE + "\n" + STARTED + "\n" + STARTED + "\n", "once per engine"),
    (LINE + "\n" + STARTED.replace("1.79035433425e+09", "soon") + "\n", "a started line is"),
    (LINE + "\n" + STARTED.replace("--port 8000", "--port 8000 --enforce-eager") + "\n", "not the planned arguments"),
    (LINE + "\n" + STARTED.replace("\t0\t/", "\t1\t/") + "\n", "runs on GPUs 1, planned 0"),
])
def test_a_malformed_deploy_record_is_refused(tmp_path, body, match):
    path = tmp_path / "engines.record"
    path.write_text(body)
    with pytest.raises(ConfigError, match=match):
        deploy_record(path)


def test_the_real_vllm_0181_line_is_read():
    labels = cache_config("e", "# TYPE vllm:cache_config_info gauge\n" + REAL_0181 + "\n")
    assert len(labels) == 23 and labels["num_gpu_blocks"] == "7515" and labels["sliding_window"] == "None"
    assert "engine" not in labels


def test_start_time_and_cache_config_come_from_the_engines_metrics():
    assert start_time("e", METRICS) == 1.79032e9
    assert cache_config("e", METRICS) == {"block_size": "16", "cache_dtype": "auto", "gpu_memory_utilization": "0.9",
                                          "num_gpu_blocks": "8123", "prefix_caching_hash_algo": "sha256",
                                          "note": 'a \\"quoted\\" value'}      # the engine index is left out


@pytest.mark.parametrize("text,fn,match", [
    ("vllm:num_requests_running 0.0\n", start_time, "0 process_start_time_seconds samples"),
    ("process_start_time_seconds 1.0\nprocess_start_time_seconds 2.0\n", start_time, "2 process_start_time_seconds"),
    ("process_start_time_seconds nan\n", start_time, "is not a time"),
    ("process_start_time_seconds_x 1.0\n", start_time, "0 process_start_time_seconds samples"),
    ("vllm:num_requests_running 0.0\n", cache_config, "0 vllm:cache_config_info samples"),
    ('vllm:cache_config_info{engine="0"} 1.0\n', cache_config, "carries no configuration"),
    ('vllm:cache_config_info{a="1",a="2"} 1.0\n', cache_config, "repeats label a"),
    ('vllm:cache_config_info{a=1} 1.0\n', cache_config, "malformed"),
])
def test_a_missing_or_malformed_sample_fails_loudly(text, fn, match):
    with pytest.raises(ConfigError, match=match):
        fn("e", text)
