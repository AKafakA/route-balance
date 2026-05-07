import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any

import aiohttp
from tqdm import tqdm
from vllm.benchmarks.lib.endpoint_request_func import RequestFunc, RequestFuncInput



def _update_headers_common(
    headers: dict[str, Any],
    request_func_input: RequestFuncInput,
) -> None:
    """Update headers with common fields. Copied from vLLM's endpoint_request_func.py."""
    if request_func_input.extra_headers:
        headers |= request_func_input.extra_headers
    if request_func_input.request_id:
        headers["x-request-id"] = request_func_input.request_id


@dataclass
class RequestFuncOutput:
    """
       The output of the request function including metrics.
       Should be aligned with vLLM's RequestFuncOutput, but extend with model name and scheduling overhead.
    """
    generated_text: str = ""
    success: bool = False
    latency: float = 0.0  # Client-side E2E latency (user-perceived)
    output_tokens: int = 0
    ttft: float = 0.0  # Time to first token (server-side measurement)
    itl: list[float] = field(default_factory=list)  # Inter-token latencies (server-side)
    tpot: float = 0.0  # avg next-token latencies
    prompt_len: int = 0
    error: str = ""
    start_time: float = 0.0
    model: str = ""
    request_id: str = ""
    scheduling_overhead: float = 0.0  # ROUTE_BALANCE-specific: client E2E - server E2E = network + ROUTE_BALANCE routing
    instance_id: str = ""  # Instance that handled the request
    host: str = ""  # Host IP address of the instance
    broadcast_results: list[dict] = field(default_factory=list)  # Broadcasting: responses from all queried models
    scheduling_overhead_breakdown: dict = field(default_factory=dict)  # ROUTE_BALANCE batch scheduling breakdown
    predicted_quality: float = 0.0  # Predicted quality score for selected model
    predicted_length: float = 0.0  # Predicted output length
    predicted_best_model: str = ""  # Model with highest predicted quality (KNN-based)
    predicted_best_hit: bool = False  # Whether scheduler picked highest-predicted-quality model
    all_model_scores: dict = field(default_factory=dict)  # All model quality scores
    budget_cost: float = 0.0  # Per-request budget (USD)
    actual_cost: float = 0.0  # Realized cost = input_tokens × c_in + output_tokens × c_out
    budget_exhausted: bool = False  # True if dispatch clamped or budget consumed mid-stream
    affordable_output_tokens: int = 0  # Output tokens the budget allowed on the chosen instance


async def async_request_route_balance_openai_completions(
    request_func_input: RequestFuncInput,
    session: aiohttp.ClientSession,
    pbar: tqdm | None = None,
) -> RequestFuncOutput:
    """
    The async request function for creating OpenAI Completions API to call RouteBalance backend.
    Args:
        request_func_input: The input for the request function.
        pbar: The progress bar to display the progress.
        session: The aiohttp session to use.
    Returns:
        The output of the request function.
    """
    api_url = request_func_input.api_url
    payload = {
        "request_id": request_func_input.request_id,
        # the model passing will be ignored and get resolved by RouteBalance server side
        "model": "route_balance",
        "prompt": request_func_input.prompt,
        "prompt_len": request_func_input.prompt_len,
        "temperature": 0.0,
        "repetition_penalty": 1.0,
        # Apr 26: matches route_balance/exp/route_balance/test_broadcasting.sh:72
        # FREQUENCY_PENALTY=1.2, tuned via sweep_broadcasting.sh to prevent
        # degenerate repetition. Identical sampling params on training and
        # inference paths so predictor sees the same response distribution.
        "frequency_penalty": 1.2,
        "max_tokens": request_func_input.output_len,
        # Apr 26: send num_predicted_output_tokens to match generate_latency_benchmark.py
        # payload schema. route_balance_serve forwards this to vLLM as predicted_decode_tokens
        # for accurate pending_decode_tokens accounting in /instance_stats. Without it,
        # vLLM falls back to max_tokens (overestimates queue load) — predictor trained
        # on generate_latency_benchmark data would see a different feature distribution
        # than at inference here. output_len is the dataset's oracle predicted output
        # length, same source generate_latency_benchmark uses.
        "num_predicted_output_tokens": request_func_input.output_len,
        # ROUTE_BALANCE server returns complete JSON response, not streaming
        "stream": False,
    }
    # Inject RSO from extra_body if present (C5: per-request RSO sampling)
    if request_func_input.extra_body:
        rso = request_func_input.extra_body.get("request_specific_objective")
        if rso:
            payload["request_specific_objective"] = rso
    headers = {
        "Content-Type": "application/json",
    }
    _update_headers_common(headers, request_func_input)
    output = RequestFuncOutput()
    output.prompt_len = request_func_input.prompt_len
    output.request_id = request_func_input.request_id

    st = time.perf_counter()
    output.start_time = st
    try:
        async with session.post(url=api_url, json=payload, headers=headers) as response:
            if response.status == 200:
                response_map = await response.json()
                # ROUTE_BALANCE server returns success field to indicate if request was processed
                if response_map.get("success", False):
                    output.prompt_len = request_func_input.prompt_len
                    output.success = True

                    # Measure client-side E2E latency (user-perceived latency)
                    # This includes: network time + ROUTE_BALANCE scheduling overhead + backend processing
                    output.latency = time.perf_counter() - st

                    # Get server-side metrics from backend instance
                    output.output_tokens = response_map.get("output_tokens", 0)
                    output.generated_text = response_map.get("generated_text", "")
                    output.ttft = response_map.get("ttft", 0.0)  # Backend's time to first token
                    output.itl = response_map.get("itl", [])     # Backend's inter-token latencies
                    output.model = response_map.get("model", "")
                    output.instance_id = response_map.get("instance_id", "")
                    output.host = response_map.get("host", "")

                    # Extract broadcast results if present (when broadcasting is enabled)
                    output.broadcast_results = response_map.get("broadcast_results", [])

                    # Get server-side E2E latency (reported by backend instance)
                    server_latency = response_map.get("server_latency", 0.0)

                    # Calculate scheduling overhead: client E2E - server E2E
                    output.scheduling_overhead = output.latency - server_latency

                    # Extract ROUTE_BALANCE batch scheduling overhead breakdown (if present)
                    output.scheduling_overhead_breakdown = response_map.get(
                        "scheduling_overhead_breakdown", {}
                    )
                    # Extract quality metrics (from ModelEstimator)
                    output.predicted_quality = response_map.get("predicted_quality", 0.0)
                    output.predicted_length = response_map.get("predicted_length", 0.0)
                    output.predicted_best_model = response_map.get("predicted_best_model", "")
                    output.predicted_best_hit = response_map.get("predicted_best_hit", False)
                    output.all_model_scores = response_map.get("all_model_scores", {})
                    output.budget_cost = float(response_map.get("budget_cost", 0.0) or 0.0)
                    output.actual_cost = float(response_map.get("actual_cost", 0.0) or 0.0)
                    output.budget_exhausted = bool(response_map.get("budget_exhausted", False))
                    output.affordable_output_tokens = int(response_map.get("affordable_output_tokens", 0) or 0)

                else:
                    # Request failed on ROUTE_BALANCE server side
                    output.success = False
                    output.error = response_map.get("error", "Unknown error from ROUTE_BALANCE server")
                    output.instance_id = response_map.get("instance_id", "")
                    output.host = response_map.get("host", "")
                    output.model = response_map.get("model", "")
            else:
                # HTTP error - try to parse response for debugging info
                output.success = False
                try:
                    error_data = await response.json()
                    output.error = error_data.get("error", f"HTTP {response.status}: {response.reason or 'Unknown error'}")
                    output.instance_id = error_data.get("instance_id", "")
                    output.host = error_data.get("host", "")
                    output.model = error_data.get("model", "")
                except:
                    output.error = f"HTTP {response.status}: {response.reason or 'Unknown error'}"
    except Exception:
        output.success = False
        exc_info = sys.exc_info()
        output.error = "".join(traceback.format_exception(*exc_info))

    if pbar:
        pbar.update(1)
    return output


async def async_request_vllm_sr_chat_completions(
    request_func_input: RequestFuncInput,
    session: aiohttp.ClientSession,
    pbar: tqdm | None = None,
) -> RequestFuncOutput:
    """vLLM-SR backend for the bench.

    Posts an OpenAI /v1/chat/completions request to a vLLM-SR Envoy endpoint
    (default port 8801). The router selects the upstream Qwen model based on
    its semantic-routing config; we capture the routing decision via the
    x-vsr-selected-model and x-upstream-host headers and write them into
    RequestFuncOutput so the standard route_balance aggregator can compute distribution
    and cost.
    """
    api_url = request_func_input.api_url
    # Match route_balance_serve's sampling settings exactly so vllm-sr cells run with
    # the same payload as route_balance/avg_pro/br4 baselines (apples-to-apples).
    # Per route_balance_end_point_func.py:75-91, route_balance cells use:
    #   temperature=0.0, repetition_penalty=1.0, frequency_penalty=1.2
    # frequency_penalty=1.2 was tuned via sweep_broadcasting.sh to prevent
    # degenerate repetition under greedy decoding; removing it changes the
    # generated output distribution.
    payload = {
        "model": "MoM",  # vllm-sr ignores; routing decides actual upstream
        "messages": [{"role": "user", "content": request_func_input.prompt}],
        "max_tokens": request_func_input.output_len,
        "temperature": 0.0,
        "frequency_penalty": 1.2,
        "repetition_penalty": 1.0,
        "stream": False,
    }
    if request_func_input.extra_body:
        for k, v in request_func_input.extra_body.items():
            if k != "request_specific_objective":
                payload.setdefault(k, v)
    headers = {"Content-Type": "application/json"}
    _update_headers_common(headers, request_func_input)
    output = RequestFuncOutput()
    output.prompt_len = request_func_input.prompt_len
    output.request_id = request_func_input.request_id

    st = time.perf_counter()
    output.start_time = st
    try:
        async with session.post(url=api_url, json=payload, headers=headers) as response:
            if response.status == 200:
                response_map = await response.json()
                output.success = True
                output.latency = time.perf_counter() - st
                # vLLM-SR routing headers — capture which Qwen tier was selected
                hdrs = response.headers
                # Prefer x-vsr-selected-model (router decision); fall back to
                # response.model (the actual model that produced the answer).
                output.model = (
                    hdrs.get("x-vsr-selected-model", "")
                    or response_map.get("model", "")
                )
                output.host = hdrs.get("x-upstream-host", "")
                output.instance_id = hdrs.get("x-upstream-host", "")
                # Token counts from OpenAI usage block
                usage = response_map.get("usage", {}) or {}
                output.output_tokens = int(usage.get("completion_tokens", 0) or 0)
                # OpenAI chat-completions doesn't expose ttft/itl — leave 0.
                output.ttft = 0.0
                output.itl = []
                # Generated text (first choice)
                choices = response_map.get("choices") or []
                if choices:
                    msg = choices[0].get("message") or {}
                    output.generated_text = msg.get("content", "") or ""
                else:
                    output.generated_text = ""
            else:
                output.success = False
                try:
                    err_data = await response.json()
                    output.error = err_data.get("error", f"HTTP {response.status}: {response.reason}")
                except Exception:
                    output.error = f"HTTP {response.status}: {response.reason or 'Unknown'}"
    except Exception:
        output.success = False
        exc_info = sys.exc_info()
        output.error = "".join(traceback.format_exception(*exc_info))

    if pbar:
        pbar.update(1)
    return output


# Create the ROUTE_BALANCE async request functions dictionary along with VLLM's request functions
ROUTE_BALANCE_ASYNC_REQUEST_FUNCS : dict[str, RequestFunc] = {
    "route_balance": async_request_route_balance_openai_completions,
    "vllm_sr": async_request_vllm_sr_chat_completions,
}