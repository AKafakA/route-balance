import argparse
import asyncio
try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass
import json
import os
import random
import ssl
import time
from argparse import Namespace
import aiohttp
import numpy as np
from typing import Any, Optional, List
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from route_balance.global_scheduler.route_balance.route_balance_instance.Instance import Instance
from route_balance.server_utils_lite import serve_http
from route_balance.global_scheduler.route_balance.utils import set_ulimit
import logging
import traceback

STOP_WORD_MAPS = {
    "Qwen": ["<|im_start|>", "<|im_end|>"]
}
TIMEOUT_KEEP_ALIVE = 600  # seconds. Was 5 — caused localhost RST race vs bench keepalive=60s.
app = FastAPI()
instances = []
num_requests = 0
start_time = 0
scheduling = "random"
served_requests = []
logging.basicConfig(level=logging.INFO,
                    filemode='a+',
                    filename='experiment_output/logs/route_balance_serve.log')
logger = logging.getLogger(__name__)
chat = False
model_family = "Qwen"
repetition_penalty = 1.0
frequency_penalty = 1.2
temperature = 0.0
broadcasting_enabled = False
broadcast_model_list: list[str] = []
enable_predictor_feedback = False
# Learned predictor instances (instance_id -> RouteBalanceLearnedPredictor)
learned_predictors: dict = {}
# Scheduling config for multi-objective (loaded from predictor config)
scoring_weights: dict = {
    "w_latency": 0.3, "w_cost": 0.2, "w_quality": 0.3, "w_balance": 0.2,
}
slo_defaults: dict = {
    "ttft_slo_ms": 5000, "tpot_slo_ms": 200,
    "budget_tokens": 256, "quality_min": 0.3,
    "budget_confidence_threshold": 0.5,
    # When False, sidecar requests skip TTFT/TPOT (E2E only) — used for the
    # main table run where no SLO filter is active. Set True for the appendix
    # filter ablation. Loaded from scheduler_config / predictor_config at boot.
    "filter_enabled": False,
}
# Instance metadata (instance_id -> {model_name, gpu_type, cost_per_token, instance_type})
instance_meta: dict = {}
# Model estimator: prompt -> per-model (score, length, budget compliance)
model_estimator = None  # Optional[ModelEstimator]
# Batch queue for ROUTE_BALANCE scheduling (initialized when scheduling == "route_balance")
batch_queue = None  # Optional[BatchQueue]
# SLO filter (pluggable, default = inline route_balance_tiered for backward compat)
slo_filter = None  # Optional[SLOFilter] — None means use inline legacy filter
# Pluggable L1 Router + L2 Dispatcher (A0.1). When both are set and the
# scheduling mode is "pipeline", the request path is:
#     router.choose_model(req) → filter to model instances → dispatcher.choose_instance(...)
# Legacy --scheduling strategies remain unchanged when these are None.
router = None        # Optional[RouterBase]
dispatcher = None    # Optional[DispatchBase]

# --- Scheduling counters for paper analysis ---
scheduling_counters = {
    "total_scheduled": 0,
    "filtered_budget": 0,
    "filtered_ttft": 0,
    "filtered_tpot": 0,
    "filtered_ttft_global": 0,
    "filtered_tpot_global": 0,
    "filtered_quality": 0,
    "relaxed_budget": 0,
    "relaxed_ttft": 0,
    "relaxed_tpot": 0,
    "relaxed_quality": 0,
    "full_relaxation": 0,
    "strict_fallback": 0,
}


class BatchRequest:
    """A request waiting in the batch queue for ROUTE_BALANCE scheduling."""
    __slots__ = (
        "future", "request_json", "prompt_text", "num_prompt_tokens",
        "max_output_tokens", "budget_tokens", "budget_cost", "ttft_slo_ms",
        "tpot_slo_ms", "quality_min", "request_id", "predicted_tokens",
        "model_estimates", "arrival_time", "constraint_mode", "relax_order",
        "_batched_tpot", "_xgb_batch_ms",
    )

    def __init__(
        self, request_json: dict, prompt_text: str, num_prompt_tokens: int,
        max_output_tokens: int, budget_tokens: int, ttft_slo_ms: float,
        tpot_slo_ms: float, quality_min: float, request_id: str,
    ):
        self.future: asyncio.Future = asyncio.get_event_loop().create_future()
        self.request_json = request_json
        self.prompt_text = prompt_text
        self.num_prompt_tokens = num_prompt_tokens
        self.max_output_tokens = max_output_tokens
        self.budget_tokens = budget_tokens
        # Cost-based budget (monetary): converted to per-model token budget at filter time
        self.budget_cost: float = float(request_json.get("budget_cost",
            (request_json.get("rso") or request_json.get("request_specific_objective") or {}).get("budget_cost", 0.0)))
        self.ttft_slo_ms = ttft_slo_ms
        self.tpot_slo_ms = tpot_slo_ms
        self.quality_min = quality_min
        self.request_id = request_id
        self.predicted_tokens: float = float(max_output_tokens)
        self.model_estimates: dict = {}
        self.arrival_time: float = time.monotonic()
        self._batched_tpot: dict = {}
        self._xgb_batch_ms: float = 0.0
        # RSO constraint handling
        rso = request_json.get("rso") or request_json.get("request_specific_objective") or {}
        self.constraint_mode: str = rso.get(
            "constraint_mode",
            slo_defaults.get("constraint_mode", "STRICT"),
        )
        self.relax_order: list = rso.get(
            "relax_order",
            slo_defaults.get("relax_order", ["ttft", "tpot", "quality", "budget"]),
        )


class BatchQueue:
    """Collects incoming requests into batches for ROUTE_BALANCE scheduling.

    Batch collection uses two triggers:
    - Size: batch is dispatched when max_batch_size requests accumulated
    - Timeout: batch is dispatched when batch_timeout_ms elapsed since first request

    Adaptive sizing adjusts parameters based on cluster load.
    """

    def __init__(
        self,
        max_batch_size: int = 16,
        batch_timeout_ms: float = 100.0,
        adaptive_sizing: bool = True,
    ):
        self._queue: asyncio.Queue = asyncio.Queue()
        self.max_batch_size = max_batch_size
        self.batch_timeout_ms = batch_timeout_ms
        self.adaptive_sizing = adaptive_sizing
        # Metrics
        self._total_batches = 0
        self._total_requests = 0

    async def enqueue(self, req: BatchRequest) -> dict:
        """Enqueue a request and wait for it to be scheduled and dispatched.

        Returns the response dict from the instance.
        """
        await self._queue.put(req)
        return await req.future

    async def collect_batch(self) -> list:
        """Collect a batch of requests.

        Blocks until at least one request arrives, then collects more
        until max_batch_size or timeout, whichever comes first.
        """
        batch = []
        deadline = None

        # RouteBalance on first request (no busy-waiting)
        first = await self._queue.get()
        batch.append(first)
        deadline = time.monotonic() + self.batch_timeout_ms / 1000.0

        # Collect more until timeout or full
        while len(batch) < self.max_batch_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                req = await asyncio.wait_for(
                    self._queue.get(), timeout=remaining
                )
                batch.append(req)
            except asyncio.TimeoutError:
                break

        self._total_batches += 1
        self._total_requests += len(batch)
        return batch

    def adapt_params(self, stats_map: dict, all_instances: list):
        """Adjust batch parameters based on cluster load — paper §3 design:
        "both limits grow with the busy-instance ratio so small batches are
        used under light load and larger batches when requests would
        otherwise queue at workers."

        Busy signal: **busy-instance ratio** = fraction of instances with
        non-zero `num_waiting`. The paper's intent — "larger batches when
        requests would otherwise queue at workers" — is precisely the
        backlog signal: an instance has work that couldn't immediately
        start running. `num_running > 0` alone reduces to "any actively
        serving cluster" → ratio≈1.0 always → adaptive policy degenerates.
        `num_waiting > 0` is non-zero only when the worker can't keep up
        with arrivals → genuine saturation pressure.
        Residual within-tier fairness handled by LRU tiebreak in
        _score_and_pick_instance.
        """
        if not self.adaptive_sizing or not all_instances:
            return

        busy_count = 0
        for inst in all_instances:
            s = stats_map.get(inst._instance_id, {}) or {}
            waiting = int(s.get("num_waiting", 0) or 0)
            if waiting > 0:
                busy_count += 1
        ratio = busy_count / len(all_instances)

        if ratio < 0.3:
            self.max_batch_size = 4
            self.batch_timeout_ms = 50.0
        elif ratio < 0.7:
            self.max_batch_size = 16
            self.batch_timeout_ms = 100.0
        else:
            self.max_batch_size = 32
            self.batch_timeout_ms = 200.0

    @property
    def stats(self) -> dict:
        return {
            "total_batches": self._total_batches,
            "total_requests": self._total_requests,
            "avg_batch_size": (
                self._total_requests / self._total_batches
                if self._total_batches > 0 else 0
            ),
            "max_batch_size": self.max_batch_size,
            "batch_timeout_ms": self.batch_timeout_ms,
            "queue_size": self._queue.qsize(),
        }


def to_ollama_tag(hf_name: str) -> str:
    """Convert HuggingFace model name to Ollama tag format.
    Example: 'Qwen/Qwen2.5-3B' -> 'qwen2.5:3b'
    """
    name = hf_name.lower()
    if "/" in name:
        name = name.split("/")[-1]
    name = name.replace("-", ":")
    return name



# align with vllm bench so we can directly leverage existing tools
@app.post("/v1/completions")
async def completion(request: Request) -> Response:
    global num_requests
    request_json = await request.json()
    num_requests += 1
    request_id = request_json.get("request_id", str(num_requests))
    request_json["request_id"] = request_id
    served_requests.append(request_id)
    selected_instance = None
    # Apply server-side sampling defaults only if client didn't specify them.
    # During broadcasting/data collection, these ensure consistent generation.
    # During benchmarking, clients can override by including params in request.
    if "repetition_penalty" not in request_json:
        request_json["repetition_penalty"] = float(repetition_penalty)
    if "frequency_penalty" not in request_json:
        request_json["frequency_penalty"] = float(frequency_penalty)
    if "temperature" not in request_json:
        request_json["temperature"] = float(temperature)

    if chat:
        # Use /v1/chat/completions endpoint with messages format
        # This is cleaner and lets vLLM handle chat template automatically
        # Signal to vllm_instance to use chat endpoint
        request_json["use_chat_endpoint"] = True
    try:
        # ROUTE_BALANCE: Query predictors before scheduling (for training data collection)
        # Only query predictors if feedback is enabled
        num_prompt_tokens = request_json.get("prompt_len", 0)
        max_output_tokens = request_json.get("max_tokens", 256)

        if enable_predictor_feedback:
            # Build predicted_num_context_tokens dict for all models
            predicted_num_context_tokens = {}
            for instance in instances:
                predicted_num_context_tokens[instance._model_name] = max_output_tokens

            # Query all instance predictors (for training data collection)
            prediction_tasks = []
            for instance in instances:
                prediction_task = instance.query_predictor(
                    request_id=request_id,
                    num_context_tokens=num_prompt_tokens,
                    predicted_num_context_tokens=predicted_num_context_tokens
                )
                prediction_tasks.append(prediction_task)

            # Wait for all predictions (run in parallel)
            try:
                predictions = await asyncio.gather(*prediction_tasks, return_exceptions=True)
                logger.debug(f"Request {request_id}: Collected {len(predictions)} predictions")
            except Exception as e:
                logger.warning(f"Request {request_id}: Predictor query failed: {e}")
                predictions = []
        else:
            logger.debug(f"Request {request_id}: Skipping predictor queries (feedback disabled)")

        # Extract optional RSO (Request Service Objectives) from request
        rso = request_json.get("rso") or request_json.get("request_specific_objective") or {}
        budget_tokens = rso.get("budget_tokens", rso.get("budget", slo_defaults.get("budget_tokens", 256)))
        ttft_slo_ms = rso.get("ttft_slo_ms", slo_defaults.get("ttft_slo_ms", 5000))
        tpot_slo_ms = rso.get("tpot_slo_ms", slo_defaults.get("tpot_slo_ms", 200))
        quality_min = rso.get("quality_min", slo_defaults.get("quality_min", 0.0))
        budget_cost = float(rso.get("budget_cost", request_json.get("budget_cost", 0.0)) or 0.0)
        prompt_text = request_json.get("prompt", "")

        # Broadcasting mode: query one instance per selected model, pick one as main response
        # Non-broadcasting mode: use scheduling strategy selection
        if broadcasting_enabled and broadcast_model_list:
            # Normalize model names for matching (support HF name or Ollama tag)
            def _norm(name: str) -> str:
                return name.strip().lower()

            # Build a set of normalized targets including possible tag forms
            target_norms = set()
            for m in broadcast_model_list:
                try:
                    tag = to_ollama_tag(m)
                except Exception:
                    tag = m
                target_norms.add(_norm(m))
                target_norms.add(_norm(tag))

            # Pick at most one instance per requested model, rotating to spread load
            chosen: dict[str, Instance] = {}
            shuffled = list(instances)
            random.shuffle(shuffled)
            for inst in shuffled:
                model_key = _norm(inst._model_name)
                if model_key in target_norms and inst._model_name not in chosen:
                    chosen[inst._model_name] = inst

            # Launch queries to all chosen instances in parallel
            tasks = []
            for model_name, inst in chosen.items():
                # Avoid mutating original payload across concurrent requests
                payload_copy = json.loads(json.dumps(request_json))
                tasks.append(inst.query_instance(
                    payload_copy,
                    predicted_num_decode_tokens=max_output_tokens
                ))

            if tasks:
                try:
                    broadcast_results = await asyncio.gather(*tasks, return_exceptions=True)
                    # Filter out exceptions
                    broadcast_results = [res for res in broadcast_results if not isinstance(res, Exception)]

                    if broadcast_results:
                        # Randomly pick one as the main response (make a copy to avoid circular reference)
                        selected_response = random.choice(broadcast_results)
                        response_dict = dict(selected_response)
                        # Include all results in broadcast_results
                        response_dict["broadcast_results"] = broadcast_results
                    else:
                        # All broadcast queries failed
                        response_dict = {
                            "success": False,
                            "error": "All broadcast queries failed",
                            "request_id": request_id,
                        }
                except Exception as e:
                    response_dict = {
                        "success": False,
                        "error": f"Broadcasting failed: {str(e)}",
                        "request_id": request_id,
                    }
            else:
                # No instances matched the broadcast model list
                response_dict = {
                    "success": False,
                    "error": f"No instances found for broadcast models: {broadcast_model_list}",
                    "request_id": request_id,
                }
        else:
            # Non-broadcasting mode: use scheduling strategy
            if scheduling == "pipeline" and router is not None and dispatcher is not None:
                # A0.1 pluggable pipeline: router → filter to model instances → dispatcher.
                selected_instance, predicted_output, sched_breakdown = await _select_via_pipeline(
                    instances,
                    prompt_text=prompt_text,
                    num_prompt_tokens=num_prompt_tokens,
                    max_output_tokens=max_output_tokens,
                    budget_tokens=budget_tokens,
                    ttft_slo_ms=ttft_slo_ms,
                    tpot_slo_ms=tpot_slo_ms,
                    quality_min=quality_min,
                    request_id=request_id,
                )
                request_json["num_predicted_output_tokens"] = predicted_output
                # Budget clamp on pipeline path (parity with route_balance-mode _dispatch_and_resolve).
                # Both modes must enforce budget identically so baselines (e.g. br4 RR) under
                # T3 budget cells are apples-to-apples with route_balance_native + budget.
                pl_budget_exhausted = False
                pl_affordable_out = None
                pl_input_cost = 0.0
                pl_c_in = 0.0
                pl_c_out = 0.0
                if budget_cost > 0:
                    chosen_meta = instance_meta.get(selected_instance._instance_id, {})
                    pl_cpt = chosen_meta.get("cost_per_token", 0.0)
                    pl_c_in = chosen_meta.get("cost_per_input_token", pl_cpt)
                    pl_c_out = chosen_meta.get("cost_per_output_token", pl_cpt)
                    if pl_c_out > 0:
                        pl_input_cost = num_prompt_tokens * pl_c_in
                        remaining = budget_cost - pl_input_cost
                        if remaining <= 0:
                            pl_affordable_out = 1
                            pl_budget_exhausted = True
                        else:
                            pl_affordable_out = max(1, int(remaining / pl_c_out))
                        requested_max = request_json.get("max_tokens", max_output_tokens)
                        request_json["max_tokens"] = min(requested_max, pl_affordable_out)
                response_dict = await selected_instance.query_instance(
                    request_json,
                    predicted_num_decode_tokens=predicted_output,
                )
                response_dict["scheduling_overhead_breakdown"] = sched_breakdown
                if budget_cost > 0:
                    actual_out = int(response_dict.get("output_tokens", 0) or 0)
                    actual_cost = pl_input_cost + actual_out * pl_c_out
                    response_dict["budget_cost"] = budget_cost
                    response_dict["actual_cost"] = actual_cost
                    response_dict["affordable_output_tokens"] = pl_affordable_out or 0
                    response_dict["budget_exhausted"] = bool(
                        pl_budget_exhausted or actual_cost >= budget_cost - 1e-9
                    )
            elif scheduling == "route_balance" and batch_queue is not None:
                # Batch path: enqueue and wait for batch scheduler
                breq = BatchRequest(
                    request_json=request_json,
                    prompt_text=prompt_text,
                    num_prompt_tokens=num_prompt_tokens,
                    max_output_tokens=max_output_tokens,
                    budget_tokens=budget_tokens,
                    ttft_slo_ms=ttft_slo_ms,
                    tpot_slo_ms=tpot_slo_ms,
                    quality_min=quality_min,
                    request_id=request_id,
                )
                response_dict = await batch_queue.enqueue(breq)
            else:
                # Guard: pipeline mode requires router+dispatcher set via /v1/config
                # before any traffic. If not yet initialized, fail loudly instead of
                # silently falling through to random (cf. data_integrity_rules #6).
                if scheduling == "pipeline":
                    raise RuntimeError(
                        "scheduling=pipeline but router/dispatcher are not initialized. "
                        "POST /v1/config with {router, dispatch, filter} before traffic, "
                        "or restart with --scheduling route_balance if you intend RouteBalance-native mode."
                    )
                # Per-request path (all baseline strategies + route_balance without batch queue)
                selected_instance, predicted_output = await _select_instance(
                    scheduling, instances, num_requests, request_json,
                    prompt_text, num_prompt_tokens, max_output_tokens,
                    budget_tokens, ttft_slo_ms, tpot_slo_ms, quality_min,
                    request_id,
                )

                # Pass model estimator's predicted output length to vLLM
                # (used for pending_decode_tokens and schedule_trace)
                request_json["num_predicted_output_tokens"] = predicted_output
                response_dict = await selected_instance.query_instance(
                    request_json,
                    predicted_num_decode_tokens=predicted_output
                )

        return JSONResponse(content=response_dict)
    except Exception as e:
        logger.error(f"Error processing request {request_id}: {e}")
        logger.error(f"Instance: {selected_instance._instance_id if selected_instance else 'None'}")
        logger.error(f"Host: {selected_instance._hostname if selected_instance else 'Unknown'}")
        logger.error(f"Model: {selected_instance._model_name if selected_instance else 'Unknown'}")
        logger.error(traceback.format_exc())

        # Include debugging info in the error response
        error_response = {
            "success": False,
            "error": str(e),
            "error_traceback": traceback.format_exc(),
            "request_id": request_id,
            "instance_id": selected_instance._instance_id if selected_instance else "Unknown",
            "host": selected_instance._hostname if selected_instance else "Unknown",
            "model": selected_instance._model_name if selected_instance else "Unknown",
        }
        return JSONResponse(content=error_response, status_code=500)


@app.get("/health")
async def health() -> Response:
    """Health check endpoint."""
    return JSONResponse(content={"status": "ok"})


@app.post("/v1/estimate")
async def estimate_endpoint(request: Request) -> Response:
    """Predict per-model quality scores and output lengths for a prompt.

    Request body:
        {"prompt": str, "budget_tokens": int (optional, default 256)}

    Response:
        {model_name: {length_expected, p_under_budget, score, score_type, bucket_probs}}
    """
    if model_estimator is None:
        return JSONResponse(
            content={"error": "ModelEstimator not loaded. Start with --predictor-config."},
            status_code=503,
        )

    data = await request.json()
    prompt = data.get("prompt", "")
    budget_tokens = data.get("budget_tokens", 256)

    if not prompt:
        return JSONResponse(
            content={"error": "prompt is required"}, status_code=400
        )

    estimates = model_estimator.estimate(prompt, budget_tokens)
    result = {}
    for mname, est in estimates.items():
        result[mname] = {
            "length_expected": est.length_expected,
            "p_under_budget": est.p_under_budget,
            "score": est.score,
            "score_type": est.score_type,
            "bucket_probs": (
                est.length_bucket_probs.tolist()
                if est.length_bucket_probs is not None
                else None
            ),
        }

    return JSONResponse(content=result)


@app.post("/v1/config")
async def update_config_endpoint(request: Request) -> Response:
    """Update scheduler config at runtime (for sweeps without restart).

    Request body (all fields optional):
        {"scoring_weights": {"w_latency": 0.3, ...},
         "slo_defaults": {"ttft_slo_ms": 10000, ...},
         "router":  {"type": "route_balance_native", ...},   # A0.1 pipeline
         "dispatch": {"type": "round_robin", ...},  # A0.1 pipeline
         "filter":   {"type": "route_balance_tiered", ...}}  # supersedes slo_defaults.filter
    """
    global scoring_weights, slo_defaults, router, dispatcher, slo_filter
    data = await request.json()
    if "scoring_weights" in data:
        scoring_weights.update(data["scoring_weights"])
    if "slo_defaults" in data:
        slo_defaults.update(data["slo_defaults"])

    router_type_loaded = None
    dispatch_type_loaded = None
    filter_type_loaded = None

    if "router" in data:
        try:
            from route_balance.global_scheduler.route_balance.routers.factory import create_router
            router = create_router(
                data["router"],
                model_estimator=model_estimator,
                scoring_weights=scoring_weights,
            )
            router_type_loaded = data["router"].get("type")
        except Exception as e:
            return JSONResponse(
                {"status": "error", "where": "router", "error": str(e)},
                status_code=400,
            )

    if "dispatch" in data:
        try:
            from route_balance.global_scheduler.route_balance.dispatch.factory import (
                create_dispatcher,
            )
            dispatcher = create_dispatcher(
                data["dispatch"],
                scoring_weights=scoring_weights,
            )
            dispatch_type_loaded = data["dispatch"].get("type")
        except Exception as e:
            return JSONResponse(
                {"status": "error", "where": "dispatch", "error": str(e)},
                status_code=400,
            )

    if "filter" in data:
        try:
            from route_balance.global_scheduler.route_balance.filters.factory import create_filter
            slo_filter = create_filter(data["filter"])
            filter_type_loaded = data["filter"].get("type")
        except Exception as e:
            return JSONResponse(
                {"status": "error", "where": "filter", "error": str(e)},
                status_code=400,
            )

    # Hot-swap batch_config (max_batch_size, batch_timeout_ms, adaptive_sizing)
    # without restarting route_balance_serve. Required for T4 batching ablation: lets us
    # disable LPT sort, disable adaptive batching, or sweep max_batch_size at
    # runtime via POST /v1/config.
    if "batch_config" in data and batch_queue is not None:
        bc = data["batch_config"]
        if "max_batch_size" in bc:
            batch_queue.max_batch_size = int(bc["max_batch_size"])
        if "batch_timeout_ms" in bc:
            batch_queue.batch_timeout_ms = float(bc["batch_timeout_ms"])
        if "adaptive_sizing" in bc:
            batch_queue.adaptive_sizing = bool(bc["adaptive_sizing"])

    return JSONResponse({
        "status": "updated",
        "scoring_weights": scoring_weights,
        "slo_defaults": {k: v for k, v in slo_defaults.items()
                         if not isinstance(v, dict)},
        "pipeline": {
            "router": router_type_loaded or (type(router).__name__ if router else None),
            "dispatch": dispatch_type_loaded or (type(dispatcher).__name__ if dispatcher else None),
            "filter": filter_type_loaded or (type(slo_filter).__name__ if slo_filter else None),
            "mode": scheduling,
        },
    })


@app.get("/v1/batch_stats")
async def batch_stats_endpoint() -> Response:
    """Return batch scheduler statistics."""
    if batch_queue is None:
        return JSONResponse(
            content={"error": "Batch scheduling not active"},
            status_code=404,
        )
    return JSONResponse(content=batch_queue.stats)


@app.get("/v1/scheduling_stats")
async def scheduling_stats_endpoint() -> Response:
    """Return scheduling filter/relaxation counters for paper analysis."""
    return JSONResponse(content=scheduling_counters)


_HTTP_SESSION: Optional[aiohttp.ClientSession] = None


async def _get_http_session() -> aiohttp.ClientSession:
    """Module-level pooled aiohttp session. Per-batch fan-out of 18×B
    sidecar calls exceeded the cost of creating fresh ClientSessions per
    request; a single pooled session with a TCPConnector reuses sockets.
    """
    global _HTTP_SESSION
    if _HTTP_SESSION is None or _HTTP_SESSION.closed:
        connector = aiohttp.TCPConnector(
            limit=4096, limit_per_host=256,
            force_close=False, enable_cleanup_closed=True,
            keepalive_timeout=600,
        )
        _HTTP_SESSION = aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=30),
        )
    return _HTTP_SESSION


async def _get_instance_stats(instance: Instance) -> dict:
    """Query /instance_stats from a vLLM instance. Returns empty dict on failure."""
    try:
        url = f"http://{instance._ip_address}:{instance._backend_port}/instance_stats"
        session = await _get_http_session()
        async with session.get(
            url, ssl=False, timeout=aiohttp.ClientTimeout(total=3)
        ) as resp:
            if resp.status == 200:
                return await resp.json()
    except Exception:
        pass
    return {}


async def _fetch_all_instance_stats(all_instances: list) -> dict:
    """Fetch /instance_stats from all instances in parallel.

    Returns {instance_id: stats_dict}.
    """
    tasks = [_get_instance_stats(inst) for inst in all_instances]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    stats_map = {}
    for inst, result in zip(all_instances, results):
        if isinstance(result, Exception) or not result:
            stats_map[inst._instance_id] = {}
        else:
            stats_map[inst._instance_id] = result
    return stats_map


async def _call_sidecar_predict_latency(
    inst: Instance, num_prompt_tokens: int, num_predicted_output_tokens: int,
    fields: Optional[list] = None,
) -> Optional[dict]:
    """Call instance's sidecar /predict_latency endpoint.

    Args:
        fields: subset of {"ttft","tpot","e2e"} to request. None = all 3
            (back-compat). When fewer fields are requested, the sidecar skips
            those XGBoost calls and returns null for them — saves predictor
            CPU during main runs that don't need TTFT/TPOT for SLO filtering.

    Returns {"ttft": float|None, "tpot": float|None, "e2e_latency": float|None,
    "probe_latency_ms": float} or None on failure.
    """
    if not inst._predictor_urls:
        return None

    counter = getattr(inst, "_predictor_url_idx", 0)
    chosen = inst._predictor_urls[counter % len(inst._predictor_urls)]
    inst._predictor_url_idx = counter + 1
    base_url = chosen.rsplit("/", 1)[0]
    url = f"{base_url}/predict_latency"

    payload = {
        "num_prompt_tokens": num_prompt_tokens,
        "num_predicted_output_tokens": num_predicted_output_tokens,
    }
    if fields is not None:
        payload["fields"] = fields

    try:
        session = await _get_http_session()
        async with session.post(
            url, json=payload, ssl=False,
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            if resp.status == 200:
                return await resp.json()
            else:
                logger.debug(
                    f"Sidecar predict_latency {inst._instance_id}: "
                    f"status {resp.status}"
                )
    except Exception as e:
        logger.debug(
            f"Sidecar predict_latency {inst._instance_id} failed: {e}"
        )
    return None


def _compute_mean_utilization(
    all_instances: list, stats_map: dict
) -> float:
    """Compute cluster mean GPU memory utilization.

    Uses kv_cache_utilization from instance_stats (0-1 scale).
    Falls back to queue-depth proxy if KV cache stats unavailable.
    """
    if not all_instances:
        return 0.0
    total_util = 0.0
    for inst in all_instances:
        stats = stats_map.get(inst._instance_id, {})
        kv_util = stats.get("kv_cache_utilization")
        if kv_util is not None:
            total_util += float(kv_util)
        else:
            inst_type = instance_meta.get(inst._instance_id, {}).get("instance_type", "")
            capacity = slo_defaults.get("instance_capacity", {}).get(inst_type, 8)
            load = stats.get("num_running", 0) + stats.get("num_waiting", 0)
            total_util += load / max(capacity, 1)
    return total_util / len(all_instances)


def _update_local_stats(stats_map: dict, inst: Instance, req: "BatchRequest"):
    """Update local copy of instance stats after assigning a request.

    Ensures later requests in the batch see load from earlier assignments.
    """
    inst_id = inst._instance_id
    stats = stats_map.get(inst_id, {})
    stats["num_waiting"] = stats.get("num_waiting", 0) + 1
    stats["pending_prefill_tokens"] = (
        stats.get("pending_prefill_tokens", 0) + req.num_prompt_tokens
    )
    stats["pending_decode_tokens"] = (
        stats.get("pending_decode_tokens", 0) + int(req.predicted_tokens)
    )
    stats_map[inst_id] = stats


async def _dispatch_and_resolve(
    inst: Instance, req: "BatchRequest", overhead: dict = None
):
    """Dispatch a request to the assigned instance and resolve its future."""
    try:
        # Use model estimator's prediction if available, else max_output_tokens
        predicted_output = getattr(req, "predicted_tokens", 0) or req.max_output_tokens
        req.request_json["num_predicted_output_tokens"] = predicted_output

        # Per-request budget clamp: if budget_cost > 0, clamp max_tokens so the
        # request can't exceed (budget − input_cost) on the chosen instance.
        # Affordable output tokens = max(0, (budget − input_cost) / out_price).
        # Final max_tokens = min(requested max_tokens, affordable output tokens).
        # If budget < input_cost the request is unaffordable on this instance —
        # clamp to 1 token and mark budget_exhausted=True (immediate near-stop).
        chosen_meta = instance_meta.get(inst._instance_id, {})
        _cpt = chosen_meta.get("cost_per_token", 0.0)
        _c_in = chosen_meta.get("cost_per_input_token", _cpt)
        _c_out = chosen_meta.get("cost_per_output_token", _cpt)
        budget_cost = float(getattr(req, "budget_cost", 0.0) or 0.0)
        budget_exhausted = False
        affordable_out = None
        if budget_cost > 0 and _c_out > 0:
            input_cost = req.num_prompt_tokens * _c_in
            remaining = budget_cost - input_cost
            if remaining <= 0:
                # Budget can't cover even the input — minimum-output mode.
                affordable_out = 1
                budget_exhausted = True
            else:
                affordable_out = max(1, int(remaining / _c_out))
            requested_max = req.request_json.get("max_tokens", req.max_output_tokens)
            req.request_json["max_tokens"] = min(requested_max, affordable_out)

        response_dict = await inst.query_instance(
            req.request_json,
            predicted_num_decode_tokens=predicted_output,
        )
        # Budget bookkeeping for response trace.
        if budget_cost > 0:
            actual_out = int(response_dict.get("output_tokens", 0) or 0)
            actual_cost = req.num_prompt_tokens * _c_in + actual_out * _c_out
            response_dict["budget_cost"] = budget_cost
            response_dict["actual_cost"] = actual_cost
            response_dict["affordable_output_tokens"] = affordable_out
            response_dict["budget_exhausted"] = bool(
                budget_exhausted or actual_cost >= budget_cost - 1e-9
            )
        # Attach scheduling overhead breakdown
        if overhead:
            response_dict["scheduling_overhead_breakdown"] = overhead

        # Attach model estimator quality metrics
        if req.model_estimates:
            selected_model = response_dict.get("model", "")
            est = req.model_estimates.get(selected_model)
            if est:
                response_dict["predicted_quality"] = est.score
                response_dict["predicted_length"] = est.length_expected
                response_dict["p_under_budget"] = est.p_under_budget
            # Model with highest *predicted* quality (KNN-based, not ground truth)
            # Note: KNN tends to favor larger models. True best-model accuracy
            # requires post-hoc comparison with annotated broadcast data.
            best_pred = max(
                req.model_estimates.items(),
                key=lambda x: x[1].score
            )
            response_dict["predicted_best_model"] = best_pred[0]
            response_dict["predicted_best_score"] = best_pred[1].score
            response_dict["predicted_best_hit"] = (selected_model == best_pred[0])
            # All model scores for post-hoc analysis
            response_dict["all_model_scores"] = {
                k: v.score for k, v in req.model_estimates.items()
            }

        if not req.future.done():
            req.future.set_result(response_dict)
    except Exception as e:
        if not req.future.done():
            req.future.set_exception(e)


# Global batched-tpot XGBoost model. Loaded once at startup. Replaces the
# per-instance RouteBalanceLearnedPredictor.predict_tpot path with a single batched
# XGBRegressor.predict() call per (request, tier). 4 parallel calls per request
# (one per active tier), distributed via run_in_executor.
_TPOT_BATCH_MODEL = None


async def _batched_tpot_per_request(req, instances, stats_map, expected_length):
    """Predict tpot per instance for ONE request via 4 parallel batched XGBoost
    calls (one per tier). Returns dict[instance_id -> tpot_seconds].

    Implementation: groups instances by instance_type, builds (n_tier × 23)
    feature matrix per tier using the same `build_feature_vector` the trained
    model expects, fires 4 `XGBRegressor.predict(matrix)` calls concurrently
    via run_in_executor, distributes results back to per-instance keys.

    No HTTP, no per-instance fan-out, no JSON. ~1-2ms wall total.
    """
    if _TPOT_BATCH_MODEL is None:
        return {}
    from route_balance.predictor.route_balance.estimators.xgboost_predictor import build_feature_vector
    import numpy as np
    # Group instances by tier (skip instances whose tier has no booster)
    by_tier = {}
    for inst in instances:
        meta = instance_meta.get(inst._instance_id, {})
        itype = meta.get("instance_type", "")
        if itype and itype in _TPOT_BATCH_MODEL.models:
            by_tier.setdefault(itype, []).append(inst)
    if not by_tier:
        return {}
    loop = asyncio.get_event_loop()

    def _predict_tier(itype, insts):
        rows = []
        for inst in insts:
            stats = stats_map.get(inst._instance_id, {}) or {}
            fv = build_feature_vector(stats, req.num_prompt_tokens, int(expected_length))
            rows.append(fv)
        X = np.stack(rows)
        booster = _TPOT_BATCH_MODEL.models[itype]
        # inplace_predict skips DMatrix construction (~32ms saved per call).
        preds = booster.inplace_predict(X)
        return itype, [float(p) for p in preds]

    # 4 parallel calls — one per tier — via thread pool
    coros = [loop.run_in_executor(None, _predict_tier, it, ins) for it, ins in by_tier.items()]
    results = await asyncio.gather(*coros)

    out = {}
    for itype, preds in results:
        for inst, p in zip(by_tier[itype], preds):
            out[inst._instance_id] = max(0.001, p)  # clamp to floor 1ms
    # Optional dump: every N requests, log full tpot map for variance audit
    if os.environ.get("ROUTE_BALANCE_DEBUG_TPOT") == "1":
        global _TPOT_DUMP_COUNTER
        try:
            _TPOT_DUMP_COUNTER += 1
        except NameError:
            _TPOT_DUMP_COUNTER = 1
        if _TPOT_DUMP_COUNTER % 25 == 0:
            items = sorted(out.items(), key=lambda kv: kv[1])
            summary = " ".join(f"{iid.replace('Qwen-2.5-','')}={v*1000:.1f}ms" for iid, v in items)
            print(f"[TPOT-DUMP req#{_TPOT_DUMP_COUNTER} prompt={req.num_prompt_tokens}t exp_out={expected_length}t] {summary}", flush=True)
    return out


async def _predict_one_for_req(req, inst, stats_map):
    """Compute the prediction tuple (inst, pred_dict, _sc_elapsed_ms) for a
    single (req, inst) pair. Module-level so it can be invoked across all
    (req × inst) pairs in a batch in one asyncio.gather (see
    _collect_batch_preds).
    """
    inst_id = inst._instance_id
    meta = instance_meta.get(inst_id, {})
    # `meta["model_name"]` may be present-but-None (config carries explicit
    # nulls); dict.get only falls back to default when the KEY is absent, not
    # when the VALUE is None. Coerce None → inst._model_name explicitly so
    # req.model_estimates lookup hits and avoids the per-(req,inst) MiniLM
    # encode in the predictor fallback path.
    model_name = meta.get("model_name") or inst._model_name
    inst_type = meta.get("instance_type", "")
    stats = stats_map.get(inst_id, {})

    est = req.model_estimates.get(model_name)
    predictor = learned_predictors.get(inst_id)

    if est is not None:
        quality = est.score
        expected_length = est.length_expected
        p_under_budget = est.p_under_budget
    elif predictor and req.prompt_text:
        pred = predictor.predict_full(
            req.prompt_text, req.num_prompt_tokens,
            req.max_output_tokens,
            schedule_state=stats, budget_tokens=req.budget_tokens,
        )
        quality = pred["quality"]
        expected_length = pred["expected_length"]
        p_under_budget = pred["p_under_budget"]
    else:
        quality = 0.5
        expected_length = float(req.max_output_tokens)
        p_under_budget = 0.5

    # Filter mode determines which sub-predictors we actually need:
    # - filter_enabled=True (appendix run): need TTFT + TPOT + E2E
    # - filter_enabled=False (main run): need E2E only
    _filter_on = slo_defaults.get("filter_enabled", True)
    _need_fields = ["ttft", "tpot", "e2e"] if _filter_on else ["e2e"]

    _sc_start = time.monotonic()
    if os.environ.get("ROUTE_BALANCE_DUMMY_SIDECAR") == "1":
        # Dummy mode for bottleneck-isolation: skip HTTP fan-out, return
        # constant predictions. Used to measure route_balance_serve overhead WITHOUT
        # the sidecar broadcasting cost. Routing decisions become near-static
        # but throughput cap is exposed cleanly.
        sidecar_pred = {"ttft": 100.0, "tpot": 30.0, "e2e_latency": 1000.0,
                        "probe_latency_ms": 0.0}
    elif os.environ.get("ROUTE_BALANCE_INPROC_PREDICTOR") == "1":
        # In-process XGBoost predictor: tpot comes from a single batched call
        # done UPSTREAM by the causal worker (`req._batched_tpot[inst_id]`).
        # Here we just look it up + compute analytical Little's-Law e2e.
        # Falls back to predictor.predict_tpot only if upstream batch missed.
        _bt = getattr(req, "_batched_tpot", None) or {}
        tpot_v = _bt.get(inst_id)
        if tpot_v is not None:
            tpot = float(tpot_v)
            # Analytical e2e (Little's-Law) using REAL per-(request,instance)
            # tpot from XGBoost. ttft is unused when filter_enabled=False.
            ttft = 0.05  # placeholder; not used downstream when filter off
            ss = stats or {}
            running_n = int(ss.get("num_running", 0) or 0)
            waiting_n = int(ss.get("num_waiting", 0) or 0)
            max_seqs = int(ss.get("max_num_seqs", 256) or 256)
            own_out = max(1, int(expected_length or 1))
            slot_open = (running_n < max_seqs) and (waiting_n == 0)
            if slot_open:
                sidecar_e2e = own_out * tpot
            else:
                pending_dec = float(ss.get("pending_decode_tokens", 0) or 0)
                queue_iters = pending_dec / max(running_n, 1)
                sidecar_e2e = (queue_iters + own_out) * tpot
            sidecar_pred = {"ttft": ttft, "tpot": tpot,
                            "e2e_latency": sidecar_e2e, "probe_latency_ms": 0.0}
        else:
            # Upstream batched-tpot didn't cover this instance — fall through to
            # legacy RouteBalanceLearnedPredictor path
            sidecar_pred = None
    else:
        sidecar_pred = await _call_sidecar_predict_latency(
            inst, req.num_prompt_tokens, int(expected_length), fields=_need_fields,
        )
    _sc_elapsed = (time.monotonic() - _sc_start) * 1000

    # Measure in-process XGBoost predict time so the breakdown reflects what
    # we actually do today: HTTP sidecar removed; predictions run directly in
    # the scheduler against the per-instance schedule_state we already fetched
    # via /instance_stats. _xgb_predict_ms = wall time for predict_ttft +
    # predict_tpot + predict_e2e calls (covers the 3 XGBoost models).
    _xgb_predict_ms = 0.0
    if sidecar_pred is not None:
        ttft = sidecar_pred.get("ttft")
        tpot = sidecar_pred.get("tpot")
        # Use dedicated XGBoost E2E prediction directly (Fix 1: no longer
        # derived as ttft + N*tpot when the sidecar provides a real e2e).
        sidecar_e2e = sidecar_pred.get("e2e_latency")
    elif predictor:
        _xgb_t0 = time.monotonic()
        ttft = predictor.predict_ttft(
            stats, req.num_prompt_tokens, int(expected_length)
        )
        tpot = predictor.predict_tpot(
            stats, req.num_prompt_tokens, int(expected_length)
        )
        # In-process predict_e2e (analytical Little's-Law formula) — async
        # in our setup, so await if coroutine.
        try:
            _e2e_call = predictor.predict_e2e(
                stats, req.num_prompt_tokens, int(expected_length)
            )
            if asyncio.iscoroutine(_e2e_call):
                sidecar_e2e = await _e2e_call
            else:
                sidecar_e2e = _e2e_call
        except Exception:
            sidecar_e2e = None
        _xgb_predict_ms = (time.monotonic() - _xgb_t0) * 1000
    else:
        ttft = 1.0
        tpot = 0.05
        sidecar_e2e = None

    # When filter is enabled, TTFT/TPOT are required for SLO checks downstream.
    # If they're missing the run is invalid — fail loudly rather than silently
    # bypass the filter.
    if _filter_on and (ttft is None or tpot is None):
        raise RuntimeError(
            f"slo_filter_enabled=True but sidecar did not return ttft/tpot for "
            f"{inst._instance_id} (ttft={ttft}, tpot={tpot}). Set "
            f"slo_defaults.filter_enabled=False or fix predictor wiring."
        )

    # Cost: support optional input/output split for vendors with asymmetric
    # prompt/completion pricing. Backward compat: if only cost_per_token set,
    # both input and output use that rate (1:1 ratio, our default).
    cost_per_token = meta.get("cost_per_token", 0.01)
    cost_in_per_tok = meta.get("cost_per_input_token", cost_per_token)
    cost_out_per_tok = meta.get("cost_per_output_token", cost_per_token)
    cost = req.num_prompt_tokens * cost_in_per_tok + expected_length * cost_out_per_tok
    # Prefer dedicated E2E XGBoost prediction; fall back to derivation only
    # when sidecar didn't return one (legacy / fallback predictor).
    if sidecar_e2e is not None:
        e2e_latency = sidecar_e2e
    elif ttft is not None and tpot is not None:
        e2e_latency = ttft + expected_length * tpot
    else:
        # Defensive default if neither path produced usable values.
        e2e_latency = float("inf")
    # When in-process predictor is active, override _sc_elapsed to reflect
    # the actual XGBoost predict wall (the HTTP sidecar was bypassed). The
    # existing breakdown field (sidecar_ms) thus tracks "per-(req,inst) per-r
    # prediction wall" regardless of whether predictions came from HTTP or
    # in-process — direct apples-to-apples comparison.
    if sidecar_pred is None and _xgb_predict_ms > 0:
        _sc_elapsed = _xgb_predict_ms
    pred_dict = {
        "quality": quality,
        "expected_length": expected_length,
        "p_under_budget": p_under_budget,
        "ttft": ttft,
        "tpot": tpot,
        "e2e_latency": e2e_latency,
        "cost_per_token": cost_per_token,
        "cost_per_input_token": cost_in_per_tok,
        "cost_per_output_token": cost_out_per_tok,
        "cost": cost,
        "instance_type": inst_type,
        "xgb_predict_ms": _xgb_predict_ms,
    }
    return inst, pred_dict, _sc_elapsed


async def _collect_batch_preds(batch, all_instances, stats_map):
    """Fan out predict_latency for ALL (req × inst) pairs in one
    asyncio.gather. Each instance's sidecar receives len(batch) concurrent
    calls in roughly the same event-loop tick → its opportunistic batcher
    folds them into a single batched predict (~constant cost for batch ≤
    max_batch). Reduces per-batch wall from B × max_per_inst_latency to
    ~1 × max_per_inst_latency (the +B factor was the previous bottleneck).

    Returns list aligned with batch:
        [(preds_for_req_i, sidecar_sum_ms_for_req_i), ...]
    where preds_for_req_i is a list of (inst, pred_dict) — same shape as
    the legacy per-req all_preds.
    """
    n_inst = len(all_instances)
    tasks = []
    for req in batch:
        for inst in all_instances:
            tasks.append(_predict_one_for_req(req, inst, stats_map))
    results = await asyncio.gather(*tasks)
    out = []
    for ri in range(len(batch)):
        slice_ = results[ri * n_inst:(ri + 1) * n_inst]
        preds = [(inst, pd) for inst, pd, _ in slice_]
        sc_sum = sum(sc for _, _, sc in slice_)
        out.append((preds, sc_sum))
    return out


async def _score_and_pick_instance(
    req: "BatchRequest",
    all_instances: list,
    stats_map: dict,
    mean_util: float,
    norm_ctx: dict = None,
    prefetched_preds: list = None,
) -> Instance:
    """Schedule a single request within a batch context (paper Algorithm 1).

    Uses pre-computed ModelEstimates for quality/length (prompt-dependent)
    and instance stats for latency (state-dependent). Falls back to
    per-instance learned predictors if ModelEstimator is not available.

    norm_ctx: Two-phase normalization context.
        - qual_min/max, cost_min/max: batch-wide (fixed for entire batch)
        - lat_min/max, bal_min/max: accumulated running range (expand only)

    prefetched_preds: optional list[(inst, pred_dict)] computed upstream by
        _collect_batch_preds. When provided, the per-instance sidecar
        fan-out is skipped (already done at batch level), making this
        function purely CPU-bound for filter/score/select.
    """
    w_lat = scoring_weights.get("w_latency", 0.3)
    w_cost = scoring_weights.get("w_cost", 0.2)
    w_qual = scoring_weights.get("w_quality", 0.3)
    w_bal = scoring_weights.get("w_balance", 0.2)
    budget_threshold = slo_defaults.get("budget_confidence_threshold", 0.5)
    # assignment_strategy: "scoring" (default, RouteBalance-full multi-objective),
    # "shortest_queue" (row 8a RouteBalance-simple-shortestQ), or "llumnix_minus"
    # (row 8b RouteBalance-simple-llumnix). Operates across the post-filter candidate
    # set, consistent with the design note ("drop joint scoring → LPT + X
    # across all instances").
    assign_strategy = slo_defaults.get("assignment_strategy", "scoring")

    # Collect predictions for all instances. Either reuse the batch-level
    # prefetch (preferred — _collect_batch_preds gathered (B × 18) calls in
    # one round, letting each instance's sidecar opportunistic batcher fold
    # B concurrent reqs into a single ~constant-cost predict) or fall back
    # to per-req fan-out (parallel within req) for callers that don't
    # prefetch (the legacy non-batch path at line 1797).
    _sidecar_total_ms = 0.0
    if prefetched_preds is not None:
        all_preds = prefetched_preds
    else:
        _gathered = await asyncio.gather(
            *[_predict_one_for_req(req, inst, stats_map)
              for inst in all_instances]
        )
        all_preds = [(inst, pd) for inst, pd, _ in _gathered]
        _sidecar_total_ms = sum(sc for _, _, sc in _gathered)

    # --- Filter stage ---
    # If a pluggable SLO filter is configured, use it instead of inline logic
    if slo_filter is not None:
        from route_balance.global_scheduler.route_balance.filters.base import SLOConstraints, InstanceState
        slo_constraints = SLOConstraints(
            ttft_ms=req.ttft_slo_ms,
            tpot_ms=req.tpot_slo_ms,
            budget_tokens=req.budget_tokens,
            quality_min=req.quality_min,
            budget_cost=float(getattr(req, "budget_cost", 0.0) or 0.0),
            num_prompt_tokens=req.num_prompt_tokens,
            max_output_tokens=int(req.request_json.get("max_tokens", req.max_output_tokens)),
        )
        instance_states = []
        for inst, pred in all_preds:
            meta = instance_meta.get(inst._instance_id, {})
            _cpt = meta.get("cost_per_token", 0.0)
            _c_in = meta.get("cost_per_input_token", _cpt)
            _c_out = meta.get("cost_per_output_token", _cpt)
            instance_states.append(InstanceState(
                instance_id=inst._instance_id,
                model_name=meta.get("model_name", ""),
                gpu_type=meta.get("gpu_type", ""),
                predicted_ttft_ms=pred["ttft"] * 1000,
                predicted_tpot_ms=pred["tpot"] * 1000,
                predicted_e2e_ms=pred["e2e_latency"] * 1000,
                num_running=inst._num_running if hasattr(inst, "_num_running") else 0,
                kv_cache_utilization=inst._kv_cache_utilization if hasattr(inst, "_kv_cache_utilization") else 0,
                cost_per_input_token=_c_in,
                cost_per_output_token=_c_out,
            ))
        filter_results = slo_filter.filter(
            slo_constraints, instance_states,
            predicted_output_tokens=int(pred.get("expected_length", 100)),
        )
        accepted_ids = {fr.instance_id for fr in filter_results if fr.accepted}
        candidates = [(inst, pred) for inst, pred in all_preds
                       if inst._instance_id in accepted_ids]

        # If no candidates and filter doesn't handle relaxation → fallback.
        # When the rejection is budget-driven (req has budget_cost set), pick the
        # CHEAPEST instance only (least-over-budget) instead of "accept all" — the
        # latter undoes the filter's intent. For non-budget rejections, keep the
        # existing accept-all behavior.
        if not candidates:
            scheduling_counters["filter_all_rejected"] = scheduling_counters.get("filter_all_rejected", 0) + 1
            req_budget = float(getattr(req, "budget_cost", 0.0) or 0.0)
            if req_budget > 0:
                # Pick instance with smallest worst-case cost (cheapest fallback).
                def _inst_worst_cost(ip):
                    inst, _pred = ip
                    meta = instance_meta.get(inst._instance_id, {})
                    cpt = meta.get("cost_per_token", 0.0)
                    c_in = meta.get("cost_per_input_token", cpt)
                    c_out = meta.get("cost_per_output_token", cpt)
                    max_out = int(req.request_json.get("max_tokens", req.max_output_tokens))
                    return req.num_prompt_tokens * c_in + max_out * c_out
                cheapest = min(all_preds, key=_inst_worst_cost)
                candidates = [cheapest]
                logger.info(
                    f"Batch {req.request_id}: budget filter rejected all, "
                    f"fallback to cheapest instance {cheapest[0]._instance_id}"
                )
            else:
                candidates = list(all_preds)  # non-budget rejection: accept all
                logger.info(
                    f"Batch {req.request_id}: filter rejected all, "
                    f"fallback to {len(candidates)} candidates"
                )
    # --- Filter stage (legacy inline, skipped when pluggable filter is active) ---
    _use_legacy_filter = (slo_filter is None)
    if _use_legacy_filter:
        candidates = []
    for inst, pred in all_preds:
        if not _use_legacy_filter:
            break  # Skip legacy filter loop; candidates set by pluggable filter
        # 1. Budget compliance (cost-based, per-model)
        # RSO budget is in monetary cost. Convert to per-model token budget
        # using the instance's cost_per_token, then check P(tokens ≤ B_model).
        cost_per_tok = pred.get("cost_per_token", 0.01)
        if req.budget_tokens > 0 and cost_per_tok > 0:
            # budget_tokens in the request is cost-based (monetary)
            # Convert: max affordable output tokens = budget / cost_per_token
            model_token_budget = req.budget_tokens  # fallback: raw tokens
            if hasattr(req, "budget_cost") and req.budget_cost > 0:
                model_token_budget = int(req.budget_cost / cost_per_tok)

            # Recompute P(tokens ≤ model_token_budget) from bucket probs
            est = req.model_estimates.get(
                instance_meta.get(inst._instance_id, {}).get("model_name", "")
            )
            if est is not None and est.length_bucket_probs is not None:
                bucket_size = slo_defaults.get("bucket_size", 64)
                budget_bucket = min(
                    model_token_budget // bucket_size,
                    len(est.length_bucket_probs) - 1
                )
                p_budget = float(sum(est.length_bucket_probs[:budget_bucket + 1]))
            else:
                p_budget = pred["p_under_budget"]
        else:
            p_budget = pred["p_under_budget"]

        # Filter block: only when slo_defaults.filter_enabled=True. Disabled
        # for the main run (no SLO filtering, scoring decides).
        _filter_on_check = slo_defaults.get("filter_enabled", True)
        if _filter_on_check and p_budget < budget_threshold:
            scheduling_counters["filtered_budget"] += 1
            logger.debug(
                f"Batch {req.request_id}: {inst._instance_id} filtered "
                f"(budget P={p_budget:.2f} < {budget_threshold})"
            )
            continue
        # 2. TTFT SLO — min(per-request RSO, system-wide global SLO)
        global_slo = slo_defaults.get("global_slo", {})
        ttft_limit = min(req.ttft_slo_ms, global_slo.get("ttft_slo_ms", float("inf")))
        if _filter_on_check and pred["ttft"] is not None and pred["ttft"] * 1000 > ttft_limit:
            scheduling_counters["filtered_ttft"] += 1
            if pred["ttft"] * 1000 > global_slo.get("ttft_slo_ms", float("inf")):
                scheduling_counters["filtered_ttft_global"] += 1
            logger.debug(
                f"Batch {req.request_id}: {inst._instance_id} filtered "
                f"(TTFT {pred['ttft']*1000:.0f}ms > {ttft_limit:.0f}ms)"
            )
            continue
        # 3. TPOT SLO — min(per-request RSO, system-wide global SLO)
        tpot_limit = min(req.tpot_slo_ms, global_slo.get("tpot_slo_ms", float("inf")))
        if _filter_on_check and pred["tpot"] is not None and pred["tpot"] * 1000 > tpot_limit:
            scheduling_counters["filtered_tpot"] += 1
            if pred["tpot"] * 1000 > global_slo.get("tpot_slo_ms", float("inf")):
                scheduling_counters["filtered_tpot_global"] += 1
            logger.debug(
                f"Batch {req.request_id}: {inst._instance_id} filtered "
                f"(TPOT {pred['tpot']*1000:.0f}ms > {tpot_limit:.0f}ms)"
            )
            continue
        # 4. Quality minimum
        if _filter_on_check and pred["quality"] is not None and pred["quality"] < req.quality_min:
            scheduling_counters["filtered_quality"] += 1
            logger.debug(
                f"Batch {req.request_id}: {inst._instance_id} filtered "
                f"(quality {pred['quality']:.2f} < {req.quality_min})"
            )
            continue

        candidates.append((inst, pred))

    # --- Tiered constraint relaxation (legacy, skipped when pluggable filter handles it) ---
    if _use_legacy_filter and not candidates and req.constraint_mode == "TIERED":
        constraint_filters = {
            "budget": lambda p: p["p_under_budget"] >= budget_threshold,
            "ttft": lambda p: p["ttft"] * 1000 <= req.ttft_slo_ms,
            "tpot": lambda p: p["tpot"] * 1000 <= req.tpot_slo_ms,
            "quality": lambda p: p["quality"] >= req.quality_min,
        }
        for skip_name in req.relax_order:
            active_filters = {
                k: v for k, v in constraint_filters.items() if k != skip_name
            }
            relaxed = [
                (inst, pred) for inst, pred in all_preds
                if all(f(pred) for f in active_filters.values())
            ]
            if relaxed:
                scheduling_counters[f"relaxed_{skip_name}"] = scheduling_counters.get(f"relaxed_{skip_name}", 0) + 1
                logger.info(
                    f"Batch {req.request_id}: relaxed {skip_name}, "
                    f"{len(relaxed)} candidates"
                )
                candidates = relaxed
                break
        if not candidates:
            # All single relaxations failed — full relaxation
            scheduling_counters["full_relaxation"] += 1
            candidates = list(all_preds)
            logger.info(
                f"Batch {req.request_id}: full relaxation, "
                f"{len(candidates)} candidates"
            )
    elif _use_legacy_filter and not candidates and req.constraint_mode == "RELAXED":
        candidates = list(all_preds)
    elif _use_legacy_filter and not candidates:
        # STRICT mode fallback to shortest queue
        scheduling_counters["strict_fallback"] += 1
        logger.warning(
            f"Batch {req.request_id}: all instances filtered (STRICT), "
            f"falling back to shortest_queue"
        )
        return await _schedule_shortest_queue(all_instances), _sidecar_total_ms

    if not candidates:
        return await _schedule_shortest_queue(all_instances), _sidecar_total_ms

    # --- Simple-assignment ablation bypass (rows 8a / 8b) ---
    # When assignment_strategy is not "scoring", skip the multi-objective
    # joint scoring and pick by a single load signal instead. This isolates
    # the contribution of RouteBalance's scoring-based joint assignment.
    if assign_strategy in ("shortest_queue", "llumnix_minus"):
        eps = 1e-8
        if assign_strategy == "shortest_queue":
            # min(num_running + num_waiting) across candidates
            def _qlen(inst):
                s = stats_map.get(inst._instance_id, {}) or {}
                return int(s.get("num_running", 0) or 0) + int(
                    s.get("num_waiting", 0) or 0
                )
            best_inst = min((inst for inst, _ in candidates), key=_qlen)
        else:  # llumnix_minus: argmax(free_blocks / max(active, 1))
            def _free_per_active(inst):
                s = stats_map.get(inst._instance_id, {}) or {}
                free = s.get("kv_free_blocks")
                if free is None:
                    util = float(s.get("kv_cache_utilization", 0.0) or 0.0)
                    max_seqs = int(s.get("max_num_seqs", 256) or 256)
                    free = max(0.0, (1.0 - util) * float(max_seqs))
                else:
                    free = float(free)
                active = int(s.get("num_running", 0) or 0) + int(
                    s.get("num_waiting", 0) or 0
                )
                return free / max(active, 1)
            best_inst = max((inst for inst, _ in candidates), key=_free_per_active)
        scheduling_counters["total_scheduled"] += 1
        scheduling_counters[f"assign_{assign_strategy}"] = (
            scheduling_counters.get(f"assign_{assign_strategy}", 0) + 1
        )
        return best_inst, _sidecar_total_ms

    # --- Normalization + Scoring (Avengers-Pro-style max-divide) ---
    # All four sub-scores in [0, 1], higher = better, scoring picks argmax.
    # Replaces legacy min-max which destroyed magnitude info (3B vs 72B both
    # became cost_norm=0 vs 1 regardless of actual 6× ratio).
    #
    #   qual_score = pred[quality]                          # raw [0,1]
    #   cost_score = 1.0 - pred[cost]        / max_cost
    #   lat_score  = 1.0 - pred[e2e_latency] / max_lat
    #   bal_score  = 1.0 - util_i            / max_util
    #   score      = w_qual*qual + w_cost*cost + w_lat*lat + w_bal*bal
    eps = 1e-8

    e2es = [p["e2e_latency"] for _, p in candidates]
    costs = [p["cost"] for _, p in candidates]
    max_lat = max(e2es) if e2es else 1.0
    max_cost = max(costs) if costs else 1.0

    # First pass: compute util_i for every candidate so we can derive max_util
    util_list = []
    for inst, pred in candidates:
        stats = stats_map.get(inst._instance_id, {})
        kv_util = stats.get("kv_cache_utilization")
        if kv_util is not None:
            kv_free = stats.get("kv_free_blocks", 0)
            inst_id = inst._instance_id
            kv_block_size_tokens = slo_defaults.get("instance_config", {}).get(
                inst_id, {}
            ).get("kv_block_size_tokens", 16)
            if kv_free > 0 and float(kv_util) < 1.0:
                kv_total_blocks = kv_free / (1.0 - float(kv_util))
                tokens_needed = req.num_prompt_tokens + pred["expected_length"]
                blocks_needed = tokens_needed / max(kv_block_size_tokens, 1)
                util_i = float(kv_util) + blocks_needed / kv_total_blocks
            else:
                util_i = float(kv_util)
        else:
            capacity = slo_defaults.get("instance_capacity", {}).get(
                pred["instance_type"], 8
            )
            util_i = (
                (stats.get("num_running", 0) + stats.get("num_waiting", 0))
                / max(capacity, 1)
            )
        util_list.append(util_i)
    max_util = max(util_list) if util_list else 1.0

    # Second pass: compute four sub-scores + final score (higher = better).
    # Track instantaneous queue depth per candidate for the shortest-queue
    # tiebreak (Fix A): when same-hardware same-model candidates produce
    # bit-identical (qual,cost,lat) sub-scores, stable-sort would otherwise
    # always pick the first-listed instance, producing pathological
    # within-A30 (or within-V100, etc.) concentration. The XGBoost lat
    # predictor isn't sensitive enough to queue depth at saturation to
    # break the tie via lat_score alone.
    scored = []
    # Per-candidate sub-score detail captured for sampled-debug logging.
    # Keyed by inst_id → dict of all sub-scores + raw inputs.
    score_detail = {}
    for (inst, pred), util_i in zip(candidates, util_list):
        qual_score = float(pred.get("quality", 0.0))                    # raw [0,1]
        cost_score = 1.0 - (pred["cost"]        / (max_cost + eps))
        lat_score  = 1.0 - (pred["e2e_latency"] / (max_lat  + eps))
        bal_score  = 1.0 - (util_i              / (max_util + eps))

        score = (
            w_qual * qual_score
            + w_cost * cost_score
            + w_lat  * lat_score
            + w_bal  * bal_score
        )
        # Instantaneous queue depth from latest stats_map (which reflects
        # local in-batch updates from _update_local_stats).
        st = stats_map.get(inst._instance_id, {})
        queue_depth = (
            int(st.get("num_running", 0) or 0)
            + int(st.get("num_waiting", 0) or 0)
        )
        scored.append((score, queue_depth, inst, pred))
        # Weighted contributions per axis — makes which dimension drove
        # the final score explicit (raw sub-scores live in *_norm fields).
        contrib_qual = w_qual * qual_score
        contrib_cost = w_cost * cost_score
        contrib_lat  = w_lat  * lat_score
        contrib_bal  = w_bal  * bal_score
        score_detail[inst._instance_id] = {
            "model": pred.get("instance_type", inst._model_name),
            "instance_id": inst._instance_id,
            # Normalized sub-scores in [0,1]
            "qual_norm": qual_score, "cost_norm": cost_score,
            "lat_norm":  lat_score,  "bal_norm":  bal_score,
            # Weighted contributions (sum = score)
            "contrib_qual": contrib_qual, "contrib_cost": contrib_cost,
            "contrib_lat":  contrib_lat,  "contrib_bal":  contrib_bal,
            "score": score,
            "queue": queue_depth,
            # Raw inputs to the normalization
            "raw_e2e_latency_s": pred["e2e_latency"],
            "raw_cost":          pred["cost"],
            "raw_quality":       qual_score,
            "raw_util_i":        util_i,
            "stats": {k: st.get(k) for k in
                      ["num_running","num_waiting","kv_cache_utilization","num_active_decode_seqs"]
                      if st.get(k) is not None},
        }

    # argmax with shortest-queue + LRU tiebreaks. Sort by:
    #   1. -score (primary: route_balance's per-instance queue-aware composite)
    #   2. queue_depth ascending (secondary: prefer least-loaded among
    #      score-tied candidates — propagated within batch via
    #      _update_local_stats on stats_map)
    #   3. last_assigned_ts ascending (tertiary: prefer least-recently-used
    #      when score and queue are both tied — eliminates the
    #      "first-listed always wins" skew at batch boundaries when
    #      stats_map is fresh-fetched and all queues=0)
    last_assigned_ts = scheduling_counters.setdefault("_last_assigned_ts", {})
    scored_lru = [
        (s, q, last_assigned_ts.get(inst._instance_id, 0.0), inst, pred)
        for (s, q, inst, pred) in scored
    ]
    scored_lru.sort(key=lambda x: (-x[0], x[1], x[2]))
    best_score, _best_queue, _best_lru, best_inst, best_pred = scored_lru[0]
    last_assigned_ts[best_inst._instance_id] = time.monotonic()
    scheduling_counters["total_scheduled"] += 1

    # Sampled debug logging: write per-candidate score breakdown for one in
    # every N requests. Lets us answer: "are scores actually tied?", "does
    # LRU fire?", "which dimension dominates?". Off by default (sample=0).
    sample_every = int(slo_defaults.get("scoring_debug_sample_every", 0) or 0)
    if sample_every > 0 and scheduling_counters["total_scheduled"] % sample_every == 0:
        try:
            import json as _json
            # rank candidates by descending score (with tiebreak order)
            ranked = sorted(score_detail.items(), key=lambda kv: -kv[1]["score"])
            top_score = ranked[0][1]["score"]
            same_model_top = [iid for iid, d in ranked
                              if d["model"] == ranked[0][1]["model"]][:6]
            tied_with_best = sum(
                1 for _, d in ranked if abs(d["score"] - top_score) < 1e-9
            )
            entry = {
                "ts": time.time(),
                "request_id": req.request_id,
                "n_candidates": len(scored_lru),
                "selected": best_inst._instance_id,
                "selected_score": best_score,
                "selected_queue": _best_queue,
                "selected_lru_ts": _best_lru,
                "tied_with_best_count": tied_with_best,
                "weights": {
                    "qual": w_qual, "lat": w_lat, "cost": w_cost, "bal": w_bal,
                },
                "max_lat": max_lat, "max_cost": max_cost, "max_util": max_util,
                # Top 6 candidates of the SAME model the winner is on
                "same_model_top": [
                    {"id": iid, **score_detail[iid],
                     "lru_ts": last_assigned_ts.get(iid, 0.0)}
                    for iid in same_model_top
                ],
            }
            os.makedirs("experiment_output/logs", exist_ok=True)
            with open("experiment_output/logs/scheduling_debug.jsonl", "a") as _df:
                _df.write(_json.dumps(entry, default=str) + "\n")
        except Exception as _e:
            logger.warning(f"scoring debug log failed: {_e}")

    logger.debug(
        f"Batch {req.request_id}: selected {best_inst._instance_id} "
        f"(score={best_score:.3f}, quality={best_pred['quality']:.2f}, "
        f"e2e={best_pred['e2e_latency']:.2f}s, E[len]={best_pred['expected_length']:.0f})"
    )
    return best_inst, _sidecar_total_ms


prepped_queue: Optional[asyncio.Queue] = None
_batch_id_counter = 0


async def _prepper_loop():
    """Option A: prepper coroutine. Pulls a batch from BatchQueue, runs the
    RoBERTa estimator (in thread pool), computes prompt-dependent
    normalization ranges and LPT sort, then hands off to the single causal
    worker via prepped_queue. Multiple preppers may run concurrently — they
    only do stateless work (no stats_map mutation, no scheduling decisions),
    so concurrency here is safe and overlaps estimator latency with the
    causal worker's per-r loop on prior batches.
    """
    global batch_queue, prepped_queue, _batch_id_counter
    while True:
        try:
            batch = await batch_queue.collect_batch()
            _batch_id_counter += 1
            batch_id = _batch_id_counter
            batch_start = time.monotonic()

            # Stage 1: estimator (offloaded to thread pool — non-blocking).
            lpt_key = slo_defaults.get("lpt_sort_key", "max")
            est_start = time.monotonic()
            if os.environ.get("ROUTE_BALANCE_DUMMY_ESTIMATOR") == "1":
                from route_balance.predictor.route_balance.model_estimator import ModelEstimate
                _model_names = ["Qwen/Qwen2.5-3B","Qwen/Qwen2.5-7B","Qwen/Qwen2.5-14B","Qwen/Qwen2.5-72B"]
                for req in batch:
                    estimates = {mn: ModelEstimate(
                        model_name=mn, length_expected=128.0,
                        length_bucket_probs=None, p_under_budget=0.5,
                        score=0.5, score_type="dummy") for mn in _model_names}
                    req.model_estimates = estimates
                    req.predicted_tokens = 128.0
            elif model_estimator is not None and hasattr(model_estimator, 'estimate_batch'):
                prompts = [req.prompt_text for req in batch]
                budget = batch[0].budget_tokens if batch else 256
                _loop = asyncio.get_event_loop()
                batch_estimates = await _loop.run_in_executor(
                    None, model_estimator.estimate_batch, prompts, budget
                )
                for req, estimates in zip(batch, batch_estimates):
                    req.model_estimates = estimates
                    if estimates:
                        lengths = [est.length_expected for est in estimates.values()]
                        if lpt_key == "max": req.predicted_tokens = max(lengths)
                        elif lpt_key == "min": req.predicted_tokens = min(lengths)
                        elif lpt_key == "mean": req.predicted_tokens = sum(lengths)/len(lengths)
                        else: req.predicted_tokens = max(lengths)
            elif model_estimator is not None:
                for req in batch:
                    req.model_estimates = model_estimator.estimate(
                        req.prompt_text, req.budget_tokens)
                    if req.model_estimates:
                        lengths = [est.length_expected for est in req.model_estimates.values()]
                        req.predicted_tokens = max(lengths) if lpt_key == "max" else (
                            min(lengths) if lpt_key == "min" else sum(lengths)/len(lengths))
            estimator_ms = (time.monotonic() - est_start) * 1000

            # Stage 2: prompt-dependent normalization ranges.
            batch_qual_min, batch_qual_max = float("inf"), float("-inf")
            batch_cost_min, batch_cost_max = float("inf"), float("-inf")
            for req in batch:
                for model_name, est in req.model_estimates.items():
                    batch_qual_min = min(batch_qual_min, est.score)
                    batch_qual_max = max(batch_qual_max, est.score)
                    for inst_id, meta in instance_meta.items():
                        if meta.get("model_name") == model_name:
                            _cpt = meta.get("cost_per_token", 0.01)
                            _cin = meta.get("cost_per_input_token", _cpt)
                            _cout = meta.get("cost_per_output_token", _cpt)
                            cost = req.num_prompt_tokens * _cin + est.length_expected * _cout
                            batch_cost_min = min(batch_cost_min, cost)
                            batch_cost_max = max(batch_cost_max, cost)

            # Stage 3: LPT sort.
            if lpt_key not in ("none", "fifo"):
                batch.sort(key=lambda r: r.predicted_tokens, reverse=True)

            # Hand off to causal worker (single causal serialization point).
            await prepped_queue.put({
                "batch": batch,
                "batch_id": batch_id,
                "batch_start": batch_start,
                "estimator_ms": estimator_ms,
                "qual_range": (batch_qual_min, batch_qual_max),
                "cost_range": (batch_cost_min, batch_cost_max),
            })
        except Exception as e:
            logger.error(f"Prepper loop error: {e}")
            logger.error(traceback.format_exc())
            try:
                if 'batch' in dir():
                    for req in batch:
                        if not req.future.done():
                            req.future.set_exception(e)
            except Exception:
                pass


async def _causal_worker_loop():
    """Option A: single causal worker. Pulls pre-prepped batches from
    prepped_queue and runs the per-r causal loop (Algorithm 1) plus dispatch.
    Strict cross-batch ordering: each batch's stats_map fetch happens AFTER
    the prior batch's per-r loop finished and dispatched. This preserves
    Algorithm 1's intent — request N+1 sees N's assignment via local-state
    updates within batch, and batch K+1 sees batch K's effects via real
    cluster state captured in the next stats_map fetch.

    With ROUTE_BALANCE_PREPPER_WORKERS=1 this should produce identical scheduling
    behavior to the legacy _batch_scheduler_loop (just with estimator off
    the asyncio event loop). With more preppers, the only effect is that
    multiple batches' estimators can overlap with the causal worker's
    per-r loop — improving throughput without altering scheduling decisions.
    """
    global batch_queue, prepped_queue
    while True:
        try:
            prepped = await prepped_queue.get()
            batch = prepped["batch"]
            batch_id = prepped["batch_id"]
            batch_start = prepped["batch_start"]
            estimator_ms = prepped["estimator_ms"]
            batch_qual_min, batch_qual_max = prepped["qual_range"]
            batch_cost_min, batch_cost_max = prepped["cost_range"]
            lpt_key = slo_defaults.get("lpt_sort_key", "max")

            norm_ctx = {
                "qual_min": batch_qual_min, "qual_max": batch_qual_max,
                "cost_min": batch_cost_min, "cost_max": batch_cost_max,
                "lat_min": float("inf"), "lat_max": float("-inf"),
            }

            # Stage 4: stats_map fetch (sees prior batches' dispatched effects).
            stats_start = time.monotonic()
            stats_map = await _fetch_all_instance_stats(instances)
            stats_fetch_ms = (time.monotonic() - stats_start) * 1000
            batch_queue.adapt_params(stats_map, instances)

            for inst in instances:
                s = stats_map.get(inst._instance_id, {})
                load = s.get("num_running", 0) + s.get("num_waiting", 0)
                est_lat = 0.1 * (1 + load * 0.5)
                norm_ctx["lat_min"] = min(norm_ctx["lat_min"], est_lat)
                norm_ctx["lat_max"] = max(norm_ctx["lat_max"], est_lat)

            mean_util = _compute_mean_utilization(instances, stats_map)
            logger.info(
                f"Batch {batch_id}: scheduling {len(batch)} requests "
                f"(mean_util={mean_util:.2f}, LPT_range="
                f"[{batch[-1].predicted_tokens:.0f}, {batch[0].predicted_tokens:.0f}], "
                f"estimator={estimator_ms:.1f}ms, stats={stats_fetch_ms:.1f}ms)"
            )

            # Stage 5: per-r causal loop (Algorithm 1 — unchanged).
            predict_rounds_total_ms = 0.0
            for i, req in enumerate(batch):
                round_start = time.monotonic()
                # Pre-step: batched tpot prediction across all instances of this
                # request, grouped by tier — 4 parallel XGBRegressor.predict()
                # calls (one per active tier). Result attaches to req for
                # _predict_one_for_req to consume. Replaces N separate per-
                # instance predictor.predict_tpot calls (one per inst) with
                # 4 batched calls (one per tier × ~3 rows).
                xgb_batch_ms = 0.0
                if os.environ.get("ROUTE_BALANCE_INPROC_PREDICTOR") == "1":
                    expected_len = 0
                    if req.model_estimates:
                        expected_len = max(1, int(max(
                            est.length_expected for est in req.model_estimates.values()
                        )))
                    else:
                        expected_len = max(1, int(req.max_output_tokens or 256))
                    _xgb_t0 = time.monotonic()
                    req._batched_tpot = await _batched_tpot_per_request(
                        req, instances, stats_map, expected_len
                    )
                    xgb_batch_ms = (time.monotonic() - _xgb_t0) * 1000
                    req._xgb_batch_ms = xgb_batch_ms
                preds_for_req_pairs = await asyncio.gather(*[
                    _predict_one_for_req(req, inst, stats_map)
                    for inst in instances
                ])
                preds_for_req = [(inst, pd) for inst, pd, _ in preds_for_req_pairs]
                sidecar_ms = sum(sc for _, _, sc in preds_for_req_pairs)
                predict_rounds_total_ms += (time.monotonic() - round_start) * 1000

                sched_start = time.monotonic()
                inst, _ = await _score_and_pick_instance(
                    req, instances, stats_map, mean_util, norm_ctx,
                    prefetched_preds=preds_for_req,
                )
                scoring_ms = (time.monotonic() - sched_start) * 1000

                _update_local_stats(stats_map, inst, req)
                mean_util = _compute_mean_utilization(instances, stats_map)

                batch_wait_ms = max(0, (batch_start - req.arrival_time)) * 1000
                total_ms = (time.monotonic() - req.arrival_time) * 1000
                # NOTE: with ROUTE_BALANCE_INPROC_PREDICTOR=1 there is no HTTP sidecar.
                # `sidecar_ms` now sums in-process XGBoost predict times across
                # the 13 per-r calls (per-instance ttft+tpot+e2e wall). Field
                # name retained for backward compat with aggregator scripts;
                # alias `xgb_predict_ms` added for clarity.
                overhead = {
                    "batch_wait_ms": round(batch_wait_ms, 2),
                    "estimator_ms": round(estimator_ms / len(batch), 2),
                    "stats_fetch_ms": round(stats_fetch_ms / len(batch), 2),
                    "sidecar_ms": round(sidecar_ms, 2),
                    "xgb_predict_ms": round(sidecar_ms, 2),
                    "xgb_batched_tpot_ms": round(getattr(req, "_xgb_batch_ms", 0.0), 3),
                    "predict_round_ms": round(predict_rounds_total_ms / len(batch), 2),
                    "scoring_ms": round(scoring_ms, 2),
                    "total_scheduling_ms": round(total_ms, 2),
                    "batch_id": batch_id,
                    "batch_size": len(batch),
                }
                asyncio.create_task(_dispatch_and_resolve(inst, req, overhead))
        except Exception as e:
            logger.error(f"Causal worker loop error: {e}")
            logger.error(traceback.format_exc())
            try:
                if 'batch' in dir():
                    for req in batch:
                        if not req.future.done():
                            req.future.set_exception(e)
            except Exception:
                pass


async def _batch_scheduler_loop():
    """Background loop that collects batches and schedules them.

    Implements paper Algorithm 1: LPT-based greedy with local state updates.
    """
    global batch_queue
    batch_id = 0

    while True:
        try:
            batch = await batch_queue.collect_batch()
            batch_id += 1

            batch_start = time.monotonic()

            # 1. ModelEstimator: predict length + quality — BATCHED for all prompts
            lpt_key = slo_defaults.get("lpt_sort_key", "max")  # max | min | mean
            est_start = time.monotonic()
            if model_estimator is not None and hasattr(model_estimator, 'estimate_batch'):
                # Batch estimation: single embedding + single BERT forward for all prompts.
                # Run the sync RoBERTa fused-batch forward in a thread pool so the
                # asyncio event loop stays free for in-flight stream forwarders (300+
                # concurrent streams under load). Profiling showed the ~80ms torch
                # forward was the single largest contributor to route_balance-mode throughput
                # cap because it blocked the asyncio loop synchronously.
                # Algorithm 1 is unaffected — this is purely a non-blocking I/O fix.
                prompts = [req.prompt_text for req in batch]
                budget = batch[0].budget_tokens if batch else 256
                _loop = asyncio.get_event_loop()
                batch_estimates = await _loop.run_in_executor(
                    None, model_estimator.estimate_batch, prompts, budget
                )
                for req, estimates in zip(batch, batch_estimates):
                    req.model_estimates = estimates
                    if estimates:
                        lengths = [
                            est.length_expected
                            for est in estimates.values()
                        ]
                        if lpt_key == "max":
                            req.predicted_tokens = max(lengths)
                        elif lpt_key == "min":
                            req.predicted_tokens = min(lengths)
                        elif lpt_key == "mean":
                            req.predicted_tokens = sum(lengths) / len(lengths)
                        else:
                            req.predicted_tokens = max(lengths)
            elif model_estimator is not None:
                # Fallback: per-request estimation
                for req in batch:
                    req.model_estimates = model_estimator.estimate(
                        req.prompt_text, req.budget_tokens
                    )
                    if req.model_estimates:
                        lengths = [
                            est.length_expected
                            for est in req.model_estimates.values()
                        ]
                        if lpt_key == "max":
                            req.predicted_tokens = max(lengths)
                        elif lpt_key == "min":
                            req.predicted_tokens = min(lengths)
                        elif lpt_key == "mean":
                            req.predicted_tokens = sum(lengths) / len(lengths)
                        else:
                            req.predicted_tokens = max(lengths)
            estimator_ms = (time.monotonic() - est_start) * 1000

            # 2. Compute batch-wide quality/cost normalization ranges
            # (prompt-dependent signals — available for all req×model pairs)
            batch_qual_min, batch_qual_max = float("inf"), float("-inf")
            batch_cost_min, batch_cost_max = float("inf"), float("-inf")
            for req in batch:
                for model_name, est in req.model_estimates.items():
                    batch_qual_min = min(batch_qual_min, est.score)
                    batch_qual_max = max(batch_qual_max, est.score)
                    # Cost = input_len × cost_per_input + expected_output_len × cost_per_output.
                    # Backward compat: if split rates absent, falls back to cost_per_token (1:1).
                    for inst_id, meta in instance_meta.items():
                        if meta.get("model_name") == model_name:
                            _cpt = meta.get("cost_per_token", 0.01)
                            _cin = meta.get("cost_per_input_token", _cpt)
                            _cout = meta.get("cost_per_output_token", _cpt)
                            cost = req.num_prompt_tokens * _cin + est.length_expected * _cout
                            batch_cost_min = min(batch_cost_min, cost)
                            batch_cost_max = max(batch_cost_max, cost)

            # Running normalization context for state-dependent signals
            # (accumulated across sequential assignments, stats-initialized)
            norm_ctx = {
                "qual_min": batch_qual_min, "qual_max": batch_qual_max,
                "cost_min": batch_cost_min, "cost_max": batch_cost_max,
                "lat_min": float("inf"), "lat_max": float("-inf"),
            }

            # 3. LPT sort (longest predicted output first).
            # lpt_sort_key in {"max","min","mean"} enables LPT; "none"/"fifo" preserves
            # arrival order (RouteBalance-no-LPT / RouteBalance-FIFO ablation row 2).
            if lpt_key not in ("none", "fifo"):
                batch.sort(key=lambda r: r.predicted_tokens, reverse=True)

            # 4. Fetch instance stats ONCE for entire batch
            stats_start = time.monotonic()
            stats_map = await _fetch_all_instance_stats(instances)
            stats_fetch_ms = (time.monotonic() - stats_start) * 1000

            # Adaptive batch sizing for next batch
            batch_queue.adapt_params(stats_map, instances)

            # Initialize latency range from instance stats (no sidecar cost)
            for inst in instances:
                s = stats_map.get(inst._instance_id, {})
                load = s.get("num_running", 0) + s.get("num_waiting", 0)
                # Rough latency estimate from queue depth
                est_lat = 0.1 * (1 + load * 0.5)
                norm_ctx["lat_min"] = min(norm_ctx["lat_min"], est_lat)
                norm_ctx["lat_max"] = max(norm_ctx["lat_max"], est_lat)

            # 5. Compute cluster mean utilization
            mean_util = _compute_mean_utilization(instances, stats_map)

            logger.info(
                f"Batch {batch_id}: scheduling {len(batch)} requests "
                f"(mean_util={mean_util:.2f}, LPT_range="
                f"[{batch[-1].predicted_tokens:.0f}, {batch[0].predicted_tokens:.0f}], "
                f"estimator={estimator_ms:.1f}ms, stats={stats_fetch_ms:.1f}ms)"
            )

            # 6. Per-request causal schedule loop (paper Algorithm 1).
            # For each r in LPT order: fan out 18 sidecar predicts FOR THIS r
            # (using current S_i state), score, pick i*, update S_i* locally,
            # then proceed to r+1. Within-batch causality: r_{k+1}'s predict
            # round sees r_k's assignment. Earlier prefetch-then-reuse made
            # all preds independent of dispatch order — wrong per Algo 1.
            # predict_rounds_total_ms accumulates per-r predict-round walls
            # across the batch (B requests × 18 sidecars/round). Reported
            # divided by B as the average per-request predict-round overhead.
            predict_rounds_total_ms = 0.0
            for i, req in enumerate(batch):
                round_start = time.monotonic()
                xgb_batch_ms = 0.0
                if os.environ.get("ROUTE_BALANCE_INPROC_PREDICTOR") == "1":
                    if req.model_estimates:
                        expected_len = max(1, int(max(
                            est.length_expected for est in req.model_estimates.values()
                        )))
                    else:
                        expected_len = max(1, int(req.max_output_tokens or 256))
                    _xgb_t0 = time.monotonic()
                    req._batched_tpot = await _batched_tpot_per_request(
                        req, instances, stats_map, expected_len
                    )
                    xgb_batch_ms = (time.monotonic() - _xgb_t0) * 1000
                preds_for_req_pairs = await asyncio.gather(*[
                    _predict_one_for_req(req, inst, stats_map)
                    for inst in instances
                ])
                preds_for_req = [(inst, pd) for inst, pd, _ in preds_for_req_pairs]
                sidecar_ms = sum(sc for _, _, sc in preds_for_req_pairs)
                predict_rounds_total_ms += (time.monotonic() - round_start) * 1000

                sched_start = time.monotonic()
                inst, _ = await _score_and_pick_instance(
                    req, instances, stats_map, mean_util, norm_ctx,
                    prefetched_preds=preds_for_req,
                )
                scoring_ms = (time.monotonic() - sched_start) * 1000

                # Local state update BEFORE next r's predict round so the
                # next request sees this assignment (causality).
                _update_local_stats(stats_map, inst, req)
                mean_util = _compute_mean_utilization(instances, stats_map)

                # Build overhead breakdown (all times use time.monotonic())
                batch_wait_ms = max(0, (batch_start - req.arrival_time)) * 1000
                total_ms = (time.monotonic() - req.arrival_time) * 1000
                # NOTE: with ROUTE_BALANCE_INPROC_PREDICTOR=1 there is no HTTP sidecar.
                # `sidecar_ms` now sums in-process XGBoost predict times across
                # the 13 per-r calls (per-instance ttft+tpot+e2e wall). Field
                # name retained for backward compat with aggregator scripts;
                # alias `xgb_predict_ms` added for clarity.
                overhead = {
                    "batch_wait_ms": round(batch_wait_ms, 2),
                    "estimator_ms": round(estimator_ms / len(batch), 2),
                    "stats_fetch_ms": round(stats_fetch_ms / len(batch), 2),
                    "sidecar_ms": round(sidecar_ms, 2),
                    "xgb_predict_ms": round(sidecar_ms, 2),
                    "xgb_batched_tpot_ms": round(xgb_batch_ms, 3),
                    "predict_round_ms": round(predict_rounds_total_ms / len(batch), 2),
                    "scoring_ms": round(scoring_ms, 2),
                    "total_scheduling_ms": round(total_ms, 2),
                    "batch_id": batch_id,
                    "batch_size": len(batch),
                }

                # 7. Dispatch immediately, resolve future when done
                asyncio.create_task(
                    _dispatch_and_resolve(inst, req, overhead)
                )

            # 8. Save batch trace for offline Pareto analysis (sampled)
            trace_rate = slo_defaults.get("trace_sample_rate", 0)
            if trace_rate > 0 and (batch_id % max(1, int(1 / trace_rate))) == 0:
                import json as _json
                trace_dir = "experiment_output/traces"
                os.makedirs(trace_dir, exist_ok=True)
                trace = {
                    "batch_id": batch_id,
                    "batch_size": len(batch),
                    "norm_ctx": {k: v for k, v in norm_ctx.items() if not isinstance(v, float) or v != float("inf")},
                    "requests": [
                        {
                            "request_id": r.request_id,
                            "prompt_tokens": r.num_prompt_tokens,
                            "predicted_tokens": r.predicted_tokens,
                            "model_estimates": {
                                m: {"quality": e.score, "length": e.length_expected}
                                for m, e in r.model_estimates.items()
                            },
                        }
                        for r in batch
                    ],
                    "instances": [
                        {
                            "id": inst._instance_id,
                            "model": instance_meta.get(inst._instance_id, {}).get("model_name", ""),
                            "stats": {k: v for k, v in stats_map.get(inst._instance_id, {}).items()
                                      if isinstance(v, (int, float))},
                        }
                        for inst in instances
                    ],
                }
                with open(f"{trace_dir}/batch_{batch_id}.json", "w") as tf:
                    _json.dump(trace, tf)

        except Exception as e:
            logger.error(f"Batch scheduler loop error: {e}")
            logger.error(traceback.format_exc())
            # Resolve any pending futures with the error
            if 'batch' in dir():
                for req in batch:
                    if not req.future.done():
                        req.future.set_exception(e)


async def _select_instance(
    strategy: str,
    all_instances: list,
    req_count: int,
    request_json: dict,
    prompt_text: str,
    num_prompt_tokens: int,
    max_output_tokens: int,
    budget_tokens: int,
    ttft_slo_ms: float,
    tpot_slo_ms: float,
    quality_min: float,
    request_id: str,
) -> tuple:
    """Select an instance based on the scheduling strategy.

    Returns:
        (Instance, predicted_output_tokens): The selected instance and the
        predicted output length (from model estimator if available, else max_tokens).
    """
    predicted_output = max_output_tokens  # default fallback

    if strategy == "random":
        return random.choice(all_instances), predicted_output

    elif strategy == "round_robin":
        return all_instances[req_count % len(all_instances)], predicted_output

    elif strategy == "shortest_queue":
        return await _schedule_shortest_queue(all_instances), predicted_output

    elif strategy == "quality_greedy":
        return await _schedule_quality_greedy(all_instances), predicted_output

    elif strategy == "cost_greedy":
        return await _schedule_cost_greedy(all_instances), predicted_output

    elif strategy == "length_aware":
        inst, pred = await _schedule_length_aware(
            all_instances, prompt_text, num_prompt_tokens, max_output_tokens
        )
        return inst, pred if pred else predicted_output

    elif strategy == "route_balance":
        inst, pred = await _schedule_route_balance(
            all_instances, prompt_text, num_prompt_tokens, max_output_tokens,
            budget_tokens, ttft_slo_ms, tpot_slo_ms, quality_min, request_id,
        )
        return inst, pred if pred else predicted_output

    else:
        raise RuntimeError(
            f"Unknown scheduling strategy '{strategy}'. "
            f"Valid: random/round_robin/shortest_queue/quality_greedy/"
            f"cost_greedy/length_aware/route_balance/pipeline. "
            f"Refusing to silently fall back to random — "
            f"cf. feedback_data_integrity_rules.md rule #6 (fail loudly)."
        )


def _unique_model_pool(all_instances: list) -> list:
    """Return deduplicated list of model names across instances."""
    seen = set()
    pool = []
    for inst in all_instances:
        mn = getattr(inst, "_model_name", None)
        if mn is None or mn in seen:
            continue
        seen.add(mn)
        pool.append(mn)
    return pool


async def _select_via_pipeline(
    all_instances: list,
    *,
    prompt_text: str,
    num_prompt_tokens: int,
    max_output_tokens: int,
    budget_tokens: int,
    ttft_slo_ms: float,
    tpot_slo_ms: float,
    quality_min: float,
    request_id: str,
) -> tuple:
    """Pluggable pipeline: router.choose_model → filter to model instances →
    dispatcher.choose_instance. Optional SLO filter applied after model
    selection, before dispatch.

    Returns: (Instance, predicted_output_tokens, overhead_breakdown_ms).
    overhead_breakdown_ms keys: router_ms, estimator_ms, stats_fetch_ms,
    filter_ms, dispatcher_ms, total_sched_ms.
    """
    from route_balance.global_scheduler.route_balance.routers import RouterRequest
    from route_balance.global_scheduler.route_balance.dispatch import DispatchRequest

    breakdown: dict = {}
    t_total = time.time()

    if not all_instances:
        raise ValueError("_select_via_pipeline: no instances available")

    pool = _unique_model_pool(all_instances)
    rreq = RouterRequest(
        prompt=prompt_text,
        num_prompt_tokens=num_prompt_tokens,
        max_output_tokens=max_output_tokens,
        budget_tokens=budget_tokens,
        ttft_slo_ms=ttft_slo_ms,
        tpot_slo_ms=tpot_slo_ms,
        quality_min=quality_min,
        request_id=request_id,
    )
    t0 = time.time()
    decision = await router.choose_model(rreq, pool)
    breakdown["router_ms"] = (time.time() - t0) * 1000.0
    chosen_model = decision.model_name

    # Passthrough sentinel: dispatcher-only baselines (RR/SQ/Random over ALL
    # instances). Skip the per-model candidate filter so the dispatcher sees
    # the full instance pool.
    if chosen_model == "__ALL__":
        candidates = list(all_instances)
    else:
        candidates = [i for i in all_instances if i._model_name == chosen_model]
        if not candidates:
            raise ValueError(
                f"Router chose {chosen_model!r} but no instance serves it "
                f"(pool={pool})"
            )

    # Predicted output length — pipeline-mode routers (random/RR/avengers/
    # routellm/best_route) don't use length prediction for routing; only the
    # route_balance batch worker does. Skip the sync RoBERTa forward entirely here —
    # it was a hidden ~70-90ms per-request cap that serialized concurrent
    # pipeline handlers on the asyncio event loop. predicted_output falls
    # back to the request's max_output_tokens, which is the safe upper bound.
    predicted_output = max_output_tokens
    breakdown["estimator_ms"] = 0.0

    # Fetch /instance_stats only when the dispatcher OR filter actually needs
    # them. Round-robin doesn't read stats; baselines with filter=none can
    # skip the per-request fan-out entirely (was a 87-255 ms unfair-overhead
    # tax on pipeline-mode wrappers vs route_balance-mode).
    needs_stats = getattr(dispatcher, "requires_stats", True) or (
        slo_filter is not None and getattr(slo_filter, "requires_stats", True)
    )
    if needs_stats:
        t0 = time.time()
        stats_map = await _fetch_all_instance_stats(candidates)
        breakdown["stats_fetch_ms"] = (time.time() - t0) * 1000.0
    else:
        stats_map = {}
        breakdown["stats_fetch_ms"] = 0.0

    # Optional SLO filter applied within the model's candidate pool.
    if slo_filter is not None:
        t0 = time.time()
        try:
            from route_balance.global_scheduler.route_balance.filters.base import (
                InstanceState,
                SLOConstraints,
            )

            inst_states = []
            for inst in candidates:
                s = stats_map.get(inst._instance_id, {}) or {}
                inst_states.append(
                    InstanceState(
                        instance_id=inst._instance_id,
                        model_name=inst._model_name,
                        gpu_type=instance_meta.get(inst._instance_id, {}).get(
                            "gpu_type", "unknown"
                        ),
                        num_running=int(s.get("num_running", 0) or 0),
                        num_waiting=int(s.get("num_waiting", 0) or 0),
                        kv_cache_utilization=float(
                            s.get("kv_cache_utilization", 0.0) or 0.0
                        ),
                    )
                )
            slo_c = SLOConstraints(
                ttft_ms=ttft_slo_ms,
                tpot_ms=tpot_slo_ms,
                budget_tokens=budget_tokens,
                quality_min=quality_min,
            )
            eligible = slo_filter.get_eligible(
                slo_c, inst_states, predicted_output
            )
            n_before = len(candidates)
            if eligible:
                eligible_ids = {e.instance_id for e in eligible}
                candidates = [
                    i for i in candidates if i._instance_id in eligible_ids
                ] or candidates
            n_filtered = n_before - len(candidates)
            if n_filtered > 0:
                # Bucket the rejection reason by inspecting the filter's per-
                # instance results. Fall back to a generic counter when reasons
                # aren't structured.
                results = slo_filter.filter(slo_c, inst_states, predicted_output)
                for fr in results:
                    if fr.accepted:
                        continue
                    reason = (fr.reason or "").lower()
                    if "ttft" in reason:
                        scheduling_counters["filtered_ttft"] += 1
                    elif "tpot" in reason:
                        scheduling_counters["filtered_tpot"] += 1
                    elif "quality" in reason:
                        scheduling_counters["filtered_quality"] += 1
                    elif "budget" in reason or "cost" in reason:
                        scheduling_counters["filtered_budget"] += 1
                    else:
                        scheduling_counters.setdefault("filtered_other", 0)
                        scheduling_counters["filtered_other"] += 1
            # If filter rejects all, we keep original candidates (fallback).
        except Exception as e:
            logger.debug("slo_filter failed in pipeline path: %s", e)
        breakdown["filter_ms"] = (time.time() - t0) * 1000.0

    dreq = DispatchRequest(
        prompt=prompt_text,
        num_prompt_tokens=num_prompt_tokens,
        max_output_tokens=max_output_tokens,
        predicted_output_tokens=predicted_output,
        ttft_slo_ms=ttft_slo_ms,
        tpot_slo_ms=tpot_slo_ms,
        request_id=request_id,
    )
    t0 = time.time()
    inst = await dispatcher.choose_instance(
        candidates, dreq, stats_map=stats_map
    )
    breakdown["dispatcher_ms"] = (time.time() - t0) * 1000.0
    breakdown["total_sched_ms"] = (time.time() - t_total) * 1000.0
    scheduling_counters["total_scheduled"] += 1
    return inst, predicted_output, breakdown


async def _schedule_shortest_queue(all_instances: list) -> Instance:
    """Pick instance with fewest pending requests."""
    if not all_instances:
        raise ValueError("No instances available for scheduling")
    stats_tasks = [_get_instance_stats(inst) for inst in all_instances]
    stats_list = await asyncio.gather(*stats_tasks, return_exceptions=True)

    best_inst = all_instances[0]
    best_queue = float("inf")

    for inst, stats in zip(all_instances, stats_list):
        if isinstance(stats, Exception) or not stats:
            queue_len = inst.total_request  # fallback to total requests served
        else:
            queue_len = stats.get("num_running", 0) + stats.get("num_waiting", 0)
        if queue_len < best_queue:
            best_queue = queue_len
            best_inst = inst

    return best_inst


def _model_size_key(model_name: str) -> int:
    """Extract numeric model size from name for ordering (larger = higher)."""
    import re
    # Match patterns like "3B", "7B", "14B", "72B"
    match = re.search(r'(\d+)[Bb]', model_name)
    if match:
        return int(match.group(1))
    return 0


async def _schedule_quality_greedy(all_instances: list) -> Instance:
    """Pick largest model with shortest queue (quality-maximizing)."""
    # Group by model size, pick largest group
    by_size: dict[int, list] = {}
    for inst in all_instances:
        size = _model_size_key(inst._model_name)
        by_size.setdefault(size, []).append(inst)

    # Try from largest to smallest
    for size in sorted(by_size.keys(), reverse=True):
        group = by_size[size]
        if group:
            # Shortest queue within the group
            stats_tasks = [_get_instance_stats(inst) for inst in group]
            stats_list = await asyncio.gather(*stats_tasks, return_exceptions=True)

            best_inst = group[0]
            best_queue = float("inf")
            for inst, stats in zip(group, stats_list):
                if isinstance(stats, Exception) or not stats:
                    queue_len = 999
                else:
                    queue_len = stats.get("num_running", 0) + stats.get("num_waiting", 0)
                if queue_len < best_queue:
                    best_queue = queue_len
                    best_inst = inst
            return best_inst

    return random.choice(all_instances)


async def _schedule_cost_greedy(all_instances: list) -> Instance:
    """Pick smallest model with shortest queue (cost-minimizing)."""
    by_size: dict[int, list] = {}
    for inst in all_instances:
        size = _model_size_key(inst._model_name)
        by_size.setdefault(size, []).append(inst)

    # Try from smallest to largest
    for size in sorted(by_size.keys()):
        group = by_size[size]
        if group:
            stats_tasks = [_get_instance_stats(inst) for inst in group]
            stats_list = await asyncio.gather(*stats_tasks, return_exceptions=True)

            best_inst = group[0]
            best_queue = float("inf")
            for inst, stats in zip(group, stats_list):
                if isinstance(stats, Exception) or not stats:
                    queue_len = 999
                else:
                    queue_len = stats.get("num_running", 0) + stats.get("num_waiting", 0)
                if queue_len < best_queue:
                    best_queue = queue_len
                    best_inst = inst
            return best_inst

    return random.choice(all_instances)


async def _schedule_length_aware(
    all_instances: list, prompt_text: str,
    num_prompt_tokens: int, max_output_tokens: int,
) -> tuple:
    """Use length prediction to route to instance that minimizes estimated
    completion time (like L4/S3 baselines).

    Returns:
        (Instance, predicted_output_tokens)
    """
    # Get predictions from learned predictors if available
    stats_tasks = [_get_instance_stats(inst) for inst in all_instances]
    stats_list = await asyncio.gather(*stats_tasks, return_exceptions=True)

    best_inst = all_instances[0]
    best_time = float("inf")
    best_predicted = max_output_tokens

    for inst, stats in zip(all_instances, stats_list):
        if isinstance(stats, Exception) or not stats:
            stats = {}

        inst_id = inst._instance_id
        predictor = learned_predictors.get(inst_id)

        if predictor and prompt_text:
            # Use learned predictor for length estimate
            pred = predictor.predict_full(
                prompt_text, num_prompt_tokens, max_output_tokens,
                schedule_state=stats,
            )
            expected_length = pred["expected_length"]
            tpot = pred["tpot"]
            ttft = pred["ttft"]
        else:
            # Fallback: assume max output tokens
            expected_length = max_output_tokens
            tpot = 0.05
            ttft = 1.0

        # Estimated completion = TTFT + expected_length * TPOT + queue wait
        queue_len = stats.get("num_running", 0) + stats.get("num_waiting", 0)
        queue_wait = queue_len * expected_length * tpot  # rough estimate
        est_time = ttft + expected_length * tpot + queue_wait

        if est_time < best_time:
            best_time = est_time
            best_inst = inst
            best_predicted = int(expected_length)

    return best_inst, best_predicted


async def _schedule_route_balance(
    all_instances: list, prompt_text: str,
    num_prompt_tokens: int, max_output_tokens: int,
    budget_tokens: int, ttft_slo_ms: float, tpot_slo_ms: float,
    quality_min: float, request_id: str,
) -> tuple:
    """Per-request ROUTE_BALANCE scheduling (batch_size=1).

    Uses the same _score_and_pick_instance() logic as the batch path to ensure
    identical behavior. This path is used when batch_queue is not initialized
    (e.g., other strategies calling route_balance as fallback).

    Returns:
        (Instance, predicted_output_tokens): Selected instance and predicted
        output length from model estimator (or max_output_tokens if no estimator).
    """
    # Build a single-request BatchRequest
    req = BatchRequest(
        request_json={},  # not needed for scheduling decision
        prompt_text=prompt_text,
        num_prompt_tokens=num_prompt_tokens,
        max_output_tokens=max_output_tokens,
        budget_tokens=budget_tokens,
        ttft_slo_ms=ttft_slo_ms,
        tpot_slo_ms=tpot_slo_ms,
        quality_min=quality_min,
        request_id=request_id,
    )

    predicted_output = max_output_tokens

    # Run ModelEstimator if available
    if model_estimator is not None:
        req.model_estimates = model_estimator.estimate(prompt_text, budget_tokens)
        if req.model_estimates:
            req.predicted_tokens = max(
                est.length_expected for est in req.model_estimates.values()
            )
            predicted_output = req.predicted_tokens

    # Fetch instance stats
    stats_map = await _fetch_all_instance_stats(all_instances)
    mean_util = _compute_mean_utilization(all_instances, stats_map)

    inst, _ = await _score_and_pick_instance(req, all_instances, stats_map, mean_util)
    return inst, predicted_output


def _load_scheduler_config(config_path: str, all_instances: list):
    """Load scheduler config from dedicated scheduler_config.json.

    Handles: scoring_weights, slo_defaults, model_estimator, instance_metadata.
    Does NOT load sidecar predictors (those come from predictor_config via sidecar).
    """
    global scoring_weights, slo_defaults, instance_meta, model_estimator, slo_filter
    global router, dispatcher

    try:
        with open(config_path) as f:
            config_dict = json.load(f)

        # Update scoring weights and SLO defaults
        if "scoring_weights" in config_dict:
            scoring_weights.update(config_dict["scoring_weights"])
        if "slo_defaults" in config_dict:
            slo_defaults.update(config_dict["slo_defaults"])

        # Load SLO filter (pluggable)
        filter_config = config_dict.get("slo_defaults", {}).get("filter")
        if filter_config:
            from route_balance.global_scheduler.route_balance.filters.factory import create_filter
            slo_filter = create_filter(filter_config)
            logger.info(f"SLO filter loaded: {filter_config.get('type', 'route_balance_tiered')}")
        else:
            slo_filter = None  # Use inline legacy filter
            logger.info("SLO filter: using inline legacy (no filter config)")

        # A0.1 pluggable pipeline (router + dispatcher). When scheduling_pipeline
        # is present, the scheduler accepts `--scheduling pipeline` and dispatches
        # via (router → filter → dispatcher). Legacy --scheduling strategies
        # continue to work unchanged when this block is absent.
        pipe_cfg = config_dict.get("scheduling_pipeline")
        if pipe_cfg:
            from route_balance.global_scheduler.route_balance.routers.factory import create_router
            from route_balance.global_scheduler.route_balance.dispatch.factory import (
                create_dispatcher,
            )
            router_cfg = pipe_cfg.get("router") or {"type": "route_balance_native"}
            dispatch_cfg = pipe_cfg.get("dispatch") or {"type": "route_balance_native"}
            # Filter may live in the pipeline block OR in slo_defaults.filter.
            if not filter_config and pipe_cfg.get("filter"):
                from route_balance.global_scheduler.route_balance.filters.factory import (
                    create_filter,
                )
                slo_filter = create_filter(pipe_cfg["filter"])
                logger.info(
                    f"SLO filter loaded from pipeline: "
                    f"{pipe_cfg['filter'].get('type', 'route_balance_tiered')}"
                )
            try:
                router = create_router(
                    router_cfg,
                    model_estimator=None,  # set below after estimator load
                    scoring_weights=scoring_weights,
                )
            except NotImplementedError as e:
                logger.error(
                    "Router %s is a stub: %s. Pipeline disabled.",
                    router_cfg.get("type"), e,
                )
                router = None
            try:
                dispatcher = create_dispatcher(
                    dispatch_cfg,
                    scoring_weights=scoring_weights,
                )
            except NotImplementedError as e:
                logger.error(
                    "Dispatcher %s is a stub: %s. Pipeline disabled.",
                    dispatch_cfg.get("type"), e,
                )
                dispatcher = None
            if router is not None and dispatcher is not None:
                logger.info(
                    "scheduling_pipeline active: router=%s dispatch=%s "
                    "filter=%s",
                    router_cfg.get("type"),
                    dispatch_cfg.get("type"),
                    (filter_config or pipe_cfg.get("filter", {})).get(
                        "type", "none"
                    ),
                )

        # Load ModelEstimator
        me_config = config_dict.get("model_estimator")
        if me_config:
            me_type = me_config.get("type", "default")
            if me_type == "pfs":
                from route_balance.predictor.route_balance.model_estimator import PFSModelEstimator
                model_estimator = PFSModelEstimator(me_config)
            elif me_type == "knn":
                from route_balance.predictor.route_balance.model_estimator import KNNModelEstimator
                model_estimator = KNNModelEstimator(me_config)
            else:
                from route_balance.predictor.route_balance.model_estimator import DefaultModelEstimator
                model_estimator = DefaultModelEstimator(me_config)
            logger.info(
                f"ModelEstimator loaded: type={me_type}, "
                f"{len(model_estimator.model_names)} models"
            )
            # Back-fill the estimator handle into the router if the router was
            # created before the estimator was loaded.
            if router is not None and hasattr(router, "_model_estimator"):
                router._model_estimator = model_estimator

        # Build instance_type mapping from instance metadata
        inst_metadata = config_dict.get("instance_metadata", {})
        for inst in all_instances:
            for inst_type, meta in inst_metadata.items():
                if meta.get("model_name") == inst._model_name:
                    instance_meta[inst._instance_id] = {
                        **meta, "instance_type": inst_type
                    }
                    break

        logger.info(
            f"Scheduler config loaded: {len(instance_meta)} instances mapped, "
            f"scoring_weights={scoring_weights}"
        )

    except Exception as e:
        logger.error(f"Failed to load scheduler config: {e}")
        import traceback
        traceback.print_exc()


def _load_learned_predictors(config_path: str, all_instances: list):
    """Legacy: Load everything from a single predictor config (backward compat)."""
    global learned_predictors, scoring_weights, slo_defaults, instance_meta, model_estimator, slo_filter

    try:
        with open(config_path) as f:
            config_dict = json.load(f)

        from route_balance.predictor.route_balance.route_balance_predictor_config import RouteBalanceBasePredictorConfig
        config = RouteBalanceBasePredictorConfig.from_dict(config_dict)

        # Update global scoring weights and SLO defaults
        scoring_weights.update(config.scoring_weights)
        slo_defaults.update(config.slo_defaults)

        # Load ModelEstimator (shared, prompt-dependent predictions)
        me_config = config_dict.get("model_estimator")
        if me_config:
            try:
                me_type = me_config.get("type", "default")
                if me_type == "pfs":
                    from route_balance.predictor.route_balance.model_estimator import PFSModelEstimator
                    model_estimator = PFSModelEstimator(me_config)
                    logger.info(
                        f"PFSModelEstimator loaded: {len(model_estimator.model_names)} models, "
                        f"window={me_config.get('window_size', 1000)}, "
                        f"binned={me_config.get('use_input_bins', True)}"
                    )
                elif me_type == "knn":
                    from route_balance.predictor.route_balance.model_estimator import KNNModelEstimator
                    model_estimator = KNNModelEstimator(me_config)
                    logger.info(
                        f"KNNModelEstimator loaded: {len(model_estimator.model_names)} models, "
                        f"score_type={me_config.get('score_type', 'judge')}"
                    )
                else:
                    from route_balance.predictor.route_balance.model_estimator import DefaultModelEstimator
                    model_estimator = DefaultModelEstimator(me_config)
                    logger.info(
                        f"DefaultModelEstimator loaded: {len(model_estimator.model_names)} models, "
                        f"score_type={me_config.get('score_type', 'judge')}"
                    )
            except Exception as e:
                logger.warning(f"Failed to load ModelEstimator: {e}")

        # Build instance_type mapping from instance metadata
        inst_metadata = config.instance_metadata

        for inst in all_instances:
            # Try to determine instance type from model name + metadata
            for inst_type, meta in inst_metadata.items():
                if meta.get("model_name") == inst._model_name:
                    instance_meta[inst._instance_id] = {
                        **meta, "instance_type": inst_type
                    }

                    # Create learned predictor for this instance
                    try:
                        from route_balance.predictor.route_balance.route_balance_learned_predictor import RouteBalanceLearnedPredictor
                        predictor = RouteBalanceLearnedPredictor(
                            config=config,
                            port=inst._backend_port,
                            hostname=inst._hostname,
                            instance_type=inst_type,
                        )
                        learned_predictors[inst._instance_id] = predictor
                        logger.info(
                            f"Loaded learned predictor for {inst._instance_id} "
                            f"(type={inst_type})"
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to load learned predictor for {inst._instance_id}: {e}"
                        )
                    break

        logger.info(
            f"Loaded {len(learned_predictors)} learned predictors, "
            f"scoring_weights={scoring_weights}"
        )

    except Exception as e:
        logger.error(f"Failed to load learned predictor config: {e}")



def _log_and_assert_deployed_predictors():
    """Log loaded predictor encoder name + file SHA. Fail loud on mismatch.

    Reads models/route_balance/length_bucket/deploy/fused_config.json and
    models/route_balance/quality/judge/deploy/fused_config.json. Log includes:
      - encoder_name (e.g. roberta-base, answerdotai/ModernBERT-base)
      - target (length_bucket / deepeval)
      - first 16 chars of fused_model.pt SHA-256 for audit.
    If env var EXPECTED_LENGTH_BUCKET_ENCODER is set, asserts equality.
    Same for EXPECTED_QUALITY_JUDGE_ENCODER. Mismatch raises RuntimeError.
    """
    import os, json, hashlib
    from pathlib import Path
    base = Path(os.environ.get('ROUTE_BALANCE_MODELS_DIR', '/users/anon/RouteBalance/models/route_balance'))
    for label, sub, env_var in [
        ('LENGTH_BUCKET', 'length_bucket/deploy', 'EXPECTED_LENGTH_BUCKET_ENCODER'),
        ('QUALITY_JUDGE', 'quality/judge/deploy', 'EXPECTED_QUALITY_JUDGE_ENCODER'),
    ]:
        cfg_p = base / sub / 'fused_config.json'
        mdl_p = base / sub / 'fused_model.pt'
        if not cfg_p.exists():
            logger.warning(f'PREDICTOR_AUDIT[{label}]: missing fused_config.json at {cfg_p}')
            continue
        try:
            cfg = json.load(open(cfg_p))
        except Exception as e:
            logger.warning(f'PREDICTOR_AUDIT[{label}]: failed to parse {cfg_p}: {e}')
            continue
        enc = cfg.get('encoder_name', 'UNKNOWN')
        tgt = cfg.get('target', 'UNKNOWN')
        sha = 'NO_MODEL_FILE'
        if mdl_p.exists():
            h = hashlib.sha256()
            with open(mdl_p, 'rb') as f:
                for chunk in iter(lambda: f.read(1 << 20), b''):
                    h.update(chunk)
            sha = h.hexdigest()[:16]
        logger.info(f'PREDICTOR_AUDIT[{label}]: encoder={enc} target={tgt} sha256_16={sha} path={mdl_p}')
        expected = os.environ.get(env_var)
        if expected and expected != enc:
            raise RuntimeError(
                f'PREDICTOR_AUDIT[{label}] MISMATCH: expected encoder={expected} but got {enc} from {cfg_p}. '
                f'Set {env_var} to the correct value or re-deploy the model.'
            )


def build_app(args: Namespace) -> FastAPI:
    global app
    app.root_path = args.root_path
    return app


async def init_app(
        args: Namespace,
        instances_list: Optional[List[Instance]] = None,
) -> FastAPI:
    app = build_app(args)
    global instances, start_time, scheduling, chat, model_family, repetition_penalty, frequency_penalty, temperature, broadcasting_enabled, broadcast_model_list, enable_predictor_feedback
    chat = args.chat
    model_family = args.model_family
    repetition_penalty = args.repetition_penalty
    frequency_penalty = args.frequency_penalty
    temperature = args.temperature
    model_config_path = args.model_config_path
    broadcasting_enabled = bool(getattr(args, "broadcasting", False))
    broadcast_model_list = list(getattr(args, "selected_broadcasted_models", []) or [])
    enable_predictor_feedback = args.enable_predictor_feedback

    model_dict = json.load(open(model_config_path))
    # Support both v2 (instances embedded) and v1 (separate host_config)
    host_config = None
    if getattr(args, "host_config", None) and os.path.exists(args.host_config):
        host_config = json.load(open(args.host_config))
    if instances_list is not None:
        instances.extend(instances_list)
    else:
        for model, model_config in model_dict.items():
            backend_type = model_config["backend"]
            hf_model_name = model_config["hf_model_name"]

            # V2: instances with embedded host info
            if "instances" in model_config:
                instance_list = model_config["instances"]
            else:
                # V1: node_hosts + separate host_config
                node_hosts = model_config["node_hosts"]
                instance_list = []
                for host in node_hosts:
                    hostname = host.split("@")[-1] if "@" in host else host
                    hc = host_config[hostname] if host_config else {}
                    instance_list.append({
                        "host": host,
                        "hostname": hostname,
                        "ip_address": hc.get("ip_address", ""),
                        "backend_port": hc.get("backend_port", 8000),
                        "predictor_ports": hc.get("predictor_ports", [8300]),
                    })

            for idx, inst_info in enumerate(instance_list):
                hostname = inst_info["hostname"]
                ip_address = inst_info["ip_address"]
                predictor_ports = inst_info["predictor_ports"]
                backend_port = inst_info["backend_port"]
                # Avoid including the backend port in predictor ports
                predictor_ports = [p for p in predictor_ports if p != backend_port]
                instance_id = f"{model}_{idx}"
                if backend_type == "ollama":
                    from route_balance.global_scheduler.route_balance.route_balance_instance.ollama_instance import OllamaInstance
                    # Convert HF model name to Ollama tag format, not supposed to be used besides testing
                    ollama_model_name = to_ollama_tag(hf_model_name)
                    instance = OllamaInstance(
                        instance_id=instance_id,
                        hostname=hostname,
                        ip_address=ip_address,
                        predictor_ports=predictor_ports,
                        model_name=ollama_model_name,
                        backend_port=backend_port,
                        enable_predictor_feedback=args.enable_predictor_feedback,
                        feedback_sample_rate=args.feedback_sample_rate
                    )
                elif backend_type == "vllm":
                    from route_balance.global_scheduler.route_balance.route_balance_instance.vllm_instance import VllmInstance
                    instance = VllmInstance(
                        instance_id=instance_id,
                        hostname=hostname,
                        ip_address=ip_address,
                        predictor_ports=predictor_ports,
                        model_name=hf_model_name,
                        backend_port=backend_port,
                        enable_predictor_feedback=args.enable_predictor_feedback,
                        feedback_sample_rate=args.feedback_sample_rate
                    )
                else:
                    raise ValueError(f"Unsupported backend type: {backend_type}")
                instances.append(instance)

    start_time = time.time()
    scheduling = args.scheduling

    # Validate scheduling strategy at startup — fail loud instead of silent random fallback
    # (cf. feedback_data_integrity_rules.md rule #6)
    _VALID_STRATEGIES = {
        "random", "round_robin", "shortest_queue", "quality_greedy",
        "cost_greedy", "length_aware", "route_balance", "pipeline",
    }
    if scheduling not in _VALID_STRATEGIES:
        raise RuntimeError(
            f"Invalid --scheduling '{scheduling}'. "
            f"Valid: {sorted(_VALID_STRATEGIES)}"
        )

    # Load scheduler config (new: separate file) or from predictor config (legacy)
    scheduler_config_path = getattr(args, "scheduler_config", None)
    if scheduler_config_path and os.path.exists(scheduler_config_path):
        _load_scheduler_config(scheduler_config_path, instances)
    elif scheduling in ("route_balance", "length_aware") and getattr(args, "predictor_config", None):
        # Legacy: extract scheduler config from predictor config
        _load_learned_predictors(args.predictor_config, instances)

    # Vendor pricing lives in model_deployment (canonical source of truth).
    # If a `pricing: {input_per_1m, output_per_1m}` block is present, override
    # the legacy cost_per_token in instance_meta. Predictor config values stay
    # only as fallback for older deployments.
    _hf_to_pricing = {}
    for _mkey, _mcfg in model_dict.items():
        _p = _mcfg.get("pricing")
        if _p:
            _hf_to_pricing[_mcfg.get("hf_model_name", _mkey)] = _p
    if _hf_to_pricing:
        for _iid, _meta in instance_meta.items():
            _p = _hf_to_pricing.get(_meta.get("model_name", ""))
            if _p:
                _in = float(_p.get("input_per_1m", 0.0)) / 1e6
                _out = float(_p.get("output_per_1m", 0.0)) / 1e6
                _meta["cost_per_input_token"] = _in
                _meta["cost_per_output_token"] = _out
                _meta["cost_per_token"] = _out  # legacy single-rate field
                _meta["pricing"] = _p
        logger.info(
            f"Pricing injected from model_deployment for {len(_hf_to_pricing)} models"
        )

    # Audit deployed predictors: log encoder + SHA, optionally assert vs env vars
    try:
        _log_and_assert_deployed_predictors()
    except Exception as _e:
        logger.error(f"PREDICTOR_AUDIT failed: {_e}")
        raise

    # Initialize batch queue and scheduler loop for ROUTE_BALANCE strategy
    if scheduling == "route_balance":
        global batch_queue
        batch_cfg = slo_defaults.get("batch_config", {})
        # adaptive_sizing ON: paper §3 design (small batches under light
        # load, larger under saturation). Signal is now busy-instance
        # ratio (count-based, paper-aligned) — see BatchQueue.adapt_params.
        batch_queue = BatchQueue(
            max_batch_size=batch_cfg.get("max_batch_size", 16),
            batch_timeout_ms=batch_cfg.get("batch_timeout_ms", 100.0),
            adaptive_sizing=batch_cfg.get("adaptive_sizing", True),
        )
        # Load global batched-tpot XGBoost model. Single XGBoostLatencyPredictor
        # with per-tier boosters. Used by _batched_tpot_per_request for the
        # in-process batched tpot prediction path (ROUTE_BALANCE_INPROC_PREDICTOR=1).
        global _TPOT_BATCH_MODEL
        if os.environ.get("ROUTE_BALANCE_INPROC_PREDICTOR") == "1" and _TPOT_BATCH_MODEL is None:
            try:
                from route_balance.predictor.route_balance.estimators.xgboost_predictor import (
                    XGBoostLatencyPredictor,
                )
                tpot_dir = "/users/anon/RouteBalance/models/route_balance/latency/deploy_tpot"
                if os.path.exists(tpot_dir):
                    _TPOT_BATCH_MODEL = XGBoostLatencyPredictor.load(tpot_dir)
                    print(
                        f"[INPROC-XGB] tpot model loaded from {tpot_dir}, "
                        f"tiers: {list(_TPOT_BATCH_MODEL.models.keys())}", flush=True
                    )
                else:
                    print(f"[INPROC-XGB] tpot dir not found: {tpot_dir}", flush=True)
            except Exception as _e:
                import traceback
                print(f"[INPROC-XGB] FAILED to load tpot model: {_e}", flush=True)
                traceback.print_exc()

        # Three pipeline modes for the route_balance batch worker:
        #   ROUTE_BALANCE_PREPPER_WORKERS=N (N>=1): Option A — single causal worker
        #     (preserving Algorithm 1 strict cross-batch ordering) + N
        #     concurrent preppers (estimator + LPT-sort offloaded). With N=1
        #     produces identical scheduling to legacy mode but with the
        #     RoBERTa estimator off the asyncio loop. With N>=2, multiple
        #     batches' estimators overlap with the causal worker's per-r
        #     loop on prior batches — pure throughput gain, no scheduling
        #     change because the causal worker is still single-coroutine.
        #   ROUTE_BALANCE_BATCH_WORKERS=N: legacy multi-worker mode (deprecated; N>1
        #     breaks cross-batch causality, kept for ablation only).
        #   default (neither set): legacy single-worker.
        n_preppers = int(os.environ.get("ROUTE_BALANCE_PREPPER_WORKERS", "0"))
        n_workers = int(os.environ.get("ROUTE_BALANCE_BATCH_WORKERS", "1"))
        global prepped_queue
        if n_preppers >= 1:
            prepped_queue = asyncio.Queue()
            for _ in range(n_preppers):
                asyncio.create_task(_prepper_loop())
            asyncio.create_task(_causal_worker_loop())
            logger.info(
                f"Batch scheduler started (Option A: {n_preppers} preppers + "
                f"1 causal worker, max_batch={batch_queue.max_batch_size}, "
                f"timeout={batch_queue.batch_timeout_ms}ms)"
            )
        else:
            for _ in range(n_workers):
                asyncio.create_task(_batch_scheduler_loop())
            logger.info(
                f"Batch scheduler started (legacy: {n_workers} workers, "
                f"max_batch={batch_queue.max_batch_size}, "
                f"timeout={batch_queue.batch_timeout_ms}ms)"
            )

    # Log registered instances and routes
    logger.info(f"ROUTE_BALANCE Scheduler initialized with {len(instances)} instances:")
    for inst in instances:
        logger.info(f"  - {inst._instance_id} ({inst._model_name}) @ {inst._ip_address}")
    logger.info(f"Scheduling strategy: {scheduling}")
    logger.info("Registered routes:")
    for route in app.routes:
        if hasattr(route, 'path') and hasattr(route, 'methods'):
            logger.info(f"  {list(route.methods)[0] if route.methods else 'GET'} {route.path}")

    return app


async def run_server(args: Namespace,
                     instances_list: Optional[List[Instance]] = None,
                     **uvicorn_kwargs: Any) -> None:
    app = await init_app(args, instances_list)
    print(str(set_ulimit()) + " set limits file ")
    assert len(instances) > 0

    if args.debugging_logs:
        logger.setLevel(logging.DEBUG)

    try:
        shutdown_task = await serve_http(
            app,
            host=args.host,
            port=args.port,
            timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
            ssl_keyfile=args.ssl_keyfile,
            ssl_certfile=args.ssl_certfile,
            ssl_ca_certs=args.ssl_ca_certs,
            ssl_cert_reqs=args.ssl_cert_reqs,
            workers=args.workers,
            **uvicorn_kwargs,
        )

        await shutdown_task
    finally:
        logger.info("Server shutdown.")
        # Flush training data from all predictors across instances (only if feedback is enabled)
        if not args.enable_predictor_feedback:
            logger.info("Predictor feedback disabled, skipping flush on shutdown.")
            return

        try:
            flush_urls = []
            for inst in instances:
                flush_urls.extend(inst.get_predictor_flush_urls())

            if flush_urls:
                timeout = aiohttp.ClientTimeout(total=15)

                async def _post_flush(session, url):
                    try:
                        async with session.post(url, ssl=False) as resp:
                            await resp.text()
                            return resp.status
                    except Exception:
                        return None

                async with aiohttp.ClientSession(timeout=timeout) as session:
                    results = await asyncio.gather(*[_post_flush(session, u) for u in flush_urls], return_exceptions=True)
                success = sum(1 for r in results if isinstance(r, int) and 200 <= r < 300)
                failed = len(flush_urls) - success
                logger.info(f"Flushed predictors: {success} ok, {failed} failed")
        except Exception as e:
            logger.warning(f"Error flushing predictors on shutdown: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8200)
    parser.add_argument("--workers", type=int, default=1,
                        help="Must be 1 — module-level state (learned_predictors, scoring_weights) is not shared across workers")
    parser.add_argument("--ssl-keyfile", type=str, default=None)
    parser.add_argument("--ssl-certfile", type=str, default=None)
    parser.add_argument("--ssl-ca-certs",
                        type=str,
                        default=None,
                        help="The CA certificates file")
    parser.add_argument(
        "--ssl-cert-reqs",
        type=int,
        default=int(ssl.CERT_NONE),
        help="Whether client certificate is required (see stdlib ssl module's)"
    )
    parser.add_argument(
        "--root-path",
        type=str,
        default=None,
        help="FastAPI root_path when app is behind a path based routing proxy")
    parser.add_argument("--model_config_path", type=str,
                        default="route_balance/config/route_balance/model_deployment.json")
    parser.add_argument("--host_config", type=str, default=None,
                        help="Path to host config JSON (optional if model_deployment has embedded instances)")
    parser.add_argument("--scheduling", type=str, default="random",
                        help="Scheduling strategy: random, round_robin, shortest_queue, "
                             "quality_greedy, cost_greedy, length_aware, route_balance, "
                             "pipeline (= pluggable router+dispatch+filter, A0.1). "
                             "When --scheduling pipeline, scheduler_config.json must "
                             "contain a scheduling_pipeline block or /v1/config must "
                             "POST {router,dispatch,filter} before traffic lands.")
    parser.add_argument("--debugging_logs", action="store_true",
                        help="Enable debug level logging")
    parser.add_argument("--chat", action="store_true",
                        help="Whether the model is a chat model to decide if stop words are appended")
    parser.add_argument("--model-family", type=str, default="Qwen",
                        help="Model family, used for append the stop words for chat models")
    parser.add_argument("--repetition-penalty", type=float, default=1.0,
                        help="Repetition penalty (multiplicative, 1.0 = no penalty). "
                             "Applied to previously generated tokens regardless of frequency.")
    parser.add_argument("--frequency-penalty", type=float, default=1.2,
                        help="Frequency penalty (additive, proportional to token count, default=1.2). "
                             "Penalizes tokens based on how many times they appear in the output so far. "
                             "Prevents degenerate repetition and keeps output_tokens under max_tokens. "
                             "Tuned via sweep: best survival rate at 1.2 (see sweep_broadcasting.sh).")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0.0 = greedy deterministic, >0 adds randomness)")
    parser.add_argument("--enable-predictor-feedback", action="store_true",
                        help="Enable sending actual metrics back to predictor for training data collection")
    parser.add_argument("--feedback-sample-rate", type=float, default=1.0,
                        help="Sampling rate for predictor feedback (0.0 to 1.0). Only applies when --enable-predictor-feedback is set")
    parser.add_argument("--broadcasting", action="store_true",
                        help="Enable broadcasting to one instance per selected model for data collection")
    parser.add_argument("--selected-broadcasted-models", nargs='+', default=[],
                        help="List of model names/tags to broadcast to when broadcasting is enabled")
    parser.add_argument("--predictor-config", type=str, default=None,
                        help="Path to learned predictor config JSON (legacy: contains all config)")
    parser.add_argument("--scheduler-config", type=str, default=None,
                        help="Path to scheduler config JSON (scoring_weights, slo_defaults, model_estimator)")
    args = parser.parse_args()
    logger.info("Starting server with args: %s", str(args))

    asyncio.run(run_server(args))
