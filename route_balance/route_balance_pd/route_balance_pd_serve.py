"""
RouteBalance P-D Disaggregated Scheduler.

Isolated module — no imports from route_balance.global_scheduler.route_balance_pd.
Reuses predictors and estimators from route_balance.predictor.route_balance via imports.

Architecture:
  - Prefill pool: vLLM instances with kv_role=kv_producer
  - Decode pool: vLLM instances with kv_role=kv_consumer
  - Per request: score prefill candidates (TTFT), score decode candidates (TPOT),
    account for KV transfer time analytically
  - Proxy: send prefill (max_tokens=1) → wait → send decode (full generation)

Usage:
    python3 -m route_balance.route_balance_pd.route_balance_pd_serve \
        --pd-config route_balance/route_balance_pd/config/pd_config.json \
        --port 8200
"""
import argparse
import asyncio
import json
import logging
import time
from argparse import Namespace
from typing import Any, Dict, List, Optional

import aiohttp
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

app = FastAPI()

# --- Global state ---
prefill_instances: List[Dict] = []
decode_instances: List[Dict] = []
model_estimator = None
scoring_weights: Dict = {}
pd_config: Dict = {}
stats = {
    "total_requests": 0,
    "total_prefill_ms": 0.0,
    "total_decode_ms": 0.0,
    "total_transfer_est_ms": 0.0,
}

# Analytical KV transfer model
BANDWIDTH_TABLE = {
    "a30_tcp": 1.67e9,       # 13.4 Gbits/s measured on CloudLab A30
    "a100_nvlink": 300e9,    # NVLink theoretical
    "a100_pcie": 25e9,       # PCIe Gen4 x16
}

KV_BYTES_PER_TOKEN = {
    "Qwen/Qwen2.5-3B": 55296,
    "Qwen/Qwen2.5-7B": 114688,
    "Qwen/Qwen2.5-14B": 393216,
    "Qwen/Qwen2.5-72B": 327680,
}


def estimate_transfer_time_s(num_prompt_tokens: int, model_name: str) -> float:
    """Analytical KV transfer time: prompt_tokens × kv_per_token / bandwidth."""
    kv_per_tok = KV_BYTES_PER_TOKEN.get(model_name, 115000)
    bw = BANDWIDTH_TABLE.get(pd_config.get("bandwidth_key", "a30_tcp"), 1.67e9)
    return (num_prompt_tokens * kv_per_tok) / bw


async def fetch_instance_stats(url: str) -> Dict:
    """Fetch /instance_stats from a vLLM instance."""
    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=3)
        ) as session:
            async with session.get(f"{url}/instance_stats", ssl=False) as resp:
                if resp.status == 200:
                    return await resp.json()
    except Exception as e:
        logger.debug(f"Failed to fetch instance_stats from {url}: {e}")
    return {}


async def call_sidecar_predict(url: str, num_prompt_tokens: int,
                                num_output_tokens: int) -> Dict:
    """Call sidecar /predict_latency on an instance."""
    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=3)
        ) as session:
            async with session.post(f"{url}/predict_latency", json={
                "num_prompt_tokens": num_prompt_tokens,
                "num_predicted_output_tokens": num_output_tokens,
            }, ssl=False) as resp:
                if resp.status == 200:
                    return await resp.json()
    except Exception as e:
        logger.debug(f"Sidecar predict failed for {url}: {e}")
    return {}


async def score_prefill_candidates(
    num_prompt_tokens: int, expected_length: float,
    ttft_slo_ms: float,
) -> Optional[Dict]:
    """Score prefill instances by predicted TTFT + balance."""
    best = None
    best_score = float("inf")

    for inst in prefill_instances:
        inst_stats = await fetch_instance_stats(inst["url"])
        pending_prefill = inst_stats.get("pending_prefill_tokens", 0)
        num_waiting = inst_stats.get("num_waiting", 0)
        kv_util = inst_stats.get("kv_cache_utilization", 0)

        sidecar_url = inst.get("sidecar_url")
        if sidecar_url:
            pred = await call_sidecar_predict(sidecar_url, num_prompt_tokens, int(expected_length))
            ttft = pred.get("ttft", 0.1)
        else:
            # Fallback: simple estimate from queue
            ttft = 0.02 + pending_prefill * 0.0001 + num_waiting * 0.05

        # SLO filter
        if ttft * 1000 > ttft_slo_ms:
            continue

        w = scoring_weights
        score = (
            w.get("w_prefill_latency", 0.4) * ttft
            + w.get("w_balance", 0.2) * kv_util
        )

        if score < best_score:
            best_score = score
            best = {**inst, "ttft": ttft, "stats": inst_stats, "score": score}

    return best


async def score_decode_candidates(
    num_prompt_tokens: int, expected_length: float,
    tpot_slo_ms: float,
) -> Optional[Dict]:
    """Score decode instances by predicted TPOT + balance."""
    best = None
    best_score = float("inf")

    for inst in decode_instances:
        inst_stats = await fetch_instance_stats(inst["url"])
        num_running = inst_stats.get("num_running", 0)
        kv_util = inst_stats.get("kv_cache_utilization", 0)

        sidecar_url = inst.get("sidecar_url")
        if sidecar_url:
            pred = await call_sidecar_predict(sidecar_url, num_prompt_tokens, int(expected_length))
            tpot = pred.get("tpot", 0.02)
        else:
            tpot = 0.02 + num_running * 0.005

        if tpot * 1000 > tpot_slo_ms:
            continue

        w = scoring_weights
        score = (
            w.get("w_decode_latency", 0.4) * tpot * expected_length
            + w.get("w_balance", 0.2) * kv_util
        )

        if score < best_score:
            best_score = score
            best = {**inst, "tpot": tpot, "stats": inst_stats, "score": score}

    return best


# --- API Endpoints ---

@app.get("/health")
async def health():
    return JSONResponse({"status": "ok", "mode": "pd_disaggregated"})


@app.get("/v1/pd_stats")
async def pd_stats():
    return JSONResponse({
        **stats,
        "prefill_pool_size": len(prefill_instances),
        "decode_pool_size": len(decode_instances),
    })


@app.post("/v1/completions")
async def completion(request: Request):
    """P-D disaggregated completion endpoint.

    1. ModelEstimator: predict quality + expected length
    2. Score prefill candidates (TTFT) → pick best
    3. Score decode candidates (TPOT) → pick best
    4. Proxy: prefill (max_tokens=1) → decode (full generation)

    Response format matches what benchmark client (route_balance_end_point_func.py) expects:
      output_tokens, generated_text, ttft, itl, server_latency, model, instance_id, host
    """
    global stats
    request_json = await request.json()
    prompt = request_json.get("prompt", "")
    max_tokens = request_json.get("max_tokens", 256)
    # Use model from request only if it matches a deployed model;
    # benchmark client sends "route_balance"/"route_balance" which is not a real model name
    model_name_raw = request_json.get("model", "")
    known_models = {inst.get("model_name") for inst in prefill_instances + decode_instances}
    model_name = model_name_raw if model_name_raw in known_models else ""
    num_prompt_tokens = request_json.get("prompt_len", len(prompt.split()) * 2)

    rso = request_json.get("request_specific_objective", request_json.get("rso", {}))
    ttft_slo_ms = rso.get("ttft_slo_ms", pd_config.get("ttft_slo_ms", 5000))
    tpot_slo_ms = rso.get("tpot_slo_ms", pd_config.get("tpot_slo_ms", 200))

    t_start = time.monotonic()

    # 1. Model estimation (quality + expected length)
    expected_length = 128.0
    quality = 0.5
    if model_estimator is not None:
        try:
            estimates = model_estimator.estimate(prompt)
            for mname, est in estimates.items():
                expected_length = est.length_expected
                quality = est.score
                break
        except Exception as e:
            logger.warning(f"ModelEstimator failed: {e}")

    # 2. Score prefill candidates
    prefill_inst = await score_prefill_candidates(
        num_prompt_tokens, expected_length, ttft_slo_ms
    )
    if prefill_inst is None:
        prefill_inst = prefill_instances[0] if prefill_instances else None

    # 3. Score decode candidates
    decode_inst = await score_decode_candidates(
        num_prompt_tokens, expected_length, tpot_slo_ms
    )
    if decode_inst is None:
        decode_inst = decode_instances[0] if decode_instances else None

    if not prefill_inst or not decode_inst:
        return JSONResponse(
            {"success": False, "error": "No prefill/decode instances available"},
            status_code=503,
        )

    # 4. Estimate transfer time
    resolved_model = model_name or prefill_inst.get("model_name", "")
    transfer_est = estimate_transfer_time_s(num_prompt_tokens, resolved_model)

    # 5. Proxy: prefill → decode
    t_prefill_start = time.monotonic()
    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=60)
        ) as session:
            # Prefill phase (max_tokens=1 triggers KV creation + nixl transfer)
            prefill_payload = {
                "model": resolved_model,
                "prompt": prompt,
                "max_tokens": 1,
                "temperature": request_json.get("temperature", 0.0),
            }
            async with session.post(
                f"{prefill_inst['url']}/v1/completions",
                json=prefill_payload, ssl=False,
            ) as resp:
                prefill_result = await resp.json()
                prefill_ms = (time.monotonic() - t_prefill_start) * 1000

            # Decode phase (KV already transferred via NixlConnector)
            t_decode_start = time.monotonic()
            decode_payload = {
                "model": resolved_model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": request_json.get("temperature", 0.0),
            }
            async with session.post(
                f"{decode_inst['url']}/v1/completions",
                json=decode_payload, ssl=False,
            ) as resp:
                decode_result = await resp.json()
                decode_ms = (time.monotonic() - t_decode_start) * 1000

    except Exception as e:
        return JSONResponse(
            {"success": False, "error": f"P-D proxy failed: {e}"},
            status_code=500,
        )

    total_ms = (time.monotonic() - t_start) * 1000
    scheduling_ms = total_ms - prefill_ms - decode_ms

    # Update stats
    stats["total_requests"] += 1
    stats["total_prefill_ms"] += prefill_ms
    stats["total_decode_ms"] += decode_ms
    stats["total_transfer_est_ms"] += transfer_est * 1000

    # Extract metrics from vLLM OpenAI completions response
    output_tokens = 0
    generated_text = ""
    try:
        choices = decode_result.get("choices", [])
        if choices:
            generated_text = choices[0].get("text", "")
        usage = decode_result.get("usage", {})
        output_tokens = usage.get("completion_tokens", 0)
    except Exception as e:
        logger.warning(f"Failed to extract decode metrics: {e}")

    # Response — flat fields for benchmark client compatibility + pd_info for analysis
    response = {
        "success": True,
        "output_tokens": output_tokens,
        "generated_text": generated_text,
        "ttft": prefill_ms / 1000.0,
        "itl": [],
        "server_latency": total_ms / 1000.0,
        "model": resolved_model,
        "instance_id": f"{prefill_inst.get('id', '')}+{decode_inst.get('id', '')}",
        "host": prefill_inst.get("url", ""),
        "predicted_quality": quality,
        "predicted_length": expected_length,
        "pd_info": {
            "prefill_instance": prefill_inst.get("id", ""),
            "decode_instance": decode_inst.get("id", ""),
            "prefill_ms": round(prefill_ms, 2),
            "decode_ms": round(decode_ms, 2),
            "transfer_est_ms": round(transfer_est * 1000, 2),
            "total_ms": round(total_ms, 2),
            "scheduling_ms": round(scheduling_ms, 2),
        },
        "decode_result": decode_result,
    }
    return JSONResponse(response)


# --- Initialization ---

def load_pd_config(config_path: str):
    """Load P-D deployment config."""
    global prefill_instances, decode_instances, pd_config

    with open(config_path) as f:
        cfg = json.load(f)

    pd_config.update(cfg.get("pd_settings", {}))

    for inst in cfg.get("prefill_instances", []):
        prefill_instances.append({
            "id": inst.get("id", ""),
            "url": f"http://{inst['ip_address']}:{inst['port']}",
            "model_name": inst.get("model_name", ""),
            "sidecar_url": f"http://{inst['ip_address']}:{inst.get('sidecar_port', 8300)}"
                if inst.get("sidecar_port") else None,
        })

    for inst in cfg.get("decode_instances", []):
        decode_instances.append({
            "id": inst.get("id", ""),
            "url": f"http://{inst['ip_address']}:{inst['port']}",
            "model_name": inst.get("model_name", ""),
            "sidecar_url": f"http://{inst['ip_address']}:{inst.get('sidecar_port', 8300)}"
                if inst.get("sidecar_port") else None,
        })

    logger.info(f"P-D config loaded: {len(prefill_instances)} prefill, {len(decode_instances)} decode")


def load_scheduler_config(config_path: str):
    """Load scheduler config (ModelEstimator + scoring weights)."""
    global model_estimator, scoring_weights

    with open(config_path) as f:
        cfg = json.load(f)

    scoring_weights.update(cfg.get("scoring_weights", {
        "w_prefill_latency": 0.4,
        "w_decode_latency": 0.3,
        "w_quality": 0.2,
        "w_balance": 0.1,
    }))

    me_config = cfg.get("model_estimator")
    if me_config:
        try:
            me_type = me_config.get("type", "default")
            if me_type == "knn":
                from route_balance.predictor.route_balance.model_estimator import KNNModelEstimator
                model_estimator = KNNModelEstimator(me_config)
            elif me_type == "pfs":
                from route_balance.predictor.route_balance.model_estimator import PFSModelEstimator
                model_estimator = PFSModelEstimator(me_config)
            else:
                from route_balance.predictor.route_balance.model_estimator import DefaultModelEstimator
                model_estimator = DefaultModelEstimator(me_config)
            logger.info(f"ModelEstimator loaded: type={me_type}")
        except Exception as e:
            logger.warning(f"Failed to load ModelEstimator: {e}")


async def init_app(args: Namespace) -> FastAPI:
    """Initialize P-D scheduler."""
    load_pd_config(args.pd_config)
    if args.scheduler_config:
        load_scheduler_config(args.scheduler_config)
    return app


async def run_server(args: Namespace):
    """Run P-D scheduler server."""
    from route_balance.server_utils_lite import serve_http
    await init_app(args)
    logger.info(
        f"RouteBalance P-D Scheduler: {len(prefill_instances)}P + {len(decode_instances)}D, "
        f"port={args.port}, bandwidth={pd_config.get('bandwidth_key', 'a30_tcp')}"
    )
    await serve_http(app, host=args.host, port=args.port)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="RouteBalance P-D Disaggregated Scheduler")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8200)
    parser.add_argument("--pd-config", required=True,
                        help="P-D deployment config (prefill/decode instances)")
    parser.add_argument("--scheduler-config", default=None,
                        help="Scheduler config (ModelEstimator, scoring weights)")
    parser.add_argument("--root-path", default="")
    args = parser.parse_args()

    asyncio.run(run_server(args))
