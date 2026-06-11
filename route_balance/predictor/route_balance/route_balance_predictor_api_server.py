"""
API server for ROUTE_BALANCE predictors.

Separate from Block's predictor API server to avoid coupling with
Block-specific configs and Vidur request transformations.
"""
import argparse
import asyncio
import logging
import ssl
import time
from argparse import Namespace
from typing import Any, Optional
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from route_balance.predictor.route_balance.route_balance_predictor_config import RouteBalanceBasePredictorConfig
from route_balance.predictor.route_balance.data_structures import PredictRequest
try:
    from route_balance.server_utils import serve_http
except ImportError:
    from route_balance.server_utils_lite import serve_http
try:
    from route_balance.global_scheduler.route_balance.utils import set_ulimit
except ImportError:
    def set_ulimit():
        """Fallback: try to raise file descriptor limit."""
        import resource
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (65536, 65536))
        except (ValueError, resource.error):
            pass

TIMEOUT_KEEP_ALIVE = 5  # seconds
app = FastAPI()
predictor: Optional[Any] = None  # ROUTE_BALANCE predictor instance
start_time = 0

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@app.post("/predict")
async def predict(request: Request) -> Response:
    """Predict metrics for a target request.

    Request JSON format:
    {
        "request_id": str,
        "num_prompt_tokens": int,
        "num_predicted_output_tokens": int
    }

    Response JSON format:
    {
        "target_metric": float,
        "gpu_blocks": int,
        "num_requests": int,
        "num_preempted": int,
        "predictor_type": str,
        "time_to_predict": float  # milliseconds
    }
    """
    if predictor is None:
        return JSONResponse({"error": "Predictor not initialized"}, status_code=503)
    pred_start_time = time.time()
    request_dict = await request.json()

    # Create PredictRequest
    target_request = PredictRequest(
        request_id=str(request_dict["request_id"]),
        num_prompt_tokens=int(request_dict["num_prompt_tokens"]),
        num_predicted_output_tokens=int(request_dict["num_predicted_output_tokens"])
    )

    metric = await predictor.predict(target_request)
    time_elapsed = (time.time() - pred_start_time) * 1000

    logger.debug(
        f"Predicted for request {target_request.request_id}: "
        f"metric={metric['target_metric']:.2f}, time={time_elapsed:.2f}ms"
    )

    metric["time_to_predict"] = time_elapsed
    return JSONResponse(metric)


@app.post("/log_actual")
async def log_actual(request: Request) -> Response:
    """Log actual metrics for training data collection.

    Request JSON format:
    {
        "request_id": str,
        "e2e_latency": float,
        "ttft": float (optional),
        "tpot": float (optional)
    }
    """
    if predictor is None:
        return JSONResponse({"error": "Predictor not initialized"}, status_code=503)
    request_dict = await request.json()

    # Only ROUTE_BALANCE predictors with data collection have this method
    if hasattr(predictor, 'log_actual_result'):
        await predictor.log_actual_result(
            request_id=request_dict['request_id'],
            e2e_latency=request_dict['e2e_latency'],
            ttft=request_dict.get('ttft'),
            tpot=request_dict.get('tpot'),
            output_tokens=request_dict.get('output_tokens'),
        )
        logger.debug(f"Logged actual for request: {request_dict['request_id']}")
    else:
        logger.warning("Predictor does not support log_actual_result")

    return Response(status_code=200)


@app.post("/predict_latency")
async def predict_latency(request: Request) -> Response:
    """Predict TTFT and TPOT for a request using local instance state.

    Sidecar mode: the predictor fetches instance_stats locally (no network
    overhead) and runs XGBoost TTFT prediction. This matches the paper's
    architecture where latency prediction is co-located with the instance.

    Request JSON format:
    {
        "num_prompt_tokens": int,
        "num_predicted_output_tokens": int
    }

    Response JSON format:
    {
        "ttft": float (seconds),
        "tpot": float (seconds),
        "e2e_latency": float (seconds),
        "probe_latency_ms": float
    }
    """
    if predictor is None:
        return JSONResponse({"error": "Predictor not initialized"}, status_code=503)
    pred_start = time.time()
    data = await request.json()
    num_prompt_tokens = int(data.get("num_prompt_tokens", 0))
    num_predicted_output_tokens = int(data.get("num_predicted_output_tokens", 256))

    # Fetch local instance stats (sidecar has direct access)
    schedule_state = {}
    if hasattr(predictor, '_schedule_trace_client'):
        try:
            state = await predictor._schedule_trace_client.fetch_instance_stats()
            if state:
                schedule_state = state.to_dict() if hasattr(state, 'to_dict') else {}
        except Exception as e:
            logger.warning(f"Failed to fetch local instance_stats: {e}")

    # Predict TTFT/TPOT/E2E. If methods are coroutines, fire all 3 in parallel
    # via asyncio.gather (XGBoost3ModelSidecarPredictor uses opportunistic
    # batching internally — concurrent gather + concurrent /predict_latency
    # calls feed each model's batcher).
    has_ttft = hasattr(predictor, 'predict_ttft')
    has_tpot = hasattr(predictor, 'predict_tpot')
    has_e2e = hasattr(predictor, 'predict_e2e')
    is_async = (
        (has_ttft and asyncio.iscoroutinefunction(predictor.predict_ttft))
        or (has_tpot and asyncio.iscoroutinefunction(predictor.predict_tpot))
        or (has_e2e and asyncio.iscoroutinefunction(predictor.predict_e2e))
    )

    if is_async:
        async def _ttft_call():
            if not has_ttft: return 1.0
            r = predictor.predict_ttft(
                schedule_state, num_prompt_tokens, num_predicted_output_tokens
            )
            return await r if asyncio.iscoroutine(r) else r

        async def _tpot_call():
            if not has_tpot: return 0.05
            r = predictor.predict_tpot(
                schedule_state, num_prompt_tokens, num_predicted_output_tokens
            )
            return await r if asyncio.iscoroutine(r) else r

        async def _e2e_call():
            if not has_e2e: return None
            r = predictor.predict_e2e(
                schedule_state, num_prompt_tokens, num_predicted_output_tokens
            )
            return await r if asyncio.iscoroutine(r) else r

        ttft, tpot, e2e_or_none = await asyncio.gather(
            _ttft_call(), _tpot_call(), _e2e_call()
        )
        e2e = e2e_or_none if e2e_or_none is not None else (
            ttft + num_predicted_output_tokens * tpot
        )
    else:
        if has_ttft:
            ttft = predictor.predict_ttft(
                schedule_state, num_prompt_tokens, num_predicted_output_tokens
            )
        else:
            ttft = 1.0
            logger.warning("predict_ttft not available, using fallback 1.0s")

        if has_tpot:
            tpot = predictor.predict_tpot(
                schedule_state, num_prompt_tokens, num_predicted_output_tokens
            )
        else:
            tpot = 0.05
            logger.warning("predict_tpot not available, using fallback 0.05s")

        if has_e2e:
            e2e = predictor.predict_e2e(
                schedule_state, num_prompt_tokens, num_predicted_output_tokens
            )
        else:
            e2e = ttft + num_predicted_output_tokens * tpot
    elapsed_ms = (time.time() - pred_start) * 1000

    return JSONResponse({
        "ttft": ttft,
        "tpot": tpot,
        "e2e_latency": e2e,
        "probe_latency_ms": elapsed_ms,
    })


@app.get("/stats")
async def stats() -> Response:
    """Get collection statistics (for data collection predictors)."""
    if hasattr(predictor, 'data_collector') and predictor.data_collector:
        stats_dict = predictor.data_collector.get_stats()
        return JSONResponse(stats_dict)
    else:
        return JSONResponse({"error": "Data collection not enabled"}, status_code=400)


@app.post("/flush")
async def flush() -> Response:
    """Force flush buffered training data to disk (for testing)."""
    if hasattr(predictor, 'data_collector') and predictor.data_collector:
        await predictor.data_collector.flush()
        stats_dict = predictor.data_collector.get_stats()
        return JSONResponse({"status": "flushed", **stats_dict})
    else:
        return JSONResponse({"error": "Data collection not enabled"}, status_code=400)


def build_app(args: Namespace) -> FastAPI:
    global app
    app.root_path = args.root_path
    return app


async def init_app(
    args: Namespace,
    instance_predictor: Optional[Any] = None,
) -> FastAPI:
    """Initialize ROUTE_BALANCE predictor API server."""
    app = build_app(args)
    global predictor

    # Load ROUTE_BALANCE predictor config
    config = RouteBalanceBasePredictorConfig.from_json_file(args.config_path)
    logger.info(f"Loaded ROUTE_BALANCE predictor config: type={config.predictor_type}")

    # Create appropriate predictor
    if instance_predictor is not None:
        predictor = instance_predictor
    elif config.predictor_type == "dummy":
        from route_balance.predictor.route_balance.dummy_route_balance_predictor import DummyRouteBalancePredictor
        predictor = DummyRouteBalancePredictor(
            config=config,
            backend_port=args.backend_port,
            predictor_port=args.port,
            hostname=args.hostname
        )
        logger.info(
            f"Created DummyRouteBalancePredictor: hostname={args.hostname}, "
            f"backend_port={args.backend_port}, predictor_port={args.port}"
        )
    elif config.predictor_type == "learned":
        from route_balance.predictor.route_balance.route_balance_learned_predictor import RouteBalanceLearnedPredictor
        # Determine instance_type from hostname (e.g., "d8545-xxx" → look up in config)
        instance_type = getattr(args, "instance_type", "unknown")
        predictor = RouteBalanceLearnedPredictor(
            config=config,
            port=args.backend_port,
            hostname=args.hostname,
            instance_type=instance_type,
        )
        logger.info(
            f"Created RouteBalanceLearnedPredictor: hostname={args.hostname}, "
            f"instance_type={instance_type}, backend_port={args.backend_port}"
        )
    elif config.predictor_type == "lstm":
        from route_balance.predictor.route_balance.route_balance_learned_predictor import LSTMSidecarPredictor
        instance_type = getattr(args, "instance_type", "unknown")
        predictor = LSTMSidecarPredictor(
            config=config,
            port=args.backend_port,
            hostname=args.hostname,
            instance_type=instance_type,
        )
        logger.info(
            f"Created LSTMSidecarPredictor: hostname={args.hostname}, "
            f"instance_type={instance_type}"
        )
    elif config.predictor_type == "roofline":
        from route_balance.predictor.route_balance.route_balance_learned_predictor import RooflineSidecarPredictor
        instance_type = getattr(args, "instance_type", "unknown")
        predictor = RooflineSidecarPredictor(
            config=config,
            port=args.backend_port,
            hostname=args.hostname,
            instance_type=instance_type,
        )
        logger.info(
            f"Created RooflineSidecarPredictor: hostname={args.hostname}, "
            f"instance_type={instance_type}"
        )
    elif config.predictor_type == "xgboost_3model":
        from route_balance.predictor.route_balance.route_balance_learned_predictor import (
            XGBoost3ModelSidecarPredictor,
        )
        instance_type = getattr(args, "instance_type", "unknown")
        predictor = XGBoost3ModelSidecarPredictor(
            config=config,
            port=args.backend_port,
            hostname=args.hostname,
            instance_type=instance_type,
        )
        await predictor.start()
        logger.info(
            f"Created XGBoost3ModelSidecarPredictor: hostname={args.hostname}, "
            f"instance_type={instance_type}"
        )
    else:
        raise ValueError(f"Unknown predictor type: {config.predictor_type}")

    return app


async def run_server(
    args: Namespace,
    instance_predictor: Optional[Any] = None,
    **uvicorn_kwargs: Any
) -> None:
    """Run ROUTE_BALANCE predictor API server."""
    global start_time
    start_time = time.time()

    app = await init_app(args, instance_predictor)
    if predictor is None:
        raise RuntimeError("Failed to initialize predictor")

    logger.info(f"Starting ROUTE_BALANCE predictor server on {args.host}:{args.port}")
    logger.info(f"Monitoring backend at {args.hostname}:{args.backend_port}")

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
        # Cleanup on shutdown
        if hasattr(predictor, 'shutdown'):
            logger.info("Shutting down predictor...")
            await predictor.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ROUTE_BALANCE Predictor API Server"
    )
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to bind to")
    parser.add_argument("--port", type=int, default=8100,
                        help="Port this predictor listens on")
    parser.add_argument("--backend-port", type=int, default=8000,
                        help="Port of backend instance (vLLM/Ollama)")
    parser.add_argument("--hostname", type=str, default="localhost",
                        help="Hostname of the backend instance being monitored")
    parser.add_argument("--config-path", type=str,
                        default="route_balance/config/route_balance/predictor_config.json",
                        help="Path to ROUTE_BALANCE predictor config JSON")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of uvicorn workers")
    parser.add_argument("--ssl-keyfile", type=str, default=None)
    parser.add_argument("--ssl-certfile", type=str, default=None)
    parser.add_argument("--ssl-ca-certs", type=str, default=None,
                        help="The CA certificates file")
    parser.add_argument("--ssl-cert-reqs", type=int,
                        default=int(ssl.CERT_NONE),
                        help="Whether client certificate is required")
    parser.add_argument("--root-path", type=str, default=None,
                        help="FastAPI root_path when app is behind a proxy")
    parser.add_argument("--instance-type", type=str, default="unknown",
                        help="Instance type identifier (e.g., qwen2.5-3b_p100) for learned predictor")

    args = parser.parse_args()

    # Set file descriptor limits
    set_ulimit()

    logger.info(f"Starting ROUTE_BALANCE predictor server with args: {args}")
    asyncio.run(run_server(args))
