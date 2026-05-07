import json
import os
import re
import sys
import argparse
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List


# --- Helpers ---

def parse_host_file(filepath: str) -> Dict[str, List[str]]:
    node_pool = {}
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: Host file '{filepath}' not found.")
        sys.exit(1)

    for line in lines:
        line = line.strip()
        if not line: continue
        # Matches patterns like: anon@d8545-10s10301...
        match = re.search(r'@([a-zA-Z0-9]+)-', line)
        if match:
            node_type = match.group(1)
            if node_type not in node_pool:
                node_pool[node_type] = []
            node_pool[node_type].append(line)
    return node_pool





def run_ssh_cmd(host: str, commands: List[str], description: str) -> bool:
    """Run commands on remote host via SSH. Returns True on success."""
    print(f"[{description}] Connecting to {host}...")

    valid_commands = [c for c in commands if c and c.strip()]
    full_command = " && ".join(valid_commands)

    ssh_params = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        host,
        full_command
    ]

    try:
        result = subprocess.run(ssh_params, capture_output=True, text=True, timeout=900)

        if result.returncode != 0:
            print(f"  [FAILED] Exit Code {result.returncode}")
            print(f"  [STDERR] {result.stderr.strip()}")
            if result.stdout:
                print(f"  [STDOUT] {result.stdout.strip()}")
            return False
        else:
            print(f"  [SUCCESS] {description} executed.")
            if result.stdout and result.stdout.strip():
                print(f"  [OUTPUT] {result.stdout.strip()}")
            return True

    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] Command took too long (check connection or extend timeout).")
        return False
    except Exception as e:
        print(f"  [ERROR] Connection failed: {str(e)}")
        return False


def to_ollama_tag(hf_name: str) -> str:
    name = hf_name.lower()
    if "/" in name:
        name = name.split("/")[-1]
    name = name.replace("-", ":")
    return name


# --- Command Generators ---

def get_cleanup_commands(backend: str) -> List[str]:
    """Generate commands to kill existing processes before deployment"""
    if backend == "vllm":
        cmds = [
            "echo 'Cleaning up existing vLLM processes...'",
            # Anchor to python so we don't kill the current remote shell running 'sh -c "... pkill ..."'
            "pkill -f '^python.*vllm\\.entrypoints\\.openai\\.api_server' || echo 'No existing vLLM process found'",
            "sleep 2"
        ]
    elif backend == "ollama":
        cmds = [
            "echo 'Cleaning up existing Ollama processes...'",
            # Kill any process using port 11434 first
            "fuser -k 11434/tcp 2>/dev/null || echo 'Port 11434 not in use'",
            # Kill Go processes related to ollama
            # Anchor to 'go' to avoid matching the current 'sh -c' command string
            "pkill -9 -f '^go .*run .*\\. .*serve' || echo 'No go run process found'",
            # Kill any ollama-related processes
            # Anchor to 'ollama' binary name if present
            "pkill -9 -f '^ollama( |$)' || echo 'No ollama process found'",
            # Wait longer for port to be released
            "sleep 5",
            # Verify port is free
            "echo 'Verifying port 11434 is free...'",
            "! lsof -ti:11434 || (echo 'WARNING: Port 11434 still in use' && lsof -ti:11434 | xargs kill -9)",
            "sleep 2"
        ]
    else:
        cmds = []
    return cmds


def get_vllm_commands(model_path: str, hf_token: str, precision: str, vllm_params: Dict = None,
                      fresh_deploy: bool = False) -> List[str]:
    # Use dtype from vllm_params if present (for old vLLM v0 versions), otherwise derive from precision
    if vllm_params and "dtype" in vllm_params:
        dtype_flag = vllm_params["dtype"]
    else:
        dtype_flag = "float16" if precision == "fp16" else "auto"

    # Ensure token is not None to prevent Python crash
    token_str = hf_token if hf_token else ""

    # Check if we should use custom HF cache
    use_hf_cache = vllm_params.get("use_hf_cache", False) if vllm_params else False

    # Build vLLM command with parameters
    vllm_cmd_parts = [
        "nohup python3 -u -m vllm.entrypoints.openai.api_server",
        f"--model {model_path}",
        f"--dtype {dtype_flag}",
        "--tensor-parallel-size $GPU_COUNT",
        "--trust-remote-code"
    ]

    # Apply vLLM parameters: explicit config values override defaults
    gpu_mem_util = 0.9  # default
    max_model_len = 8192  # default — matches latency training data collection
    enforce_eager = False
    if vllm_params:
        gpu_mem_util = vllm_params.get("gpu-memory-utilization", gpu_mem_util)
        max_model_len = vllm_params.get("max-model-len", max_model_len)
        enforce_eager = vllm_params.get("enforce-eager", enforce_eager)
    vllm_cmd_parts.append(f"--gpu-memory-utilization {gpu_mem_util}")
    vllm_cmd_parts.append(f"--max-model-len {max_model_len}")
    if enforce_eager:
        vllm_cmd_parts.append("--enforce-eager")
    # APC policy: enabled by default (vLLM default — used by main-table
    # cells where every cell sees a freshly-deployed cluster, so APC is
    # fair across baselines). Disabled via --disable-prefix-caching for
    # ablation/sensitivity sweeps where consecutive cells reuse the same
    # cluster and prefix-cache hits from earlier cells would bias
    # cell-to-cell comparison.
    if vllm_params and vllm_params.get("disable-prefix-caching"):
        vllm_cmd_parts.append("--no-enable-prefix-caching")

    vllm_cmd_str = " ".join(vllm_cmd_parts)

    cmds = [
        "echo 'Step 1: Checking vllm directory...'",
        "if [ ! -d ~/vllm ]; then echo 'ERROR: vllm directory not found'; exit 1; fi",
        "echo 'Step 2: Changing to vllm directory...'",
        "cd ~/vllm",
        "echo 'Step 3: Setting environment variables...'",
        f"export HF_TOKEN={token_str}",
    ]

    # Only export HF_HOME if use_hf_cache is true
    if use_hf_cache:
        cmds.append("export HF_HOME=/mydata/hf_cache")
        cmds.append("echo 'Using custom HF cache at /mydata/hf_cache'")
    else:
        cmds.append("echo 'Using system default HF cache'")

    # Set attention backend as environment variable for old vLLM v0
    if vllm_params and "attention_backend" in vllm_params:
        backend_value = vllm_params["attention_backend"].upper()
        cmds.append(f"export VLLM_ATTENTION_BACKEND={backend_value}")
        cmds.append(f"echo 'Using attention backend: {backend_value}'")

    sleep_time = 600 if fresh_deploy else 10

    cmds.extend([
        "export LD_LIBRARY_PATH=$(python3 -c 'import nvidia, os, glob; print(\":\".join(glob.glob(os.path.join(nvidia.__path__[0], \"*\", \"lib\"))))' 2>/dev/null):$LD_LIBRARY_PATH",
        "echo 'Step 4: Detecting GPU count...'",
        "export GPU_COUNT=$(python3 -c 'import torch; print(torch.cuda.device_count())')",
        "echo \"GPU_COUNT=$GPU_COUNT\"",
        "echo 'Step 5: Launching vLLM server in background...'",
        f"sh -c 'cd ~/vllm && {vllm_cmd_str} > vllm_server.log 2>&1 < /dev/null &'",
        f"sleep {sleep_time}",
        # Health check loop: wait up to 180s for vLLM to respond
        "echo 'Waiting for vLLM health check...'",
        "for i in $(seq 1 36); do "
        "  curl -sf http://localhost:8000/health > /dev/null 2>&1 && echo 'vLLM ready!' && break; "
        "  echo \"  attempt $i/36...\"; sleep 5; "
        "done",
        "curl -sf http://localhost:8000/health > /dev/null 2>&1 || { echo 'ERROR: vLLM failed to start'; exit 1; }",
        "echo 'Deployment completed!'",
        "exit 0"
    ])
    return cmds


def get_predictor_deployment_commands(
    hostname: str,
    backend_port: int,
    predictor_config: Dict,
    host_config: Dict,
    predictor_type: str = "dummy",
    config_path: str = "route_balance/config/route_balance/predictor_deployment_config.json",
    instance_type: str = "unknown",
) -> List[str]:
    """Generate commands to deploy ROUTE_BALANCE predictors on a host.

    Args:
        hostname: The hostname (without user@)
        backend_port: Backend port from host_config
        predictor_config: Predictor deployment configuration
        host_config: Host configuration containing predictor_ports
        predictor_type: Type of predictor ('dummy' or 'learned')
        config_path: Path to predictor config file on remote host
        instance_type: Instance type identifier for learned predictor

    Returns:
        List of shell commands to deploy predictors
    """
    predictor_ports = host_config[hostname]["predictor_ports"]
    # Avoid port collision with backend by skipping backend_port if present
    predictor_ports = [p for p in predictor_ports if p != backend_port]

    # Get data collection settings from predictor config (currently handled by config file)
    _enable_data_collection = predictor_config.get("enable_data_collection", False)
    _data_collection_sample_rate = predictor_config.get("data_collection_sample_rate", 1.0)
    data_output_dir = predictor_config.get("data_output_dir", "./training_data/route_balance")
    _save_batch_size = predictor_config.get("save_batch_size", 100)

    # Build a single aggregated command string to avoid "& &&" join issues.
    header_parts = [
        "echo 'Deploying ROUTE_BALANCE Predictors...'",
        # Anchor to python to avoid killing the remote 'sh -c' that contains this string in its command line
        "pkill -f '^python.*route_balance\\.predictor\\.route_balance\\.route_balance_predictor_api_server' || echo 'No existing predictors'",
        "sleep 2",
        "mkdir -p RouteBalance/experiment_output/logs",
        f"mkdir -p RouteBalance/{data_output_dir}",
    ]
    header_cmd = " && ".join(header_parts)

    # Start all predictors in background within one grouped command after cd
    bg_cmds = []
    for predictor_port in predictor_ports:
        cmd_parts = [
            f"nohup $PYTHON_BIN -u -m route_balance.predictor.route_balance.route_balance_predictor_api_server",
            f"--host 0.0.0.0",
            f"--port {predictor_port}",
            f"--backend-port {backend_port}",
            f"--hostname {hostname}",
            f"--config-path {config_path}",
        ]
        # Add instance-type for learned/lstm/roofline/xgboost_3model predictors
        if predictor_type in ("learned", "lstm", "roofline", "xgboost_3model"):
            cmd_parts.append(f"--instance-type {instance_type}")
        cmd_parts.append(
            f"> experiment_output/logs/predictor_{predictor_port}.log 2>&1 < /dev/null &"
        )
        bg_cmds.append(" ".join(cmd_parts))

    # Group background launches so the outer command does not end with '&'.
    # Detect usable python binary inside the group.
    # CUDA_VISIBLE_DEVICES="" forces XGBoost to CPU. With CUDA visible, xgboost
    # dispatches 1-row predicts to GPU; on nodes where vLLM saturates the GPU
    # (e.g. 72B TP=4 on A100, 14B TP=4 on V100) this gates predict at ~800ms
    # waiting for an idle SM. CPU predict is sub-ms and deterministic.
    # OMP/MKL=1: XGBoost predict is single-threaded fast; default per-CPU
    # forking adds latency on high-core nodes (96-core EPYC etc.).
    group_cmd = (
        "cd RouteBalance && ( "
        "export PYTHONUNBUFFERED=1; "
        "export PYTHONPATH=$HOME/RouteBalance:$PYTHONPATH; "
        "export CUDA_VISIBLE_DEVICES=; "
        "export OMP_NUM_THREADS=1; "
        "export MKL_NUM_THREADS=1; "
        "export OPENBLAS_NUM_THREADS=1; "
        "PYTHON_BIN=${PREDICTOR_PYTHON_BIN:-$(command -v python3 || command -v python)}; "
        + " ".join(bg_cmds) +
        " true )"
    )

    # Verify processes are up; fail if any did not start. Print concise summary only.
    ports_list = " ".join(str(p) for p in predictor_ports)
    verify_cmd = (
        "sleep 5 && "
        "fail=0; failed_ports=''; "
        f"for p in {ports_list}; do "
        "if pgrep -f \"route_balance.predictor.route_balance.route_balance_predictor_api_server.*--port $p\" >/dev/null; then :; "
        "else echo \"Predictor failed on port $p\" 1>&2; failed_ports=\"$failed_ports $p\"; fail=1; fi; done; "
        "if [ $fail -ne 0 ]; then echo \"Failed predictor ports:$failed_ports\" 1>&2; exit 1; fi"
    )

    # Final echo (concise)
    tail_cmd = f"echo 'Predictors OK: {len(predictor_ports)}/{len(predictor_ports)}'"

    combined = " && ".join([header_cmd, group_cmd, verify_cmd, tail_cmd])
    return [combined]


def get_monitor_deployment_commands(
    node_id: str,
    backend_port: int,
    interval: int = 5,
    output_dir: str = "experiment_output/monitor",
    is_scheduler_node: bool = False,
) -> List[str]:
    """Generate commands to deploy the per-node resource monitor.

    The monitor collects GPU (via NVML), vLLM (/metrics), CPU/memory, and
    optionally scheduler/predictor process stats. Runs in background.

    Args:
        node_id: Unique node identifier for the CSV filename.
        backend_port: Local vLLM port for /metrics endpoint.
        interval: Polling interval in seconds.
        output_dir: Remote directory for CSV output.
        is_scheduler_node: If True, also track scheduler + predictor PIDs.
    """
    header_parts = [
        "echo 'Deploying RouteBalance Monitor...'",
        # Kill existing monitor by PID file (pkill -f would kill the SSH session itself)
        "test -f /tmp/monitor_pid && kill $(cat /tmp/monitor_pid) 2>/dev/null ; true",
        "sleep 1",
        "mkdir -p ~/RouteBalance/experiment_output/monitor ~/RouteBalance/experiment_output/logs",
        "cd ~/RouteBalance",
    ]

    monitor_args = [
        f"--node-id {node_id}",
        f"--vllm-port {backend_port}",
        f"--interval {interval}",
        f"--output-dir {output_dir}",
    ]

    if is_scheduler_node:
        monitor_args.append("--scheduler-pid $(pgrep -f 'route_balance_serve.*8200' | head -1)")
        monitor_args.append("--predictor-pid $(pgrep -f 'route_balance_predictor_api_server' | head -1)")

    monitor_args_str = " ".join(monitor_args)

    # Write launcher script to avoid & and quoting issues in SSH command chains
    script_path = f"/tmp/start_monitor_{node_id}.sh"

    # Escape $ for SSH passthrough (run_ssh_cmd passes as double-quoted string)
    cmds = list(header_parts) + [
        f"echo '#!/bin/bash' > {script_path}",
        f"echo 'cd ~/RouteBalance' >> {script_path}",
        f"echo 'export PYTHONPATH=~/RouteBalance:\\$PYTHONPATH' >> {script_path}",
        f"echo 'nohup python3 -u route_balance/exp/route_balance/monitor.py {monitor_args_str}"
        f" > experiment_output/logs/monitor_{node_id}.log 2>&1 &' >> {script_path}",
        f'echo "echo \\$! > /tmp/monitor_pid" >> {script_path}',
        f"bash {script_path}",
        "sleep 3",
        # Verify by checking PID file (pgrep -f would kill SSH session)
        (
            f"test -f /tmp/monitor_pid && kill -0 $(cat /tmp/monitor_pid) 2>/dev/null"
            f" && echo 'Monitor OK: {node_id}'"
            f" || echo 'Monitor FAILED: {node_id}'"
        ),
    ]

    return cmds


def get_scheduler_deployment_commands(
    scheduler_host: str,
    port: int = 8200,
    model_config_path: str = "route_balance/config/route_balance/model_deployment.json",
    scheduling: str = "route_balance",
    scheduler_config: str = "",
    predictor_config: str = "",
    extra_args: str = "",
) -> List[str]:
    """Generate commands to deploy the RouteBalance scheduler (route_balance_serve.py).

    Runs on the designated coordinator node.
    """
    header_parts = [
        "echo 'Deploying RouteBalance Scheduler...'",
        "(pkill -f '^python.*route_balance_serve' || true)",
        "sleep 2",
        "cd ~/RouteBalance",
        "mkdir -p experiment_output/logs",
    ]

    sched_cmd_parts = [
        "nohup python3 -u -m route_balance.global_scheduler.route_balance.route_balance_serve",
        f"--host 0.0.0.0 --port {port}",
        f"--model_config_path {model_config_path}",
        f"--scheduling {scheduling}",
        "--chat",
    ]
    if scheduler_config:
        sched_cmd_parts.append(f"--scheduler-config {scheduler_config}")
    if predictor_config:
        sched_cmd_parts.append(f"--predictor-config {predictor_config}")
    sched_cmd_parts.append("--host_config route_balance/config/host_configs.json")
    if extra_args:
        sched_cmd_parts.append(extra_args)

    sched_cmd = " ".join(sched_cmd_parts)

    # Use script file to avoid & quoting issues in SSH
    script_path = "/tmp/start_route_balance_scheduler.sh"
    cmds = list(header_parts) + [
        f"echo '#!/bin/bash' > {script_path}",
        f"echo 'cd ~/RouteBalance' >> {script_path}",
        f"echo 'export CUDA_VISIBLE_DEVICES=' >> {script_path}",
        f"echo 'export PYTHONPATH=~/RouteBalance:~/vllm:\\$PYTHONPATH' >> {script_path}",
        f"echo '{sched_cmd} > experiment_output/logs/route_balance_scheduler.log 2>&1 &' >> {script_path}",
        f'echo "echo \\$! > /tmp/scheduler_pid" >> {script_path}',
        f"bash {script_path}",
        "sleep 8",
        # Health check: wait for scheduler to respond
        f"for i in $(seq 1 12); do "
        f"  curl -sf http://localhost:{port}/v1/batch_stats > /dev/null 2>&1 && echo 'Scheduler ready!' && break; "
        f"  echo \"  scheduler attempt $i/12...\"; sleep 5; "
        f"done",
        f"curl -sf http://localhost:{port}/v1/batch_stats > /dev/null 2>&1"
        f" || {{ echo 'ERROR: Scheduler failed to start'; exit 1; }}",
    ]

    return cmds


def get_ollama_commands(hf_name: str, num_parallel: int = 4, fresh_deploy: bool = False) -> List[str]:
    ollama_tag = to_ollama_tag(hf_name)

    sleeping_time = 600 if fresh_deploy else 10

    cmds = [
        "if [ ! -d ~/ollama ]; then echo 'ERROR: ollama directory not found'; exit 1; fi",
        "cd ~/ollama",
        "echo 'Setting environment variables...'",
        # Export PATH to include Go and CUDA (same as in setup.sh)
        "export PATH=$PATH:/usr/local/go/bin:/usr/local/cuda-12.8/bin:/usr/local/cuda/bin",
        "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12.8/lib64",
        # CRITICAL: Set OLLAMA_HOST to listen on all interfaces, not just localhost
        "export OLLAMA_HOST=0.0.0.0:11434",
        # Enable parallel request processing to maximize GPU utilization
        f"export OLLAMA_NUM_PARALLEL={num_parallel}",
        "export OLLAMA_MAX_LOADED_MODELS=1",
        f"echo 'Ollama server starting with {num_parallel} parallel requests (listening on 0.0.0.0:11434)...'",
        # Start server with go run wrapped in sh -c - environment variables will be inherited
        "sh -c 'cd ~/ollama && go run . serve > ollama_server.log 2>&1 < /dev/null &'",
        f"sleep {sleeping_time}",
        f"echo 'Pulling {ollama_tag} via REST API (synchronous)...'",
        # Pull synchronously to wait for completion before warmup
        f'curl -s http://localhost:11434/api/pull -d \'{{\"model\": \"{ollama_tag}\"}}\' > ollama_pull.log 2>&1',
        f"echo 'Model pulled successfully. Warming up {ollama_tag} to load into GPU...'",
        # Warmup inference to load model into GPU memory
        f'curl -s http://localhost:11434/api/generate -d \'{{\"model\": \"{ollama_tag}\", \"prompt\": \"Hello\", \"stream\": false}}\' > ollama_warmup.log 2>&1',
        "echo 'Warmup completed. Model loaded into GPU memory.'",
        "echo 'Ollama deployment completed!'",
        "exit 0"
    ]
    return cmds


# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="Model Deployment Script v4")

    parser.add_argument("--hosts", help="Host file path",
                        default="route_balance/config/hosts")
    parser.add_argument("--config", help="Model config JSON",
                        default="route_balance/config/route_balance/model_config_template.json")

    # NOTE: Hardcoded for local testing. Remove before committing to public repo.
    parser.add_argument("--hf-token", help="Hugging Face Token",
                        default="")

    parser.add_argument("--output", default="route_balance/config/route_balance/model_deployment.json",
                        help="Output config path")

    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated list of models to deploy (e.g., 'Qwen-2.5-3B' or 'Qwen-2.5-3B,Qwen-2.5-7B'). Deploy all if not specified.")

    parser.add_argument("--ollama-num-parallel", type=int, default=4,
                        help="Number of parallel requests Ollama can handle (default: 4). Higher values increase GPU utilization.")

    parser.add_argument("--fresh-deploy", action="store_true",
                        help="If set, it usually need much longer time to download models from HF and golang packages for Ollama.")
    parser.add_argument("--disable-prefix-caching", action="store_true", default=False,
                        help="Disable vLLM automatic prefix caching cluster-wide. Default OFF "
                             "(APC enabled, vLLM default). Set this for ablation/sensitivity "
                             "sweeps where consecutive cells must not benefit from prefix-cache "
                             "hits accumulated by earlier cells. For main-table comparisons, "
                             "leave APC enabled and ensure each cell gets a freshly-deployed "
                             "cluster so APC is fair across baselines.")

    parser.add_argument("--deploy-predictors", action="store_true", default=True,
                        help="Deploy ROUTE_BALANCE predictors alongside backend instances")
    parser.add_argument("--no-deploy-predictors", action="store_false", dest="deploy_predictors",
                        help="Skip predictor deployment")
    parser.add_argument("--predictor-config", type=str,
                        default="route_balance/config/route_balance/predictor_deployment_config.json",
                        help="Path to predictor deployment config")
    parser.add_argument("--host-config", type=str,
                        default="route_balance/config/host_configs.json",
                        help="Path to host config file (contains backend_port and predictor_ports)")
    parser.add_argument(
        "-d", "--deploy-services",
        nargs="+",
        default=["model_instance", "predictor", "monitor"],
        help=(
            "Services to deploy: model_instance predictor monitor scheduler. "
            "Default: model_instance predictor monitor"
        )
    )
    parser.add_argument("--scheduler-host", type=str, default=None,
                        help="Host to deploy scheduler on (user@hostname). "
                        "Defaults to the first host in the hosts file.")
    parser.add_argument("--scheduler-port", type=int, default=8200,
                        help="Scheduler port (default: 8200)")
    parser.add_argument("--scheduling", type=str, default="route_balance",
                        help="Scheduling strategy (default: route_balance)")
    parser.add_argument("--scheduler-config", type=str, default=None,
                        help="Path to scheduler_config.json on remote host")
    parser.add_argument("--scheduler-extra-args", type=str, default="",
                        help="Extra args to pass to route_balance_serve.py")
    parser.add_argument("--monitor-interval", type=int, default=5,
                        help="Monitor polling interval in seconds (default: 5)")
    parser.add_argument("--monitor-output-dir", type=str,
                        default="experiment_output/monitor",
                        help="Remote directory for per-node monitor CSV files")
    parser.add_argument("--predictor-type", type=str, default="dummy",
                        choices=["dummy", "learned"],
                        help="Type of predictor to deploy (dummy or learned)")
    parser.add_argument("--learned-predictor-config", type=str,
                        default="route_balance/config/route_balance/predictor_config_learned.json",
                        help="Path to learned predictor config (used when --predictor-type=learned)")
    parser.add_argument("--model-checkpoint-dir", type=str,
                        default="/mydata/models/route_balance",
                        help="Base directory containing trained model checkpoints")
    parser.add_argument("--parallel-deploy", type=int, default=1,
                        help="Max concurrent host-deployments (default: 1 = sequential). "
                             "Set to N>1 to run up to N hosts in parallel via ThreadPoolExecutor. "
                             "Each thread runs cleanup→vllm→predictor→monitor for one host serially. "
                             "Different hosts run independently. 18 = full cluster parallel.")

    args = parser.parse_args()

    # 1. Load Data
    available_nodes = parse_host_file(args.hosts)
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Error: Config file '{args.config}' not found.")
        sys.exit(1)

    # Determine which services to deploy (list of strings)
    deploy_services = [s.lower() for s in (args.deploy_services or [])]
    deploy_model_instances = "model_instance" in deploy_services
    deploy_predictors_flag = ("predictor" in deploy_services) and args.deploy_predictors
    deploy_monitor_flag = "monitor" in deploy_services
    deploy_scheduler_flag = "scheduler" in deploy_services

    # Load host config and predictor config if deploying predictors
    host_config = None
    predictor_config = None
    if deploy_predictors_flag:
        try:
            with open(args.host_config, 'r') as f:
                host_config = json.load(f)
            print(f"--- Loaded host config from {args.host_config} ---")
        except FileNotFoundError:
            print(f"Error: Host config file '{args.host_config}' not found.")
            sys.exit(1)

        try:
            with open(args.predictor_config, 'r') as f:
                predictor_config = json.load(f)
            print(f"--- Loaded predictor config from {args.predictor_config} ---")
        except FileNotFoundError:
            print(f"Error: Predictor config file '{args.predictor_config}' not found.")
            sys.exit(1)

    # Load existing deployment config if doing selective deployment
    final_config = {}
    if args.models and os.path.exists(args.output):
        try:
            with open(args.output, 'r') as f:
                final_config = json.load(f)
            print(f"--- Loaded existing deployment config from {args.output} ---")
        except:
            print(f"--- Could not load existing config, starting fresh ---")

    print(f"--- Loaded {sum(len(v) for v in available_nodes.values())} nodes ---")

    # Filter models if --models specified
    models_to_deploy = None
    if args.models:
        models_to_deploy = [m.strip() for m in args.models.split(',')]
        print(f"--- Deploying only specified models: {models_to_deploy} ---")
        print(f"--- Existing models will be preserved in output config ---")
    else:
        print(f"--- Deploying all models ---")

    # 2. Allocation & Deployment
    # Phase 2a (serial): allocation pass — mutates available_nodes, builds
    #   final_config and a flat task list. Allocation is sub-second per model.
    # Phase 2b (parallel): runs all (model, host) deploy tasks in ONE pool —
    #   so 14B+3B+7B+72B all overlap. Cuts total wall to slowest single model
    #   instead of sum-of-models-sequential.
    deployment_results = []  # [(model_key, host, service, success)]
    deployment_tasks = []    # [(model_key, host, deploy_fn)] — flat across all models
    for model_key, details in config.items():
        # Skip if not in the list of models to deploy
        if models_to_deploy and model_key not in models_to_deploy:
            print(f"\nSkipping: {model_key} (not in deployment list)")
            # Still add to final config if it exists in output file
            continue
        print(f"\nProcessing: {model_key}...")

        # --- Allocation Logic ---
        new_entry = details.copy()
        assigned_hosts = []
        vllm_params = {}  # Store vLLM-specific parameters
        if getattr(args, "disable_prefix_caching", False):
            vllm_params["disable-prefix-caching"] = True

        if 'node_type' in details:
            for ntype, node_config in details['node_type'].items():
                # Support both old format (just count) and new format (dict with count + params)
                if isinstance(node_config, int):
                    # Old format: "d8545": 2
                    count = node_config
                else:
                    # New format: "d8545": {"count": 2, "gpu-memory-utilization": 0.95, ...}
                    count = node_config.get('count', 1)
                    # Extract vLLM parameters (everything except 'count')
                    vllm_params = {k: v for k, v in node_config.items() if k != 'count'}

                if ntype not in available_nodes or len(available_nodes[ntype]) < count:
                    print(f"CRITICAL ERROR: Not enough nodes of type {ntype} for {model_key}")
                    # In a real script you might want to continue, but exiting is safer here
                    sys.exit(1)

                nodes = available_nodes[ntype][:count]
                # Remove used nodes from pool
                available_nodes[ntype] = available_nodes[ntype][count:]
                assigned_hosts.extend(nodes)

            del new_entry['node_type']
            new_entry['node_hosts'] = assigned_hosts
        else:
            assigned_hosts = details.get('node_hosts', [])
            # Support top-level vllm_params for node_hosts format
            if 'vllm_params' in details:
                vllm_params = details['vllm_params']

        # Embed host_config info into instances list (v2 format)
        # so route_balance_serve.py only needs model_deployment.json
        if host_config:
            embedded_instances = []
            for host in assigned_hosts:
                hostname = host.split("@")[-1] if "@" in host else host
                hc = host_config.get(hostname, {})
                embedded_instances.append({
                    "host": host,
                    "hostname": hostname,
                    "ip_address": hc.get("ip_address", ""),
                    "backend_port": hc.get("backend_port", 8000),
                    "predictor_ports": hc.get("predictor_ports", [8300]),
                })
            new_entry['instances'] = embedded_instances

        final_config[model_key] = new_entry

        # --- SSH Deployment Logic ---
        backend = details.get('backend', 'vllm')
        hf_name = details.get('hf_model_name')
        precision = details.get('precision', 'fp16')

        # Per-host deploy worker — runs cleanup → vllm → predictor → monitor for ONE host.
        # Returns list of (model, host, service, ok) tuples. Safe to call concurrently
        # across different hosts (no shared mutable state inside).
        # Default-arg trick captures per-iter state so the closure doesn't bind to
        # the LAST loop iteration's model_key/hf_name when run later in a pool.
        def deploy_one_host(host, model_key=model_key, hf_name=hf_name,
                            vllm_params=vllm_params, precision=precision,
                            backend=backend):
            local_results = []
            # Optionally cleanup and deploy model instances
            if deploy_model_instances:
                cleanup_cmds = get_cleanup_commands(backend)
                if cleanup_cmds:
                    run_ssh_cmd(host, cleanup_cmds, f"Cleanup ({model_key})")

                if backend == "vllm":
                    if vllm_params:
                        print(f"  vLLM params: {vllm_params}")
                    cmds = get_vllm_commands(hf_name, args.hf_token, precision, vllm_params,
                                             fresh_deploy=args.fresh_deploy)
                    ok = run_ssh_cmd(host, cmds, f"Deploying vLLM ({model_key})")
                    local_results.append((model_key, host, "vllm", ok))
                elif backend == "ollama":
                    print(f"  Ollama parallel requests: {args.ollama_num_parallel}")
                    cmds = get_ollama_commands(hf_name, num_parallel=args.ollama_num_parallel,
                                               fresh_deploy=args.fresh_deploy)
                    run_ssh_cmd(host, cmds, f"Deploying Ollama ({model_key})")

            # Deploy predictors if requested
            if deploy_predictors_flag:
                # Extract hostname from "user@hostname" format
                hostname = host.split("@")[-1] if "@" in host else host
                # Get backend_port from host_config (single source of truth!)
                backend_port = host_config[hostname]["backend_port"]

                # Determine predictor type and config path
                pred_type = args.predictor_type
                if pred_type == "learned":
                    pred_config_path = args.learned_predictor_config
                    # Infer instance_type from model key + GPU type (not node type)
                    # so it matches the per-(model, GPU) .xgb files in deploy_e2e/
                    model_short = hf_name.split("/")[-1].lower() if hf_name else "unknown"
                    match = re.search(r'@([a-zA-Z0-9]+)-', host)
                    node_type = match.group(1) if match else "unknown"
                    NODE_TO_GPU = {"d7525": "a30", "c240g5": "p100", "c4130": "v100", "d8545": "a100"}
                    gpu_type = NODE_TO_GPU.get(node_type, node_type)
                    inst_type = f"{model_short}_{gpu_type}"
                else:
                    pred_config_path = "route_balance/config/route_balance/predictor_deployment_config.json"
                    inst_type = "unknown"

                predictor_cmds = get_predictor_deployment_commands(
                    hostname=hostname,
                    backend_port=backend_port,
                    predictor_config=predictor_config,
                    host_config=host_config,
                    predictor_type=pred_type,
                    config_path=pred_config_path,
                    instance_type=inst_type,
                )
                ok = run_ssh_cmd(host, predictor_cmds, f"Deploying Predictors ({model_key})")
                local_results.append((model_key, host, "predictor", ok))

            # Deploy monitor if requested
            if deploy_monitor_flag:
                hostname = host.split("@")[-1] if "@" in host else host
                backend_port = 8000
                if host_config and hostname in host_config:
                    backend_port = host_config[hostname]["backend_port"]

                # Read monitor config from predictor config if available
                monitor_cfg = {}
                if predictor_config:
                    monitor_cfg = predictor_config.get("monitor", {})
                mon_interval = monitor_cfg.get("interval_sec", args.monitor_interval)
                mon_output = monitor_cfg.get("output_dir", args.monitor_output_dir)

                # Use model_key as node_id (e.g., "Qwen-2.5-7B")
                node_id = model_key.lower().replace(" ", "_").replace(".", "")
                # Add hostname suffix for uniqueness
                host_short = hostname.split(".")[0].split("-")[-1] if "." in hostname else hostname
                node_id = f"{node_id}_{host_short}"

                monitor_cmds = get_monitor_deployment_commands(
                    node_id=node_id,
                    backend_port=backend_port,
                    interval=mon_interval,
                    output_dir=mon_output,
                    is_scheduler_node=False,
                )
                ok = run_ssh_cmd(host, monitor_cmds, f"Deploying Monitor ({node_id})")
                local_results.append((model_key, host, "monitor", ok))

            return local_results

        # End of deploy_one_host. Defer execution: collect tasks across ALL
        # models, then run in one big pool below. This lets 14B+3B+7B+72B
        # overlap their per-host SSH deploys instead of running model-size
        # groups serially.
        for host in assigned_hosts:
            deployment_tasks.append((model_key, host, deploy_one_host))

    # Phase 2b — execute all (model, host) tasks in ONE pool across all models.
    max_workers = max(1, int(args.parallel_deploy))
    if max_workers == 1:
        for (mk, h, fn) in deployment_tasks:
            deployment_results.extend(fn(h))
    elif deployment_tasks:
        print(f"  [parallel-deploy={max_workers}] launching {len(deployment_tasks)} (model,host) tasks "
              f"concurrently across {len(set(t[0] for t in deployment_tasks))} models")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(fn, h): (mk, h) for (mk, h, fn) in deployment_tasks}
            for fut in as_completed(futures):
                mk, h = futures[fut]
                try:
                    deployment_results.extend(fut.result())
                except Exception as e:
                    print(f"  ERROR deploying {mk} to {h}: {e}")
                    deployment_results.append((mk, h, "exception", False))

    # 3. Deploy Scheduler (on coordinator node, after all instances are up)
    if deploy_scheduler_flag:
        # Determine scheduler host: explicit arg or first host from hosts file
        all_hosts_flat = []
        for hosts_list in available_nodes.values():
            all_hosts_flat.extend(hosts_list)
        # Also include assigned hosts from config
        for mk, md in final_config.items():
            for inst in md.get("instances", []):
                if inst["host"] not in all_hosts_flat:
                    all_hosts_flat.append(inst["host"])

        scheduler_host = args.scheduler_host
        if not scheduler_host and all_hosts_flat:
            scheduler_host = all_hosts_flat[0]

        if scheduler_host:
            sched_cmds = get_scheduler_deployment_commands(
                scheduler_host=scheduler_host,
                port=args.scheduler_port,
                model_config_path=args.output,
                scheduling=args.scheduling,
                scheduler_config=args.scheduler_config or "",
                predictor_config=getattr(args, "learned_predictor_config", "") or "",
                extra_args=args.scheduler_extra_args,
            )
            ok = run_ssh_cmd(scheduler_host, sched_cmds, "Deploying Scheduler")
            deployment_results.append(("scheduler", scheduler_host, "scheduler", ok))
        else:
            print("WARNING: No scheduler host available, skipping scheduler deployment")

    # 4. Save Config
    with open(args.output, 'w') as f:
        json.dump(final_config, f, indent=2)
    print(f"\nFinal configuration saved to {args.output}")

    # 4. Deployment Summary
    if deployment_results:
        print(f"\n{'='*60}")
        print("DEPLOYMENT SUMMARY")
        print(f"{'='*60}")
        failures = []
        for model, host, service, ok in deployment_results:
            status = "OK" if ok else "FAILED"
            host_short = host.split("@")[-1].split(".")[0] if "@" in host else host
            print(f"  {status:6s}  {model:20s}  {host_short:20s}  {service}")
            if not ok:
                failures.append((model, host_short, service))
        total = len(deployment_results)
        passed = sum(1 for _, _, _, ok in deployment_results if ok)
        print(f"\n  {passed}/{total} deployments succeeded")
        if failures:
            print(f"  FAILURES:")
            for model, host, service in failures:
                print(f"    - {model} / {host} / {service}")
            sys.exit(1)
        else:
            print("  All deployments successful!")


if __name__ == "__main__":
    main()
