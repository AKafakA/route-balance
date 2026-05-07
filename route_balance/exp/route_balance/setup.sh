BLOCK_GITHUB_LINK="https://github.com/anon/RouteBalance"
VLLM_GITHUB_LINK="https://github.com/anon/vllm.git"
OLLAMA_GITHUB_LINK="https://github.com/anon/ollama.git"

# general setup for all hosts
echo "Install CUDA and dependencies on all hosts..."


# also need to manually run the following command to create the directory /mydata/hf_cache on all hosts and reboot will reset
# for d8545
# sudo mkfs.ext4 -F /dev/nvme0n1 && sudo mkdir -p /mydata && sudo mount /dev/nvme0n1 /mydata && sudo chmod 777 /mydata
# for c4130
# sudo mkfs.ext4 /dev/sdb && sudo mkdir -p /mydata && sudo mount /dev/sdb /mydata && sudo chmod 777 /mydata
#
# =============================================================================
# P100 vLLM (route_balance_p100_v_6.0) patches — Apr 26 2026
# =============================================================================
# Two bugs in route_balance_p100_v_6.0 caused /instance_stats and /scheduler_stats to
# hang indefinitely AND missing /instance_stats URL alias:
#
#   1. RPCAggregatedStatsResponse missing from output_handler dispatch tuple
#      in vllm/engine/multiprocessing/client.py:256 → server processes the
#      request but client silently drops the response.
#   2. /instance_stats URL alias missing in vllm/entrypoints/openai/api_server.py
#      (route_balance_v_11 uses /instance_stats, route_balance_p100_v_6.0 only had /scheduler_stats).
#   3. /schedule_trace per-request schema differs from route_balance_v_11 (4 fields vs 6).
#
# After git clone of route_balance_p100_v_6.0 on pascal_hosts (line 56), copy the
# patched source files BEFORE pip install --editable . step:
#
#   parallel-scp -h route_balance/config/pascal_hosts \
#     route_balance_paper/smoke_test_apr_13/p100_patches/api_server.py \
#     '~/vllm/vllm/entrypoints/openai/api_server.py'
#   parallel-scp -h route_balance/config/pascal_hosts \
#     route_balance_paper/smoke_test_apr_13/p100_patches/client.py \
#     '~/vllm/vllm/engine/multiprocessing/client.py'
#
# These patched files are saved in the repo at the path above and are byte-
# identical to what's running on the 5 P100 nodes today. Schema verified
# matches route_balance_v_11 /instance_stats (16 fields) and /schedule_trace (6
# per-request fields).
# =============================================================================

parallel-ssh -t 0 -h route_balance/config/hosts "sudo apt update && sudo apt full-upgrade -y"
parallel-ssh -t 0 -h route_balance/config/hosts "sudo apt install -y python3-pip python3-venv ccache"
parallel-ssh -t 0 -h route_balance/config/hosts "pip install --upgrade pip"
parallel-ssh -t 0 -h route_balance/config/hosts "git clone ${BLOCK_GITHUB_LINK} && cd RouteBalance && git checkout route_balance  && pip install -r requirements.txt"
parallel-ssh -t 0 -h route_balance/config/hosts "pip3 install torch torchvision"
parallel-ssh -t 0 -h route_balance/config/hosts "pip install dacite"
parallel-ssh -t 0 -h route_balance/config/hosts "wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin && sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600"
parallel-ssh -t 0 -h route_balance/config/hosts "wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda-repo-ubuntu2204-12-8-local_12.8.0-570.86.10-1_amd64.deb && sudo dpkg -i cuda-repo-ubuntu2204-12-8-local_12.8.0-570.86.10-1_amd64.deb"
parallel-ssh -t 0 -h route_balance/config/hosts "sudo cp /var/cuda-repo-ubuntu2204-12-8-local/cuda-*-keyring.gpg /usr/share/keyrings/ && sudo apt-get update"
parallel-ssh -t 0 -h route_balance/config/hosts "sudo dpkg --configure -a && sudo apt-get -y install cuda-toolkit-12-8"
parallel-ssh -t 0 -h route_balance/config/ampere_hosts "sudo apt-get install -y nvidia-open"
parallel-ssh -t 0 -h route_balance/config/volta_hosts "sudo apt-get install -y cuda-drivers"
parallel-ssh -t 0 -h route_balance/config/pascal_hosts "sudo apt-get install -y cuda-drivers"
parallel-ssh -t 0 -h route_balance/config/ampere_hosts "pip install --upgrade torch"
parallel-ssh -t 0 -h route_balance/config/volta_hosts "pip install --upgrade torch"
parallel-ssh -t 0 -h route_balance/config/hosts "pip install --upgrade "ray[cgraph]""
# Set environment variables BEFORE the interactive guard in .bashrc
# Uses a separate bash script (setup_env.sh) to avoid quoting issues with parallel-ssh
parallel-ssh -t 0 -h route_balance/config/hosts "bash ~/RouteBalance/route_balance/exp/route_balance/setup_env.sh"

echo "cuda installation completed on all hosts and now tested with nvidia-smi..."
parallel-ssh -t 0 -h route_balance/config/hosts "sudo nvidia-smi -mig 0"
parallel-ssh -t 0 -h route_balance/config/hosts  "rm -r ~/cuda-repo-*.deb"

# For ampere hosts and volta hosts, which are able to run vllm
echo "Starting setup for vllm hosts..."
parallel-ssh -t 0 -h route_balance/config/ampere_hosts "git clone ${VLLM_GITHUB_LINK} && cd vllm && git checkout route_balance_v_11"
parallel-ssh -t 0 -h route_balance/config/ampere_hosts  "cd vllm && sudo VLLM_USE_PRECOMPILED=1 pip install --editable ."
parallel-ssh -t 0 -h route_balance/config/ampere_hosts "git clone ${BLOCK_GITHUB_LINK} && cd RouteBalance && git checkout route_balance  && pip install -r requirements.txt"
parallel-ssh -t 0 -h route_balance/config/volta_hosts "git clone ${VLLM_GITHUB_LINK} && cd vllm && git checkout route_balance_v_11"
parallel-ssh -t 0 -h route_balance/config/volta_hosts  "cd vllm && sudo VLLM_USE_PRECOMPILED=1 pip install --editable ."




# install customized vllm for pascal hosts (using the P100-compatible branch)
echo "Starting setup for pascal hosts..."
# 1. Install Build Dependencies
# vLLM requires cmake to build from source
parallel-ssh -t 0 -h route_balance/config/pascal_hosts "sudo apt install -y cmake"
# 2. Clone vLLM and checkout your P100 branch
parallel-ssh -t 0 -h route_balance/config/pascal_hosts "git clone ${VLLM_GITHUB_LINK} && cd vllm && git checkout route_balance_p100_v_6.0"
# 3. Install Xformers
# P100 cannot run FlashAttn, so we install xformers as the fallback backend
parallel-ssh -t 0 -h route_balance/config/pascal_hosts "pip install xformers"
# 4. Build and Install vLLM
# We explicit set TORCH_CUDA_ARCH_LIST=6.0 to force the compiler to generate Pascal (sm_60) binaries.
# We pass the env var into sudo to ensure the build process sees it.
parallel-ssh -t 0 -h route_balance/config/pascal_hosts "cd vllm && sudo CUDACXX=/usr/local/cuda-12.8/bin/nvcc TORCH_CUDA_ARCH_LIST=6.0 MAX_JOBS=7 CMAKE_BUILD_PARALLEL_LEVEL=7 pip install --editable ."

# 5. Install model estimator dependencies + pin transformers
# Pin to a version compatible with the currently-checked-out vLLM branch(es):
#   - route_balance_v_11 (ampere/volta)  requires: transformers >= 4.56.0, < 5
#   - route_balance_p100_v_6.0 (pascal)  requires: transformers == 4.50.3
# sentence-transformers may pull transformers>=5.0 — must override with the vLLM
# fork's declared floor. Keep this pin in sync with the branches' requirements/common.txt.
parallel-ssh -t 0 -h route_balance/config/hosts "pip install sentence-transformers faiss-cpu"
# Pin transformers at BOTH user and system level.
# 2026-04-13: bumped 4.50.3 → 4.56.0 after anon/vllm route_balance_v_11 rebased onto upstream
# requiring transformers>=4.56 (commit 2894a9872 added predicted_decode_tokens to /schedule_trace).
parallel-ssh -t 0 -h route_balance/config/hosts "pip install transformers==4.56.0"
parallel-ssh -t 0 -h route_balance/config/hosts "sudo pip install transformers==4.56.0"

# 5b. P-D disaggregation runtime (NVIDIA NIXL) — required for vLLM's NixlConnector
# when --kv-transfer-config uses kv_connector=NixlConnector. Without this, vLLM
# fails with "RuntimeError: NIXL is not available" at engine init.
# 2026-04-13: added after P-D deploy on Apr 13 A30 smoke revealed missing dep.
parallel-ssh -t 0 -h route_balance/config/ampere_hosts "pip install nixl==1.0.0"
parallel-ssh -t 0 -h route_balance/config/volta_hosts "pip install nixl==1.0.0"
# Pascal (P100) doesn't support NIXL (requires CUDA compute capability 7.0+ typically);
# P-D on Pascal is not a supported configuration.

# 5c. ONNX runtime for the fused RoBERTa predictor (optional fast path).
# Enables the "runtime: onnx" mode in predictor_config_*.json (see model_estimator.py).
# Projected ~4-5× speedup for estimator_ms (100ms → ~20ms CPU) per Apr 14 plan.
# 2026-04-14: added for ONNX dual-runtime support on all GPU-capable nodes.
parallel-ssh -t 0 -h route_balance/config/hosts "pip install onnxruntime==1.23.2 onnx==1.17.0 onnxscript"

# =============================================================================
# IMPORTANT: Reboot all nodes after setup to load the NVIDIA driver!
# nvidia-smi will NOT work until reboot.
# Run: parallel-ssh -t 0 -h route_balance/config/hosts "sudo reboot"
# Wait ~2 min, then verify: parallel-ssh -t 0 -h route_balance/config/hosts "nvidia-smi --query-gpu=name --format=csv,noheader"
# =============================================================================
echo ""
echo "============================================================"
echo "  SETUP COMPLETE — REBOOT REQUIRED"
echo "  Run: parallel-ssh -t 0 -h route_balance/config/hosts \"sudo reboot\""
echo "  Then wait ~2 min and verify nvidia-smi works"
echo "============================================================"
