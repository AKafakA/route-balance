#!/bin/bash
# Setup script for P100 smoke test nodes (2× c240g5)
# Run from local machine (not on CloudLab nodes)
#
# Usage:
#   bash route_balance/exp/route_balance/setup_p100_smoketest.sh
#
# Prerequisites:
#   - SSH access to both nodes via ssh-agent forwarding
#   - Nodes listed in route_balance/config/p100_smoketest_hosts

set -e

HOSTS_FILE="route_balance/config/p100_smoketest_hosts"
BLOCK_GITHUB_LINK="https://github.com/AKafakA/Block"
VLLM_GITHUB_LINK="https://github.com/AKafakA/vllm.git"
HF_TOKEN="${HF_TOKEN}"

NODE0="asdwb@c240g5-110131.wisc.cloudlab.us"
NODE1="asdwb@c240g5-110211.wisc.cloudlab.us"

run_on_all() {
    echo ">>> Running on all nodes: $1"
    for host in $NODE0 $NODE1; do
        echo "  [$host]"
        ssh -A -o StrictHostKeyChecking=no "$host" "$1" 2>&1 | tail -3
    done
    echo ""
}

run_on_host() {
    echo ">>> Running on $1: $2"
    ssh -A -o StrictHostKeyChecking=no "$1" "$2" 2>&1
}

echo "============================================================"
echo "  P100 Smoke Test Setup — $(date)"
echo "============================================================"

# Step 1: System update + dependencies
echo "=== Step 1: System update + dependencies ==="
run_on_all "sudo apt update -qq && sudo apt install -y -qq python3-pip python3-venv ccache cmake"

# Step 2: CUDA 12.8 install
echo "=== Step 2: CUDA 12.8 ==="
run_on_all "wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin && sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600"
run_on_all "wget -q https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda-repo-ubuntu2204-12-8-local_12.8.0-570.86.10-1_amd64.deb && sudo dpkg -i cuda-repo-ubuntu2204-12-8-local_12.8.0-570.86.10-1_amd64.deb"
run_on_all "sudo cp /var/cuda-repo-ubuntu2204-12-8-local/cuda-*-keyring.gpg /usr/share/keyrings/ && sudo apt-get update -qq"
run_on_all "sudo dpkg --configure -a && sudo apt-get -y -qq install cuda-toolkit-12-8"
run_on_all "sudo apt-get install -y -qq cuda-drivers"
echo "Verifying CUDA..."
run_on_all "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader"

# Step 3: Environment variables
echo "=== Step 3: Environment setup ==="
run_on_all 'cat >> ~/.bashrc << "ENVEOF"
# === Environment (sourced for ALL shells) ===
export CUDA_HOME=/usr/local/cuda
export PATH="${PATH}:${CUDA_HOME}/bin:/usr/local/cuda-12.8/bin"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CUDA_HOME}/lib64:/usr/local/cuda-12.8/lib64:${CUDA_HOME}/targets/x86_64-linux/lib:/usr/lib/x86_64-linux-gnu"
for dir in ${HOME}/.local/lib/python3.10/site-packages/nvidia/*/lib ${HOME}/.local/lib/python3.10/site-packages/cusparselt/lib /usr/local/lib/python3.10/dist-packages/nvidia/*/lib; do [ -d "${dir}" ] && export LD_LIBRARY_PATH="${dir}:${LD_LIBRARY_PATH}"; done
export PYTHONPATH="${HOME}/vllm:${HOME}/Block:${PYTHONPATH}"
export HF_TOKEN='"$HF_TOKEN"'
# === End environment block ===
ENVEOF'

# Step 4: Clone repos
echo "=== Step 4: Clone Block + vLLM ==="
run_on_all "cd ~ && [ -d Block ] || git clone ${BLOCK_GITHUB_LINK} && cd Block && git checkout route_balance"
run_on_all "cd ~/Block && pip install -r requirements.txt 2>&1 | tail -3"
run_on_all "pip3 install torch torchvision 2>&1 | tail -3"
run_on_all "pip install sentence-transformers faiss-cpu 2>&1 | tail -3"

# Step 5: vLLM for P100 (Pascal, sm_60)
echo "=== Step 5: vLLM for P100 (build from source) ==="
run_on_all "cd ~ && [ -d vllm ] || git clone ${VLLM_GITHUB_LINK} && cd vllm && git checkout route_balance_p100_v_6.0"
run_on_all "pip install xformers 2>&1 | tail -3"
echo "Building vLLM (this takes ~15-20 min per node)..."
run_on_all "cd ~/vllm && sudo CUDACXX=/usr/local/cuda-12.8/bin/nvcc TORCH_CUDA_ARCH_LIST=6.0 MAX_JOBS=7 CMAKE_BUILD_PARALLEL_LEVEL=7 pip install --editable . 2>&1 | tail -5"

# Step 6: Pin transformers (sentence-transformers may pull newer version)
echo "=== Step 6: Pin transformers==4.50.3 ==="
run_on_all "pip install transformers==4.50.3 2>&1 | tail -3"

# Step 7: Verify
echo "=== Step 7: Verification ==="
run_on_all "source ~/.bashrc && python3 -c 'import vllm; print(f\"vLLM {vllm.__version__}\")' 2>&1"
run_on_all "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"

echo ""
echo "============================================================"
echo "  Setup complete — $(date)"
echo "============================================================"
echo ""
echo "Next: Deploy with deploy_route_balance.py using model_deployment_p100_smoketest.json"
