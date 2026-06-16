BLOCK_GITHUB_LINK="https://github.com/AKafakA/Block"
VLLM_GITHUB_LINK="https://github.com/AKafakA/vllm.git"
OLLAMA_GITHUB_LINK="https://github.com/AKafakA/ollama.git"

 general setup for all hosts
echo "Install CUDA and dependencies on all hosts..."


# also need to manually run the following command to create the directory /mydata/hf_cache on all hosts and reboot will reset
# for d8545
# sudo mkfs.ext4 -F /dev/nvme0n1 && sudo mkdir -p /mydata && sudo mount /dev/nvme0n1 /mydata && sudo chmod 777 /mydata
# for c4130
# sudo mkfs.ext4 /dev/sdb && sudo mkdir -p /mydata && sudo mount /dev/sdb /mydata && sudo chmod 777 /mydata

parallel-ssh -t 0 -h route_balance/config/p100_smoke/hosts "sudo apt update && sudo apt full-upgrade -y"
parallel-ssh -t 0 -h route_balance/config/p100_smoke/hosts "sudo apt install -y python3-pip python3-venv ccache"
parallel-ssh -t 0 -h route_balance/config/p100_smoke/hosts "pip install --upgrade pip"
parallel-ssh -t 0 -h route_balance/config/p100_smoke/hosts "git clone ${BLOCK_GITHUB_LINK} && cd Block && git checkout route_balance  && pip install -r requirements.txt"
parallel-ssh -t 0 -h route_balance/config/p100_smoke/hosts "pip3 install torch torchvision"
parallel-ssh -t 0 -h route_balance/config/p100_smoke/hosts "pip install dacite"
parallel-ssh -t 0 -h route_balance/config/p100_smoke/hosts "wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin && sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600"
parallel-ssh -t 0 -h route_balance/config/p100_smoke/hosts "wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda-repo-ubuntu2204-12-8-local_12.8.0-570.86.10-1_amd64.deb && sudo dpkg -i cuda-repo-ubuntu2204-12-8-local_12.8.0-570.86.10-1_amd64.deb"
parallel-ssh -t 0 -h route_balance/config/p100_smoke/hosts "sudo cp /var/cuda-repo-ubuntu2204-12-8-local/cuda-*-keyring.gpg /usr/share/keyrings/ && sudo apt-get update"
parallel-ssh -t 0 -h route_balance/config/p100_smoke/hosts "sudo dpkg --configure -a && sudo apt-get -y install cuda-toolkit-12-8"
parallel-ssh -t 0 -h route_balance/config/p100_smoke/ampere_hosts "sudo apt-get install -y nvidia-open"
parallel-ssh -t 0 -h route_balance/config/p100_smoke/volta_hosts "sudo apt-get install -y cuda-drivers"
parallel-ssh -t 0 -h route_balance/config/p100_smoke/pascal_hosts "sudo apt-get install -y cuda-drivers"
parallel-ssh -t 0 -h route_balance/config/p100_smoke/ampere_hosts "pip install --upgrade torch"
parallel-ssh -t 0 -h route_balance/config/p100_smoke/volta_hosts "pip install --upgrade torch"
parallel-ssh -t 0 -h route_balance/config/p100_smoke/hosts "pip install --upgrade "ray[cgraph]""
# Set environment variables BEFORE the interactive guard in .bashrc
# Uses a separate bash script (setup_env.sh) to avoid quoting issues with parallel-ssh
parallel-ssh -t 0 -h route_balance/config/p100_smoke/hosts "bash ~/Block/route_balance/exp/route_balance/setup_env.sh"

echo "cuda installation completed on all hosts and now tested with nvidia-smi..."
parallel-ssh -t 0 -h route_balance/config/p100_smoke/hosts "sudo nvidia-smi -mig 0"
parallel-ssh -t 0 -h route_balance/config/p100_smoke/hosts  "rm -r ~/cuda-repo-*.deb"

# For ampere hosts and volta hosts, which are able to run vllm
echo "Starting setup for vllm hosts..."
parallel-ssh -t 0 -h route_balance/config/p100_smoke/ampere_hosts "git clone ${VLLM_GITHUB_LINK} && cd vllm && git checkout route_balance_v_11"
parallel-ssh -t 0 -h route_balance/config/p100_smoke/ampere_hosts  "cd vllm && sudo VLLM_USE_PRECOMPILED=1 pip install --editable ."
parallel-ssh -t 0 -h route_balance/config/p100_smoke/ampere_hosts "git clone ${BLOCK_GITHUB_LINK} && cd Block && git checkout route_balance  && pip install -r requirements.txt"
parallel-ssh -t 0 -h route_balance/config/p100_smoke/volta_hosts "git clone ${VLLM_GITHUB_LINK} && cd vllm && git checkout route_balance_v_11"
parallel-ssh -t 0 -h route_balance/config/p100_smoke/volta_hosts  "cd vllm && sudo VLLM_USE_PRECOMPILED=1 pip install --editable ."




# install customized vllm for pascal hosts (using the P100-compatible branch)
echo "Starting setup for pascal hosts..."
# 1. Install Build Dependencies
# vLLM requires cmake to build from source
parallel-ssh -t 0 -h route_balance/config/p100_smoke/pascal_hosts "sudo apt install -y cmake"
# Upgrade cmake via pip (vLLM requires cmake >= 3.26, Ubuntu 22.04 ships 3.22)
parallel-ssh -t 0 -h route_balance/config/p100_smoke/pascal_hosts "pip install cmake --upgrade"
# 2. Clone vLLM and checkout your P100 branch
parallel-ssh -t 0 -h route_balance/config/p100_smoke/pascal_hosts "git clone ${VLLM_GITHUB_LINK} && cd vllm && git checkout route_balance_p100_v_6.0"
# 3. Install Xformers
# P100 cannot run FlashAttn, so we install xformers as the fallback backend
parallel-ssh -t 0 -h route_balance/config/p100_smoke/pascal_hosts "pip install xformers"
# 4. Build and Install vLLM
# We explicit set TORCH_CUDA_ARCH_LIST=6.0 to force the compiler to generate Pascal (sm_60) binaries.
# We pass the env var into sudo to ensure the build process sees it.
parallel-ssh -t 0 -h route_balance/config/p100_smoke/pascal_hosts "cd vllm && sudo PATH=\"/users/${CLOUDLAB_USER}/.local/bin:\$PATH\" CUDACXX=/usr/local/cuda-12.8/bin/nvcc TORCH_CUDA_ARCH_LIST=6.0 MAX_JOBS=4 CMAKE_BUILD_PARALLEL_LEVEL=4 pip install --editable ."

# =============================================================================
# IMPORTANT: Reboot all nodes after setup to load the NVIDIA driver!
# nvidia-smi will NOT work until reboot.
# Run: parallel-ssh -t 0 -h route_balance/config/p100_smoke/hosts "sudo reboot"
# Wait ~2 min, then verify: parallel-ssh -t 0 -h route_balance/config/p100_smoke/hosts "nvidia-smi --query-gpu=name --format=csv,noheader"
# =============================================================================
echo ""
echo "============================================================"
echo "  SETUP COMPLETE — REBOOT REQUIRED"
echo "  Run: parallel-ssh -t 0 -h route_balance/config/p100_smoke/hosts \"sudo reboot\""
echo "  Then wait ~2 min and verify nvidia-smi works"
echo "============================================================"
