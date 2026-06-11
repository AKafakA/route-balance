#!/bin/bash
# Write CUDA/nvidia environment variables to .bashrc on all nodes.
# Run separately from setup.sh or as part of it.
# This replaces the fragile inline Python approach with a simple bash heredoc.
#
# Usage:
#   parallel-ssh -t 0 -h route_balance/config/hosts "bash ~/Block/route_balance/exp/route_balance/setup_env.sh"
#   OR
#   parallel-ssh -t 0 -h route_balance/config/p100_smoke/hosts "bash ~/Block/route_balance/exp/route_balance/setup_env.sh"

MARKER="# === Environment (sourced for ALL shells) ==="

if grep -q "$MARKER" ~/.bashrc 2>/dev/null; then
    echo "Env block already present in .bashrc, skipping"
    exit 0
fi

# Find the interactive guard line to insert BEFORE it
if grep -q "# If not running interactively" ~/.bashrc; then
    # Insert env block before the interactive guard
    sed -i "/# If not running interactively/i\\
$MARKER\\
export CUDA_HOME=/usr/local/cuda\\
export PATH=\"\${PATH}:\${CUDA_HOME}/bin:/usr/local/cuda-12.8/bin\"\\
export LD_LIBRARY_PATH=\"\${LD_LIBRARY_PATH}:\${CUDA_HOME}/lib64:/usr/local/cuda-12.8/lib64:\${CUDA_HOME}/targets/x86_64-linux/lib:/usr/lib/x86_64-linux-gnu\"\\
for dir in \${HOME}/.local/lib/python3.10/site-packages/nvidia/*/lib /usr/local/lib/python3.10/dist-packages/nvidia/*/lib \${HOME}/.local/lib/python3.10/site-packages/cusparselt/lib; do [ -d \"\${dir}\" ] \\&\\& export LD_LIBRARY_PATH=\"\${dir}:\${LD_LIBRARY_PATH}\"; done\\
export PYTHONPATH=\"\${HOME}/vllm:\${HOME}/Block:\${PYTHONPATH}\"\\
export HF_TOKEN=${HF_TOKEN}\\
# === End environment block ===" ~/.bashrc
    echo "Env block inserted before interactive guard"
else
    # No interactive guard found, append to end
    cat >> ~/.bashrc << 'ENVEOF'
# === Environment (sourced for ALL shells) ===
export CUDA_HOME=/usr/local/cuda
export PATH="${PATH}:${CUDA_HOME}/bin:/usr/local/cuda-12.8/bin"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CUDA_HOME}/lib64:/usr/local/cuda-12.8/lib64:${CUDA_HOME}/targets/x86_64-linux/lib:/usr/lib/x86_64-linux-gnu"
for dir in ${HOME}/.local/lib/python3.10/site-packages/nvidia/*/lib /usr/local/lib/python3.10/dist-packages/nvidia/*/lib ${HOME}/.local/lib/python3.10/site-packages/cusparselt/lib; do [ -d "${dir}" ] && export LD_LIBRARY_PATH="${dir}:${LD_LIBRARY_PATH}"; done
export PYTHONPATH="${HOME}/vllm:${HOME}/Block:${PYTHONPATH}"
export HF_TOKEN=${HF_TOKEN}
# === End environment block ===
ENVEOF
    echo "Env block appended to .bashrc"
fi

# Verify
source ~/.bashrc
python3 -c "import torch; print(f'torch OK: {torch.__version__}')" 2>/dev/null && echo "Verification: PASS" || echo "Verification: torch import failed (may need reboot first)"
