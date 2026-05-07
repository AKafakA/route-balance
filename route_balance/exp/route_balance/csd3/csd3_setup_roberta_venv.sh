#!/bin/bash
# Setup RoBERTa venv on CSD3 login node.
# Run on login node (no GPU needed): bash route_balance/exp/route_balance/csd3_setup_roberta_venv.sh
#
# Creates a SEPARATE venv from the main one to avoid breaking ModernBERT.
# Main venv: /rds/user/anon/hpc-work/venv/        (transformers 5.x, for ModernBERT)
# RoBERTa:   /rds/user/anon/hpc-work/venv_roberta/ (transformers 4.50.3)
#
# Both share the same /rds filesystem, visible to SLURM compute nodes.

set -e

VENV_DIR="/rds/user/anon/hpc-work/venv_roberta"
WORK_DIR="/rds/user/anon/hpc-work/llm/RouteBalance"

echo "=== Creating RoBERTa venv at $VENV_DIR ==="

if [ -d "$VENV_DIR" ]; then
    echo "Venv already exists. Checking..."
    source "$VENV_DIR/bin/activate"
    TF_VER=$(python3 -c "import transformers; print(transformers.__version__)" 2>/dev/null || echo "none")
    if [ "$TF_VER" = "4.50.3" ]; then
        echo "Venv OK: transformers=$TF_VER"
        python3 -c "import torch; print(f'torch={torch.__version__}')" 2>/dev/null || echo "torch not installed yet"
        exit 0
    fi
    echo "Wrong transformers version ($TF_VER), rebuilding..."
    deactivate 2>/dev/null || true
    rm -rf "$VENV_DIR"
fi

python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "=== Installing packages ==="
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.50.3 accelerate datasets
pip install scikit-learn scipy numpy

echo ""
echo "=== Verification ==="
python3 -c "
import transformers, torch
print(f'transformers={transformers.__version__}')
print(f'torch={torch.__version__}')
# CPU-only check on login node (no GPU)
print(f'CUDA compiled: {torch.cuda.is_available()}')
"

echo ""
echo "=== Testing RoBERTa model load (CPU) ==="
python3 -c "
from transformers import AutoModelForSequenceClassification, AutoTokenizer
tok = AutoTokenizer.from_pretrained('roberta-base')
model = AutoModelForSequenceClassification.from_pretrained('roberta-base', num_labels=1)
print(f'RoBERTa: {sum(p.numel() for p in model.parameters())/1e6:.0f}M params')
print(f'Max position embeddings: {model.config.max_position_embeddings} (context: 512 tokens)')
print('RoBERTa loads OK on CPU')
"

echo ""
echo "=== Testing training script import ==="
cd "$WORK_DIR"
export PYTHONPATH="$WORK_DIR:\$PYTHONPATH"
python3 -c "
from route_balance.predictor.route_balance.offline_training.train_bert_predictor import train_model_for_target
print('train_bert_predictor imports OK')
"

echo ""
echo "=== Setup complete ==="
echo "Venv: $VENV_DIR"
echo "Use in SLURM: source $VENV_DIR/bin/activate"
echo "Main venv (ModernBERT) untouched at: /rds/user/anon/hpc-work/venv/"
