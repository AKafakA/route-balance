#!/bin/bash
#SBATCH -J route_balance_refsim
#SBATCH -A KALYVIANAKI-SL3-GPU
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --output=/rds/user/anon/hpc-work/llm/RouteBalance/experiment_output/logs/ref_similarity_%j.log

cd /rds/user/anon/hpc-work/llm/RouteBalance
source .venv/bin/activate
export PYTHONPATH=/rds/user/anon/hpc-work/llm/RouteBalance:$PYTHONPATH
export HF_HOME=/rds/user/anon/hpc-work/hf_cache

echo "=== Starting reference similarity computation ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Python: $(which python)"

python route_balance/predictor/route_balance/offline_training/add_reference_similarity.py \
  --raw-data data/route_balance/training_data/route_balance_v3_all_training_fixed.json \
  --train-data data/route_balance/training_data/train_fixed.jsonl \
  --test-data data/route_balance/training_data/test_fixed.jsonl \
  --output-dir data/route_balance/training_data_with_ref/ \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --device cuda

echo "=== Done ==="
ls -la data/route_balance/training_data_with_ref/
