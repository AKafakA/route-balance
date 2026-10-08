# Interactive Debugging Guide — March 26, 2026

## Overview
Two 1-hour interactive sessions on CSD3 A100:
1. **Session 1**: RoBERTa validation (~40 min)
2. **Session 2**: Length exploration — batch=32, cosine scheduler (~40 min)

## Pre-requisites (already done)
- ✅ CSD3 MFA authenticated
- ✅ RoBERTa venv created: `.venv_roberta` with transformers==4.50.3
- ✅ Main venv has all other packages
- ✅ All data present (train_fixed.jsonl, test_fixed.jsonl)
- ✅ `--scheduler` CLI arg added to train_bert_predictor.py

## Pending CSD3 batch job
- Job 26127544 (12h batch): log-transform 100ep + bucket 100ep + judge fused 5ep + judge 7B 5ep
- Status: queued (Priority), may take days to start
- Judge 7B already done separately (57.0% acc), judge fused has 1 epoch only

---

## Session 1: RoBERTa Validation

### Get interactive GPU node
```bash
ssh csd3
sintr -A KALYVIANAKI-SL3-GPU -p ampere -N1 -n1 --gres=gpu:1 -t 1:0:0 --qos=INTR
```

### Once on GPU node
```bash
cd /rds/user/wd312/hpc-work/llm/Block
export PYTHONPATH=.

# Verify RoBERTa venv works on GPU
.venv_roberta/bin/python -c "import torch; print('CUDA:', torch.cuda.is_available()); import transformers; print('transformers:', transformers.__version__)"
```

### Run RoBERTa experiments (3 targets, ~12 min each)

```bash
# 1. RoBERTa length (MSE) — ~12 min
.venv_roberta/bin/python -m block.predictor.cara.offline_training.train_bert_predictor \
    --input data/cara/training_data/train_fixed.jsonl \
    --test-input data/cara/training_data/test_fixed.jsonl \
    --regression-model-name roberta-base \
    --target length --lr 1e-5 --epochs 5 --seed 42 --device cuda \
    --output-dir models/cara/study/roberta_fused_length

# 2. RoBERTa similarity — ~12 min
.venv_roberta/bin/python -m block.predictor.cara.offline_training.train_bert_predictor \
    --input data/cara/training_data/train_fixed.jsonl \
    --test-input data/cara/training_data/test_fixed.jsonl \
    --regression-model-name roberta-base \
    --target similarity --lr 1e-5 --epochs 5 --seed 42 --device cuda \
    --output-dir models/cara/study/roberta_fused_similarity

# 3. RoBERTa judge_class — ~12 min
.venv_roberta/bin/python -m block.predictor.cara.offline_training.train_bert_predictor \
    --input data/cara/training_data/train_fixed.jsonl \
    --test-input data/cara/training_data/test_fixed.jsonl \
    --regression-model-name roberta-base \
    --target judge_class --lr 1e-5 --epochs 5 --seed 42 --device cuda \
    --output-dir models/cara/study/roberta_fused_judge_class
```

### What to check
- Does training start without CUDA errors? (transformers 4.50.3 should fix the SDPA mask bug)
- Check epoch 1 eval metrics are printed
- If all 3 complete → RoBERTa is validated, results saved

### If something fails
- Note the error message
- If OOM: try adding `--batch-size 8`
- If import error: check `PYTHONPATH=.` is set
- Report the error, don't debug further

---

## Session 2: Length Exploration

### Get another interactive GPU node
```bash
sintr -A KALYVIANAKI-SL3-GPU -p ampere -N1 -n1 --gres=gpu:1 -t 1:0:0 --qos=INTR
```

### Once on GPU node
```bash
cd /rds/user/wd312/hpc-work/llm/Block
export PYTHONPATH=.
```

### Test A: batch=32, lr=4e-5, cosine scheduler, 10 epochs (Qwen-14B only)
```bash
.venv/bin/python -m block.predictor.cara.offline_training.train_bert_predictor \
    --input data/cara/training_data/train_fixed.jsonl \
    --test-input data/cara/training_data/test_fixed.jsonl \
    --regression-model-name answerdotai/ModernBERT-base \
    --target-models Qwen/Qwen2.5-14B \
    --target length --lr 4e-5 --epochs 10 --batch-size 32 --scheduler cosine --seed 42 --device cuda \
    --output-dir models/cara/exploration/modernbert_length_b32_lr4e5_cosine
```

### Test B: batch=32, lr=4e-5, cosine_with_restarts, 10 epochs
```bash
.venv/bin/python -m block.predictor.cara.offline_training.train_bert_predictor \
    --input data/cara/training_data/train_fixed.jsonl \
    --test-input data/cara/training_data/test_fixed.jsonl \
    --regression-model-name answerdotai/ModernBERT-base \
    --target-models Qwen/Qwen2.5-14B \
    --target length --lr 4e-5 --epochs 10 --batch-size 32 --scheduler cosine_with_restarts --seed 42 --device cuda \
    --output-dir models/cara/exploration/modernbert_length_b32_lr4e5_cosine_restart
```

### Test C (if time): batch=32, lr=1e-5, cosine, 10 epochs (same LR as before, just different scheduler)
```bash
.venv/bin/python -m block.predictor.cara.offline_training.train_bert_predictor \
    --input data/cara/training_data/train_fixed.jsonl \
    --test-input data/cara/training_data/test_fixed.jsonl \
    --regression-model-name answerdotai/ModernBERT-base \
    --target-models Qwen/Qwen2.5-14B \
    --target length --lr 1e-5 --epochs 10 --batch-size 32 --scheduler cosine --seed 42 --device cuda \
    --output-dir models/cara/exploration/modernbert_length_b32_lr1e5_cosine
```

### What to check
- Does batch=32 fit in A100-80GB? (batch=64 OOMed)
- Compare eval_loss at epoch 10 across tests vs our baseline:
  - **Baseline** (batch=16, lr=1e-5, polynomial, epoch 10): loss=29,300, MAE=102
  - If Test A/B/C show lower loss at epoch 10 → worth submitting full 300ep run
- Check if cosine_with_restarts shows loss recovery after restart

### What to report back
For each test, note:
1. Did it run? Any errors?
2. Epoch 1 eval_loss and MAE
3. Epoch 5 eval_loss and MAE
4. Epoch 10 eval_loss and MAE (if reached)
5. Any signs of faster convergence vs baseline?

---

## Quick Reference

### Key paths on CSD3
```
/rds/user/wd312/hpc-work/llm/Block/          # repo root
    .venv/bin/python                           # main venv (transformers 5.3)
    .venv_roberta/bin/python                   # roberta venv (transformers 4.50.3)
    data/cara/training_data/train_fixed.jsonl  # training data
    data/cara/training_data/test_fixed.jsonl   # test data
    models/cara/study/                         # completed models
    models/cara/exploration/                   # exploration results
    logs/                                      # job logs
```

### Check existing results
```bash
# See what models are trained
find models/cara/study -name "training_results.json" | sort

# Check a result
cat models/cara/study/modernbert_7b_judge_class/training_results.json | python3 -m json.tool
```

### Check batch job status
```bash
squeue -u wd312
```

---

## After debugging — what to tell Claude
1. RoBERTa: worked / failed (with error)
2. Length exploration: epoch 10 loss/MAE for each test
3. Any surprises or issues
4. Then we decide:
   - If RoBERTa works → submit sbatch for full runs
   - If cosine scheduler helps → submit 300ep batch job
   - If neither helps → stick with current results, focus on bucket approach
