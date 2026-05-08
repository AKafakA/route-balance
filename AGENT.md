# Project Rules

## CRITICAL: No Autonomous Experiment Decisions

**NEVER change experiment parameters without explicit user approval.** This includes:

- Hyperparameters: batch size, learning rate, epochs, max_length, warmup, scheduler
- Data processing: truncation length, padding strategy, filtering, transforms
- Model configs: loss function, model architecture, num_labels
- Deployment configs: GPU memory utilization, enforce-eager, max-model-len, vllm params
- Infrastructure workarounds: any change made to avoid OOM, errors, or resource constraints

**When hitting any issue (OOM, crash, unexpected result):**
1. STOP immediately
2. Report the issue and what caused it
3. List possible options with trade-offs
4. WAIT for user decision — do NOT proceed with a workaround

**If user is away:** Queue the question and wait. A delayed experiment is better than a wasted one.

**Reason:** Silently changing max_length from 1024 to 512 to work around OOM invalidated hours of Phase 1 training results. The user must make all experiment design decisions.

## CRITICAL: Process Management on Remote Machines

**Before launching ANY training job:**
1. Check for existing processes: `pgrep -u $(whoami) -f train -a`
2. Kill ALL stale processes first
3. Verify GPU is clean: `nvidia-smi`
4. Launch ONE process only
5. Verify exactly 1 process running after launch

**Never retry OOM failures by changing parameters.** Report the OOM and wait.

## Allowed Auto-Fixes (no user approval needed)

Fixing non-semantic infrastructure issues that don't change training results:
- Creating missing directories
- Fixing import errors (missing PYTHONPATH, missing packages)
- Killing stale/zombie processes
- Retrying after transient network errors
- Fixing file permissions

These must NOT change any hyperparameters, data processing, or model configuration.

## CRITICAL: Use /mydata for CloudLab Storage

**Save models, checkpoints, and large outputs to `/mydata/` on CloudLab nodes when available.**
Root filesystem is only 63GB and fills quickly with model checkpoints.
- d8545 (A100): `/mydata/` available (1.5TB NVMe)
- c4130 (V100): `/mydata/` available (880GB)
- d7525 (A30): **NO /mydata** — use root carefully or `/tmp`
- c240g5 (P100): **NO /mydata** — use root carefully or `/tmp`

Default paths when `/mydata` available:
- Model output: `/mydata/models/cara/`
- Training logs: `/mydata/training_logs/`
- Venvs: `/mydata/training_venv/`
- HF cache: `/mydata/hf_cache/`

## CRITICAL: Verify Before Reporting

**Before telling the user "everything is fine" or "launched successfully":**
1. Verify the actual running command matches what was intended
2. Check process count — should be exactly what's expected
3. Check for errors in the first few lines of output
4. Do NOT report success until verified
