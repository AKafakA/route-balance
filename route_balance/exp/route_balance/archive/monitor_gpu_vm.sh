#!/bin/bash
# Monitor GPU VM training status
# Run via cron every 30 min from VPS/desktop
# Usage: bash route_balance/exp/route_balance/monitor_gpu_vm.sh

LOGFILE="/home/anon/Code/llm/RouteBalance/route_balance_paper/plans/gpu_vm_monitor.log"
REMOTE="anon@gxp-l40s-2.cluster.example"
REPORT="/home/anon/Code/llm/RouteBalance/route_balance_paper/plans/experiment_report_march23_25.md"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

echo "$(timestamp) === Monitor check ===" >> "$LOGFILE"

# Check SSH connectivity (single attempt, don't spam)
if ! ssh -o ConnectTimeout=10 -o BatchMode=yes "$REMOTE" "echo ok" >/dev/null 2>&1; then
    echo "$(timestamp) GPU VM unreachable — skipping" >> "$LOGFILE"
    exit 0
fi

# All checks in one SSH session
STATUS=$(ssh -o ConnectTimeout=10 "$REMOTE" 'cd ~/Code/llm/RouteBalance 2>/dev/null || exit 1

echo "DISK_FREE=$(df -BG /home/anon --output=avail | tail -1 | tr -d " G")"

# Check which study is running
if ps aux | grep -q "[r]un_gpu_vm_long_training"; then
    echo "RUNNING=long_training"
elif ps aux | grep -q "[r]un_gpu_vm_full_study_resume"; then
    echo "RUNNING=resume_study"
elif ps aux | grep -q "[r]un_gpu_vm_full_study"; then
    echo "RUNNING=full_study"
elif ps aux | grep -q "[r]un_gpu_vm_long_training\|[r]un_gpu_vm_rerun"; then
    echo "RUNNING=long_or_rerun"
elif ps aux | grep -q "[t]rain_bert\|[t]rain_llm\|[t]rain_xgb\|[t]rain_knn\|[t]rain_mlp\|[t]rain_lstm"; then
    echo "RUNNING=training_process"
else
    echo "RUNNING=none"
fi

# GPU memory
echo "GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | head -1 | tr -d " ")"

# Latest log entries
for log in /tmp/long_training.log /tmp/full_study_resume.log /tmp/full_study.log; do
    if [ -f "$log" ]; then
        echo "LATEST_LOG=$log"
        echo "LAST_TIMESTAMP=$(grep -E "^\[" "$log" | tail -1)"
        echo "FAILURES=$(grep -c "FAILED" "$log" 2>/dev/null)"
        break
    fi
done

# Model dirs and sizes
echo "STUDY_SIZE=$(du -sh models/route_balance/study/ 2>/dev/null | cut -f1)"
echo "EARLY_SIZE=$(du -sh models/route_balance/early_study/ 2>/dev/null | cut -f1)"
echo "LONG_SIZE=$(du -sh models/route_balance/long_study/ 2>/dev/null | cut -f1)"

# Count checkpoints that should be cleaned
CKPT_COUNT=$(find models/route_balance -type d -name "checkpoint-*" 2>/dev/null | wc -l)
echo "CHECKPOINT_DIRS=$CKPT_COUNT"

# Completed experiments
echo "COMPLETED_MODELS=$(find models/route_balance/study models/route_balance/long_study -name "training_results.json" -o -name "training_metrics.json" 2>/dev/null | wc -l)"
' 2>&1)

echo "$STATUS" >> "$LOGFILE"

# Parse status
DISK_FREE=$(echo "$STATUS" | grep "DISK_FREE=" | cut -d= -f2)
RUNNING=$(echo "$STATUS" | grep "RUNNING=" | cut -d= -f2)
CKPT_COUNT=$(echo "$STATUS" | grep "CHECKPOINT_DIRS=" | cut -d= -f2)

# Action: clean checkpoints if any exist (always safe — best model is at top level)
if [ -n "$CKPT_COUNT" ] && [ "$CKPT_COUNT" -gt 0 ]; then
    echo "$(timestamp) Cleaning old checkpoint dirs (keeping last per model)..." >> "$LOGFILE"
    ssh -o ConnectTimeout=10 "$REMOTE" 'cd ~/Code/llm/RouteBalance && for model_dir in models/route_balance/study/*/  models/route_balance/long_study/*/ models/route_balance/early_study/*/; do
        ckpts=$(ls -d "$model_dir"checkpoint-* 2>/dev/null | sort -V)
        count=$(echo "$ckpts" | grep -c "checkpoint" 2>/dev/null || echo 0)
        if [ "$count" -gt 1 ]; then
            echo "$ckpts" | head -n -1 | xargs rm -rf 2>/dev/null
        fi
    done
    find ~/Code/llm/RouteBalance/models/route_balance -name "optimizer.pt" -not -path "*/checkpoint-*/*" -delete 2>/dev/null
    find ~/Code/llm/RouteBalance/models/route_balance -name "rng_state*.pth" -not -path "*/checkpoint-*/*" -delete 2>/dev/null' 2>/dev/null
    echo "$(timestamp) Checkpoints cleaned (kept last per model)" >> "$LOGFILE"
fi

# Alert + action: disk critically low — clean HF cache for already-downloaded models
if [ -n "$DISK_FREE" ] && [ "$DISK_FREE" -lt 20 ]; then
    echo "$(timestamp) CRITICAL: Disk very low — ${DISK_FREE}GB free. Cleaning HF cache..." >> "$LOGFILE"
    ssh -o ConnectTimeout=10 "$REMOTE" 'python3 -c "
from huggingface_hub import scan_cache_dir
cache = scan_cache_dir()
print(f\"HF cache: {cache.size_on_disk / 1e9:.1f}GB\")
# Only delete if cache > 2GB
if cache.size_on_disk > 2e9:
    for repo in cache.repos:
        for rev in repo.revisions:
            print(f\"  Deleting {repo.repo_id} rev {rev.commit_hash[:8]}...\")
    strategy = cache.delete_revisions(*[rev.commit_hash for repo in cache.repos for rev in repo.revisions])
    strategy.execute()
    print(\"HF cache cleaned\")
" 2>&1' >> "$LOGFILE" 2>&1
elif [ -n "$DISK_FREE" ] && [ "$DISK_FREE" -lt 100 ]; then
    echo "$(timestamp) WARNING: Disk low — ${DISK_FREE}GB free" >> "$LOGFILE"
fi

# Action: if nothing running, launch next study in pipeline:
# 1. Resume study → 2. Rerun failed → 3. Long training → 4. All done (collect results)
if [ "$RUNNING" = "none" ]; then
    echo "$(timestamp) No training running — checking pipeline stage" >> "$LOGFILE"

    # Check what's been completed
    RERUN_DONE=$(ssh -o ConnectTimeout=10 "$REMOTE" 'ls ~/Code/llm/RouteBalance/models/route_balance/study/qwen05b_fused_judge/adapter_config.json 2>/dev/null && echo yes || echo no' 2>&1)
    LONG_DONE=$(ssh -o ConnectTimeout=10 "$REMOTE" 'ls ~/Code/llm/RouteBalance/models/route_balance/long_study/modernbert_fused_length_mse/training_results.json 2>/dev/null && echo yes || echo no' 2>&1)

    if [ "$RERUN_DONE" = "no" ]; then
        echo "$(timestamp) Launching failed experiment re-runs..." >> "$LOGFILE"
        ssh -o ConnectTimeout=10 "$REMOTE" 'cd ~/Code/llm/RouteBalance && nohup bash route_balance/exp/route_balance/run_gpu_vm_rerun_failed.sh > /tmp/rerun_failed.log 2>&1 &' 2>/dev/null
        echo "$(timestamp) Re-runs launched" >> "$LOGFILE"
    elif [ "$LONG_DONE" = "no" ]; then
        # Check lock file to prevent double-launch
        LONG_LAUNCHED=$(ssh -o ConnectTimeout=10 "$REMOTE" 'ls /tmp/long_training_launched 2>/dev/null && echo yes || echo no' 2>&1)
        if [ "$LONG_LAUNCHED" = "no" ]; then
            echo "$(timestamp) Launching long training study..." >> "$LOGFILE"
            ssh -o ConnectTimeout=10 "$REMOTE" 'cd ~/Code/llm/RouteBalance && touch /tmp/long_training_launched && nohup bash route_balance/exp/route_balance/run_gpu_vm_long_training.sh > /tmp/long_training.log 2>&1 &' 2>/dev/null
            echo "$(timestamp) Long training launched" >> "$LOGFILE"
        else
            echo "$(timestamp) Long training already launched (lock exists), waiting..." >> "$LOGFILE"
        fi
    else
        # Check if judge retries done
        JUDGE_DONE=$(ssh -o ConnectTimeout=10 "$REMOTE" 'ls ~/Code/llm/RouteBalance/models/route_balance/study/modernbert_fused_judge_class/training_results.json 2>/dev/null && echo yes || echo no' 2>&1)
        if [ "$JUDGE_DONE" = "no" ]; then
            echo "$(timestamp) Long training done. Launching judge_class retries..." >> "$LOGFILE"
            ssh -o ConnectTimeout=10 "$REMOTE" 'cd ~/Code/llm/RouteBalance && nohup bash route_balance/exp/route_balance/run_gpu_vm_judge_retry.sh > /tmp/judge_retry.log 2>&1 &' 2>/dev/null
            echo "$(timestamp) Judge retries launched" >> "$LOGFILE"
        else
            # Check if final eval done
            EVAL_DONE=$(ssh -o ConnectTimeout=10 "$REMOTE" 'ls ~/Code/llm/RouteBalance/models/route_balance/evaluation/eval_length.json 2>/dev/null && echo yes || echo no' 2>&1)
            if [ "$EVAL_DONE" = "no" ]; then
                echo "$(timestamp) All training done. Launching final evaluation..." >> "$LOGFILE"
                ssh -o ConnectTimeout=10 "$REMOTE" 'cd ~/Code/llm/RouteBalance && nohup bash route_balance/exp/route_balance/run_gpu_vm_final_eval.sh > /tmp/final_eval.log 2>&1 &' 2>/dev/null
                echo "$(timestamp) Final evaluation launched" >> "$LOGFILE"
            else
                echo "$(timestamp) ALL COMPLETE (training + evaluation)" >> "$LOGFILE"

                # Collect all results
                ssh -o ConnectTimeout=10 "$REMOTE" 'cd ~/Code/llm/RouteBalance && find models/route_balance/evaluation -name "*.json" 2>/dev/null | sort | while read f; do echo "=== $f ==="; cat "$f" | python3 -c "import sys,json; d=json.load(sys.stdin); print(json.dumps(d,indent=2)[:1000])" 2>/dev/null; done' >> "$LOGFILE" 2>&1

                echo "$(timestamp) All results collected. Ready for analysis." >> "$LOGFILE"
            fi
        fi
    fi
fi

echo "$(timestamp) === Monitor done ===" >> "$LOGFILE"
