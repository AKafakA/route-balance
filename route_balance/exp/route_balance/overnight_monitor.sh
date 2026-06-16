#!/bin/bash
# Overnight monitor for Vast training + evaluation
# Run via: nohup bash route_balance/exp/route_balance/overnight_monitor.sh > /tmp/monitor_cron.log 2>&1 &
# Checks every 30 min, logs to OVERNIGHT_MONITOR.md

LOG="${HOME}/Block/route_balance_paper/claude/OVERNIGHT_MONITOR.md"
INTERVAL=1800  # 30 min

# Initialize log
cat > "$LOG" << 'HEADER'
# Overnight Monitor Log — March 30, 2026

## Monitoring: Vast (RoBERTa training + full evaluation) + CSD3 (LoRA per-model + ModernBERT long)
HEADER

check_vast() {
    echo ""
    echo "### $(date '+%Y-%m-%d %H:%M UTC') — Vast Check"
    echo ""

    # Check if any training/eval process is running
    PROCS=$(ssh -o ConnectTimeout=10 -o BatchMode=yes vast 'pgrep -af "train_fused\|run_evaluation\|train_llm" 2>/dev/null' 2>/dev/null)
    if [ -z "$PROCS" ]; then
        echo "**Status: No training/eval process running**"
        echo ""
        # Check if overnight script completed
        LAST=$(ssh -o ConnectTimeout=10 vast 'tail -5 /tmp/vast_overnight.log 2>/dev/null' 2>/dev/null)
        echo '```'
        echo "$LAST"
        echo '```'

        if echo "$LAST" | grep -q "ALL COMPLETE"; then
            echo ""
            echo "**EVALUATION COMPLETE — extracting results**"
            echo ""
            # Extract results
            for F in length bucket judge similarity reference_score; do
                RESULT=$(ssh -o ConnectTimeout=10 vast "cat /root/eval_results/${F}_results.json 2>/dev/null | python3 -c \"
import sys,json
try:
    data=json.load(sys.stdin)
    for r in data:
        print(f'  {r[\"name\"]}:')
        for m,v in r.get(\"per_model\",{}).items():
            ms=m.split(\"/\")[-1]
            mae=v.get(\"mae\",0)
            rho=v.get(\"spearman_r\",0)
            acc=v.get(\"bucket\",{}).get(\"accuracy\",\"\")
            if acc: print(f'    {ms}: MAE={mae:.3f}, ρ={rho:.3f}, bucket_acc={acc:.3f}')
            else: print(f'    {ms}: MAE={mae:.3f}, ρ={rho:.3f}')
except: print('  (parse error)')
\" 2>/dev/null" 2>/dev/null)
                if [ -n "$RESULT" ]; then
                    echo "#### ${F}"
                    echo '```'
                    echo "$RESULT"
                    echo '```'
                fi
            done
            return 0  # Signal completion
        fi

        # Check for errors
        ERRORS=$(ssh -o ConnectTimeout=10 vast 'grep -c "ERROR\|Failed\|Traceback" /tmp/vast_overnight.log 2>/dev/null' 2>/dev/null)
        if [ "$ERRORS" -gt 0 ] 2>/dev/null; then
            echo ""
            echo "**ERRORS FOUND ($ERRORS):**"
            echo '```'
            ssh -o ConnectTimeout=10 vast 'grep -A2 "ERROR\|Failed\|Traceback" /tmp/vast_overnight.log 2>/dev/null | tail -20' 2>/dev/null
            echo '```'
        fi
    else
        echo "**Status: Running**"
        echo '```'
        echo "$PROCS"
        echo '```'
        # Get latest progress
        echo ""
        echo "Latest output:"
        echo '```'
        ssh -o ConnectTimeout=10 vast 'grep -E "^---|MAE=|accuracy|PHASE|TARGET|COMPLETE" /tmp/vast_overnight.log 2>/dev/null | tail -10' 2>/dev/null
        echo '```'
    fi
    return 1  # Not complete
}

check_csd3() {
    echo ""
    echo "### $(date '+%Y-%m-%d %H:%M UTC') — CSD3 Check"
    echo ""

    QUEUE=$(ssh -o ConnectTimeout=10 -o BatchMode=yes csd3 'squeue -u ${CLOUDLAB_USER} 2>/dev/null' 2>/dev/null)
    if [ -z "$QUEUE" ]; then
        echo "**CSD3: SSH failed or no jobs**"
        return
    fi
    echo '```'
    echo "$QUEUE"
    echo '```'

    # Check for completed jobs
    COMPLETED=$(ssh -o ConnectTimeout=10 csd3 'sacct -u ${CLOUDLAB_USER} --starttime=2026-03-29 --format=JobID,JobName%15,State,Elapsed --noheader 2>/dev/null | grep -E "COMPLETED|FAILED" | grep -v ".batch\|.extern"' 2>/dev/null)
    if [ -n "$COMPLETED" ]; then
        echo ""
        echo "Completed/Failed jobs:"
        echo '```'
        echo "$COMPLETED"
        echo '```'
    fi
}

# Main loop
ROUND=0
while true; do
    ROUND=$((ROUND + 1))
    echo "" >> "$LOG"
    echo "---" >> "$LOG"
    echo "## Check #$ROUND" >> "$LOG"

    check_vast >> "$LOG" 2>&1
    VAST_DONE=$?

    check_csd3 >> "$LOG" 2>&1

    if [ $VAST_DONE -eq 0 ]; then
        echo "" >> "$LOG"
        echo "## Vast evaluation COMPLETE. Monitor stopping." >> "$LOG"
        echo "Final results extracted above." >> "$LOG"
        break
    fi

    sleep $INTERVAL
done

echo "" >> "$LOG"
echo "---" >> "$LOG"
echo "## Monitor ended at $(date '+%Y-%m-%d %H:%M UTC')" >> "$LOG"
