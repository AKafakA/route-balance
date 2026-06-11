#!/bin/bash
# Collect all experiment results and monitor CSVs from all nodes to coordinator
#
# Usage: bash route_balance/exp/route_balance/collect/collect_results.sh [output_dir]
set -euo pipefail

OUTPUT_DIR=${1:-"experiment_output/collected"}
HOSTS_FILE="route_balance/config/hosts"
COORDINATOR=$(head -1 $HOSTS_FILE)

mkdir -p $OUTPUT_DIR/monitor

echo "Collecting results to $OUTPUT_DIR — $(date)"

# Collect monitor CSVs from all nodes
while IFS= read -r host; do
    [ -z "$host" ] && continue
    hostname=$(echo $host | sed 's/.*@//' | cut -d. -f1)
    echo "  Collecting monitor from $hostname..."
    scp -o StrictHostKeyChecking=no "$host:~/Block/experiment_output/monitor/*.csv" \
      "$OUTPUT_DIR/monitor/" 2>/dev/null || echo "    (no monitor data)"
done < "$HOSTS_FILE"

# Collect experiment results from coordinator
echo "  Collecting experiment results..."
for dir in e2e sensitivity ablation comprehensive_smoketest filter_validation; do
    if ssh -o StrictHostKeyChecking=no $COORDINATOR "test -d ~/Block/experiment_output/$dir" 2>/dev/null; then
        mkdir -p $OUTPUT_DIR/$dir
        scp -r -o StrictHostKeyChecking=no "$COORDINATOR:~/Block/experiment_output/$dir/*.json" \
          "$OUTPUT_DIR/$dir/" 2>/dev/null && echo "    $dir: $(ls $OUTPUT_DIR/$dir/*.json 2>/dev/null | wc -l) files" \
          || echo "    $dir: no results"
    fi
done

# Aggregate monitor data
echo ""
echo "Aggregating monitor data..."
python3 route_balance/exp/route_balance/aggregate_monitor.py --input-dir $OUTPUT_DIR/monitor

echo ""
echo "Collection COMPLETE — $OUTPUT_DIR/"
find $OUTPUT_DIR -name "*.json" -o -name "*.csv" | wc -l
echo "total files"
