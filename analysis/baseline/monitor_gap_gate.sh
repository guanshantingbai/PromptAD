#!/bin/bash

# Monitor gap-gating batch job progress

LOG_FILE="gap_gate_visa.log"

if [ ! -f "$LOG_FILE" ]; then
    echo "Log file not found: $LOG_FILE"
    exit 1
fi

echo "======================================================================"
echo "Gap-Gating Batch Job Monitor"
echo "======================================================================"
echo ""

# Count completed classes
n_completed=$(grep -c "✓ Completed:" "$LOG_FILE" 2>/dev/null || echo "0")
echo "Completed classes: ${n_completed} / 12"
echo ""

# Show last completed class
last_completed=$(grep "✓ Completed:" "$LOG_FILE" | tail -1)
if [ ! -z "$last_completed" ]; then
    echo "Last completed: $last_completed"
    echo ""
fi

# Show current class being processed
current_class=$(grep "Class:" "$LOG_FILE" | tail -1)
if [ ! -z "$current_class" ]; then
    echo "Current: $current_class"
    echo ""
fi

# Show latest AUROC results
echo "Latest results:"
grep "Gap-Gated" "$LOG_FILE" | tail -3
echo ""

# Check if job finished
if grep -q "Gap-Gating Validation Completed!" "$LOG_FILE" 2>/dev/null; then
    echo "======================================================================"
    echo "✓ Batch job COMPLETED!"
    echo "======================================================================"
else
    echo "Status: Running..."
    echo "Monitor in real-time: tail -f $LOG_FILE"
fi
