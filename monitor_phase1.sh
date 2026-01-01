#!/bin/bash
# 监控 Phase 1 批量推理进度

LOG_FILE="phase1_reinference.log"
CSV_FILE="result/phase1_cleaned/mvtec/k_2/csv/Seed_111-results.csv"

echo "======================================================================"
echo "Phase 1 Re-inference Progress Monitor"
echo "======================================================================"
echo ""

# 检查日志文件
if [ ! -f "$LOG_FILE" ]; then
    echo "❌ Log file not found: $LOG_FILE"
    echo "   Is the script running?"
    exit 1
fi

echo "📊 Progress from log file:"
echo "------------------------------"
grep -E "\[[0-9]+/[0-9]+\] Processing:" "$LOG_FILE" | tail -3
echo ""

# 检查结果文件
if [ -f "$CSV_FILE" ]; then
    echo "📈 Current Results:"
    echo "------------------------------"
    cat "$CSV_FILE"
    echo ""
    
    COMPLETED=$(tail -n +2 "$CSV_FILE" | wc -l)
    echo "✅ Completed: $COMPLETED classes"
else
    echo "⚠️  No results file yet: $CSV_FILE"
fi

echo ""
echo "======================================================================"
echo "Recent log entries:"
echo "======================================================================"
tail -20 "$LOG_FILE"
