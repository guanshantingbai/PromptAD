#!/bin/bash
# 监控k=4训练进度

LOG_FILE="logs/retrain_all_cls_k4.log"
CSV_FILE="retrain_all_cls_k4_execution.csv"

echo "=========================================="
echo "k=4训练监控"
echo "=========================================="

# 检查进程
PID=$(cat logs/retrain_all_cls_k4.pid 2>/dev/null)
if [ -n "$PID" ]; then
    if ps -p $PID > /dev/null 2>&1; then
        echo "✅ 训练进程运行中 (PID: $PID)"
    else
        echo "❌ 训练进程已结束"
    fi
else
    echo "⚠️  未找到PID文件"
fi

echo ""
echo "最新日志 (最后20行):"
echo "----------------------------------------"
tail -20 $LOG_FILE 2>/dev/null || echo "日志文件不存在"

echo ""
echo "=========================================="
echo "按Ctrl+C退出监控"
echo "完整日志: tail -f $LOG_FILE"
echo "=========================================="
