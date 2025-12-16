#!/bin/bash
################################################################################
# 高频 CPU 监控（1秒间隔，捕捉初始化峰值）
################################################################################

LOG_FILE="cpu_monitor_highfreq_$(date +%Y%m%d_%H%M%S).log"

echo "高频监控启动 (1秒间隔)"
echo "日志: $LOG_FILE"
echo "按 Ctrl+C 停止"
echo ""

{
    echo "监控开始: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
} > "$LOG_FILE"

while true; do
    {
        TIMESTAMP=$(date '+%H:%M:%S')
        TRAIN_PROCS=$(ps aux | grep -E "python.*(train_cls|train_seg)" | grep -v grep | wc -l)
        TOTAL_CPU=$(ps aux | grep -E "python.*(train_cls|train_seg)" | grep -v grep | awk '{sum += $3} END {print (sum == "" ? 0 : sum)}')
        
        printf "%s | 进程数: %d | CPU: %.1f%% (%.1f核)\n" \
            "$TIMESTAMP" "$TRAIN_PROCS" "$TOTAL_CPU" "$(echo "scale=1; $TOTAL_CPU/100" | bc)"
        
    } | tee -a "$LOG_FILE"
    
    sleep 1
done
