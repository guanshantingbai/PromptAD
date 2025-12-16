#!/bin/bash
################################################################################
# CPU 负载监控脚本
# 每 5 秒记录一次 CPU 和 GPU 状态
################################################################################

LOG_FILE="cpu_monitor_$(date +%Y%m%d_%H%M%S).log"

echo "=========================================="
echo "CPU & GPU 监控"
echo "日志文件: $LOG_FILE"
echo "按 Ctrl+C 停止监控"
echo "=========================================="
echo ""

# 写入表头
{
    echo "=========================================="
    echo "监控开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="
    echo ""
} > "$LOG_FILE"

while true; do
    {
        echo "────────────────────────────────────────"
        echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
        echo ""
        
        # 系统负载
        echo "=== 系统负载 ==="
        uptime
        echo ""
        
        # 训练进程 CPU 占用
        echo "=== 训练进程 CPU 使用率 ==="
        ps aux | head -1
        ps aux | grep -E "python.*(train_cls|train_seg)" | grep -v grep | head -10
        echo ""
        
        # CPU 总体使用情况
        echo "=== CPU 总体使用 ==="
        top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print "CPU 总使用率: " 100 - $1 "%"}'
        echo ""
        
        # GPU 使用情况
        echo "=== GPU 状态 ==="
        nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits | \
            awk -F', ' '{printf "GPU %s: 利用率 %s%%, 显存 %s/%s MB (%.1f%%)\n", $1, $2, $4, $5, $4/$5*100}'
        echo ""
        
        # 统计分析
        echo "=== 实时统计 ==="
        TRAIN_PROCS=$(ps aux | grep -E "python.*(train_cls|train_seg)" | grep -v grep | wc -l)
        TOTAL_CPU=$(ps aux | grep -E "python.*(train_cls|train_seg)" | grep -v grep | awk '{sum += $3} END {print (sum == "" ? 0 : sum)}')
        TOTAL_MEM=$(ps aux | grep -E "python.*(train_cls|train_seg)" | grep -v grep | awk '{sum += $4} END {print (sum == "" ? 0 : sum)}')
        echo "训练进程数: $TRAIN_PROCS"
        if [ "$TRAIN_PROCS" -gt 0 ]; then
            echo "总 CPU 占用: ${TOTAL_CPU}% (相当于 $(echo "scale=1; $TOTAL_CPU/100" | bc) 个核心满载)"
            echo "总内存占用: ${TOTAL_MEM}%"
        else
            echo "总 CPU 占用: 0% (无训练进程运行)"
        fi
        echo ""
        
    } | tee -a "$LOG_FILE"
    
    sleep 5
done
