#!/bin/bash
# GPU监控和自动启动测试脚本
# 等待GPU空闲后自动开始批量测试

echo "GPU监控和自动测试启动器"
echo "========================================"
echo "启动时间: $(date)"
echo ""

# 配置参数
GPU_ID=0                    # GPU编号
MEMORY_THRESHOLD=1000       # 显存阈值(MB)，低于此值认为空闲
CHECK_INTERVAL=60           # 检查间隔(秒)
MAX_WAIT_TIME=7200          # 最大等待时间(秒) = 2小时

# 检查GPU使用情况
check_gpu_idle() {
    # 获取GPU显存使用(MB)
    local memory_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU_ID)
    
    if [ $memory_used -lt $MEMORY_THRESHOLD ]; then
        return 0  # GPU空闲
    else
        return 1  # GPU忙碌
    fi
}

# 主循环
elapsed_time=0
while [ $elapsed_time -lt $MAX_WAIT_TIME ]; do
    echo -n "$(date '+%H:%M:%S') - 检查GPU状态... "
    
    if check_gpu_idle; then
        echo "✓ GPU空闲！"
        echo ""
        echo "========================================"
        echo "开始执行批量测试"
        echo "========================================"
        
        # 执行批量测试
        ./run_batch_tests.sh
        
        exit 0
    else
        # 获取详细GPU信息
        memory_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU_ID)
        utilization=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $GPU_ID)
        
        echo "✗ GPU忙碌 (显存: ${memory_used}MB, 利用率: ${utilization}%)"
        
        # 等待
        sleep $CHECK_INTERVAL
        elapsed_time=$((elapsed_time + CHECK_INTERVAL))
        
        # 显示剩余等待时间
        remaining=$((MAX_WAIT_TIME - elapsed_time))
        remaining_min=$((remaining / 60))
        echo "   等待中... (剩余最大等待时间: ${remaining_min}分钟)"
    fi
done

echo ""
echo "========================================"
echo "超时退出 - GPU持续忙碌超过${MAX_WAIT_TIME}秒"
echo "请手动检查GPU状态并重新运行"
echo "========================================"
exit 1
