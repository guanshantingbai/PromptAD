#!/bin/bash

################################################################################
# PromptAD 实验进度监控脚本
################################################################################

ROOT_DIR="result/backbone1"
LOG_DIR="$ROOT_DIR/logs"
MAIN_LOG="$LOG_DIR/run_all_experiments.log"

echo "=========================================="
echo "PromptAD 实验进度监控"
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
echo ""

# 检查主日志是否存在
if [ ! -f "$MAIN_LOG" ]; then
    echo "❌ 主日志文件不存在: $MAIN_LOG"
    echo "实验可能尚未开始"
    exit 1
fi

# GPU 运行状态
echo "📊 GPU 运行状态："
echo "---"
for gpu in 1 2 3; do
    count=$(pgrep -f "train_.*\.py.*--gpu-id ${gpu}" 2>/dev/null | wc -l)
    if [ "$count" -gt 0 ]; then
        echo "GPU $gpu: ✅ $count 个任务运行中"
        # 显示任务简要信息
        pgrep -af "train_.*\.py.*--gpu-id ${gpu}" 2>/dev/null | head -2 | while read line; do
            if echo "$line" | grep -q "train_cls.py"; then
                class=$(echo "$line" | grep -o "class_name [^ ]*" | awk '{print $2}')
                config=$(echo "$line" | grep -o "exp_config [^ ]*" | awk '{print $2}')
                dataset=$(echo "$line" | grep -o "dataset [^ ]*" | awk '{print $2}')
                echo "  └─ CLS: $dataset/$class ($config)"
            elif echo "$line" | grep -q "train_seg.py"; then
                class=$(echo "$line" | grep -o "class_name [^ ]*" | awk '{print $2}')
                config=$(echo "$line" | grep -o "exp_config [^ ]*" | awk '{print $2}')
                dataset=$(echo "$line" | grep -o "dataset [^ ]*" | awk '{print $2}')
                echo "  └─ SEG: $dataset/$class ($config)"
            fi
        done
    else
        echo "GPU $gpu: ⏸️  空闲"
    fi
done
echo ""

# 任务统计
echo "📈 任务统计："
echo "---"
if [ -f "$MAIN_LOG" ]; then
    total_start=$(grep -c "\[START\]" "$MAIN_LOG" 2>/dev/null || echo 0)
    total_success=$(grep -c "\[SUCCESS\]" "$MAIN_LOG" 2>/dev/null || echo 0)
    total_failed=$(grep -c "\[FAILED\]" "$MAIN_LOG" 2>/dev/null || echo 0)
    total_skip=$(grep -c "\[SKIP\]" "$MAIN_LOG" 2>/dev/null || echo 0)
    
    echo "已启动: $total_start"
    echo "已完成: $total_success ✅"
    echo "已失败: $total_failed ❌"
    echo "已跳过: $total_skip ⏭️"
    
    # 计算完成率
    if [ "$total_start" -gt 0 ]; then
        completed=$((total_success + total_failed + total_skip))
        running=$((total_start - completed))
        percent=$((completed * 100 / total_start))
        echo "进行中: $running 🔄"
        echo "完成率: $percent%"
    fi
else
    echo "主日志文件不存在"
fi
echo ""

# 最近的任务状态
echo "🕒 最近任务（最后10条）："
echo "---"
if [ -f "$MAIN_LOG" ]; then
    tail -20 "$MAIN_LOG" | grep -E "\[(START|SUCCESS|FAILED|SKIP)\]" | tail -10
else
    echo "无日志记录"
fi
echo ""

# 锁文件状态
echo "🔒 锁文件状态："
echo "---"
if [ -d "$ROOT_DIR/locks" ]; then
    lock_count=$(ls -1 "$ROOT_DIR/locks"/*.lock 2>/dev/null | wc -l)
    if [ "$lock_count" -gt 0 ]; then
        echo "存在 $lock_count 个锁文件"
        ls -lh "$ROOT_DIR/locks"/*.lock 2>/dev/null
    else
        echo "无锁文件"
    fi
else
    echo "锁目录不存在"
fi
echo ""

# Checkpoint 统计
echo "💾 Checkpoint 统计："
echo "---"
if [ -d "$ROOT_DIR" ]; then
    mvtec_ck=$(find "$ROOT_DIR/mvtec" -name "*_check_point.pt" 2>/dev/null | wc -l)
    visa_ck=$(find "$ROOT_DIR/visa" -name "*_check_point.pt" 2>/dev/null | wc -l)
    echo "MVTec checkpoints: $mvtec_ck"
    echo "VisA checkpoints: $visa_ck"
    echo "总计: $((mvtec_ck + visa_ck))"
else
    echo "结果目录不存在"
fi
echo ""

# 磁盘使用
echo "💽 磁盘使用："
echo "---"
if [ -d "$ROOT_DIR" ]; then
    size=$(du -sh "$ROOT_DIR" 2>/dev/null | awk '{print $1}')
    echo "结果目录大小: $size"
else
    echo "结果目录不存在"
fi
echo ""

# 实时日志尾部
echo "📝 主日志尾部（最后5行）："
echo "---"
if [ -f "$MAIN_LOG" ]; then
    tail -5 "$MAIN_LOG"
else
    echo "无日志记录"
fi

echo ""
echo "=========================================="
echo "提示："
echo "  - 查看完整日志: tail -f $MAIN_LOG"
echo "  - 监控 GPU: watch -n 1 nvidia-smi"
echo "  - 查看特定任务: tail -f $LOG_DIR/cls_mvtec_bottle_*.log"
echo "=========================================="
