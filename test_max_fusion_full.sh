#!/bin/bash

# Max Fusion 完整实验脚本 - 测试所有类别（并行数=2）
echo "============================================================"
echo "Max Fusion vs Harmonic Mean 完整对比实验"
echo "============================================================"
echo ""
echo "实验设置:"
echo "  - Dataset: MVTec AD (15类)"
echo "  - k-shot: 4"
echo "  - seed: 111"
echo "  - Epochs: 100"
echo "  - 并行数: 2 (经过验证的安全设置)"
echo ""
echo "============================================================"

# 创建结果目录
mkdir -p result/max_fusion/mvtec/k_4/csv
mkdir -p result/max_fusion/logs

# 定义所有类别
classes="carpet grid leather tile wood bottle cable capsule hazelnut metal_nut pill screw toothbrush transistor zipper"

# 并行数设置
MAX_PARALLEL=2

# 并行运行函数（带checkpoint跳过机制）
run_parallel() {
    local task_type=$1
    local script_name=$2
    local log_file=$3
    local task_prefix=$4  # CLS 或 SEG
    
    echo ""
    echo "开始 $task_type 任务 (并行数=$MAX_PARALLEL, 支持断点续传)"
    echo "============================================================"
    
    local count=0
    local skipped=0
    local running=0
    
    for class_name in $classes; do
        # 检查checkpoint是否存在
        local checkpoint="result/max_fusion/mvtec/k_4/checkpoint/${task_prefix}-Seed_111-${class_name}-check_point.pt"
        
        if [ -f "$checkpoint" ]; then
            echo "[$class_name] ✓ 已完成 (checkpoint存在)，跳过"
            ((skipped++))
            continue
        fi
        
        echo "[$class_name] 启动 $task_type 任务..."
        
        conda run -n prompt_ad python $script_name \
            --class_name $class_name \
            --k-shot 4 \
            --seed 111 \
            --Epoch 100 \
            --log-interval 10 \
            --resume true \
            2>&1 | tee -a $log_file &
        
        ((count++))
        ((running++))
        
        # 每启动MAX_PARALLEL个任务后等待
        if [ $((running % MAX_PARALLEL)) -eq 0 ]; then
            echo "等待当前批次完成..."
            wait
            echo "当前批次完成，继续下一批次"
            running=0
        fi
    done
    
    # 等待最后一批任务完成
    if [ $running -gt 0 ]; then
        echo "等待最后一批任务完成..."
        wait
    fi
    
    echo ""
    echo "$task_type 任务完成！"
    echo "  - 新运行: $count 个"
    echo "  - 已跳过: $skipped 个"
    echo "============================================================"
}

# 分类任务
run_parallel "分类" "train_cls_max_fusion.py" "result/max_fusion/logs/train_cls_all.log" "CLS"

# 分割任务
run_parallel "分割" "train_seg_max_fusion.py" "result/max_fusion/logs/train_seg_all.log" "SEG"

echo ""
echo "============================================================"
echo "完整实验完成！"
echo "============================================================"
echo ""
echo "结果文件："
echo "  - CSV: result/max_fusion/mvtec/k_4/csv/Seed_111-results.csv"
echo "  - 日志: result/max_fusion/logs/"
echo ""
echo "对比分析："
echo "  python analyze_max_fusion.py"
echo ""
echo "============================================================"
