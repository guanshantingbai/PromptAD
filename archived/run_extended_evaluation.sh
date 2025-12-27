#!/bin/bash
"""
批量执行扩展评估 - 27个类别
仅执行评估，不修改训练
"""

# MVTec classes
mvtec_classes=(
    "carpet" "grid" "leather" "tile" "wood"
    "bottle" "cable" "capsule" "hazelnut" "metal_nut"
    "pill" "screw" "toothbrush" "transistor" "zipper"
)

# VisA classes
visa_classes=(
    "candle" "capsules" "cashew" "chewinggum" "fryum"
    "macaroni1" "macaroni2" "pcb1" "pcb2" "pcb3"
    "pcb4" "pipe_fryum"
)

echo "========================================================================"
echo "扩展评估批量执行 - 27个类别 (k=2, n_pro=1)"
echo "========================================================================"
echo ""

# 创建输出目录
mkdir -p analysis/extended_metrics
mkdir -p logs/extended_eval

total_classes=27
current=0

# MVTec
for cls in "${mvtec_classes[@]}"; do
    current=$((current + 1))
    echo "[$current/$total_classes] MVTec - $cls"
    
    python evaluate_extended_metrics.py \
        --dataset mvtec \
        --class_name $cls \
        --k-shot 2 \
        --n_pro 1 \
        --n_pro_ab 4 \
        --version prompt2 \
        --gpu-id 0 \
        > logs/extended_eval/mvtec_${cls}_k2.log 2>&1
    
    if [ $? -eq 0 ]; then
        echo "   ✅ 完成"
    else
        echo "   ❌ 失败 (查看 logs/extended_eval/mvtec_${cls}_k2.log)"
    fi
    echo ""
done

# VisA
for cls in "${visa_classes[@]}"; do
    current=$((current + 1))
    echo "[$current/$total_classes] VisA - $cls"
    
    python evaluate_extended_metrics.py \
        --dataset visa \
        --class_name $cls \
        --k-shot 2 \
        --n_pro 1 \
        --n_pro_ab 4 \
        --version prompt2 \
        --gpu-id 0 \
        > logs/extended_eval/visa_${cls}_k2.log 2>&1
    
    if [ $? -eq 0 ]; then
        echo "   ✅ 完成"
    else
        echo "   ❌ 失败 (查看 logs/extended_eval/visa_${cls}_k2.log)"
    fi
    echo ""
done

echo "========================================================================"
echo "✅ 批量扩展评估完成！"
echo "========================================================================"
echo ""
echo "下一步: 运行汇总分析"
echo "  python aggregate_extended_metrics.py"
