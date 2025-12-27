#!/bin/bash
"""
快速测试 - 6个代表性类别的扩展评估
用于验证流程，完整运行使用 run_extended_evaluation.sh
"""

echo "========================================================================"
echo "扩展评估快速测试 - 6个代表性类别"
echo "========================================================================"
echo ""

# 选择6个代表性类别（涵盖不同性能组）
test_classes=(
    "mvtec:carpet:Stable"           # Stable组
    "mvtec:capsule:Severe"          # Severe退化
    "mvtec:screw:Improved"          # 唯一改进类
    "mvtec:toothbrush:Severe"       # Severe退化
    "visa:candle:Mild"              # Mild退化
    "visa:macaroni1:Stable"         # Stable组
)

total=${#test_classes[@]}
current=0

for entry in "${test_classes[@]}"; do
    IFS=':' read -r dataset cls group <<< "$entry"
    current=$((current + 1))
    
    echo "[$current/$total] $dataset-$cls ($group)"
    
    python evaluate_extended_metrics.py \
        --dataset $dataset \
        --class_name $cls \
        --k-shot 2 \
        --n_pro 1 \
        --n_pro_ab 4 \
        --version prompt2 \
        --gpu-id 0 \
        > logs/extended_eval/${dataset}_${cls}_k2.log 2>&1
    
    if [ $? -eq 0 ]; then
        echo "   ✅ 完成"
    else
        echo "   ❌ 失败"
    fi
    echo ""
done

echo "========================================================================"
echo "✅ 快速测试完成！"
echo "========================================================================"
echo ""
echo "现在运行汇总:"
echo "  python aggregate_extended_metrics.py"
