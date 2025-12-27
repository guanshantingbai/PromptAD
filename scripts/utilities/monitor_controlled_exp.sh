#!/bin/bash
# 受控实验训练监控

echo "受控实验训练进度监控"
echo "========================================================================"

# 检查训练是否完成
if pgrep -f "train_6class_controlled.sh" > /dev/null; then
    echo "⏳ 训练进行中..."
else
    echo "✅ 训练已完成（或未启动）"
fi

echo ""
echo "各类别训练状态:"
echo "------------------------------------------------------------------------"

classes=(
    "mvtec:toothbrush"
    "mvtec:capsule"
    "mvtec:carpet"
    "mvtec:leather"
    "mvtec:screw"
    "visa:pcb2"
)

for class_key in "${classes[@]}"; do
    IFS=':' read -r dataset cls <<< "$class_key"
    log_file="logs/controlled_exp/${dataset}_${cls}_k2.log"
    
    if [ -f "$log_file" ]; then
        # 提取Image-AUROC
        auroc=$(grep "Image-AUROC:" "$log_file" | tail -1 | awk -F'Image-AUROC:' '{print $2}' | tr -d ' ')
        
        if [ -n "$auroc" ]; then
            echo "  ✅ $dataset-$cls: AUROC=$auroc"
        else
            echo "  ⏳ $dataset-$cls: 训练中..."
        fi
    else
        echo "  ⏸️  $dataset-$cls: 未开始"
    fi
done

echo ""
echo "========================================================================"

# 如果全部完成，提示下一步
completed_count=$(grep -l "Image-AUROC:" logs/controlled_exp/*_k2.log 2>/dev/null | wc -l)

if [ "$completed_count" -eq 6 ]; then
    echo "🎉 所有6个类别训练完成！"
    echo ""
    echo "下一步操作："
    echo "  1. 运行评估: ./evaluate_controlled_comparison.sh"
    echo "  2. 运行分析: python analyze_controlled_experiment.py"
fi
