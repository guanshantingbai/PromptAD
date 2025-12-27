#!/bin/bash
"""
6类对照实验评估脚本
对比三组：Baseline vs Prompt2 vs Ours
"""

echo "========================================================================"
echo "6类代表性类别 - 对照实验评估"
echo "========================================================================"
echo ""

# 定义6个代表性类别
declare -A class_info=(
    ["mvtec:toothbrush"]="Severe"
    ["mvtec:capsule"]="Severe"
    ["visa:pcb2"]="Severe"
    ["mvtec:carpet"]="Stable"
    ["mvtec:leather"]="Stable"
    ["mvtec:screw"]="Improved"
)

# 配置
K_SHOT=2
SEED=111
GPU_ID=0

mkdir -p logs/extended_eval_6class
mkdir -p analysis/6class_comparison

total=6
current=0

# 评估每个类别的三个版本
for class_key in "${!class_info[@]}"; do
    IFS=':' read -r dataset cls <<< "$class_key"
    group=${class_info[$class_key]}
    current=$((current + 1))
    
    echo "========================================================================"
    echo "[$current/$total] $dataset-$cls ($group)"
    echo "========================================================================"
    
    # (A) Baseline评估
    echo "  [A] Baseline..."
    python evaluate_extended_metrics.py \
        --dataset $dataset \
        --class_name $cls \
        --k-shot $K_SHOT \
        --n_pro 3 \
        --n_pro_ab 11 \
        --version baseline \
        --seed $SEED \
        --gpu-id $GPU_ID \
        > logs/extended_eval_6class/${dataset}_${cls}_baseline_k${K_SHOT}.log 2>&1
    
    if [ $? -eq 0 ]; then
        echo "     ✅ Baseline完成"
        # 重命名输出文件以区分版本
        mv analysis/extended_metrics/${dataset}_${cls}_k${K_SHOT}_split_auroc.csv \
           analysis/6class_comparison/${dataset}_${cls}_baseline_split_auroc.csv 2>/dev/null
        mv analysis/extended_metrics/${dataset}_${cls}_k${K_SHOT}_margin_stats.csv \
           analysis/6class_comparison/${dataset}_${cls}_baseline_margin_stats.csv 2>/dev/null
        mv analysis/extended_metrics/${dataset}_${cls}_k${K_SHOT}_semantic_contrib.csv \
           analysis/6class_comparison/${dataset}_${cls}_baseline_semantic_contrib.csv 2>/dev/null
        mv analysis/extended_metrics/${dataset}_${cls}_k${K_SHOT}_sample_scores.csv \
           analysis/6class_comparison/${dataset}_${cls}_baseline_sample_scores.csv 2>/dev/null
    else
        echo "     ❌ Baseline失败"
    fi
    
    # (B) Prompt2评估
    echo "  [B] Prompt2..."
    python evaluate_extended_metrics.py \
        --dataset $dataset \
        --class_name $cls \
        --k-shot $K_SHOT \
        --n_pro 1 \
        --n_pro_ab 4 \
        --version prompt2 \
        --seed $SEED \
        --gpu-id $GPU_ID \
        > logs/extended_eval_6class/${dataset}_${cls}_prompt2_k${K_SHOT}.log 2>&1
    
    if [ $? -eq 0 ]; then
        echo "     ✅ Prompt2完成"
        mv analysis/extended_metrics/${dataset}_${cls}_k${K_SHOT}_split_auroc.csv \
           analysis/6class_comparison/${dataset}_${cls}_prompt2_split_auroc.csv 2>/dev/null
        mv analysis/extended_metrics/${dataset}_${cls}_k${K_SHOT}_margin_stats.csv \
           analysis/6class_comparison/${dataset}_${cls}_prompt2_margin_stats.csv 2>/dev/null
        mv analysis/extended_metrics/${dataset}_${cls}_k${K_SHOT}_semantic_contrib.csv \
           analysis/6class_comparison/${dataset}_${cls}_prompt2_semantic_contrib.csv 2>/dev/null
        mv analysis/extended_metrics/${dataset}_${cls}_k${K_SHOT}_sample_scores.csv \
           analysis/6class_comparison/${dataset}_${cls}_prompt2_sample_scores.csv 2>/dev/null
    else
        echo "     ❌ Prompt2失败"
    fi
    
    # (C) Ours评估（需要先训练完成）
    echo "  [C] Ours (fix_ema_rep_margin)..."
    python evaluate_extended_metrics.py \
        --dataset $dataset \
        --class_name $cls \
        --k-shot $K_SHOT \
        --n_pro 1 \
        --n_pro_ab 4 \
        --version ours_fix_ema_rep_margin \
        --seed $SEED \
        --gpu-id $GPU_ID \
        > logs/extended_eval_6class/${dataset}_${cls}_ours_k${K_SHOT}.log 2>&1
    
    if [ $? -eq 0 ]; then
        echo "     ✅ Ours完成"
        mv analysis/extended_metrics/${dataset}_${cls}_k${K_SHOT}_split_auroc.csv \
           analysis/6class_comparison/${dataset}_${cls}_ours_split_auroc.csv 2>/dev/null
        mv analysis/extended_metrics/${dataset}_${cls}_k${K_SHOT}_margin_stats.csv \
           analysis/6class_comparison/${dataset}_${cls}_ours_margin_stats.csv 2>/dev/null
        mv analysis/extended_metrics/${dataset}_${cls}_k${K_SHOT}_semantic_contrib.csv \
           analysis/6class_comparison/${dataset}_${cls}_ours_semantic_contrib.csv 2>/dev/null
        mv analysis/extended_metrics/${dataset}_${cls}_k${K_SHOT}_sample_scores.csv \
           analysis/6class_comparison/${dataset}_${cls}_ours_sample_scores.csv 2>/dev/null
    else
        echo "     ❌ Ours失败"
    fi
    
    echo ""
done

echo "========================================================================"
echo "✅ 6类对照实验评估完成！"
echo "========================================================================"
echo ""
echo "结果保存在: analysis/6class_comparison/"
echo ""
echo "下一步: 生成汇总对比报告"
echo "  python aggregate_6class_comparison.py"
