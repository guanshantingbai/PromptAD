#!/bin/bash
# 受控实验评估脚本
# 对比4个版本：Baseline, Prompt2, Ours_v1(全改动), Ours_v2(EMA+Rep)

echo "========================================================================"
echo "受控实验评估 - 4版本对比"
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

mkdir -p logs/controlled_eval
mkdir -p analysis/controlled_comparison

total=6
current=0

# 评估每个类别的4个版本
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
        > logs/controlled_eval/${dataset}_${cls}_baseline_k${K_SHOT}.log 2>&1
    
    if [ $? -eq 0 ]; then
        echo "     ✅ Baseline完成"
        mv analysis/extended_metrics/${dataset}_${cls}_k${K_SHOT}_split_auroc.csv \
           analysis/controlled_comparison/${dataset}_${cls}_baseline_split_auroc.csv 2>/dev/null
        mv analysis/extended_metrics/${dataset}_${cls}_k${K_SHOT}_margin_stats.csv \
           analysis/controlled_comparison/${dataset}_${cls}_baseline_margin_stats.csv 2>/dev/null
        mv analysis/extended_metrics/${dataset}_${cls}_k${K_SHOT}_semantic_contrib.csv \
           analysis/controlled_comparison/${dataset}_${cls}_baseline_semantic_contrib.csv 2>/dev/null
        mv analysis/extended_metrics/${dataset}_${cls}_k${K_SHOT}_sample_scores.csv \
           analysis/controlled_comparison/${dataset}_${cls}_baseline_sample_scores.csv 2>/dev/null
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
        > logs/controlled_eval/${dataset}_${cls}_prompt2_k${K_SHOT}.log 2>&1
    
    if [ $? -eq 0 ]; then
        echo "     ✅ Prompt2完成"
        mv analysis/extended_metrics/${dataset}_${cls}_k${K_SHOT}_split_auroc.csv \
           analysis/controlled_comparison/${dataset}_${cls}_prompt2_split_auroc.csv 2>/dev/null
        mv analysis/extended_metrics/${dataset}_${cls}_k${K_SHOT}_margin_stats.csv \
           analysis/controlled_comparison/${dataset}_${cls}_prompt2_margin_stats.csv 2>/dev/null
        mv analysis/extended_metrics/${dataset}_${cls}_k${K_SHOT}_semantic_contrib.csv \
           analysis/controlled_comparison/${dataset}_${cls}_prompt2_semantic_contrib.csv 2>/dev/null
        mv analysis/extended_metrics/${dataset}_${cls}_k${K_SHOT}_sample_scores.csv \
           analysis/controlled_comparison/${dataset}_${cls}_prompt2_sample_scores.csv 2>/dev/null
    else
        echo "     ❌ Prompt2失败"
    fi
    
    # (C) Ours_v1评估（全部改动：EMA+Rep+Margin）
    echo "  [C] Ours_v1 (EMA+Rep+Margin)..."
    python evaluate_extended_metrics.py \
        --dataset $dataset \
        --class_name $cls \
        --k-shot $K_SHOT \
        --n_pro 1 \
        --n_pro_ab 4 \
        --version ours_fix_ema_rep_margin \
        --seed $SEED \
        --gpu-id $GPU_ID \
        > logs/controlled_eval/${dataset}_${cls}_ours_v1_k${K_SHOT}.log 2>&1
    
    if [ $? -eq 0 ]; then
        echo "     ✅ Ours_v1完成"
        mv analysis/extended_metrics/${dataset}_${cls}_k${K_SHOT}_split_auroc.csv \
           analysis/controlled_comparison/${dataset}_${cls}_ours_v1_split_auroc.csv 2>/dev/null
        mv analysis/extended_metrics/${dataset}_${cls}_k${K_SHOT}_margin_stats.csv \
           analysis/controlled_comparison/${dataset}_${cls}_ours_v1_margin_stats.csv 2>/dev/null
        mv analysis/extended_metrics/${dataset}_${cls}_k${K_SHOT}_semantic_contrib.csv \
           analysis/controlled_comparison/${dataset}_${cls}_ours_v1_semantic_contrib.csv 2>/dev/null
        mv analysis/extended_metrics/${dataset}_${cls}_k${K_SHOT}_sample_scores.csv \
           analysis/controlled_comparison/${dataset}_${cls}_ours_v1_sample_scores.csv 2>/dev/null
    else
        echo "     ❌ Ours_v1失败"
    fi
    
    # (D) Ours_v2评估（受控实验：EMA+Rep，无Margin）
    echo "  [D] Ours_v2 (EMA+Rep only)..."
    python evaluate_extended_metrics.py \
        --dataset $dataset \
        --class_name $cls \
        --k-shot $K_SHOT \
        --n_pro 1 \
        --n_pro_ab 4 \
        --version ema_rep_only \
        --seed $SEED \
        --gpu-id $GPU_ID \
        > logs/controlled_eval/${dataset}_${cls}_ours_v2_k${K_SHOT}.log 2>&1
    
    if [ $? -eq 0 ]; then
        echo "     ✅ Ours_v2完成"
        mv analysis/extended_metrics/${dataset}_${cls}_k${K_SHOT}_split_auroc.csv \
           analysis/controlled_comparison/${dataset}_${cls}_ours_v2_split_auroc.csv 2>/dev/null
        mv analysis/extended_metrics/${dataset}_${cls}_k${K_SHOT}_margin_stats.csv \
           analysis/controlled_comparison/${dataset}_${cls}_ours_v2_margin_stats.csv 2>/dev/null
        mv analysis/extended_metrics/${dataset}_${cls}_k${K_SHOT}_semantic_contrib.csv \
           analysis/controlled_comparison/${dataset}_${cls}_ours_v2_semantic_contrib.csv 2>/dev/null
        mv analysis/extended_metrics/${dataset}_${cls}_k${K_SHOT}_sample_scores.csv \
           analysis/controlled_comparison/${dataset}_${cls}_ours_v2_sample_scores.csv 2>/dev/null
    else
        echo "     ❌ Ours_v2失败"
    fi
    
    echo ""
done

echo "========================================================================"
echo "✅ 受控实验评估完成！"
echo "========================================================================"
echo ""
echo "结果保存在: analysis/controlled_comparison/"
echo ""
echo "下一步: 生成受控实验对比报告"
echo "  python analyze_controlled_experiment.py"
