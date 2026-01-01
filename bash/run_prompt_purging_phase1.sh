#!/bin/bash

# Prompt Purging Phase 1 批量运行脚本
# 分析所有类别的正常侧语义失配风险
# Checkpoint路径自动构建：result/baseline/{dataset}/k_{k_shot}/checkpoint/CLS-Seed_{seed}-{class}-check_point.pt

# 配置
K_SHOT=2
EPSILON=0.05
DEVICE=0
TASK=cls  # 使用分类任务（CLS）

echo "========================================"
echo "Prompt Purging Phase 1: Batch Analysis"
echo "========================================"
echo "K-shot: $K_SHOT"
echo "Task: $TASK"
echo "Epsilon: $EPSILON"
echo ""
echo "注意: Phase 1 不需要checkpoint"
echo "      Text prototypes通过prompts实时编码"
echo ""

# ===== MVTec-AD 数据集 =====
echo "Processing MVTec-AD classes..."

MVTEC_CLASSES=(
    "bottle" "cable" "capsule" "carpet" "grid"
    "hazelnut" "leather" "metal_nut" "pill" "screw"
    "tile" "toothbrush" "transistor" "wood" "zipper"
)

for class in "${MVTEC_CLASSES[@]}"; do
    echo ""
    echo "----------------------------------------"
    echo "MVTec: $class"
    echo "----------------------------------------"
    
    python prompt_purging_phase1.py \
        --dataset mvtec \
        --class "$class" \
        --k_shot $K_SHOT \
        --task $TASK \
        --epsilon $EPSILON \
        --device $DEVICE
    
    if [ $? -ne 0 ]; then
        echo "⚠ Warning: Failed for $class"
    fi
done

# ===== VisA 数据集 =====
echo ""
echo "Processing VisA classes..."

VISA_CLASSES=(
    "candle" "capsules" "cashew" "chewinggum"
    "fryum" "macaroni1" "macaroni2"
    "pcb1" "pcb2" "pcb3" "pcb4" "pipe_fryum"
)

for class in "${VISA_CLASSES[@]}"; do
    echo ""
    echo "----------------------------------------"
    echo "VisA: $class"
    echo "----------------------------------------"
    
    python prompt_purging_phase1.py \
        --dataset visa \
        --class "$class" \
        --k_shot $K_SHOT \
        --task $TASK \
        --epsilon $EPSILON \
        --device $DEVICE
    
    if [ $? -ne 0 ]; then
        echo "⚠ Warning: Failed for $class"
    fi
done

echo ""
echo "========================================"
echo "批量分析完成！"
echo "结果保存在: result/prompt_purging/phase1/{dataset}/k_{k_shot}/"
echo "========================================"
