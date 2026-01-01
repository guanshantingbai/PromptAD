#!/bin/bash

# Phase 2 批量重新推理脚本（基于 Phase 1 清洗后的权重）
# 只对 6 个目标类别进行推理

DATASET=$1
K_SHOT=$2
SEED=$3
TARGET_CLASS=$4  # 可选：指定单个类别

# Phase 2 目标类别
TIER1_CLASSES="metal_nut pill cable"
TIER2_CLASSES="screw capsule transistor"
ALL_CLASSES="$TIER1_CLASSES $TIER2_CLASSES"

# 如果指定了单个类别，只运行该类别
if [ -n "$TARGET_CLASS" ]; then
    CLASSES=$TARGET_CLASS
else
    CLASSES=$ALL_CLASSES
fi

echo "======================================================================"
echo "Phase 2: Re-inference with Class-Specific Cleaning"
echo "======================================================================"
echo "Dataset: $DATASET"
echo "K-shot: $K_SHOT"
echo "Seed: $SEED"
echo "Classes: $(echo $CLASSES | wc -w)"
echo ""
echo "Note: Using Phase 1 cleaned prompts + Phase 2 class-specific cleaning"
echo "======================================================================"
echo ""

# 开始时间
START_TIME=$(date +%s)

# 逐类别运行
CLASS_COUNT=0
for CLASS in $CLASSES; do
    CLASS_COUNT=$((CLASS_COUNT + 1))
    TOTAL_CLASSES=$(echo $CLASSES | wc -w)
    
    echo ""
    echo "======================================================================"
    echo "[$CLASS_COUNT/$TOTAL_CLASSES] Processing: $CLASS"
    echo "======================================================================"
    
    # 检查权重文件（使用 baseline 权重，因为只是推理）
    WEIGHT_PATH="result/baseline/${DATASET}/k_${K_SHOT}/checkpoint/CLS-Seed_${SEED}-${CLASS}-check_point.pt"
    
    if [ ! -f "$WEIGHT_PATH" ]; then
        echo "❌ 权重文件不存在: $WEIGHT_PATH"
        echo "   跳过类别: $CLASS"
        continue
    fi
    
    echo "✓ Found weights: $WEIGHT_PATH"
    echo "  Running inference with Phase 2 cleaned prompts..."
    
    # 运行推理（使用 Phase 1 的测试脚本，因为配置相同）
    python test_with_cleaned_prompts.py \
        --dataset $DATASET \
        --class_name $CLASS \
        --k-shot $K_SHOT \
        --seed $SEED \
        --gpu-id 0 \
        --output-dir result/phase2_cleaned
    
    if [ $? -eq 0 ]; then
        echo "✓ Inference completed for $CLASS"
    else
        echo "❌ Inference failed for $CLASS"
    fi
done

# 结束时间
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "======================================================================"
echo "✅ Phase 2 Re-inference Complete"
echo "======================================================================"
echo "Total classes: $(echo $CLASSES | wc -w)"
echo "Time elapsed: ${MINUTES}m ${SECONDS}s"
echo "Output directory: result/phase2_cleaned/${DATASET}/k_${K_SHOT}/Seed_${SEED}"
echo ""
echo "Next steps:"
echo "  1. Compare results with Phase 1 baseline"
echo "  2. Check AUROC improvements"
echo "  3. Analyze per-class changes"
echo "======================================================================"
