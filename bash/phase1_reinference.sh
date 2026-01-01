#!/bin/bash
# Phase 1: 使用清洗后的 prompt table 重新推理

DATASET=${1:-"mvtec"}
K_SHOT=${2:-2}
SEED=${3:-111}
SINGLE_CLASS=${4:-""}  # 可选：只运行单个类别

# 定义数据集类别
if [ "$DATASET" == "mvtec" ]; then
    ALL_CLASSES="bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper"
elif [ "$DATASET" == "visa" ]; then
    ALL_CLASSES="candle capsules cashew chewinggum fryum macaroni1 macaroni2 pcb1 pcb2 pcb3 pcb4 pipe_fryum"
else
    echo "❌ Unknown dataset: $DATASET"
    exit 1
fi

# 如果指定了单个类别，只运行该类别
if [ -n "$SINGLE_CLASS" ]; then
    CLASSES="$SINGLE_CLASS"
    echo "Running single class: $SINGLE_CLASS"
else
    CLASSES="$ALL_CLASSES"
fi

echo "======================================================================"
echo "Phase 1: Re-inference with Cleaned Prompts"
echo "======================================================================"
echo "Dataset: $DATASET"
echo "K-shot: $K_SHOT"
echo "Seed: $SEED"
echo "Classes: $(echo $CLASSES | wc -w)"
echo ""
echo "Note: Using cleaned prompts from prompts/manual_prompts_master_table.csv"
echo "      (6 global problematic prompts disabled)"
echo "======================================================================"
echo ""

# 输出目录
OUTPUT_DIR="result/phase1_cleaned/${DATASET}/k_${K_SHOT}/Seed_${SEED}"
mkdir -p "$OUTPUT_DIR"

# 记录开始时间
START_TIME=$(date +%s)

# 计数器
TOTAL=$(echo $CLASSES | wc -w)
COUNT=0

# 遍历所有类别
for CLASS in $CLASSES; do
    COUNT=$((COUNT+1))
    
    echo ""
    echo "======================================================================"
    echo "[$COUNT/$TOTAL] Processing: $CLASS"
    echo "======================================================================"
    
    # 查找已训练的权重
    WEIGHT_PATH="result/baseline/${DATASET}/k_${K_SHOT}/checkpoint/CLS-Seed_${SEED}-${CLASS}-check_point.pt"
    
    if [ ! -f "$WEIGHT_PATH" ]; then
        echo "⚠️  Warning: Model weights not found at $WEIGHT_PATH"
        echo "   Skipping $CLASS"
        continue
    fi
    
    echo "✓ Found weights: $WEIGHT_PATH"
    echo "  Running inference with cleaned prompts..."
    
    # 使用专门的测试脚本（不重新训练）
    python test_with_cleaned_prompts.py \
        --dataset "$DATASET" \
        --class_name "$CLASS" \
        --k-shot "$K_SHOT" \
        --seed "$SEED" \
        --gpu-id 0 \
        2>&1 | tee "$OUTPUT_DIR/${CLASS}_inference.log"
    
    # 检查是否成功
    if [ $? -eq 0 ]; then
        echo "✓ Inference completed for $CLASS"
    else
        echo "❌ Inference failed for $CLASS"
    fi
done

# 计算总时间
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "======================================================================"
echo "✅ Phase 1 Re-inference Complete"
echo "======================================================================"
echo "Total classes: $TOTAL"
echo "Time elapsed: ${MINUTES}m ${SECONDS}s"
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "  1. Compare results with baseline (before cleaning)"
echo "  2. Check AUROC improvements"
echo "  3. Analyze per-class changes"
echo "======================================================================"
