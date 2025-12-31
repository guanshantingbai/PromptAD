#!/bin/bash

# Training script for purged prompts (Tier 1 & Tier 2)
# Train the 6 MVTec classes with reduced prompts

OUTPUT_DIR="./result/baseline_reducedprompt2"
K_SHOT=2
GPU_ID=0

echo "=========================================="
echo "Training Purged Prompts (6 MVTec Classes)"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

# Tier 1 classes (必做)
declare -a TIER1_CLASSES=("metal_nut" "pill" "cable")

# Tier 2 classes (推荐)
declare -a TIER2_CLASSES=("screw" "capsule" "transistor")

# Combine all classes
ALL_CLASSES=("${TIER1_CLASSES[@]}" "${TIER2_CLASSES[@]}")

# Train each class
for cls in "${ALL_CLASSES[@]}"
do
    echo ""
    echo "----------------------------------------"
    echo "Training: mvtec-$cls (k=$K_SHOT)"
    echo "----------------------------------------"
    
    python train_cls.py \
        --dataset mvtec \
        --class_name $cls \
        --k-shot $K_SHOT \
        --gpu-id $GPU_ID \
        --root-dir $OUTPUT_DIR
    
    if [ $? -eq 0 ]; then
        echo "✅ $cls training completed"
    else
        echo "❌ $cls training failed"
    fi
done

echo ""
echo "=========================================="
echo "✅ All training completed!"
echo "Checkpoints: $OUTPUT_DIR/mvtec/k_$K_SHOT/checkpoint/"
echo "Results: $OUTPUT_DIR/mvtec/k_$K_SHOT/csv/Seed_111-results.csv"
echo "=========================================="
