#!/bin/bash

# Test script for purged prompts (Tier 1 & Tier 2)
# Only test the 6 MVTec classes that had Dangerous-and-Useless prompts removed

OUTPUT_DIR="./result/baseline_reducedprompt2"
K_SHOT=2
GPU_ID=0

echo "=========================================="
echo "Testing Purged Prompts (6 MVTec Classes)"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

# Tier 1 classes (必做)
declare -a TIER1_CLASSES=("metal_nut" "pill" "cable")

# Tier 2 classes (推荐)
declare -a TIER2_CLASSES=("screw" "capsule" "transistor")

# Combine all classes
ALL_CLASSES=("${TIER1_CLASSES[@]}" "${TIER2_CLASSES[@]}")

# Test each class
for cls in "${ALL_CLASSES[@]}"
do
    echo ""
    echo "----------------------------------------"
    echo "Testing: mvtec-$cls (k=$K_SHOT)"
    echo "----------------------------------------"
    
    python test_cls.py \
        --dataset mvtec \
        --class_name $cls \
        --k-shot $K_SHOT \
        --gpu-id $GPU_ID \
        --root-dir $OUTPUT_DIR
    
    if [ $? -eq 0 ]; then
        echo "✅ $cls completed"
    else
        echo "❌ $cls failed"
    fi
done

echo ""
echo "=========================================="
echo "✅ All tests completed!"
echo "Results: $OUTPUT_DIR/mvtec/k_$K_SHOT/csv/Seed_111-results.csv"
echo "=========================================="

