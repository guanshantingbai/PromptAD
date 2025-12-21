#!/bin/bash
# Test all MVTec classes using baseline k=2 checkpoints with diagnostics

DATASET="mvtec"
K_SHOT=2
SEED=111

# MVTec classes
CLASSES=(
    "bottle" "cable" "capsule" "carpet" "grid"
    "hazelnut" "leather" "metal_nut" "pill" "screw"
    "tile" "toothbrush" "transistor" "wood" "zipper"
)

echo "=========================================="
echo "Testing Baseline Checkpoints (k=${K_SHOT})"
echo "=========================================="
echo ""

for class_name in "${CLASSES[@]}"; do
    echo "--------------------------------------"
    echo "Testing: ${class_name}"
    echo "--------------------------------------"
    
    python test_baseline_checkpoint.py \
        --dataset ${DATASET} \
        --class_name ${class_name} \
        --k-shot ${K_SHOT} \
        --seed ${SEED} \
        --n_pro 1 \
        --n_pro_ab 4 \
        --save-diagnostics True \
        --resolution 400 \
        --batch-size 32 \
        --gpu-id 0
    
    echo ""
done

echo "=========================================="
echo "All tests completed!"
echo "Results saved to: result/baseline_results/"
echo "Diagnostics saved to: result/baseline_diagnostics/"
echo "=========================================="
