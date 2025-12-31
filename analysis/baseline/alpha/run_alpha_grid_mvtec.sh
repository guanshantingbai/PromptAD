#!/bin/bash

# ======================================================================
# Alpha Grid Search for Semantic Weight - MVTec Dataset
# ======================================================================

DATASET="mvtec"
K_SHOT=2
SEED=111
GPU_ID=0

MVTEC_CLASSES=(
    "bottle"
    "cable"
    "capsule"
    "carpet"
    "grid"
    "hazelnut"
    "leather"
    "metal_nut"
    "pill"
    "screw"
    "tile"
    "toothbrush"
    "transistor"
    "wood"
    "zipper"
)

echo "======================================================================"
echo "Alpha Grid Search - MVTec Dataset"
echo "======================================================================"
echo "Dataset: ${DATASET}"
echo "K-shot: ${K_SHOT}"
echo "Total classes: ${#MVTEC_CLASSES[@]}"
echo "Alpha values: [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0]"
echo "======================================================================"
echo ""

cd ../..  # Go to project root
 
completed=0
total=${#MVTEC_CLASSES[@]}

for class_name in "${MVTEC_CLASSES[@]}"; do
    echo ""
    echo "======================================================================"
    echo "Progress: $((completed + 1))/${total}"
    echo "Class: ${class_name}"
    echo "======================================================================"
    
    python test_cls_alpha_grid.py \
        --dataset ${DATASET} \
        --class_name ${class_name} \
        --k-shot ${K_SHOT} \
        --gpu-id ${GPU_ID} \
        --use-lap True \
        --seed ${SEED}
    
    ((completed++))
    echo "✓ Completed: ${class_name} (${completed}/${total})"
done

echo ""
echo "======================================================================"
echo "Alpha Grid Search Completed!"
echo "======================================================================"
echo ""
echo "Results saved in: ./result/baseline/alpha/${DATASET}/k_${K_SHOT}/csv/${DATASET}.csv"
echo ""
