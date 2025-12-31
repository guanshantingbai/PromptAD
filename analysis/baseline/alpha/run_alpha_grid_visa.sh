#!/bin/bash

# ======================================================================
# Alpha Grid Search for Semantic Weight - VisA Dataset
# ======================================================================
#
# Purpose: Find optimal alpha for semantic weight scaling
# Formula: semantic_new = alpha * semantic_original
#          fusion = 1 / (1/semantic_new + 1/memory)
#
# Alpha values: [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0]
#   - alpha < 1: Suppress semantic
#   - alpha = 1: Baseline (no change)
#   - alpha > 1: Amplify semantic
#
# Output: ./result/baseline/alpha/visa/k_2/csv/visa.csv
# ======================================================================

DATASET="visa"
K_SHOT=2
SEED=111
GPU_ID=0

VISA_CLASSES=(
    "candle"
    "capsules"
    "cashew"
    "chewinggum"
    "fryum"
    "macaroni1"
    "macaroni2"
    "pcb1"
    "pcb2"
    "pcb3"
    "pcb4"
    "pipe_fryum"
)

echo "======================================================================"
echo "Alpha Grid Search - VisA Dataset"
echo "======================================================================"
echo "Dataset: ${DATASET}"
echo "K-shot: ${K_SHOT}"
echo "Total classes: ${#VISA_CLASSES[@]}"
echo "Alpha values: [0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.1, 1.2]"
echo "======================================================================"
echo ""

cd ../..  # Go to project root

completed=0
total=${#VISA_CLASSES[@]}

for class_name in "${VISA_CLASSES[@]}"; do
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
