#!/bin/bash

# ======================================================================
# Gap-Based Semantic Gating Validation Script - MVTec Dataset
# ======================================================================
# 
# Purpose: Validate gap-based semantic suppression on MVTec (15 classes)
# Note: Based on gap screening, most MVTec classes have weak gap informativeness
# (only toothbrush is "usable", most are "disable")
#
# Experiment Design:
#   1. Run baseline test (no gating) to get reference AUROC
#   2. Run gated test (with gap-based semantic suppression)
#   3. Compare: Baseline vs Gated fusion AUROC
#
# Output:
#   - Individual results: ./result/semantic_gate1/mvtec/k_2/
#   - Comparison CSV: will be aggregated by separate script
# ======================================================================

# Don't use set -e to allow continuation even if some tests fail
# set -e

# Configuration
DATASET="mvtec"
K_SHOT=2
SEED=111
GPU_ID=0
GAP_BETA=10.0  # Default beta for sigmoid gate

# MVTec classes (15 classes)
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
echo "Gap-Based Gating Validation - MVTec Dataset"
echo "======================================================================"
echo "Dataset: ${DATASET}"
echo "K-shot: ${K_SHOT}"
echo "Seed: ${SEED}"
echo "Gap Beta: ${GAP_BETA}"
echo "Total classes: ${#MVTEC_CLASSES[@]}"
echo ""
echo "⚠️  Note: Based on gap screening, most MVTec classes have weak gap"
echo "    informativeness. Expect neutral or negative delta for most classes."
echo "    Only toothbrush (Gap AUC=0.71) is expected to potentially benefit."
echo "======================================================================"
echo ""

# Function to run gated test only (baseline results already exist)
run_gated_test() {
    local class_name=$1
    
    echo "----------------------------------------------------------------------"
    echo "[GAP-GATED] Processing: ${DATASET} - ${class_name}"
    echo "----------------------------------------------------------------------"
    
    cd ../..  # Return to project root
    
    # Always use ./result/baseline as root_dir (for loading checkpoint)
    # Gated results will be automatically saved to ./result/semantic_gate1
    python test_cls.py \
        --dataset ${DATASET} \
        --class_name ${class_name} \
        --root-dir ./result/baseline \
        --k-shot ${K_SHOT} \
        --gpu-id ${GPU_ID} \
        --use-lap True \
        --seed ${SEED} \
        --enable-gap-gate True \
        --gap-beta ${GAP_BETA}
    
    cd analysis/baseline
    
    echo ""
}

# Track progress
completed=0
total=${#MVTEC_CLASSES[@]}

for class_name in "${MVTEC_CLASSES[@]}"; do
    echo ""
    echo "======================================================================"
    echo "Progress: $((completed + 1))/${total}"
    echo "Class: ${class_name}"
    echo "======================================================================"
    echo ""
    
    # Only run gated test (baseline results already exist in ./result/baseline)
    echo "[INFO] Loading baseline checkpoint from ./result/baseline"
    echo "[INFO] Saving gated results to ./result/semantic_gate1"
    run_gated_test ${class_name}
    
    ((completed++))
    echo "✓ Completed: ${class_name} (${completed}/${total})"
    echo ""
done

echo ""
echo "======================================================================"
echo "Gap-Gating Validation Completed!"
echo "======================================================================"
echo ""
echo "Results saved in:"
echo "  - Baseline:   ./result/baseline/${DATASET}/k_${K_SHOT}/"
echo "  - Gap-Gated:  ./result/semantic_gate1/${DATASET}/k_${K_SHOT}/"
echo ""
echo "Next step: Run comparison analysis script to aggregate results"
echo "======================================================================"
