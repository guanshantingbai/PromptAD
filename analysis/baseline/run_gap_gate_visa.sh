#!/bin/bash

# ======================================================================
# Gap-Based Semantic Gating Validation Script - VisA Dataset Only
# ======================================================================
# 
# Purpose: Validate gap-based semantic suppression on VisA (all 12 classes)
# Based on gap screening results showing VisA has strong gap informativeness
# (All 12 VisA classes are "usable" with Gap AUC 0.78-0.92)
#
# Experiment Design:
#   1. Run baseline test (no gating) to get reference AUROC
#   2. Run gated test (with gap-based semantic suppression)
#   3. Compare: Baseline vs Gated fusion AUROC
#
# Output:
#   - Individual results: ./result/semantic_gate1/visa/k_2/
#   - Comparison CSV: will be aggregated by separate script
# ======================================================================

# Don't use set -e to allow continuation even if some tests fail
# set -e

# Configuration
DATASET="visa"
K_SHOT=2
SEED=111
GPU_ID=0
GAP_BETA=10.0  # Default beta for sigmoid gate

# VisA classes (all 12 are usable based on gap screening)
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
echo "Gap-Based Gating Validation - VisA Dataset"
echo "======================================================================"
echo "Dataset: ${DATASET}"
echo "K-shot: ${K_SHOT}"
echo "Seed: ${SEED}"
echo "Gap Beta: ${GAP_BETA}"
echo "Total classes: ${#VISA_CLASSES[@]}"
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
total=${#VISA_CLASSES[@]}

for class_name in "${VISA_CLASSES[@]}"; do
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
