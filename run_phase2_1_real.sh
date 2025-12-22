#!/bin/bash

# Phase 2.1: Run 5 representative classes with REAL indicator computation
# This script will compute per-sample reliability indicators for selected classes

DEVICE="cuda:0"
OUTPUT_DIR="result/gate2"
CHECKPOINT_DIR="result/gate"  # Use existing checkpoints

echo "=============================================="
echo "Phase 2.1: REAL Indicator Computation"
echo "=============================================="
echo ""
echo "Strategy:"
echo "  - Use existing checkpoints from result/gate"
echo "  - Compute per-sample reliability indicators"
echo "  - Save to result/gate2/*/per_sample/*.json"
echo ""
echo "Selected classes (5):"
echo "  1. mvtec/screw     (oracle gain: +33.82%)"
echo "  2. mvtec/capsule   (oracle gain: +7.94%)"
echo "  3. visa/capsules   (oracle gain: +29.72%)"
echo "  4. visa/macaroni2  (oracle gain: +26.72%)"
echo "  5. visa/candle     (oracle gain: +4.89%)"
echo ""
echo "Estimated time: 30-60 minutes (no training, only inference)"
echo "=============================================="
echo ""

# Classes to run
declare -a CLASSES=(
    "mvtec screw"
    "mvtec capsule"
    "visa capsules"
    "visa macaroni2"
    "visa candle"
)

K_SHOT=4
TASK="cls"

# Run each class
for class_info in "${CLASSES[@]}"; do
    IFS=' ' read -r dataset class_name <<< "$class_info"
    
    echo ""
    echo "=========================================="
    echo "Processing: ${dataset}/${class_name}"
    echo "=========================================="
    
    python run_gate_experiment.py \
        --dataset $dataset \
        --class_name $class_name \
        --k-shot $K_SHOT \
        --task $TASK \
        --gpu-id 0 \
        --root-dir $OUTPUT_DIR \
        --checkpoint-dir $CHECKPOINT_DIR \
        --seed 111 \
        --backbone ViT-B-16-plus-240 \
        --pretrained_dataset laion400m_e32
    
    if [ $? -eq 0 ]; then
        echo "✓ ${dataset}/${class_name} completed"
    else
        echo "✗ ${dataset}/${class_name} failed"
        exit 1
    fi
done

echo ""
echo "=============================================="
echo "Phase 2.1 Real Computation Complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "1. Run analysis: python phase2_1_oracle_correlation.py --use-real-data --dataset mvtec visa --k-shot 4 --task cls"
echo "2. Check results in result/gate2/*/per_sample/"
echo "3. Review AUC scores to decide next phase"
echo ""
