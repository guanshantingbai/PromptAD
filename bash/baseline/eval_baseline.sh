#!/bin/bash
# Baseline evaluation script with comprehensive margin analysis
# Usage: bash bash/baseline/eval_baseline.sh [class_name] [dataset] [k_shot]

CLASS_NAME=${1:-carpet}
DATASET=${2:-mvtec}
K_SHOT=${3:-2}
GPU_ID=${4:-0}

echo "======================================"
echo "PromptAD Baseline Analysis"
echo "======================================"
echo "Dataset: $DATASET"
echo "Class: $CLASS_NAME"
echo "K-shot: $K_SHOT"
echo "======================================"
echo ""

python test_baseline_analysis.py \
    --dataset $DATASET \
    --class_name $CLASS_NAME \
    --k-shot $K_SHOT \
    --root-dir ./result \
    --seed 111 \
    --gpu-id $GPU_ID \
    --batch-size 100 \
    --resolution 400

echo ""
echo "======================================"
echo "Analysis completed!"
echo "Results saved to: result/baseline/$DATASET/k_$K_SHOT/seed_111/results/"
echo "======================================"
