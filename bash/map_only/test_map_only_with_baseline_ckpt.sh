#!/bin/bash

# Test MAP-only performance using baseline checkpoints (post-hoc evaluation)
# Usage: bash bash/map_only/test_map_only_with_baseline_ckpt.sh <dataset> <k_shot> <gpu_id>
# Example: bash bash/map_only/test_map_only_with_baseline_ckpt.sh mvtec 2 0

DATASET=$1
K_SHOT=$2
GPU_ID=$3

if [ -z "$DATASET" ] || [ -z "$K_SHOT" ] || [ -z "$GPU_ID" ]; then
    echo "Usage: bash $0 <dataset> <k_shot> <gpu_id>"
    echo "Example: bash $0 mvtec 2 0"
    exit 1
fi

# Define class lists
if [ "$DATASET" == "mvtec" ]; then
    CLASSES="bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper"
elif [ "$DATASET" == "visa" ]; then
    CLASSES="candle capsules cashew chewinggum fryum macaroni1 macaroni2 pcb1 pcb2 pcb3 pcb4 pipe_fryum"
else
    echo "Unknown dataset: $DATASET"
    exit 1
fi

# Create output directories
OUTPUT_DIR="./result/map_only_posthoc/${DATASET}/k_${K_SHOT}"
mkdir -p ${OUTPUT_DIR}/csv
mkdir -p logs/map_only_posthoc/${DATASET}_k${K_SHOT}

echo "==========================================="
echo "MAP-only Post-hoc Evaluation (Baseline Checkpoints)"
echo "Dataset: $DATASET"
echo "K-shot: $K_SHOT"
echo "GPU: $GPU_ID"
echo "Output: $OUTPUT_DIR"
echo "==========================================="

# Test each class
for CLASS in $CLASSES; do
    echo ""
    echo "Testing MAP-only: $CLASS"
    
    # Test with MAP-only (use-lap=False), load from baseline checkpoint
    python test_cls.py \
        --dataset $DATASET \
        --class_name $CLASS \
        --k-shot $K_SHOT \
        --seed 111 \
        --use-lap False \
        --checkpoint-dir "./result/baseline/${DATASET}/k_${K_SHOT}" \
        --root-dir "$OUTPUT_DIR" \
        --gpu-id $GPU_ID \
        2>&1 | tee logs/map_only_posthoc/${DATASET}_k${K_SHOT}/${CLASS}.log
    
    echo "  Completed: $CLASS"
done

echo ""
echo "==========================================="
echo "MAP-only Post-hoc Testing Complete!"
echo "Results saved to: $OUTPUT_DIR/csv"
echo "Logs saved to: logs/map_only_posthoc/${DATASET}_k${K_SHOT}/"
echo "==========================================="
echo ""
echo "Next step: Run comparison analysis"
echo "  bash bash/map_only/compare_map_vs_baseline.sh $DATASET $K_SHOT"
