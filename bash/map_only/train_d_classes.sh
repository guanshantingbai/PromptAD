#!/bin/bash

# Train MAP-only models for classes with LAP harmful (failure_type D)
# Usage: bash bash/map_only/train_d_classes.sh <gpu_id>

GPU_ID=${1:-0}
K_SHOT=2
DATASET="mvtec"
OUTPUT_DIR="./result/map_only"

echo "=========================================="
echo "Training MAP-only for LAP Harmful Classes"
echo "Dataset: $DATASET, k-shot: $K_SHOT, GPU: $GPU_ID"
echo "=========================================="

# D classes: carpet, tile, wood, grid
D_CLASSES=("carpet" "tile" "wood" "grid")

for CLASS in "${D_CLASSES[@]}"; do
    echo ""
    echo ">>> Training MAP-only for $CLASS"
    python train_cls.py \
        --dataset $DATASET \
        --class_name $CLASS \
        --k-shot $K_SHOT \
        --use-lap False \
        --gpu-id $GPU_ID \
        --Epoch 100 \
        2>&1 | tee logs/map_only/${DATASET}_${CLASS}_k${K_SHOT}.log
    
    echo ">>> Finished $CLASS"
done

echo ""
echo "=========================================="
echo "All MAP-only training completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="
