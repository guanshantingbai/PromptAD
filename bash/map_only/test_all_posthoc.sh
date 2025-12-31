#!/bin/bash

# Post-hoc MAP-only testing: Load baseline checkpoint, test with MAP-only
# Usage: bash bash/map_only/test_all_posthoc.sh <dataset> <k_shot> <gpu_id>

DATASET=${1:-mvtec}
K_SHOT=${2:-2}
GPU_ID=${3:-0}

echo "=========================================="
echo "Post-hoc MAP-only Testing"
echo "Dataset: $DATASET, k-shot: $K_SHOT, GPU: $GPU_ID"
echo "=========================================="

# MVTec classes
if [ "$DATASET" == "mvtec" ]; then
    CLASSES=(bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper)
# ViSA classes
elif [ "$DATASET" == "visa" ]; then
    CLASSES=(candle capsules cashew chewinggum fryum macaroni1 macaroni2 pcb1 pcb2 pcb3 pcb4 pipe_fryum)
else
    echo "Unknown dataset: $DATASET"
    exit 1
fi

for CLASS in "${CLASSES[@]}"; do
    echo ""
    echo ">>> Testing MAP-only (post-hoc) for $CLASS"
    python test_cls.py \
        --dataset $DATASET \
        --class_name $CLASS \
        --k-shot $K_SHOT \
        --seed 111 \
        --use-lap False \
        --root-dir "./result/baseline" \
        --gpu-id $GPU_ID \
        2>&1 | grep -E "(Object:|MAP-only)"
done

echo ""
echo "=========================================="
echo "All post-hoc testing completed!"
echo "=========================================="
