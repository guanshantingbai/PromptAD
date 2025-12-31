#!/bin/bash
# Train MAP-only models (without LAP) for comparison

set -e

DATASET=${1:-"visa"}
K_SHOT=${2:-2}
GPU_ID=${3:-0}

echo "=========================================="
echo "Training MAP-only Models"
echo "=========================================="
echo "Dataset: ${DATASET}"
echo "K-shot: ${K_SHOT}"
echo "GPU: ${GPU_ID}"
echo "Output: ./result/map_only/${DATASET}/k_${K_SHOT}"
echo "=========================================="
echo ""

# Get class list
if [ "$DATASET" = "mvtec" ]; then
    CLASSES="bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper"
elif [ "$DATASET" = "visa" ]; then
    CLASSES="candle capsules cashew chewinggum fryum macaroni1 macaroni2 pcb1 pcb2 pcb3 pcb4 pipe_fryum"
else
    echo "Unknown dataset: ${DATASET}"
    exit 1
fi

# Create output directory
mkdir -p logs/map_only

# Train each class
COUNT=0
TOTAL=$(echo $CLASSES | wc -w)

for CLASS in $CLASSES; do
    COUNT=$((COUNT + 1))
    echo "----------------------------------------"
    echo "[${COUNT}/${TOTAL}] Training: ${CLASS}"
    echo "----------------------------------------"
    
    python train_cls.py \
        --dataset ${DATASET} \
        --class_name ${CLASS} \
        --k-shot ${K_SHOT} \
        --gpu-id ${GPU_ID} \
        --root-dir ./result/map_only \
        --use-lap False \
        > logs/map_only/k${K_SHOT}_${DATASET}_${CLASS}.log 2>&1
    
    echo "✅ Completed: ${CLASS}"
    echo ""
done

echo "=========================================="
echo "All training completed!"
echo "=========================================="
echo "Checkpoints: ./result/map_only/${DATASET}/k_${K_SHOT}/checkpoint/"
echo "Logs: logs/map_only/"
echo ""
echo "Next step: Run comparison analysis"
echo "  bash bash/map_only/compare_map_vs_baseline.sh ${DATASET} ${K_SHOT}"
