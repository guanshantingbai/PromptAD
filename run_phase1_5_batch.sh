#!/bin/bash

# Batch run Phase 1.5: Prompt Classification
# 对所有 MVTec 和 VisA 类别运行

DATASET=$1
K_SHOT=${2:-2}

if [ "$DATASET" == "mvtec" ]; then
    CLASSES="bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper"
elif [ "$DATASET" == "visa" ]; then
    CLASSES="candle capsules cashew chewinggum fryum macaroni1 macaroni2 pcb1 pcb2 pcb3 pcb4 pipe_fryum"
else
    echo "Usage: bash run_phase1_5_batch.sh <mvtec|visa> [k_shot]"
    exit 1
fi

echo "======================================================================"
echo "Running Phase 1.5: Prompt Classification"
echo "Dataset: $DATASET"
echo "K-shot: $K_SHOT"
echo "======================================================================"

for class in $CLASSES; do
    echo ""
    echo "----------------------------------------------------------------------"
    echo "Processing: $class"
    echo "----------------------------------------------------------------------"
    
    python prompt_purging_phase1_5.py \
        --dataset $DATASET \
        --class $class \
        --k_shot $K_SHOT \
        --batch_size 8 \
        --device 0
    
    if [ $? -ne 0 ]; then
        echo "✗ Failed: $class"
    else
        echo "✓ Completed: $class"
    fi
done

echo ""
echo "======================================================================"
echo "Phase 1.5 Batch Processing Complete!"
echo "======================================================================"
echo "Results saved in: result/prompt_purging/phase1_5/$DATASET/k_$K_SHOT/"
