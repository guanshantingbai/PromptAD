#!/bin/bash
# Batch baseline analysis after training with run_cls.py
# Usage: bash bash/baseline/batch_analysis.sh [dataset] [k_shot] [checkpoint_dir]

DATASET=${1:-mvtec}
K_SHOT=${2:-2}
CHECKPOINT_DIR=${3:-./result/${DATASET}/k_${K_SHOT}/checkpoints}

echo "======================================"
echo "Batch Baseline Analysis"
echo "======================================"
echo "Dataset: $DATASET"
echo "K-shot: $K_SHOT"
echo "Checkpoint dir: $CHECKPOINT_DIR"
echo "======================================"
echo ""

# Check if checkpoint directory exists
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "❌ Checkpoint directory not found: $CHECKPOINT_DIR"
    echo ""
    echo "Please train models first using:"
    echo "  python run_cls.py"
    exit 1
fi

# Get all checkpoint files
CHECKPOINTS=$(ls $CHECKPOINT_DIR/*.pth 2>/dev/null)

if [ -z "$CHECKPOINTS" ]; then
    echo "❌ No checkpoints found in $CHECKPOINT_DIR"
    exit 1
fi

echo "Found checkpoints:"
echo "$CHECKPOINTS"
echo ""

# Run analysis for each checkpoint
for CHECKPOINT in $CHECKPOINTS; do
    # Extract class name from filename (e.g., carpet.pth -> carpet)
    CLASS_NAME=$(basename $CHECKPOINT .pth)
    
    echo "----------------------------------------"
    echo "Analyzing: $CLASS_NAME"
    echo "----------------------------------------"
    
    python test_baseline_analysis.py \
        --dataset $DATASET \
        --class_name $CLASS_NAME \
        --k-shot $K_SHOT \
        --checkpoint-path $CHECKPOINT \
        --root-dir ./result \
        --seed 111 \
        --gpu-id 0 \
        --batch-size 100 \
        --resolution 400
    
    if [ $? -eq 0 ]; then
        echo "✅ $CLASS_NAME completed"
    else
        echo "❌ $CLASS_NAME failed"
    fi
    echo ""
done

echo "======================================"
echo "Batch analysis completed!"
echo "Results saved to: result/baseline_analysis/$DATASET/k_$K_SHOT/seed_111/results/"
echo "======================================"
