#!/bin/bash
# Quick test of baseline analysis framework
# Tests on carpet with existing checkpoint

echo "🧪 Testing Baseline Analysis Framework"
echo "======================================="
echo ""

# Check if checkpoint exists
CHECKPOINT="./result/test/semantic_memory_split/mvtec/k_2/checkpoints/carpet.pth"

if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ Checkpoint not found: $CHECKPOINT"
    echo "Please run training first:"
    echo "  bash bash/run_test_semantic_memory.sh"
    exit 1
fi

echo "✅ Checkpoint found"
echo ""
echo "Running baseline analysis on carpet..."
echo ""

python test_baseline_analysis.py \
    --dataset mvtec \
    --class_name carpet \
    --k-shot 2 \
    --root-dir ./result/test \
    --seed 111 \
    --gpu-id 0 \
    --batch-size 50 \
    --resolution 256

echo ""
echo "======================================="
echo "Test completed!"
echo ""
echo "Check outputs at:"
echo "  result/test/baseline/mvtec/k_2/seed_111/results/"
echo "======================================="
