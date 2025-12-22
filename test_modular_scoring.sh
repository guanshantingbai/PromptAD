#!/bin/bash
# Quick test of modular scoring framework

echo "=================================="
echo "Testing Modular Scoring Framework"
echo "=================================="

# Test on a single class with small batch for quick validation
python demo_modular_scoring.py \
    --dataset mvtec \
    --class_name carpet \
    --k-shot 4 \
    --task cls \
    --batch-size 16 \
    --gpu-id 0

echo ""
echo "=================================="
echo "Test Complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Check that oracle AUROC >= max AUROC >= {semantic, memory}"
echo "2. Verify component scores match when mode='semantic' or 'memory'"
echo "3. Compare max mode results with current baseline"
echo ""
