#!/bin/bash
# 测试 semantic 和 memory 分支隔离的实验脚本
# 输出目录: result/test/semantic_memory_split/

echo "======================================"
echo "Semantic & Memory Branch Split Test"
echo "======================================"
echo ""
echo "This experiment validates:"
echo "  1. Training loss ONLY depends on semantic branch (no memory bank influence)"
echo "  2. Evaluation computes 3 separate AUROCs: semantic, memory, fusion"
echo ""
echo "Running 5 epochs on carpet dataset..."
echo ""

python train_cls.py \
    --dataset mvtec \
    --class_name carpet \
    --Epoch 5 \
    --root-dir ./result/test/semantic_memory_split \
    --seed 111 \
    --gpu-id 0 \
    --k-shot 2 \
    --batch-size 100 \
    --resolution 256 \
    --vis False

echo ""
echo "======================================"
echo "Test completed!"
echo "Results saved to: result/test/semantic_memory_split/"
echo "======================================"
