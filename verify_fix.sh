#!/bin/bash
################################################################################
# 快速验证修复后的代码
################################################################################

echo "=========================================="
echo "验证数据类型兼容性修复"
echo "=========================================="
echo ""

# 测试 num_workers=0（原始模式）
echo "测试 1: num_workers=0 (无 transform 优化)"
python train_cls.py \
    --exp_config qq_residual \
    --class_name bottle \
    --dataset mvtec \
    --k-shot 2 \
    --Epoch 2 \
    --batch-size 8 \
    --gpu-id 0 \
    --num-workers 0 \
    --backbone ViT-B-16 \
    --pretrained_dataset laion400m_e32

if [ $? -eq 0 ]; then
    echo "✅ num_workers=0 测试通过"
else
    echo "❌ num_workers=0 测试失败"
    exit 1
fi

echo ""
echo "测试 2: num_workers=2 (多进程加载)"
python train_cls.py \
    --exp_config qq_residual \
    --class_name bottle \
    --dataset mvtec \
    --k-shot 2 \
    --Epoch 2 \
    --batch-size 8 \
    --gpu-id 0 \
    --num-workers 2 \
    --backbone ViT-B-16 \
    --pretrained_dataset laion400m_e32

if [ $? -eq 0 ]; then
    echo "✅ num_workers=2 测试通过"
else
    echo "❌ num_workers=2 测试失败"
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ 所有测试通过！代码修复成功"
echo "=========================================="
