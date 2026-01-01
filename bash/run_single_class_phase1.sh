#!/bin/bash

# Prompt Purging Phase 1 - 单类别测试脚本
# 用于快速测试单个类别

# 使用示例:
# bash bash/run_single_class_phase1.sh mvtec bottle
# bash bash/run_single_class_phase1.sh visa candle

DATASET=$1
CLASS=$2

if [ -z "$DATASET" ] || [ -z "$CLASS" ]; then
    echo "用法: bash bash/run_single_class_phase1.sh <dataset> <class>"
    echo ""
    echo "示例:"
    echo "  bash bash/run_single_class_phase1.sh mvtec bottle"
    echo "  bash bash/run_single_class_phase1.sh visa candle"
    exit 1
fi

# 配置
K_SHOT=2
EPSILON=0.05
DEVICE=0
TASK=cls

echo "========================================"
echo "Prompt Purging Phase 1: Single Class"
echo "========================================"
echo "Dataset: $DATASET"
echo "Class: $CLASS"
echo "K-shot: $K_SHOT"
echo "Task: $TASK"
echo "Epsilon: $EPSILON"
echo ""
echo "注意: Phase 1 不需要checkpoint"
echo "      Text prototypes通过prompts实时编码"
echo ""

# 运行分析
python prompt_purging_phase1.py \
    --dataset "$DATASET" \
    --class "$CLASS" \
    --k_shot $K_SHOT \
    --task $TASK \
    --epsilon $EPSILON \
    --device $DEVICE

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ 分析完成！"
    echo "结果保存在: result/prompt_purging/phase1/$DATASET/k_$K_SHOT/${CLASS}_phase1_normal_side_risk_eps${EPSILON}.csv"
else
    echo ""
    echo "✗ 分析失败"
    exit 1
fi
