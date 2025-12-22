#!/bin/bash
# 单任务测试脚本 - 验证只运行一次的机制

set -e

# 测试配置
DATASET="mvtec"
CLASS="carpet"
K_SHOT=4
TASK="cls"
OUTPUT_DIR="./test_gate_single"

echo "=================================="
echo "单任务测试：Gate实验框架"
echo "=================================="
echo "数据集: $DATASET"
echo "类别: $CLASS"
echo "K值: $K_SHOT"
echo "任务: $TASK"
echo "输出目录: $OUTPUT_DIR"
echo ""

# 运行第一次
echo "【第一次运行】- 应该进行训练"
echo "=================================="
python run_gate_experiment.py \
    --dataset $DATASET \
    --class_name $CLASS \
    --k-shot $K_SHOT \
    --task $TASK \
    --root-dir $OUTPUT_DIR \
    --gpu-id 0 \
    --batch-size 16

echo ""
echo ""
echo "=================================="
echo "【第二次运行】- 应该跳过训练"
echo "=================================="
python run_gate_experiment.py \
    --dataset $DATASET \
    --class_name $CLASS \
    --k-shot $K_SHOT \
    --task $TASK \
    --root-dir $OUTPUT_DIR \
    --gpu-id 0 \
    --batch-size 16

echo ""
echo "=================================="
echo "测试完成！"
echo ""
echo "验证结果:"
echo "1. 第一次应该显示 '将进行训练'"
echo "2. 第二次应该显示 '跳过训练，直接加载模型'"
echo "3. 两次结果应该完全一致"
echo ""
echo "输出文件位置:"
echo "  - Checkpoint: $OUTPUT_DIR/$DATASET/k_${K_SHOT}/checkpoint/"
echo "  - Metadata: $OUTPUT_DIR/$DATASET/k_${K_SHOT}/metadata/"
echo "  - Results: $OUTPUT_DIR/$DATASET/k_${K_SHOT}/gate_results/"
echo "=================================="

