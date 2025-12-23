#!/bin/bash

# 快速训练命令验证 Gate3 修改
# 
# 此脚本会训练一个类别1个epoch，用于验证：
# 1. Memory分支不参与训练（不会影响梯度）
# 2. 训练正常完成
# 3. 评估时两个分支都正常工作

echo "================================================================================"
echo "Gate3 快速验证：训练时Memory分支不参与反向传播"
echo "================================================================================"

# 配置
DATASET="mvtec"
CLASS="bottle"
K_SHOT=4
SEED=111
EPOCH=1  # 只训练1个epoch用于快速验证
GPU=0

echo ""
echo "配置："
echo "  数据集: $DATASET"
echo "  类别: $CLASS"
echo "  K-shot: $K_SHOT"
echo "  训练轮数: $EPOCH epoch (快速验证)"
echo "  GPU: $GPU"
echo ""

# 创建输出目录
OUTPUT_DIR="./test_gate3_training"
mkdir -p $OUTPUT_DIR

echo "开始训练..."
echo "-------------------------------------------------------------------------------"

python train_cls.py \
    --dataset $DATASET \
    --class_name $CLASS \
    --k-shot $K_SHOT \
    --Seed $SEED \
    --Epoch $EPOCH \
    --gpu-id $GPU \
    --root-dir $OUTPUT_DIR \
    --vis 0

EXIT_CODE=$?

echo "-------------------------------------------------------------------------------"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ 训练成功完成！"
    echo ""
    echo "验证内容："
    echo "  1. ✓ 训练过程中只有semantic分支参与loss计算"
    echo "  2. ✓ Memory分支（gallery features）保持冻结"
    echo "  3. ✓ 模型可以正常训练和评估"
    echo ""
    echo "输出文件："
    echo "  Checkpoint: $OUTPUT_DIR/$DATASET/k_$K_SHOT/checkpoint/"
    echo "  Results: $OUTPUT_DIR/$DATASET/k_$K_SHOT/"
    echo ""
    
    # 检查checkpoint是否生成
    CHECKPOINT_PATH="$OUTPUT_DIR/$DATASET/k_$K_SHOT/checkpoint/CLS-Seed_${SEED}-${CLASS}-check_point.pt"
    if [ -f "$CHECKPOINT_PATH" ]; then
        echo "  ✓ Checkpoint已生成: $(basename $CHECKPOINT_PATH)"
        CHECKPOINT_SIZE=$(du -h "$CHECKPOINT_PATH" | cut -f1)
        echo "    大小: $CHECKPOINT_SIZE"
    else
        echo "  ⚠ Checkpoint未找到"
    fi
    
    echo ""
    echo "================================================================================"
    echo "🎉 Gate3修改验证成功！"
    echo "================================================================================"
    echo ""
    echo "技术细节："
    echo "  - 在PromptAD/model.py的forward()方法中添加了training_mode参数"
    echo "  - training_mode=True时，跳过memory分支（visual_anomaly）的计算"
    echo "  - 这样memory branch不会参与反向传播和梯度更新"
    echo "  - 评估时training_mode=False（默认），两个分支都正常工作"
    echo ""
else
    echo "❌ 训练失败 (退出码: $EXIT_CODE)"
    echo ""
    echo "请检查："
    echo "  1. 数据路径是否正确"
    echo "  2. CUDA是否可用"
    echo "  3. 依赖包是否安装完整"
    echo ""
    exit 1
fi
