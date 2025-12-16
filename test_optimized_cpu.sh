#!/bin/bash
################################################################################
# CPU 优化测试脚本
# 测试优化后的 CPU 占用情况
################################################################################

echo "=========================================="
echo "PromptAD CPU 优化测试"
echo "优化内容："
echo "  1. 降低 Dataset resize: 1024×1024 → 256×256"
echo "  2. 添加 pin_memory=True 加速 GPU 传输"
echo "  3. 将图像预处理移到 Dataset（避免训练循环重复转换）"
echo "=========================================="
echo ""

# 测试配置
GPU_ID=0
DATASET="mvtec"
CLASS="bottle"
K_SHOT=2
EXP_CONFIG="qq_residual"
EPOCH=5
BATCH_SIZE=8

echo "测试参数："
echo "  GPU: $GPU_ID"
echo "  Dataset: $DATASET"
echo "  Class: $CLASS"
echo "  K-shot: $K_SHOT"
echo "  Config: $EXP_CONFIG"
echo "  Epoch: $EPOCH (快速测试)"
echo ""

# 创建日志目录
LOG_DIR="./test_logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/cpu_test_cls_${TIMESTAMP}.log"

echo "日志文件: $LOG_FILE"
echo ""
echo "=========================================="
echo "开始测试..."
echo "=========================================="
echo ""

# 运行训练（前台运行以便观察）
python train_cls.py \
    --exp_config "$EXP_CONFIG" \
    --class_name "$CLASS" \
    --dataset "$DATASET" \
    --k-shot "$K_SHOT" \
    --Epoch "$EPOCH" \
    --batch-size "$BATCH_SIZE" \
    --gpu-id "$GPU_ID" \
    --backbone ViT-B-16 \
    --pretrained_dataset laion400m_e32 \
    --n_ctx 4 \
    --n_pro 3 \
    --n_ctx_ab 1 \
    --n_pro_ab 4 \
    --lr 0.002 \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ 测试完成！"
else
    echo "✗ 测试失败 (exit code: $EXIT_CODE)"
fi
echo "=========================================="
echo ""
echo "请检查运行期间的 CPU 占用情况："
echo "  - 查看 htop 中单进程占用的核心数"
echo "  - 对比优化前后的差异"
echo ""
echo "优化前预期："
echo "  - 单进程占用 6 核左右"
echo "  - GPU 利用率 ~20%"
echo ""
echo "优化后预期："
echo "  - 单进程占用 3-4 核"
echo "  - GPU 利用率提升到 40-60%"
echo "  - 训练速度提升 1.5-2 倍"
echo ""
