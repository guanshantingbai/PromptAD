#!/bin/bash
################################################################################
# num_workers 对比测试脚本
# 测试 num_workers=0 vs num_workers=2 的性能差异
################################################################################

echo "=========================================="
echo "num_workers 对比测试"
echo "=========================================="
echo ""

GPU_ID=0
DATASET="mvtec"
CLASS="bottle"
K_SHOT=2
EXP_CONFIG="qq_residual"
EPOCH=10  # 短测试
BATCH_SIZE=8

LOG_DIR="./test_logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "测试配置："
echo "  Dataset: $DATASET / $CLASS"
echo "  K-shot: $K_SHOT"
echo "  Epoch: $EPOCH (快速对比)"
echo "  GPU: $GPU_ID"
echo ""

# ============================================
# 测试 1: num_workers=0 (当前优化后的基线)
# ============================================
echo "=========================================="
echo "测试 1/2: num_workers=0 (单线程加载)"
echo "=========================================="
LOG_FILE_0="$LOG_DIR/test_workers0_${TIMESTAMP}.log"

START_TIME=$(date +%s)

python train_cls.py \
    --exp_config "$EXP_CONFIG" \
    --class_name "$CLASS" \
    --dataset "$DATASET" \
    --k-shot "$K_SHOT" \
    --Epoch "$EPOCH" \
    --batch-size "$BATCH_SIZE" \
    --gpu-id "$GPU_ID" \
    --num-workers 0 \
    --backbone ViT-B-16 \
    --pretrained_dataset laion400m_e32 \
    --n_ctx 4 \
    --n_pro 3 \
    --n_ctx_ab 1 \
    --n_pro_ab 4 \
    --lr 0.002 \
    2>&1 | tee "$LOG_FILE_0"

END_TIME=$(date +%s)
DURATION_0=$((END_TIME - START_TIME))

echo ""
echo "测试 1 完成，耗时: ${DURATION_0}s"
echo ""
sleep 5

# ============================================
# 测试 2: num_workers=2 (2 个子进程)
# ============================================
echo "=========================================="
echo "测试 2/2: num_workers=2 (2 个子进程)"
echo "=========================================="
LOG_FILE_2="$LOG_DIR/test_workers2_${TIMESTAMP}.log"

START_TIME=$(date +%s)

python train_cls.py \
    --exp_config "$EXP_CONFIG" \
    --class_name "$CLASS" \
    --dataset "$DATASET" \
    --k-shot "$K_SHOT" \
    --Epoch "$EPOCH" \
    --batch-size "$BATCH_SIZE" \
    --gpu-id "$GPU_ID" \
    --num-workers 2 \
    --backbone ViT-B-16 \
    --pretrained_dataset laion400m_e32 \
    --n_ctx 4 \
    --n_pro 3 \
    --n_ctx_ab 1 \
    --n_pro_ab 4 \
    --lr 0.002 \
    2>&1 | tee "$LOG_FILE_2"

END_TIME=$(date +%s)
DURATION_2=$((END_TIME - START_TIME))

echo ""
echo "测试 2 完成，耗时: ${DURATION_2}s"
echo ""

# ============================================
# 对比结果
# ============================================
echo "=========================================="
echo "对比结果"
echo "=========================================="
echo ""
echo "num_workers=0: ${DURATION_0}s"
echo "num_workers=2: ${DURATION_2}s"
echo ""

if [ "$DURATION_2" -lt "$DURATION_0" ]; then
    SPEEDUP=$(echo "scale=2; $DURATION_0 / $DURATION_2" | bc)
    IMPROVEMENT=$(echo "scale=1; ($DURATION_0 - $DURATION_2) * 100 / $DURATION_0" | bc)
    echo "✅ num_workers=2 更快！"
    echo "   加速比: ${SPEEDUP}x"
    echo "   提升: ${IMPROVEMENT}%"
else
    echo "⚠️  num_workers=2 反而更慢，可能原因："
    echo "   - CPU 已经饱和"
    echo "   - 进程创建/通信开销大于收益"
    echo "   建议: 保持 num_workers=0"
fi

echo ""
echo "=========================================="
echo "详细日志："
echo "  num_workers=0: $LOG_FILE_0"
echo "  num_workers=2: $LOG_FILE_2"
echo ""
echo "建议配合监控脚本查看 CPU/GPU 占用："
echo "  ./monitor_cpu.sh"
echo "=========================================="
