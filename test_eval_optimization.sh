#!/bin/bash
################################################################################
# 评估优化验证脚本
# 测试优化后的训练速度和 CPU 占用
################################################################################

echo "=========================================="
echo "评估优化效果测试"
echo "=========================================="
echo ""

GPU_ID=0
DATASET="mvtec"
CLASS="bottle"
K_SHOT=2
EXP_CONFIG="qq_residual"
EPOCH=20  # 测试 20 个 epoch
BATCH_SIZE=8

LOG_DIR="./test_logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "测试参数："
echo "  Dataset: $DATASET / $CLASS"
echo "  K-shot: $K_SHOT"
echo "  Epoch: $EPOCH"
echo "  GPU: $GPU_ID"
echo ""

echo "优化内容："
echo "  ✓ 方向 1: 每 5 个 epoch 评估一次（降低评估频率）"
echo "  ✓ 方向 2: 移除不必要的 denormalization（仅在 --vis 时）"
echo "  ✓ 方向 3.1: 移除 score_maps 的重复 resize"
echo "  ✓ 方向 3.2: 降低 resolution 从 400 → 256"
echo "  ✓ 方向 3.3: 仅在可视化时收集 test_imgs"
echo "  ✓ 方向 5: 缓存测试集预处理结果"
echo ""

# 启动监控
./monitor_cpu_highfreq.sh &
MONITOR_PID=$!

echo "=========================================="
echo "开始训练（监控 PID: $MONITOR_PID）"
echo "=========================================="
echo ""

START_TIME=$(date +%s)

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
    --vis False \
    2>&1 | tee "$LOG_DIR/eval_opt_test_${TIMESTAMP}.log"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# 停止监控
kill $MONITOR_PID 2>/dev/null

echo ""
echo "=========================================="
echo "测试完成"
echo "=========================================="
echo "总耗时: ${DURATION}s"
echo ""

# 分析 CPU 占用
LOG_FILE=$(ls -t cpu_monitor_highfreq_*.log | head -1)
if [ -f "$LOG_FILE" ]; then
    echo "CPU 占用统计："
    grep "CPU:" "$LOG_FILE" | awk '{print $4}' | sort -rn | head -1 | \
        awk '{printf "  峰值: %.1f 核\n", $1}'
    grep "CPU:" "$LOG_FILE" | awk '{sum += $4; count++} END {printf "  平均: %.1f 核\n", sum/count}'
    echo ""
fi

echo "预期效果："
echo "  优化前: 8 核，每 epoch ~22s"
echo "  优化后: 1.5-2 核，每 epoch ~5s"
echo ""
echo "实际每 epoch 时间: $(echo "scale=1; $DURATION / $EPOCH" | bc)s"
echo ""

if [ "$DURATION" -lt 150 ]; then
    echo "✅ 性能提升显著！训练速度达到预期"
else
    echo "⚠️  性能提升有限，可能需要进一步调查"
fi
echo ""
echo "详细日志:"
echo "  训练: $LOG_DIR/eval_opt_test_${TIMESTAMP}.log"
echo "  监控: $LOG_FILE"
echo "=========================================="
