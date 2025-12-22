#!/bin/bash

# 检查可用的checkpoint
CHECKPOINT_DIR="result/max_score"

echo "============================================================"
echo "可用Checkpoint统计"
echo "============================================================"
echo ""

# MVTec统计
echo "MVTec AD:"
echo "--------"
for k in 1 2 4; do
    cls_count=$(find ${CHECKPOINT_DIR}/mvtec/k_${k}/checkpoint -name "CLS-*.pt" 2>/dev/null | wc -l)
    seg_count=$(find ${CHECKPOINT_DIR}/mvtec/k_${k}/checkpoint -name "SEG-*.pt" 2>/dev/null | wc -l)
    echo "  k=${k}: CLS=${cls_count}, SEG=${seg_count}"
done
echo ""

# VisA统计
echo "VisA:"
echo "-----"
for k in 1 2 4; do
    cls_count=$(find ${CHECKPOINT_DIR}/visa/k_${k}/checkpoint -name "CLS-*.pt" 2>/dev/null | wc -l)
    seg_count=$(find ${CHECKPOINT_DIR}/visa/k_${k}/checkpoint -name "SEG-*.pt" 2>/dev/null | wc -l)
    echo "  k=${k}: CLS=${cls_count}, SEG=${seg_count}"
done
echo ""

# 总计
total=$(find ${CHECKPOINT_DIR} -name "*.pt" 2>/dev/null | wc -l)
echo "总计: ${total} 个checkpoint"
echo ""
echo "============================================================"
echo "理论任务数:"
echo "  MVTec: 15类 × 3个k值 × 2任务 = 90"
echo "  VisA:  12类 × 3个k值 × 2任务 = 72"
echo "  合计: 162任务"
echo "============================================================"
