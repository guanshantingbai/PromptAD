#!/bin/bash
#
# 简化评估脚本：直接评估v1和v2的27类
#

set -e

echo "========================================================================"
echo "v1/v2评估 - 27类"
echo "========================================================================"

# MVTec 15类
MVTEC_CLASSES=(
    "carpet" "grid" "leather" "tile" "wood"
    "bottle" "cable" "capsule" "hazelnut" "metal_nut"
    "pill" "screw" "toothbrush" "transistor" "zipper"
)

# VisA 12类
VISA_CLASSES=(
    "candle" "capsules" "cashew" "chewinggum" "fryum"
    "macaroni1" "macaroni2" "pcb1" "pcb2" "pcb3"
    "pcb4" "pipe_fryum"
)

K_SHOT=2
SEED=111
GPU_ID=0

mkdir -p logs/v1_v2_eval

total=54
current=0

# 评估MVTec类别
for cls in "${MVTEC_CLASSES[@]}"; do
    echo ""
    echo "========================================================================"
    echo "[$((current/2 + 1))/27] mvtec-$cls"
    echo "========================================================================"
    
    # v1评估
    current=$((current + 1))
    echo "  [$current/$total] v1..."
    
    python evaluate_extended_metrics.py \
        --dataset mvtec \
        --class_name $cls \
        --root-dir "result" \
        --version "v1_ema_rep05_margin01" \
        --k-shot $K_SHOT \
        --seed $SEED \
        --gpu-id $GPU_ID \
        > logs/v1_v2_eval/mvtec_${cls}_v1.log 2>&1
    
    [ $? -eq 0 ] && echo "     ✅ v1完成" || echo "     ❌ v1失败"
    
    # v2评估
    current=$((current + 1))
    echo "  [$current/$total] v2..."
    
    python evaluate_extended_metrics.py \
        --dataset mvtec \
        --class_name $cls \
        --root-dir "result" \
        --version "v2_ema_rep10_nomargin" \
        --k-shot $K_SHOT \
        --seed $SEED \
        --gpu-id $GPU_ID \
        > logs/v1_v2_eval/mvtec_${cls}_v2.log 2>&1
    
    [ $? -eq 0 ] && echo "     ✅ v2完成" || echo "     ❌ v2失败"
done

# 评估VisA类别
for cls in "${VISA_CLASSES[@]}"; do
    echo ""
    echo "========================================================================"
    echo "[$((current/2 + 1))/27] visa-$cls"
    echo "========================================================================"
    
    # v1评估
    current=$((current + 1))
    echo "  [$current/$total] v1..."
    
    python evaluate_extended_metrics.py \
        --dataset visa \
        --class_name $cls \
        --root-dir "result" \
        --version "v1_ema_rep05_margin01" \
        --k-shot $K_SHOT \
        --seed $SEED \
        --gpu-id $GPU_ID \
        > logs/v1_v2_eval/visa_${cls}_v1.log 2>&1
    
    [ $? -eq 0 ] && echo "     ✅ v1完成" || echo "     ❌ v1失败"
    
    # v2评估
    current=$((current + 1))
    echo "  [$current/$total] v2..."
    
    python evaluate_extended_metrics.py \
        --dataset visa \
        --class_name $cls \
        --root-dir "result" \
        --version "v2_ema_rep10_nomargin" \
        --k-shot $K_SHOT \
        --seed $SEED \
        --gpu-id $GPU_ID \
        > logs/v1_v2_eval/visa_${cls}_v2.log 2>&1
    
    [ $? -eq 0 ] && echo "     ✅ v2完成" || echo "     ❌ v2失败"
done

echo ""
echo "========================================================================"
echo "✅ v1/v2评估完成！"
echo "========================================================================"
echo "总评估数: $total"
echo "输出目录: analysis/extended_metrics/"
echo "========================================================================"
