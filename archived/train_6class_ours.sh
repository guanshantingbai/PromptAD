#!/bin/bash
"""
6类代表性类别训练脚本 - Ours版本
配置：prompt2_fix_ema_rep_margin
改动：(1)修正EMA逐原型对齐 (2)Repulsion loss (3)Margin loss
"""

# 固定配置
VERSION="ours_fix_ema_rep_margin"
K_SHOT=2
N_PRO=1
N_PRO_AB=4
SEED=111
GPU_ID=0

# 超参数（三项改动的权重）
LAMBDA1=0.001       # EMA loss（修正版）
LAMBDA_REP=0.05     # Repulsion loss（轻量级）
LAMBDA_MARGIN=0.1   # Margin loss（中等强度）

echo "========================================================================"
echo "Ours 配置训练 - 6类代表性类别"
echo "版本: $VERSION"
echo "改动: 修正EMA + Repulsion + Margin"
echo "========================================================================"
echo ""
echo "超参数:"
echo "  lambda1 (EMA):        $LAMBDA1"
echo "  lambda_rep (Repulsion): $LAMBDA_REP"
echo "  lambda_margin (Margin): $LAMBDA_MARGIN"
echo ""

# 定义6个代表性类别
# Severe degrade (3): toothbrush, capsule, visa-pcb2
# Stable (2): carpet, leather
# Improved (1): screw

classes_mvtec=(
    "toothbrush"  # Severe
    "capsule"     # Severe
    "carpet"      # Stable
    "leather"     # Stable
    "screw"       # Improved
)

classes_visa=(
    "pcb2"        # Severe
)

mkdir -p logs/ours_training
mkdir -p result/$VERSION

total=6
current=0

# 训练MVTec类别
for cls in "${classes_mvtec[@]}"; do
    current=$((current + 1))
    echo "[$current/$total] 训练 mvtec-$cls"
    
    python train_cls.py \
        --dataset mvtec \
        --class_name $cls \
        --k-shot $K_SHOT \
        --n_pro $N_PRO \
        --n_pro_ab $N_PRO_AB \
        --lambda1 $LAMBDA1 \
        --lambda_rep $LAMBDA_REP \
        --lambda_margin $LAMBDA_MARGIN \
        --seed $SEED \
        --gpu-id $GPU_ID \
        --root-dir ./result/$VERSION \
        --Epoch 100 \
        --lr 0.002 \
        --batch-size 400 \
        --vis False \
        > logs/ours_training/mvtec_${cls}_k${K_SHOT}.log 2>&1
    
    if [ $? -eq 0 ]; then
        echo "   ✅ 完成"
    else
        echo "   ❌ 失败 (查看 logs/ours_training/mvtec_${cls}_k${K_SHOT}.log)"
    fi
    echo ""
done

# 训练VisA类别
for cls in "${classes_visa[@]}"; do
    current=$((current + 1))
    echo "[$current/$total] 训练 visa-$cls"
    
    python train_cls.py \
        --dataset visa \
        --class_name $cls \
        --k-shot $K_SHOT \
        --n_pro $N_PRO \
        --n_pro_ab $N_PRO_AB \
        --lambda1 $LAMBDA1 \
        --lambda_rep $LAMBDA_REP \
        --lambda_margin $LAMBDA_MARGIN \
        --seed $SEED \
        --gpu-id $GPU_ID \
        --root-dir ./result/$VERSION \
        --Epoch 100 \
        --lr 0.002 \
        --batch-size 400 \
        --vis False \
        > logs/ours_training/visa_${cls}_k${K_SHOT}.log 2>&1
    
    if [ $? -eq 0 ]; then
        echo "   ✅ 完成"
    else
        echo "   ❌ 失败 (查看 logs/ours_training/visa_${cls}_k${K_SHOT}.log)"
    fi
    echo ""
done

echo "========================================================================"
echo "✅ Ours配置训练完成！"
echo "========================================================================"
echo ""
echo "Checkpoints保存在: result/checkpoint/"
echo "训练日志在: logs/ours_training/"
echo ""
echo "下一步: 运行扩展评估"
echo "  bash evaluate_6class_comparison.sh"
