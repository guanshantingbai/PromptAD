#!/bin/bash
# 受控实验：EMA修正 + Repulsion Loss（移除Margin Loss）
# 版本：ema_rep_only
# 配置：
#   - EMA: 逐原型对齐（修正版）
#   - Repulsion: lambda_rep=0.1（增强原型多样性）
#   - Margin: 移除（避免破坏Stable类）

# 固定配置
VERSION="ema_rep_only"
K_SHOT=2
N_PRO=1
N_PRO_AB=4
SEED=111
GPU_ID=0

# 超参数（受控实验）
LAMBDA1=0.001       # EMA loss（修正版，不变）
LAMBDA_REP=0.10     # Repulsion loss（0.05 → 0.10，增大2倍）

echo "========================================================================"
echo "受控实验：EMA + Repulsion（无Margin）"
echo "版本: $VERSION"
echo "改动: (1) EMA逐原型对齐 (2) Repulsion增强 (3) 移除Margin Loss"
echo "========================================================================"
echo ""
echo "超参数:"
echo "  lambda1 (EMA):        $LAMBDA1"
echo "  lambda_rep (Repulsion): $LAMBDA_REP"
echo "  lambda_margin:        [已移除]"
echo ""
echo "实验目标:"
echo "  - 验证在不破坏Stable类的前提下，Repulsion能否缓解原型坍缩"
echo "  - 隔离Margin Loss的负面影响"
echo ""

# 定义6个代表性类别
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

mkdir -p logs/controlled_exp
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
        --seed $SEED \
        --gpu-id $GPU_ID \
        --root-dir ./result/$VERSION \
        --Epoch 100 \
        --lr 0.002 \
        --batch-size 400 \
        --vis False \
        > logs/controlled_exp/mvtec_${cls}_k${K_SHOT}.log 2>&1
    
    if [ $? -eq 0 ]; then
        echo "   ✅ 完成"
    else
        echo "   ❌ 失败 (查看 logs/controlled_exp/mvtec_${cls}_k${K_SHOT}.log)"
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
        --seed $SEED \
        --gpu-id $GPU_ID \
        --root-dir ./result/$VERSION \
        --Epoch 100 \
        --lr 0.002 \
        --batch-size 400 \
        --vis False \
        > logs/controlled_exp/visa_${cls}_k${K_SHOT}.log 2>&1
    
    if [ $? -eq 0 ]; then
        echo "   ✅ 完成"
    else
        echo "   ❌ 失败 (查看 logs/controlled_exp/visa_${cls}_k${K_SHOT}.log)"
    fi
    echo ""
done

echo "========================================================================"
echo "✅ 受控实验训练完成！"
echo "========================================================================"
echo ""
echo "Checkpoints保存在: result/$VERSION/{dataset}/k_2/checkpoint/"
echo "训练日志在: logs/controlled_exp/"
echo ""
echo "下一步: 运行对比评估"
echo "  1. 对比 Prompt2 vs Ours_v1(全部改动) vs Ours_v2(EMA+Rep)"
echo "  2. 验证移除Margin Loss后，Separation是否不再下降"
echo "  3. 验证Repulsion增强后，Collapse是否进一步减少"
