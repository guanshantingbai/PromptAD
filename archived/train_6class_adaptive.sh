#!/bin/bash
#
# v3训练脚本：类别自适应Repulsion策略 (EMA + Adaptive Repulsion)
# 基于受控实验发现：不同类别需要不同的Repulsion强度
#
# 配置:
# - toothbrush: λ_rep=0.05 (降低，避免过度分散)
# - capsule:    λ_rep=0.10 (保持，Collapse严重)
# - pcb2:       λ_rep=0.10 (保持，Collapse严重)
# - carpet:     λ_rep=0.02 (极低，Stable类)
# - leather:    λ_rep=0.02 (极低，Stable类)
# - screw:      λ_rep=0.10 (增强，低Separation)

VERSION="ema_adaptive_rep"
LAMBDA1=0.001  # EMA per-prototype alignment

# 创建日志目录
mkdir -p logs/adaptive_exp

# 定义类别和对应的λ_rep值
declare -A CLASSES=(
    ["mvtec-toothbrush"]="0.05"
    ["mvtec-capsule"]="0.10"
    ["visa-pcb2"]="0.10"
    ["mvtec-carpet"]="0.02"
    ["mvtec-leather"]="0.02"
    ["mvtec-screw"]="0.10"
)

echo "========================================================================"
echo "v3训练：类别自适应Repulsion (EMA + Adaptive Rep)"
echo "========================================================================"
echo "版本: $VERSION"
echo "EMA权重 (lambda1): $LAMBDA1"
echo ""
echo "类别自适应配置:"
for cls in "${!CLASSES[@]}"; do
    printf "  %-20s λ_rep=%.2f\n" "$cls" "${CLASSES[$cls]}"
done | sort
echo "========================================================================"
echo ""

# 训练函数
train_class() {
    local dataset=$1
    local lambda_rep=$2
    
    echo "========================================================================" | tee -a logs/adaptive_exp/train_${dataset}.log
    echo "训练: $dataset (λ_rep=$lambda_rep)" | tee -a logs/adaptive_exp/train_${dataset}.log
    echo "========================================================================" | tee -a logs/adaptive_exp/train_${dataset}.log
    
    python train_cls.py \
        --dataset "$dataset" \
        --root-dir "./result/$VERSION" \
        --lambda1 $LAMBDA1 \
        --lambda_rep $lambda_rep \
        --Epoch 5 \
        --k_value 2 \
        2>&1 | tee -a logs/adaptive_exp/train_${dataset}.log
    
    # 提取AUROC
    auroc=$(grep "Image-AUROC" logs/adaptive_exp/train_${dataset}.log | tail -1 | grep -oP '(?<=Image-AUROC: )\d+\.\d+')
    
    if [ -n "$auroc" ]; then
        echo "  ✅ $dataset 训练完成: AUROC=$auroc%" | tee -a logs/train_adaptive.log
    else
        echo "  ⚠️  $dataset 训练失败或AUROC未找到" | tee -a logs/train_adaptive.log
    fi
    
    echo ""
}

# 启动时间
start_time=$(date +%s)
echo "开始时间: $(date)" > logs/train_adaptive.log

# 按顺序训练6个类别
for dataset in mvtec-toothbrush mvtec-capsule visa-pcb2 mvtec-carpet mvtec-leather mvtec-screw; do
    lambda_rep=${CLASSES[$dataset]}
    train_class "$dataset" "$lambda_rep"
done

# 结束时间
end_time=$(date +%s)
duration=$((end_time - start_time))
minutes=$((duration / 60))
seconds=$((duration % 60))

echo "========================================================================" | tee -a logs/train_adaptive.log
echo "🎉 v3训练完成！" | tee -a logs/train_adaptive.log
echo "总耗时: ${minutes}分${seconds}秒" | tee -a logs/train_adaptive.log
echo "========================================================================" | tee -a logs/train_adaptive.log

# 快速汇总
echo "" | tee -a logs/train_adaptive.log
echo "训练结果汇总:" | tee -a logs/train_adaptive.log
echo "------------------------------------------------------------------------" | tee -a logs/train_adaptive.log
for dataset in mvtec-toothbrush mvtec-capsule visa-pcb2 mvtec-carpet mvtec-leather mvtec-screw; do
    lambda_rep=${CLASSES[$dataset]}
    auroc=$(grep "Image-AUROC" logs/adaptive_exp/train_${dataset}.log | tail -1 | grep -oP '(?<=Image-AUROC: )\d+\.\d+')
    if [ -n "$auroc" ]; then
        printf "  %-20s λ_rep=%.2f  AUROC=%s%%\n" "$dataset" "$lambda_rep" "$auroc" | tee -a logs/train_adaptive.log
    fi
done
echo "========================================================================" | tee -a logs/train_adaptive.log

echo ""
echo "下一步: 运行5版本对比评估"
echo "  ./evaluate_5version_comparison.sh"
