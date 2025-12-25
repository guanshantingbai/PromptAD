#!/bin/bash

# 策略2：修复bug后重训关键类别
# 重点：语义提升大但融合下降的类别

echo "=========================================="
echo "策略2：重训关键类别（修复后）"
echo "=========================================="
echo ""
echo "修复内容：train_cls.py 使用纯语义分支选择最佳模型"
echo "重训原因：之前用融合分数选择模型，导致保存的不是语义最佳checkpoint"
echo ""

# 关键类别：语义提升大但融合下降
# 根据verify_semantic_fusion_hypothesis.py的结果选择
CLASSES=(
    "screw"          # 语义+13.15% → 融合-5.66%
    "toothbrush"     # 语义+19.86% → 融合-8.62%
    "hazelnut"       # 语义+11.03% → 融合-8.52%
    "capsule"        # 语义+6.96% → 融合+4.59%（唯一保持的）
    "pill"           # 语义+0.62% → 融合-7.42%
    "metal_nut"      # 语义+3.15% → 融合-5.47%
)

DATASET="mvtec"
K_SHOTS=(1 2 4)  # 训练所有k值
ROOT_DIR="result/prompt1_fixed"  # 新目录，与原结果分开

echo "重训配置："
echo "  数据集: ${DATASET}"
echo "  类别数: ${#CLASSES[@]} (${CLASSES[@]})"
echo "  K值: ${K_SHOTS[@]}"
echo "  总任务数: $((${#CLASSES[@]} * ${#K_SHOTS[@]}))"
echo "  预计时间: ~1-2小时"
echo "  结果目录: ${ROOT_DIR}"
echo ""

# 计算预计时间
TOTAL_TASKS=$((${#CLASSES[@]} * ${#K_SHOTS[@]}))
echo "即将开始训练 ${TOTAL_TASKS} 个任务..."
echo ""

read -p "确认开始训练? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "已取消"
    exit 1
fi

# 创建结果目录
mkdir -p ${ROOT_DIR}

# 记录开始时间
START_TIME=$(date +%s)
echo "[$(date)] 开始重训..." | tee ${ROOT_DIR}/retraining.log

# 训练每个配置
TASK_NUM=0
for class_name in "${CLASSES[@]}"; do
    for k_shot in "${K_SHOTS[@]}"; do
        TASK_NUM=$((TASK_NUM + 1))
        echo ""
        echo "=========================================="
        echo "任务 ${TASK_NUM}/${TOTAL_TASKS}: ${class_name} k=${k_shot}"
        echo "=========================================="
        
        # 运行训练
        python train_cls.py \
            --dataset ${DATASET} \
            --class_name ${class_name} \
            --k-shot ${k_shot} \
            --n_pro 3 \
            --n_pro_ab 4 \
            --Epoch 100 \
            --lr 0.002 \
            --root-dir ${ROOT_DIR} \
            --vis False \
            2>&1 | tee -a ${ROOT_DIR}/retraining.log
        
        if [ $? -eq 0 ]; then
            echo "[$(date)] ✅ 完成: ${class_name} k=${k_shot}" | tee -a ${ROOT_DIR}/retraining.log
        else
            echo "[$(date)] ❌ 失败: ${class_name} k=${k_shot}" | tee -a ${ROOT_DIR}/retraining.log
        fi
    done
done

# 记录结束时间
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))

echo ""
echo "=========================================="
echo "重训完成！"
echo "=========================================="
echo "总任务数: ${TOTAL_TASKS}"
echo "耗时: ${HOURS}小时 ${MINUTES}分钟"
echo "结果目录: ${ROOT_DIR}"
echo ""
echo "下一步："
echo "1. 验证重训后的语义性能:"
echo "   python verify_checkpoint_semantic.py  # 需要修改RESULT_DIR"
echo ""
echo "2. 对比重训前后:"
echo "   python compare_retraining_results.py"
echo ""
