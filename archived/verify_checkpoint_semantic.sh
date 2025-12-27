#!/bin/bash

# 策略1验证脚本：测试现有checkpoint的纯语义性能
# 对比baseline语义分支 vs prompt1语义分支（使用现有checkpoint）

echo "=========================================="
echo "策略1：验证现有checkpoint的纯语义性能"
echo "=========================================="
echo ""

# 测试关键类别（语义提升大但融合下降的类别）
CLASSES=("screw" "toothbrush" "hazelnut" "capsule" "pill")
K_SHOT=2
DATASET="mvtec"

# Baseline语义分支的已知性能（来自fair_comparison_semantic_only_k2.csv）
declare -A BASELINE_SEMANTIC
BASELINE_SEMANTIC["screw"]=66.42
BASELINE_SEMANTIC["toothbrush"]=69.58
BASELINE_SEMANTIC["hazelnut"]=80.11
BASELINE_SEMANTIC["capsule"]=73.69
BASELINE_SEMANTIC["pill"]=85.50

# Prompt1语义分支的期望性能（训练时应该达到的）
declare -A EXPECTED_SEMANTIC
EXPECTED_SEMANTIC["screw"]=79.57
EXPECTED_SEMANTIC["toothbrush"]=89.44
EXPECTED_SEMANTIC["hazelnut"]=91.14
EXPECTED_SEMANTIC["capsule"]=80.65
EXPECTED_SEMANTIC["pill"]=86.12

echo "测试配置："
echo "  数据集: ${DATASET}"
echo "  K-shot: ${K_SHOT}"
echo "  测试类别: ${CLASSES[@]}"
echo ""
echo "将测试现有checkpoint的纯语义性能，对比："
echo "  1. Baseline语义分支（已知）"
echo "  2. Prompt1期望语义性能（训练时的）"
echo ""
echo "判断标准："
echo "  ✅ 如果现有checkpoint达到期望性能 → 不需要重新训练"
echo "  ⚠️  如果现有checkpoint低于期望但高于baseline → 考虑重训部分类别"
echo "  ❌如果现有checkpoint低于或接近baseline → 需要重新训练"
echo ""
echo "=========================================="
echo ""

# 创建结果目录
RESULT_DIR="result/prompt1_memory"
VALIDATION_LOG="${RESULT_DIR}/semantic_validation_k${K_SHOT}.txt"

echo "验证结果将保存到: ${VALIDATION_LOG}"
echo ""

# 清空之前的日志
> ${VALIDATION_LOG}

# 记录验证信息
cat >> ${VALIDATION_LOG} << EOF
========================================
策略1验证：现有checkpoint的纯语义性能
========================================
日期: $(date)
数据集: ${DATASET}
K-shot: ${K_SHOT}
评估模式: --semantic-only True

类别 | Baseline语义 | 期望语义 | 实际语义 | vs Baseline | vs 期望 | 判断
-----|------------|---------|---------|------------|---------|------
EOF

echo "开始测试..."
echo ""

# 测试每个类别
for class_name in "${CLASSES[@]}"; do
    echo "测试类别: ${class_name}"
    echo "  Baseline语义: ${BASELINE_SEMANTIC[$class_name]}%"
    echo "  期望语义: ${EXPECTED_SEMANTIC[$class_name]}%"
    echo "  测试中..."
    
    # 运行测试（纯语义模式）
    python test_cls.py \
        --dataset ${DATASET} \
        --class_name ${class_name} \
        --k-shot ${K_SHOT} \
        --semantic-only True \
        --vis False \
        --n_pro 3 \
        --n_pro_ab 4 \
        --root-dir result/prompt1_memory \
        2>&1 | grep "Pixel-AUROC" | tail -1
    
    echo ""
done

echo ""
echo "=========================================="
echo "测试完成！"
echo "=========================================="
echo ""
echo "结果已保存到: ${VALIDATION_LOG}"
echo ""
echo "请查看各类别的实际语义性能，判断是否需要重新训练："
echo "  - 查看详细结果: cat ${VALIDATION_LOG}"
echo "  - 查看CSV结果: 在 ${RESULT_DIR}/mvtec/cls_logs/<class>_k${K_SHOT}/test_results.csv"
echo ""
