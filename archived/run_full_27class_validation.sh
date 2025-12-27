#!/bin/bash
#
# 全类别v1/v2验证 - 主控脚本
# 目的: 验证6类结论在27类上的一致性
# 约束: 并行数=2, 区分semantic和fusion结果
#

set -e

echo "========================================================================"
echo "全类别(27类) v1/v2 验证实验"
echo "========================================================================"
echo "实验目的:"
echo "  1. 验证6类结论在全类别上的一致性"
echo "  2. 检验Fusion vs Semantic解耦现象"
echo "  3. 评估6类样本的代表性"
echo ""
echo "实验配置:"
echo "  - 总类别数: 27 (MVTec 15 + VisA 12)"
echo "  - 训练版本: v1 (EMA+Rep+Margin), v2 (EMA+Rep only)"
echo "  - 并行数: 2 (保证稳定性)"
echo "  - 评估指标: Fusion AUROC, Semantic AUROC, Separation"
echo "========================================================================"
echo ""

# 检查是否已有训练结果
echo "【阶段0】检查现有结果"
echo "------------------------------------------------------------------------"

# 检查v1结果
v1_count=$(find result/ours_fix_ema_rep_margin/*/k_2/checkpoint -name "*.pt" 2>/dev/null | wc -l)
echo "v1已训练: $v1_count/27 类"

# 检查v2结果
v2_count=$(find result/ema_rep_only/*/k_2/checkpoint -name "*.pt" 2>/dev/null | wc -l)
echo "v2已训练: $v2_count/27 类"

# 检查评估结果
eval_count=$(ls analysis/full_27class_comparison/*_split_auroc.csv 2>/dev/null | wc -l)
echo "已评估: $((eval_count / 3))/27 类 (应有 81 个文件)"

echo ""
read -p "是否需要重新训练？[y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    NEED_TRAIN=true
else
    NEED_TRAIN=false
fi

# 阶段1: 训练
if [ "$NEED_TRAIN" = true ]; then
    echo ""
    echo "【阶段1】开始训练"
    echo "========================================================================"
    
    # 给脚本添加执行权限
    chmod +x train_full_27class.sh
    
    # 执行训练（前台运行，显示进度）
    ./train_full_27class.sh
    
    echo ""
    echo "✅ 训练完成"
else
    echo ""
    echo "⏭️  跳过训练阶段"
fi

# 阶段2: 评估
echo ""
echo "【阶段2】开始评估"
echo "========================================================================"

read -p "是否需要运行评估？[Y/n] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    chmod +x evaluate_full_27class.sh
    
    # 评估可以后台运行
    read -p "是否后台运行评估？[Y/n] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        nohup ./evaluate_full_27class.sh > logs/evaluate_full.log 2>&1 &
        eval_pid=$!
        echo "评估已在后台运行 (PID: $eval_pid)"
        echo "监控进度: tail -f logs/evaluate_full.log"
        echo ""
        echo "等待评估完成后，运行阶段3:"
        echo "  python analyze_full_27class.py"
        exit 0
    else
        ./evaluate_full_27class.sh
        echo "✅ 评估完成"
    fi
else
    echo "⏭️  跳过评估阶段"
fi

# 阶段3: 分析
echo ""
echo "【阶段3】开始分析"
echo "========================================================================"

# 检查评估是否完成
eval_count=$(ls analysis/full_27class_comparison/*_split_auroc.csv 2>/dev/null | wc -l)
expected_count=81  # 27类 × 3版本

if [ "$eval_count" -lt "$expected_count" ]; then
    echo "⚠️  评估未完成: $eval_count/$expected_count"
    echo "请等待评估完成后再运行分析"
    echo "监控命令: tail -f logs/evaluate_full.log"
    exit 1
fi

echo "✅ 评估数据完整: $eval_count/$expected_count"
echo ""
echo "运行分析脚本..."

python analyze_full_27class.py

echo ""
echo "========================================================================"
echo "🎉 全类别验证实验完成！"
echo "========================================================================"
echo ""
echo "查看结果:"
echo "  - 数据: analysis/full_27class_comparison/full_27class_data.csv"
echo "  - 可视化: analysis/full_27class_comparison/full_27class_analysis.png"
echo "  - 日志: logs/evaluate_full.log"
echo ""
echo "核心结论已输出到终端，请查看一致性评分。"
echo "========================================================================"
