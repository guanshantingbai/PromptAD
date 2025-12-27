#!/bin/bash
#
# 全类别验证 - 快速启动脚本
# 自动检测已完成的阶段，从断点继续
#

set -e

echo "========================================================================"
echo "全类别(27类) v1/v2 验证 - 快速启动"
echo "========================================================================"
echo ""

# 检查v1训练状态
v1_trained=$(find result/ours_fix_ema_rep_margin -name "*.pt" -type f 2>/dev/null | wc -l)
echo "v1已训练: $v1_trained/27 类"

# 检查v2训练状态
v2_trained=$(find result/ema_rep_only -name "*.pt" -type f 2>/dev/null | wc -l)
echo "v2已训练: $v2_trained/27 类"

# 检查评估状态
eval_files=$(ls analysis/full_27class_comparison/*_split_auroc.csv 2>/dev/null | wc -l)
eval_classes=$((eval_files / 3))
echo "已评估: $eval_classes/27 类 (共需81个文件)"

echo ""
echo "========================================================================"
echo "推荐执行方案:"
echo "========================================================================"

if [ $v1_trained -lt 27 ] || [ $v2_trained -lt 27 ]; then
    echo "⚠️  训练未完成，需要先训练全部27类"
    echo ""
    echo "预计时间:"
    echo "  - v1训练: 约 $(( (27 - v1_trained) * 90 / 60 )) 分钟 (剩余 $((27 - v1_trained)) 类)"
    echo "  - v2训练: 约 $(( (27 - v2_trained) * 90 / 60 )) 分钟 (剩余 $((27 - v2_trained)) 类)"
    echo "  - 并行数=2，每类约1.5分钟"
    echo ""
    echo "启动命令:"
    echo "  nohup ./train_full_27class.sh > logs/train_full_all.log 2>&1 &"
    echo ""
    echo "监控进度:"
    echo "  tail -f logs/train_full_v1.log"
    echo "  tail -f logs/train_full_v2.log"
    
elif [ $eval_classes -lt 27 ]; then
    echo "✅ 训练已完成！"
    echo "⚠️  评估未完成，需要评估全部27类×3版本"
    echo ""
    echo "预计时间:"
    echo "  - 评估: 约 $(( (81 - eval_files) * 10 / 60 )) 分钟 (剩余 $((81 - eval_files)) 个)"
    echo "  - 每次评估约10秒"
    echo ""
    echo "启动命令:"
    echo "  nohup ./evaluate_full_27class.sh > logs/evaluate_full_all.log 2>&1 &"
    echo ""
    echo "监控进度:"
    echo "  tail -f logs/evaluate_full.log"
    
else
    echo "✅ 训练和评估均已完成！"
    echo "🔍 可以直接运行分析"
    echo ""
    echo "启动命令:"
    echo "  python analyze_full_27class.py"
    echo ""
    echo "输出:"
    echo "  - 终端: 详细分析报告"
    echo "  - CSV: analysis/full_27class_comparison/full_27class_data.csv"
    echo "  - 图表: analysis/full_27class_comparison/full_27class_analysis.png"
fi

echo "========================================================================"
echo ""

# 提供快速启动选项
if [ $v1_trained -lt 27 ] || [ $v2_trained -lt 27 ]; then
    read -p "是否立即开始训练？[y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "启动后台训练..."
        nohup ./train_full_27class.sh > logs/train_full_all.log 2>&1 &
        train_pid=$!
        echo "✅ 训练已启动 (PID: $train_pid)"
        echo "监控命令: tail -f logs/train_full_v1.log"
        echo ""
        echo "预计完成时间: $(date -d "+$((((27 - v1_trained) + (27 - v2_trained)) * 90 / 2 / 60)) minutes" '+%H:%M')"
    fi
    
elif [ $eval_classes -lt 27 ]; then
    read -p "是否立即开始评估？[y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "启动后台评估..."
        nohup ./evaluate_full_27class.sh > logs/evaluate_full_all.log 2>&1 &
        eval_pid=$!
        echo "✅ 评估已启动 (PID: $eval_pid)"
        echo "监控命令: tail -f logs/evaluate_full.log"
        echo ""
        echo "预计完成时间: $(date -d "+$(((81 - eval_files) * 10 / 60)) minutes" '+%H:%M')"
    fi
    
else
    read -p "是否立即运行分析？[Y/n] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        python analyze_full_27class.py
    fi
fi
