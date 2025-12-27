#!/bin/bash
#
# 全类别验证 - 监控面板
# 实时显示训练和评估进度
#

echo "========================================================================"
echo "全类别验证 - 实时监控面板"
echo "========================================================================"
echo "刷新间隔: 10秒 (Ctrl+C 退出)"
echo ""

while true; do
    clear
    echo "========================================================================"
    echo "全类别(27类) v1/v2 验证 - 监控面板"
    echo "========================================================================"
    echo "更新时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # 训练进度
    echo "【训练进度】"
    echo "------------------------------------------------------------------------"
    v1_count=$(find result/ours_fix_ema_rep_margin -name "*.pt" -type f 2>/dev/null | wc -l)
    v2_count=$(find result/ema_rep_only -name "*.pt" -type f 2>/dev/null | wc -l)
    
    v1_progress=$((v1_count * 100 / 27))
    v2_progress=$((v2_count * 100 / 27))
    
    printf "v1: [%-27s] %2d/27 (%3d%%)\n" "$(printf '#%.0s' $(seq 1 $v1_count))" $v1_count $v1_progress
    printf "v2: [%-27s] %2d/27 (%3d%%)\n" "$(printf '#%.0s' $(seq 1 $v2_count))" $v2_count $v2_progress
    
    echo ""
    
    # 评估进度
    echo "【评估进度】"
    echo "------------------------------------------------------------------------"
    eval_count=$(ls analysis/full_27class_comparison/*_split_auroc.csv 2>/dev/null | wc -l)
    eval_classes=$((eval_count / 3))
    eval_progress=$((eval_count * 100 / 81))
    
    printf "评估: [%-27s] %2d/27 (%3d%%) [%d/81文件]\n" \
        "$(printf '#%.0s' $(seq 1 $eval_classes))" $eval_classes $eval_progress $eval_count
    
    echo ""
    
    # 最新日志
    echo "【v1训练最新日志】(logs/train_full_v1.log)"
    echo "------------------------------------------------------------------------"
    if [ -f logs/train_full_v1.log ]; then
        tail -3 logs/train_full_v1.log
    else
        echo "(日志文件不存在)"
    fi
    
    echo ""
    echo "【v2训练最新日志】(logs/train_full_v2.log)"
    echo "------------------------------------------------------------------------"
    if [ -f logs/train_full_v2.log ]; then
        tail -3 logs/train_full_v2.log
    else
        echo "(日志文件不存在)"
    fi
    
    echo ""
    echo "【评估最新日志】(logs/evaluate_full.log)"
    echo "------------------------------------------------------------------------"
    if [ -f logs/evaluate_full.log ]; then
        tail -3 logs/evaluate_full.log
    else
        echo "(日志文件不存在)"
    fi
    
    echo ""
    echo "========================================================================"
    
    # 检查是否完成
    if [ $v1_count -eq 27 ] && [ $v2_count -eq 27 ] && [ $eval_count -eq 81 ]; then
        echo "🎉 全流程已完成！"
        echo ""
        echo "运行分析: python analyze_full_27class.py"
        echo "========================================================================"
        break
    fi
    
    echo "按 Ctrl+C 退出监控"
    sleep 10
done
