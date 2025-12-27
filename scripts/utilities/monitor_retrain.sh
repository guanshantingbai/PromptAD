#!/bin/bash

# 监控重训进度的便捷脚本

echo "=========================================="
echo "重训进度监控"
echo "=========================================="
echo ""

# 检查进程是否在运行
if ps aux | grep "retrain_key_classes.py" | grep -v grep > /dev/null; then
    echo "✅ 训练进程正在运行"
    echo ""
else
    echo "❌ 训练进程未运行"
    echo ""
    exit 1
fi

# 显示当前进度
echo "【当前进度】"
grep "进度:" retrain.log | tail -1
echo ""

# 显示最近完成的任务
echo "【最近完成的任务】"
grep "✅ 完成:" retrain.log | tail -5
echo ""

# 显示失败的任务
FAILED=$(grep "❌ 失败:" retrain.log | wc -l)
if [ $FAILED -gt 0 ]; then
    echo "【失败的任务: $FAILED】"
    grep "❌ 失败:" retrain.log
    echo ""
fi

# 显示是否已完成
if grep "重训完成" retrain.log > /dev/null; then
    echo "🎉 重训已完成！"
    echo ""
    echo "查看结果汇总:"
    tail -n 30 retrain.log
else
    echo "⏳ 训练进行中..."
    echo ""
    echo "实时监控: tail -f retrain.log"
    echo "检查进度: bash monitor_retrain.sh"
fi

echo ""
echo "=========================================="
