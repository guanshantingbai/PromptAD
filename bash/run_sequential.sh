#!/bin/bash
# 顺序执行训练脚本：先 run_cls.py，完成后自动执行 run_seg.py
# 使用方法：bash bash/run_sequential.sh

set -e  # 遇到错误立即退出

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# 创建日志目录
mkdir -p logs

# 记录开始时间
echo "========================================"
echo "开始时间: $(date)"
echo "========================================"

# 执行 run_cls.py
echo ""
echo ">>> 阶段 1/2: 执行 run_cls.py (分类任务)"
echo ">>> 日志文件: logs/run_cls.log"
echo ""

python run_cls.py > logs/run_cls.log 2>&1

if [ $? -eq 0 ]; then
    echo "✅ run_cls.py 完成！"
else
    echo "❌ run_cls.py 执行失败，退出码: $?"
    exit 1
fi

# 执行 run_seg.py
echo ""
echo ">>> 阶段 2/2: 执行 run_seg.py (分割任务)"
echo ">>> 日志文件: logs/run_seg.log"
echo ""

python run_seg.py > logs/run_seg.log 2>&1

if [ $? -eq 0 ]; then
    echo "✅ run_seg.py 完成！"
else
    echo "❌ run_seg.py 执行失败，退出码: $?"
    exit 1
fi

# 记录结束时间
echo ""
echo "========================================"
echo "全部完成时间: $(date)"
echo "========================================"
echo ""
echo "查看日志："
echo "  分类: tail -f logs/run_cls.log"
echo "  分割: tail -f logs/run_seg.log"
