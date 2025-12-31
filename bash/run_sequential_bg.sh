#!/bin/bash
# 后台顺序执行训练脚本：先 run_cls.py，完成后自动执行 run_seg.py
# 使用方法：nohup bash bash/run_sequential_bg.sh > logs/sequential.log 2>&1 &

set -e  # 遇到错误立即退出

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# 创建日志目录
mkdir -p logs

# 记录开始时间到主日志
echo "========================================" | tee -a logs/sequential.log
echo "开始时间: $(date)" | tee -a logs/sequential.log
echo "========================================" | tee -a logs/sequential.log

# 执行 run_cls.py
echo "" | tee -a logs/sequential.log
echo ">>> 阶段 1/2: 执行 run_cls.py (分类任务)" | tee -a logs/sequential.log
echo ">>> 日志文件: logs/run_cls.log" | tee -a logs/sequential.log
echo ">>> 开始时间: $(date)" | tee -a logs/sequential.log
echo "" | tee -a logs/sequential.log

python run_cls.py > logs/run_cls.log 2>&1

if [ $? -eq 0 ]; then
    echo "✅ run_cls.py 完成！时间: $(date)" | tee -a logs/sequential.log
else
    echo "❌ run_cls.py 执行失败，退出码: $?，时间: $(date)" | tee -a logs/sequential.log
    exit 1
fi

# 执行 run_seg.py
echo "" | tee -a logs/sequential.log
echo ">>> 阶段 2/2: 执行 run_seg.py (分割任务)" | tee -a logs/sequential.log
echo ">>> 日志文件: logs/run_seg.log" | tee -a logs/sequential.log
echo ">>> 开始时间: $(date)" | tee -a logs/sequential.log
echo "" | tee -a logs/sequential.log

python run_seg.py > logs/run_seg.log 2>&1

if [ $? -eq 0 ]; then
    echo "✅ run_seg.py 完成！时间: $(date)" | tee -a logs/sequential.log
else
    echo "❌ run_seg.py 执行失败，退出码: $?，时间: $(date)" | tee -a logs/sequential.log
    exit 1
fi

# 记录结束时间
echo "" | tee -a logs/sequential.log
echo "========================================" | tee -a logs/sequential.log
echo "全部完成时间: $(date)" | tee -a logs/sequential.log
echo "========================================" | tee -a logs/sequential.log
echo "" | tee -a logs/sequential.log
echo "查看各阶段日志：" | tee -a logs/sequential.log
echo "  分类: tail -f logs/run_cls.log" | tee -a logs/sequential.log
echo "  分割: tail -f logs/run_seg.log" | tee -a logs/sequential.log
