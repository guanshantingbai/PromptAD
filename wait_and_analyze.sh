#!/bin/bash

# Wait for experiment to complete and run analysis

echo "等待Phase 2.1实验完成..."
echo "监控进程: run_phase2_1_real.sh"
echo ""

# Wait for the process to finish
while pgrep -f "run_phase2_1_real.sh" > /dev/null; do
    sleep 10
done

echo "实验已完成，开始分析..."
echo ""

# Run correlation analysis
cd /home/zju/codes/AD/PromptAD
python phase2_1_oracle_correlation.py \
    --use_real_data \
    --datasets mvtec visa \
    --k_shots 4 \
    --task cls \
    --result_dir result_gate

echo ""
echo "运行验证脚本..."
python verify_phase2_1_fixed.py

echo ""
echo "所有分析完成!"
