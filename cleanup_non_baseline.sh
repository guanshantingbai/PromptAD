#!/bin/bash
# 清理所有非baseline的实验数据

echo "开始清理非baseline数据..."

# 1. 清理result目录中的非baseline数据
echo "清理 result/prompt1..."
rm -rf result/prompt1/

echo "清理 result/prompt1_fixed..."
rm -rf result/prompt1_fixed/

# 2. 清理测试结果CSV
echo "清理测试结果CSV..."
rm -f retest_all_cls_k*.csv
rm -f test_key_6_fixed_results.csv
rm -f retrain_all_cls_k*_execution.csv

# 3. 清理日志文件
echo "清理日志文件..."
rm -f logs/retrain_all_cls_k*.log
rm -f logs/retrain_all_cls_k*.pid
rm -f logs/retest_all_cls_k*.log

# 4. 清理临时文件
echo "清理临时文件..."
rm -f 0
rm -f check_consistency.py
rm -f compare_fix_impact.py
rm -f monitor_k4_training.sh
rm -f comprehensive_analysis.py

# 5. 清理测试和重训练脚本（保留源码脚本）
echo "清理测试脚本..."
rm -f retest_all_cls_k*.py
rm -f retrain_all_cls_k*.py
rm -f test_key_6_classes_fixed.py

echo "清理完成！"
echo ""
echo "保留的内容:"
echo "  - result/baseline/ (baseline数据)"
echo "  - 源代码文件 (PromptAD/, datasets/, utils/)"
echo "  - 原始训练/测试脚本 (train_cls.py, test_cls.py等)"
