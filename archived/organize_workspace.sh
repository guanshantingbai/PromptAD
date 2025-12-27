#!/bin/bash
# 项目目录整理脚本

echo "=== 开始整理项目目录 ==="

# 1. 创建目录结构
mkdir -p docs/reports
mkdir -p docs/guides
mkdir -p scripts/evaluation
mkdir -p scripts/training
mkdir -p scripts/analysis
mkdir -p scripts/debugging
mkdir -p scripts/utilities
mkdir -p archived

echo "✅ 目录结构已创建"

# 2. 移动文档文件
echo "📄 整理文档文件..."
mv *REPORT*.md docs/reports/ 2>/dev/null
mv *GUIDE*.md docs/guides/ 2>/dev/null
mv *README*.md docs/ 2>/dev/null
mv *ANALYSIS*.md docs/reports/ 2>/dev/null
mv DECISION_POINT*.md docs/reports/ 2>/dev/null
mv DIAGNOSTICS*.md docs/guides/ 2>/dev/null
mv EXPANSION*.md docs/guides/ 2>/dev/null
mv EXPERIMENT_PLAN.md docs/guides/ 2>/dev/null
mv INCONSISTENCY*.md docs/reports/ 2>/dev/null
mv ROOT_CAUSE*.md docs/reports/ 2>/dev/null

# 3. 移动评估脚本
echo "📊 整理评估脚本..."
mv evaluate_*.sh scripts/evaluation/ 2>/dev/null
mv evaluate_*.py scripts/evaluation/ 2>/dev/null

# 4. 移动训练脚本
echo "🏋️ 整理训练脚本..."
mv train_full_27class.sh scripts/training/ 2>/dev/null
mv retrain_*.sh scripts/training/ 2>/dev/null
mv run_baseline.sh scripts/training/ 2>/dev/null
mv run_batch_tests.sh scripts/training/ 2>/dev/null
mv quick_start*.sh scripts/training/ 2>/dev/null

# 5. 移动分析脚本
echo "🔍 整理分析脚本..."
mv analyze_*.py scripts/analysis/ 2>/dev/null
mv aggregate_*.py scripts/analysis/ 2>/dev/null
mv compare_*.py scripts/analysis/ 2>/dev/null
mv consolidate_*.py scripts/analysis/ 2>/dev/null
mv correct_analysis.py scripts/analysis/ 2>/dev/null
mv deep_analysis*.py scripts/analysis/ 2>/dev/null
mv diagnose_*.py scripts/analysis/ 2>/dev/null
mv estimate_*.py scripts/analysis/ 2>/dev/null
mv fair_semantic*.py scripts/analysis/ 2>/dev/null
mv generate_report.py scripts/analysis/ 2>/dev/null
mv prepare_*.py scripts/analysis/ 2>/dev/null
mv quick_comparison*.py scripts/analysis/ 2>/dev/null
mv quick_test*.py scripts/analysis/ 2>/dev/null

# 6. 移动调试脚本
echo "🐛 整理调试脚本..."
mv debug_*.py scripts/debugging/ 2>/dev/null
mv check_*.py scripts/debugging/ 2>/dev/null

# 7. 移动工具脚本
echo "🔧 整理工具脚本..."
mv cleanup*.sh scripts/utilities/ 2>/dev/null
mv install.sh scripts/utilities/ 2>/dev/null
mv monitor_*.sh scripts/utilities/ 2>/dev/null

# 8. 移动运行脚本
echo "🚀 整理运行脚本..."
mv run_all_experiments.py scripts/ 2>/dev/null
mv RUN_ALL*.md docs/guides/ 2>/dev/null

# 9. 保留核心文件在根目录
echo "✅ 核心文件保留在根目录:"
ls -1 *.py 2>/dev/null | grep -E "^(train_cls|run_cls|test)\.py$"

echo ""
echo "=== 整理完成 ==="
echo "目录结构："
echo "  docs/          - 所有文档"
echo "  docs/reports/  - 实验报告"
echo "  docs/guides/   - 使用指南"
echo "  scripts/       - 所有脚本"
echo "  scripts/evaluation/  - 评估脚本"
echo "  scripts/training/    - 训练脚本"
echo "  scripts/analysis/    - 分析脚本"
echo "  scripts/debugging/   - 调试脚本"
echo "  scripts/utilities/   - 工具脚本"
echo ""
echo "核心训练代码保留在根目录"
