#!/bin/bash
# Quick view of baseline analysis results

echo "=========================================="
echo "PromptAD Baseline Analysis Results"
echo "=========================================="
echo ""

# 1. Show failure mode table
echo "【1】Failure Mode Table (Top 10 + Bottom 5)"
echo "------------------------------------------"
head -11 result/baseline/baseline_analysis/combined_analysis/failure_mode_table.csv | column -t -s,
echo "..."
tail -6 result/baseline/baseline_analysis/combined_analysis/failure_mode_table.csv | column -t -s,
echo ""

# 2. Count by failure type
echo "【2】Failure Type Distribution"
echo "------------------------------------------"
tail -n +2 result/baseline/baseline_analysis/combined_analysis/failure_mode_table.csv | \
    cut -d',' -f10 | sort | uniq -c | sort -rn
echo ""

# 3. Key findings
echo "【3】Key Statistics"
echo "------------------------------------------"
echo "Total classes analyzed: 27 (15 MVTec + 12 ViSA)"
echo ""
echo "Margin correlation with Semantic:"
echo "  Pearson:  r = 0.613, p = 0.0007"
echo "  Spearman: ρ = 0.685, p = 0.00008"
echo ""
echo "Performance by Margin threshold:"
echo "  Margin < 0.05: 77.0% avg semantic"
echo "  Margin ≥ 0.10: 95.1% avg semantic"
echo ""

# 4. Show visualization
echo "【4】Visualization"
echo "------------------------------------------"
if command -v display &> /dev/null; then
    echo "Opening correlation plot..."
    display result/baseline/baseline_analysis/combined_analysis/margin_semantic_correlation.png &
else
    echo "Plot saved at: result/baseline/baseline_analysis/combined_analysis/margin_semantic_correlation.png"
    echo "Use image viewer to open it."
fi

echo ""
echo "=========================================="
echo "Full analysis files:"
echo "  • failure_mode_table.csv"
echo "  • margin_semantic_correlation.png"
echo "  • Individual class reports in:"
echo "    - result/baseline/baseline_analysis/mvtec/k_2/seed_111/results/"
echo "    - result/baseline/baseline_analysis/visa/k_2/seed_111/results/"
echo "=========================================="
