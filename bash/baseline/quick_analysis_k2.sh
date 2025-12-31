#!/bin/bash
# Quick analysis for k=2 on both MVTec and ViSA datasets

set -e

echo "=========================================="
echo "Quick Baseline Analysis (k=2)"
echo "=========================================="
echo ""

# Analyze MVTec
echo ">>> Analyzing MVTec dataset (k=2)..."
bash bash/baseline/run_all_analysis.sh mvtec 2

echo ""
echo "=========================================="
echo ""

# Analyze ViSA
echo ">>> Analyzing ViSA dataset (k=2)..."
bash bash/baseline/run_all_analysis.sh visa 2

echo ""
echo "=========================================="
echo "All datasets analyzed!"
echo "=========================================="
echo ""
echo "Results:"
echo "  MVTec: ./result/baseline/mvtec/k_2/analysis/"
echo "  ViSA:  ./result/baseline/visa/k_2/analysis/"
echo ""
echo "Summary reports:"
echo "  MVTec: ./result/baseline/mvtec/k_2/analysis/summary_report.csv"
echo "  ViSA:  ./result/baseline/visa/k_2/analysis/summary_report.csv"
