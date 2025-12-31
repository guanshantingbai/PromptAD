#!/bin/bash

# Compute MAP/LAP reliability metrics for baseline models
# Usage: bash analysis/baseline/MAP_LAP/run_analysis.sh <gpu_id>

GPU_ID=${1:-0}

echo "=========================================="
echo "MAP/LAP Reliability Metrics Analysis"
echo "=========================================="
echo "GPU: $GPU_ID"
echo ""

# Step 1: Compute reliability metrics
echo "Step 1: Computing reliability metrics..."
python analysis/baseline/MAP_LAP/compute_reliability_metrics.py \
    --datasets mvtec visa \
    --k-shot 2 \
    --seed 111 \
    --gpu-id $GPU_ID \
    --epsilon 0.05 \
    --output-dir ./result/baseline/baseline_analysis/MAP_LAP

if [ $? -ne 0 ]; then
    echo "Error: Metric computation failed"
    exit 1
fi

# Step 2: Generate visualizations
echo ""
echo "Step 2: Generating visualizations..."
python analysis/baseline/MAP_LAP/visualize_reliability.py \
    --metrics-path ./result/baseline/baseline_analysis/MAP_LAP/reliability_metrics_k2.csv \
    --failure-mode-path ./result/baseline/baseline_analysis/combined_analysis/failure_mode_table.csv \
    --output-dir ./result/baseline/baseline_analysis/MAP_LAP

if [ $? -ne 0 ]; then
    echo "Error: Visualization failed"
    exit 1
fi

# Step 3: Display summary
echo ""
echo "=========================================="
echo "Analysis Complete!"
echo "=========================================="
echo ""
echo "Output files:"
echo "  - reliability_metrics_k2.csv: Raw metrics for all classes"
echo "  - reliability_summary.txt: Text summary report"
echo "  - normal_side_risk.png: Normal-side risk indicators"
echo "  - consistency_metrics.png: MAP-LAP consistency analysis"
echo "  - geometric_metrics.png: Anchor geometry visualization"
echo ""
echo "Location: ./result/baseline/baseline_analysis/MAP_LAP/"
echo ""

# Quick preview of summary
echo "Quick Summary:"
head -n 30 ./result/baseline/baseline_analysis/MAP_LAP/reliability_summary.txt

echo ""
echo "=========================================="
echo "To view full report:"
echo "  cat ./result/baseline/baseline_analysis/MAP_LAP/reliability_summary.txt"
echo ""
echo "To view plots:"
echo "  eog ./result/baseline/baseline_analysis/MAP_LAP/*.png"
echo "=========================================="
