#!/bin/bash

# 监控gate实验进度

OUTPUT_DIR="result/gate"

echo "============================================================"
echo "Gate实验进度监控"
echo "============================================================"
echo ""

# 统计已完成的结果文件(只统计gate_results目录下的JSON)
completed_mvtec=$(find ${OUTPUT_DIR}/mvtec -path "*/gate_results/*.json" 2>/dev/null | wc -l)
completed_visa=$(find ${OUTPUT_DIR}/visa -path "*/gate_results/*.json" 2>/dev/null | wc -l)
total_completed=$((completed_mvtec + completed_visa))

echo "已完成任务:"
echo "  MVTec: ${completed_mvtec} / 90"
echo "  VisA:  ${completed_visa} / 72"
echo "  总计:  ${total_completed} / 162"
echo ""

# 进度百分比
progress=$(awk "BEGIN {printf \"%.1f\", ${total_completed}/162*100}")
echo "总进度: ${progress}%"
echo ""

# 最近完成的5个任务
echo "最近完成的任务:"
find ${OUTPUT_DIR} -path "*/gate_results/*.json" -type f 2>/dev/null | xargs ls -lt 2>/dev/null | head -5 | awk '{print "  " $9}'
echo ""

echo "============================================================"
