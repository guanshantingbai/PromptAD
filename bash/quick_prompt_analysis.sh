#!/bin/bash

# Manual Prompts 分析工具 - 快速使用示例

echo "=========================================="
echo "Manual Prompts Analysis - Quick Examples"
echo "=========================================="
echo ""

# 1. 查看所有类别的prompt概览
echo "1. Showing all classes prompt summary..."
python analysis_manual_prompts.py --mode summary
echo ""

# 2. 查看特定类别的详细prompt
echo "2. Showing detailed prompts for 'bottle'..."
python analysis_manual_prompts.py --mode detail --class bottle
echo ""

# 3. 导出完整prompt表格
echo "3. Exporting full prompt table to CSV..."
python analysis_manual_prompts.py --mode export --output result/manual_prompts_full_table.csv
echo ""

# 4. 在训练好的模型上运行贡献度分析（示例）
# 需要替换为实际的checkpoint路径
echo "4. Running contribution analysis on trained model (example)..."
echo "Example command:"
echo "python run_prompt_contribution_analysis.py \\"
echo "  --dataset mvtec \\"
echo "  --class bottle \\"
echo "  --k_shot 2 \\"
echo "  --seed 111 \\"
echo "  --ckpt output/mvtec/bottle/k_2/Seed_111_best.pth \\"
echo "  --task seg \\"
echo "  --top_k 20"
echo ""

echo "=========================================="
echo "Analysis Complete!"
echo "=========================================="
