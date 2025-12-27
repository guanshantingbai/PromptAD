#!/bin/bash
#
# 5版本对比评估脚本
# 版本: Baseline, Prompt2, Ours_v1, Ours_v2, Ours_v3 (新增)
#

CLASSES=(
    "mvtec-toothbrush"
    "mvtec-capsule"
    "visa-pcb2"
    "mvtec-carpet"
    "mvtec-leather"
    "mvtec-screw"
)

# 类别分组标签
declare -A CLASS_GROUPS=(
    ["mvtec-toothbrush"]="Severe"
    ["mvtec-capsule"]="Severe"
    ["visa-pcb2"]="Severe"
    ["mvtec-carpet"]="Stable"
    ["mvtec-leather"]="Stable"
    ["mvtec-screw"]="Improved"
)

OUTPUT_DIR="analysis/5version_comparison"
mkdir -p "$OUTPUT_DIR"

echo "========================================================================"
echo "5版本对比评估"
echo "========================================================================"
echo "评估类别: ${#CLASSES[@]}"
echo "评估版本: 5 (Baseline, Prompt2, v1, v2, v3)"
echo "总评估数: $((${#CLASSES[@]} * 5))"
echo "输出目录: $OUTPUT_DIR"
echo "========================================================================"
echo ""

# 评估单个版本的函数
evaluate_version() {
    local dataset=$1
    local version_name=$2
    local checkpoint_dir=$3
    local n_pro=$4
    local n_pro_ab=$5
    
    local prefix="${dataset}_${version_name}"
    
    echo "  [$version_name] ..."
    
    python evaluate_extended_metrics.py \
        --dataset "$dataset" \
        --checkpoint-dir "$checkpoint_dir" \
        --n-pro $n_pro \
        --n-pro-ab $n_pro_ab \
        --output-dir "$OUTPUT_DIR" \
        --prefix "$prefix" \
        > /dev/null 2>&1
    
    if [ $? -eq 0 ]; then
        echo "     ✅ ${version_name}完成"
    else
        echo "     ❌ ${version_name}失败"
    fi
}

# 主循环
class_count=0
for dataset in "${CLASSES[@]}"; do
    class_count=$((class_count + 1))
    group=${CLASS_GROUPS[$dataset]}
    
    echo "========================================================================"
    echo "[$class_count/${#CLASSES[@]}] $dataset ($group)"
    echo "========================================================================"
    
    # [A] Baseline
    evaluate_version "$dataset" "Baseline" \
        "result/$dataset/k_3" 3 11
    
    # [B] Prompt2 (原始EMA)
    evaluate_version "$dataset" "Prompt2" \
        "result/$dataset/k_2" 1 4
    
    # [C] Ours_v1 (EMA + Rep0.05 + Margin)
    evaluate_version "$dataset" "Ours_v1" \
        "result/ours_fix_ema_rep_margin/$dataset/k_2" 1 4
    
    # [D] Ours_v2 (EMA + Rep0.10, No Margin)
    evaluate_version "$dataset" "Ours_v2" \
        "result/ema_rep_only/$dataset/k_2" 1 4
    
    # [E] Ours_v3 (EMA + Adaptive Rep) ⭐ 新增
    evaluate_version "$dataset" "Ours_v3" \
        "result/ema_adaptive_rep/$dataset/k_2" 1 4
    
    echo ""
done

echo "========================================================================"
echo "✅ 5版本对比评估完成！"
echo "========================================================================"
echo ""
echo "结果保存在: $OUTPUT_DIR"
echo ""
echo "下一步: 生成5版本对比分析"
echo "  python analyze_5version_comparison.py"
