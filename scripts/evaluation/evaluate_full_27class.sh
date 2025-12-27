#!/bin/bash
#
# 全类别v1/v2评估脚本（27类 × 3版本 = 81次评估）
# 版本: Prompt2 baseline, v1, v2
# 严格区分: semantic-only 和 fusion 结果
#

set -e

# MVTec 15类
MVTEC_CLASSES=(
    "carpet" "grid" "leather" "tile" "wood"
    "bottle" "cable" "capsule" "hazelnut" "metal_nut"
    "pill" "screw" "toothbrush" "transistor" "zipper"
)

# VisA 12类
VISA_CLASSES=(
    "candle" "capsules" "cashew" "chewinggum" "fryum"
    "macaroni1" "macaroni2" "pcb1" "pcb2" "pcb3"
    "pcb4" "pipe_fryum"
)

OUTPUT_DIR="analysis/full_27class_comparison"
mkdir -p "$OUTPUT_DIR"

MAX_WORKERS=2  # 并行数=2

# 版本配置 - 对应训练脚本中的独立目录
PROMPT2_VERSION="prompt2"  # 修正：使用实际存在的prompt2目录
V1_VERSION="v1_ema_rep05_margin01"
V2_VERSION="v2_ema_rep10_nomargin"

echo "======================================================================="
echo "全类别v1/v2评估计划"
echo "======================================================================="
echo "总类别数: 27 (MVTec 15 + VisA 12)"
echo "评估版本: 2 (v1, v2) - Prompt2已有结果，作为baseline对比"
echo "总评估数: 54"
echo "并行数: $MAX_WORKERS"
echo "输出目录: $OUTPUT_DIR"
echo ""
echo "Prompt2 baseline: result/$PROMPT2_VERSION/ (已有结果)"
echo "v1目录: result/$V1_VERSION/"
echo "v2目录: result/$V2_VERSION/"
echo "========================================================================"
echo ""

# 评估单个类别的函数
evaluate_one() {
    local full_name=$1       # 如 mvtec-carpet
    local version_name=$2
    local root_dir=$3
    
    # 拆分dataset和class_name
    local dataset=$(echo $full_name | cut -d'-' -f1)
    local class_name=$(echo $full_name | cut -d'-' -f2-)
    
    local log_file="logs/eval_full_27class/${full_name}_${version_name}.log"
    
    mkdir -p logs/eval_full_27class
    
    echo "[$(date '+%H:%M:%S')] 评估: $full_name ($version_name)" | tee -a logs/evaluate_full.log
    
    python evaluate_extended_metrics.py \
        --dataset "$dataset" \
        --class_name "$class_name" \
        --root-dir "$root_dir" \
        --version-tag "$version_name" \
        --k-shot 2 \
        --seed 111 \
        > $log_file 2>&1
    
    if [ $? -eq 0 ]; then
        echo "  ✅ $version_name 完成" | tee -a logs/evaluate_full.log
    else
        echo "  ❌ $version_name 失败" | tee -a logs/evaluate_full.log
    fi
}

# 导出函数供xargs使用
export -f evaluate_one
export OUTPUT_DIR

# 合并所有类别
ALL_CLASSES=()
for cls in "${MVTEC_CLASSES[@]}"; do
    ALL_CLASSES+=("mvtec-$cls")
done
for cls in "${VISA_CLASSES[@]}"; do
    ALL_CLASSES+=("visa-$cls")
done

echo "开始评估..." | tee logs/evaluate_full.log
start_time=$(date +%s)

# 为每个类别评估2个版本（v1和v2）
eval_count=0
total_count=$((${#ALL_CLASSES[@]} * 2))

for full_name in "${ALL_CLASSES[@]}"; do
    echo ""
    echo "========================================================================"
    echo "[$((eval_count / 2 + 1))/${#ALL_CLASSES[@]}] $full_name"
    echo "========================================================================"
    
    # 拆分dataset和class_name（如mvtec-carpet → mvtec, carpet）
    dataset=$(echo $full_name | cut -d'-' -f1)
    class_name=$(echo $full_name | cut -d'-' -f2-)
    
    # 只评估v1和v2（Prompt2已有结果，作为baseline对比）
    # v1
    evaluate_one "$full_name" "v1" "result/$V1_VERSION"
    eval_count=$((eval_count + 1))
    
    # v2
    evaluate_one "$full_name" "v2" "result/$V2_VERSION"
    eval_count=$((eval_count + 1))
    
    echo "进度: $eval_count / $total_count"
done

end_time=$(date +%s)
duration=$((end_time - start_time))

echo ""
echo "========================================================================"
echo "✅ 全类别评估完成！"
echo "========================================================================"
echo "总评估数: $total_count"
echo "总耗时: $((duration / 60))分 $((duration % 60))秒"
echo "平均每次: $((duration / total_count))秒"
echo ""
echo "结果保存在: $OUTPUT_DIR"
echo ""
echo "下一步: 生成全类别对比分析"
echo "  python analyze_full_27class.py"
echo "========================================================================"
