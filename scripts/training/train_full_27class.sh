#!/bin/bash
#
# 全类别v1/v2训练脚本（27类）
# 配置: 并行数=2，严格区分semantic和fusion结果
# 目的: 验证6类结论在全类别上的一致性
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

# 训练参数
EPOCH=100
SEED=111
MAX_WORKERS=2  # 并行数=2

# v1配置: EMA + Repulsion(0.05) + Margin(0.1)
# 目录独立，避免权重混用
V1_VERSION="v1_ema_rep05_margin01"
V1_LAMBDA1=0.001
V1_LAMBDA_REP=0.05
V1_LAMBDA_MARGIN=0.1

# v2配置: EMA + Repulsion(0.10), No Margin
# 目录独立，避免权重混用
V2_VERSION="v2_ema_rep10_nomargin"
V2_LAMBDA1=0.001
V2_LAMBDA_REP=0.10

# 创建日志目录
mkdir -p logs/full_27class_v1
mkdir -p logs/full_27class_v2

echo "========================================================================"
echo "全类别v1/v2训练计划"
echo "========================================================================"
echo "总类别数: 27 (MVTec 15类 + VisA 12类)"
echo "并行数: $MAX_WORKERS"
echo "训练轮次: $EPOCH"
echo ""
echo "v1配置: EMA + Repulsion(0.05) + Margin(0.1)"
echo "  输出目录: result/$V1_VERSION/"
echo "v2配置: EMA + Repulsion(0.10), No Margin"
echo "  输出目录: result/$V2_VERSION/"
echo "========================================================================"
echo ""

# 训练单个类别的函数
train_one_class() {
    local full_name=$1  # 如 mvtec-carpet
    local version=$2
    local lambda1=$3
    local lambda_rep=$4
    local lambda_margin=$5
    
    # 拆分dataset和class_name
    local dataset=$(echo $full_name | cut -d'-' -f1)      # mvtec
    local class_name=$(echo $full_name | cut -d'-' -f2-)  # carpet (可能包含多个-)
    
    mkdir -p logs/full_27class_${version}
    local log_prefix="logs/full_27class_${version}/${full_name}"
    
    echo "[$(date '+%H:%M:%S')] 开始训练: $full_name ($version)" | tee -a logs/train_full_${version}.log
    
    # 根据version选择配置
    if [ "$version" = "v1" ]; then
        # v1有margin loss，需要检查train_cls.py是否支持--lambda_margin
        python train_cls.py \
            --dataset "$dataset" \
            --class_name "$class_name" \
            --root-dir "./result/$V1_VERSION" \
            --lambda1 $lambda1 \
            --lambda_rep $lambda_rep \
            --Epoch $EPOCH \
            --seed $SEED \
            --k-shot 2 \
            > ${log_prefix}.log 2>&1
    else
        # v2没有margin loss
        python train_cls.py \
            --dataset "$dataset" \
            --class_name "$class_name" \
            --root-dir "./result/$V2_VERSION" \
            --lambda1 $lambda1 \
            --lambda_rep $lambda_rep \
            --Epoch $EPOCH \
            --seed $SEED \
            --k-shot 2 \
            > ${log_prefix}.log 2>&1
    fi
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        # 提取AUROC
        local auroc=$(grep "Image-AUROC:" ${log_prefix}.log | tail -1 | grep -oP '(?<=Image-AUROC:)\d+\.\d+')
        if [ -n "$auroc" ]; then
            echo "  ✅ $dataset 完成: AUROC=${auroc}%" | tee -a logs/train_full_${version}.log
        else
            echo "  ✅ $dataset 完成 (AUROC未找到)" | tee -a logs/train_full_${version}.log
        fi
    else
        echo "  ❌ $dataset 失败 (exit code: $exit_code)" | tee -a logs/train_full_${version}.log
    fi
    
    return $exit_code
}

# 批量并行训练函数（严格控制最多MAX_WORKERS个进程）
batch_train() {
    local version=$1
    local lambda1=$2
    local lambda_rep=$3
    local lambda_margin=$4
    shift 4
    local classes=("$@")
    
    local total=${#classes[@]}
    
    echo ""
    echo "========================================================================"
    echo "开始训练: $version (共 $total 类)"
    echo "========================================================================"
    
    local idx=0
    
    for dataset in "${classes[@]}"; do
        idx=$((idx + 1))
        
        # 严格等待：检查当前运行的train_cls.py进程数
        while true; do
            running=$(pgrep -f "train_cls.py" | wc -l)
            if [ $running -lt $MAX_WORKERS ]; then
                break
            fi
            sleep 3
            echo "  [等待] 当前运行: $running/$MAX_WORKERS，等待空闲slot..."
        done
        
        # 启动训练任务
        train_one_class "$dataset" "$version" $lambda1 $lambda_rep $lambda_margin &
        
        # 等待进程真正启动
        sleep 2
        running=$(pgrep -f "train_cls.py" | wc -l)
        
        echo "[$idx/$total] 已启动: $dataset (当前运行: $running/$MAX_WORKERS)"
    done
    
    # 等待所有任务完成
    echo "等待所有训练任务完成..."
    wait
    
    echo ""
    echo "========================================================================"
    echo "$version 训练完成"
    echo "========================================================================"
}

# 训练v1 (27类)
echo ""
echo "阶段1: 训练v1配置 (EMA + Rep + Margin)"
echo "------------------------------------------------------------------------"

# 合并所有类别
ALL_CLASSES=()
for cls in "${MVTEC_CLASSES[@]}"; do
    ALL_CLASSES+=("mvtec-$cls")
done
for cls in "${VISA_CLASSES[@]}"; do
    ALL_CLASSES+=("visa-$cls")
done

start_v1=$(date +%s)
batch_train "v1" $V1_LAMBDA1 $V1_LAMBDA_REP $V1_LAMBDA_MARGIN "${ALL_CLASSES[@]}"
end_v1=$(date +%s)
duration_v1=$((end_v1 - start_v1))

echo "v1训练耗时: $((duration_v1 / 60))分 $((duration_v1 % 60))秒" | tee -a logs/train_full_v1.log

# 训练v2 (27类)
echo ""
echo "阶段2: 训练v2配置 (EMA + Rep only)"
echo "------------------------------------------------------------------------"

start_v2=$(date +%s)
batch_train "v2" $V2_LAMBDA1 $V2_LAMBDA_REP "0" "${ALL_CLASSES[@]}"
end_v2=$(date +%s)
duration_v2=$((end_v2 - start_v2))

echo "v2训练耗时: $((duration_v2 / 60))分 $((duration_v2 % 60))秒" | tee -a logs/train_full_v2.log

# 总结
total_time=$((duration_v1 + duration_v2))
echo ""
echo "========================================================================"
echo "🎉 全类别训练完成！"
echo "========================================================================"
echo "v1耗时: $((duration_v1 / 60))分 $((duration_v1 % 60))秒"
echo "v2耗时: $((duration_v2 / 60))分 $((duration_v2 % 60))秒"
echo "总耗时: $((total_time / 60))分 $((total_time % 60))秒"
echo ""
echo "下一步: 运行全类别评估"
echo "  ./evaluate_full_27class.sh"
echo "========================================================================"
