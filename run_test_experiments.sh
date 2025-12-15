#!/bin/bash

################################################################################
# PromptAD 实验配置批量训练脚本 - 测试版
# 仅运行少量任务以验证脚本逻辑
################################################################################

set -u
set -o pipefail

ROOT_DIR="result/backbone1"
BACKBONE="ViT-B-16"
PRETRAINED_DATASET="laion400m_e32"
EPOCH=1  # 测试版：只训练1个epoch
LR=0.002

GPU_IDS=(1 2 3)

# 测试版：只运行2个配置
EXP_CONFIGS=(
    "original"
    "qq_residual"
)

# 测试版：每个数据集只选2个类别
MVTEC_CLASSES=("bottle" "cable")
VISA_CLASSES=("candle" "cashew")

# 测试版：只测试 k=2
MVTEC_K_SHOTS=(2)
VISA_K_SHOTS=(2)

declare -A MVTEC_CLS_PARALLEL=( [2]=2 )
declare -A MVTEC_SEG_PARALLEL=( [2]=2 )
declare -A VISA_PARALLEL=( [2]=2 )

N_CTX=2      # 测试版：减小参数
N_PRO=2
N_CTX_AB=1
N_PRO_AB=2
BATCH_SIZE=4  # 测试版：减小batch size

LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"
MAIN_LOG="$LOG_DIR/run_test_experiments.log"

START_TIME=$(date +%s)

echo "========================================" | tee "$MAIN_LOG"
echo "PromptAD 实验脚本测试版"                 | tee -a "$MAIN_LOG"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$MAIN_LOG"
echo "========================================" | tee -a "$MAIN_LOG"

wait_for_slots() {
    local max_parallel=$1
    local gpu_id=$2
    local lock_file="$ROOT_DIR/locks/gpu${gpu_id}_parallel.lock"
    mkdir -p "$ROOT_DIR/locks"
    
    while true; do
        (
            flock -x 200
            local running=$(pgrep -f "train_.*\.py.*--gpu-id ${gpu_id}" | wc -l)
            [ "$running" -lt "$max_parallel" ] && exit 0 || exit 1
        ) 200>"$lock_file"
        [ $? -eq 0 ] && break
        sleep 3
    done
}

check_checkpoint() {
    local task=$1
    local dataset=$2
    local class=$3
    local k=$4
    local exp_config=$5
    local ck_path="$ROOT_DIR/${dataset}/k_${k}/checkpoint/${task}_${exp_config}_${class}_check_point.pt"
    [ -f "$ck_path" ]
}

run_training() {
    local task=$1
    local dataset=$2
    local class=$3
    local k=$4
    local exp_config=$5
    local parallel=$6
    local gpu_id=$7

    local script="train_${task,,}.py"
    local log_file="$LOG_DIR/${task}_${dataset}_${class}_k${k}_${exp_config}_gpu${gpu_id}.log"

    if check_checkpoint "$task" "$dataset" "$class" "$k" "$exp_config"; then
        echo "[SKIP] $task $dataset/$class k=$k config=$exp_config GPU=$gpu_id" | tee -a "$MAIN_LOG"
        return
    fi

    local lock_file="$ROOT_DIR/locks/gpu${gpu_id}_parallel.lock"
    
    (
        flock -x 200
        local count=$(pgrep -f "train_.*\.py.*--gpu-id ${gpu_id}" | wc -l)
        [ "$count" -ge "$parallel" ] && exit 1
        
        check_checkpoint "$task" "$dataset" "$class" "$k" "$exp_config" && exit 0
        
        echo "[START] $task $dataset/$class k=$k config=$exp_config GPU=$gpu_id" | tee -a "$MAIN_LOG"
        
        (
            nohup python "$script" \
                --dataset "$dataset" \
                --class_name "$class" \
                --k-shot "$k" \
                --exp_config "$exp_config" \
                --gpu-id "$gpu_id" \
                --backbone "$BACKBONE" \
                --pretrained_dataset "$PRETRAINED_DATASET" \
                --Epoch "$EPOCH" \
                --batch-size "$BATCH_SIZE" \
                --n_ctx "$N_CTX" \
                --n_pro "$N_PRO" \
                --n_ctx_ab "$N_CTX_AB" \
                --n_pro_ab "$N_PRO_AB" \
                --lr "$LR" \
                >> "$log_file" 2>&1 &
            pid=$!
            sleep 1
            wait "$pid"
            exit_code=$?
            if [ "$exit_code" -eq 0 ]; then
                echo "[SUCCESS] $task $dataset/$class k=$k config=$exp_config GPU=$gpu_id" | tee -a "$MAIN_LOG"
            else
                echo "[FAILED]  $task $dataset/$class k=$k config=$exp_config GPU=$gpu_id" | tee -a "$MAIN_LOG"
            fi
        ) &
        
    ) 200>"$lock_file" || { wait_for_slots "$parallel" "$gpu_id"; run_training "$@"; }
}

get_gpu_id() {
    local index=$1
    echo "${GPU_IDS[$((index % ${#GPU_IDS[@]}))]}"
}

TASK_COUNTER=0

for exp_config in "${EXP_CONFIGS[@]}"; do
    echo "" | tee -a "$MAIN_LOG"
    echo "配置: $exp_config" | tee -a "$MAIN_LOG"

    # MVTec CLS (测试版：只运行2个类别)
    for k in "${MVTEC_K_SHOTS[@]}"; do
        for class in "${MVTEC_CLASSES[@]}"; do
            gpu_id=$(get_gpu_id $TASK_COUNTER)
            wait_for_slots "${MVTEC_CLS_PARALLEL[$k]}" "$gpu_id"
            run_training "cls" "mvtec" "$class" "$k" "$exp_config" "${MVTEC_CLS_PARALLEL[$k]}" "$gpu_id"
            TASK_COUNTER=$((TASK_COUNTER + 1))
        done
    done
    
    # 等待完成
    for gpu_id in "${GPU_IDS[@]}"; do
        while [ $(pgrep -f "train_cls\.py.*--gpu-id ${gpu_id}" | wc -l) -gt 0 ]; do
            sleep 5
        done
    done
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "" | tee -a "$MAIN_LOG"
echo "测试完成！耗时: $((ELAPSED/60))m $((ELAPSED%60))s" | tee -a "$MAIN_LOG"
echo "总任务数: $TASK_COUNTER" | tee -a "$MAIN_LOG"
