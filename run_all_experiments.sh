#!/bin/bash

################################################################################
# PromptAD 实验配置批量训练脚本（3 GPU 并行版）
# - 支持所有实验配置（original + 6个变体）
# - 支持 MVTec 和 VisA 数据集
# - 支持分类和分割任务
# - 3个GPU固定分配任务，避免动态轮询
# - 双层 nohup 防 SIGHUP
# - 文件锁防止竞态条件和重复执行
################################################################################

# ========= Bash 安全设置 =========
set -u              # 未定义变量直接报错
set -o pipefail     # 管道中任一失败即失败

# ============ 配置参数 ============
ROOT_DIR="result/backbone1"
BACKBONE="ViT-B-16"
PRETRAINED_DATASET="laion400m_e32"
EPOCH=100
LR=0.002

# GPU 配置
GPU_IDS=(1 2 3)

# 实验配置列表（original + 6个变体）
EXP_CONFIGS=(
    "original"
    "qq_residual"
    "kk_residual"
    "vv_residual"
    "qq_no_residual"
    "kk_no_residual"
    "vv_no_residual"
)

# MVTec 数据集
MVTEC_CLASSES=(
    "carpet" "grid" "leather" "tile" "wood"
    "bottle" "cable" "capsule" "hazelnut" "metal_nut"
    "pill" "screw" "toothbrush" "transistor" "zipper"
)

# VisA 数据集
VISA_CLASSES=(
    "candle" "capsules" "cashew" "chewinggum" "fryum"
    "macaroni1" "macaroni2" "pcb1" "pcb2" "pcb3" "pcb4" "pipe_fryum"
)

# k-shot 配置
MVTEC_K_SHOTS=(1 2 4)
VISA_K_SHOTS=(1 2 4)

# 并行数配置（基于 k-shot）
# MVTec: k=1->6, k=2->4, k=4->3
# VisA: 一律 2
declare -A MVTEC_CLS_PARALLEL=( [1]=6 [2]=4 [4]=3 )
declare -A MVTEC_SEG_PARALLEL=( [1]=6 [2]=4 [4]=3 )
declare -A VISA_PARALLEL=( [1]=2 [2]=2 [4]=2 )

# Prompt 参数
N_CTX=4
N_PRO=3
N_CTX_AB=1
N_PRO_AB=4

# Batch size
BATCH_SIZE=8

# ============ 日志 ============
LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"
MAIN_LOG="$LOG_DIR/run_all_experiments.log"

START_TIME=$(date +%s)

echo "========================================" | tee -a "$MAIN_LOG"
echo "PromptAD 实验配置批量训练 (3 GPU)"      | tee -a "$MAIN_LOG"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$MAIN_LOG"
echo "ROOT_DIR: $ROOT_DIR"                    | tee -a "$MAIN_LOG"
echo "GPU_IDS: ${GPU_IDS[*]}"                 | tee -a "$MAIN_LOG"
echo "实验配置数: ${#EXP_CONFIGS[@]}"         | tee -a "$MAIN_LOG"
echo "========================================" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"

# ============ 辅助函数 ============

# 基于进程名的并行控制（带文件锁防止竞态条件）
wait_for_slots() {
    local max_parallel=$1
    local gpu_id=$2
    local lock_file="$ROOT_DIR/locks/gpu${gpu_id}_parallel.lock"
    
    # 创建锁文件目录
    mkdir -p "$ROOT_DIR/locks"
    
    while true; do
        # 使用文件锁保护计数检查
        (
            flock -x 200
            local running
            # 统计在指定 GPU 上运行的 train_*.py 进程数
            running=$(pgrep -f "train_.*\.py.*--gpu-id ${gpu_id}" | wc -l)
            if [ "$running" -lt "$max_parallel" ]; then
                exit 0  # 有空闲槽位
            else
                exit 1  # 已满
            fi
        ) 200>"$lock_file"
        
        if [ $? -eq 0 ]; then
            break  # 有空位,退出循环
        fi
        sleep 5
    done
}

# checkpoint 检查
check_checkpoint() {
    local task=$1
    local dataset=$2
    local class=$3
    local k=$4
    local exp_config=$5

    # 根据训练脚本的 checkpoint 路径检查
    local ck_path="$ROOT_DIR/${dataset}/k_${k}/checkpoint/${task}_${exp_config}_${class}_check_point.pt"
    [ -f "$ck_path" ]
}

# 启动训练（双层 nohup）
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

    # 使用文件锁保护任务启动,防止竞态条件导致重复
    local lock_file="$ROOT_DIR/locks/gpu${gpu_id}_parallel.lock"
    mkdir -p "$ROOT_DIR/locks"
    
    (
        # 获取排他锁
        flock -x 200
        
        # 再次检查槽位(双重检查锁)
        local count=$(pgrep -f "train_.*\.py.*--gpu-id ${gpu_id}" | wc -l)
        if [ "$count" -ge "$parallel" ]; then
            # 槽位已满,释放锁后重新等待
            exit 1
        fi
        
        # 二次检查 checkpoint（在锁内，防止重复启动）
        if check_checkpoint "$task" "$dataset" "$class" "$k" "$exp_config"; then
            echo "[SKIP-LOCKED] $task $dataset/$class k=$k config=$exp_config GPU=$gpu_id" | tee -a "$MAIN_LOG"
            exit 0
        fi
        
        echo "[START] $task $dataset/$class k=$k config=$exp_config GPU=$gpu_id" | tee -a "$MAIN_LOG"
        
        # 在锁保护下启动后台任务
        (
            echo "========================================" > "$log_file"
            echo "EXP_CONFIG=$exp_config" >> "$log_file"
            echo "TASK=$task DATASET=$dataset CLASS=$class" >> "$log_file"
            echo "K_SHOT=$k GPU_ID=$gpu_id" >> "$log_file"
            echo "START_TIME=$(date '+%Y-%m-%d %H:%M:%S')" >> "$log_file"
            echo "========================================" >> "$log_file"
            echo "" >> "$log_file"

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
            
            # 短暂等待确保进程被pgrep捕获
            sleep 1
            
            wait "$pid"
            exit_code=$?

            if [ "$exit_code" -eq 0 ]; then
                echo "[SUCCESS] $task $dataset/$class k=$k config=$exp_config GPU=$gpu_id" | tee -a "$MAIN_LOG"
            else
                echo "[FAILED]  $task $dataset/$class k=$k config=$exp_config GPU=$gpu_id (exit=$exit_code)" | tee -a "$MAIN_LOG"
            fi
        ) &
        
    ) 200>"$lock_file" || { wait_for_slots "$parallel" "$gpu_id"; run_training "$@"; }
}

# 获取 GPU ID（轮询分配）
get_gpu_id() {
    local index=$1
    echo "${GPU_IDS[$((index % ${#GPU_IDS[@]}))]}"
}

# ============ 执行实验 ============

# 全局任务计数器（用于 GPU 分配）
TASK_COUNTER=0

for exp_config in "${EXP_CONFIGS[@]}"; do
    echo "" | tee -a "$MAIN_LOG"
    echo "================================================" | tee -a "$MAIN_LOG"
    echo "实验配置: $exp_config" | tee -a "$MAIN_LOG"
    echo "================================================" | tee -a "$MAIN_LOG"

    # ---- MVTec CLS ----
    echo "[MVTec CLS]" | tee -a "$MAIN_LOG"
    for k in "${MVTEC_K_SHOTS[@]}"; do
        parallel=${MVTEC_CLS_PARALLEL[$k]}
        echo "  k=$k, parallel=$parallel" | tee -a "$MAIN_LOG"
        
        for class in "${MVTEC_CLASSES[@]}"; do
            gpu_id=$(get_gpu_id $TASK_COUNTER)
            wait_for_slots "$parallel" "$gpu_id"
            run_training "cls" "mvtec" "$class" "$k" "$exp_config" "$parallel" "$gpu_id"
            TASK_COUNTER=$((TASK_COUNTER + 1))
        done
    done
    
    # 等待当前 exp_config 的 MVTec CLS 任务完成
    echo "  等待 MVTec CLS 任务完成..." | tee -a "$MAIN_LOG"
    for gpu_id in "${GPU_IDS[@]}"; do
        while [ $(pgrep -f "train_cls\.py.*--gpu-id ${gpu_id}.*mvtec" | wc -l) -gt 0 ]; do
            sleep 10
        done
    done

    # ---- MVTec SEG ----
    echo "[MVTec SEG]" | tee -a "$MAIN_LOG"
    for k in "${MVTEC_K_SHOTS[@]}"; do
        parallel=${MVTEC_SEG_PARALLEL[$k]}
        echo "  k=$k, parallel=$parallel" | tee -a "$MAIN_LOG"
        
        for class in "${MVTEC_CLASSES[@]}"; do
            gpu_id=$(get_gpu_id $TASK_COUNTER)
            wait_for_slots "$parallel" "$gpu_id"
            run_training "seg" "mvtec" "$class" "$k" "$exp_config" "$parallel" "$gpu_id"
            TASK_COUNTER=$((TASK_COUNTER + 1))
        done
    done
    
    # 等待当前 exp_config 的 MVTec SEG 任务完成
    echo "  等待 MVTec SEG 任务完成..." | tee -a "$MAIN_LOG"
    for gpu_id in "${GPU_IDS[@]}"; do
        while [ $(pgrep -f "train_seg\.py.*--gpu-id ${gpu_id}.*mvtec" | wc -l) -gt 0 ]; do
            sleep 10
        done
    done

    # ---- VisA CLS ----
    echo "[VisA CLS]" | tee -a "$MAIN_LOG"
    for k in "${VISA_K_SHOTS[@]}"; do
        parallel=${VISA_PARALLEL[$k]}
        echo "  k=$k, parallel=$parallel" | tee -a "$MAIN_LOG"
        
        for class in "${VISA_CLASSES[@]}"; do
            gpu_id=$(get_gpu_id $TASK_COUNTER)
            wait_for_slots "$parallel" "$gpu_id"
            run_training "cls" "visa" "$class" "$k" "$exp_config" "$parallel" "$gpu_id"
            TASK_COUNTER=$((TASK_COUNTER + 1))
        done
    done
    
    # 等待当前 exp_config 的 VisA CLS 任务完成
    echo "  等待 VisA CLS 任务完成..." | tee -a "$MAIN_LOG"
    for gpu_id in "${GPU_IDS[@]}"; do
        while [ $(pgrep -f "train_cls\.py.*--gpu-id ${gpu_id}.*visa" | wc -l) -gt 0 ]; do
            sleep 10
        done
    done

    # ---- VisA SEG ----
    echo "[VisA SEG]" | tee -a "$MAIN_LOG"
    for k in "${VISA_K_SHOTS[@]}"; do
        parallel=${VISA_PARALLEL[$k]}
        echo "  k=$k, parallel=$parallel" | tee -a "$MAIN_LOG"
        
        for class in "${VISA_CLASSES[@]}"; do
            gpu_id=$(get_gpu_id $TASK_COUNTER)
            wait_for_slots "$parallel" "$gpu_id"
            run_training "seg" "visa" "$class" "$k" "$exp_config" "$parallel" "$gpu_id"
            TASK_COUNTER=$((TASK_COUNTER + 1))
        done
    done
    
    # 等待当前 exp_config 的 VisA SEG 任务完成
    echo "  等待 VisA SEG 任务完成..." | tee -a "$MAIN_LOG"
    for gpu_id in "${GPU_IDS[@]}"; do
        while [ $(pgrep -f "train_seg\.py.*--gpu-id ${gpu_id}.*visa" | wc -l) -gt 0 ]; do
            sleep 10
        done
    done
    
    echo "[完成] 实验配置 $exp_config 的所有任务已完成" | tee -a "$MAIN_LOG"
done

# ============ 总结 ============
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "" | tee -a "$MAIN_LOG"
echo "========================================" | tee -a "$MAIN_LOG"
echo "所有实验完成！" | tee -a "$MAIN_LOG"
echo "总耗时: $((ELAPSED/3600))h $(((ELAPSED%3600)/60))m $((ELAPSED%60))s" | tee -a "$MAIN_LOG"
echo "总任务数: $TASK_COUNTER" | tee -a "$MAIN_LOG"
echo "日志目录: $LOG_DIR" | tee -a "$MAIN_LOG"
echo "结果目录: $ROOT_DIR" | tee -a "$MAIN_LOG"
echo "========================================" | tee -a "$MAIN_LOG"
