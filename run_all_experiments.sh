#!/bin/bash
################################################################################
# PromptAD 实验配置批量训练脚本（3 GPU 固定分配 + 并行槽位控制）
#
# 修正版要点：
# 1) 全局单实例锁：防止重复启动导致任务重复、并行失控
# 2) 等待阶段完成不再依赖参数顺序：用 ps args 做包含匹配（order-free）
# 3) 并行计数/槽位控制：只统计当前用户的 train_*.py 且匹配 --gpu-id
# 4) 去掉 per-GPU flock + 递归：单实例串行调度下过度设计且更易出错
################################################################################

set -u
set -o pipefail

# ============ 配置参数 ============
ROOT_DIR="result/backbone1"
BACKBONE="ViT-B-16"
PRETRAINED_DATASET="laion400m_e32"
EPOCH=100
LR=0.002

# GPU 配置（固定分配）
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

# ============ 目录与日志 ============
LOG_DIR="$ROOT_DIR/logs"
LOCK_DIR="$ROOT_DIR/locks"
mkdir -p "$LOG_DIR" "$LOCK_DIR"

MAIN_LOG="$LOG_DIR/run_all_experiments.log"
MAIN_LOG_LOCK="$LOCK_DIR/main_log.lock"

RUN_USER="$(id -un)"
START_TIME=$(date +%s)

# ============ 全局单实例锁（强烈建议保留） ============
# 防止你重复 nohup 启动两份脚本，导致重复实验和并行翻倍
exec 9>"$LOCK_DIR/run_all_experiments.lock"
if ! flock -n 9; then
  echo "[ERROR] Another instance of run_all_experiments.sh is already running."
  echo "        Lock: $LOCK_DIR/run_all_experiments.lock"
  exit 1
fi

# ============ 日志函数（防并发写互相打架） ============
log_main() {
  local msg="$1"
  (
    flock -x 200
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $msg"
  ) 200>"$MAIN_LOG_LOCK" >> "$MAIN_LOG"
}

log_main "========================================"
log_main "PromptAD 实验配置批量训练 (3 GPU 固定分配)"
log_main "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
log_main "ROOT_DIR: $ROOT_DIR"
log_main "GPU_IDS: ${GPU_IDS[*]}"
log_main "实验配置数: ${#EXP_CONFIGS[@]}"
log_main "RUN_USER: $RUN_USER"
log_main "========================================"

# ============ 辅助函数 ============

# 取得 GPU ID（固定轮询分配）
get_gpu_id() {
  local index="$1"
  echo "${GPU_IDS[$((index % ${#GPU_IDS[@]}))]}"
}

# checkpoint 检查（保持你原来的逻辑）
check_checkpoint() {
  local task="$1"     # cls / seg
  local dataset="$2"  # mvtec / visa
  local class="$3"
  local k="$4"
  local exp_config="$5"
  local ck_path="$ROOT_DIR/${dataset}/k_${k}/checkpoint/${task}_${exp_config}_${class}_check_point.pt"
  [[ -f "$ck_path" ]]
}

# 获取当前用户的 train_*.py 进程 PID 列表
train_pids() {
  # 只抓 python 启动的 train_*.py（避免把 grep/pstree 等算进去）
  pgrep -u "$RUN_USER" -f "python .*train_.*\.py" 2>/dev/null || true
}

# 判断某个 pid 的命令行是否包含指定片段（不依赖参数顺序）
pid_args_contains_all() {
  local pid="$1"; shift
  local args
  args="$(ps -p "$pid" -o args= 2>/dev/null || true)"
  [[ -n "$args" ]] || return 1

  local token
  for token in "$@"; do
    [[ "$args" == *"$token"* ]] || return 1
  done
  return 0
}

# 统计某张 GPU 上正在跑的训练进程数（只按 --gpu-id，不区分 dataset/task）
count_gpu_running() {
  local gpu_id="$1"
  local cnt=0
  local pid
  while read -r pid; do
    pid_args_contains_all "$pid" "--gpu-id $gpu_id" && cnt=$((cnt+1))
  done < <(train_pids)
  echo "$cnt"
}

# 统计某阶段（task+dataset）在某 GPU 上的训练进程数（用于阶段等待）
count_stage_running_on_gpu() {
  local task="$1"     # cls/seg
  local dataset="$2"  # mvtec/visa
  local gpu_id="$3"

  local script="train_${task}.py"
  local cnt=0
  local pid
  while read -r pid; do
    pid_args_contains_all "$pid" "$script" "--dataset $dataset" "--gpu-id $gpu_id" && cnt=$((cnt+1))
  done < <(train_pids)
  echo "$cnt"
}

# 判断“完全同参任务”是否已经在跑（防重复启动）
is_same_job_running() {
  local task="$1"
  local dataset="$2"
  local class="$3"
  local k="$4"
  local exp_config="$5"
  local gpu_id="$6"

  local script="train_${task}.py"
  local pid
  while read -r pid; do
    if pid_args_contains_all "$pid" \
      "$script" \
      "--dataset $dataset" \
      "--class_name $class" \
      "--k-shot $k" \
      "--exp_config $exp_config" \
      "--gpu-id $gpu_id"; then
      return 0
    fi
  done < <(train_pids)

  return 1
}

# 等待某 GPU 有空闲槽位（并行限制）
wait_for_slots() {
  local max_parallel="$1"
  local gpu_id="$2"

  while true; do
    local running
    running="$(count_gpu_running "$gpu_id")"
    if [[ "$running" -lt "$max_parallel" ]]; then
      return 0
    fi
    sleep 5
  done
}

# 等待某个阶段（task+dataset）在所有 GPU 上跑完
wait_stage_done() {
  local task="$1"
  local dataset="$2"

  log_main "[WAIT] 等待阶段完成: ${dataset^^} ${task^^}"
  local gpu_id
  for gpu_id in "${GPU_IDS[@]}"; do
    while true; do
      local c
      c="$(count_stage_running_on_gpu "$task" "$dataset" "$gpu_id")"
      if [[ "$c" -eq 0 ]]; then
        break
      fi
      sleep 10
    done
  done
}

# 启动训练（后台），并在结束后写 SUCCESS/FAILED 到 MAIN_LOG
run_training() {
  local task="$1"        # cls / seg
  local dataset="$2"     # mvtec / visa
  local class="$3"
  local k="$4"
  local exp_config="$5"
  local parallel="$6"
  local gpu_id="$7"

  local script="train_${task}.py"
  local log_file="$LOG_DIR/${task}_${dataset}_${class}_k${k}_${exp_config}_gpu${gpu_id}.log"

  # checkpoint 已存在就跳过
  if check_checkpoint "$task" "$dataset" "$class" "$k" "$exp_config"; then
    log_main "[SKIP] $task $dataset/$class k=$k config=$exp_config GPU=$gpu_id (checkpoint exists)"
    return 0
  fi

  # 如果同参任务已经在跑，也跳过（防重复启动）
  if is_same_job_running "$task" "$dataset" "$class" "$k" "$exp_config" "$gpu_id"; then
    log_main "[SKIP-RUNNING] $task $dataset/$class k=$k config=$exp_config GPU=$gpu_id (already running)"
    return 0
  fi

  # 等槽位
  wait_for_slots "$parallel" "$gpu_id"

  log_main "[START] $task $dataset/$class k=$k config=$exp_config GPU=$gpu_id"

  (
    {
      echo "========================================"
      echo "EXP_CONFIG=$exp_config"
      echo "TASK=$task DATASET=$dataset CLASS=$class"
      echo "K_SHOT=$k GPU_ID=$gpu_id"
      echo "START_TIME=$(date '+%Y-%m-%d %H:%M:%S')"
      echo "========================================"
      echo
    } > "$log_file"

    # 直接跑（外层脚本本身通常是 nohup 启的；这里不再双层 nohup，减少复杂性/诡异行为）
    python "$script" \
      --dataset "$dataset" \
      --class_name "$class" \
      --k-shot "$k" \
      --exp_config "$exp_config" \
      --gpu-id "$gpu_id" \
      --backbone "$BACKBONE" \
      --pretrained_dataset "$PRETRAINED_DATASET" \
      --root-dir "$ROOT_DIR" \
      --Epoch "$EPOCH" \
      --batch-size "$BATCH_SIZE" \
      --n_ctx "$N_CTX" \
      --n_pro "$N_PRO" \
      --n_ctx_ab "$N_CTX_AB" \
      --n_pro_ab "$N_PRO_AB" \
      --lr "$LR" \
      >> "$log_file" 2>&1

    ec=$?
    if [[ "$ec" -eq 0 ]]; then
      log_main "[SUCCESS] $task $dataset/$class k=$k config=$exp_config GPU=$gpu_id"
    else
      log_main "[FAILED]  $task $dataset/$class k=$k config=$exp_config GPU=$gpu_id (exit=$ec)"
    fi
    exit "$ec"
  ) &
}

# ============ 执行实验 ============
TASK_COUNTER=0

for exp_config in "${EXP_CONFIGS[@]}"; do
  log_main ""
  log_main "================================================"
  log_main "实验配置: $exp_config"
  log_main "================================================"

  # ---- MVTec CLS ----
  log_main "[MVTec CLS]"
  for k in "${MVTEC_K_SHOTS[@]}"; do
    parallel="${MVTEC_CLS_PARALLEL[$k]}"
    log_main "  k=$k, parallel=$parallel"
    for class in "${MVTEC_CLASSES[@]}"; do
      gpu_id="$(get_gpu_id "$TASK_COUNTER")"
      run_training "cls" "mvtec" "$class" "$k" "$exp_config" "$parallel" "$gpu_id"
      TASK_COUNTER=$((TASK_COUNTER + 1))
    done
  done
  wait_stage_done "cls" "mvtec"

  # ---- MVTec SEG ----
  log_main "[MVTec SEG]"
  for k in "${MVTEC_K_SHOTS[@]}"; do
    parallel="${MVTEC_SEG_PARALLEL[$k]}"
    log_main "  k=$k, parallel=$parallel"
    for class in "${MVTEC_CLASSES[@]}"; do
      gpu_id="$(get_gpu_id "$TASK_COUNTER")"
      run_training "seg" "mvtec" "$class" "$k" "$exp_config" "$parallel" "$gpu_id"
      TASK_COUNTER=$((TASK_COUNTER + 1))
    done
  done
  wait_stage_done "seg" "mvtec"

  # ---- VisA CLS ----
  log_main "[VisA CLS]"
  for k in "${VISA_K_SHOTS[@]}"; do
    parallel="${VISA_PARALLEL[$k]}"
    log_main "  k=$k, parallel=$parallel"
    for class in "${VISA_CLASSES[@]}"; do
      gpu_id="$(get_gpu_id "$TASK_COUNTER")"
      run_training "cls" "visa" "$class" "$k" "$exp_config" "$parallel" "$gpu_id"
      TASK_COUNTER=$((TASK_COUNTER + 1))
    done
  done
  wait_stage_done "cls" "visa"

  # ---- VisA SEG ----
  log_main "[VisA SEG]"
  for k in "${VISA_K_SHOTS[@]}"; do
    parallel="${VISA_PARALLEL[$k]}"
    log_main "  k=$k, parallel=$parallel"
    for class in "${VISA_CLASSES[@]}"; do
      gpu_id="$(get_gpu_id "$TASK_COUNTER")"
      run_training "seg" "visa" "$class" "$k" "$exp_config" "$parallel" "$gpu_id"
      TASK_COUNTER=$((TASK_COUNTER + 1))
    done
  done
  wait_stage_done "seg" "visa"

  log_main "[完成] 实验配置 $exp_config 的所有阶段已完成"
done

# 等待所有后台训练子任务彻底结束（保险）
wait

# ============ 总结 ============
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
log_main ""
log_main "========================================"
log_main "所有实验完成！"
log_main "总耗时: $((ELAPSED/3600))h $(((ELAPSED%3600)/60))m $((ELAPSED%60))s"
log_main "总任务数(调度计数): $TASK_COUNTER"
log_main "日志目录: $LOG_DIR"
log_main "结果目录: $ROOT_DIR"
log_main "========================================"