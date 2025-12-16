#!/bin/bash
set -u
set -o pipefail

# ============ 配置参数 ============
ROOT_DIR="result/backbone1"
BACKBONE="ViT-B-16"
PRETRAINED_DATASET="laion400m_e32"
EPOCH=100
LR=0.002

# 只使用 GPU 1
GPU_IDS=(1)

# 实验配置列表
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

# 只跑 k=1
MVTEC_K_SHOTS=(1)

# 并行数配置（只会用到 [1]）
declare -A MVTEC_CLS_PARALLEL=( [1]=4 )

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

MAIN_LOG="$LOG_DIR/run_mvtec_k1_cls_gpu1.log"
MAIN_LOG_LOCK="$LOCK_DIR/main_log.lock"

RUN_USER="$(id -un)"
START_TIME=$(date +%s)

# ============ 全局单实例锁 ============
exec 9>"$LOCK_DIR/run_mvtec_k1_cls_gpu1.lock"
if ! flock -n 9; then
  echo "[ERROR] Another instance is already running."
  echo "        Lock: $LOCK_DIR/run_mvtec_k1_cls_gpu1.lock"
  exit 1
fi

# ============ 日志函数 ============
log_main() {
  local msg="$1"
  (
    flock -x 200
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $msg"
  ) 200>"$MAIN_LOG_LOCK" >> "$MAIN_LOG"
}

log_main "========================================"
log_main "只跑 MVTec CLS k=1（GPU1）"
log_main "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
log_main "ROOT_DIR: $ROOT_DIR"
log_main "GPU_IDS: ${GPU_IDS[*]}"
log_main "实验配置数: ${#EXP_CONFIGS[@]}"
log_main "RUN_USER: $RUN_USER"
log_main "========================================"

# ============ 辅助函数 ============

get_gpu_id() {
  local index="$1"
  echo "${GPU_IDS[$((index % ${#GPU_IDS[@]}))]}"
}

check_checkpoint() {
  local task="$1"     # cls / seg
  local dataset="$2"  # mvtec / visa
  local class="$3"
  local k="$4"
  local exp_config="$5"
  local ck_path="$ROOT_DIR/${dataset}/k_${k}/checkpoint/${task}_${exp_config}_${class}_check_point.pt"
  [[ -f "$ck_path" ]]
}

train_pids() {
  pgrep -u "$RUN_USER" -f "python .*train_.*\.py" 2>/dev/null || true
}

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

count_gpu_running() {
  local gpu_id="$1"
  local cnt=0
  local pid
  while read -r pid; do
    pid_args_contains_all "$pid" "--gpu-id $gpu_id" && cnt=$((cnt+1))
  done < <(train_pids)
  echo "$cnt"
}

count_stage_running_on_gpu() {
  local task="$1"
  local dataset="$2"
  local gpu_id="$3"
  local script="train_${task}.py"

  local cnt=0
  local pid
  while read -r pid; do
    pid_args_contains_all "$pid" "$script" "--dataset $dataset" "--gpu-id $gpu_id" && cnt=$((cnt+1))
  done < <(train_pids)
  echo "$cnt"
}

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

run_training() {
  local task="$1"
  local dataset="$2"
  local class="$3"
  local k="$4"
  local exp_config="$5"
  local parallel="$6"
  local gpu_id="$7"

  local script="train_${task}.py"
  local log_file="$LOG_DIR/${task}_${dataset}_${class}_k${k}_${exp_config}_gpu${gpu_id}.log"

  if check_checkpoint "$task" "$dataset" "$class" "$k" "$exp_config"; then
    log_main "[SKIP] $task $dataset/$class k=$k config=$exp_config GPU=$gpu_id (checkpoint exists)"
    return 0
  fi

  if is_same_job_running "$task" "$dataset" "$class" "$k" "$exp_config" "$gpu_id"; then
    log_main "[SKIP-RUNNING] $task $dataset/$class k=$k config=$exp_config GPU=$gpu_id (already running)"
    return 0
  fi

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

# ============ 执行：仅 MVTec CLS k=1 ============
TASK_COUNTER=0

for exp_config in "${EXP_CONFIGS[@]}"; do
  log_main ""
  log_main "================================================"
  log_main "实验配置: $exp_config"
  log_main "================================================"

  log_main "[MVTec CLS] k=1 only"
  k=1
  parallel="${MVTEC_CLS_PARALLEL[$k]}"
  log_main "  k=$k, parallel=$parallel, gpu=1"

  for class in "${MVTEC_CLASSES[@]}"; do
    gpu_id="$(get_gpu_id "$TASK_COUNTER")"   # 永远返回 1
    run_training "cls" "mvtec" "$class" "$k" "$exp_config" "$parallel" "$gpu_id"
    TASK_COUNTER=$((TASK_COUNTER + 1))
  done

  wait_stage_done "cls" "mvtec"
  log_main "[完成] exp_config=$exp_config 的 MVTec CLS k=1 已完成"
done

wait

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
log_main ""
log_main "========================================"
log_main "任务完成！(仅 MVTec CLS k=1, GPU1)"
log_main "总耗时: $((ELAPSED/3600))h $(((ELAPSED%3600)/60))m $((ELAPSED%60))s"
log_main "总任务数(调度计数): $TASK_COUNTER"
log_main "日志: $MAIN_LOG"
log_main "========================================"