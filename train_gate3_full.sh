#!/bin/bash

# Gate3 全类别训练脚本
# 
# 在gate3分支上重新训练所有类别，验证修复后的训练效果
# 
# 主要改进：
# - Memory分支不再参与训练时的反向传播
# - 训练更符合原始设计理念
# - 理论上semantic branch的优化更纯粹

echo "================================================================================"
echo "Gate3 全类别训练"
echo "================================================================================"

# 配置
K_SHOT=4
SEED=111
EPOCH=10  # 与gate2保持一致，用于对比
GPU=0
OUTPUT_DIR="./result_gate3"

# 并行配置（与gate2保持一致）
PARALLEL=3
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

echo ""
echo "配置："
echo "  K-shot: $K_SHOT"
echo "  Epoch: $EPOCH"
echo "  Seed: $SEED"
echo "  GPU: $GPU"
echo "  输出目录: $OUTPUT_DIR"
echo "  并行任务数: $PARALLEL"
echo "  每任务线程数: 2"
echo ""

# 创建输出目录
mkdir -p $OUTPUT_DIR
mkdir -p logs

# MVTec 类别
MVTEC_CLASSES=(
    "bottle" "cable" "capsule" "carpet" "grid" 
    "hazelnut" "leather" "metal_nut" "pill" "screw" 
    "tile" "toothbrush" "transistor" "wood" "zipper"
)

# VisA 类别
VISA_CLASSES=(
    "candle" "capsules" "cashew" "chewinggum" "fryum"
    "macaroni1" "macaroni2" "pcb1" "pcb2" "pcb3"
    "pcb4" "pipe_fryum"
)

# 总任务数
TOTAL_TASKS=$((${#MVTEC_CLASSES[@]} + ${#VISA_CLASSES[@]}))

echo "================================================================================"
echo "开始训练 (共 $TOTAL_TASKS 个任务)"
echo "================================================================================"
echo ""

# 任务队列
declare -a TASK_QUEUE=()

# 添加MVTec任务
for class in "${MVTEC_CLASSES[@]}"; do
    TASK_QUEUE+=("mvtec:$class")
done

# 添加VisA任务
for class in "${VISA_CLASSES[@]}"; do
    TASK_QUEUE+=("visa:$class")
done

# 运行任务的函数
run_task() {
    local dataset=$1
    local class=$2
    local task_id=$3
    local total=$4
    
    local log_file="logs/gate3_${dataset}_${class}_k${K_SHOT}_seed${SEED}.log"
    
    echo "[$(date '+%H:%M:%S')] [$task_id/$total] 开始训练: $dataset/$class"
    
    python train_cls.py \
        --dataset $dataset \
        --class_name $class \
        --k-shot $K_SHOT \
        --Seed $SEED \
        --Epoch $EPOCH \
        --gpu-id $GPU \
        --root-dir $OUTPUT_DIR \
        --vis 0 \
        > $log_file 2>&1
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] [$task_id/$total] ✓ $dataset/$class 完成"
    else
        echo "[$(date '+%H:%M:%S')] [$task_id/$total] ✗ $dataset/$class 失败 (退出码: $exit_code)"
    fi
    
    return $exit_code
}

# 并行执行任务
running_jobs=0
task_counter=0
failed_tasks=()

for task in "${TASK_QUEUE[@]}"; do
    task_counter=$((task_counter + 1))
    
    IFS=':' read -r dataset class <<< "$task"
    
    # 等待空闲槽位
    while [ $running_jobs -ge $PARALLEL ]; do
        wait -n
        running_jobs=$((running_jobs - 1))
    done
    
    # 启动新任务
    run_task "$dataset" "$class" "$task_counter" "$TOTAL_TASKS" &
    running_jobs=$((running_jobs + 1))
    
    # 错开启动时间
    sleep 2
done

# 等待所有任务完成
echo ""
echo "等待所有任务完成..."
wait

echo ""
echo "================================================================================"
echo "训练完成！"
echo "================================================================================"
echo ""

# 统计结果
checkpoint_count=$(find $OUTPUT_DIR -name "*check_point.pt" 2>/dev/null | wc -l)
echo "生成的checkpoint数量: $checkpoint_count / $TOTAL_TASKS"
echo ""

# 检查失败的任务
if [ ${#failed_tasks[@]} -gt 0 ]; then
    echo "⚠ 失败的任务："
    for task in "${failed_tasks[@]}"; do
        echo "  - $task"
    done
    echo ""
fi

echo "输出目录: $OUTPUT_DIR"
echo "日志目录: logs/"
echo ""
echo "================================================================================"
echo "下一步："
echo "================================================================================"
echo ""
echo "1. 检查训练结果"
echo "   find $OUTPUT_DIR -name '*.csv' | head -5"
echo ""
echo "2. 与gate2结果对比"
echo "   - Gate2: result_gate/"
echo "   - Gate3: result_gate3/"
echo ""
echo "3. 运行gate实验对比（可选）"
echo "   python run_gate_experiment.py \\"
echo "     --dataset mvtec \\"
echo "     --class_name bottle \\"
echo "     --k_shot 4 \\"
echo "     --checkpoint_dir result_gate3/mvtec/k_4/checkpoint"
echo ""
echo "================================================================================"
