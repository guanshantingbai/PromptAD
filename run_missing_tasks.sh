#!/bin/bash

# 运行缺失的gate实验任务

CHECKPOINT_DIR="result/max_score"
OUTPUT_DIR="result/gate"
SEED=111
GPU_ID=0

echo "============================================================"
echo "运行缺失的Gate实验任务"
echo "============================================================"
echo ""

# 缺失的任务列表
MISSING_TASKS=(
    "visa pcb3 4 seg"
    "visa pcb4 4 cls"
    "visa pcb4 4 seg"
    "visa pipe_fryum 4 cls"
    "visa pipe_fryum 4 seg"
)

total=${#MISSING_TASKS[@]}
completed=0

for task_info in "${MISSING_TASKS[@]}"; do
    read -r dataset class_name k task <<< "$task_info"
    ((completed++))
    
    echo ""
    echo "============================================================"
    echo "[${completed}/${total}] 运行: ${dataset}/${class_name} k=${k} ${task}"
    echo "============================================================"
    
    # 运行实验
    conda run -n prompt_ad python run_gate_experiment.py \
        --dataset "$dataset" \
        --class_name "$class_name" \
        --k-shot "$k" \
        --task "$task" \
        --seed "$SEED" \
        --gpu-id "$GPU_ID" \
        --checkpoint-dir "$CHECKPOINT_DIR" \
        --root-dir "$OUTPUT_DIR"
    
    if [ $? -eq 0 ]; then
        echo "✓ 完成: ${dataset}/${class_name} k=${k} ${task}"
    else
        echo "✗ 失败: ${dataset}/${class_name} k=${k} ${task}"
    fi
done

echo ""
echo "============================================================"
echo "所有缺失任务已完成"
echo "============================================================"
