#!/bin/bash

# 运行剩余的4个缺失任务

CHECKPOINT_DIR="result/max_score"
OUTPUT_DIR="result/gate"
SEED=111
GPU_ID=0

echo "运行剩余的4个任务..."
echo ""

# Task 1: pcb4 cls
echo "[1/4] visa/pcb4 k=4 cls"
conda run -n prompt_ad python run_gate_experiment.py \
    --dataset visa --class_name pcb4 --k-shot 4 --task cls \
    --seed $SEED --gpu-id $GPU_ID \
    --checkpoint-dir $CHECKPOINT_DIR --root-dir $OUTPUT_DIR
echo "✓ 完成 1/4"
echo ""

# Task 2: pcb4 seg
echo "[2/4] visa/pcb4 k=4 seg"
conda run -n prompt_ad python run_gate_experiment.py \
    --dataset visa --class_name pcb4 --k-shot 4 --task seg \
    --seed $SEED --gpu-id $GPU_ID \
    --checkpoint-dir $CHECKPOINT_DIR --root-dir $OUTPUT_DIR
echo "✓ 完成 2/4"
echo ""

# Task 3: pipe_fryum cls
echo "[3/4] visa/pipe_fryum k=4 cls"
conda run -n prompt_ad python run_gate_experiment.py \
    --dataset visa --class_name pipe_fryum --k-shot 4 --task cls \
    --seed $SEED --gpu-id $GPU_ID \
    --checkpoint-dir $CHECKPOINT_DIR --root-dir $OUTPUT_DIR
echo "✓ 完成 3/4"
echo ""

# Task 4: pipe_fryum seg
echo "[4/4] visa/pipe_fryum k=4 seg"
conda run -n prompt_ad python run_gate_experiment.py \
    --dataset visa --class_name pipe_fryum --k-shot 4 --task seg \
    --seed $SEED --gpu-id $GPU_ID \
    --checkpoint-dir $CHECKPOINT_DIR --root-dir $OUTPUT_DIR
echo "✓ 完成 4/4"
echo ""

echo "============================================================"
echo "所有任务完成!"
echo "============================================================"
./monitor_gate_progress.sh
