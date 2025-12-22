#!/bin/bash

# ============================================================
# Gate实验批量运行脚本
# 功能：复用max_score实验的checkpoint，对所有数据集进行gate评估
# ============================================================

# 配置
CHECKPOINT_DIR="result/max_score"  # 已有checkpoint的目录
OUTPUT_DIR="result/gate"           # gate实验结果输出目录
SEED=111
GPU_ID=0

GPU_ID=0

# MVTec数据集类别
MVTEC_CLASSES=(
    "bottle" "cable" "capsule" "carpet" "grid"
    "hazelnut" "leather" "metal_nut" "pill" "screw"
    "tile" "toothbrush" "transistor" "wood" "zipper"
)

# VisA数据集类别
VISA_CLASSES=(
    "candle" "capsules" "cashew" "chewinggum" "fryum"
    "macaroni1" "macaroni2" "pcb1" "pcb2" "pcb3" "pcb4" "pipe_fryum"
)

# K值
K_SHOTS=(1 2 4)

# 任务类型
TASKS=("cls" "seg")

echo "============================================================"
echo "Gate实验批量运行 - 复用max_score checkpoint"
echo "============================================================"
echo "Checkpoint来源: ${CHECKPOINT_DIR}"
echo "结果输出目录: ${OUTPUT_DIR}"
echo "============================================================"
echo ""

# 统计
total_tasks=0
completed_tasks=0
skipped_tasks=0

# 计算总任务数
for dataset in "mvtec" "visa"; do
    if [ "$dataset" == "mvtec" ]; then
        classes=("${MVTEC_CLASSES[@]}")
    else
        classes=("${VISA_CLASSES[@]}")
    fi
    
    for k in "${K_SHOTS[@]}"; do
        for class_name in "${classes[@]}"; do
            for task in "${TASKS[@]}"; do
                ((total_tasks++))
            done
        done
    done
done

echo "总任务数: ${total_tasks}"
echo ""

# 运行实验
for dataset in "mvtec" "visa"; do
    if [ "$dataset" == "mvtec" ]; then
        classes=("${MVTEC_CLASSES[@]}")
    else
        classes=("${VISA_CLASSES[@]}")
    fi
    
    for k in "${K_SHOTS[@]}"; do
        for class_name in "${classes[@]}"; do
            for task in "${TASKS[@]}"; do
                ((completed_tasks++))
                
                # 检查外部checkpoint是否存在
                task_upper=$(echo "$task" | tr '[:lower:]' '[:upper:]')
                checkpoint_path="${CHECKPOINT_DIR}/${dataset}/k_${k}/checkpoint/${task_upper}-Seed_${SEED}-${class_name}-check_point.pt"
                
                if [ ! -f "$checkpoint_path" ]; then
                    echo "[${completed_tasks}/${total_tasks}] ✗ 跳过 ${dataset}/${class_name} k=${k} ${task} - checkpoint不存在"
                    ((skipped_tasks++))
                    continue
                fi
                
                echo ""
                echo "============================================================"
                echo "[${completed_tasks}/${total_tasks}] 运行: ${dataset}/${class_name} k=${k} ${task}"
                echo "============================================================"
                
                # 运行gate实验
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
        done
    done
done

echo ""
echo "============================================================"
echo "批量运行完成"
echo "============================================================"
echo "总任务: ${total_tasks}"
echo "完成: $((completed_tasks - skipped_tasks))"
echo "跳过: ${skipped_tasks}"
echo "============================================================"
