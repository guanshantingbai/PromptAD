#!/bin/bash

OUTPUT_DIR="result/gate"
CHECKPOINT_DIR="result/max_score"

echo "检查缺失的任务..."
echo ""

missing_count=0

# VisA classes
VISA_CLASSES=(candle capsules cashew chewinggum fryum macaroni1 macaroni2 pcb1 pcb2 pcb3 pcb4 pipe_fryum)
K_SHOTS=(1 2 4)
TASKS=(cls seg)

for k in "${K_SHOTS[@]}"; do
    for class_name in "${VISA_CLASSES[@]}"; do
        for task in "${TASKS[@]}"; do
            result_file="${OUTPUT_DIR}/visa/k_${k}/gate_results/${class_name}_seed111_${task}.json"
            
            if [ ! -f "$result_file" ]; then
                echo "✗ 缺失: visa/${class_name} k=${k} ${task}"
                ((missing_count++))
            fi
        done
    done
done

echo ""
echo "总缺失任务数: ${missing_count}"
