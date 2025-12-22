#!/bin/bash

# Phase 2.1: Full Dataset Validation
# 验证修复后的reliability indicators在完整数据集上的表现
# 
# 策略:
# - 所有MVTec类别 (15个)
# - 所有VisA类别 (12个)  
# - k=4 (与原实验一致)
# - task=cls (分类任务,语义indicators可用)
# - 并行度=3, 每个任务限制2线程
#
# 预计时间: 3-4小时 (27类 × 5-8分钟/类)

set -e  # Exit on error

# Activate environment
source ~/miniconda3/bin/activate prompt_ad

# Configuration
DEVICE_IDS=(0 0 0)  # 3个并行任务都用GPU 0
PARALLEL=3
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

OUTPUT_DIR="result_gate"
K_SHOT=4
TASK="cls"
SEED=111

# All MVTec classes
MVTEC_CLASSES=(
    "bottle" "cable" "capsule" "carpet" "grid"
    "hazelnut" "leather" "metal_nut" "pill" "screw"
    "tile" "toothbrush" "transistor" "wood" "zipper"
)

# All VisA classes  
VISA_CLASSES=(
    "candle" "capsules" "cashew" "chewinggum"
    "fryum" "macaroni1" "macaroni2"
    "pcb1" "pcb2" "pcb3" "pcb4" "pipe_fryum"
)

# Function to run single class
run_class() {
    local dataset=$1
    local class_name=$2
    local gpu_id=$3
    
    echo "[$(date '+%H:%M:%S')] Starting ${dataset}/${class_name} on GPU ${gpu_id}"
    
    CUDA_VISIBLE_DEVICES=$gpu_id python run_gate_experiment.py \
        --dataset $dataset \
        --class_name $class_name \
        --k-shot $K_SHOT \
        --task $TASK \
        --seed $SEED \
        --root-dir $OUTPUT_DIR \
        --checkpoint-dir $OUTPUT_DIR \
        --backbone ViT-B-16-plus-240 \
        --pretrained_dataset laion400m_e32 \
        --num-workers 2 \
        > logs/phase2_1_${dataset}_${class_name}.log 2>&1
    
    if [ $? -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] ✓ ${dataset}/${class_name} completed"
    else
        echo "[$(date '+%H:%M:%S')] ✗ ${dataset}/${class_name} FAILED"
        return 1
    fi
}

# Create log directory
mkdir -p logs

echo "================================================================================"
echo "Phase 2.1: Full Dataset Validation (Fixed Implementation)"
echo "================================================================================"
echo ""
echo "Configuration:"
echo "  - Datasets: MVTec-AD (15 classes) + VisA (12 classes) = 27 classes"
echo "  - K-shot: $K_SHOT"
echo "  - Task: $TASK"
echo "  - Seed: $SEED"
echo "  - Parallel jobs: $PARALLEL"
echo "  - Threads per job: 2 (OMP/MKL/OPENBLAS)"
echo "  - Output: $OUTPUT_DIR"
echo ""
echo "P0 Fixes Applied:"
echo "  ✓ Per-prompt semantic reliability (real implementation)"
echo "  ✓ Memory centroid normalization (support set statistics)"
echo "  ✓ Entropy degeneracy handling (MAD=0 check)"
echo ""
echo "Estimated time: 3-4 hours"
echo "================================================================================"
echo ""

# Combine all classes with dataset prefix
ALL_TASKS=()
for class_name in "${MVTEC_CLASSES[@]}"; do
    ALL_TASKS+=("mvtec:$class_name")
done
for class_name in "${VISA_CLASSES[@]}"; do
    ALL_TASKS+=("visa:$class_name")
done

TOTAL_TASKS=${#ALL_TASKS[@]}
echo "Total tasks: $TOTAL_TASKS"
echo ""

# Run tasks in parallel
task_idx=0
pids=()

for task in "${ALL_TASKS[@]}"; do
    IFS=':' read -r dataset class_name <<< "$task"
    
    # Wait if we've reached parallel limit
    while [ ${#pids[@]} -ge $PARALLEL ]; do
        for i in "${!pids[@]}"; do
            if ! kill -0 ${pids[$i]} 2>/dev/null; then
                unset 'pids[$i]'
            fi
        done
        pids=("${pids[@]}")  # Reindex array
        sleep 1
    done
    
    # Assign GPU in round-robin
    gpu_id=${DEVICE_IDS[$((task_idx % PARALLEL))]}
    
    # Run in background
    run_class $dataset $class_name $gpu_id &
    pids+=($!)
    
    task_idx=$((task_idx + 1))
    
    echo "Progress: $task_idx/$TOTAL_TASKS tasks launched"
    sleep 2  # Stagger launches slightly
done

# Wait for all remaining jobs
echo ""
echo "Waiting for all tasks to complete..."
for pid in "${pids[@]}"; do
    wait $pid
done

echo ""
echo "================================================================================"
echo "Phase 2.1 Full Dataset Execution Complete!"
echo "================================================================================"
echo ""
echo "Next steps:"
echo "1. Run correlation analysis:"
echo "   python phase2_1_oracle_correlation.py \\"
echo "     --use_real_data \\"
echo "     --datasets mvtec visa \\"
echo "     --k_shots 4 \\"
echo "     --task cls \\"
echo "     --result_dir result_gate"
echo ""
echo "2. Check detailed logs in logs/ directory"
echo "3. Per-sample data: result_gate/{dataset}/k_4/per_sample/cls/"
echo ""
echo "================================================================================"
