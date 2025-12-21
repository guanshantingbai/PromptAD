#!/bin/bash
# Train all MVTec classes with multi-prototype PromptAD (k=2)

DATASET="mvtec"
K_SHOT=2
SEED=111
EPOCH=100

# Multi-prototype settings
N_PRO=3        # 3 normal prototypes for multi-modal normal manifold
N_PRO_AB=6     # 6 abnormal prototypes for structured abnormal directions

# MVTec classes
CLASSES=(
    "bottle" "cable" "capsule" "carpet" "grid"
    "hazelnut" "leather" "metal_nut" "pill" "screw"
    "tile" "toothbrush" "transistor" "wood" "zipper"
)

echo "=========================================="
echo "Training Multi-Prototype PromptAD"
echo "Dataset: ${DATASET}, K-shot: ${K_SHOT}"
echo "Normal prototypes: ${N_PRO}"
echo "Abnormal prototypes: ${N_PRO_AB}"
echo "Epochs: ${EPOCH}"
echo "=========================================="
echo ""

for class_name in "${CLASSES[@]}"; do
    echo "--------------------------------------"
    echo "Training: ${class_name}"
    echo "--------------------------------------"
    
    python train_cls.py \
        --dataset ${DATASET} \
        --class_name ${class_name} \
        --k-shot ${K_SHOT} \
        --seed ${SEED} \
        --n_ctx 4 \
        --n_ctx_ab 1 \
        --n_pro ${N_PRO} \
        --n_pro_ab ${N_PRO_AB} \
        --Epoch ${EPOCH} \
        --lr 0.002 \
        --momentum 0.9 \
        --weight_decay 0.0005 \
        --lambda1 0.001 \
        --resolution 256 \
        --batch-size 400 \
        --vis False \
        --root-dir ./result \
        --gpu-id 0
    
    echo ""
done

echo "=========================================="
echo "All training completed!"
echo "Checkpoints saved to: result/mvtec/k_${K_SHOT}/checkpoint/"
echo "=========================================="
