#!/bin/bash
# Train multi-prototype PromptAD on MVTec with k=2 shot

DATASET="mvtec"
K_SHOT=2
SEED=111
EPOCH=100

# Training hyperparameters for multi-prototype
N_PRO=3        # 3 normal prototypes
N_PRO_AB=6     # 6 abnormal prototypes (more diverse abnormal directions)

echo "=========================================="
echo "Training Multi-Prototype PromptAD"
echo "Dataset: ${DATASET}, K-shot: ${K_SHOT}"
echo "Normal prototypes: ${N_PRO}"
echo "Abnormal prototypes: ${N_PRO_AB}"
echo "=========================================="
echo ""

# Single class for quick test
CLASS="carpet"

echo "Training: ${CLASS}"
python train_cls.py \
    --dataset ${DATASET} \
    --class_name ${CLASS} \
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
echo "Training completed!"
echo "Checkpoint saved to: result/mvtec/k_${K_SHOT}/checkpoint/"
