#!/bin/bash

# Train Purge3 (混合版本)
# 只需要重新训练 capsule（还原到 Purge1 状态）

OUTPUT_DIR="./result/baseline_reducedprompt3"
K_SHOT=2
GPU_ID=0

echo "=========================================="
echo "Training Purge3 - Capsule Only (Restored)"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

# 只训练 capsule（其他5个类别从 Purge2 复制）
echo ""
echo "Training: mvtec-capsule (k=$K_SHOT)"
echo "  - Restoring prompts: poke + squeezed with compression"
echo ""

python train_cls.py \
    --dataset mvtec \
    --class_name capsule \
    --k-shot $K_SHOT \
    --gpu-id $GPU_ID \
    --root-dir $OUTPUT_DIR

if [ $? -eq 0 ]; then
    echo "✅ Capsule training completed"
    
    # 从 Purge2 复制其他 5 个类别的结果
    echo ""
    echo "Copying other 5 classes from Purge2..."
    
    PURGE2_DIR="./result/baseline_reducedprompt2/mvtec/k_2"
    PURGE3_DIR="$OUTPUT_DIR/mvtec/k_2"
    
    # 创建目录
    mkdir -p "$PURGE3_DIR/checkpoint"
    mkdir -p "$PURGE3_DIR/csv"
    
    # 复制 checkpoint 和结果
    for cls in metal_nut pill cable screw transistor
    do
        echo "  - Copying $cls..."
        cp "$PURGE2_DIR/checkpoint/CLS-Seed_111-${cls}-check_point.pt" "$PURGE3_DIR/checkpoint/" 2>/dev/null
    done
    
    # 合并 CSV
    python -c "
import pandas as pd
import os

purge2_csv = '$PURGE2_DIR/csv/Seed_111-results.csv'
purge3_csv = '$PURGE3_DIR/csv/Seed_111-results.csv'

if os.path.exists(purge2_csv):
    df_purge2 = pd.read_csv(purge2_csv, index_col=0)
    df_purge3 = pd.read_csv(purge3_csv, index_col=0)
    
    # 从 Purge2 复制 5 个类别
    for cls in ['mvtec-metal_nut', 'mvtec-pill', 'mvtec-cable', 'mvtec-screw', 'mvtec-transistor']:
        if cls in df_purge2.index:
            df_purge3.loc[cls] = df_purge2.loc[cls]
    
    df_purge3.to_csv(purge3_csv, float_format='%.2f')
    print('  ✅ CSV merged successfully')
else:
    print('  ⚠️ Purge2 CSV not found')
"
    
    echo ""
    echo "=========================================="
    echo "✅ Purge3 created successfully!"
    echo "   - Capsule: Retrained (Purge1 prompts)"
    echo "   - Other 5: Copied from Purge2"
    echo "Results: $OUTPUT_DIR/mvtec/k_$K_SHOT/csv/Seed_111-results.csv"
    echo "=========================================="
else
    echo "❌ Capsule training failed"
fi
