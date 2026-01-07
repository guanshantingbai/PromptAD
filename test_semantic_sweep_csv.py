#!/usr/bin/env python3
"""
测试 semantic sweep CSV 输出
验证会创建两个CSV文件：
1. Seed_111-results.csv - 基础指标 + fusion_alpha_X.XX
2. Seed_111-semantic-sweep.csv - semantic_alpha_X.XX
"""

import os
import sys
sys.path.append(os.path.dirname(__file__))

from utils.csv_utils import save_metric

# MVTec classes
mvtec_classes = [
    'carpet', 'grid', 'leather', 'tile', 'wood',
    'bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut',
    'pill', 'screw', 'toothbrush', 'transistor', 'zipper'
]

# Test output directory
test_dir = './test_csv_output'
os.makedirs(test_dir, exist_ok=True)
test_csv = os.path.join(test_dir, 'Seed_111-results.csv')

print("Testing semantic sweep CSV output...")
print("="*70)

# Test different alpha values for carpet
alphas = [0.0, 0.05, 0.1, 0.15, 0.2]

for alpha in alphas:
    print(f"\nWriting results for alpha={alpha:.2f}...")
    
    # Simulate changing metrics with alpha
    test_results = {
        'i_roc': 97.71,
        'semantic_i_roc': 99.56 - alpha * 5,  # Simulate change with alpha
        'memory_i_roc': 99.88,
        'fusion_i_roc': 97.71 + alpha * 10,
    }
    
    save_metric(
        metrics=test_results,
        total_classes=mvtec_classes,
        class_name='carpet',
        dataset='mvtec',
        csv_path=test_csv,
        semantic_weight=alpha
    )

print("\n" + "="*70)
print("✅ Test completed!")
print("\nGenerated files:")
print(f"  1. {test_csv}")
print(f"  2. {test_csv.replace('-results.csv', '-semantic-sweep.csv')}")

# Display both CSVs
import pandas as pd

print("\n📊 Main Results CSV (fusion sweep):")
df_main = pd.read_csv(test_csv, index_col=0)
print(df_main.head())
print(f"\nShape: {df_main.shape}")
print(f"Columns: {list(df_main.columns)}")

semantic_csv = test_csv.replace('-results.csv', '-semantic-sweep.csv')
if os.path.exists(semantic_csv):
    print("\n📊 Semantic Sweep CSV (semantic_alpha_X.XX):")
    df_semantic = pd.read_csv(semantic_csv, index_col=0)
    print(df_semantic.head())
    print(f"\nShape: {df_semantic.shape}")
    print(f"Columns: {list(df_semantic.columns)}")

# Cleanup
import shutil
if os.path.exists(test_dir):
    print(f"\n🗑️  Cleaning up test directory: {test_dir}")
    shutil.rmtree(test_dir)
    print("✅ Done!")
