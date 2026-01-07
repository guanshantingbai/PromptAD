#!/usr/bin/env python3
"""
快速测试 alpha sweep CSV 输出
验证不同 alpha 会创建不同的列
"""

import os
import sys
sys.path.append(os.path.dirname(__file__))

from utils.csv_utils import save_metric

# Simulate test results for different alpha values
test_results = {
    'i_roc': 97.71,
    'semantic_i_roc': 99.56,
    'memory_i_roc': 99.88,
    'fusion_i_roc': 97.71,  # This will go to fusion_alpha_X.XX column
}

# MVTec classes
mvtec_classes = [
    'carpet', 'grid', 'leather', 'tile', 'wood',
    'bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut',
    'pill', 'screw', 'toothbrush', 'transistor', 'zipper'
]

# Test output
test_csv = './test_alpha_sweep_output.csv'

print("Testing alpha sweep CSV output...")
print("="*70)

# Test different alpha values
alphas = [0.0, 0.05, 0.1, 0.15, 0.2]

for alpha in alphas:
    print(f"\nWriting results for alpha={alpha:.2f}...")
    
    # Simulate changing fusion_i_roc with alpha
    test_results['fusion_i_roc'] = 97.71 + alpha * 10  # Dummy change
    
    save_metric(
        metrics=test_results,
        total_classes=mvtec_classes,
        class_name='carpet',
        dataset='mvtec',
        csv_path=test_csv,
        semantic_weight=alpha
    )

print("\n" + "="*70)
print("✅ Test completed! Check the output:")
print(f"   {test_csv}")
print("\nExpected columns:")
print("  - i_roc (baseline)")
print("  - semantic_i_roc")
print("  - memory_i_roc")
print("  - fusion_alpha_0.00")
print("  - fusion_alpha_0.05")
print("  - fusion_alpha_0.10")
print("  - fusion_alpha_0.15")
print("  - fusion_alpha_0.20")
print("="*70)

# Display the CSV
import pandas as pd
df = pd.read_csv(test_csv, index_col=0)
print("\n📊 Result CSV:")
print(df.head())
print(f"\nShape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Cleanup
import os
if os.path.exists(test_csv):
    print(f"\n🗑️  Cleaning up test file: {test_csv}")
    os.remove(test_csv)
    print("✅ Done!")
