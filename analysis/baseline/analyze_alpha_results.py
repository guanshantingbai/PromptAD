#!/usr/bin/env python3
"""
Analyze alpha scaling results across different alpha values
"""

import pandas as pd
import os

csv_dir = "./result/test_alpha/visa/k_1/csv"

# Alpha values to analyze
alphas = [0.1, 0.2, 0.3, 0.5, 0.7, 1.5, 2.0]

results = []

for alpha in alphas:
    alpha_str = str(alpha).replace('.', '')
    csv_path = f"{csv_dir}/visa_alpha{alpha_str}.csv"
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        
        # Print columns for debugging
        if alpha == alphas[0]:
            print(f"DEBUG: Columns in CSV: {df.columns.tolist()}")
        
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
        
        # Calculate average (excluding NaN values)
        avg_baseline = df['fusion_baseline_i_roc'].mean()
        avg_scaled = df['fusion_scaled_i_roc'].mean()
        avg_delta = df['delta_i_roc'].mean()
        avg_semantic = df['semantic_i_roc'].mean()
        avg_memory = df['memory_i_roc'].mean()
        
        results.append({
            'alpha': alpha,
            'fusion_baseline_i_roc': avg_baseline,
            'fusion_scaled_i_roc': avg_scaled,
            'delta_i_roc': avg_delta,
            'semantic_i_roc': avg_semantic,
            'memory_i_roc': avg_memory
        })
    else:
        print(f"[WARNING] File not found: {csv_path}")

# Create summary dataframe
summary_df = pd.DataFrame(results)

print("=" * 80)
print("Alpha Scaling Analysis - VisA Dataset (k=1)")
print("=" * 80)
print(summary_df.to_string(index=False))
print("=" * 80)

# Find best alpha
best_idx = summary_df['fusion_scaled_i_roc'].idxmax()
best_alpha = summary_df.loc[best_idx, 'alpha']
best_score = summary_df.loc[best_idx, 'fusion_scaled_i_roc']
baseline_score = summary_df.loc[best_idx, 'fusion_baseline_i_roc']

print(f"\nBest Alpha: {best_alpha}")
print(f"Best Scaled AUROC: {best_score:.2f}")
print(f"Baseline AUROC: {baseline_score:.2f}")
print(f"Improvement: {best_score - baseline_score:+.2f}")

# Analyze trend
print("\n" + "=" * 80)
print("Observations:")
print("=" * 80)
for i, row in summary_df.iterrows():
    trend = "↑" if row['delta_i_roc'] > 0 else "↓" if row['delta_i_roc'] < 0 else "="
    print(f"Alpha={row['alpha']:.1f}: Baseline={row['fusion_baseline_i_roc']:.2f}, "
          f"Scaled={row['fusion_scaled_i_roc']:.2f}, "
          f"Δ={row['delta_i_roc']:+.2f} {trend}")

print("\n" + "=" * 80)
print("Insight:")
avg_delta = summary_df['delta_i_roc'].mean()
if avg_delta > 0:
    print(f"Overall Average Δ: {avg_delta:+.2f} (POSITIVE)")
    print("→ Larger alpha values IMPROVE performance (counterintuitive!)")
    print("→ This suggests: Semantic branch benefits from UPWEIGHTING, not suppression")
    print("→ Possible reason: semantic_scaled = alpha * semantic makes semantic MORE important")
else:
    print(f"Overall Average Δ: {avg_delta:+.2f} (NEGATIVE)")
    print("→ Larger alpha values DEGRADE performance")
    print("→ This suggests: Semantic branch benefits from suppression")
print("=" * 80)
