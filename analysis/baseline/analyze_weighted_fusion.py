#!/usr/bin/env python3
"""
Analyze weighted harmonic mean fusion results
Formula: fusion = 1 / (1/memory + alpha/semantic)
"""

import pandas as pd
import numpy as np
import os

csv_dir = "./result/test_alpha_weighted/visa/k_1/csv"

# Alpha values tested
alphas = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.2, 1.5, 2.0]

results = []

print("="*80)
print("Loading results from all alpha values...")
print("="*80)

for alpha in alphas:
    alpha_str = str(alpha).replace('.', '')
    csv_path = f"{csv_dir}/visa_alpha{alpha_str}.csv"
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        
        # Calculate average (excluding NaN values)
        avg_baseline = df['fusion_baseline_i_roc'].mean()
        avg_weighted = df['fusion_weighted_i_roc'].mean()
        avg_delta = df['delta_i_roc'].mean()
        avg_semantic = df['semantic_i_roc'].mean()
        avg_memory = df['memory_i_roc'].mean()
        
        results.append({
            'alpha': alpha,
            'fusion_baseline_i_roc': avg_baseline,
            'fusion_weighted_i_roc': avg_weighted,
            'delta_i_roc': avg_delta,
            'semantic_i_roc': avg_semantic,
            'memory_i_roc': avg_memory
        })
        print(f"✓ Alpha={alpha}: Loaded {len(df)} classes")
    else:
        print(f"✗ Alpha={alpha}: File not found")

# Create summary dataframe
summary_df = pd.DataFrame(results)

print("\n" + "="*80)
print("WEIGHTED HARMONIC MEAN FUSION ANALYSIS")
print("Formula: fusion = 1/(1/memory + alpha/semantic)")
print("="*80)
print(summary_df.to_string(index=False, float_format=lambda x: f'{x:.2f}'))
print("="*80)

# Find best alpha
best_idx = summary_df['fusion_weighted_i_roc'].idxmax()
best_alpha = summary_df.loc[best_idx, 'alpha']
best_score = summary_df.loc[best_idx, 'fusion_weighted_i_roc']
baseline_score = summary_df.loc[best_idx, 'fusion_baseline_i_roc']

print(f"\n🏆 BEST RESULT:")
print(f"   Alpha: {best_alpha}")
print(f"   Weighted Fusion AUROC: {best_score:.2f}")
print(f"   Baseline AUROC: {baseline_score:.2f}")
print(f"   Improvement: {best_score - baseline_score:+.2f}")

# Detailed observations
print("\n" + "="*80)
print("DETAILED OBSERVATIONS:")
print("="*80)
for i, row in summary_df.iterrows():
    trend = "✅" if row['delta_i_roc'] > 0 else "❌" if row['delta_i_roc'] < -0.1 else "➖"
    print(f"Alpha={row['alpha']:4.1f}: Baseline={row['fusion_baseline_i_roc']:5.2f}, "
          f"Weighted={row['fusion_weighted_i_roc']:5.2f}, "
          f"Δ={row['delta_i_roc']:+6.2f} {trend}")

# Key insights
print("\n" + "="*80)
print("KEY INSIGHTS:")
print("="*80)

avg_delta = summary_df['delta_i_roc'].mean()
positive_count = (summary_df['delta_i_roc'] > 0).sum()
negative_count = (summary_df['delta_i_roc'] < 0).sum()

print(f"1. Overall Average Δ: {avg_delta:+.2f}")
print(f"2. Results: {positive_count} improved, {negative_count} degraded")

# Find optimal range
optimal_alphas = summary_df[summary_df['delta_i_roc'] > 0]['alpha'].tolist()
if optimal_alphas:
    print(f"3. Alpha values with improvement: {optimal_alphas}")
else:
    print(f"3. No alpha value improves over baseline")

# Semantic vs Memory comparison
avg_semantic = summary_df.loc[0, 'semantic_i_roc']
avg_memory = summary_df.loc[0, 'memory_i_roc']
print(f"4. Branch strength: Semantic={avg_semantic:.2f}, Memory={avg_memory:.2f}")

if avg_semantic > avg_memory:
    print(f"   → Semantic branch is stronger, benefit from alpha > 1.0")
else:
    print(f"   → Memory branch is stronger, benefit from alpha < 1.0")

# Alpha interpretation
print("\n" + "="*80)
print("ALPHA INTERPRETATION:")
print("="*80)
print("• alpha=0.0: fusion = memory (ignore semantic completely)")
print("• alpha=0.5: memory has 2× weight compared to semantic")
print("• alpha=1.0: standard harmonic mean (equal weights) [BASELINE]")
print("• alpha=1.5: semantic has 1.5× weight compared to memory")
print("• alpha=2.0: semantic has 2× weight compared to memory")

print("\n" + "="*80)
print(f"CONCLUSION:")
print("="*80)
if best_alpha < 1.0:
    print(f"✓ Best strategy: REDUCE semantic weight (alpha={best_alpha})")
    print(f"✓ This suggests: Memory branch is more reliable than semantic")
elif best_alpha > 1.0:
    print(f"✓ Best strategy: INCREASE semantic weight (alpha={best_alpha})")
    print(f"✓ This suggests: Semantic branch is more reliable than memory")
else:
    print(f"✓ Best strategy: EQUAL weights (alpha=1.0, standard baseline)")
    print(f"✓ This suggests: Both branches are equally reliable")
print("="*80)
