#!/usr/bin/env python3
"""
Compare results: Baseline vs Prompt1 (semantic only) vs Prompt1+Memory Bank (fusion)
"""

import pandas as pd
import numpy as np

# Baseline complete system (single-proto semantic + memory bank)
baseline_full = {
    'bottle': 97.87, 'cable': 87.69, 'capsule': 76.06, 'carpet': 99.92,
    'grid': 99.42, 'hazelnut': 97.07, 'leather': 100.00, 'metal_nut': 94.18,
    'pill': 93.54, 'screw': 62.65, 'tile': 100.00, 'toothbrush': 98.06,
    'transistor': 96.82, 'wood': 99.89, 'zipper': 97.02
}

# Baseline semantic-only (from your report)
baseline_semantic = {
    'bottle': 95.52, 'cable': 83.60, 'capsule': 73.69, 'carpet': 100.00,
    'grid': 98.87, 'hazelnut': 80.11, 'leather': 100.00, 'metal_nut': 85.56,
    'pill': 85.50, 'screw': 66.42, 'tile': 99.96, 'toothbrush': 69.58,
    'transistor': 89.60, 'wood': 98.82, 'zipper': 94.22
}

# Prompt1 semantic-only (multi-prototype)
prompt1_semantic = {
    'bottle': 98.25, 'cable': 86.00, 'capsule': 80.65, 'carpet': 100.00,
    'grid': 99.00, 'hazelnut': 91.14, 'leather': 100.00, 'metal_nut': 88.71,
    'pill': 86.12, 'screw': 79.57, 'tile': 99.93, 'toothbrush': 89.44,
    'transistor': 78.08, 'wood': 99.65, 'zipper': 95.46
}

# Prompt1 + Memory Bank (fusion) - from test results
prompt1_fusion = {
    'bottle': 98.17, 'cable': 86.00, 'capsule': 80.65, 'carpet': 100.00,
    'grid': 99.00, 'hazelnut': 88.55, 'leather': 100.00, 'metal_nut': 88.71,
    'pill': 86.12, 'screw': 56.99, 'tile': 99.93, 'toothbrush': 89.44,
    'transistor': 86.35, 'wood': 99.65, 'zipper': 95.46
}

# Create comparison dataframe
classes = list(baseline_full.keys())
df = pd.DataFrame({
    'Class': classes,
    'Baseline_Full': [baseline_full[c] for c in classes],
    'Baseline_Semantic': [baseline_semantic[c] for c in classes],
    'Prompt1_Semantic': [prompt1_semantic[c] for c in classes],
    'Prompt1_Fusion': [prompt1_fusion[c] for c in classes]
})

# Calculate improvements
df['Semantic_Improvement'] = df['Prompt1_Semantic'] - df['Baseline_Semantic']
df['Fusion_vs_Baseline_Full'] = df['Prompt1_Fusion'] - df['Baseline_Full']
df['Fusion_vs_Semantic'] = df['Prompt1_Fusion'] - df['Prompt1_Semantic']

print("="*120)
print("COMPREHENSIVE COMPARISON: MVTec k=2")
print("="*120)
print()

print("## Performance Summary")
print("-"*120)
print(f"{'Method':<30} {'Average AUROC':<15} {'Description':<60}")
print("-"*120)
print(f"{'Baseline Complete':<30} {df['Baseline_Full'].mean():>14.2f}% {'Single-proto semantic + Memory bank':<60}")
print(f"{'Baseline Semantic-Only':<30} {df['Baseline_Semantic'].mean():>14.2f}% {'Single-proto semantic (no memory bank)':<60}")
print(f"{'Prompt1 Semantic-Only':<30} {df['Prompt1_Semantic'].mean():>14.2f}% {'Multi-proto semantic (no memory bank)':<60}")
print(f"{'Prompt1 + Memory Bank':<30} {df['Prompt1_Fusion'].mean():>14.2f}% {'Multi-proto semantic + Memory bank (FUSION)':<60}")
print("-"*120)
print()

print("## Key Findings")
print("-"*120)

baseline_full_avg = df['Baseline_Full'].mean()
prompt1_fusion_avg = df['Prompt1_Fusion'].mean()
improvement = prompt1_fusion_avg - baseline_full_avg

if improvement > 0:
    print(f"✅ Multi-prototype + Memory Bank OUTPERFORMS baseline: +{improvement:.2f}%")
else:
    print(f"❌ Multi-prototype + Memory Bank underperforms baseline: {improvement:.2f}%")

print(f"\n   Baseline Complete:       {baseline_full_avg:.2f}%")
print(f"   Prompt1 + Memory Bank:   {prompt1_fusion_avg:.2f}%")
print(f"   Difference:              {improvement:+.2f}%")
print()

# Semantic improvement analysis
semantic_improvement = df['Prompt1_Semantic'].mean() - df['Baseline_Semantic'].mean()
print(f"📊 Semantic Branch Improvement: +{semantic_improvement:.2f}%")
print(f"   (Multi-proto vs Single-proto semantic)")
print()

# Fusion effect analysis  
fusion_effect = prompt1_fusion_avg - df['Prompt1_Semantic'].mean()
print(f"🔗 Memory Bank Contribution to Multi-proto: {fusion_effect:+.2f}%")
print(f"   (How much memory bank adds to multi-proto semantic)")
print()

print("## Per-Class Detailed Comparison")
print("-"*120)
print(f"{'Class':<12} {'Baseline':<10} {'B-Sem':<10} {'P1-Sem':<10} {'P1+MB':<10} {'Sem Δ':<10} {'Fusion Δ':<12} {'Status':<15}")
print("-"*120)

for _, row in df.iterrows():
    cls = row['Class']
    b_full = row['Baseline_Full']
    b_sem = row['Baseline_Semantic']
    p1_sem = row['Prompt1_Semantic']
    p1_fusion = row['Prompt1_Fusion']
    sem_imp = row['Semantic_Improvement']
    fusion_vs_full = row['Fusion_vs_Baseline_Full']
    fusion_vs_sem = row['Fusion_vs_Semantic']
    
    if fusion_vs_full > 1:
        status = "✓ Better"
    elif fusion_vs_full > -1:
        status = "≈ Similar"
    else:
        status = "✗ Worse"
    
    print(f"{cls:<12} {b_full:>9.2f}% {b_sem:>9.2f}% {p1_sem:>9.2f}% {p1_fusion:>9.2f}% "
          f"{sem_imp:>+9.2f}% {fusion_vs_full:>+10.2f}%  {status:<15}")

print("-"*120)
print(f"{'AVERAGE':<12} {baseline_full_avg:>9.2f}% {df['Baseline_Semantic'].mean():>9.2f}% "
      f"{df['Prompt1_Semantic'].mean():>9.2f}% {prompt1_fusion_avg:>9.2f}% "
      f"{semantic_improvement:>+9.2f}% {improvement:>+10.2f}%")
print("-"*120)
print()

# Analysis of classes where fusion helps/hurts
print("## Fusion Effect Analysis")
print("-"*120)

df_sorted = df.sort_values('Fusion_vs_Semantic', ascending=False)

print("\n### Classes where Memory Bank HELPS Multi-prototype:")
helps = df_sorted[df_sorted['Fusion_vs_Semantic'] > 1.0]
if len(helps) > 0:
    for _, row in helps.iterrows():
        print(f"  {row['Class']:<12} Semantic: {row['Prompt1_Semantic']:>6.2f}% → Fusion: {row['Prompt1_Fusion']:>6.2f}%  (+{row['Fusion_vs_Semantic']:>5.2f}%)")
else:
    print("  None (all approximately same or worse)")

print("\n### Classes where Memory Bank HURTS Multi-prototype:")
hurts = df_sorted[df_sorted['Fusion_vs_Semantic'] < -1.0]
if len(hurts) > 0:
    for _, row in hurts.iterrows():
        print(f"  {row['Class']:<12} Semantic: {row['Prompt1_Semantic']:>6.2f}% → Fusion: {row['Prompt1_Fusion']:>6.2f}%  ({row['Fusion_vs_Semantic']:>5.2f}%)")
else:
    print("  None")

print("\n### Observation:")
if len(hurts) > len(helps):
    print("⚠️  Memory bank is DEGRADING multi-prototype performance for most classes!")
    print("   Possible reasons:")
    print("   1. k=2 is too small to build robust memory bank")
    print("   2. Harmonic fusion with numerator=1 may be too conservative")
    print("   3. Memory bank and multi-prototype may be redundant")
elif len(helps) > len(hurts):
    print("✓ Memory bank is ENHANCING multi-prototype performance!")
else:
    print("≈ Memory bank has mixed effect on multi-prototype")

print()
print("="*120)

# Save to CSV
df.to_csv('result/prompt1/fusion_comparison_k2.csv', index=False, float_format='%.2f')
print("\nComparison saved to: result/prompt1/fusion_comparison_k2.csv")
