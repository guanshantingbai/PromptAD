#!/usr/bin/env python3
"""
Fair Comparison: Multi-Prototype Semantic vs Single-Prototype Semantic
Both methods use ONLY semantic discrimination (no memory bank).
"""

import pandas as pd
import numpy as np
from pathlib import Path


def create_fair_comparison():
    """
    Compare prompt1 (multi-prototype) vs baseline-semantic (single-prototype).
    Both use only semantic discrimination, so comparison is fair.
    """
    
    # Prompt1 (Multi-Prototype) k=2 results
    prompt1_data = {
        'bottle': 98.25, 'cable': 86.00, 'capsule': 80.65, 'carpet': 100.00,
        'grid': 99.00, 'hazelnut': 91.14, 'leather': 100.00, 'metal_nut': 88.71,
        'pill': 86.12, 'screw': 79.57, 'tile': 99.93, 'toothbrush': 89.44,
        'transistor': 78.08, 'wood': 99.65, 'zipper': 95.46
    }
    
    # Baseline Semantic-Only k=2 results (from your report)
    baseline_semantic_data = {
        'bottle': 95.52, 'cable': 83.60, 'capsule': 73.69, 'carpet': 100.00,
        'grid': 98.87, 'hazelnut': 80.11, 'leather': 100.00, 'metal_nut': 85.56,
        'pill': 85.50, 'screw': 66.42, 'tile': 99.96, 'toothbrush': 69.58,
        'transistor': 89.60, 'wood': 98.82, 'zipper': 94.22
    }
    
    # Create comparison dataframe
    comparison = pd.DataFrame({
        'class': list(prompt1_data.keys()),
        'baseline_semantic': [baseline_semantic_data[k] for k in prompt1_data.keys()],
        'multi_prototype': [prompt1_data[k] for k in prompt1_data.keys()]
    })
    
    comparison['diff'] = comparison['multi_prototype'] - comparison['baseline_semantic']
    comparison = comparison.sort_values('diff', ascending=False)
    
    return comparison


def print_fair_comparison(comparison):
    """Print detailed fair comparison analysis."""
    
    print("\n" + "="*100)
    print("FAIR COMPARISON: Multi-Prototype vs Single-Prototype Semantic (k=2, MVTec)")
    print("="*100)
    print("\n⚖️  Both methods use ONLY semantic discrimination (NO memory bank)")
    print("   This is a fair, apples-to-apples comparison!\n")
    
    # Overall statistics
    baseline_avg = comparison['baseline_semantic'].mean()
    prompt1_avg = comparison['multi_prototype'].mean()
    diff_avg = prompt1_avg - baseline_avg
    
    print("## 1. Overall Performance (Image-level AUROC)")
    print("-" * 100)
    print(f"Single-Prototype Semantic:  {baseline_avg:6.2f}%")
    print(f"Multi-Prototype Semantic:   {prompt1_avg:6.2f}%")
    print(f"Improvement:                {diff_avg:+6.2f}% {'🎉 SIGNIFICANT IMPROVEMENT' if diff_avg > 2 else '✓ IMPROVEMENT' if diff_avg > 0 else '✗ DEGRADATION'}")
    print()
    
    # Win/lose statistics
    improvements = comparison[comparison['diff'] > 0]
    degradations = comparison[comparison['diff'] < 0]
    unchanged = comparison[comparison['diff'] == 0]
    
    print(f"Classes improved:  {len(improvements)}/15 ({len(improvements)/15*100:.1f}%)")
    print(f"Classes degraded:  {len(degradations)}/15 ({len(degradations)/15*100:.1f}%)")
    print(f"Classes unchanged: {len(unchanged)}/15 ({len(unchanged)/15*100:.1f}%)")
    
    # Detailed table
    print("\n## 2. Per-Class Comparison")
    print("-" * 100)
    print(f"{'Class':<15} {'Baseline':>12} {'Multi-Proto':>12} {'Δ':>10} {'Status':<20}")
    print("-" * 100)
    
    for _, row in comparison.iterrows():
        change = row['diff']
        if change > 0:
            marker = "↑"
            status = "✓ Improved"
        elif change < 0:
            marker = "↓"
            status = "✗ Degraded"
        else:
            marker = "="
            status = "= Unchanged"
        
        print(f"{row['class']:<15} {row['baseline_semantic']:>11.2f}% {row['multi_prototype']:>11.2f}% "
              f"{marker} {change:>8.2f}% {status:<20}")
    
    print("-" * 100)
    print(f"{'AVERAGE':<15} {baseline_avg:>11.2f}% {prompt1_avg:>11.2f}% "
          f"{'↑' if diff_avg > 0 else '↓'} {diff_avg:>8.2f}%")
    print("-" * 100)
    
    # Top improvements
    print("\n## 3. Top 5 Improvements")
    print("-" * 100)
    print(f"{'Rank':<6} {'Class':<15} {'Baseline':>12} {'Multi-Proto':>12} {'Improvement':>12}")
    print("-" * 100)
    for idx, (_, row) in enumerate(improvements.head(5).iterrows(), 1):
        print(f"{idx:<6} {row['class']:<15} {row['baseline_semantic']:>11.2f}% "
              f"{row['multi_prototype']:>11.2f}% +{row['diff']:>10.2f}%")
    
    # Top degradations
    print("\n## 4. Top 5 Degradations")
    print("-" * 100)
    if len(degradations) > 0:
        print(f"{'Rank':<6} {'Class':<15} {'Baseline':>12} {'Multi-Proto':>12} {'Degradation':>12}")
        print("-" * 100)
        for idx, (_, row) in enumerate(degradations.tail(5).iloc[::-1].iterrows(), 1):
            print(f"{idx:<6} {row['class']:<15} {row['baseline_semantic']:>11.2f}% "
                  f"{row['multi_prototype']:>11.2f}% {row['diff']:>11.2f}%")
    else:
        print("No degradations!")
    
    # Analysis by magnitude
    print("\n## 5. Analysis by Improvement Magnitude")
    print("-" * 100)
    
    huge_improvement = comparison[comparison['diff'] > 10.0]
    significant_improvement = comparison[(comparison['diff'] > 5.0) & (comparison['diff'] <= 10.0)]
    moderate_improvement = comparison[(comparison['diff'] > 1.0) & (comparison['diff'] <= 5.0)]
    minor_improvement = comparison[(comparison['diff'] > 0) & (comparison['diff'] <= 1.0)]
    
    minor_degradation = comparison[(comparison['diff'] < 0) & (comparison['diff'] >= -1.0)]
    moderate_degradation = comparison[(comparison['diff'] < -1.0) & (comparison['diff'] >= -5.0)]
    significant_degradation = comparison[comparison['diff'] < -5.0]
    
    print(f"Huge Improvements (>10%):         {len(huge_improvement):2d} classes")
    if len(huge_improvement) > 0:
        print(f"  → {', '.join(huge_improvement['class'].tolist())}")
    
    print(f"Significant Improvements (5-10%): {len(significant_improvement):2d} classes")
    if len(significant_improvement) > 0:
        print(f"  → {', '.join(significant_improvement['class'].tolist())}")
    
    print(f"Moderate Improvements (1-5%):     {len(moderate_improvement):2d} classes")
    if len(moderate_improvement) > 0:
        print(f"  → {', '.join(moderate_improvement['class'].tolist())}")
    
    print(f"Minor Improvements (0-1%):        {len(minor_improvement):2d} classes")
    if len(minor_improvement) > 0:
        print(f"  → {', '.join(minor_improvement['class'].tolist())}")
    
    print()
    print(f"Minor Degradations (0-1%):        {len(minor_degradation):2d} classes")
    if len(minor_degradation) > 0:
        print(f"  → {', '.join(minor_degradation['class'].tolist())}")
    
    print(f"Moderate Degradations (1-5%):     {len(moderate_degradation):2d} classes")
    if len(moderate_degradation) > 0:
        print(f"  → {', '.join(moderate_degradation['class'].tolist())}")
    
    print(f"Significant Degradations (>5%):   {len(significant_degradation):2d} classes")
    if len(significant_degradation) > 0:
        print(f"  → {', '.join(significant_degradation['class'].tolist())}")
    
    # Key insights
    print("\n## 6. Key Insights")
    print("-" * 100)
    
    print("\n### 🎯 Multi-Prototype Semantic is BETTER Overall!")
    print(f"   Average improvement: +{diff_avg:.2f}%")
    print(f"   Win rate: {len(improvements)}/15 = {len(improvements)/15*100:.1f}%")
    
    print("\n### 💪 Strengths of Multi-Prototype:")
    if len(improvements) > 0:
        avg_improvement = improvements['diff'].mean()
        print(f"   - {len(improvements)} classes improved")
        print(f"   - Average gain on improved classes: +{avg_improvement:.2f}%")
        print(f"   - Best improvement: {improvements.iloc[0]['class']} (+{improvements.iloc[0]['diff']:.2f}%)")
        
        print("\n   Why multi-prototype works better:")
        print("   • Captures multi-modal normal distributions")
        print("   • Provides richer semantic representations")
        print("   • More robust to intra-class variance")
    
    print("\n### ⚠️  Weaknesses (if any):")
    if len(degradations) > 0:
        avg_degradation = degradations['diff'].mean()
        print(f"   - {len(degradations)} classes degraded")
        print(f"   - Average loss on degraded classes: {avg_degradation:.2f}%")
        print(f"   - Worst degradation: {degradations.iloc[-1]['class']} ({degradations.iloc[-1]['diff']:.2f}%)")
        
        print("\n   Possible reasons for degradation:")
        print("   • Some classes may have simple, single-mode distributions")
        print("   • More parameters → harder to optimize with limited data")
    else:
        print("   - No weaknesses! All classes improved or stayed same.")
    
    # Comparison with unfair baseline
    print("\n## 7. Why Previous Comparison Was Misleading")
    print("-" * 100)
    print("\n📊 Previous Comparison (UNFAIR):")
    print("   - Baseline: Single-Proto Semantic + Memory Bank = 94.30%")
    print("   - Prompt1:  Multi-Proto Semantic (no MB)      = 91.47%")
    print("   - Difference: -2.83% (misleading!)")
    print("\n   ❌ This compared: (Semantic + Visual) vs (Semantic only)")
    print("   ❌ The -2.83% was mainly due to removing memory bank!")
    
    print("\n✅ Current Comparison (FAIR):")
    print(f"   - Baseline: Single-Proto Semantic             = {baseline_avg:.2f}%")
    print(f"   - Prompt1:  Multi-Proto Semantic              = {prompt1_avg:.2f}%")
    print(f"   - Difference: {diff_avg:+.2f}% (true improvement!)")
    print("\n   ✓ This compares: Semantic vs Semantic")
    print("   ✓ Shows multi-prototype is genuinely better!")
    
    print("\n" + "="*100)


if __name__ == "__main__":
    comparison = create_fair_comparison()
    print_fair_comparison(comparison)
    
    # Save to CSV
    output_file = "result/prompt1/fair_comparison_semantic_only_k2.csv"
    comparison.to_csv(output_file, index=False, float_format='%.2f')
    print(f"\nFair comparison saved to: {output_file}\n")
