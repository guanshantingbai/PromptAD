#!/usr/bin/env python3
"""
Analyze whether multi-prototype semantic improvement translates to fused results.

Hypothesis: Multi-prototype improvement on semantic branch should faithfully 
reflect in the final fused results.
"""

import pandas as pd
import numpy as np

print("="*90)
print("Multi-Prototype Improvement Analysis")
print("="*90)

# Data from previous analysis
data = {
    'bottle': {
        'baseline_complete': 100.00,
        'prompt1_semantic': 100.00,  # From fair_comparison
        'prompt1_fused_buggy': 98.25,
        'prompt1_fused_fixed': 99.52,
    },
    'screw': {
        'baseline_complete': 79.57,
        'prompt1_semantic': 79.57,   # From fair_comparison
        'prompt1_fused_buggy': 56.99,
        'prompt1_fused_fixed': 68.96,
    },
}

# Calculate improvements and verify hypothesis
print("\n" + "="*90)
print(f"{'Class':<12} {'Baseline':>10} {'Semantic':>10} {'Fused(Fix)':>10} {'Semantic Δ':>12} {'Fused Δ':>12} {'Preserved?':>12}")
print("-"*90)

for class_name, metrics in data.items():
    baseline = metrics['baseline_complete']
    semantic = metrics['prompt1_semantic']
    fused = metrics['prompt1_fused_fixed']
    
    # Single-prototype is implicitly the baseline visual + baseline semantic
    # Multi-prototype changes only the semantic branch
    semantic_improvement = semantic - baseline
    fused_improvement = fused - baseline
    
    # Check if semantic improvement is preserved in fusion
    preservation_rate = (fused_improvement / semantic_improvement * 100) if semantic_improvement != 0 else float('inf')
    preserved = "✓" if abs(fused_improvement - semantic_improvement) < 2.0 else "✗"
    
    print(f"{class_name:<12} {baseline:>10.2f} {semantic:>10.2f} {fused:>10.2f} {semantic_improvement:>+11.2f}% {fused_improvement:>+11.2f}% {preserved:>12}")

print("\n" + "="*90)
print("Detailed Analysis:")
print("="*90)

print("\n1. BOTTLE - Perfect Case:")
print("   - Semantic improvement: +0.00% (already at 100%)")
print("   - Fused improvement: -0.48%")
print("   - Conclusion: At ceiling performance, small fusion loss is expected")

print("\n2. SCREW - Degradation Case:")
print("   - Semantic improvement: +0.00% (no improvement from multi-prototype)")
print("   - Fused improvement: -10.61% (68.96 vs 79.57)")
print("   - Baseline visual branch: probably strong")
print("   - Multi-prototype semantic: same as baseline")
print("   - Issue: Fusion degrades performance when both branches are weak")

print("\n" + "="*90)
print("Key Insights:")
print("="*90)
print("✓ When semantic branch improves, we need to check if it preserves in fusion")
print("✗ When semantic branch has NO improvement (screw), fusion can degrade")
print("⚠  Issue: We don't have cases yet where semantic branch actually improved!")
print("")
print("Need to test classes where:")
print("  - Prompt1 semantic > Baseline semantic (e.g., from previous fair_comparison)")
print("  - Then check if: Prompt1 fused ≥ Baseline complete")

print("\n" + "="*90)
print("From fair_comparison_semantic_only_k2.csv (Semantic-only results):")
print("="*90)

# Read the fair comparison results
try:
    df = pd.read_csv('result/prompt1/fair_comparison_semantic_only_k2.csv')
    print(f"\n{'Class':<15} {'Baseline Sem':>13} {'Prompt1 Sem':>13} {'Improvement':>13}")
    print("-"*90)
    
    # Show classes with positive semantic improvement
    improved = []
    for _, row in df.iterrows():
        if pd.notna(row.get('Semantic_Improvement_%')):
            improvement = row['Semantic_Improvement_%']
            if improvement > 0:
                improved.append({
                    'class': row['Class'],
                    'baseline': row['Baseline_Semantic'],
                    'prompt1': row['Prompt1_Semantic'],
                    'improvement': improvement
                })
    
    if improved:
        for item in sorted(improved, key=lambda x: x['improvement'], reverse=True)[:5]:
            print(f"{item['class']:<15} {item['baseline']:>13.2f} {item['prompt1']:>13.2f} {item['improvement']:>+12.2f}%")
        
        print("\n" + "="*90)
        print("RECOMMENDATION:")
        print("="*90)
        print("Test these classes with improved semantics to verify hypothesis:")
        for item in improved[:3]:
            print(f"  • {item['class']}: semantic improved by {item['improvement']:+.2f}%")
    else:
        print("No classes found with semantic improvement")
        
except FileNotFoundError:
    print("\nCannot find fair_comparison_semantic_only_k2.csv")
    print("Need to check which classes have semantic improvements from multi-prototype")

print("\n" + "="*90)
