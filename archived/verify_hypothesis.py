#!/usr/bin/env python3
"""
Verify hypothesis: Multi-prototype semantic improvement should preserve in fusion.
"""

import pandas as pd

print("="*100)
print("HYPOTHESIS VERIFICATION: Multi-Prototype Semantic Improvement → Fusion Improvement")
print("="*100)

# Read semantic-only comparison
semantic_df = pd.read_csv('result/prompt1/fair_comparison_semantic_only_k2.csv')
semantic_df.columns = ['class', 'baseline_semantic', 'prompt1_semantic', 'semantic_diff']

# Current tested fusion results (after bug fix)
tested_fusion = {
    'bottle': {
        'baseline_complete': 100.00,
        'prompt1_fused': 99.52,
    },
    'screw': {
        'baseline_complete': 79.57,
        'prompt1_fused': 68.96,
    },
}

print("\n" + "="*100)
print("TESTED CLASSES - Semantic Improvement vs Fusion Result:")
print("="*100)
print(f"{'Class':<12} {'Baseline':>10} {'Semantic':>12} {'Fused':>10} {'Sem Δ':>10} {'Fusion Δ':>12} {'Preserved?':>12}")
print("-"*100)

for class_name, fusion_data in tested_fusion.items():
    # Get semantic data
    sem_row = semantic_df[semantic_df['class'] == class_name].iloc[0]
    
    baseline_complete = fusion_data['baseline_complete']
    baseline_semantic = sem_row['baseline_semantic']
    prompt1_semantic = sem_row['prompt1_semantic']
    prompt1_fused = fusion_data['prompt1_fused']
    
    semantic_improvement = prompt1_semantic - baseline_semantic
    fusion_improvement = prompt1_fused - baseline_complete
    
    # Check preservation
    if semantic_improvement > 1.0:  # Only check if there's meaningful semantic improvement
        preserved = "✓" if fusion_improvement > 0 else "✗"
    else:
        preserved = "N/A"
    
    print(f"{class_name:<12} {baseline_complete:>10.2f} {prompt1_semantic:>12.2f} {prompt1_fused:>10.2f} "
          f"{semantic_improvement:>+9.2f}% {fusion_improvement:>+11.2f}% {preserved:>12}")

print("\n" + "="*100)
print("ANALYSIS:")
print("="*100)

print("\n1. BOTTLE (small semantic improvement):")
print("   • Baseline semantic: 95.52% → Prompt1 semantic: 98.25% (+2.73%)")
print("   • Baseline complete: 100.00% → Prompt1 fused: 99.52% (-0.48%)")
print("   • Analysis: Baseline already at ceiling (100%), can't improve further")
print("   • Conclusion: ✗ Cannot verify hypothesis (ceiling effect)")

print("\n2. SCREW (large semantic improvement):")
print("   • Baseline semantic: 66.42% → Prompt1 semantic: 79.57% (+13.15%)")
print("   • Baseline complete: 79.57% → Prompt1 fused: 68.96% (-10.61%)")
print("   • Analysis: Semantic improved +13.15%, but fusion degraded -10.61%")
print("   • Conclusion: ✗ HYPOTHESIS VIOLATED!")

print("\n" + "="*100)
print("CRITICAL FINDING:")
print("="*100)
print("For SCREW:")
print("  • Multi-prototype semantic branch: 79.57%")
print("  • Baseline complete (semantic+visual fusion): 79.57%")
print("  • Prompt1 complete (semantic+visual fusion): 68.96%")
print("")
print("This suggests:")
print("  ⚠  Baseline visual branch is WEAK for screw (otherwise complete would be > semantic)")
print("  ⚠  When visual branch is weak, multi-prototype+memory fusion degrades performance")
print("  ⚠  The +13.15% semantic improvement is LOST in fusion!")

print("\n" + "="*100)
print("HYPOTHESIS TEST RECOMMENDATIONS:")
print("="*100)
print("Need to test classes with STRONG baseline visual branch:")
print("")

# Find classes with strong visual contribution
print("Classes with large semantic improvements to test:")
top_improvements = semantic_df.nlargest(8, 'semantic_diff')
print(f"\n{'Class':<15} {'Baseline Sem':>13} {'Prompt1 Sem':>13} {'Improvement':>13} {'Priority':>10}")
print("-"*100)

for _, row in top_improvements.iterrows():
    class_name = row['class']
    improvement = row['semantic_diff']
    
    # Priority: high improvement + likely strong visual branch
    if class_name in ['toothbrush', 'hazelnut', 'capsule']:
        priority = "HIGH"
    elif improvement > 5:
        priority = "MEDIUM"
    else:
        priority = "LOW"
    
    print(f"{class_name:<15} {row['baseline_semantic']:>13.2f} {row['prompt1_semantic']:>13.2f} "
          f"{improvement:>+12.2f}% {priority:>10}")

print("\n" + "="*100)
print("CONCLUSION:")
print("="*100)
print("Current evidence: ✗ Hypothesis NOT confirmed")
print("")
print("Possible reasons:")
print("  1. Visual memory bank is not strong enough (only 2-shot, 450 patches)")
print("  2. Harmonic fusion may suppress improvements when one branch is weak")
print("  3. Visual features [2][3] may not be optimal for memory matching")
print("")
print("Next steps:")
print("  • Test hazelnut (+11.03% semantic) - likely has better visual branch")
print("  • Test toothbrush (+19.86% semantic) - largest improvement")
print("  • Analyze why visual branch degrades for screw")
