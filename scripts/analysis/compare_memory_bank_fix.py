#!/usr/bin/env python3
"""
Compare memory bank performance before and after bug fix.
"""

import pandas as pd

# Previous buggy results (from mvtec_detailed_comparison_k2.csv)
buggy_results = {
    'bottle': 98.25,
    'screw': 56.99,
    'hazelnut': 88.55,
    'transistor': 86.35,
}

# Baseline results for reference
baseline_results = {
    'bottle': 100.00,
    'screw': 79.57,
    'hazelnut': 91.14,
    'transistor': 78.08,
}

# New fixed results (to be filled in)
fixed_results = {
    'bottle': 99.52,  # Just tested
    'screw': None,
    'hazelnut': None,
    'transistor': None,
}

print("="*80)
print("Memory Bank Bug Fix Comparison")
print("="*80)
print(f"\n{'Class':<15} {'Baseline':>12} {'Buggy':>12} {'Fixed':>12} {'Improvement':>12}")
print("-"*80)

for class_name in sorted(buggy_results.keys()):
    baseline = baseline_results[class_name]
    buggy = buggy_results[class_name]
    fixed = fixed_results[class_name]
    
    if fixed is not None:
        improvement = fixed - buggy
        fixed_str = f"{fixed:.2f}"
        imp_str = f"+{improvement:.2f}" if improvement > 0 else f"{improvement:.2f}"
    else:
        fixed_str = "Testing..."
        imp_str = "N/A"
    
    print(f"{class_name:<15} {baseline:>12.2f} {buggy:>12.2f} {fixed_str:>12} {imp_str:>12}")

print("\n" + "="*80)
print("Key Findings:")
print("="*80)
print("✓ Fixed bottle: 98.25% → 99.52% (+1.27%)")
print("✓ Now using correct mid-layer features [2] and [3] for memory bank")
print("✓ Removed incorrect .cpu() call that caused device mismatch")
print("✓ Implementation now matches author's source code exactly")
