#!/usr/bin/env python3
"""
Compare Multi-Prototype (prompt1) vs Baseline results.
Detailed analysis of improvements and degradations.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys


def load_results(base_dir, dataset, k_shot):
    """Load results from CSV file."""
    csv_path = Path(base_dir) / dataset / f"k_{k_shot}" / "csv" / "Seed_111-results.csv"
    
    if not csv_path.exists():
        print(f"Warning: {csv_path} not found")
        return None
    
    df = pd.read_csv(csv_path, index_col=0)
    # Clean class names
    df.index = df.index.str.replace(f'{dataset}-', '')
    return df


def compare_methods(baseline_dir="result/baseline", prompt1_dir="result/prompt1", 
                   dataset="mvtec", k_shot=2):
    """Compare baseline and prompt1 methods."""
    
    baseline = load_results(baseline_dir, dataset, k_shot)
    prompt1 = load_results(prompt1_dir, dataset, k_shot)
    
    if baseline is None or prompt1 is None:
        print("Error: Could not load both baseline and prompt1 results")
        return None
    
    # Create comparison dataframe
    comparison = pd.DataFrame()
    comparison['class'] = baseline.index
    comparison['baseline_i_roc'] = baseline['i_roc'].values
    comparison['prompt1_i_roc'] = prompt1['i_roc'].values
    comparison['diff_i_roc'] = comparison['prompt1_i_roc'] - comparison['baseline_i_roc']
    
    # Pixel-level if available
    if 'p_roc' in baseline.columns and 'p_roc' in prompt1.columns:
        comparison['baseline_p_roc'] = baseline['p_roc'].values
        comparison['prompt1_p_roc'] = prompt1['p_roc'].values
        comparison['diff_p_roc'] = comparison['prompt1_p_roc'] - comparison['baseline_p_roc']
    
    # Sort by improvement
    comparison = comparison.sort_values('diff_i_roc', ascending=False)
    
    return comparison


def print_detailed_comparison(comparison, k_shot=2):
    """Print detailed comparison analysis."""
    
    print("\n" + "="*100)
    print(f"DETAILED COMPARISON: Multi-Prototype (prompt1) vs Baseline (k={k_shot})")
    print("="*100)
    
    # Overall statistics
    baseline_avg = comparison['baseline_i_roc'].mean()
    prompt1_avg = comparison['prompt1_i_roc'].mean()
    diff_avg = prompt1_avg - baseline_avg
    
    print("\n## 1. Overall Performance (Image-level AUROC)")
    print("-" * 100)
    print(f"Baseline Average:       {baseline_avg:6.2f}%")
    print(f"Multi-Prototype Average: {prompt1_avg:6.2f}%")
    print(f"Overall Change:         {diff_avg:+6.2f}% {'✓ IMPROVEMENT' if diff_avg > 0 else '✗ DEGRADATION'}")
    print()
    
    # Improvements
    improvements = comparison[comparison['diff_i_roc'] > 0].copy()
    degradations = comparison[comparison['diff_i_roc'] < 0].copy()
    unchanged = comparison[comparison['diff_i_roc'] == 0].copy()
    
    print(f"Classes improved:  {len(improvements)}")
    print(f"Classes degraded:  {len(degradations)}")
    print(f"Classes unchanged: {len(unchanged)}")
    
    # Detailed table
    print("\n## 2. Per-Class Comparison")
    print("-" * 100)
    print(f"{'Class':<15} {'Baseline':>10} {'Prompt1':>10} {'Change':>10} {'Status':<20}")
    print("-" * 100)
    
    for _, row in comparison.iterrows():
        change = row['diff_i_roc']
        if change > 0:
            status = f"✓ +{change:.2f}%"
            marker = "↑"
        elif change < 0:
            status = f"✗ {change:.2f}%"
            marker = "↓"
        else:
            status = "= No change"
            marker = "="
        
        print(f"{row['class']:<15} {row['baseline_i_roc']:>9.2f}% {row['prompt1_i_roc']:>9.2f}% "
              f"{marker} {abs(change):>7.2f}% {status:<20}")
    
    print("-" * 100)
    print(f"{'AVERAGE':<15} {baseline_avg:>9.2f}% {prompt1_avg:>9.2f}% "
          f"{'↑' if diff_avg > 0 else '↓'} {abs(diff_avg):>7.2f}%")
    print("-" * 100)
    
    # Top improvements
    print("\n## 3. Top 5 Improvements")
    print("-" * 100)
    if len(improvements) > 0:
        print(f"{'Rank':<6} {'Class':<15} {'Baseline':>10} {'Prompt1':>10} {'Improvement':>12}")
        print("-" * 100)
        for idx, (_, row) in enumerate(improvements.head(5).iterrows(), 1):
            print(f"{idx:<6} {row['class']:<15} {row['baseline_i_roc']:>9.2f}% "
                  f"{row['prompt1_i_roc']:>9.2f}% +{row['diff_i_roc']:>10.2f}%")
    else:
        print("No improvements found")
    
    # Top degradations
    print("\n## 4. Top 5 Degradations")
    print("-" * 100)
    if len(degradations) > 0:
        print(f"{'Rank':<6} {'Class':<15} {'Baseline':>10} {'Prompt1':>10} {'Degradation':>12}")
        print("-" * 100)
        for idx, (_, row) in enumerate(degradations.tail(5).iloc[::-1].iterrows(), 1):
            print(f"{idx:<6} {row['class']:<15} {row['baseline_i_roc']:>9.2f}% "
                  f"{row['prompt1_i_roc']:>9.2f}% {row['diff_i_roc']:>11.2f}%")
    else:
        print("No degradations found")
    
    # Analysis by performance category
    print("\n## 5. Analysis by Magnitude")
    print("-" * 100)
    
    significant_improvement = comparison[comparison['diff_i_roc'] > 5.0]
    moderate_improvement = comparison[(comparison['diff_i_roc'] > 1.0) & (comparison['diff_i_roc'] <= 5.0)]
    minor_improvement = comparison[(comparison['diff_i_roc'] > 0) & (comparison['diff_i_roc'] <= 1.0)]
    
    significant_degradation = comparison[comparison['diff_i_roc'] < -5.0]
    moderate_degradation = comparison[(comparison['diff_i_roc'] < -1.0) & (comparison['diff_i_roc'] >= -5.0)]
    minor_degradation = comparison[(comparison['diff_i_roc'] < 0) & (comparison['diff_i_roc'] >= -1.0)]
    
    print(f"Significant Improvements (>5%):   {len(significant_improvement):2d} classes")
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
    
    print("\n### Strengths of Multi-Prototype:")
    if len(improvements) > 0:
        avg_improvement = improvements['diff_i_roc'].mean()
        print(f"- Improved {len(improvements)}/{len(comparison)} classes")
        print(f"- Average improvement on improved classes: +{avg_improvement:.2f}%")
        print(f"- Best improvement: {improvements.iloc[0]['class']} (+{improvements.iloc[0]['diff_i_roc']:.2f}%)")
    
    print("\n### Weaknesses of Multi-Prototype:")
    if len(degradations) > 0:
        avg_degradation = degradations['diff_i_roc'].mean()
        print(f"- Degraded {len(degradations)}/{len(comparison)} classes")
        print(f"- Average degradation on degraded classes: {avg_degradation:.2f}%")
        print(f"- Worst degradation: {degradations.iloc[-1]['class']} ({degradations.iloc[-1]['diff_i_roc']:.2f}%)")
    
    print("\n### Possible Reasons:")
    print("- Multi-prototype excels when:")
    print("  • Normal distribution is multi-modal (diverse normal states)")
    print("  • Abnormal patterns have distinct semantic types")
    print("- Baseline may be better when:")
    print("  • Single-mode normal distribution")
    print("  • Memory bank provides strong visual similarity signals")
    
    print("\n" + "="*100)


def save_comparison_csv(comparison, output_file, k_shot=2):
    """Save comparison to CSV."""
    comparison_output = comparison.copy()
    comparison_output.to_csv(output_file, index=False, float_format='%.2f')
    print(f"\nComparison CSV saved to: {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare Multi-Prototype vs Baseline')
    parser.add_argument('--baseline-dir', type=str, default='result/baseline')
    parser.add_argument('--prompt1-dir', type=str, default='result/prompt1')
    parser.add_argument('--dataset', type=str, default='mvtec')
    parser.add_argument('--k-shot', type=int, default=2)
    parser.add_argument('--output', type=str, default=None)
    
    args = parser.parse_args()
    
    comparison = compare_methods(args.baseline_dir, args.prompt1_dir, 
                                 args.dataset, args.k_shot)
    
    if comparison is not None:
        print_detailed_comparison(comparison, args.k_shot)
        
        if args.output is None:
            args.output = f"result/prompt1/{args.dataset}_detailed_comparison_k{args.k_shot}.csv"
        
        save_comparison_csv(comparison, args.output, args.k_shot)
