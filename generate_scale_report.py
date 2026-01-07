#!/usr/bin/env python3
"""
Scale Analysis Report Generator

读取 scale_stats.csv 并生成人类可读的分析报告
"""

import pandas as pd
import numpy as np
import argparse

def generate_report(csv_path):
    """
    生成scale分析报告
    """
    # Read CSV
    df = pd.read_csv(csv_path)
    
    print("\n" + "="*80)
    print("Scale Analysis Report")
    print("="*80)
    print(f"Dataset: {df['dataset'].iloc[0]}")
    print(f"K-shot: {df['k_shot'].iloc[0]}")
    print(f"Alpha: {df['alpha'].iloc[0]}")
    print(f"Total classes: {len(df)}")
    print("="*80)
    
    # Summary statistics
    print("\n📊 Overall Statistics")
    print("-" * 80)
    print(f"Average Semantic AUROC: {df['semantic_auroc'].mean():.2f} ± {df['semantic_auroc'].std():.2f}")
    print(f"Average Scale Ratio (sem/geom): {df['scale_ratio'].mean():.2f} ± {df['scale_ratio'].std():.2f}")
    print(f"Classes with balanced scales: {(df['warning'] == 'Balanced').sum()}/{len(df)}")
    
    # Discriminability analysis
    print("\n🔍 Discriminability Analysis (Mean Difference: Abnormal - Normal)")
    print("-" * 80)
    
    df['E_geom_diff'] = df['E_geom_abnormal_mean'] - df['E_geom_normal_mean']
    df['E_sem_diff'] = df['E_sem_abnormal_mean'] - df['E_sem_normal_mean']
    
    # Expected: E_geom_abnormal > E_geom_normal (positive diff)
    #           E_sem_abnormal < E_sem_normal (negative diff, less aligned with normal)
    
    df_sorted = df.sort_values('semantic_auroc', ascending=False)
    
    print(f"\n{'Class':<15} {'AUROC':>7} {'E_geom_diff':>12} {'E_sem_diff':>12} {'Signal':>10}")
    print("-" * 80)
    
    for _, row in df_sorted.iterrows():
        geom_signal = "✅ Good" if row['E_geom_diff'] > 0.01 else "⚠️  Weak" if row['E_geom_diff'] > 0 else "❌ Bad"
        sem_signal = "✅ Good" if row['E_sem_diff'] < -0.01 else "⚠️  Weak" if row['E_sem_diff'] < 0 else "❌ Bad"
        
        # Overall signal
        if geom_signal == "✅ Good" and sem_signal == "✅ Good":
            overall = "✅✅"
        elif "❌" in geom_signal or "❌" in sem_signal:
            overall = "❌"
        else:
            overall = "⚠️"
        
        print(f"{row['class']:<15} {row['semantic_auroc']:>7.2f} "
              f"{row['E_geom_diff']:>12.4f} {row['E_sem_diff']:>12.4f} "
              f"{overall:>10}")
    
    # Problem classes
    print("\n⚠️  Problem Classes (AUROC < 70)")
    print("-" * 80)
    
    problem_df = df[df['semantic_auroc'] < 70].sort_values('semantic_auroc')
    
    if len(problem_df) > 0:
        for _, row in problem_df.iterrows():
            print(f"\n{row['class'].upper()}:")
            print(f"  Semantic AUROC: {row['semantic_auroc']:.2f}")
            print(f"  E_geom: Normal={row['E_geom_normal_mean']:.4f}, Abnormal={row['E_geom_abnormal_mean']:.4f}, Diff={row['E_geom_diff']:+.4f}")
            print(f"  E_sem:  Normal={row['E_sem_normal_mean']:.4f}, Abnormal={row['E_sem_abnormal_mean']:.4f}, Diff={row['E_sem_diff']:+.4f}")
            
            # Diagnosis
            if abs(row['E_geom_diff']) < 0.01 and abs(row['E_sem_diff']) < 0.01:
                print(f"  🔴 Critical: Both E_geom and E_sem have no discriminability!")
            elif abs(row['E_geom_diff']) < 0.01:
                print(f"  🔴 E_geom has no discriminability (geometric features fail)")
            elif abs(row['E_sem_diff']) < 0.01:
                print(f"  🔴 E_sem has no discriminability (MAP prompts ineffective)")
            elif row['E_sem_diff'] > 0:
                print(f"  🔴 E_sem wrong direction (abnormal MORE aligned with normal!)")
    else:
        print("  ✅ No problem classes (all AUROC >= 70)")
    
    # Scale mismatch issues
    print("\n⚖️  Scale Analysis")
    print("-" * 80)
    
    scale_issues = df[(df['scale_ratio'] > 2.0) | (df['scale_ratio'] < 0.5)]
    
    if len(scale_issues) > 0:
        print(f"Classes with scale mismatch: {len(scale_issues)}")
        for _, row in scale_issues.iterrows():
            if row['scale_ratio'] > 2.0:
                print(f"  {row['class']}: E_sem is {row['scale_ratio']:.2f}x larger (may dominate fusion)")
            else:
                print(f"  {row['class']}: E_sem is {1/row['scale_ratio']:.2f}x smaller (may be ignored)")
    else:
        print("  ✅ All classes have balanced scales (0.5 < ratio < 2.0)")
    
    # Top performing classes
    print("\n🏆 Top 5 Classes (by Semantic AUROC)")
    print("-" * 80)
    
    top5 = df.nlargest(5, 'semantic_auroc')
    for i, (_, row) in enumerate(top5.iterrows(), 1):
        print(f"{i}. {row['class']:<15} AUROC={row['semantic_auroc']:.2f}, "
              f"E_geom_diff={row['E_geom_diff']:+.4f}, E_sem_diff={row['E_sem_diff']:+.4f}")
    
    # Bottom performing classes
    print("\n⬇️  Bottom 5 Classes (by Semantic AUROC)")
    print("-" * 80)
    
    bottom5 = df.nsmallest(5, 'semantic_auroc')
    for i, (_, row) in enumerate(bottom5.iterrows(), 1):
        print(f"{i}. {row['class']:<15} AUROC={row['semantic_auroc']:.2f}, "
              f"E_geom_diff={row['E_geom_diff']:+.4f}, E_sem_diff={row['E_sem_diff']:+.4f}")
    
    print("\n" + "="*80)
    print("Report Complete")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Generate scale analysis report')
    parser.add_argument('--csv', type=str, 
                       default='./result/scale_analysis/scale_stats.csv',
                       help='Path to scale_stats.csv')
    
    args = parser.parse_args()
    
    generate_report(args.csv)


if __name__ == '__main__':
    main()
