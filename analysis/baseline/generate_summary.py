"""
Generate summary report from individual baseline analysis results.

This script collects all individual class analysis results and generates:
1. Comprehensive summary CSV with all classes
2. Statistical summary (mean, std, median)
3. Key insights printed to console
"""

import os
import pandas as pd
import numpy as np
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Generate summary report from baseline analysis')
    parser.add_argument('--analysis-dir', type=str, required=True,
                        help='Directory containing individual analysis results')
    parser.add_argument('--output', type=str, required=True,
                        help='Output summary CSV file')
    args = parser.parse_args()
    
    analysis_dir = Path(args.analysis_dir)
    
    if not analysis_dir.exists():
        print(f"❌ Analysis directory not found: {analysis_dir}")
        return 1
    
    # Collect all baseline metrics CSV files
    csv_files = list(analysis_dir.glob('*_baseline_metrics.csv'))
    
    if not csv_files:
        print(f"❌ No *_baseline_metrics.csv files found in {analysis_dir}")
        print(f"Expected files like: candle_baseline_metrics.csv")
        return 1
    
    print(f"Found {len(csv_files)} analysis results")
    print("")
    
    # Read and concatenate all CSVs
    all_results = []
    for csv_file in sorted(csv_files):
        try:
            df = pd.read_csv(csv_file)
            if not df.empty:
                all_results.append(df)
                print(f"  ✓ Loaded: {csv_file.name}")
            else:
                print(f"  ⚠ Empty: {csv_file.name}")
        except Exception as e:
            print(f"  ✗ Error loading {csv_file.name}: {e}")
    
    if not all_results:
        print(f"❌ No valid results to summarize")
        return 1
    
    print("")
    summary_df = pd.concat(all_results, ignore_index=True)
    
    # Sort by class name
    summary_df = summary_df.sort_values('class_name').reset_index(drop=True)
    
    # Calculate statistical summary rows
    numeric_cols = summary_df.select_dtypes(include=[np.number]).columns.tolist()
    
    mean_row = {'class_name': 'MEAN'}
    std_row = {'class_name': 'STD'}
    median_row = {'class_name': 'MEDIAN'}
    min_row = {'class_name': 'MIN'}
    max_row = {'class_name': 'MAX'}
    
    for col in numeric_cols:
        mean_row[col] = summary_df[col].mean()
        std_row[col] = summary_df[col].std()
        median_row[col] = summary_df[col].median()
        min_row[col] = summary_df[col].min()
        max_row[col] = summary_df[col].max()
    
    summary_rows = pd.DataFrame([mean_row, std_row, median_row, min_row, max_row])
    summary_df = pd.concat([summary_df, summary_rows], ignore_index=True)
    
    # Save summary
    summary_df.to_csv(args.output, index=False, float_format='%.4f')
    print(f"✅ Summary saved to: {args.output}")
    print("")
    
    # Print key statistics to console
    print("=" * 90)
    print("BASELINE ANALYSIS SUMMARY")
    print("=" * 90)
    print(f"Total classes analyzed: {len(all_results)}")
    print("")
    
    # Define key metrics to display
    key_metrics = [
        ('semantic_auroc', 'Semantic AUROC'),
        ('fusion_auroc', 'Fusion AUROC'),
        ('delta_fusion', 'Δ Fusion (vs Semantic)'),
        ('margin_mean_normal', 'Margin Mean (Normal)'),
        ('margin_mean_abnormal', 'Margin Mean (Abnormal)'),
        ('margin_separation', 'Margin Separation'),
        ('overlap_ratio', 'Overlap Ratio'),
        ('normal_side_risk', 'Normal-side Risk@0'),
        ('abnormal_side_risk', 'Abnormal-side Risk@0'),
        ('anchor_cosine_sim', 'Anchor Cosine Sim'),
        ('anchor_l2_dist', 'Anchor L2 Distance'),
        ('anchor_angular_dist', 'Anchor Angular Dist'),
    ]
    
    # Add MAP/LAP metrics if available
    if 'MAP_auroc' in numeric_cols:
        key_metrics.extend([
            ('MAP_auroc', 'MAP-only AUROC'),
            ('LAP_auroc', 'LAP-only AUROC'),
            ('Combined_auroc', 'Combined AUROC'),
        ])
    
    print(f"{'Metric':<35} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print("-" * 90)
    
    for metric_key, metric_name in key_metrics:
        if metric_key in numeric_cols:
            mean_val = summary_df[summary_df['class_name'] == 'MEAN'][metric_key].values[0]
            std_val = summary_df[summary_df['class_name'] == 'STD'][metric_key].values[0]
            min_val = summary_df[summary_df['class_name'] == 'MIN'][metric_key].values[0]
            max_val = summary_df[summary_df['class_name'] == 'MAX'][metric_key].values[0]
            print(f"{metric_name:<35} {mean_val:<10.4f} {std_val:<10.4f} {min_val:<10.4f} {max_val:<10.4f}")
    
    print("=" * 90)
    print("")
    
    # Identify problem classes (high overlap, low separation)
    print("PROBLEM CLASSES (High Overlap / Low Separation):")
    print("-" * 90)
    
    problem_df = summary_df[summary_df['class_name'] != 'MEAN']
    problem_df = problem_df[problem_df['class_name'] != 'STD']
    problem_df = problem_df[problem_df['class_name'] != 'MEDIAN']
    problem_df = problem_df[problem_df['class_name'] != 'MIN']
    problem_df = problem_df[problem_df['class_name'] != 'MAX']
    
    # Sort by overlap ratio (descending)
    if 'overlap_ratio' in problem_df.columns:
        problem_classes = problem_df.nlargest(5, 'overlap_ratio')[['class_name', 'overlap_ratio', 'margin_separation', 'semantic_auroc', 'fusion_auroc']]
        print(problem_classes.to_string(index=False))
    
    print("")
    print("=" * 90)
    
    return 0


if __name__ == '__main__':
    exit(main())
