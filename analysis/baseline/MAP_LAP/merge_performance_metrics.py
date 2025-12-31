#!/usr/bin/env python3
"""
Merge reliability metrics with baseline performance metrics.

Input:
- reliability_metrics_k2.csv (from compute_reliability_metrics.py)
- Seed_111-results.csv (from baseline training)

Output:
- full_metrics_k2.csv (combined data for failure mode analysis)
"""

import pandas as pd
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Merge reliability and performance metrics')
    parser.add_argument('--reliability-csv', type=str, required=True,
                        help='Path to reliability metrics CSV')
    parser.add_argument('--baseline-csvs', type=str, nargs='+', required=True,
                        help='Paths to baseline results CSV files (e.g., mvtec and visa)')
    parser.add_argument('--output-csv', type=str, required=True,
                        help='Output path for merged CSV')
    
    args = parser.parse_args()
    
    # Load reliability metrics
    reliability_df = pd.read_csv(args.reliability_csv)
    print(f"Loaded reliability metrics: {len(reliability_df)} classes")
    
    # Load and concatenate baseline results
    baseline_dfs = []
    for csv_path in args.baseline_csvs:
        # First column is the class name (index)
        df = pd.read_csv(csv_path, index_col=0)
        df['class'] = df.index
        baseline_dfs.append(df)
    baseline_df = pd.concat(baseline_dfs, ignore_index=True)
    print(f"Loaded baseline results: {len(baseline_df)} classes")
    
    # Prepare merge keys
    # reliability_df has: dataset, class, k_shot, seed
    # baseline_df has: class (from index), i_roc, p_roc, semantic_i_roc, memory_i_roc
    
    # Merge on class name
    merged_df = reliability_df.merge(
        baseline_df[['class', 'i_roc', 'p_roc', 'semantic_i_roc', 'memory_i_roc']],
        on='class',
        how='left'
    )
    
    # Rename for clarity
    merged_df = merged_df.rename(columns={
        'i_roc': 'fusion_auroc',
        'p_roc': 'fusion_p_auroc',
        'semantic_i_roc': 'semantic_auroc',
        'memory_i_roc': 'memory_auroc'
    })
    
    # Add derived metrics
    merged_df['delta_fusion'] = merged_df['fusion_auroc'] - merged_df['semantic_auroc']
    merged_df['delta_memory'] = merged_df['memory_auroc'] - merged_df['semantic_auroc']
    
    # Calculate margin separation (simplified from full distribution analysis)
    merged_df['margin_separation'] = abs(merged_df['margin_MAP_mean'])
    
    # Add anchor cosine similarity (using MAP-LAP alignment)
    merged_df['anchor_cosine_sim'] = merged_df['cos_MAP_LAP']
    
    # Save
    merged_df.to_csv(args.output_csv, index=False)
    print(f"\n✅ Merged metrics saved to: {args.output_csv}")
    print(f"Total classes: {len(merged_df)}")
    print(f"\nColumns: {list(merged_df.columns)}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"\nSemantic AUROC: {merged_df['semantic_auroc'].mean():.2f}±{merged_df['semantic_auroc'].std():.2f}")
    print(f"Fusion AUROC:   {merged_df['fusion_auroc'].mean():.2f}±{merged_df['fusion_auroc'].std():.2f}")
    print(f"Delta (Fusion): {merged_df['delta_fusion'].mean():.2f}±{merged_df['delta_fusion'].std():.2f}")
    print(f"\nMAP Risk (R_0): {merged_df['R_MAP_0'].mean():.3f}±{merged_df['R_MAP_0'].std():.3f}")
    print(f"LAP Risk (R_0): {merged_df['R_LAP_0'].mean():.3f}±{merged_df['R_LAP_0'].std():.3f}")


if __name__ == '__main__':
    main()
