#!/usr/bin/env python3
"""
Simple Failure Mode Classification based on full_metrics_k2.csv

This script classifies failure modes without needing the full baseline_analysis structure.
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path


def classify_failure_mode(row):
    """
    Classify failure mode for a single class.
    
    Returns: (failure_type, failure_reason)
    """
    semantic = row['semantic_auroc']
    fusion = row['fusion_auroc']
    delta = row['delta_fusion']
    
    # Use margin separation from MAP
    sep = abs(row['margin_MAP_mean'])
    
    # Use MAP-LAP cosine similarity as proxy for anchor collapse
    cos_sim = row['cos_MAP_LAP']
    
    # MAP/LAP reliability
    r_map = row['R_MAP_0']
    r_lap = row['R_LAP_0']
    
    # Define thresholds
    MARGIN_COLLAPSE_THRESHOLD = 0.05
    ANCHOR_CLOSE_THRESHOLD = 0.92
    SEMANTIC_WEAK_THRESHOLD = 85.0
    DELTA_LARGE_THRESHOLD = 8.0
    SEMANTIC_STRONG_THRESHOLD = 95.0
    
    failure_types = []
    reasons = []
    
    # Mode A: Margin Collapse
    if sep < MARGIN_COLLAPSE_THRESHOLD:
        failure_types.append('A')
        reasons.append(f'margin_collapse(sep={sep:.3f})')
    
    # Mode B: Anchor Direction Collapse
    if cos_sim > ANCHOR_CLOSE_THRESHOLD:
        failure_types.append('B')
        reasons.append(f'anchor_collapse(cos={cos_sim:.3f})')
    
    # Mode C: Semantic Weak但Fusion拯救
    if semantic < SEMANTIC_WEAK_THRESHOLD and delta > DELTA_LARGE_THRESHOLD:
        failure_types.append('C')
        reasons.append(f'semantic_weak_fusion_saves(Δ={delta:.1f})')
    
    # Mode D: Semantic Strong且Memory无帮助
    if semantic > SEMANTIC_STRONG_THRESHOLD and abs(delta) < 2.0:
        failure_types.append('D')
        reasons.append(f'semantic_strong_no_memory_help(sem={semantic:.1f})')
    
    # Default: Normal performance
    if not failure_types:
        failure_types.append('Normal')
        reasons.append('good_performance')
    
    failure_type = '+'.join(failure_types)
    failure_reason = '; '.join(reasons)
    
    return failure_type, failure_reason


def main():
    parser = argparse.ArgumentParser(description='Simple failure mode classification')
    parser.add_argument('--input-csv', type=str, required=True,
                        help='Path to full_metrics_k2.csv')
    parser.add_argument('--output-csv', type=str, required=True,
                        help='Output path for failure mode table')
    
    args = parser.parse_args()
    
    # Load data
    df = pd.read_csv(args.input_csv)
    print(f"Loaded {len(df)} classes from {args.input_csv}")
    
    # Classify each class
    results = []
    for idx, row in df.iterrows():
        failure_type, failure_reason = classify_failure_mode(row)
        
        results.append({
            'class': row['class'],
            'semantic': row['semantic_auroc'],
            'fusion': row['fusion_auroc'],
            'delta': row['delta_fusion'],
            'sep': abs(row['margin_MAP_mean']),
            'anchor_cos': row['cos_MAP_LAP'],
            'R_MAP_0': row['R_MAP_0'],
            'R_LAP_0': row['R_LAP_0'],
            'failure_type': failure_type,
            'failure_reason': failure_reason
        })
    
    result_df = pd.DataFrame(results)
    
    # Save
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)
    
    print(f"\n✅ Failure mode table saved to: {output_path}")
    print(f"\nFailure Mode Distribution:")
    print(result_df['failure_type'].value_counts())
    
    # Print summary by type
    print("\n" + "="*80)
    print("FAILURE MODE SUMMARY")
    print("="*80)
    
    for mode in sorted(result_df['failure_type'].unique()):
        mode_df = result_df[result_df['failure_type'] == mode]
        print(f"\n【{mode}】 ({len(mode_df)} classes)")
        print(f"  Semantic: {mode_df['semantic'].mean():.2f}±{mode_df['semantic'].std():.2f}")
        print(f"  Fusion:   {mode_df['fusion'].mean():.2f}±{mode_df['fusion'].std():.2f}")
        print(f"  Delta:    {mode_df['delta'].mean():.2f}±{mode_df['delta'].std():.2f}")
        print(f"  Classes: {', '.join(mode_df['class'].tolist())}")


if __name__ == '__main__':
    main()
