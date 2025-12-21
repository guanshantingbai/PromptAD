#!/usr/bin/env python3
"""
Consolidate multi-prototype PromptAD results from result/prompt1/
Creates a comprehensive summary CSV with all k-shot results.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse


def load_results(base_dir, dataset, k_shot):
    """Load results from CSV file."""
    csv_path = Path(base_dir) / dataset / f"k_{k_shot}" / "csv" / "Seed_111-results.csv"
    
    if not csv_path.exists():
        print(f"Warning: {csv_path} not found")
        return None
    
    df = pd.read_csv(csv_path, index_col=0)
    return df


def consolidate_results(base_dir="result/prompt1", dataset="mvtec", output_file=None):
    """Consolidate results across different k-shot settings."""
    
    # Available k-shot configurations
    k_shots = [1, 2, 4]
    
    # Load all results
    results = {}
    for k in k_shots:
        df = load_results(base_dir, dataset, k)
        if df is not None:
            results[k] = df
    
    if not results:
        print("No results found!")
        return None
    
    # Get all classes
    classes = sorted(results[list(results.keys())[0]].index.tolist())
    
    # Create consolidated dataframe
    consolidated = []
    
    for cls_name in classes:
        row = {'class': cls_name.replace(f'{dataset}-', '')}
        
        for k in k_shots:
            if k in results and cls_name in results[k].index:
                row[f'i_roc_k{k}'] = results[k].loc[cls_name, 'i_roc']
                if 'p_roc' in results[k].columns:
                    row[f'p_roc_k{k}'] = results[k].loc[cls_name, 'p_roc']
            else:
                row[f'i_roc_k{k}'] = np.nan
                row[f'p_roc_k{k}'] = np.nan
        
        consolidated.append(row)
    
    df_consolidated = pd.DataFrame(consolidated)
    
    # Calculate averages
    avg_row = {'class': 'AVERAGE'}
    for k in k_shots:
        i_roc_col = f'i_roc_k{k}'
        p_roc_col = f'p_roc_k{k}'
        
        if i_roc_col in df_consolidated.columns:
            avg_row[i_roc_col] = df_consolidated[i_roc_col].mean()
        if p_roc_col in df_consolidated.columns:
            avg_row[p_roc_col] = df_consolidated[p_roc_col].mean()
    
    df_consolidated = pd.concat([df_consolidated, pd.DataFrame([avg_row])], ignore_index=True)
    
    # Save consolidated results
    if output_file is None:
        output_file = Path(base_dir) / f"{dataset}_consolidated_results.csv"
    
    df_consolidated.to_csv(output_file, index=False, float_format='%.2f')
    print(f"\n{'='*80}")
    print(f"Consolidated results saved to: {output_file}")
    print(f"{'='*80}\n")
    
    return df_consolidated


def print_summary(df):
    """Print a nice summary table."""
    print("\n" + "="*80)
    print("MULTI-PROTOTYPE PromptAD RESULTS SUMMARY")
    print("="*80)
    print()
    
    # Print table
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(df.to_string(index=False))
    print()
    
    # Print key statistics
    avg_row = df[df['class'] == 'AVERAGE'].iloc[0]
    print("="*80)
    print("KEY STATISTICS (Image-level AUROC)")
    print("="*80)
    
    for col in df.columns:
        if col.startswith('i_roc_k'):
            k_shot = col.split('_k')[1]
            value = avg_row[col]
            print(f"  k={k_shot} shot: {value:.2f}%")
    
    print("="*80)
    print()


def create_comparison_table(base_dir="result/prompt1", baseline_dir="result/baseline", dataset="mvtec"):
    """Create comparison table between multi-prototype and baseline."""
    
    # Load multi-prototype results
    mp_k2 = load_results(base_dir, dataset, 2)
    
    # Load baseline results if available
    baseline_k2 = load_results(baseline_dir, dataset, 2)
    
    if mp_k2 is None:
        print("Multi-prototype k=2 results not found")
        return None
    
    # Create comparison
    comparison = pd.DataFrame()
    comparison['class'] = mp_k2.index
    comparison['class'] = comparison['class'].str.replace(f'{dataset}-', '')
    comparison['multi_prototype'] = mp_k2['i_roc'].values
    
    if baseline_k2 is not None:
        comparison['baseline'] = baseline_k2['i_roc'].values
        comparison['improvement'] = comparison['multi_prototype'] - comparison['baseline']
    
    # Add average
    avg_row = {'class': 'AVERAGE'}
    avg_row['multi_prototype'] = comparison['multi_prototype'].mean()
    if 'baseline' in comparison.columns:
        avg_row['baseline'] = comparison['baseline'].mean()
        avg_row['improvement'] = comparison['improvement'].mean()
    
    comparison = pd.concat([comparison, pd.DataFrame([avg_row])], ignore_index=True)
    
    # Save
    output_file = Path(base_dir) / f"{dataset}_comparison_k2.csv"
    comparison.to_csv(output_file, index=False, float_format='%.2f')
    
    print("\n" + "="*80)
    print("MULTI-PROTOTYPE vs BASELINE COMPARISON (k=2)")
    print("="*80)
    print()
    print(comparison.to_string(index=False))
    print()
    print(f"Saved to: {output_file}")
    print("="*80)
    print()
    
    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Consolidate multi-prototype results')
    parser.add_argument('--base-dir', type=str, default='result/prompt1',
                        help='Base directory containing results')
    parser.add_argument('--dataset', type=str, default='mvtec',
                        help='Dataset name')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV file path')
    parser.add_argument('--compare-baseline', action='store_true',
                        help='Compare with baseline results')
    
    args = parser.parse_args()
    
    # Consolidate results
    df = consolidate_results(args.base_dir, args.dataset, args.output)
    
    if df is not None:
        print_summary(df)
    
    # Compare with baseline if requested
    if args.compare_baseline:
        create_comparison_table(args.base_dir, "result/baseline", args.dataset)
