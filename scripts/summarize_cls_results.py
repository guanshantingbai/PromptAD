#!/usr/bin/env python3
"""
Summarize CLS (Classification) results for different k-shot settings
"""

import pandas as pd
import os
from pathlib import Path

def summarize_results(base_dir="result/repulsive_ablation/lambda_0.05"):
    """
    Summarize classification results for MVTec and ViSA datasets
    
    Args:
        base_dir: Base directory containing results
    """
    
    results = {
        'MVTec': {},
        'ViSA': {}
    }
    
    datasets = {
        'MVTec': 'mvtec',
        'ViSA': 'visa'
    }
    
    k_shots = [1, 2, 4]
    
    print("=" * 80)
    print("CLS Task Results Summary (Classification)")
    print("=" * 80)
    print(f"\nResults directory: {base_dir}")
    print(f"Seed: 111")
    print(f"Configuration: lambda_rep=0.05, margin=0.8, fusion_lambda=0.3, fusion_weight=0.1")
    print()
    
    # Collect results
    for dataset_name, dataset_dir in datasets.items():
        for k in k_shots:
            csv_path = Path(base_dir) / dataset_dir / f"k_{k}" / "csv" / "Seed_111-results.csv"
            
            if not csv_path.exists():
                print(f"⚠️  Warning: {csv_path} not found")
                results[dataset_name][k] = None
                continue
            
            # Read CSV
            df = pd.read_csv(csv_path, index_col=0)
            
            # Extract metrics (assuming columns: i_roc, semantic_i_roc, memory_i_roc, p_roc)
            metrics = {
                'I-AUROC': df['i_roc'].mean(),
                'Semantic': df['semantic_i_roc'].mean(),
                'Memory': df['memory_i_roc'].mean(),
                'P-AUROC': df['p_roc'].mean() if 'p_roc' in df.columns else 0.0,
                'Count': len(df)
            }
            
            results[dataset_name][k] = metrics
    
    # Print results table
    print("\n" + "=" * 80)
    print("MVTec Dataset Results")
    print("=" * 80)
    print(f"{'K-Shot':<10} {'I-AUROC':<12} {'Semantic':<12} {'Memory':<12} {'Classes':<10}")
    print("-" * 80)
    
    for k in k_shots:
        metrics = results['MVTec'].get(k)
        if metrics:
            print(f"{k:<10} {metrics['I-AUROC']:>10.2f}%  {metrics['Semantic']:>10.2f}%  "
                  f"{metrics['Memory']:>10.2f}%  {metrics['Count']:<10}")
        else:
            print(f"{k:<10} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<10}")
    
    print("\n" + "=" * 80)
    print("ViSA Dataset Results")
    print("=" * 80)
    print(f"{'K-Shot':<10} {'I-AUROC':<12} {'Semantic':<12} {'Memory':<12} {'Classes':<10}")
    print("-" * 80)
    
    for k in k_shots:
        metrics = results['ViSA'].get(k)
        if metrics:
            print(f"{k:<10} {metrics['I-AUROC']:>10.2f}%  {metrics['Semantic']:>10.2f}%  "
                  f"{metrics['Memory']:>10.2f}%  {metrics['Count']:<10}")
        else:
            print(f"{k:<10} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<10}")
    
    # Print combined average
    print("\n" + "=" * 80)
    print("Combined Average (MVTec + ViSA)")
    print("=" * 80)
    print(f"{'K-Shot':<10} {'I-AUROC':<12} {'Semantic':<12} {'Memory':<12} {'Total Classes':<15}")
    print("-" * 80)
    
    for k in k_shots:
        mvtec_metrics = results['MVTec'].get(k)
        visa_metrics = results['ViSA'].get(k)
        
        if mvtec_metrics and visa_metrics:
            # Weighted average by number of classes
            total_count = mvtec_metrics['Count'] + visa_metrics['Count']
            
            combined_i_roc = (mvtec_metrics['I-AUROC'] * mvtec_metrics['Count'] + 
                             visa_metrics['I-AUROC'] * visa_metrics['Count']) / total_count
            combined_sem = (mvtec_metrics['Semantic'] * mvtec_metrics['Count'] + 
                           visa_metrics['Semantic'] * visa_metrics['Count']) / total_count
            combined_mem = (mvtec_metrics['Memory'] * mvtec_metrics['Count'] + 
                           visa_metrics['Memory'] * visa_metrics['Count']) / total_count
            
            print(f"{k:<10} {combined_i_roc:>10.2f}%  {combined_sem:>10.2f}%  "
                  f"{combined_mem:>10.2f}%  {total_count:<15}")
        else:
            print(f"{k:<10} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<15}")
    
    print("\n" + "=" * 80)
    print("Column Descriptions:")
    print("=" * 80)
    print("  I-AUROC:  Image-level AUROC (fusion of semantic + memory)")
    print("  Semantic: Semantic branch AUROC (text prototype similarity)")
    print("  Memory:   Memory branch AUROC (K-NN distance)")
    print("=" * 80)
    
    # Save to CSV
    output_file = Path(base_dir) / "cls_summary.csv"
    summary_data = []
    
    for dataset_name in ['MVTec', 'ViSA']:
        for k in k_shots:
            metrics = results[dataset_name].get(k)
            if metrics:
                summary_data.append({
                    'Dataset': dataset_name,
                    'K-Shot': k,
                    'I-AUROC': f"{metrics['I-AUROC']:.2f}",
                    'Semantic-AUROC': f"{metrics['Semantic']:.2f}",
                    'Memory-AUROC': f"{metrics['Memory']:.2f}",
                    'Classes': metrics['Count']
                })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_file, index=False)
    print(f"\n✓ Summary saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    summarize_results()
