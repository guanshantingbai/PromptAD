#!/usr/bin/env python3
"""
Analyze Lambda Sweep Results
Aggregates results from all lambda sweep experiments and generates summary tables
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import sys

def collect_results(output_root="result/lambda_sweep_full", k_shot=2, seed=111):
    """Collect all results from lambda sweep experiments"""
    
    results = []
    
    # Lambda values tested
    lambdas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    datasets = ['mvtec', 'visa']
    
    for dataset in datasets:
        for lambda_val in lambdas:
            csv_path = f"{output_root}/{dataset}_k{k_shot}_lambda{lambda_val}/{dataset}/k_{k_shot}/csv/Seed_{seed}-results.csv"
            
            if not os.path.exists(csv_path):
                print(f"Warning: Missing CSV for {dataset} λ={lambda_val}: {csv_path}")
                continue
            
            try:
                df = pd.read_csv(csv_path, index_col=0)
                
                # Process each class
                for idx, row in df.iterrows():
                    # Extract class name (remove dataset prefix)
                    class_name = idx.replace(f"{dataset}-", "")
                    
                    # Skip if all zeros (not tested)
                    if row['i_roc'] == 0.0:
                        continue
                    
                    result = {
                        'dataset': dataset,
                        'class': class_name,
                        'lambda': lambda_val,
                        'i_roc': row.get('i_roc', np.nan),
                        'semantic_i_roc': row.get('semantic_i_roc', np.nan),
                        'memory_i_roc': row.get('memory_i_roc', np.nan),
                        'fusion_i_roc': row.get('fusion_i_roc', np.nan),
                        'sim_before': row.get('sim_before', np.nan),
                        'sim_after': row.get('sim_after', np.nan),
                        'sim_to_learned': row.get('sim_to_learned', np.nan),
                        'sim_individual_mean': row.get('sim_individual_mean', np.nan),
                        'sim_individual_std': row.get('sim_individual_std', np.nan),
                    }
                    
                    # Calculate delta similarity
                    if pd.notna(result['sim_before']) and pd.notna(result['sim_after']):
                        result['delta_sim'] = result['sim_after'] - result['sim_before']
                    else:
                        result['delta_sim'] = np.nan
                    
                    results.append(result)
                    
            except Exception as e:
                print(f"Error processing {csv_path}: {e}")
    
    return pd.DataFrame(results)

def generate_summary_tables(df, output_dir="result/lambda_sweep_full"):
    """Generate various summary tables"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("Lambda Sweep Analysis Results")
    print("="*80 + "\n")
    
    # 1. Overall statistics by lambda
    print("1. Overall Statistics by Lambda")
    print("-" * 80)
    lambda_stats = df.groupby('lambda').agg({
        'semantic_i_roc': ['mean', 'std', 'min', 'max'],
        'sim_after': ['mean', 'std'],
        'delta_sim': ['mean', 'std']
    }).round(4)
    print(lambda_stats)
    lambda_stats.to_csv(f"{output_dir}/stats_by_lambda.csv")
    print(f"Saved to: {output_dir}/stats_by_lambda.csv\n")
    
    # 2. Best lambda per class
    print("2. Best Lambda per Class (by Semantic AUROC)")
    print("-" * 80)
    best_lambda = df.loc[df.groupby(['dataset', 'class'])['semantic_i_roc'].idxmax()]
    best_lambda_summary = best_lambda[['dataset', 'class', 'lambda', 'semantic_i_roc', 'sim_before', 'sim_after', 'delta_sim']].sort_values(['dataset', 'class'])
    print(best_lambda_summary.to_string(index=False))
    best_lambda_summary.to_csv(f"{output_dir}/best_lambda_per_class.csv", index=False)
    print(f"\nSaved to: {output_dir}/best_lambda_per_class.csv\n")
    
    # 3. Performance improvement summary
    print("3. Performance Improvement (λ_best vs λ=0)")
    print("-" * 80)
    
    # Get baseline (λ=0) performance
    baseline = df[df['lambda'] == 0.0].set_index(['dataset', 'class'])['semantic_i_roc']
    
    # Get best performance for each class
    best_perf = df.loc[df.groupby(['dataset', 'class'])['semantic_i_roc'].idxmax()]
    best_perf = best_perf.set_index(['dataset', 'class'])
    
    improvement = pd.DataFrame({
        'baseline_auroc': baseline,
        'best_auroc': best_perf['semantic_i_roc'],
        'best_lambda': best_perf['lambda'],
        'improvement': best_perf['semantic_i_roc'] - baseline,
        'baseline_sim': best_perf['sim_before'],
        'best_sim': best_perf['sim_after'],
        'sim_gain': best_perf['delta_sim']
    }).round(4)
    
    improvement = improvement.sort_values('improvement', ascending=False)
    print(improvement)
    improvement.to_csv(f"{output_dir}/performance_improvement.csv")
    print(f"\nSaved to: {output_dir}/performance_improvement.csv\n")
    
    # 4. Classes with significant improvement (>1% AUROC)
    significant = improvement[improvement['improvement'] > 1.0]
    if len(significant) > 0:
        print("4. Classes with Significant Improvement (>1% AUROC)")
        print("-" * 80)
        print(significant)
        print(f"\nTotal: {len(significant)} classes\n")
    else:
        print("4. No classes with >1% AUROC improvement\n")
    
    # 5. Correlation analysis
    print("5. Correlation: Similarity vs Performance")
    print("-" * 80)
    
    # Filter valid data
    valid_data = df[pd.notna(df['sim_after']) & pd.notna(df['semantic_i_roc'])]
    
    if len(valid_data) > 0:
        corr_sim_perf = valid_data['sim_after'].corr(valid_data['semantic_i_roc'])
        corr_delta_sim_perf = valid_data['delta_sim'].corr(valid_data['semantic_i_roc'])
        
        print(f"Correlation(sim_after, semantic_auroc) = {corr_sim_perf:.4f}")
        print(f"Correlation(delta_sim, semantic_auroc) = {corr_delta_sim_perf:.4f}")
        print()
    
    # 6. Dataset-wise summary
    print("6. Dataset-wise Summary")
    print("-" * 80)
    dataset_summary = improvement.groupby(level='dataset').agg({
        'baseline_auroc': 'mean',
        'best_auroc': 'mean',
        'improvement': ['mean', 'max'],
        'sim_gain': ['mean', 'max']
    }).round(4)
    print(dataset_summary)
    print()
    
    # Save full results
    df_sorted = df.sort_values(['dataset', 'class', 'lambda'])
    df_sorted.to_csv(f"{output_dir}/full_results.csv", index=False)
    print(f"Full results saved to: {output_dir}/full_results.csv")
    
    return {
        'lambda_stats': lambda_stats,
        'best_lambda': best_lambda_summary,
        'improvement': improvement,
        'dataset_summary': dataset_summary
    }

def main():
    output_root = "result/lambda_sweep_full"
    
    if not os.path.exists(output_root):
        print(f"Error: Output directory not found: {output_root}")
        print("Please run the lambda sweep experiment first.")
        sys.exit(1)
    
    print("Collecting results from lambda sweep experiments...")
    df = collect_results(output_root)
    
    if len(df) == 0:
        print("Error: No results found. Please check that experiments have completed.")
        sys.exit(1)
    
    print(f"\nCollected {len(df)} result entries")
    print(f"Datasets: {df['dataset'].unique()}")
    print(f"Classes: {df['class'].nunique()}")
    print(f"Lambda values: {sorted(df['lambda'].unique())}")
    
    # Generate summary tables
    summaries = generate_summary_tables(df, output_root)
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)

if __name__ == "__main__":
    main()
