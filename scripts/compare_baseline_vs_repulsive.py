#!/usr/bin/env python3
"""
Compare Baseline vs Repulsive Ablation (lambda=0.05) Results
Focus on I-AUROC improvements
"""

import pandas as pd
from pathlib import Path

def compare_results():
    """Compare baseline and repulsive ablation results"""
    
    baseline_dir = Path("result/baseline")
    repulsive_dir = Path("result/repulsive_ablation/lambda_0.05")
    
    datasets = {
        'MVTec': 'mvtec',
        'ViSA': 'visa'
    }
    
    k_shots = [1, 2, 4]
    
    results = {
        'Baseline': {'MVTec': {}, 'ViSA': {}},
        'Repulsive': {'MVTec': {}, 'ViSA': {}}
    }
    
    # Collect results
    for dataset_name, dataset_dir in datasets.items():
        for k in k_shots:
            # Baseline
            baseline_csv = baseline_dir / dataset_dir / f"k_{k}" / "csv" / "Seed_111-results.csv"
            if baseline_csv.exists():
                df = pd.read_csv(baseline_csv, index_col=0)
                results['Baseline'][dataset_name][k] = {
                    'I-AUROC': df['i_roc'].mean(),
                    'Semantic': df['semantic_i_roc'].mean(),
                    'Memory': df['memory_i_roc'].mean(),
                    'Count': len(df)
                }
            
            # Repulsive
            repulsive_csv = repulsive_dir / dataset_dir / f"k_{k}" / "csv" / "Seed_111-results.csv"
            if repulsive_csv.exists():
                df = pd.read_csv(repulsive_csv, index_col=0)
                results['Repulsive'][dataset_name][k] = {
                    'I-AUROC': df['i_roc'].mean(),
                    'Semantic': df['semantic_i_roc'].mean(),
                    'Memory': df['memory_i_roc'].mean(),
                    'Count': len(df)
                }
    
    # Print comparison
    print("=" * 100)
    print("BASELINE vs REPULSIVE ABLATION (λ_rep=0.05) - CLS Task Comparison")
    print("=" * 100)
    print("\nConfiguration:")
    print("  Baseline:   Standard training (no repulsive loss)")
    print("  Repulsive:  λ_rep=0.05, margin=0.8, fusion_lambda=0.3, fusion_weight=0.1")
    print()
    
    # MVTec comparison
    print("=" * 100)
    print("MVTec Dataset (15 classes)")
    print("=" * 100)
    print(f"{'K-Shot':<8} {'Method':<12} {'I-AUROC':<12} {'Semantic':<12} {'Memory':<12} {'Δ I-AUROC':<12}")
    print("-" * 100)
    
    for k in k_shots:
        baseline = results['Baseline']['MVTec'].get(k)
        repulsive = results['Repulsive']['MVTec'].get(k)
        
        if baseline and repulsive:
            delta = repulsive['I-AUROC'] - baseline['I-AUROC']
            
            print(f"{k:<8} {'Baseline':<12} {baseline['I-AUROC']:>10.2f}%  {baseline['Semantic']:>10.2f}%  "
                  f"{baseline['Memory']:>10.2f}%  {'-':<12}")
            print(f"{'':<8} {'Repulsive':<12} {repulsive['I-AUROC']:>10.2f}%  {repulsive['Semantic']:>10.2f}%  "
                  f"{repulsive['Memory']:>10.2f}%  {delta:>+10.2f}%")
            print()
    
    # ViSA comparison
    print("=" * 100)
    print("ViSA Dataset (12 classes)")
    print("=" * 100)
    print(f"{'K-Shot':<8} {'Method':<12} {'I-AUROC':<12} {'Semantic':<12} {'Memory':<12} {'Δ I-AUROC':<12}")
    print("-" * 100)
    
    for k in k_shots:
        baseline = results['Baseline']['ViSA'].get(k)
        repulsive = results['Repulsive']['ViSA'].get(k)
        
        if baseline and repulsive:
            delta = repulsive['I-AUROC'] - baseline['I-AUROC']
            
            print(f"{k:<8} {'Baseline':<12} {baseline['I-AUROC']:>10.2f}%  {baseline['Semantic']:>10.2f}%  "
                  f"{baseline['Memory']:>10.2f}%  {'-':<12}")
            print(f"{'':<8} {'Repulsive':<12} {repulsive['I-AUROC']:>10.2f}%  {repulsive['Semantic']:>10.2f}%  "
                  f"{repulsive['Memory']:>10.2f}%  {delta:>+10.2f}%")
            print()
    
    # Combined average comparison
    print("=" * 100)
    print("Combined Average (MVTec + ViSA, 27 classes)")
    print("=" * 100)
    print(f"{'K-Shot':<8} {'Method':<12} {'I-AUROC':<12} {'Semantic':<12} {'Memory':<12} {'Δ I-AUROC':<12}")
    print("-" * 100)
    
    for k in k_shots:
        # Baseline combined
        mvtec_base = results['Baseline']['MVTec'].get(k)
        visa_base = results['Baseline']['ViSA'].get(k)
        
        # Repulsive combined
        mvtec_rep = results['Repulsive']['MVTec'].get(k)
        visa_rep = results['Repulsive']['ViSA'].get(k)
        
        if mvtec_base and visa_base and mvtec_rep and visa_rep:
            total_count = mvtec_base['Count'] + visa_base['Count']
            
            # Baseline weighted average
            base_i_roc = (mvtec_base['I-AUROC'] * mvtec_base['Count'] + 
                         visa_base['I-AUROC'] * visa_base['Count']) / total_count
            base_sem = (mvtec_base['Semantic'] * mvtec_base['Count'] + 
                       visa_base['Semantic'] * visa_base['Count']) / total_count
            base_mem = (mvtec_base['Memory'] * mvtec_base['Count'] + 
                       visa_base['Memory'] * visa_base['Count']) / total_count
            
            # Repulsive weighted average
            rep_i_roc = (mvtec_rep['I-AUROC'] * mvtec_rep['Count'] + 
                        visa_rep['I-AUROC'] * visa_rep['Count']) / total_count
            rep_sem = (mvtec_rep['Semantic'] * mvtec_rep['Count'] + 
                      visa_rep['Semantic'] * visa_rep['Count']) / total_count
            rep_mem = (mvtec_rep['Memory'] * mvtec_rep['Count'] + 
                      visa_rep['Memory'] * visa_rep['Count']) / total_count
            
            delta = rep_i_roc - base_i_roc
            
            print(f"{k:<8} {'Baseline':<12} {base_i_roc:>10.2f}%  {base_sem:>10.2f}%  "
                  f"{base_mem:>10.2f}%  {'-':<12}")
            print(f"{'':<8} {'Repulsive':<12} {rep_i_roc:>10.2f}%  {rep_sem:>10.2f}%  "
                  f"{rep_mem:>10.2f}%  {delta:>+10.2f}%")
            print()
    
    print("=" * 100)
    print("Summary:")
    print("=" * 100)
    
    # Calculate average improvements
    improvements = []
    for k in k_shots:
        mvtec_base = results['Baseline']['MVTec'].get(k)
        visa_base = results['Baseline']['ViSA'].get(k)
        mvtec_rep = results['Repulsive']['MVTec'].get(k)
        visa_rep = results['Repulsive']['ViSA'].get(k)
        
        if all([mvtec_base, visa_base, mvtec_rep, visa_rep]):
            total_count = mvtec_base['Count'] + visa_base['Count']
            base_avg = (mvtec_base['I-AUROC'] * mvtec_base['Count'] + 
                       visa_base['I-AUROC'] * visa_base['Count']) / total_count
            rep_avg = (mvtec_rep['I-AUROC'] * mvtec_rep['Count'] + 
                      visa_rep['I-AUROC'] * visa_rep['Count']) / total_count
            improvements.append(rep_avg - base_avg)
    
    if improvements:
        avg_improvement = sum(improvements) / len(improvements)
        print(f"\n✓ Average I-AUROC improvement across all k-shots: {avg_improvement:+.2f}%")
        print(f"✓ Best improvement: {max(improvements):+.2f}% (k={k_shots[improvements.index(max(improvements))]})")
        print(f"✓ Smallest improvement: {min(improvements):+.2f}% (k={k_shots[improvements.index(min(improvements))]})")
    
    print("=" * 100)
    
    # Save comparison to CSV
    output_file = Path("result") / "baseline_vs_repulsive_comparison.csv"
    comparison_data = []
    
    for dataset_name in ['MVTec', 'ViSA']:
        for k in k_shots:
            baseline = results['Baseline'][dataset_name].get(k)
            repulsive = results['Repulsive'][dataset_name].get(k)
            
            if baseline and repulsive:
                comparison_data.append({
                    'Dataset': dataset_name,
                    'K-Shot': k,
                    'Baseline-I-AUROC': f"{baseline['I-AUROC']:.2f}",
                    'Repulsive-I-AUROC': f"{repulsive['I-AUROC']:.2f}",
                    'Improvement': f"{repulsive['I-AUROC'] - baseline['I-AUROC']:+.2f}",
                    'Baseline-Semantic': f"{baseline['Semantic']:.2f}",
                    'Repulsive-Semantic': f"{repulsive['Semantic']:.2f}",
                    'Baseline-Memory': f"{baseline['Memory']:.2f}",
                    'Repulsive-Memory': f"{repulsive['Memory']:.2f}"
                })
    
    df = pd.DataFrame(comparison_data)
    df.to_csv(output_file, index=False)
    print(f"\n✓ Detailed comparison saved to: {output_file}\n")


if __name__ == "__main__":
    compare_results()
