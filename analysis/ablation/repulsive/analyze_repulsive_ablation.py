#!/usr/bin/env python3
"""
Analyze repulsive loss ablation results
Compare different lambda_rep values
"""

import pandas as pd
from pathlib import Path
import numpy as np

# Lambda_rep values
LAMBDA_REPS = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]
base_path = Path('result/repulsive_ablation')

def load_results(lambda_rep):
    """Load results for a specific lambda_rep value"""
    mvtec_csv = base_path / f'lambda_{lambda_rep}' / 'mvtec' / 'k_2' / 'csv' / 'Seed_111-results.csv'
    visa_csv = base_path / f'lambda_{lambda_rep}' / 'visa' / 'k_2' / 'csv' / 'Seed_111-results.csv'
    
    results = {}
    
    if mvtec_csv.exists():
        df = pd.read_csv(mvtec_csv)
        df.columns = df.columns.str.strip()
        df = df.set_index(df.columns[0])
        results['mvtec'] = df
    
    if visa_csv.exists():
        df = pd.read_csv(visa_csv)
        df.columns = df.columns.str.strip()
        df = df.set_index(df.columns[0])
        results['visa'] = df
        
    return results

def calculate_averages(df):
    """Calculate average performance"""
    metrics = {}
    if 'i_roc' in df.columns:
        metrics['fusion'] = df['i_roc'].mean()
    if 'semantic_i_roc' in df.columns:
        metrics['semantic'] = df['semantic_i_roc'].mean()
    if 'memory_i_roc' in df.columns:
        metrics['memory'] = df['memory_i_roc'].mean()
    return metrics

def main():
    print("=" * 100)
    print("Repulsive Loss Ablation Study - Results Analysis")
    print("=" * 100)
    print()
    
    # Collect all results
    all_results = {}
    for lambda_rep in LAMBDA_REPS:
        results = load_results(lambda_rep)
        all_results[lambda_rep] = results
    
    # MVTec analysis
    print("=" * 100)
    print("MVTec Dataset (15 classes)")
    print("=" * 100)
    print()
    print(f"{'Lambda_rep':<12} {'Fusion':<12} {'Semantic':<12} {'Memory':<12} {'Δ Fusion':<12} {'Δ Semantic'}")
    print("-" * 80)
    
    baseline_lambda = 0.0
    baseline_mvtec = all_results[baseline_lambda].get('mvtec')
    baseline_avg_mvtec = calculate_averages(baseline_mvtec) if baseline_mvtec is not None else {}
    
    mvtec_fusion_results = []
    mvtec_semantic_results = []
    
    for lambda_rep in LAMBDA_REPS:
        results = all_results[lambda_rep].get('mvtec')
        if results is not None:
            avg = calculate_averages(results)
            fusion = avg.get('fusion', 0)
            semantic = avg.get('semantic', 0)
            memory = avg.get('memory', 0)
            
            delta_fusion = fusion - baseline_avg_mvtec.get('fusion', 0)
            delta_semantic = semantic - baseline_avg_mvtec.get('semantic', 0)
            
            mvtec_fusion_results.append((lambda_rep, fusion))
            mvtec_semantic_results.append((lambda_rep, semantic))
            
            marker = ' ★' if lambda_rep == baseline_lambda else ''
            print(f"{lambda_rep:<12.2f} {fusion:<12.2f} {semantic:<12.2f} {memory:<12.2f} "
                  f"{delta_fusion:>+11.2f} {delta_semantic:>+11.2f}{marker}")
    
    # ViSA analysis
    print()
    print("=" * 100)
    print("ViSA Dataset (12 classes)")
    print("=" * 100)
    print()
    print(f"{'Lambda_rep':<12} {'Fusion':<12} {'Semantic':<12} {'Memory':<12} {'Δ Fusion':<12} {'Δ Semantic'}")
    print("-" * 80)
    
    baseline_visa = all_results[baseline_lambda].get('visa')
    baseline_avg_visa = calculate_averages(baseline_visa) if baseline_visa is not None else {}
    
    visa_fusion_results = []
    visa_semantic_results = []
    
    for lambda_rep in LAMBDA_REPS:
        results = all_results[lambda_rep].get('visa')
        if results is not None:
            avg = calculate_averages(results)
            fusion = avg.get('fusion', 0)
            semantic = avg.get('semantic', 0)
            memory = avg.get('memory', 0)
            
            delta_fusion = fusion - baseline_avg_visa.get('fusion', 0)
            delta_semantic = semantic - baseline_avg_visa.get('semantic', 0)
            
            visa_fusion_results.append((lambda_rep, fusion))
            visa_semantic_results.append((lambda_rep, semantic))
            
            marker = ' ★' if lambda_rep == baseline_lambda else ''
            print(f"{lambda_rep:<12.2f} {fusion:<12.2f} {semantic:<12.2f} {memory:<12.2f} "
                  f"{delta_fusion:>+11.2f} {delta_semantic:>+11.2f}{marker}")
    
    # Overall analysis
    print()
    print("=" * 100)
    print("Overall Average (27 classes)")
    print("=" * 100)
    print()
    print(f"{'Lambda_rep':<12} {'Fusion':<12} {'Semantic':<12} {'Δ Fusion':<12} {'Δ Semantic':<12} {'Ranking'}")
    print("-" * 85)
    
    overall_results = []
    for lambda_rep in LAMBDA_REPS:
        mvtec_r = all_results[lambda_rep].get('mvtec')
        visa_r = all_results[lambda_rep].get('visa')
        
        if mvtec_r is not None and visa_r is not None:
            mvtec_avg = calculate_averages(mvtec_r)
            visa_avg = calculate_averages(visa_r)
            
            fusion = (mvtec_avg.get('fusion', 0) * 15 + visa_avg.get('fusion', 0) * 12) / 27
            semantic = (mvtec_avg.get('semantic', 0) * 15 + visa_avg.get('semantic', 0) * 12) / 27
            
            overall_results.append({
                'lambda_rep': lambda_rep,
                'fusion': fusion,
                'semantic': semantic
            })
    
    baseline_overall = next((r for r in overall_results if r['lambda_rep'] == baseline_lambda), None)
    baseline_fusion_o = baseline_overall['fusion'] if baseline_overall else 0
    baseline_semantic_o = baseline_overall['semantic'] if baseline_overall else 0
    
    # Rankings
    overall_sorted_fusion = sorted(overall_results, key=lambda x: x['fusion'], reverse=True)
    overall_sorted_semantic = sorted(overall_results, key=lambda x: x['semantic'], reverse=True)
    
    fusion_ranks = {r['lambda_rep']: i+1 for i, r in enumerate(overall_sorted_fusion)}
    semantic_ranks = {r['lambda_rep']: i+1 for i, r in enumerate(overall_sorted_semantic)}
    
    for r in overall_results:
        delta_f = r['fusion'] - baseline_fusion_o
        delta_s = r['semantic'] - baseline_semantic_o
        f_rank = fusion_ranks[r['lambda_rep']]
        s_rank = semantic_ranks[r['lambda_rep']]
        
        marker = ''
        if f_rank == 1 and s_rank == 1:
            marker = ' ★★★'
        elif f_rank == 1:
            marker = ' ★F'
        elif s_rank == 1:
            marker = ' ★S'
        elif r['lambda_rep'] == baseline_lambda:
            marker = ' ★baseline'
        
        print(f"{r['lambda_rep']:<12.2f} {r['fusion']:<12.2f} {r['semantic']:<12.2f} "
              f"{delta_f:>+11.2f} {delta_s:>+11.2f} F{f_rank}/S{s_rank}{marker}")
    
    # Best lambda_rep
    print()
    print("=" * 100)
    print("Best Lambda_rep Values")
    print("=" * 100)
    print()
    
    best_fusion = overall_sorted_fusion[0]
    best_semantic = overall_sorted_semantic[0]
    
    print(f"Best for Fusion:   lambda_rep={best_fusion['lambda_rep']:.2f} ({best_fusion['fusion']:.2f}%)")
    print(f"Best for Semantic: lambda_rep={best_semantic['lambda_rep']:.2f} ({best_semantic['semantic']:.2f}%)")
    
    if best_fusion['lambda_rep'] == best_semantic['lambda_rep']:
        print(f"\n✓ Both metrics agree on lambda_rep={best_fusion['lambda_rep']:.2f}")
    else:
        print(f"\n✗ Metrics disagree (Fusion: {best_fusion['lambda_rep']:.2f}, Semantic: {best_semantic['lambda_rep']:.2f})")
    
    print()
    print("=" * 100)
    print("★★★ = Best for both Fusion and Semantic")
    print("★F  = Best for Fusion")
    print("★S  = Best for Semantic")
    print("★baseline = Baseline (lambda_rep=0.0)")
    print("=" * 100)

if __name__ == "__main__":
    main()
