#!/usr/bin/env python3
"""
Compare margin ablation results
Analyze the impact of different margin values on model performance
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Margin values tested
MARGINS = [0.0, 0.1, 0.2, 0.5, 0.8]

# Result paths
base_path = Path("result/margin_ablation")

def load_results(margin):
    """Load results for a specific margin value"""
    mvtec_csv = base_path / f"margin_{margin}" / "mvtec" / "k_2" / "csv" / "Seed_111-results.csv"
    visa_csv = base_path / f"margin_{margin}" / "visa" / "k_2" / "csv" / "Seed_111-results.csv"
    
    results = {}
    
    # Load MVTec results
    if mvtec_csv.exists():
        df_mvtec = pd.read_csv(mvtec_csv)
        results['mvtec'] = df_mvtec
    else:
        print(f"Warning: {mvtec_csv} not found")
        
    # Load ViSA results
    if visa_csv.exists():
        df_visa = pd.read_csv(visa_csv)
        results['visa'] = df_visa
    else:
        print(f"Warning: {visa_csv} not found")
        
    return results

def calculate_averages(df):
    """Calculate average performance metrics"""
    # Column names: i_roc (fusion), memory_i_roc (memory), semantic_i_roc (semantic)
    metrics_mapping = {
        'i_roc': 'image_rocauc_fusion',
        'memory_i_roc': 'image_rocauc_memory', 
        'semantic_i_roc': 'image_rocauc_semantic'
    }
    
    avg_results = {}
    
    for csv_col, result_key in metrics_mapping.items():
        if csv_col in df.columns:
            avg_results[result_key] = df[csv_col].mean()
            
    return avg_results

def main():
    print("=" * 80)
    print("Margin Ablation Study - Results Comparison")
    print("=" * 80)
    print()
    
    # Store all results
    all_results = {}
    
    for margin in MARGINS:
        print(f"\n{'='*80}")
        print(f"Margin = {margin}")
        print(f"{'='*80}")
        
        results = load_results(margin)
        all_results[margin] = results
        
        for dataset, df in results.items():
            print(f"\n[{dataset.upper()}]")
            print(f"Classes: {len(df)}")
            
            avg = calculate_averages(df)
            print(f"  Fusion:   {avg.get('image_rocauc_fusion', 0):.2f}%")
            print(f"  Memory:   {avg.get('image_rocauc_memory', 0):.2f}%")
            print(f"  Semantic: {avg.get('image_rocauc_semantic', 0):.2f}%")
            
    # Summary comparison table
    print("\n" + "=" * 80)
    print("Summary Comparison Table")
    print("=" * 80)
    print()
    
    # MVTec comparison
    print("MVTec Dataset (15 classes):")
    print(f"{'Margin':<10} {'Fusion':<10} {'Memory':<10} {'Semantic':<10} {'Δ Fusion':<10} {'Δ Semantic':<10}")
    print("-" * 70)
    
    baseline_margin = 0.5
    baseline_mvtec = all_results[baseline_margin].get('mvtec')
    baseline_avg_mvtec = calculate_averages(baseline_mvtec) if baseline_mvtec is not None else {}
    
    for margin in MARGINS:
        results = all_results[margin].get('mvtec')
        if results is not None:
            avg = calculate_averages(results)
            fusion = avg.get('image_rocauc_fusion', 0)
            memory = avg.get('image_rocauc_memory', 0)
            semantic = avg.get('image_rocauc_semantic', 0)
            
            delta_fusion = fusion - baseline_avg_mvtec.get('image_rocauc_fusion', 0)
            delta_semantic = semantic - baseline_avg_mvtec.get('image_rocauc_semantic', 0)
            
            marker = " ★" if margin == baseline_margin else ""
            print(f"{margin:<10.1f} {fusion:<10.2f} {memory:<10.2f} {semantic:<10.2f} "
                  f"{delta_fusion:>+9.2f} {delta_semantic:>+9.2f}{marker}")
    
    print()
    
    # ViSA comparison
    print("ViSA Dataset (12 classes):")
    print(f"{'Margin':<10} {'Fusion':<10} {'Memory':<10} {'Semantic':<10} {'Δ Fusion':<10} {'Δ Semantic':<10}")
    print("-" * 70)
    
    baseline_visa = all_results[baseline_margin].get('visa')
    baseline_avg_visa = calculate_averages(baseline_visa) if baseline_visa is not None else {}
    
    for margin in MARGINS:
        results = all_results[margin].get('visa')
        if results is not None:
            avg = calculate_averages(results)
            fusion = avg.get('image_rocauc_fusion', 0)
            memory = avg.get('image_rocauc_memory', 0)
            semantic = avg.get('image_rocauc_semantic', 0)
            
            delta_fusion = fusion - baseline_avg_visa.get('image_rocauc_fusion', 0)
            delta_semantic = semantic - baseline_avg_visa.get('image_rocauc_semantic', 0)
            
            marker = " ★" if margin == baseline_margin else ""
            print(f"{margin:<10.1f} {fusion:<10.2f} {memory:<10.2f} {semantic:<10.2f} "
                  f"{delta_fusion:>+9.2f} {delta_semantic:>+9.2f}{marker}")
    
    print()
    
    # Overall average
    print("Overall Average (27 classes):")
    print(f"{'Margin':<10} {'Fusion':<10} {'Memory':<10} {'Semantic':<10} {'Δ Fusion':<10} {'Δ Semantic':<10}")
    print("-" * 70)
    
    baseline_fusion_overall = (baseline_avg_mvtec.get('image_rocauc_fusion', 0) * 15 + 
                               baseline_avg_visa.get('image_rocauc_fusion', 0) * 12) / 27
    baseline_semantic_overall = (baseline_avg_mvtec.get('image_rocauc_semantic', 0) * 15 + 
                                 baseline_avg_visa.get('image_rocauc_semantic', 0) * 12) / 27
    
    for margin in MARGINS:
        mvtec_results = all_results[margin].get('mvtec')
        visa_results = all_results[margin].get('visa')
        
        if mvtec_results is not None and visa_results is not None:
            mvtec_avg = calculate_averages(mvtec_results)
            visa_avg = calculate_averages(visa_results)
            
            fusion = (mvtec_avg.get('image_rocauc_fusion', 0) * 15 + 
                     visa_avg.get('image_rocauc_fusion', 0) * 12) / 27
            memory = (mvtec_avg.get('image_rocauc_memory', 0) * 15 + 
                     visa_avg.get('image_rocauc_memory', 0) * 12) / 27
            semantic = (mvtec_avg.get('image_rocauc_semantic', 0) * 15 + 
                       visa_avg.get('image_rocauc_semantic', 0) * 12) / 27
            
            delta_fusion = fusion - baseline_fusion_overall
            delta_semantic = semantic - baseline_semantic_overall
            
            marker = " ★" if margin == baseline_margin else ""
            print(f"{margin:<10.1f} {fusion:<10.2f} {memory:<10.2f} {semantic:<10.2f} "
                  f"{delta_fusion:>+9.2f} {delta_semantic:>+9.2f}{marker}")
    
    print()
    print("=" * 80)
    print("★ = Baseline (margin=0.5)")
    print("Δ = Delta from baseline")
    print("=" * 80)
    
    # Find best margin for each metric
    print("\nBest Margin Values:")
    print("-" * 40)
    
    best_fusion_margin = None
    best_fusion_score = 0
    best_semantic_margin = None
    best_semantic_score = 0
    
    for margin in MARGINS:
        mvtec_results = all_results[margin].get('mvtec')
        visa_results = all_results[margin].get('visa')
        
        if mvtec_results is not None and visa_results is not None:
            mvtec_avg = calculate_averages(mvtec_results)
            visa_avg = calculate_averages(visa_results)
            
            fusion = (mvtec_avg.get('image_rocauc_fusion', 0) * 15 + 
                     visa_avg.get('image_rocauc_fusion', 0) * 12) / 27
            semantic = (mvtec_avg.get('image_rocauc_semantic', 0) * 15 + 
                       visa_avg.get('image_rocauc_semantic', 0) * 12) / 27
            
            if fusion > best_fusion_score:
                best_fusion_score = fusion
                best_fusion_margin = margin
                
            if semantic > best_semantic_score:
                best_semantic_score = semantic
                best_semantic_margin = margin
    
    print(f"Best Fusion:   margin={best_fusion_margin} ({best_fusion_score:.2f}%)")
    print(f"Best Semantic: margin={best_semantic_margin} ({best_semantic_score:.2f}%)")
    print()
    
    # Per-class analysis for extreme cases
    print("\n" + "=" * 80)
    print("Per-Class Analysis (Classes with Large Variance)")
    print("=" * 80)
    
    for dataset in ['mvtec', 'visa']:
        print(f"\n[{dataset.upper()}]")
        
        # Collect all class results across margins
        class_results = {}
        
        for margin in MARGINS:
            results = all_results[margin].get(dataset)
            if results is not None:
                for idx, row in results.iterrows():
                    # Index contains class name (e.g., 'mvtec-carpet')
                    class_name = str(idx).split('-')[-1] if '-' in str(idx) else str(idx)
                    if class_name not in class_results:
                        class_results[class_name] = {'margins': [], 'fusion': [], 'semantic': []}
                    
                    class_results[class_name]['margins'].append(margin)
                    class_results[class_name]['fusion'].append(row.get('i_roc', 0))
                    class_results[class_name]['semantic'].append(row.get('semantic_i_roc', 0))
        
        # Find classes with high variance
        variance_data = []
        for class_name, data in class_results.items():
            fusion_variance = np.var(data['fusion'])
            semantic_variance = np.var(data['semantic'])
            variance_data.append((class_name, fusion_variance, semantic_variance))
        
        # Sort by fusion variance
        variance_data.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\nTop 5 classes with highest Fusion variance:")
        for class_name, fusion_var, semantic_var in variance_data[:5]:
            print(f"  {class_name:<15} Fusion Var: {fusion_var:>8.2f}, Semantic Var: {semantic_var:>8.2f}")
            data = class_results[class_name]
            print(f"    {'Margin':<10} {'Fusion':<10} {'Semantic':<10}")
            for i, margin in enumerate(data['margins']):
                print(f"    {margin:<10.1f} {data['fusion'][i]:<10.2f} {data['semantic'][i]:<10.2f}")

if __name__ == "__main__":
    main()
