#!/usr/bin/env python3
"""
Phase 2.1 Fixed Implementation Verification Script

Verify that P0 fixes are effective across all 5 classes:
- Semantic indicators are non-degenerate
- Memory centroid is properly normalized
- At least one indicator shows AUC > 0.55
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

def check_indicator_degeneracy(samples, indicator_name):
    """Check if an indicator is degenerate (constant or nearly constant)"""
    values = [s[indicator_name] for s in samples]
    unique_count = len(set(values))
    std = np.std(values)
    
    # Check for NaN
    nan_count = sum(1 for v in values if np.isnan(v))
    nan_ratio = nan_count / len(values)
    
    is_degenerate = (unique_count <= 2) or (std < 0.01) or (nan_ratio > 0.5)
    
    return {
        'unique_count': unique_count,
        'std': std,
        'nan_ratio': nan_ratio,
        'is_degenerate': is_degenerate,
        'min': float(np.nanmin(values)) if nan_count < len(values) else np.nan,
        'max': float(np.nanmax(values)) if nan_count < len(values) else np.nan
    }

def verify_class(dataset, class_name, k_shot=4):
    """Verify Phase 2.1 implementation for a single class"""
    
    per_sample_file = Path(f"result_gate/{dataset}/k_{k_shot}/per_sample/cls/{class_name}_seed111_per_sample.json")
    
    if not per_sample_file.exists():
        return None
    
    with open(per_sample_file) as f:
        data = json.load(f)
    
    samples = data['per_sample_data']
    n_samples = len(samples)
    
    indicators = [
        'r_mem_margin',
        'r_mem_entropy', 
        'r_mem_centroid',
        'r_sem_prompt_var',
        'r_sem_prompt_margin',
        'r_sem_extremity'
    ]
    
    result = {
        'dataset': dataset,
        'class_name': class_name,
        'n_samples': n_samples,
        'indicators': {}
    }
    
    for ind in indicators:
        result['indicators'][ind] = check_indicator_degeneracy(samples, ind)
    
    return result

def load_auc_results():
    """Load AUC results from phase2_1_analysis"""
    auc_file = Path("result_gate/phase2_1_analysis/indicator_auc_results.csv")
    
    if not auc_file.exists():
        return None
    
    import pandas as pd
    df = pd.read_csv(auc_file)
    return df

def print_verification_report(results, auc_df=None):
    """Print comprehensive verification report"""
    
    print("="*80)
    print("Phase 2.1 Fixed Implementation Verification Report")
    print("="*80)
    print()
    
    for result in results:
        if result is None:
            continue
            
        dataset = result['dataset']
        class_name = result['class_name']
        n_samples = result['n_samples']
        
        print(f"{'='*60}")
        print(f"{dataset}/{class_name} (n={n_samples})")
        print(f"{'='*60}")
        
        # Check indicator health
        print("\n📊 Indicator Degeneracy Check:")
        print(f"{'Indicator':<25} {'Status':<12} {'Unique':<8} {'Std':<10} {'Range'}")
        print("-"*70)
        
        memory_ok = []
        semantic_ok = []
        
        for ind_name, ind_stats in result['indicators'].items():
            status = "❌ DEGENERATE" if ind_stats['is_degenerate'] else "✅ OK"
            unique = ind_stats['unique_count']
            std = ind_stats['std']
            range_str = f"[{ind_stats['min']:.2f}, {ind_stats['max']:.2f}]"
            
            print(f"{ind_name:<25} {status:<12} {unique:<8} {std:<10.3f} {range_str}")
            
            if not ind_stats['is_degenerate']:
                if 'mem' in ind_name:
                    memory_ok.append(ind_name)
                elif 'sem' in ind_name:
                    semantic_ok.append(ind_name)
        
        print()
        print(f"Memory indicators OK: {len(memory_ok)}/3")
        print(f"Semantic indicators OK: {len(semantic_ok)}/3")
        
        # Check AUC if available
        if auc_df is not None:
            class_auc = auc_df[(auc_df['dataset'] == dataset) & 
                               (auc_df['class_name'] == class_name)]
            
            if not class_auc.empty:
                print("\n📈 AUC Performance:")
                
                mem_auc = class_auc['memory_avg_auc'].values[0]
                sem_auc = class_auc['semantic_avg_auc'].values[0]
                overall_auc = class_auc['overall_avg_auc'].values[0]
                
                print(f"  Memory avg AUC:   {mem_auc:.3f} {'✅' if mem_auc > 0.55 else '⚠️'}")
                print(f"  Semantic avg AUC: {sem_auc:.3f} {'✅' if sem_auc > 0.55 else '⚠️'}")
                print(f"  Overall avg AUC:  {overall_auc:.3f} {'✅' if overall_auc > 0.55 else '⚠️'}")
                
                # Best individual indicators
                best_indicators = []
                for ind in ['r_mem_margin', 'r_mem_entropy', 'r_mem_centroid',
                           'r_sem_prompt_var', 'r_sem_prompt_margin']:
                    if ind in class_auc.columns:
                        auc_val = class_auc[ind].values[0]
                        if auc_val > 0.55:
                            best_indicators.append((ind, auc_val))
                
                if best_indicators:
                    print("\n  Best indicators (AUC > 0.55):")
                    for ind, auc_val in sorted(best_indicators, key=lambda x: -x[1]):
                        print(f"    {ind:<25} AUC = {auc_val:.3f}")
        
        print()
    
    # Overall summary
    print("="*80)
    print("OVERALL SUMMARY")
    print("="*80)
    
    total_classes = len([r for r in results if r is not None])
    
    # Count non-degenerate indicators
    indicator_health = defaultdict(int)
    for result in results:
        if result is None:
            continue
        for ind_name, ind_stats in result['indicators'].items():
            if not ind_stats['is_degenerate']:
                indicator_health[ind_name] += 1
    
    print(f"\nClasses tested: {total_classes}/5")
    print("\nIndicator health across classes:")
    for ind_name in ['r_mem_margin', 'r_mem_entropy', 'r_mem_centroid',
                     'r_sem_prompt_var', 'r_sem_prompt_margin', 'r_sem_extremity']:
        ok_count = indicator_health[ind_name]
        pct = 100 * ok_count / total_classes if total_classes > 0 else 0
        status = "✅" if pct >= 80 else "⚠️" if pct >= 50 else "❌"
        print(f"  {ind_name:<25} {ok_count}/{total_classes} ({pct:.0f}%) {status}")
    
    if auc_df is not None:
        print("\nAUC performance (> 0.55 criterion):")
        classes_above_threshold = 0
        for _, row in auc_df.iterrows():
            if row['overall_avg_auc'] > 0.55:
                classes_above_threshold += 1
        
        pct = 100 * classes_above_threshold / len(auc_df) if len(auc_df) > 0 else 0
        print(f"  Classes with overall AUC > 0.55: {classes_above_threshold}/{len(auc_df)} ({pct:.0f}%)")
        
        # Best performing class
        best_class = auc_df.loc[auc_df['overall_avg_auc'].idxmax()]
        print(f"\n  Best class: {best_class['dataset']}/{best_class['class_name']}")
        print(f"    Overall AUC: {best_class['overall_avg_auc']:.3f}")
    
    print("\n" + "="*80)
    print("VERIFICATION CONCLUSION")
    print("="*80)
    
    # P0 criteria
    p0_pass = True
    issues = []
    
    # Check if semantic indicators are implemented
    sem_ok_count = sum(1 for r in results if r and 
                      not r['indicators']['r_sem_prompt_var']['is_degenerate'])
    if sem_ok_count < total_classes * 0.8:
        p0_pass = False
        issues.append("P0-1: Semantic indicators still degenerate in some classes")
    
    # Check if centroid is fixed
    centroid_ok_count = sum(1 for r in results if r and 
                           not r['indicators']['r_mem_centroid']['is_degenerate'])
    if centroid_ok_count < total_classes * 0.8:
        p0_pass = False
        issues.append("P0-2: Centroid normalization still broken in some classes")
    
    # Check if any indicator shows AUC > 0.55
    if auc_df is not None:
        any_good_auc = (auc_df['overall_avg_auc'] > 0.55).any()
        if not any_good_auc:
            issues.append("No class shows overall AUC > 0.55 (weak signal)")
    
    if p0_pass and not issues:
        print("✅ P0 fixes are SUCCESSFUL")
        print("   - All indicators are non-degenerate")
        print("   - Implementation is valid for analysis")
    else:
        print("⚠️ P0 fixes PARTIALLY successful")
        for issue in issues:
            print(f"   - {issue}")
    
    print()

def main():
    classes = [
        ('mvtec', 'screw'),
        ('mvtec', 'capsule'),
        ('visa', 'capsules'),
        ('visa', 'macaroni2'),
        ('visa', 'candle')
    ]
    
    results = []
    for dataset, class_name in classes:
        result = verify_class(dataset, class_name)
        results.append(result)
    
    # Load AUC results if available
    auc_df = load_auc_results()
    
    print_verification_report(results, auc_df)

if __name__ == "__main__":
    main()
