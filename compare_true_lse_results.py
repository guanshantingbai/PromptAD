#!/usr/bin/env python3
"""
比较真正的 LSE 聚合与 baseline 的性能
分析不同 τ 值的影响
"""

import os
import pandas as pd
import numpy as np

DATASETS = {
    'mvtec': ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 
              'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 
              'transistor', 'wood', 'zipper'],
    'visa': ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
             'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
}

TAU_VALUES = [5, 10, 20]
K = 2

def read_result(dataset, class_name, result_dir, k=2):
    """读取测试结果 - 从聚合结果文件中提取"""
    csv_path = f"{result_dir}/{dataset}/k_{k}/csv/Seed_111-results.csv"
    
    if not os.path.exists(csv_path):
        return None
    
    try:
        df = pd.read_csv(csv_path, index_col=0)
        # 查找对应类别的行
        row_name = f"{dataset}-{class_name}"
        if row_name in df.index and 'i_roc' in df.columns:
            return df.loc[row_name, 'i_roc']
        return None
    except Exception as e:
        print(f"Error reading {csv_path} for {class_name}: {e}")
        return None

def main():
    print("=" * 80)
    print("True LSE Aggregation Performance Comparison")
    print("=" * 80)
    print()
    
    # 收集所有结果
    results = {}
    
    for dataset_name, classes in DATASETS.items():
        print(f"\n{'='*80}")
        print(f"{dataset_name.upper()} (k={K})")
        print('='*80)
        print(f"{'Class':<15} {'Baseline':>10} ", end='')
        for tau in TAU_VALUES:
            print(f"{'τ='+str(tau):>10} {'Δ':>8}", end=' ')
        print()
        print('-'*80)
        
        baseline_scores = []
        tau_scores = {tau: [] for tau in TAU_VALUES}
        
        for class_name in classes:
            # Baseline from fusion_normal
            baseline = read_result(dataset_name, class_name, f'result/fusion_normal', k=K)
            
            # LSE results
            lse_results = {}
            for tau in TAU_VALUES:
                lse_val = read_result(dataset_name, class_name, f'result/test_true_lse_tau{tau}', k=K)
                lse_results[tau] = lse_val
            
            # 打印结果
            if baseline is not None:
                print(f"{class_name:<15} {baseline:>9.2f}% ", end='')
                baseline_scores.append(baseline)
                
                for tau in TAU_VALUES:
                    if lse_results[tau] is not None:
                        delta = lse_results[tau] - baseline
                        tau_scores[tau].append(lse_results[tau])
                        
                        # 标记改善/退化
                        marker = '↑' if delta > 0 else '↓' if delta < 0 else '='
                        print(f"{lse_results[tau]:>9.2f}% {delta:>+7.2f}{marker}", end=' ')
                    else:
                        print(f"{'N/A':>10} {'N/A':>8}", end=' ')
                print()
            else:
                print(f"{class_name:<15} {'N/A':>10}")
        
        # 计算平均值
        if baseline_scores:
            print('-'*80)
            avg_baseline = np.mean(baseline_scores)
            print(f"{'Average':<15} {avg_baseline:>9.2f}% ", end='')
            
            for tau in TAU_VALUES:
                if tau_scores[tau]:
                    avg_tau = np.mean(tau_scores[tau])
                    avg_delta = avg_tau - avg_baseline
                    marker = '↑' if avg_delta > 0 else '↓' if avg_delta < 0 else '='
                    print(f"{avg_tau:>9.2f}% {avg_delta:>+7.2f}{marker}", end=' ')
            print()
            
            results[dataset_name] = {
                'baseline': baseline_scores,
                'tau_scores': tau_scores
            }
    
    # 总体统计
    print("\n" + "="*80)
    print("Overall Statistics")
    print("="*80)
    
    all_baseline = []
    all_tau_scores = {tau: [] for tau in TAU_VALUES}
    
    for dataset_name, data in results.items():
        all_baseline.extend(data['baseline'])
        for tau in TAU_VALUES:
            all_tau_scores[tau].extend(data['tau_scores'][tau])
    
    print(f"\nBaseline Average: {np.mean(all_baseline):.2f}%")
    for tau in TAU_VALUES:
        if all_tau_scores[tau]:
            avg = np.mean(all_tau_scores[tau])
            delta = avg - np.mean(all_baseline)
            print(f"τ={tau:2d} Average:    {avg:.2f}% (Δ {delta:+.2f}%)")
    
    # τ 敏感性分析
    print("\n" + "="*80)
    print("τ Sensitivity Analysis")
    print("="*80)
    
    tau_values_list = [np.mean(all_tau_scores[tau]) for tau in TAU_VALUES if all_tau_scores[tau]]
    if len(tau_values_list) >= 2:
        tau_range = max(tau_values_list) - min(tau_values_list)
        print(f"Range across τ={TAU_VALUES}: {tau_range:.4f}%")
        
        if tau_range < 0.1:
            print("⚠️  Very low τ sensitivity (< 0.1%)")
        elif tau_range < 0.5:
            print("⚠️  Low τ sensitivity (< 0.5%)")
        elif tau_range < 2.0:
            print("✓  Moderate τ sensitivity")
        else:
            print("✓✓ High τ sensitivity")
    
    # 改善/退化统计
    print("\n" + "="*80)
    print("Improvement/Degradation Analysis")
    print("="*80)
    
    for tau in TAU_VALUES:
        improved = sum(1 for b, t in zip(all_baseline, all_tau_scores[tau]) if t > b)
        degraded = sum(1 for b, t in zip(all_baseline, all_tau_scores[tau]) if t < b)
        total = len(all_baseline)
        
        print(f"\nτ={tau}:")
        print(f"  Improved:  {improved}/{total} ({100*improved/total:.1f}%)")
        print(f"  Degraded:  {degraded}/{total} ({100*degraded/total:.1f}%)")
        print(f"  Unchanged: {total-improved-degraded}/{total}")

if __name__ == '__main__':
    main()
