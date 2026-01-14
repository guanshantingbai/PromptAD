#!/usr/bin/env python3
"""
分析LSE测试结果 (包括旧的softmax版本)
对比不同tau值和baseline
"""

import os
import pandas as pd
import numpy as np

# 数据集配置
DATASETS = {
    'mvtec': ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 
              'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 
              'transistor', 'wood', 'zipper'],
    'visa': ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
             'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
}

K = 2

def read_aggregated_results(result_dir, dataset):
    """读取聚合后的结果文件"""
    csv_path = f"{result_dir}/{dataset}/k_{K}/csv/Seed_111-results.csv"
    
    if not os.path.exists(csv_path):
        return None
    
    try:
        df = pd.read_csv(csv_path, index_col=0)
        # 提取image AUROC (第一列: i_roc)
        results = {}
        for idx, row in df.iterrows():
            class_name = idx.split('-')[1]  # 'mvtec-bottle' -> 'bottle'
            results[class_name] = row['i_roc']
        return results
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return None

def main():
    print("="*80)
    print("LSE Aggregation Results Analysis")
    print("="*80)
    print("\n[注意] 旧LSE测试使用的是 softmax-weighted average")
    print("[注意] 需要用新的真正LSE实现重新测试以获得准确对比")
    print("="*80)
    
    # 定义要对比的结果目录
    result_dirs = {
        'Baseline (fusion_normal)': 'result/fusion_normal',
        'LSE τ=0.1 (OLD)': 'result/test_lse_tau0.1_fixed',
        'LSE τ=1.0 (OLD)': 'result/test_lse_tau1_fixed',
        'LSE τ=10 (OLD)': 'result/test_lse_tau10_fixed',
    }
    
    # 检查哪些结果目录存在
    available_dirs = {}
    for name, path in result_dirs.items():
        if os.path.exists(path):
            available_dirs[name] = path
        else:
            print(f"⚠️  {name} 不存在: {path}")
    
    if not available_dirs:
        print("\n❌ 没有找到任何结果目录！")
        return
    
    print(f"\n找到 {len(available_dirs)} 个结果目录\n")
    
    # 分析每个数据集
    for dataset_name, class_list in DATASETS.items():
        print(f"\n{'='*80}")
        print(f"{dataset_name.upper()} Dataset (k={K})")
        print('='*80)
        
        # 读取所有结果
        all_results = {}
        for name, path in available_dirs.items():
            results = read_aggregated_results(path, dataset_name)
            if results:
                all_results[name] = results
        
        if not all_results:
            print(f"⚠️  没有找到 {dataset_name} 的结果")
            continue
        
        # 打印表头
        header_cols = list(all_results.keys())
        print(f"{'Class':<15}", end=' ')
        for col in header_cols:
            print(f"{col:>18}", end=' ')
        
        # 如果有baseline，打印delta列
        if 'Baseline (fusion_normal)' in all_results:
            for col in header_cols[1:]:  # 跳过baseline自己
                print(f"{'Δ':>8}", end=' ')
        print()
        print('-'*80)
        
        # 打印每个类别的结果
        class_scores = {name: [] for name in all_results.keys()}
        
        for class_name in class_list:
            print(f"{class_name:<15}", end=' ')
            
            baseline_val = None
            if 'Baseline (fusion_normal)' in all_results:
                baseline_val = all_results['Baseline (fusion_normal)'].get(class_name)
            
            # 打印每列的值
            for col in header_cols:
                val = all_results[col].get(class_name)
                if val is not None:
                    print(f"{val:>17.2f}%", end=' ')
                    class_scores[col].append(val)
                else:
                    print(f"{'N/A':>18}", end=' ')
            
            # 打印delta
            if baseline_val is not None:
                for col in header_cols[1:]:
                    val = all_results[col].get(class_name)
                    if val is not None:
                        delta = val - baseline_val
                        marker = '↑' if delta > 0.1 else '↓' if delta < -0.1 else '='
                        print(f"{delta:>+7.2f}{marker}", end=' ')
                    else:
                        print(f"{'':>8}", end=' ')
            print()
        
        # 打印平均值
        print('-'*80)
        print(f"{'Average':<15}", end=' ')
        
        baseline_avg = None
        for col in header_cols:
            if class_scores[col]:
                avg = np.mean(class_scores[col])
                print(f"{avg:>17.2f}%", end=' ')
                if col == 'Baseline (fusion_normal)':
                    baseline_avg = avg
            else:
                print(f"{'N/A':>18}", end=' ')
        
        # 打印平均delta
        if baseline_avg is not None:
            for col in header_cols[1:]:
                if class_scores[col]:
                    avg = np.mean(class_scores[col])
                    delta = avg - baseline_avg
                    marker = '↑' if delta > 0.1 else '↓' if delta < -0.1 else '='
                    print(f"{delta:>+7.2f}{marker}", end=' ')
        print()
    
    # 总体统计
    print("\n" + "="*80)
    print("Overall Analysis")
    print("="*80)
    
    # 1. 不同tau之间的差异
    print("\n1️⃣  不同 τ 值之间的差异:")
    tau_cols = [col for col in available_dirs.keys() if 'LSE' in col]
    if len(tau_cols) >= 2:
        print(f"   发现 {len(tau_cols)} 个LSE配置")
        
        # 收集所有LSE结果
        lse_averages = {}
        for col in tau_cols:
            all_scores = []
            for dataset_name in DATASETS.keys():
                results = read_aggregated_results(available_dirs[col], dataset_name)
                if results:
                    all_scores.extend(results.values())
            if all_scores:
                lse_averages[col] = np.mean(all_scores)
        
        if lse_averages:
            print("\n   各τ的总平均 AUROC:")
            for col, avg in sorted(lse_averages.items()):
                print(f"     {col:30s}: {avg:.2f}%")
            
            tau_range = max(lse_averages.values()) - min(lse_averages.values())
            print(f"\n   τ 敏感性 (极差): {tau_range:.4f}%")
            
            if tau_range < 0.05:
                print("   ⚠️  极低的τ敏感性 (< 0.05%) - 旧softmax实现的已知问题")
            elif tau_range < 0.5:
                print("   ⚠️  低τ敏感性 (< 0.5%)")
            else:
                print("   ✓  有明显的τ敏感性")
    
    # 2. 与baseline对比
    print("\n2️⃣  与 Baseline 对比:")
    if 'Baseline (fusion_normal)' in available_dirs:
        baseline_all = []
        for dataset_name in DATASETS.keys():
            results = read_aggregated_results(available_dirs['Baseline (fusion_normal)'], dataset_name)
            if results:
                baseline_all.extend(results.values())
        
        baseline_avg = np.mean(baseline_all) if baseline_all else None
        
        if baseline_avg:
            print(f"   Baseline 总平均: {baseline_avg:.2f}%")
            print()
            
            for col in tau_cols:
                lse_all = []
                for dataset_name in DATASETS.keys():
                    results = read_aggregated_results(available_dirs[col], dataset_name)
                    if results:
                        lse_all.extend(results.values())
                
                if lse_all:
                    lse_avg = np.mean(lse_all)
                    delta = lse_avg - baseline_avg
                    marker = '✓' if delta > 0 else '✗'
                    print(f"   {col:30s}: {lse_avg:.2f}% (Δ {delta:+.2f}%) {marker}")
    
    # 3. 最佳tau
    print("\n3️⃣  最佳 τ 配置:")
    if lse_averages:
        best_tau = max(lse_averages.items(), key=lambda x: x[1])
        print(f"   {best_tau[0]}: {best_tau[1]:.2f}%")
        
        if baseline_avg:
            improvement = best_tau[1] - baseline_avg
            if improvement > 0:
                print(f"   相比baseline改进: +{improvement:.2f}% ✓")
            else:
                print(f"   相比baseline退化: {improvement:.2f}% ✗")
    
    print("\n" + "="*80)
    print("⚠️  重要提示:")
    print("   这些结果使用的是旧的 softmax-weighted LSE 实现")
    print("   需要使用新的真正LSE实现 (τ * logsumexp) 重新测试")
    print("   预期新实现会展现出更强的 τ 敏感性")
    print("="*80)

if __name__ == '__main__':
    main()
