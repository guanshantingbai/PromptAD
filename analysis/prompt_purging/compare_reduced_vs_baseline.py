"""
比较 baseline_reducedprompt 和 baseline 的结果
对比 i_roc 和 semantic_i_roc 的变化
"""

import os
import pandas as pd
import numpy as np

def load_results(root_dir, dataset, k_shot):
    """加载指定路径的结果CSV"""
    csv_path = f"{root_dir}/{dataset}/k_{k_shot}/csv/Seed_111-results.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, index_col=0)
        return df
    else:
        print(f"[WARNING] File not found: {csv_path}")
        return None

def compare_results(baseline_dir, reduced_dir, dataset, k_shot, output_dir):
    """比较两个结果并生成对比报告"""
    
    # 加载数据
    df_baseline = load_results(baseline_dir, dataset, k_shot)
    df_reduced = load_results(reduced_dir, dataset, k_shot)
    
    if df_baseline is None or df_reduced is None:
        print(f"[SKIP] Missing data for {dataset} k={k_shot}")
        return None
    
    # 找到两边都存在的类别（取交集）
    common_classes = df_baseline.index.intersection(df_reduced.index)
    
    if len(common_classes) == 0:
        print(f"[ERROR] No common classes found for {dataset} k={k_shot}")
        return None
    
    # 检查缺失的类别
    missing_in_reduced = set(df_baseline.index) - set(df_reduced.index)
    missing_in_baseline = set(df_reduced.index) - set(df_baseline.index)
    
    if missing_in_reduced:
        print(f"[WARNING] Classes in baseline but not in reduced: {missing_in_reduced}")
    if missing_in_baseline:
        print(f"[WARNING] Classes in reduced but not in baseline: {missing_in_baseline}")
    
    # 只对比共同的类别
    df_baseline = df_baseline.loc[common_classes]
    df_reduced = df_reduced.loc[common_classes]
    
    # 创建对比DataFrame
    comparison = pd.DataFrame()
    comparison['class'] = df_baseline.index
    
    # Baseline结果
    comparison['baseline_i_roc'] = df_baseline['i_roc'].values
    comparison['baseline_semantic_i_roc'] = df_baseline['semantic_i_roc'].values
    
    # Reduced结果
    comparison['reduced_i_roc'] = df_reduced['i_roc'].values
    comparison['reduced_semantic_i_roc'] = df_reduced['semantic_i_roc'].values
    
    # 计算差值 (Reduced - Baseline)
    comparison['delta_i_roc'] = comparison['reduced_i_roc'] - comparison['baseline_i_roc']
    comparison['delta_semantic_i_roc'] = comparison['reduced_semantic_i_roc'] - comparison['baseline_semantic_i_roc']
    
    # 计算平均值
    avg_row = {
        'class': 'AVERAGE',
        'baseline_i_roc': comparison['baseline_i_roc'].mean(),
        'baseline_semantic_i_roc': comparison['baseline_semantic_i_roc'].mean(),
        'reduced_i_roc': comparison['reduced_i_roc'].mean(),
        'reduced_semantic_i_roc': comparison['reduced_semantic_i_roc'].mean(),
        'delta_i_roc': comparison['delta_i_roc'].mean(),
        'delta_semantic_i_roc': comparison['delta_semantic_i_roc'].mean()
    }
    
    comparison = pd.concat([comparison, pd.DataFrame([avg_row])], ignore_index=True)
    
    # 保存对比结果
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/{dataset}_k{k_shot}_comparison.csv"
    comparison.to_csv(output_path, index=False, float_format='%.2f')
    
    print(f"\n{'='*80}")
    print(f"[{dataset.upper()} k={k_shot}] Comparison Results")
    print(f"{'='*80}")
    print(f"Output: {output_path}")
    print(f"\n{'Class':<25} {'Baseline':<10} {'Reduced':<10} {'Δ i_roc':<10} {'Δ Semantic':<12}")
    print(f"{'-'*80}")
    
    for idx, row in comparison.iterrows():
        class_name = row['class']
        baseline_iroc = row['baseline_i_roc']
        reduced_iroc = row['reduced_i_roc']
        delta_iroc = row['delta_i_roc']
        delta_semantic = row['delta_semantic_i_roc']
        
        if class_name == 'AVERAGE':
            print(f"{'-'*80}")
            print(f"{'AVERAGE':<25} {baseline_iroc:<10.2f} {reduced_iroc:<10.2f} {delta_iroc:+10.2f} {delta_semantic:+12.2f}")
        else:
            print(f"{class_name:<25} {baseline_iroc:<10.2f} {reduced_iroc:<10.2f} {delta_iroc:+10.2f} {delta_semantic:+12.2f}")
    
    return comparison

def generate_summary_report(baseline_dir, reduced_dir, output_dir):
    """生成汇总报告"""
    
    datasets = ['mvtec', 'visa']
    k_shots = [1, 2, 4]
    
    summary_data = []
    
    for dataset in datasets:
        for k_shot in k_shots:
            result = compare_results(baseline_dir, reduced_dir, dataset, k_shot, output_dir)
            
            if result is not None:
                # 提取平均值行
                avg_row = result[result['class'] == 'AVERAGE'].iloc[0]
                
                summary_data.append({
                    'dataset': dataset,
                    'k_shot': k_shot,
                    'baseline_i_roc_avg': avg_row['baseline_i_roc'],
                    'reduced_i_roc_avg': avg_row['reduced_i_roc'],
                    'delta_i_roc_avg': avg_row['delta_i_roc'],
                    'baseline_semantic_i_roc_avg': avg_row['baseline_semantic_i_roc'],
                    'reduced_semantic_i_roc_avg': avg_row['reduced_semantic_i_roc'],
                    'delta_semantic_i_roc_avg': avg_row['delta_semantic_i_roc']
                })
    
    # 生成汇总表
    summary_df = pd.DataFrame(summary_data)
    summary_path = f"{output_dir}/summary_all.csv"
    summary_df.to_csv(summary_path, index=False, float_format='%.2f')
    
    print(f"\n{'='*80}")
    print(f"OVERALL SUMMARY")
    print(f"{'='*80}")
    print(f"Output: {summary_path}")
    print(f"\n{'Dataset':<10} {'k':<5} {'Baseline':<12} {'Reduced':<12} {'Δ i_roc':<12} {'Δ Semantic':<12}")
    print(f"{'-'*80}")
    
    for idx, row in summary_df.iterrows():
        print(f"{row['dataset']:<10} {row['k_shot']:<5} "
              f"{row['baseline_i_roc_avg']:<12.2f} {row['reduced_i_roc_avg']:<12.2f} "
              f"{row['delta_i_roc_avg']:+12.2f} {row['delta_semantic_i_roc_avg']:+12.2f}")
    
    print(f"\n{'='*80}")
    print(f"✅ Analysis complete!")
    print(f"   - Individual comparisons: {output_dir}/*_comparison.csv")
    print(f"   - Summary report: {summary_path}")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    # 定义路径
    # 对比2: Purge1 (通用清洗) vs Purge2 (通用+类别清洗)
    baseline_dir = './result/baseline_reducedprompt'  # Purge1 作为新 baseline
    reduced_dir = './result/baseline_reducedprompt2'  # Purge2 (类别级清洗)
    output_dir = './analysis/prompt_purging'
    
    # 生成对比报告
    generate_summary_report(baseline_dir, reduced_dir, output_dir)
