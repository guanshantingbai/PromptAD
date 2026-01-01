#!/usr/bin/env python
"""
Phase 1 综合分析工具
关联prompt风险指标与baseline semantic_auroc性能
"""

import os
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


def load_phase1_results(dataset, k_shot=2):
    """加载所有Phase 1结果"""
    result_dir = f'result/prompt_purging/phase1/{dataset}/k_{k_shot}'
    
    if not os.path.exists(result_dir):
        print(f"❌ 目录不存在: {result_dir}")
        return None
    
    csv_files = sorted([f for f in os.listdir(result_dir) if f.endswith('.csv')])
    
    all_results = []
    class_summaries = []
    
    for csv_file in csv_files:
        # 提取类别名
        classname = csv_file.replace('_phase1_normal_side_risk_eps0.05.csv', '')
        csv_path = os.path.join(result_dir, csv_file)
        
        df = pd.read_csv(csv_path)
        
        # 类别级别摘要
        summary = {
            'dataset': dataset,
            'class': classname,
            'total_prompts': len(df),
            'generic_prompts': (df['type'] == 'generic').sum(),
            'specific_prompts': (df['type'] == 'specific').sum(),
            'high_risk_count': (df['risk_level'] == 'high').sum(),
            'high_risk_pct': (df['risk_level'] == 'high').sum() / len(df) * 100,
            'mean_R_j_eps': df['R_j_eps'].mean(),
            'median_R_j_eps': df['R_j_eps'].median(),
            'max_R_j_eps': df['R_j_eps'].max(),
            'mean_R_j_0': df['R_j_0'].mean(),
            'max_R_j_0': df['R_j_0'].max(),
            'mean_median_margin': df['median_margin'].mean(),
            'min_median_margin': df['median_margin'].min(),
            'num_negative_margin': (df['median_margin'] < 0).sum(),
            'pct_negative_margin': (df['median_margin'] < 0).sum() / len(df) * 100,
        }
        
        class_summaries.append(summary)
        
        # 保留每条prompt的详细信息
        df['class'] = classname
        df['dataset'] = dataset
        all_results.append(df)
    
    summary_df = pd.DataFrame(class_summaries)
    detail_df = pd.concat(all_results, ignore_index=True) if all_results else None
    
    return summary_df, detail_df


def load_baseline_results(dataset, k_shot=2, seed=111):
    """加载baseline性能结果"""
    result_file = f'result/baseline/{dataset}/k_{k_shot}/csv/Seed_{seed}-results.csv'
    
    if not os.path.exists(result_file):
        print(f"❌ Baseline结果不存在: {result_file}")
        return None
    
    df = pd.read_csv(result_file)
    
    # 提取类别名（去掉dataset前缀）
    df['class'] = df.iloc[:, 0].str.replace(f'{dataset}-', '')
    
    # 重命名列
    df = df.rename(columns={
        'i_roc': 'image_auroc',
        'p_roc': 'pixel_auroc',
        'semantic_i_roc': 'semantic_auroc',
        'memory_i_roc': 'memory_auroc'
    })
    
    return df[['class', 'image_auroc', 'pixel_auroc', 'semantic_auroc', 'memory_auroc']]


def merge_and_analyze(phase1_summary, baseline_results):
    """合并Phase 1和baseline结果，进行关联分析"""
    
    merged = pd.merge(phase1_summary, baseline_results, on='class', how='inner')
    
    # 计算相关性
    correlations = {}
    
    risk_metrics = [
        'high_risk_pct', 'mean_R_j_eps', 'median_R_j_eps', 'max_R_j_eps',
        'mean_R_j_0', 'mean_median_margin', 'pct_negative_margin'
    ]
    
    performance_metrics = ['semantic_auroc', 'memory_auroc', 'image_auroc']
    
    for risk_m in risk_metrics:
        for perf_m in performance_metrics:
            corr = merged[risk_m].corr(merged[perf_m])
            correlations[f'{risk_m}_vs_{perf_m}'] = corr
    
    return merged, correlations


def print_analysis_report(dataset, merged_df, correlations):
    """打印分析报告"""
    
    print("="*80)
    print(f"Phase 1 分析报告 - {dataset.upper()}")
    print("="*80)
    
    # 1. 类别级别摘要
    print("\n📊 各类别Prompt风险与性能总览")
    print("-"*80)
    
    # 按semantic_auroc排序显示
    display_df = merged_df.sort_values('semantic_auroc', ascending=False)
    
    print(f"{'Class':<15} {'Total':>5} {'High%':>6} {'R_eps':>6} {'NegMar%':>7} {'SemAUC':>7} {'MemAUC':>7}")
    print("-"*80)
    
    for _, row in display_df.iterrows():
        classname = row['class']
        total = int(row['total_prompts'])
        high_pct = row['high_risk_pct']
        r_eps = row['mean_R_j_eps']
        neg_pct = row['pct_negative_margin']
        sem_auc = row['semantic_auroc']
        mem_auc = row['memory_auroc']
        
        print(f"{classname:<15} {total:>5} {high_pct:>6.1f} {r_eps:>6.3f} {neg_pct:>6.1f}% {sem_auc:>7.2f} {mem_auc:>7.2f}")
    
    # 2. 统计摘要
    print("\n📈 统计摘要")
    print("-"*80)
    print(f"类别数: {len(merged_df)}")
    print(f"\nPrompt风险指标:")
    print(f"  平均高风险比例: {merged_df['high_risk_pct'].mean():.1f}% (std={merged_df['high_risk_pct'].std():.1f})")
    print(f"  平均R_j_eps: {merged_df['mean_R_j_eps'].mean():.3f} (std={merged_df['mean_R_j_eps'].std():.3f})")
    print(f"  平均负margin比例: {merged_df['pct_negative_margin'].mean():.1f}% (std={merged_df['pct_negative_margin'].std():.1f})")
    
    print(f"\n性能指标:")
    print(f"  平均Semantic AUROC: {merged_df['semantic_auroc'].mean():.2f} (std={merged_df['semantic_auroc'].std():.2f})")
    print(f"  平均Memory AUROC: {merged_df['memory_auroc'].mean():.2f} (std={merged_df['memory_auroc'].std():.2f})")
    print(f"  平均Image AUROC: {merged_df['image_auroc'].mean():.2f} (std={merged_df['image_auroc'].std():.2f})")
    
    # 3. 关键相关性分析
    print("\n🔗 关键相关性分析 (Prompt风险 vs 性能)")
    print("-"*80)
    
    key_correlations = [
        ('high_risk_pct_vs_semantic_auroc', '高风险比例 vs Semantic AUROC'),
        ('mean_R_j_eps_vs_semantic_auroc', '平均R_j_eps vs Semantic AUROC'),
        ('pct_negative_margin_vs_semantic_auroc', '负margin比例 vs Semantic AUROC'),
        ('high_risk_pct_vs_memory_auroc', '高风险比例 vs Memory AUROC'),
        ('mean_R_j_eps_vs_memory_auroc', '平均R_j_eps vs Memory AUROC'),
    ]
    
    for key, label in key_correlations:
        if key in correlations:
            corr = correlations[key]
            strength = "强" if abs(corr) > 0.5 else "中等" if abs(corr) > 0.3 else "弱"
            direction = "负相关" if corr < 0 else "正相关"
            print(f"{label:<45}: {corr:>7.3f} ({strength}{direction})")
    
    # 4. 问题类别识别
    print("\n⚠️  高风险类别识别")
    print("-"*80)
    
    # 高风险且性能差的类别
    high_risk_threshold = merged_df['high_risk_pct'].quantile(0.75)
    low_performance_threshold = merged_df['semantic_auroc'].quantile(0.25)
    
    problem_classes = merged_df[
        (merged_df['high_risk_pct'] >= high_risk_threshold) & 
        (merged_df['semantic_auroc'] <= low_performance_threshold)
    ]
    
    if len(problem_classes) > 0:
        print("高风险 + 低性能类别:")
        for _, row in problem_classes.iterrows():
            print(f"  - {row['class']}: {row['high_risk_pct']:.1f}% 高风险, Semantic AUROC={row['semantic_auroc']:.2f}")
    else:
        print("未发现明显的高风险+低性能类别")
    
    # 5. 优秀案例识别
    print("\n✅ 优秀案例识别")
    print("-"*80)
    
    low_risk_threshold = merged_df['high_risk_pct'].quantile(0.25)
    high_performance_threshold = merged_df['semantic_auroc'].quantile(0.75)
    
    good_classes = merged_df[
        (merged_df['high_risk_pct'] <= low_risk_threshold) & 
        (merged_df['semantic_auroc'] >= high_performance_threshold)
    ]
    
    if len(good_classes) > 0:
        print("低风险 + 高性能类别:")
        for _, row in good_classes.iterrows():
            print(f"  - {row['class']}: {row['high_risk_pct']:.1f}% 高风险, Semantic AUROC={row['semantic_auroc']:.2f}")
    else:
        print("未发现明显的低风险+高性能类别")
    
    print("\n" + "="*80)
    
    return merged_df


def save_detailed_analysis(dataset, merged_df, detail_df, output_dir='result/prompt_purging/analysis'):
    """保存详细分析结果"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存类别级别摘要
    summary_file = os.path.join(output_dir, f'{dataset}_class_summary.csv')
    merged_df.to_csv(summary_file, index=False)
    print(f"\n✓ 类别摘要已保存: {summary_file}")
    
    # 保存每条prompt的详细信息（合并性能数据）
    if detail_df is not None:
        detail_with_perf = pd.merge(
            detail_df, 
            merged_df[['class', 'semantic_auroc', 'memory_auroc', 'image_auroc']], 
            on='class', 
            how='left'
        )
        detail_file = os.path.join(output_dir, f'{dataset}_prompt_details.csv')
        detail_with_perf.to_csv(detail_file, index=False)
        print(f"✓ Prompt详情已保存: {detail_file}")


def compare_datasets(mvtec_merged, visa_merged):
    """比较MVTec和VisA数据集"""
    
    print("\n" + "="*80)
    print("跨数据集比较: MVTec vs VisA")
    print("="*80)
    
    print("\nPrompt风险指标对比:")
    print(f"{'Metric':<30} {'MVTec':>12} {'VisA':>12} {'Diff':>12}")
    print("-"*80)
    
    metrics = [
        ('high_risk_pct', '平均高风险比例 (%)'),
        ('mean_R_j_eps', '平均R_j_eps'),
        ('pct_negative_margin', '平均负margin比例 (%)'),
    ]
    
    for col, label in metrics:
        mvtec_val = mvtec_merged[col].mean()
        visa_val = visa_merged[col].mean()
        diff = mvtec_val - visa_val
        print(f"{label:<30} {mvtec_val:>12.2f} {visa_val:>12.2f} {diff:>12.2f}")
    
    print("\n性能指标对比:")
    print(f"{'Metric':<30} {'MVTec':>12} {'VisA':>12} {'Diff':>12}")
    print("-"*80)
    
    perf_metrics = [
        ('semantic_auroc', 'Semantic AUROC'),
        ('memory_auroc', 'Memory AUROC'),
        ('image_auroc', 'Image AUROC'),
    ]
    
    for col, label in perf_metrics:
        mvtec_val = mvtec_merged[col].mean()
        visa_val = visa_merged[col].mean()
        diff = mvtec_val - visa_val
        print(f"{label:<30} {mvtec_val:>12.2f} {visa_val:>12.2f} {diff:>12.2f}")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description='Phase 1 Comprehensive Analysis')
    parser.add_argument('--dataset', type=str, choices=['mvtec', 'visa', 'both'], default='both')
    parser.add_argument('--k_shot', type=int, default=2)
    parser.add_argument('--seed', type=int, default=111)
    parser.add_argument('--output_dir', type=str, default='result/prompt_purging/analysis')
    
    args = parser.parse_args()
    
    datasets = ['mvtec', 'visa'] if args.dataset == 'both' else [args.dataset]
    
    all_merged = {}
    
    for dataset in datasets:
        print(f"\n{'='*80}")
        print(f"处理数据集: {dataset.upper()}")
        print(f"{'='*80}\n")
        
        # 加载Phase 1结果
        phase1_summary, phase1_detail = load_phase1_results(dataset, args.k_shot)
        if phase1_summary is None:
            continue
        
        print(f"✓ 加载了 {len(phase1_summary)} 个类别的Phase 1结果")
        
        # 加载baseline结果
        baseline_results = load_baseline_results(dataset, args.k_shot, args.seed)
        if baseline_results is None:
            continue
        
        print(f"✓ 加载了 {len(baseline_results)} 个类别的Baseline结果")
        
        # 合并和分析
        merged_df, correlations = merge_and_analyze(phase1_summary, baseline_results)
        print(f"✓ 成功关联 {len(merged_df)} 个类别")
        
        # 打印分析报告
        merged_df = print_analysis_report(dataset, merged_df, correlations)
        
        # 保存详细结果
        save_detailed_analysis(dataset, merged_df, phase1_detail, args.output_dir)
        
        all_merged[dataset] = merged_df
    
    # 跨数据集比较
    if len(all_merged) == 2:
        compare_datasets(all_merged['mvtec'], all_merged['visa'])
    
    print("\n✅ 分析完成！")


if __name__ == '__main__':
    main()
