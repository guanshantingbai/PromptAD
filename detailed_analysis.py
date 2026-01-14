#!/usr/bin/env python3
"""
Lambda Sweep Detailed Analysis
深度分析原型融合策略的影响
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Load full results
df = pd.read_csv("result/lambda_sweep_full/full_results.csv")

print("="*80)
print("原型融合策略 - 详细分析报告")
print("="*80)
print()

# ============================================================================
# 1. 退化分析
# ============================================================================
print("━"*80)
print("【1】性能退化分析")
print("━"*80)

# Get baseline (lambda=0) and best performance for each class
baseline = df[df['lambda'] == 0.0].set_index(['dataset', 'class'])['semantic_i_roc']
best_perf = df.loc[df.groupby(['dataset', 'class'])['semantic_i_roc'].idxmax()]
best_perf_indexed = best_perf.set_index(['dataset', 'class'])

# Calculate improvement/degradation
improvement = best_perf_indexed['semantic_i_roc'] - baseline

# Count categories
total_classes = len(baseline)
degraded = (improvement < 0).sum()
no_change = (improvement == 0).sum()
improved = (improvement > 0).sum()

print(f"\n总类别数: {total_classes}")
print(f"  ✓ 有提升: {improved} 类 ({improved/total_classes*100:.1f}%)")
print(f"  - 无变化: {no_change} 类 ({no_change/total_classes*100:.1f}%)")
print(f"  ✗ 退化:   {degraded} 类 ({degraded/total_classes*100:.1f}%)")

# Show degraded classes
if degraded > 0:
    print(f"\n退化类别详情 ({degraded}个):")
    degraded_classes = improvement[improvement < 0].sort_values()
    print(f"{'数据集':<8} {'类别':<12} {'baseline':<10} {'最佳λ':<6} {'最佳AUROC':<10} {'退化幅度':<10}")
    print("-"*70)
    for (dataset, cls), deg in degraded_classes.items():
        base_auroc = baseline.loc[(dataset, cls)]
        best_lambda = best_perf_indexed.loc[(dataset, cls), 'lambda']
        best_auroc = best_perf_indexed.loc[(dataset, cls), 'semantic_i_roc']
        print(f"{dataset:<8} {cls:<12} {base_auroc:>8.2f}%  {best_lambda:>4.1f}  {best_auroc:>8.2f}%  {deg:>8.2f}%")

# ============================================================================
# 2. Semantic弱类别分析
# ============================================================================
print("\n" + "━"*80)
print("【2】Semantic弱类别（baseline<85%）提升分析")
print("━"*80)

# Define weak classes (semantic_i_roc < 85%)
weak_classes = baseline[baseline < 85.0]
print(f"\nSemantic弱类别数量: {len(weak_classes)} / {total_classes}")

if len(weak_classes) > 0:
    weak_improvement = improvement.loc[weak_classes.index]
    
    print(f"\n统计摘要:")
    print(f"  平均baseline:     {weak_classes.mean():.2f}%")
    print(f"  平均提升幅度:     {weak_improvement.mean():+.2f}%")
    print(f"  最大提升:         {weak_improvement.max():+.2f}%")
    print(f"  最小变化:         {weak_improvement.min():+.2f}%")
    print(f"  标准差:           {weak_improvement.std():.2f}%")
    
    # Count by improvement level
    weak_sig_improved = (weak_improvement > 1.0).sum()
    weak_minor_improved = ((weak_improvement > 0) & (weak_improvement <= 1.0)).sum()
    weak_no_change = (weak_improvement == 0).sum()
    weak_degraded = (weak_improvement < 0).sum()
    
    print(f"\n提升分布:")
    print(f"  显著提升(>1%):   {weak_sig_improved} 类")
    print(f"  轻微提升(0-1%):  {weak_minor_improved} 类")
    print(f"  无变化:          {weak_no_change} 类")
    print(f"  退化:            {weak_degraded} 类")
    
    # Show top improved weak classes
    print(f"\n弱类别Top提升:")
    weak_top = weak_improvement.sort_values(ascending=False).head(10)
    print(f"{'数据集':<8} {'类别':<12} {'baseline':<10} {'最佳λ':<6} {'最佳AUROC':<10} {'提升幅度':<10}")
    print("-"*70)
    for (dataset, cls), imp in weak_top.items():
        base_auroc = baseline.loc[(dataset, cls)]
        best_lambda = best_perf_indexed.loc[(dataset, cls), 'lambda']
        best_auroc = best_perf_indexed.loc[(dataset, cls), 'semantic_i_roc']
        print(f"{dataset:<8} {cls:<12} {base_auroc:>8.2f}%  {best_lambda:>4.1f}  {best_auroc:>8.2f}%  {imp:>+8.2f}%")

# ============================================================================
# 3. 全局最优Lambda分析
# ============================================================================
print("\n" + "━"*80)
print("【3】全局最优Lambda分析")
print("━"*80)

# Calculate mean performance for each lambda across all classes
lambda_avg_perf = df.groupby('lambda')['semantic_i_roc'].agg(['mean', 'std', 'min', 'max'])

print(f"\n各Lambda值的平均性能:")
print(f"{'Lambda':<8} {'平均AUROC':<12} {'标准差':<10} {'最小值':<10} {'最大值':<10} {'vs baseline':<12}")
print("-"*70)
baseline_mean = lambda_avg_perf.loc[0.0, 'mean']
for lambda_val in sorted(df['lambda'].unique()):
    row = lambda_avg_perf.loc[lambda_val]
    delta = row['mean'] - baseline_mean
    print(f"{lambda_val:<8.1f} {row['mean']:>10.2f}%  {row['std']:>8.2f}%  {row['min']:>8.2f}%  {row['max']:>8.2f}%  {delta:>+10.2f}%")

# Find global optimal lambda
best_global_lambda = lambda_avg_perf['mean'].idxmax()
best_global_mean = lambda_avg_perf.loc[best_global_lambda, 'mean']
global_improvement = best_global_mean - baseline_mean

print(f"\n🏆 全局最优Lambda: {best_global_lambda}")
print(f"   平均AUROC: {best_global_mean:.2f}%")
print(f"   vs baseline: {global_improvement:+.2f}%")

# Analyze impact of choosing global optimal lambda
print(f"\n若所有类别使用λ={best_global_lambda}:")
global_lambda_results = df[df['lambda'] == best_global_lambda].set_index(['dataset', 'class'])
global_improvements = global_lambda_results['semantic_i_roc'] - baseline

print(f"  ✓ 有提升的类别: {(global_improvements > 0).sum()} / {total_classes}")
print(f"  ✗ 退化的类别:   {(global_improvements < 0).sum()} / {total_classes}")
print(f"  平均提升:       {global_improvements.mean():+.2f}%")
print(f"  最大提升:       {global_improvements.max():+.2f}%")
print(f"  最大退化:       {global_improvements.min():+.2f}%")

# Compare global lambda vs adaptive lambda
adaptive_improvements = improvement  # This uses best lambda per class
print(f"\n📊 策略对比:")
print(f"{'策略':<20} {'平均提升':<12} {'正提升类别':<15} {'最大提升':<12}")
print("-"*60)
print(f"{'全局λ=' + str(best_global_lambda):<20} {global_improvements.mean():>+10.2f}%  {(global_improvements > 0).sum():>6} / {total_classes:<5} {global_improvements.max():>+10.2f}%")
print(f"{'自适应λ(逐类最优)':<20} {adaptive_improvements.mean():>+10.2f}%  {(adaptive_improvements > 0).sum():>6} / {total_classes:<5} {adaptive_improvements.max():>+10.2f}%")

adaptive_gain = adaptive_improvements.mean() - global_improvements.mean()
print(f"\n自适应策略额外收益: {adaptive_gain:+.2f}%")

# ============================================================================
# 4. 数据集差异分析
# ============================================================================
print("\n" + "━"*80)
print("【4】数据集差异分析")
print("━"*80)

for dataset in df['dataset'].unique():
    dataset_df = df[df['dataset'] == dataset]
    dataset_baseline = baseline[baseline.index.get_level_values('dataset') == dataset]
    dataset_improvement = improvement[improvement.index.get_level_values('dataset') == dataset]
    
    # Find best lambda for this dataset
    dataset_lambda_avg = dataset_df.groupby('lambda')['semantic_i_roc'].mean()
    best_dataset_lambda = dataset_lambda_avg.idxmax()
    best_dataset_mean = dataset_lambda_avg.max()
    dataset_baseline_mean = dataset_lambda_avg.loc[0.0]
    dataset_global_improvement = best_dataset_mean - dataset_baseline_mean
    
    print(f"\n{dataset.upper()}:")
    print(f"  类别数:           {len(dataset_baseline)}")
    print(f"  Baseline平均:     {dataset_baseline_mean:.2f}%")
    print(f"  最优λ:            {best_dataset_lambda}")
    print(f"  最优平均性能:     {best_dataset_mean:.2f}%")
    print(f"  全局提升:         {dataset_global_improvement:+.2f}%")
    print(f"  自适应策略提升:   {dataset_improvement.mean():+.2f}%")
    print(f"  有提升类别:       {(dataset_improvement > 0).sum()} / {len(dataset_baseline)}")

# ============================================================================
# 5. 相似度与性能关系深度分析
# ============================================================================
print("\n" + "━"*80)
print("【5】相似度分段分析")
print("━"*80)

# Segment by baseline similarity
baseline_sim = df[df['lambda'] == 0.0].set_index(['dataset', 'class'])['sim_before']

# Define similarity bins
bins = [0, 0.4, 0.6, 0.8, 1.0]
labels = ['极低(<0.4)', '低(0.4-0.6)', '中(0.6-0.8)', '高(≥0.8)']

sim_category = pd.cut(baseline_sim, bins=bins, labels=labels, include_lowest=True)

print(f"\n按baseline相似度分段:")
print(f"{'相似度区间':<15} {'类别数':<10} {'平均baseline':<15} {'平均提升':<12} {'提升>0类别':<15}")
print("-"*70)

for label in labels:
    mask = sim_category == label
    if mask.sum() > 0:
        seg_classes = baseline_sim[mask].index
        seg_baseline = baseline.loc[seg_classes]
        seg_improvement = improvement.loc[seg_classes]
        
        print(f"{label:<15} {mask.sum():<10} {seg_baseline.mean():>13.2f}%  {seg_improvement.mean():>+10.2f}%  {(seg_improvement > 0).sum():>5} / {mask.sum():<5}")

print("\n" + "="*80)
print("分析完成！")
print("="*80)
