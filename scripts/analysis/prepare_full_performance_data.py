#!/usr/bin/env python3
"""
Step 1: 整理 Prompt2 vs Baseline 的全类别性能对比数据（k=2）
"""

import pandas as pd
import numpy as np

# 读取baseline数据
baseline_mvtec = pd.read_csv('result/baseline/mvtec/k_2/csv/Seed_111-results.csv', index_col=0)
baseline_visa = pd.read_csv('result/baseline/visa/k_2/csv/Seed_111-results.csv', index_col=0)
baseline = pd.concat([baseline_mvtec, baseline_visa])

# 读取prompt2数据
prompt2_mvtec = pd.read_csv('result/prompt2/mvtec/k_2/csv/Seed_111-results.csv', index_col=0)
prompt2_visa = pd.read_csv('result/prompt2/visa/k_2/csv/Seed_111-results.csv', index_col=0)
prompt2 = pd.concat([prompt2_mvtec, prompt2_visa])

# 合并数据
comparison = pd.DataFrame({
    'class': baseline.index,
    'baseline_acc': baseline['i_roc'],
    'prompt2_acc': prompt2['i_roc'],
})

# 计算delta
comparison['delta_acc'] = comparison['prompt2_acc'] - comparison['baseline_acc']

# 添加数据集标记
comparison['dataset'] = comparison['class'].apply(lambda x: x.split('-')[0])
comparison['class_name'] = comparison['class'].apply(lambda x: x.split('-')[1])

# 按delta_acc分组
def classify_performance(delta):
    if delta < -5:
        return 'Severe Degrade'
    elif delta < -2:
        return 'Mild Degrade'
    elif delta < 2:
        return 'Stable'
    else:
        return 'Improved'

comparison['performance_group'] = comparison['delta_acc'].apply(classify_performance)

# 排序
comparison = comparison.sort_values('delta_acc')

# 保存完整数据
comparison.to_csv('analysis/full_performance_comparison_k2.csv', index=False)

# 生成分组统计
print("="*80)
print("Prompt2 vs Baseline 全类别性能对比 (k=2)")
print("="*80)
print()

# 总体统计
print("📊 总体统计:")
print(f"  总类别数: {len(comparison)}")
print(f"  平均 Baseline: {comparison['baseline_acc'].mean():.2f}%")
print(f"  平均 Prompt2: {comparison['prompt2_acc'].mean():.2f}%")
print(f"  平均 ΔAcc: {comparison['delta_acc'].mean():.2f}%")
print()

# 按数据集统计
print("📊 按数据集统计:")
for dataset in ['mvtec', 'visa']:
    subset = comparison[comparison['dataset'] == dataset]
    print(f"  {dataset.upper()}:")
    print(f"    类别数: {len(subset)}")
    print(f"    平均 Baseline: {subset['baseline_acc'].mean():.2f}%")
    print(f"    平均 Prompt2: {subset['prompt2_acc'].mean():.2f}%")
    print(f"    平均 ΔAcc: {subset['delta_acc'].mean():.2f}%")
print()

# 按性能分组统计
print("📊 按性能变化分组:")
group_stats = comparison.groupby('performance_group').agg({
    'class': 'count',
    'baseline_acc': 'mean',
    'delta_acc': 'mean'
}).rename(columns={'class': 'count'})

# 按分组顺序排列
group_order = ['Severe Degrade', 'Mild Degrade', 'Stable', 'Improved']
for group in group_order:
    if group in group_stats.index:
        stats = group_stats.loc[group]
        print(f"  {group}:")
        print(f"    类别数: {int(stats['count'])}")
        print(f"    平均 Baseline: {stats['baseline_acc']:.2f}%")
        print(f"    平均 ΔAcc: {stats['delta_acc']:.2f}%")
print()

# Top-5 改进和退化类别
print("="*80)
print("🔝 Top-5 改进类别:")
print("="*80)
top_improved = comparison.nlargest(5, 'delta_acc')[['class', 'baseline_acc', 'prompt2_acc', 'delta_acc']]
for idx, row in top_improved.iterrows():
    print(f"  {row['class']:<25} {row['baseline_acc']:>6.2f}% → {row['prompt2_acc']:>6.2f}%  Δ={row['delta_acc']:>6.2f}%")
print()

print("="*80)
print("💔 Top-5 退化类别:")
print("="*80)
top_degraded = comparison.nsmallest(5, 'delta_acc')[['class', 'baseline_acc', 'prompt2_acc', 'delta_acc']]
for idx, row in top_degraded.iterrows():
    print(f"  {row['class']:<25} {row['baseline_acc']:>6.2f}% → {row['prompt2_acc']:>6.2f}%  Δ={row['delta_acc']:>6.2f}%")
print()

# 按baseline强度分层
print("="*80)
print("📊 按 Baseline 强度分层:")
print("="*80)

def classify_baseline(acc):
    if acc >= 95:
        return 'Strong (≥95%)'
    elif acc >= 85:
        return 'Medium (85-95%)'
    else:
        return 'Weak (<85%)'

comparison['baseline_strength'] = comparison['baseline_acc'].apply(classify_baseline)

strength_stats = comparison.groupby('baseline_strength').agg({
    'class': 'count',
    'baseline_acc': 'mean',
    'delta_acc': 'mean'
}).rename(columns={'class': 'count'})

# 按强度顺序排列
strength_order = ['Strong (≥95%)', 'Medium (85-95%)', 'Weak (<85%)']
for strength in strength_order:
    if strength in strength_stats.index:
        stats = strength_stats.loc[strength]
        print(f"  {strength}:")
        print(f"    类别数: {int(stats['count'])}")
        print(f"    平均 Baseline: {stats['baseline_acc']:.2f}%")
        print(f"    平均 ΔAcc: {stats['delta_acc']:.2f}%")
print()

# 生成完整表格
print("="*80)
print("完整类别列表 (按 ΔAcc 排序):")
print("="*80)
print()
print(comparison[['class', 'baseline_acc', 'prompt2_acc', 'delta_acc', 'performance_group', 'baseline_strength']].to_string(index=False))
print()

print("="*80)
print(f"✅ 数据已保存到: analysis/full_performance_comparison_k2.csv")
print("="*80)
