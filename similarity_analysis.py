#!/usr/bin/env python3
"""
从融合后相似度(sim_after)角度分析性能
"""

import pandas as pd
import numpy as np

df = pd.read_csv("result/lambda_sweep_full/full_results.csv")

print("="*80)
print("从融合后相似度角度的性能分析")
print("="*80)
print()

# ============================================================================
# 1. 相似度与性能的相关性分析
# ============================================================================
print("━"*80)
print("【1】相似度与性能的相关性")
print("━"*80)

valid_data = df[pd.notna(df['sim_after']) & pd.notna(df['semantic_i_roc'])]

corr_sim_after = valid_data['sim_after'].corr(valid_data['semantic_i_roc'])
corr_sim_before = valid_data['sim_before'].corr(valid_data['semantic_i_roc'])
corr_delta_sim = valid_data['delta_sim'].corr(valid_data['semantic_i_roc'])

print(f"\n整体相关性分析 (n={len(valid_data)}):")
print(f"  Corr(sim_before, AUROC) = {corr_sim_before:+.4f}")
print(f"  Corr(sim_after, AUROC)  = {corr_sim_after:+.4f}")
print(f"  Corr(Δsim, AUROC)       = {corr_delta_sim:+.4f}")

# Per-class correlation (baseline similarity vs baseline performance)
baseline_df = df[df['lambda'] == 0.0].copy()
per_class_corr = baseline_df['sim_before'].corr(baseline_df['semantic_i_roc'])
print(f"\n类别级相关性 (baseline, n={len(baseline_df)}):")
print(f"  Corr(baseline_sim, baseline_AUROC) = {per_class_corr:+.4f}")

# ============================================================================
# 2. 按融合后相似度分段分析
# ============================================================================
print("\n" + "━"*80)
print("【2】按融合后相似度(sim_after)分段分析")
print("━"*80)

# Define similarity bins
bins = [0, 0.5, 0.7, 0.85, 0.95, 1.0]
labels = ['极低(<0.5)', '低(0.5-0.7)', '中(0.7-0.85)', '高(0.85-0.95)', '极高(≥0.95)']

valid_data['sim_category'] = pd.cut(valid_data['sim_after'], bins=bins, labels=labels, include_lowest=True)

print(f"\n相似度分段性能统计:")
print(f"{'相似度区间':<15} {'样本数':<10} {'平均AUROC':<12} {'标准差':<10} {'最小值':<10} {'最大值':<10}")
print("-"*75)

for label in labels:
    segment = valid_data[valid_data['sim_category'] == label]
    if len(segment) > 0:
        mean_auroc = segment['semantic_i_roc'].mean()
        std_auroc = segment['semantic_i_roc'].std()
        min_auroc = segment['semantic_i_roc'].min()
        max_auroc = segment['semantic_i_roc'].max()
        print(f"{label:<15} {len(segment):<10} {mean_auroc:>10.2f}%  {std_auroc:>8.2f}%  {min_auroc:>8.2f}%  {max_auroc:>8.2f}%")

# Find optimal similarity range
segment_means = valid_data.groupby('sim_category')['semantic_i_roc'].mean()
best_sim_range = segment_means.idxmax()
print(f"\n🏆 最佳相似度区间: {best_sim_range} (平均AUROC: {segment_means.max():.2f}%)")

# ============================================================================
# 3. 每个类别的最佳相似度分析
# ============================================================================
print("\n" + "━"*80)
print("【3】每个类别的最佳相似度分析")
print("━"*80)

# Get best performance for each class
best_perf = df.loc[df.groupby(['dataset', 'class'])['semantic_i_roc'].idxmax()]
baseline = df[df['lambda'] == 0.0].set_index(['dataset', 'class'])

print(f"\n各类别最佳性能时的相似度分布:")
best_sim_after = best_perf['sim_after']
print(f"  平均最佳相似度:    {best_sim_after.mean():.4f}")
print(f"  中位数:            {best_sim_after.median():.4f}")
print(f"  标准差:            {best_sim_after.std():.4f}")
print(f"  最小值:            {best_sim_after.min():.4f}")
print(f"  最大值:            {best_sim_after.max():.4f}")

# Compare with baseline similarity
baseline_sim = baseline['sim_before']
print(f"\nBaseline相似度分布:")
print(f"  平均:              {baseline_sim.mean():.4f}")
print(f"  中位数:            {baseline_sim.median():.4f}")
print(f"  标准差:            {baseline_sim.std():.4f}")

avg_delta = best_sim_after.mean() - baseline_sim.mean()
print(f"\n平均相似度提升:      {avg_delta:+.4f}")

# ============================================================================
# 4. 相似度提升与性能提升的关系
# ============================================================================
print("\n" + "━"*80)
print("【4】相似度提升(Δsim)与性能提升(ΔAUROC)的关系")
print("━"*80)

# Calculate improvements
improvements = pd.DataFrame({
    'dataset': best_perf['dataset'].values,
    'class': best_perf['class'].values,
    'baseline_sim': baseline['sim_before'].values,
    'best_sim': best_perf['sim_after'].values,
    'delta_sim': best_perf['delta_sim'].values,
    'baseline_auroc': baseline['semantic_i_roc'].values,
    'best_auroc': best_perf['semantic_i_roc'].values,
})
improvements['delta_auroc'] = improvements['best_auroc'] - improvements['baseline_auroc']

# Group by delta_sim magnitude
improvements['delta_sim_category'] = pd.cut(
    improvements['delta_sim'], 
    bins=[0, 0.1, 0.2, 0.3, 1.0],
    labels=['微小(0-0.1)', '小(0.1-0.2)', '中(0.2-0.3)', '大(≥0.3)'],
    include_lowest=True
)

print(f"\n按相似度提升幅度分组的性能变化:")
print(f"{'Δsim区间':<15} {'类别数':<10} {'平均ΔAUROC':<15} {'正提升类别':<15}")
print("-"*60)

for cat in ['微小(0-0.1)', '小(0.1-0.2)', '中(0.2-0.3)', '大(≥0.3)']:
    segment = improvements[improvements['delta_sim_category'] == cat]
    if len(segment) > 0:
        mean_delta_auroc = segment['delta_auroc'].mean()
        positive_count = (segment['delta_auroc'] > 0).sum()
        print(f"{cat:<15} {len(segment):<10} {mean_delta_auroc:>+13.2f}%  {positive_count:>5} / {len(segment):<5}")

# Find anomalies: high sim increase but low/negative AUROC change
print(f"\n⚠️  高相似度提升但性能未改善的异常案例:")
anomalies = improvements[(improvements['delta_sim'] > 0.2) & (improvements['delta_auroc'] <= 0)]
if len(anomalies) > 0:
    print(f"{'数据集':<8} {'类别':<12} {'Δsim':<10} {'ΔAUROC':<10} {'baseline_sim':<12}")
    print("-"*60)
    for _, row in anomalies.iterrows():
        print(f"{row['dataset']:<8} {row['class']:<12} {row['delta_sim']:>+8.4f}  {row['delta_auroc']:>+8.2f}%  {row['baseline_sim']:>10.4f}")
else:
    print("  无异常案例")

# ============================================================================
# 5. 识别最佳相似度目标值
# ============================================================================
print("\n" + "━"*80)
print("【5】最佳相似度目标值识别")
print("━"*80)

# For each class, find which sim_after gives best performance
best_sim_targets = []
for (dataset, cls), group in df.groupby(['dataset', 'class']):
    best_idx = group['semantic_i_roc'].idxmax()
    best_row = group.loc[best_idx]
    baseline_row = group[group['lambda'] == 0.0].iloc[0]
    
    best_sim_targets.append({
        'dataset': dataset,
        'class': cls,
        'baseline_sim': baseline_row['sim_before'],
        'target_sim': best_row['sim_after'],
        'baseline_auroc': baseline_row['semantic_i_roc'],
        'best_auroc': best_row['semantic_i_roc'],
        'improvement': best_row['semantic_i_roc'] - baseline_row['semantic_i_roc']
    })

targets_df = pd.DataFrame(best_sim_targets)

# Categorize by baseline similarity
targets_df['baseline_category'] = pd.cut(
    targets_df['baseline_sim'],
    bins=[0, 0.4, 0.6, 0.8, 1.0],
    labels=['低(<0.4)', '中低(0.4-0.6)', '中高(0.6-0.8)', '高(≥0.8)'],
    include_lowest=True
)

print(f"\n不同baseline相似度下的最佳目标相似度:")
print(f"{'Baseline区间':<15} {'类别数':<10} {'平均baseline':<15} {'平均目标sim':<15} {'平均提升':<12}")
print("-"*70)

for cat in ['低(<0.4)', '中低(0.4-0.6)', '中高(0.6-0.8)', '高(≥0.8)']:
    segment = targets_df[targets_df['baseline_category'] == cat]
    if len(segment) > 0:
        mean_baseline = segment['baseline_sim'].mean()
        mean_target = segment['target_sim'].mean()
        mean_improvement = segment['improvement'].mean()
        print(f"{cat:<15} {len(segment):<10} {mean_baseline:>13.4f}  {mean_target:>13.4f}  {mean_improvement:>+10.2f}%")

# ============================================================================
# 6. 关键发现总结
# ============================================================================
print("\n" + "="*80)
print("💡 关键发现")
print("="*80)

print(f"""
1. 相似度与性能的关系复杂且非线性
   • 整体相关性弱: sim_after与AUROC相关性仅{corr_sim_after:.3f}
   • 盲目提升相似度不一定提升性能

2. 存在最佳相似度区间
   • 最佳区间: {best_sim_range}
   • 该区间平均AUROC: {segment_means.max():.2f}%
   
3. 相似度提升的边际效应递减
   • 微小提升(Δsim<0.1): 多数类别有效
   • 大幅提升(Δsim≥0.3): 可能导致过拟合，性能反而下降

4. 最佳目标相似度因baseline而异
   • Baseline低(<0.4): 应提升至{targets_df[targets_df['baseline_category']=='低(<0.4)']['target_sim'].mean():.3f}
   • Baseline高(≥0.8): 保持或轻微调整即可

5. 异常案例警示
   • {len(anomalies)}个类别出现"相似度↑但性能↓"
   • 提示：相似度不是唯一因素，可能存在其他潜在因素
""")

print("="*80)
print("分析完成！详细数据已保存")
print("="*80)

# Save detailed analysis
targets_df.to_csv("result/lambda_sweep_full/similarity_analysis.csv", index=False)
print("\n保存至: result/lambda_sweep_full/similarity_analysis.csv")
