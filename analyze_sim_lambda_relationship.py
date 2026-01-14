#!/usr/bin/env python3
"""
分析baseline相似度(sim_before)与最佳lambda的关系
目标：建立基于先验相似度选择lambda的规则，避免heuristic
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("result/lambda_sweep_full/full_results.csv")

print("="*80)
print("Baseline相似度(sim_before) vs 最佳Lambda - 关系分析")
print("="*80)
print()

# Get best lambda for each class
best_lambda_per_class = df.loc[df.groupby(['dataset', 'class'])['semantic_i_roc'].idxmax()]
baseline = df[df['lambda'] == 0.0].set_index(['dataset', 'class'])

# Merge to get baseline similarity
results = pd.DataFrame({
    'dataset': best_lambda_per_class['dataset'].values,
    'class': best_lambda_per_class['class'].values,
    'baseline_sim': baseline['sim_before'].values,
    'best_lambda': best_lambda_per_class['lambda'].values,
    'baseline_auroc': baseline['semantic_i_roc'].values,
    'best_auroc': best_lambda_per_class['semantic_i_roc'].values,
    'improvement': best_lambda_per_class['semantic_i_roc'].values - baseline['semantic_i_roc'].values
})

print("━"*80)
print("【1】整体相关性分析")
print("━"*80)

correlation = results['baseline_sim'].corr(results['best_lambda'])
print(f"\nCorr(baseline_sim, best_lambda) = {correlation:.4f}")

if abs(correlation) < 0.3:
    print("  → 相关性弱，关系不显著")
elif correlation < 0:
    print("  → 负相关：相似度越低，需要越大的λ")
else:
    print("  → 正相关：相似度越高，需要越大的λ")

# ============================================================================
# 【2】分组分析：λ=0最佳 vs λ>0最佳
# ============================================================================
print("\n" + "━"*80)
print("【2】关键分组：λ=0最佳 vs λ>0最佳")
print("━"*80)

lambda_0_classes = results[results['best_lambda'] == 0.0]
lambda_pos_classes = results[results['best_lambda'] > 0.0]

print(f"\nλ=0最佳的类别 (n={len(lambda_0_classes)}):")
print(f"  平均baseline_sim:    {lambda_0_classes['baseline_sim'].mean():.4f}")
print(f"  中位数baseline_sim:  {lambda_0_classes['baseline_sim'].median():.4f}")
print(f"  范围:                [{lambda_0_classes['baseline_sim'].min():.4f}, {lambda_0_classes['baseline_sim'].max():.4f}]")
print(f"  平均baseline_auroc:  {lambda_0_classes['baseline_auroc'].mean():.2f}%")

print(f"\nλ>0最佳的类别 (n={len(lambda_pos_classes)}):")
print(f"  平均baseline_sim:    {lambda_pos_classes['baseline_sim'].mean():.4f}")
print(f"  中位数baseline_sim:  {lambda_pos_classes['baseline_sim'].median():.4f}")
print(f"  范围:                [{lambda_pos_classes['baseline_sim'].min():.4f}, {lambda_pos_classes['baseline_sim'].max():.4f}]")
print(f"  平均baseline_auroc:  {lambda_pos_classes['baseline_auroc'].mean():.2f}%")

# Statistical test
from scipy import stats
t_stat, p_value = stats.ttest_ind(lambda_0_classes['baseline_sim'], lambda_pos_classes['baseline_sim'])
print(f"\nt检验: t={t_stat:.3f}, p={p_value:.4f}")
if p_value < 0.05:
    print("  → 两组相似度差异显著 ✓")
else:
    print("  → 两组相似度差异不显著 ✗")

# Find threshold
print(f"\n相似度阈值分析:")
threshold_candidates = np.arange(0.3, 0.9, 0.05)
best_threshold = None
best_accuracy = 0

for threshold in threshold_candidates:
    # Predict: sim < threshold → lambda > 0
    predicted_need_lambda = results['baseline_sim'] < threshold
    actual_need_lambda = results['best_lambda'] > 0
    accuracy = (predicted_need_lambda == actual_need_lambda).mean()
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_threshold = threshold

print(f"  最佳阈值: {best_threshold:.2f}")
print(f"  预测准确率: {best_accuracy*100:.1f}%")
print(f"  规则: baseline_sim < {best_threshold:.2f} → 建议λ>0")

# ============================================================================
# 【3】按baseline_sim分段，查看λ分布
# ============================================================================
print("\n" + "━"*80)
print("【3】baseline_sim分段下的λ分布")
print("━"*80)

bins = [0, 0.4, 0.6, 0.8, 1.0]
labels = ['极低(<0.4)', '低(0.4-0.6)', '中(0.6-0.8)', '高(≥0.8)']
results['sim_category'] = pd.cut(results['baseline_sim'], bins=bins, labels=labels, include_lowest=True)

print(f"\n{'相似度区间':<15} {'样本数':<8} {'λ=0':<8} {'λ∈(0,0.5]':<12} {'λ>0.5':<8} {'平均λ':<10} {'平均提升':<10}")
print("-"*80)

for label in labels:
    segment = results[results['sim_category'] == label]
    if len(segment) > 0:
        n = len(segment)
        lambda_0 = (segment['best_lambda'] == 0.0).sum()
        lambda_small = ((segment['best_lambda'] > 0) & (segment['best_lambda'] <= 0.5)).sum()
        lambda_large = (segment['best_lambda'] > 0.5).sum()
        avg_lambda = segment['best_lambda'].mean()
        avg_improvement = segment['improvement'].mean()
        
        print(f"{label:<15} {n:<8} {lambda_0:<8} {lambda_small:<12} {lambda_large:<8} {avg_lambda:<10.2f} {avg_improvement:>+8.2f}%")

# ============================================================================
# 【4】低baseline性能类别的特殊分析
# ============================================================================
print("\n" + "━"*80)
print("【4】低baseline性能类别(<85%)的λ选择")
print("━"*80)

weak_classes = results[results['baseline_auroc'] < 85.0]
print(f"\n低性能类别 (n={len(weak_classes)}):")
print(f"  平均baseline_sim:    {weak_classes['baseline_sim'].mean():.4f}")
print(f"  平均最佳λ:           {weak_classes['best_lambda'].mean():.2f}")
print(f"  中位数最佳λ:         {weak_classes['best_lambda'].median():.2f}")

# Check if sim and lambda are correlated for weak classes
if len(weak_classes) > 3:
    weak_corr = weak_classes['baseline_sim'].corr(weak_classes['best_lambda'])
    print(f"  Corr(sim, λ):        {weak_corr:.4f}")

print(f"\n弱类别的λ分布:")
print(f"  λ=0:     {(weak_classes['best_lambda'] == 0.0).sum()} / {len(weak_classes)}")
print(f"  λ∈(0,0.5]: {((weak_classes['best_lambda'] > 0) & (weak_classes['best_lambda'] <= 0.5)).sum()} / {len(weak_classes)}")
print(f"  λ>0.5:   {(weak_classes['best_lambda'] > 0.5).sum()} / {len(weak_classes)}")

# ============================================================================
# 【5】建立先验规则
# ============================================================================
print("\n" + "="*80)
print("💡 基于先验相似度的λ选择规则（可解释、非heuristic）")
print("="*80)

# Rule based on similarity bins
print(f"\n规则A - 基于相似度分段（简单）:")
for label in labels:
    segment = results[results['sim_category'] == label]
    if len(segment) > 0:
        avg_lambda = segment['best_lambda'].mean()
        median_lambda = segment['best_lambda'].median()
        lambda_0_ratio = (segment['best_lambda'] == 0.0).sum() / len(segment)
        
        if lambda_0_ratio > 0.5:
            suggestion = f"λ=0 (基线已优，{lambda_0_ratio*100:.0f}%类别无需融合)"
        else:
            suggestion = f"λ≈{median_lambda:.1f} (中位值，{100-lambda_0_ratio*100:.0f}%类别受益)"
        
        print(f"  {label:<15} → {suggestion}")

# Rule based on formula
print(f"\n规则B - 基于线性公式（连续）:")

# Fit linear model: lambda = a * sim + b (只对lambda>0的类别)
lambda_pos_for_fit = lambda_pos_classes.copy()
if len(lambda_pos_for_fit) > 2:
    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(
        lambda_pos_for_fit['baseline_sim'], 
        lambda_pos_for_fit['best_lambda']
    )
    
    print(f"  公式: λ = {slope:.3f} × sim_before + {intercept:.3f}")
    print(f"  R² = {r_value**2:.3f}, p = {p_value:.4f}")
    
    if p_value > 0.05:
        print(f"  ⚠️  线性关系不显著，建议使用分段规则")
    else:
        print(f"  ✓ 线性关系显著")
        print(f"\n  示例:")
        for sim_val in [0.3, 0.5, 0.7, 0.9]:
            predicted_lambda = max(0, min(1.0, slope * sim_val + intercept))
            print(f"    sim_before = {sim_val:.1f} → 建议 λ ≈ {predicted_lambda:.2f}")

# Rule based on threshold
print(f"\n规则C - 基于阈值（二分类，最简单）:")
print(f"  IF baseline_sim < {best_threshold:.2f}:")
print(f"      建议 λ ∈ [0.1, 0.5]")
print(f"  ELSE:")
print(f"      建议 λ = 0 (无融合)")
print(f"  准确率: {best_accuracy*100:.1f}%")

# ============================================================================
# 【6】详细案例表
# ============================================================================
print("\n" + "━"*80)
print("【6】所有类别的baseline_sim与最佳λ对照表")
print("━"*80)

results_sorted = results.sort_values('baseline_sim')
print(f"\n{'数据集':<8} {'类别':<12} {'baseline_sim':<13} {'最佳λ':<8} {'提升':<10} {'baseline_AUROC':<15}")
print("-"*75)
for _, row in results_sorted.iterrows():
    print(f"{row['dataset']:<8} {row['class']:<12} {row['baseline_sim']:>12.4f}  {row['best_lambda']:>6.1f}  {row['improvement']:>+8.2f}%  {row['baseline_auroc']:>13.2f}%")

print("\n" + "="*80)
print("分析完成！")
print("="*80)

# Save results
results_sorted.to_csv("result/lambda_sweep_full/sim_lambda_relationship.csv", index=False)
print("\n保存至: result/lambda_sweep_full/sim_lambda_relationship.csv")
