#!/usr/bin/env python3
"""
汇总27个类别的扩展评估指标
生成统一的分析报告

输出：
1. 拆分AUROC汇总表
2. Margin分布统计汇总
3. Semantic贡献汇总
4. 定性结论文档
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def aggregate_split_auroc():
    """汇总拆分AUROC结果"""
    metrics_dir = Path('analysis/extended_metrics')
    all_files = list(metrics_dir.glob('*_split_auroc.csv'))
    
    if not all_files:
        print("⚠️  未找到拆分AUROC文件")
        return None
    
    dfs = []
    for f in all_files:
        df = pd.read_csv(f)
        dfs.append(df)
    
    combined = pd.concat(dfs, ignore_index=True)
    
    # 合并性能分组
    performance_data = pd.read_csv('analysis/full_performance_comparison_k2.csv')
    combined = combined.merge(
        performance_data[['class', 'delta_acc', 'performance_group', 'baseline_acc']],
        on='class',
        how='left'
    )
    
    output_path = 'analysis/extended_metrics/split_auroc_summary.csv'
    combined.to_csv(output_path, index=False)
    print(f"✅ 拆分AUROC汇总: {output_path}")
    
    return combined


def aggregate_margin_stats():
    """汇总Margin统计"""
    metrics_dir = Path('analysis/extended_metrics')
    all_files = list(metrics_dir.glob('*_margin_stats.csv'))
    
    if not all_files:
        print("⚠️  未找到Margin统计文件")
        return None
    
    dfs = []
    for f in all_files:
        df = pd.read_csv(f)
        dfs.append(df)
    
    combined = pd.concat(dfs, ignore_index=True)
    
    # 透视表：每个类别 × 组别
    pivot = combined.pivot_table(
        index='class',
        columns='group',
        values=['mean', 'std', 'median', 'p10', 'p90']
    )
    
    output_path = 'analysis/extended_metrics/margin_stats_summary.csv'
    combined.to_csv(output_path, index=False)
    print(f"✅ Margin统计汇总: {output_path}")
    
    return combined


def aggregate_semantic_contrib():
    """汇总Semantic贡献"""
    metrics_dir = Path('analysis/extended_metrics')
    all_files = list(metrics_dir.glob('*_semantic_contrib.csv'))
    
    if not all_files:
        print("⚠️  未找到Semantic贡献文件")
        return None
    
    dfs = []
    for f in all_files:
        df = pd.read_csv(f)
        dfs.append(df)
    
    combined = pd.concat(dfs, ignore_index=True)
    
    # 合并性能分组
    performance_data = pd.read_csv('analysis/full_performance_comparison_k2.csv')
    combined = combined.merge(
        performance_data[['class', 'delta_acc', 'performance_group', 'baseline_acc']],
        on='class',
        how='left'
    )
    
    output_path = 'analysis/extended_metrics/semantic_contrib_summary.csv'
    combined.to_csv(output_path, index=False)
    print(f"✅ Semantic贡献汇总: {output_path}")
    
    return combined


def analyze_split_auroc(df):
    """分析拆分AUROC结果"""
    print("\n" + "="*80)
    print("【任务1】拆分AUROC分析")
    print("="*80)
    
    # 整体统计
    print("\n📊 整体AUROC统计:")
    print(f"  Overall Semantic: {df['overall_semantic_auroc'].mean():.4f} ± {df['overall_semantic_auroc'].std():.4f}")
    print(f"  Overall Fusion:   {df['overall_fusion_auroc'].mean():.4f} ± {df['overall_fusion_auroc'].std():.4f}")
    
    print(f"\n  Normal-only Semantic: {df['normal_semantic_auroc'].mean():.4f} ± {df['normal_semantic_auroc'].std():.4f}")
    print(f"  Normal-only Fusion:   {df['normal_fusion_auroc'].mean():.4f} ± {df['normal_fusion_auroc'].std():.4f}")
    
    print(f"\n  Abnormal-only Semantic: {df['abnormal_semantic_auroc'].mean():.4f} ± {df['abnormal_semantic_auroc'].std():.4f}")
    print(f"  Abnormal-only Fusion:   {df['abnormal_fusion_auroc'].mean():.4f} ± {df['abnormal_fusion_auroc'].std():.4f}")
    
    # 按性能组统计
    print("\n📊 按性能组分层:")
    for group in ['Severe Degrade', 'Mild Degrade', 'Stable', 'Improved']:
        group_df = df[df['performance_group'] == group]
        if len(group_df) == 0:
            continue
        print(f"\n  {group} (n={len(group_df)}):")
        print(f"    Overall AUROC: {group_df['overall_semantic_auroc'].mean():.4f}")
        print(f"    Normal-only:   {group_df['normal_semantic_auroc'].mean():.4f}")
        print(f"    Abnormal-only: {group_df['abnormal_semantic_auroc'].mean():.4f}")
    
    # 定性结论
    print("\n💡 定性结论:")
    
    # 1. Normal vs Abnormal侧哪个更差？
    normal_mean = df['normal_semantic_auroc'].mean()
    abnormal_mean = df['abnormal_semantic_auroc'].mean()
    
    if normal_mean < abnormal_mean - 0.05:
        print(f"  ✅ 证据充分: Normal侧区分能力更差 ({normal_mean:.3f} < {abnormal_mean:.3f})")
        print(f"     → 假阳性问题严重（正常样本被误判为异常）")
    elif abnormal_mean < normal_mean - 0.05:
        print(f"  ✅ 证据充分: Abnormal侧区分能力更差 ({abnormal_mean:.3f} < {normal_mean:.3f})")
        print(f"     → 召回不足问题严重（异常样本未被检出）")
    else:
        print(f"  ⚖️ 趋势不明显: Normal ({normal_mean:.3f}) 与 Abnormal ({abnormal_mean:.3f}) 相当")
    
    # 2. 退化类别的特征
    severe_df = df[df['performance_group'] == 'Severe Degrade']
    stable_df = df[df['performance_group'] == 'Stable']
    
    if len(severe_df) > 0 and len(stable_df) > 0:
        severe_normal = severe_df['normal_semantic_auroc'].mean()
        stable_normal = stable_df['normal_semantic_auroc'].mean()
        
        if severe_normal < stable_normal - 0.05:
            print(f"  ✅ 证据充分: Severe组Normal侧更差 ({severe_normal:.3f} < {stable_normal:.3f})")
            print(f"     → 退化主要来自正常样本被误判")


def analyze_margin_distribution(df):
    """分析Margin分布"""
    print("\n" + "="*80)
    print("【任务2】Margin分布分析")
    print("="*80)
    
    # 提取Normal和Abnormal组的统计
    normal_df = df[df['group'] == 'normal']
    abnormal_df = df[df['group'] == 'abnormal']
    
    print("\n📊 整体Margin统计:")
    print(f"  Normal样本:   均值={normal_df['mean'].mean():.4f}, 中位数={normal_df['median'].mean():.4f}")
    print(f"  Abnormal样本: 均值={abnormal_df['mean'].mean():.4f}, 中位数={abnormal_df['median'].mean():.4f}")
    print(f"  Separation:   {normal_df['mean'].mean() - abnormal_df['mean'].mean():.4f}")
    
    # 合并性能分组
    performance_data = pd.read_csv('analysis/full_performance_comparison_k2.csv')
    normal_df = normal_df.merge(
        performance_data[['class', 'performance_group']],
        on='class',
        how='left'
    )
    abnormal_df = abnormal_df.merge(
        performance_data[['class', 'performance_group']],
        on='class',
        how='left'
    )
    
    print("\n📊 按性能组分层:")
    for group in ['Severe Degrade', 'Mild Degrade', 'Stable', 'Improved']:
        normal_group = normal_df[normal_df['performance_group'] == group]
        abnormal_group = abnormal_df[abnormal_df['performance_group'] == group]
        
        if len(normal_group) == 0:
            continue
        
        print(f"\n  {group} (n={len(normal_group)}):")
        print(f"    Normal Margin:   {normal_group['mean'].mean():.4f} (P10={normal_group['p10'].mean():.4f})")
        print(f"    Abnormal Margin: {abnormal_group['mean'].mean():.4f} (P90={abnormal_group['p90'].mean():.4f})")
        print(f"    Separation:      {normal_group['mean'].mean() - abnormal_group['mean'].mean():.4f}")
    
    # 定性结论
    print("\n💡 定性结论:")
    
    # 1. Margin是否足够？
    normal_mean = normal_df['mean'].mean()
    abnormal_mean = abnormal_df['mean'].mean()
    separation = normal_mean - abnormal_mean
    
    if separation < 0.1:
        print(f"  ✅ 证据充分: Margin严重不足 (separation={separation:.4f})")
        print(f"     → Normal-Abnormal判别边界过窄")
    elif separation < 0.2:
        print(f"  ⚖️ 趋势明显: Margin偏低 (separation={separation:.4f})")
    
    # 2. 哪一侧Margin更小？
    normal_p10 = normal_df['p10'].mean()
    abnormal_p90 = abnormal_df['p90'].mean()
    
    if normal_p10 < 0:
        print(f"  ✅ 证据充分: Normal样本10%分位数为负 ({normal_p10:.4f})")
        print(f"     → 正常样本中有大量被误判为异常")
    
    if abnormal_p90 < 0:
        print(f"  ✅ 证据充分: Abnormal样本90%分位数为负 ({abnormal_p90:.4f})")
        print(f"     → 异常样本中多数更接近异常原型")
    
    # 3. Severe vs Stable的Margin差异
    severe_normal = normal_df[normal_df['performance_group'] == 'Severe Degrade']
    stable_normal = normal_df[normal_df['performance_group'] == 'Stable']
    
    if len(severe_normal) > 0 and len(stable_normal) > 0:
        severe_margin = severe_normal['mean'].mean()
        stable_margin = stable_normal['mean'].mean()
        
        if stable_margin > severe_margin + 0.05:
            print(f"  ✅ 证据充分: Stable组Normal Margin更大 ({stable_margin:.3f} > {severe_margin:.3f})")
            print(f"     → 稳定类别确实拥有更好的判别裕度")


def analyze_semantic_contribution(df):
    """分析Semantic贡献"""
    print("\n" + "="*80)
    print("【任务3】Semantic分支贡献分析")
    print("="*80)
    
    print("\n📊 整体相关性统计:")
    print(f"  Semantic-Fusion Pearson: {df['overall_pearson'].mean():.4f} ± {df['overall_pearson'].std():.4f}")
    print(f"    Normal侧:   {df['normal_pearson'].mean():.4f}")
    print(f"    Abnormal侧: {df['abnormal_pearson'].mean():.4f}")
    
    print(f"\n  Semantic-Visual 分支差异:")
    print(f"    Overall: {df['semantic_visual_diff_mean'].mean():.4f}")
    print(f"    Normal:  {df['semantic_visual_diff_normal'].mean():.4f}")
    print(f"    Abnormal: {df['semantic_visual_diff_abnormal'].mean():.4f}")
    
    # 按性能组统计
    print("\n📊 按性能组分层:")
    for group in ['Severe Degrade', 'Mild Degrade', 'Stable', 'Improved']:
        group_df = df[df['performance_group'] == group]
        if len(group_df) == 0:
            continue
        print(f"\n  {group} (n={len(group_df)}):")
        print(f"    Semantic-Fusion 相关性: {group_df['overall_pearson'].mean():.4f}")
        print(f"    Semantic-Visual 差异:   {group_df['semantic_visual_diff_mean'].mean():.4f}")
    
    # 定性结论
    print("\n💡 定性结论:")
    
    # 1. Semantic分支贡献强度
    overall_corr = df['overall_pearson'].mean()
    
    if overall_corr > 0.9:
        print(f"  ✅ 证据充分: Semantic主导Fusion (r={overall_corr:.3f})")
        print(f"     → Visual分支贡献较弱")
    elif overall_corr > 0.7:
        print(f"  ⚖️ 趋势明显: Semantic对Fusion贡献较大 (r={overall_corr:.3f})")
    else:
        print(f"  ⚠️  异常: Semantic与Fusion相关性偏低 (r={overall_corr:.3f})")
    
    # 2. Normal vs Abnormal侧差异
    normal_corr = df['normal_pearson'].mean()
    abnormal_corr = df['abnormal_pearson'].mean()
    
    if abs(normal_corr - abnormal_corr) > 0.1:
        if normal_corr > abnormal_corr:
            print(f"  ⚖️ 趋势: Normal侧Semantic贡献更稳定 ({normal_corr:.3f} > {abnormal_corr:.3f})")
        else:
            print(f"  ⚖️ 趋势: Abnormal侧Semantic贡献更稳定 ({abnormal_corr:.3f} > {normal_corr:.3f})")
    
    # 3. Semantic-Visual差异是否与退化相关
    severe_df = df[df['performance_group'] == 'Severe Degrade']
    stable_df = df[df['performance_group'] == 'Stable']
    
    if len(severe_df) > 0 and len(stable_df) > 0:
        severe_diff = severe_df['semantic_visual_diff_mean'].mean()
        stable_diff = stable_df['semantic_visual_diff_mean'].mean()
        
        if abs(severe_diff - stable_diff) > 0.05:
            print(f"  ⚖️ 趋势: Severe组分支差异{'更大' if severe_diff > stable_diff else '更小'} ({severe_diff:.3f} vs {stable_diff:.3f})")


def plot_extended_metrics_summary(split_auroc_df, margin_df, semantic_df):
    """可视化扩展指标汇总"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. 拆分AUROC对比
    ax = axes[0, 0]
    metrics = ['overall_semantic_auroc', 'normal_semantic_auroc', 'abnormal_semantic_auroc']
    means = [split_auroc_df[m].mean() for m in metrics]
    ax.bar(['Overall', 'Normal-only', 'Abnormal-only'], means, alpha=0.7)
    ax.set_ylabel('AUROC')
    ax.set_title('Split AUROC Comparison')
    ax.set_ylim([0.4, 1.0])
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    
    # 2. Margin分布箱线图
    ax = axes[0, 1]
    normal_margins = margin_df[margin_df['group'] == 'normal']['mean'].values
    abnormal_margins = margin_df[margin_df['group'] == 'abnormal']['mean'].values
    ax.boxplot([normal_margins, abnormal_margins], labels=['Normal', 'Abnormal'])
    ax.set_ylabel('Margin (max_normal - max_abnormal)')
    ax.set_title('Margin Distribution by Label')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax.grid(True, alpha=0.3)
    
    # 3. Semantic相关性分布
    ax = axes[0, 2]
    ax.hist(semantic_df['overall_pearson'].dropna(), bins=20, alpha=0.7, edgecolor='black')
    ax.axvline(x=semantic_df['overall_pearson'].mean(), color='red', linestyle='--', 
              label=f'Mean={semantic_df["overall_pearson"].mean():.3f}')
    ax.set_xlabel('Pearson Correlation')
    ax.set_ylabel('Frequency')
    ax.set_title('Semantic-Fusion Correlation Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. 按性能组的AUROC
    ax = axes[1, 0]
    for group in ['Severe Degrade', 'Mild Degrade', 'Stable']:
        group_df = split_auroc_df[split_auroc_df['performance_group'] == group]
        if len(group_df) > 0:
            ax.scatter([group]*len(group_df), group_df['overall_semantic_auroc'], 
                      alpha=0.6, s=50, label=f'{group} (n={len(group_df)})')
    ax.set_ylabel('Overall AUROC')
    ax.set_title('AUROC by Performance Group')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 5. Margin vs Delta_Acc
    ax = axes[1, 1]
    # 需要合并数据
    performance_data = pd.read_csv('analysis/full_performance_comparison_k2.csv')
    normal_margin_df = margin_df[margin_df['group'] == 'normal'].merge(
        performance_data[['class', 'delta_acc']], on='class', how='left'
    )
    ax.scatter(normal_margin_df['mean'], normal_margin_df['delta_acc'], alpha=0.6)
    ax.set_xlabel('Normal Margin')
    ax.set_ylabel('Delta Acc (%)')
    ax.set_title('Normal Margin vs Performance Change')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    
    # 6. Semantic贡献 vs Delta_Acc
    ax = axes[1, 2]
    ax.scatter(semantic_df['overall_pearson'], semantic_df['delta_acc'], alpha=0.6)
    ax.set_xlabel('Semantic-Fusion Correlation')
    ax.set_ylabel('Delta Acc (%)')
    ax.set_title('Semantic Contribution vs Performance Change')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = 'analysis/extended_metrics/extended_metrics_summary.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ 可视化汇总: {output_path}")
    plt.close()


def main():
    print("="*80)
    print("扩展评估指标汇总分析")
    print("="*80)
    
    # 1. 汇总数据
    print("\n📥 汇总数据文件...")
    split_auroc_df = aggregate_split_auroc()
    margin_df = aggregate_margin_stats()
    semantic_df = aggregate_semantic_contrib()
    
    if split_auroc_df is None or margin_df is None or semantic_df is None:
        print("\n❌ 数据文件不完整，请先运行 run_extended_evaluation.sh")
        return
    
    print(f"\n✅ 成功汇总 {len(split_auroc_df)} 个类别的数据")
    
    # 2. 分析
    analyze_split_auroc(split_auroc_df)
    analyze_margin_distribution(margin_df)
    analyze_semantic_contribution(semantic_df)
    
    # 3. 可视化
    print("\n📊 生成可视化...")
    plot_extended_metrics_summary(split_auroc_df, margin_df, semantic_df)
    
    # 4. 生成总结报告
    print("\n" + "="*80)
    print("✅ 扩展评估分析完成！")
    print("="*80)
    print("\n关键文件:")
    print("  - analysis/extended_metrics/split_auroc_summary.csv")
    print("  - analysis/extended_metrics/margin_stats_summary.csv")
    print("  - analysis/extended_metrics/semantic_contrib_summary.csv")
    print("  - analysis/extended_metrics/extended_metrics_summary.png")


if __name__ == '__main__':
    main()
