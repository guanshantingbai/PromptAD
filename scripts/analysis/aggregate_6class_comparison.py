#!/usr/bin/env python3
"""
6类对照实验汇总分析
对比 Baseline vs Prompt2 vs Ours

输出：
1. 性能对比表（AUROC）
2. Margin/Separation对比表
3. Collapse指标对比
4. 定性结论
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# 设置显示
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)


# 定义6个代表性类别
CLASS_INFO = {
    'mvtec-toothbrush': 'Severe',
    'mvtec-capsule': 'Severe',
    'visa-pcb2': 'Severe',
    'mvtec-carpet': 'Stable',
    'mvtec-leather': 'Stable',
    'mvtec-screw': 'Improved',
}


def load_split_auroc(class_key, version):
    """加载拆分AUROC结果"""
    dataset, cls = class_key.split('-')
    file_path = f'analysis/6class_comparison/{dataset}_{cls}_{version}_split_auroc.csv'
    
    if not Path(file_path).exists():
        return None
    
    df = pd.read_csv(file_path)
    return {
        'overall_semantic': df['overall_semantic_auroc'].values[0],
        'overall_fusion': df['overall_fusion_auroc'].values[0],
        'normal_semantic': df['normal_semantic_auroc'].values[0],
        'abnormal_semantic': df['abnormal_semantic_auroc'].values[0],
    }


def load_margin_stats(class_key, version):
    """加载Margin统计"""
    dataset, cls = class_key.split('-')
    file_path = f'analysis/6class_comparison/{dataset}_{cls}_{version}_margin_stats.csv'
    
    if not Path(file_path).exists():
        return None
    
    df = pd.read_csv(file_path)
    
    # 提取normal和abnormal组的统计
    normal_row = df[df['group'] == 'normal']
    abnormal_row = df[df['group'] == 'abnormal']
    
    if len(normal_row) == 0 or len(abnormal_row) == 0:
        return None
    
    return {
        'normal_margin_mean': normal_row['mean'].values[0],
        'normal_margin_p10': normal_row['p10'].values[0],
        'abnormal_margin_mean': abnormal_row['mean'].values[0],
        'abnormal_margin_p90': abnormal_row['p90'].values[0],
        'separation': normal_row['mean'].values[0] - abnormal_row['mean'].values[0],
    }


def calculate_collapse_from_samples(class_key, version):
    """从样本分数计算collapse指标（近似）"""
    dataset, cls = class_key.split('-')
    file_path = f'analysis/6class_comparison/{dataset}_{cls}_{version}_sample_scores.csv'
    
    if not Path(file_path).exists():
        return None
    
    # 这里简化处理：从样本分数的方差推测collapse程度
    # 实际collapse需要从原型相似度计算，这里用样本分数方差作为代理
    df = pd.read_csv(file_path)
    
    # Collapse代理：semantic_score的标准差（低方差表示高collapse）
    semantic_std = df['semantic_score'].std()
    
    return {
        'semantic_score_std': semantic_std,
    }


def aggregate_comparison():
    """汇总三版本对比"""
    print("="*80)
    print("6类代表性类别 - 三版本对比汇总")
    print("="*80)
    print()
    
    results = []
    
    for class_key, group in CLASS_INFO.items():
        print(f"处理 {class_key} ({group})...")
        
        # 加载三个版本的数据
        baseline_auroc = load_split_auroc(class_key, 'baseline')
        prompt2_auroc = load_split_auroc(class_key, 'prompt2')
        ours_auroc = load_split_auroc(class_key, 'ours')
        
        baseline_margin = load_margin_stats(class_key, 'baseline')
        prompt2_margin = load_margin_stats(class_key, 'prompt2')
        ours_margin = load_margin_stats(class_key, 'ours')
        
        baseline_collapse = calculate_collapse_from_samples(class_key, 'baseline')
        prompt2_collapse = calculate_collapse_from_samples(class_key, 'prompt2')
        ours_collapse = calculate_collapse_from_samples(class_key, 'ours')
        
        # 构造结果行
        result = {
            'class': class_key,
            'group': group,
        }
        
        # AUROC对比
        if baseline_auroc and prompt2_auroc and ours_auroc:
            result['baseline_auroc'] = baseline_auroc['overall_semantic']
            result['prompt2_auroc'] = prompt2_auroc['overall_semantic']
            result['ours_auroc'] = ours_auroc['overall_semantic']
            result['delta_prompt2'] = prompt2_auroc['overall_semantic'] - baseline_auroc['overall_semantic']
            result['delta_ours'] = ours_auroc['overall_semantic'] - baseline_auroc['overall_semantic']
            result['improvement_vs_prompt2'] = ours_auroc['overall_semantic'] - prompt2_auroc['overall_semantic']
        
        # Margin对比
        if baseline_margin and prompt2_margin and ours_margin:
            result['baseline_separation'] = baseline_margin['separation']
            result['prompt2_separation'] = prompt2_margin['separation']
            result['ours_separation'] = ours_margin['separation']
            result['separation_change'] = ours_margin['separation'] - prompt2_margin['separation']
            
            result['baseline_normal_margin'] = baseline_margin['normal_margin_mean']
            result['prompt2_normal_margin'] = prompt2_margin['normal_margin_mean']
            result['ours_normal_margin'] = ours_margin['normal_margin_mean']
        
        # Collapse对比（代理指标）
        if baseline_collapse and prompt2_collapse and ours_collapse:
            result['baseline_semantic_std'] = baseline_collapse['semantic_score_std']
            result['prompt2_semantic_std'] = prompt2_collapse['semantic_score_std']
            result['ours_semantic_std'] = ours_collapse['semantic_score_std']
        
        results.append(result)
    
    # 转为DataFrame
    df = pd.DataFrame(results)
    
    # 保存详细对比表
    output_path = 'analysis/6class_comparison/comparison_summary.csv'
    df.to_csv(output_path, index=False, float_format='%.4f')
    print(f"\n✅ 详细对比表已保存: {output_path}")
    
    return df


def print_summary_analysis(df):
    """打印汇总分析"""
    print("\n" + "="*80)
    print("📊 关键指标汇总")
    print("="*80)
    
    # 按组统计
    for group in ['Severe', 'Stable', 'Improved']:
        group_df = df[df['group'] == group]
        if len(group_df) == 0:
            continue
        
        print(f"\n【{group}组】(n={len(group_df)})")
        
        # AUROC变化
        if 'delta_prompt2' in group_df.columns:
            print(f"  AUROC变化:")
            print(f"    Baseline→Prompt2: {group_df['delta_prompt2'].mean():+.4f} (平均)")
            print(f"    Baseline→Ours:    {group_df['delta_ours'].mean():+.4f} (平均)")
            print(f"    Prompt2→Ours:     {group_df['improvement_vs_prompt2'].mean():+.4f} (平均)")
        
        # Separation变化
        if 'separation_change' in group_df.columns:
            print(f"  Separation变化:")
            print(f"    Prompt2: {group_df['prompt2_separation'].mean():.4f}")
            print(f"    Ours:    {group_df['ours_separation'].mean():.4f}")
            print(f"    变化:    {group_df['separation_change'].mean():+.4f}")
    
    # 整体统计
    print(f"\n【整体】(n={len(df)})")
    
    if 'improvement_vs_prompt2' in df.columns:
        improvement_count = (df['improvement_vs_prompt2'] > 0).sum()
        print(f"  Ours相对Prompt2:")
        print(f"    改善类别数: {improvement_count}/{len(df)}")
        print(f"    平均AUROC提升: {df['improvement_vs_prompt2'].mean():+.4f}")
        print(f"    最大提升: {df['improvement_vs_prompt2'].max():+.4f} ({df.loc[df['improvement_vs_prompt2'].idxmax(), 'class']})")
        print(f"    最大下降: {df['improvement_vs_prompt2'].min():+.4f} ({df.loc[df['improvement_vs_prompt2'].idxmin(), 'class']})")
    
    if 'separation_change' in df.columns:
        separation_improve = (df['separation_change'] > 0).sum()
        print(f"\n  Separation改善:")
        print(f"    改善类别数: {separation_improve}/{len(df)}")
        print(f"    平均变化: {df['separation_change'].mean():+.4f}")


def generate_qualitative_conclusions(df):
    """生成定性结论"""
    print("\n" + "="*80)
    print("💡 定性结论")
    print("="*80)
    
    # 1. Severe组退化是否缓解？
    severe_df = df[df['group'] == 'Severe']
    if len(severe_df) > 0 and 'improvement_vs_prompt2' in severe_df.columns:
        severe_improvement = severe_df['improvement_vs_prompt2'].mean()
        if severe_improvement > 0.02:
            print(f"\n✅ 证据充分: Severe组退化显著缓解")
            print(f"   平均AUROC提升: {severe_improvement:+.4f}")
            print(f"   → 三项改动对严重退化类别有效")
        elif severe_improvement > 0:
            print(f"\n⚖️ 趋势: Severe组略有改善")
            print(f"   平均AUROC提升: {severe_improvement:+.4f}")
        else:
            print(f"\n❌ Severe组未明显改善")
            print(f"   平均AUROC变化: {severe_improvement:+.4f}")
    
    # 2. Margin/Separation是否改善？
    if 'separation_change' in df.columns:
        avg_sep_change = df['separation_change'].mean()
        if avg_sep_change > 0.05:
            print(f"\n✅ 证据充分: Separation显著提升")
            print(f"   平均变化: {avg_sep_change:+.4f}")
            print(f"   → Margin loss有效扩大判别裕度")
        elif avg_sep_change > 0:
            print(f"\n⚖️ 趋势: Separation略有提升")
            print(f"   平均变化: {avg_sep_change:+.4f}")
    
    # 3. Screw是否保持改进？
    screw_df = df[df['class'] == 'mvtec-screw']
    if len(screw_df) > 0 and 'improvement_vs_prompt2' in screw_df.columns:
        screw_change = screw_df['improvement_vs_prompt2'].values[0]
        if screw_change >= -0.02:
            print(f"\n✅ Screw保持改进或轻微回退")
            print(f"   相对Prompt2变化: {screw_change:+.4f}")
            print(f"   → 改动未破坏困难类的提升")
        else:
            print(f"\n⚠️ Screw显著回退")
            print(f"   相对Prompt2变化: {screw_change:+.4f}")
    
    # 4. 主要改善在哪一侧？
    # 这需要从split AUROC数据推断，这里简化处理
    if 'prompt2_normal_margin' in df.columns and 'ours_normal_margin' in df.columns:
        normal_margin_change = (df['ours_normal_margin'] - df['prompt2_normal_margin']).mean()
        if normal_margin_change > 0.05:
            print(f"\n✅ 证据充分: 主要改善在Normal侧")
            print(f"   Normal margin平均提升: {normal_margin_change:+.4f}")
            print(f"   → 减少假阳性（正常样本被误判）")


def plot_comparison_charts(df):
    """生成对比图表"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. AUROC对比（三版本）
    ax = axes[0, 0]
    if 'baseline_auroc' in df.columns:
        x = np.arange(len(df))
        width = 0.25
        ax.bar(x - width, df['baseline_auroc'], width, label='Baseline', alpha=0.8)
        ax.bar(x, df['prompt2_auroc'], width, label='Prompt2', alpha=0.8)
        ax.bar(x + width, df['ours_auroc'], width, label='Ours', alpha=0.8)
        ax.set_xlabel('Class')
        ax.set_ylabel('AUROC')
        ax.set_title('AUROC Comparison (3 Versions)')
        ax.set_xticks(x)
        ax.set_xticklabels([c.split('-')[1] for c in df['class']], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    # 2. AUROC变化（Ours vs Prompt2）
    ax = axes[0, 1]
    if 'improvement_vs_prompt2' in df.columns:
        colors = ['green' if x > 0 else 'red' for x in df['improvement_vs_prompt2']]
        ax.barh(df['class'], df['improvement_vs_prompt2'], color=colors, alpha=0.7)
        ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
        ax.set_xlabel('AUROC Change (Ours - Prompt2)')
        ax.set_title('Performance Improvement')
        ax.grid(True, alpha=0.3, axis='x')
    
    # 3. Separation对比
    ax = axes[1, 0]
    if 'prompt2_separation' in df.columns:
        x = np.arange(len(df))
        width = 0.35
        ax.bar(x - width/2, df['prompt2_separation'], width, label='Prompt2', alpha=0.8)
        ax.bar(x + width/2, df['ours_separation'], width, label='Ours', alpha=0.8)
        ax.set_xlabel('Class')
        ax.set_ylabel('Separation (Normal - Abnormal Margin)')
        ax.set_title('Margin Separation Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([c.split('-')[1] for c in df['class']], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    # 4. 按组汇总
    ax = axes[1, 1]
    if 'improvement_vs_prompt2' in df.columns:
        group_means = df.groupby('group')['improvement_vs_prompt2'].mean()
        colors_map = {'Severe': 'red', 'Stable': 'green', 'Improved': 'blue'}
        colors = [colors_map.get(g, 'gray') for g in group_means.index]
        ax.bar(group_means.index, group_means.values, color=colors, alpha=0.7)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
        ax.set_ylabel('Avg AUROC Change (Ours - Prompt2)')
        ax.set_title('Improvement by Performance Group')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = 'analysis/6class_comparison/comparison_charts.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ 对比图表已保存: {output_path}")
    plt.close()


def main():
    # 1. 汇总数据
    df = aggregate_comparison()
    
    if len(df) == 0:
        print("\n❌ 未找到任何数据，请先运行 evaluate_6class_comparison.sh")
        return
    
    # 2. 打印汇总分析
    print_summary_analysis(df)
    
    # 3. 生成定性结论
    generate_qualitative_conclusions(df)
    
    # 4. 生成图表
    plot_comparison_charts(df)
    
    print("\n" + "="*80)
    print("✅ 6类对照实验汇总完成！")
    print("="*80)
    print("\n关键文件:")
    print("  - analysis/6class_comparison/comparison_summary.csv")
    print("  - analysis/6class_comparison/comparison_charts.png")


if __name__ == '__main__':
    main()
