#!/usr/bin/env python3
"""
Step 3 + Step 4C: 完整的假设验证与分层分析
包含baseline_strength分层相关性分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

def analyze_correlation(df):
    """计算所有指标与delta_acc的相关性"""
    print("\n" + "="*80)
    print("1️⃣  相关性分析")
    print("="*80 + "\n")
    
    metrics = {
        'A_hit_mean': '异常max偶然命中(均值)',
        'A_hit_p95': '异常max偶然命中(P95)',
        'B_separation': '判别分离度',
        'B_overlap': '裕度重叠率',
        'C_collapse_score': '原型塌缩分数',
        'D_max_proto_count': '坏原型责任(最大计数)',
        'E_semantic_gap': '语义分数区分度'
    }
    
    results = []
    for metric, label in metrics.items():
        # 计算相关性
        pearson_r, pearson_p = stats.pearsonr(df[metric], df['delta_acc'])
        spearman_r, spearman_p = stats.spearmanr(df[metric], df['delta_acc'])
        
        # 显著性标记
        def sig_mark(p):
            if p < 0.001: return '***'
            elif p < 0.01: return '**'
            elif p < 0.05: return '*'
            else: return 'ns'
        
        print(f"{label}:")
        print(f"  Pearson  r={pearson_r:7.4f}, p={pearson_p:.4f} {sig_mark(pearson_p)}")
        print(f"  Spearman ρ={spearman_r:7.4f}, p={spearman_p:.4f} {sig_mark(spearman_p)}")
        print()
        
        results.append({
            'metric': metric,
            'label': label,
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'n_samples': len(df)
        })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv('analysis/correlation_results.csv', index=False)
    
    return results_df

def plot_correlation_heatmap(results_df):
    """绘制相关性热力图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Pearson相关性
    pearson_data = results_df.set_index('label')['pearson_r'].values.reshape(-1, 1)
    pearson_p = results_df.set_index('label')['pearson_p'].values.reshape(-1, 1)
    
    sns.heatmap(pearson_data, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                vmin=-0.5, vmax=0.5, cbar_kws={'label': 'Pearson r'},
                yticklabels=results_df['label'].values, xticklabels=['ΔAcc'],
                ax=axes[0])
    axes[0].set_title('Pearson Correlation with ΔAcc', fontsize=12, pad=10)
    
    # 添加显著性标记
    for i, p in enumerate(pearson_p.flatten()):
        if p < 0.05:
            axes[0].text(0.5, i+0.5, '*', ha='center', va='center', 
                        color='white' if abs(pearson_data[i][0]) > 0.25 else 'black',
                        fontsize=16, weight='bold')
    
    # Spearman相关性
    spearman_data = results_df.set_index('label')['spearman_r'].values.reshape(-1, 1)
    spearman_p = results_df.set_index('label')['spearman_p'].values.reshape(-1, 1)
    
    sns.heatmap(spearman_data, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                vmin=-0.5, vmax=0.5, cbar_kws={'label': 'Spearman ρ'},
                yticklabels=results_df['label'].values, xticklabels=['ΔAcc'],
                ax=axes[1])
    axes[1].set_title('Spearman Correlation with ΔAcc', fontsize=12, pad=10)
    
    # 添加显著性标记
    for i, p in enumerate(spearman_p.flatten()):
        if p < 0.05:
            axes[1].text(0.5, i+0.5, '*', ha='center', va='center',
                        color='white' if abs(spearman_data[i][0]) > 0.25 else 'black',
                        fontsize=16, weight='bold')
    
    plt.tight_layout()
    plt.savefig('analysis/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print("✅ 相关性热力图已保存: analysis/correlation_heatmap.png")
    plt.close()

def plot_scatter_matrix(df):
    """绘制关键指标散点图矩阵"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    key_metrics = [
        ('A_hit_mean', '异常max偶然命中'),
        ('B_separation', '判别分离度'),
        ('C_collapse_score', '原型塌缩分数'),
        ('B_overlap', '裕度重叠率')
    ]
    
    # 性能组颜色映射
    colors = {'Severe': '#d62728', 'Mild': '#ff7f0e', 'Stable': '#2ca02c', 'Improved': '#1f77b4'}
    
    for idx, (metric, label) in enumerate(key_metrics):
        ax = axes[idx // 2, idx % 2]
        
        for group in df['performance_group'].unique():
            group_data = df[df['performance_group'] == group]
            ax.scatter(group_data[metric], group_data['delta_acc'],
                      label=group, color=colors.get(group, 'gray'), alpha=0.6, s=50)
        
        # 回归线
        z = np.polyfit(df[metric], df['delta_acc'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df[metric].min(), df[metric].max(), 100)
        ax.plot(x_line, p(x_line), 'k--', alpha=0.5, linewidth=1.5)
        
        # 相关系数
        r, p_val = stats.pearsonr(df[metric], df['delta_acc'])
        ax.text(0.05, 0.95, f'r={r:.3f}\np={p_val:.3f}',
               transform=ax.transAxes, va='top', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel(label, fontsize=10)
        ax.set_ylabel('ΔAcc (%)', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
        
        if idx == 0:
            ax.legend(loc='lower right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('analysis/scatter_matrix.png', dpi=300, bbox_inches='tight')
    print("✅ 散点图矩阵已保存: analysis/scatter_matrix.png")
    plt.close()

def group_comparison(df):
    """按性能组对比指标分布"""
    print("\n" + "="*80)
    print("2️⃣  分组对比分析")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    metrics = [
        ('B_separation', '判别分离度'),
        ('C_collapse_score', '原型塌缩分数'),
        ('A_hit_mean', '异常max偶然命中'),
        ('B_overlap', '裕度重叠率')
    ]
    
    colors = {'Severe': '#d62728', 'Mild': '#ff7f0e', 'Stable': '#2ca02c', 'Improved': '#1f77b4'}
    
    for idx, (metric, label) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        data_by_group = [df[df['performance_group'] == g][metric].values 
                        for g in ['Severe', 'Mild', 'Stable', 'Improved']]
        
        bp = ax.boxplot(data_by_group, labels=['Severe', 'Mild', 'Stable', 'Improved'],
                       patch_artist=True, widths=0.6)
        
        for patch, group in zip(bp['boxes'], ['Severe', 'Mild', 'Stable', 'Improved']):
            patch.set_facecolor(colors[group])
            patch.set_alpha(0.7)
        
        ax.set_ylabel(label, fontsize=10)
        ax.set_xlabel('Performance Group', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='x', rotation=15)
    
    plt.tight_layout()
    plt.savefig('analysis/group_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ 分组对比箱线图已保存: analysis/group_comparison.png\n")
    
    # 统计每组均值
    group_stats = df.groupby('performance_group')[['B_separation', 'C_collapse_score', 
                                                    'A_hit_mean', 'B_overlap']].mean()
    print("📊 各组指标均值:")
    print(group_stats.to_string(float_format=lambda x: f"{x:.6f}"))
    
    # 保存详细统计
    detailed_stats = []
    for metric in ['C_collapse_score', 'B_separation', 'B_overlap', 'A_hit_mean']:
        for group in df['performance_group'].unique():
            group_data = df[df['performance_group'] == group][metric]
            detailed_stats.append({
                'metric': metric,
                'group': group,
                'mean': group_data.mean(),
                'std': group_data.std(),
                'median': group_data.median(),
                'n': len(group_data)
            })
    
    pd.DataFrame(detailed_stats).to_csv('analysis/group_statistics.csv', index=False)
    plt.close()

def baseline_strength_analysis(df):
    """按baseline强度分层的相关性分析 - Step 4C"""
    print("\n" + "="*80)
    print("3️⃣  Baseline强度分层分析 (Step 4C)")
    print("="*80 + "\n")
    
    metrics = {
        'A_hit_mean': '异常max偶然命中',
        'B_separation': '判别分离度',
        'B_overlap': '裕度重叠率',
        'C_collapse_score': '原型塌缩分数'
    }
    
    stratified_results = []
    
    for strength in ['Strong', 'Medium', 'Weak']:
        subset = df[df['baseline_strength'] == strength]
        n = len(subset)
        
        if n < 3:
            print(f"⚠️  {strength} 组样本量不足 (n={n})，跳过分析\n")
            continue
        
        print(f"📊 {strength} Baseline (n={n}):")
        print(f"   Baseline准确率范围: {subset['baseline_acc'].min():.2f}% - {subset['baseline_acc'].max():.2f}%")
        print(f"   平均ΔAcc: {subset['delta_acc'].mean():.2f}%")
        print(f"   性能分组: {subset['performance_group'].value_counts().to_dict()}\n")
        
        for metric, label in metrics.items():
            if subset[metric].std() < 1e-6:  # 方差过小
                print(f"   {label}: 方差过小，无法计算相关性")
                continue
            
            pearson_r, pearson_p = stats.pearsonr(subset[metric], subset['delta_acc'])
            spearman_r, spearman_p = stats.spearmanr(subset[metric], subset['delta_acc'])
            
            def sig_mark(p):
                if p < 0.05: return '*'
                else: return 'ns'
            
            print(f"   {label}:")
            print(f"      Pearson  r={pearson_r:6.3f}, p={pearson_p:.3f} {sig_mark(pearson_p)}")
            print(f"      Spearman ρ={spearman_r:6.3f}, p={spearman_p:.3f} {sig_mark(spearman_p)}")
            
            stratified_results.append({
                'baseline_strength': strength,
                'n_samples': n,
                'metric': metric,
                'label': label,
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'spearman_r': spearman_r,
                'spearman_p': spearman_p,
                'mean_delta_acc': subset['delta_acc'].mean(),
                'metric_mean': subset[metric].mean(),
                'metric_std': subset[metric].std()
            })
        
        print()
    
    # 保存分层结果
    stratified_df = pd.DataFrame(stratified_results)
    stratified_df.to_csv('analysis/baseline_strength_correlations.csv', index=False)
    print("✅ 分层相关性分析已保存: analysis/baseline_strength_correlations.csv\n")
    
    # 可视化分层相关性
    plot_stratified_correlations(stratified_df)
    
    return stratified_df

def plot_stratified_correlations(stratified_df):
    """可视化分层相关性结果"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics = ['B_separation', 'C_collapse_score', 'B_overlap', 'A_hit_mean']
    labels = ['判别分离度', '原型塌缩分数', '裕度重叠率', '异常max偶然命中']
    
    for idx, (metric, label) in enumerate(zip(metrics, labels)):
        ax = axes[idx // 2, idx % 2]
        
        metric_data = stratified_df[stratified_df['metric'] == metric]
        
        x = np.arange(len(metric_data))
        width = 0.35
        
        pearson_bars = ax.bar(x - width/2, metric_data['pearson_r'], width, 
                             label='Pearson r', alpha=0.8)
        spearman_bars = ax.bar(x + width/2, metric_data['spearman_r'], width,
                              label='Spearman ρ', alpha=0.8)
        
        # 标记显著性
        for i, row in enumerate(metric_data.itertuples()):
            if row.pearson_p < 0.05:
                ax.text(i - width/2, row.pearson_r + 0.02, '*', 
                       ha='center', fontsize=14, weight='bold')
            if row.spearman_p < 0.05:
                ax.text(i + width/2, row.spearman_r + 0.02, '*',
                       ha='center', fontsize=14, weight='bold')
        
        ax.set_ylabel('Correlation with ΔAcc', fontsize=10)
        ax.set_title(label, fontsize=11, pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(metric_data['baseline_strength'].values)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('analysis/stratified_correlations.png', dpi=300, bbox_inches='tight')
    print("✅ 分层相关性可视化已保存: analysis/stratified_correlations.png")
    plt.close()

def threshold_analysis(df):
    """阈值分析 - 找到预测退化的最佳阈值"""
    print("\n" + "="*80)
    print("4️⃣  阈值分析（预测退化）")
    print("="*80 + "\n")
    
    # 定义退化标签 (ΔAcc < -2%)
    df['is_degraded'] = (df['delta_acc'] < -2).astype(int)
    
    metrics = {
        'C_collapse_score': '原型塌缩分数',
        'B_separation': '判别分离度',
        'B_overlap': '裕度重叠率'
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    threshold_results = []
    
    for idx, (metric, label) in enumerate(metrics.items()):
        # 计算ROC曲线
        if metric == 'B_separation':
            # 分离度越低越可能退化，需要反转
            fpr, tpr, thresholds = roc_curve(df['is_degraded'], -df[metric])
            thresholds = -thresholds
        else:
            fpr, tpr, thresholds = roc_curve(df['is_degraded'], df[metric])
        
        roc_auc = auc(fpr, tpr)
        
        # 找最佳阈值 (Youden's J statistic)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        best_threshold = thresholds[best_idx]
        best_sensitivity = tpr[best_idx]
        best_specificity = 1 - fpr[best_idx]
        
        # 计算准确率
        if metric == 'B_separation':
            predictions = (df[metric] < best_threshold).astype(int)
        else:
            predictions = (df[metric] > best_threshold).astype(int)
        accuracy = (predictions == df['is_degraded']).mean()
        
        print(f"{label}:")
        print(f"  AUC: {roc_auc:.3f}")
        print(f"  最佳阈值: {best_threshold:.4f}")
        print(f"  灵敏度: {best_sensitivity*100:.2f}%")
        print(f"  特异性: {best_specificity*100:.2f}%")
        print(f"  准确率: {accuracy*100:.2f}%")
        print()
        
        threshold_results.append({
            'metric': metric,
            'label': label,
            'auc': roc_auc,
            'best_threshold': best_threshold,
            'sensitivity': best_sensitivity,
            'specificity': best_specificity,
            'accuracy': accuracy
        })
        
        # 绘制ROC曲线
        ax = axes[idx]
        ax.plot(fpr, tpr, linewidth=2, label=f'AUC = {roc_auc:.3f}')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
        ax.scatter([fpr[best_idx]], [tpr[best_idx]], s=100, c='red', 
                  marker='o', label=f'Best: {best_threshold:.3f}')
        ax.set_xlabel('False Positive Rate', fontsize=10)
        ax.set_ylabel('True Positive Rate', fontsize=10)
        ax.set_title(label, fontsize=11, pad=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('analysis/roc_curves.png', dpi=300, bbox_inches='tight')
    print("✅ ROC曲线已保存: analysis/roc_curves.png")
    
    pd.DataFrame(threshold_results).to_csv('analysis/threshold_analysis.csv', index=False)
    plt.close()

def main():
    print("="*80)
    print("Step 3 + Step 4C: 相关性与分层分析 - 完整验证")
    print("="*80)
    
    # 读取数据
    df = pd.read_csv('analysis/full_metrics_k2.csv')
    
    # 计算语义分数区分度
    df['E_semantic_gap'] = abs(df['E_abnormal_semantic'] - df['E_normal_semantic'])
    
    print(f"✅ 加载数据: {len(df)} 个类别\n")
    
    # Step 3: 整体相关性分析
    results_df = analyze_correlation(df)
    plot_correlation_heatmap(results_df)
    plot_scatter_matrix(df)
    
    # Step 3: 性能组对比
    group_comparison(df)
    
    # Step 4C: Baseline强度分层分析（关键！）
    stratified_df = baseline_strength_analysis(df)
    
    # Step 3: 阈值分析
    threshold_analysis(df)
    
    print("\n" + "="*80)
    print("✅ Step 3 + Step 4C 完成！")
    print("="*80 + "\n")
    print("关键文件:")
    print("  - analysis/correlation_results.csv")
    print("  - analysis/baseline_strength_correlations.csv  ⭐ Step 4C")
    print("  - analysis/correlation_heatmap.png")
    print("  - analysis/scatter_matrix.png")
    print("  - analysis/stratified_correlations.png  ⭐ Step 4C")
    print("  - analysis/group_comparison.png")
    print("  - analysis/roc_curves.png\n")

if __name__ == '__main__':
    main()
