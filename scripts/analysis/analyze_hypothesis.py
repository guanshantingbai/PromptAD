#!/usr/bin/env python3
"""
Step 3: 相关性分析 - 验证已提出假设在全类别上的成立性
目标：找出能稳定预测性能变化的指标，定位强类退化/难类改进的关键因素
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

# 设置绘图风格
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 10

def load_data():
    """加载完整数据"""
    df = pd.read_csv('analysis/full_metrics_k2.csv')
    print(f"✅ 加载数据: {len(df)} 个类别")
    return df

def analyze_correlation(df):
    """分析指标与性能变化的相关性"""
    print("\n" + "="*80)
    print("1️⃣  相关性分析")
    print("="*80)
    
    # 关键指标
    metrics = {
        'A_hit_mean': '异常max偶然命中(均值)',
        'A_hit_p95': '异常max偶然命中(P95)',
        'B_separation': '判别分离度',
        'B_overlap': '裕度重叠率',
        'C_collapse_score': '原型塌缩分数',
        'D_max_proto_count': '坏原型责任(最大计数)',
    }
    
    # 计算语义分数区分度
    df['E_semantic_gap'] = abs(df['E_abnormal_semantic'] - df['E_normal_semantic'])
    metrics['E_semantic_gap'] = '语义分数区分度'
    
    correlations = []
    
    for metric, label in metrics.items():
        # 去除NaN
        valid_data = df[[metric, 'delta_acc']].dropna()
        
        if len(valid_data) < 3:
            print(f"⚠️  {label}: 数据不足")
            continue
        
        # Pearson相关系数
        pearson_r, pearson_p = pearsonr(valid_data[metric], valid_data['delta_acc'])
        
        # Spearman相关系数
        spearman_r, spearman_p = spearmanr(valid_data[metric], valid_data['delta_acc'])
        
        correlations.append({
            'metric': metric,
            'label': label,
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'n_samples': len(valid_data)
        })
        
        # 显著性标记
        sig_p = '***' if pearson_p < 0.001 else '**' if pearson_p < 0.01 else '*' if pearson_p < 0.05 else 'ns'
        sig_s = '***' if spearman_p < 0.001 else '**' if spearman_p < 0.01 else '*' if spearman_p < 0.05 else 'ns'
        
        print(f"\n{label}:")
        print(f"  Pearson  r={pearson_r:>7.4f}, p={pearson_p:.4f} {sig_p}")
        print(f"  Spearman ρ={spearman_r:>7.4f}, p={spearman_p:.4f} {sig_s}")
    
    corr_df = pd.DataFrame(correlations)
    corr_df.to_csv('analysis/correlation_results.csv', index=False)
    
    return corr_df

def plot_correlation_heatmap(corr_df, output_path):
    """绘制相关系数热力图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Pearson相关系数
    labels = corr_df['label'].tolist()
    pearson_values = corr_df['pearson_r'].values.reshape(-1, 1)
    
    im1 = ax1.imshow(pearson_values, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    ax1.set_yticks(range(len(labels)))
    ax1.set_yticklabels(labels, fontsize=9)
    ax1.set_xticks([0])
    ax1.set_xticklabels(['ΔAcc'])
    ax1.set_title('Pearson Correlation (r)', fontsize=12, fontweight='bold')
    
    # 添加数值和显著性标记
    for i, (r, p) in enumerate(zip(corr_df['pearson_r'], corr_df['pearson_p'])):
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        color = 'white' if abs(r) > 0.5 else 'black'
        ax1.text(0, i, f'{r:.3f}{sig}', ha='center', va='center', color=color, fontsize=9)
    
    plt.colorbar(im1, ax=ax1)
    
    # Spearman相关系数
    spearman_values = corr_df['spearman_r'].values.reshape(-1, 1)
    
    im2 = ax2.imshow(spearman_values, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    ax2.set_yticks(range(len(labels)))
    ax2.set_yticklabels(labels, fontsize=9)
    ax2.set_xticks([0])
    ax2.set_xticklabels(['ΔAcc'])
    ax2.set_title('Spearman Correlation (ρ)', fontsize=12, fontweight='bold')
    
    # 添加数值和显著性标记
    for i, (r, p) in enumerate(zip(corr_df['spearman_r'], corr_df['spearman_p'])):
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        color = 'white' if abs(r) > 0.5 else 'black'
        ax2.text(0, i, f'{r:.3f}{sig}', ha='center', va='center', color=color, fontsize=9)
    
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✅ 相关性热力图已保存: {output_path}")

def plot_scatter_matrix(df, output_path):
    """绘制散点图矩阵"""
    metrics = {
        'A_hit_mean': '异常max偶然命中',
        'B_separation': '判别分离度',
        'C_collapse_score': '原型塌缩分数',
        'B_overlap': '裕度重叠率',
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for idx, (metric, label) in enumerate(metrics.items()):
        ax = axes[idx]
        
        # 根据性能分组着色
        colors = []
        for delta in df['delta_acc']:
            if delta < -5:
                colors.append('#d62728')  # 深红：严重退化
            elif delta < -2:
                colors.append('#ff7f0e')  # 橙色：轻微退化
            elif delta < 2:
                colors.append('#2ca02c')  # 绿色：持平
            else:
                colors.append('#1f77b4')  # 蓝色：改进
        
        # 散点图
        ax.scatter(df[metric], df['delta_acc'], c=colors, alpha=0.7, s=80, edgecolors='black', linewidth=0.5)
        
        # 拟合线
        valid_data = df[[metric, 'delta_acc']].dropna()
        if len(valid_data) > 3:
            z = np.polyfit(valid_data[metric], valid_data['delta_acc'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(valid_data[metric].min(), valid_data[metric].max(), 100)
            ax.plot(x_line, p(x_line), 'k--', alpha=0.5, linewidth=2)
            
            # 相关系数
            r, p_val = pearsonr(valid_data[metric], valid_data['delta_acc'])
            sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
            ax.text(0.05, 0.95, f'r={r:.3f} {sig}', transform=ax.transAxes, 
                   fontsize=11, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel(label, fontsize=11, fontweight='bold')
        ax.set_ylabel('ΔAcc (%)', fontsize=11, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.set_title(f'{label} vs Performance Change', fontsize=12)
    
    # 图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#d62728', label='严重退化 (Δ<-5%)'),
        Patch(facecolor='#ff7f0e', label='轻微退化 (-5%≤Δ<-2%)'),
        Patch(facecolor='#2ca02c', label='持平 (-2%≤Δ<2%)'),
        Patch(facecolor='#1f77b4', label='改进 (Δ≥2%)')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, 
              frameon=True, fontsize=10, bbox_to_anchor=(0.5, -0.02))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 散点图矩阵已保存: {output_path}")

def group_comparison(df, output_path):
    """按性能分组对比指标分布"""
    print("\n" + "="*80)
    print("2️⃣  分组对比分析")
    print("="*80)
    
    metrics = {
        'C_collapse_score': '原型塌缩分数',
        'B_separation': '判别分离度',
        'B_overlap': '裕度重叠率',
        'A_hit_mean': '异常max偶然命中',
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    group_stats = []
    
    for idx, (metric, label) in enumerate(metrics.items()):
        ax = axes[idx]
        
        # 按性能分组
        groups_data = []
        groups_labels = []
        for group in ['Severe Degrade', 'Mild Degrade', 'Stable', 'Improved']:
            group_df = df[df['performance_group'] == group]
            if len(group_df) > 0:
                groups_data.append(group_df[metric].dropna())
                groups_labels.append(f'{group}\n(n={len(group_df)})')
        
        # 箱线图
        bp = ax.boxplot(groups_data, labels=groups_labels, patch_artist=True,
                        widths=0.6, showmeans=True)
        
        # 着色
        colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']
        for patch, color in zip(bp['boxes'], colors[:len(groups_data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        ax.set_ylabel(label, fontsize=11, fontweight='bold')
        ax.set_title(f'{label} by Performance Group', fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='x', labelsize=9)
        
        # 统计
        for i, (group, data) in enumerate(zip(['Severe', 'Mild', 'Stable', 'Improved'], groups_data)):
            group_stats.append({
                'metric': label,
                'group': group,
                'mean': data.mean(),
                'std': data.std(),
                'median': data.median(),
                'n': len(data)
            })
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 分组对比箱线图已保存: {output_path}")
    
    # 输出统计表
    stats_df = pd.DataFrame(group_stats)
    pivot_table = stats_df.pivot_table(index='group', columns='metric', values='mean')
    
    print("\n📊 各组指标均值:")
    print(pivot_table.to_string())
    
    stats_df.to_csv('analysis/group_statistics.csv', index=False)
    
    return stats_df

def baseline_strength_analysis(df):
    """按baseline强度分层分析"""
    print("\n" + "="*80)
    print("3️⃣  Baseline强度分层分析")
    print("="*80)
    
    metrics = ['C_collapse_score', 'B_separation', 'A_hit_mean']
    
    for strength in ['Strong (≥95%)', 'Medium (85-95%)', 'Weak (<85%)']:
        subset = df[df['baseline_strength'] == strength]
        if len(subset) == 0:
            continue
        
        print(f"\n{strength} (n={len(subset)}):")
        print(f"  平均 ΔAcc: {subset['delta_acc'].mean():.2f}%")
        
        for metric in metrics:
            valid_data = subset[[metric, 'delta_acc']].dropna()
            if len(valid_data) < 3:
                continue
            r, p = pearsonr(valid_data[metric], valid_data['delta_acc'])
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            print(f"  {metric}: r={r:.3f} {sig}")

def threshold_analysis(df):
    """阈值分析：找出能区分"会退化"的阈值"""
    print("\n" + "="*80)
    print("4️⃣  阈值分析（预测退化）")
    print("="*80)
    
    # 定义"退化"为 delta < -2%
    df['is_degrade'] = (df['delta_acc'] < -2).astype(int)
    
    metrics = {
        'C_collapse_score': '原型塌缩分数',
        'B_separation': '判别分离度',
        'B_overlap': '裕度重叠率',
    }
    
    threshold_results = []
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, (metric, label) in enumerate(metrics.items()):
        ax = axes[idx]
        
        valid_data = df[[metric, 'is_degrade']].dropna()
        
        if len(valid_data) < 10:
            print(f"⚠️  {label}: 数据不足")
            continue
        
        # 对于separation，值越大越好，需要取反
        if metric == 'B_separation':
            fpr, tpr, thresholds = roc_curve(valid_data['is_degrade'], -valid_data[metric])
        else:
            fpr, tpr, thresholds = roc_curve(valid_data['is_degrade'], valid_data[metric])
        
        roc_auc = auc(fpr, tpr)
        
        # 找最佳阈值（Youden's J statistic）
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        best_threshold = thresholds[best_idx]
        best_sensitivity = tpr[best_idx]
        best_specificity = 1 - fpr[best_idx]
        
        # ROC曲线
        ax.plot(fpr, tpr, linewidth=2, label=f'AUC={roc_auc:.3f}')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
        ax.plot(fpr[best_idx], tpr[best_idx], 'ro', markersize=8, 
               label=f'Best: Sens={best_sensitivity:.2f}, Spec={best_specificity:.2f}')
        ax.set_xlabel('False Positive Rate', fontsize=11)
        ax.set_ylabel('True Positive Rate', fontsize=11)
        ax.set_title(f'{label}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        
        threshold_results.append({
            'metric': metric,
            'label': label,
            'auc': roc_auc,
            'best_threshold': best_threshold,
            'sensitivity': best_sensitivity,
            'specificity': best_specificity,
            'accuracy': (best_sensitivity * valid_data['is_degrade'].sum() + 
                        best_specificity * (len(valid_data) - valid_data['is_degrade'].sum())) / len(valid_data)
        })
        
        print(f"\n{label}:")
        print(f"  AUC: {roc_auc:.3f}")
        print(f"  最佳阈值: {best_threshold:.4f}")
        print(f"  灵敏度: {best_sensitivity:.2%}")
        print(f"  特异性: {best_specificity:.2%}")
        print(f"  准确率: {threshold_results[-1]['accuracy']:.2%}")
    
    plt.tight_layout()
    plt.savefig('analysis/roc_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✅ ROC曲线已保存: analysis/roc_curves.png")
    
    threshold_df = pd.DataFrame(threshold_results)
    threshold_df.to_csv('analysis/threshold_analysis.csv', index=False)
    
    return threshold_df

def main():
    print("="*80)
    print("Step 3: 相关性与分组分析 - 验证已提出假设")
    print("="*80)
    
    # 加载数据
    df = load_data()
    
    # 1. 相关性分析
    corr_df = analyze_correlation(df)
    plot_correlation_heatmap(corr_df, 'analysis/correlation_heatmap.png')
    
    # 2. 散点图矩阵
    plot_scatter_matrix(df, 'analysis/scatter_matrix.png')
    
    # 3. 分组对比
    stats_df = group_comparison(df, 'analysis/group_comparison.png')
    
    # 4. Baseline强度分层
    baseline_strength_analysis(df)
    
    # 5. 阈值分析
    threshold_df = threshold_analysis(df)
    
    print("\n" + "="*80)
    print("✅ Step 3 完成！")
    print("="*80)
    print("\n关键文件:")
    print("  - analysis/correlation_results.csv")
    print("  - analysis/correlation_heatmap.png")
    print("  - analysis/scatter_matrix.png")
    print("  - analysis/group_comparison.png")
    print("  - analysis/roc_curves.png")

if __name__ == '__main__':
    main()
