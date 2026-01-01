#!/usr/bin/env python
"""
Phase 1 可视化分析工具
生成prompt风险与性能关系的可视化图表
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

sns.set_style("whitegrid")
sns.set_palette("husl")


def load_summary(dataset):
    """加载分析摘要"""
    file_path = f'result/prompt_purging/analysis/{dataset}_class_summary.csv'
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return None
    return pd.read_csv(file_path)


def plot_risk_vs_performance(mvtec_df, visa_df, output_dir='result/prompt_purging/analysis'):
    """绘制风险指标 vs 性能的散点图"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Prompt Risk Metrics vs Semantic AUROC', fontsize=16, fontweight='bold')
    
    datasets = [
        (mvtec_df, 'MVTec-AD', 'blue'),
        (visa_df, 'VisA', 'red')
    ]
    
    risk_metrics = [
        ('high_risk_pct', 'High Risk Prompt Percentage (%)'),
        ('mean_R_j_eps', 'Mean R_j_eps'),
        ('pct_negative_margin', 'Negative Margin Percentage (%)')
    ]
    
    performance_metrics = [
        ('semantic_auroc', 'Semantic AUROC'),
        ('memory_auroc', 'Memory AUROC')
    ]
    
    for row_idx, (perf_col, perf_label) in enumerate(performance_metrics):
        for col_idx, (risk_col, risk_label) in enumerate(risk_metrics):
            ax = axes[row_idx, col_idx]
            
            for df, label, color in datasets:
                if df is not None and len(df) > 0:
                    x = df[risk_col]
                    y = df[perf_col]
                    
                    # 散点图
                    ax.scatter(x, y, alpha=0.6, s=100, label=label, color=color)
                    
                    # 添加类别标签
                    for _, row in df.iterrows():
                        ax.annotate(row['class'], 
                                  (row[risk_col], row[perf_col]),
                                  fontsize=7, alpha=0.7,
                                  xytext=(3, 3), textcoords='offset points')
                    
                    # 拟合趋势线
                    if len(x) > 2:
                        z = np.polyfit(x, y, 1)
                        p = np.poly1d(z)
                        x_line = np.linspace(x.min(), x.max(), 100)
                        ax.plot(x_line, p(x_line), "--", alpha=0.5, color=color, linewidth=1.5)
                        
                        # 计算相关系数
                        corr, p_val = stats.pearsonr(x, y)
                        ax.text(0.05, 0.95 if label == 'MVTec-AD' else 0.88, 
                               f'{label} r={corr:.3f}',
                               transform=ax.transAxes, 
                               fontsize=9,
                               verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor=color, alpha=0.2))
            
            ax.set_xlabel(risk_label, fontsize=10)
            ax.set_ylabel(perf_label, fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'risk_vs_performance_scatter.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ 散点图已保存: {output_file}")
    plt.close()


def plot_class_comparison(mvtec_df, visa_df, output_dir='result/prompt_purging/analysis'):
    """绘制各类别的风险和性能对比"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Class-wise Risk and Performance Comparison', fontsize=16, fontweight='bold')
    
    # MVTec
    ax1 = axes[0, 0]
    mvtec_sorted = mvtec_df.sort_values('semantic_auroc', ascending=True)
    y_pos = np.arange(len(mvtec_sorted))
    
    ax1.barh(y_pos, mvtec_sorted['semantic_auroc'], alpha=0.6, color='blue', label='Semantic AUROC')
    ax1.barh(y_pos, mvtec_sorted['memory_auroc'], alpha=0.4, color='green', label='Memory AUROC')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(mvtec_sorted['class'], fontsize=9)
    ax1.set_xlabel('AUROC (%)', fontsize=10)
    ax1.set_title('MVTec-AD: Performance by Class', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis='x')
    
    # VisA
    ax2 = axes[0, 1]
    visa_sorted = visa_df.sort_values('semantic_auroc', ascending=True)
    y_pos = np.arange(len(visa_sorted))
    
    ax2.barh(y_pos, visa_sorted['semantic_auroc'], alpha=0.6, color='red', label='Semantic AUROC')
    ax2.barh(y_pos, visa_sorted['memory_auroc'], alpha=0.4, color='orange', label='Memory AUROC')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(visa_sorted['class'], fontsize=9)
    ax2.set_xlabel('AUROC (%)', fontsize=10)
    ax2.set_title('VisA: Performance by Class', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='x')
    
    # MVTec 风险指标
    ax3 = axes[1, 0]
    mvtec_risk = mvtec_df.sort_values('high_risk_pct', ascending=True)
    y_pos = np.arange(len(mvtec_risk))
    
    ax3.barh(y_pos, mvtec_risk['high_risk_pct'], alpha=0.7, color='purple')
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(mvtec_risk['class'], fontsize=9)
    ax3.set_xlabel('High Risk Prompt Percentage (%)', fontsize=10)
    ax3.set_title('MVTec-AD: Prompt Risk by Class', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    
    # VisA 风险指标
    ax4 = axes[1, 1]
    visa_risk = visa_df.sort_values('high_risk_pct', ascending=True)
    y_pos = np.arange(len(visa_risk))
    
    ax4.barh(y_pos, visa_risk['high_risk_pct'], alpha=0.7, color='orange')
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(visa_risk['class'], fontsize=9)
    ax4.set_xlabel('High Risk Prompt Percentage (%)', fontsize=10)
    ax4.set_title('VisA: Prompt Risk by Class', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'class_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ 类别对比图已保存: {output_file}")
    plt.close()


def plot_correlation_heatmap(mvtec_df, visa_df, output_dir='result/prompt_purging/analysis'):
    """绘制相关性热力图"""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Correlation Heatmap: Risk Metrics vs Performance', fontsize=16, fontweight='bold')
    
    risk_cols = ['high_risk_pct', 'mean_R_j_eps', 'mean_R_j_0', 'pct_negative_margin']
    perf_cols = ['semantic_auroc', 'memory_auroc', 'image_auroc']
    
    # MVTec
    if mvtec_df is not None:
        mvtec_corr = mvtec_df[risk_cols + perf_cols].corr().loc[risk_cols, perf_cols]
        sns.heatmap(mvtec_corr, annot=True, fmt='.3f', cmap='RdBu_r', 
                   center=0, vmin=-1, vmax=1, ax=axes[0],
                   cbar_kws={'label': 'Correlation'})
        axes[0].set_title('MVTec-AD', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Performance Metrics', fontsize=10)
        axes[0].set_ylabel('Risk Metrics', fontsize=10)
    
    # VisA
    if visa_df is not None:
        visa_corr = visa_df[risk_cols + perf_cols].corr().loc[risk_cols, perf_cols]
        sns.heatmap(visa_corr, annot=True, fmt='.3f', cmap='RdBu_r', 
                   center=0, vmin=-1, vmax=1, ax=axes[1],
                   cbar_kws={'label': 'Correlation'})
        axes[1].set_title('VisA', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Performance Metrics', fontsize=10)
        axes[1].set_ylabel('Risk Metrics', fontsize=10)
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'correlation_heatmap.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ 相关性热力图已保存: {output_file}")
    plt.close()


def plot_risk_distribution(mvtec_df, visa_df, output_dir='result/prompt_purging/analysis'):
    """绘制风险分布对比"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Risk Metric Distribution Comparison', fontsize=16, fontweight='bold')
    
    metrics = [
        ('high_risk_pct', 'High Risk %'),
        ('mean_R_j_eps', 'Mean R_j_eps'),
        ('pct_negative_margin', 'Negative Margin %'),
        ('semantic_auroc', 'Semantic AUROC')
    ]
    
    for idx, (col, label) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        if mvtec_df is not None:
            ax.hist(mvtec_df[col], bins=10, alpha=0.5, color='blue', label='MVTec', edgecolor='black')
        
        if visa_df is not None:
            ax.hist(visa_df[col], bins=10, alpha=0.5, color='red', label='VisA', edgecolor='black')
        
        ax.set_xlabel(label, fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(f'Distribution: {label}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'risk_distribution.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ 分布对比图已保存: {output_file}")
    plt.close()


def main():
    print("="*80)
    print("Phase 1 可视化分析")
    print("="*80)
    
    output_dir = 'result/prompt_purging/analysis'
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    print("\n加载数据...")
    mvtec_df = load_summary('mvtec')
    visa_df = load_summary('visa')
    
    if mvtec_df is not None:
        print(f"✓ MVTec: {len(mvtec_df)} 个类别")
    if visa_df is not None:
        print(f"✓ VisA: {len(visa_df)} 个类别")
    
    # 生成图表
    print("\n生成可视化图表...")
    
    plot_risk_vs_performance(mvtec_df, visa_df, output_dir)
    plot_class_comparison(mvtec_df, visa_df, output_dir)
    plot_correlation_heatmap(mvtec_df, visa_df, output_dir)
    plot_risk_distribution(mvtec_df, visa_df, output_dir)
    
    print("\n" + "="*80)
    print("✅ 所有可视化图表已生成！")
    print(f"输出目录: {output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
