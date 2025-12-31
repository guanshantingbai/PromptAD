#!/usr/bin/env python3
"""
Memory-Semantic Branch Analysis

Analyze the relationship between Semantic and Memory branches,
including margin-semantic correlation and branch performance comparison.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from pathlib import Path
import argparse

sns.set_style('whitegrid')
plt.rcParams['font.size'] = 10


def plot_margin_semantic_correlation(df, output_path):
    """Plot margin separation vs semantic AUROC correlation"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Scatter plot with regression
    ax1.scatter(df['margin_separation'], df['semantic_auroc'], 
                alpha=0.6, s=100, c=df['delta_fusion'], cmap='RdYlGn', 
                edgecolors='black', linewidth=0.5)
    
    # Add regression line
    z = np.polyfit(df['margin_separation'], df['semantic_auroc'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['margin_separation'].min(), df['margin_separation'].max(), 100)
    ax1.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label=f'y = {z[0]:.1f}x + {z[1]:.1f}')
    
    # Add correlation text
    r_pearson, p_pearson = pearsonr(df['margin_separation'], df['semantic_auroc'])
    r_spearman, p_spearman = spearmanr(df['margin_separation'], df['semantic_auroc'])
    
    textstr = f'Pearson: r={r_pearson:.3f}, p={p_pearson:.4f}\nSpearman: ρ={r_spearman:.3f}, p={p_spearman:.4f}'
    ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, 
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax1.set_xlabel('Margin Separation', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Semantic AUROC (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Margin Separation vs Semantic Performance', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='RdYlGn', 
                                norm=plt.Normalize(vmin=df['delta_fusion'].min(), 
                                                   vmax=df['delta_fusion'].max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax1)
    cbar.set_label('Δ Fusion (Memory Boost)', rotation=270, labelpad=20)
    
    # Right: Box plot by failure mode
    # Group classes by margin level
    df_plot = df.copy()
    df_plot['margin_group'] = pd.cut(df_plot['margin_separation'], 
                                      bins=[0, 0.05, 0.10, 1.0],
                                      labels=['Low (<0.05)', 'Medium (0.05-0.10)', 'High (>0.10)'])
    
    df_melted = df_plot.melt(id_vars=['margin_group'], 
                              value_vars=['semantic_auroc', 'memory_auroc', 'fusion_auroc'],
                              var_name='Branch', value_name='AUROC')
    
    df_melted['Branch'] = df_melted['Branch'].map({
        'semantic_auroc': 'Semantic',
        'memory_auroc': 'Memory', 
        'fusion_auroc': 'Fusion'
    })
    
    sns.boxplot(data=df_melted, x='margin_group', y='AUROC', hue='Branch', ax=ax2)
    ax2.set_xlabel('Margin Separation Level', fontsize=12, fontweight='bold')
    ax2.set_ylabel('AUROC (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Performance by Margin Level', fontsize=13, fontweight='bold')
    ax2.legend(title='Branch')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {output_path}")


def plot_branch_comparison(df, output_path):
    """Plot comprehensive branch comparison"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Semantic vs Memory scatter
    ax = axes[0, 0]
    scatter = ax.scatter(df['semantic_auroc'], df['memory_auroc'], 
                         alpha=0.6, s=100, c=df['fusion_auroc'], 
                         cmap='viridis', edgecolors='black', linewidth=0.5)
    ax.plot([50, 100], [50, 100], 'r--', alpha=0.5, label='y=x')
    ax.set_xlabel('Semantic AUROC (%)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Memory AUROC (%)', fontsize=11, fontweight='bold')
    ax.set_title('Semantic vs Memory Performance', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Fusion AUROC')
    
    # 2. Delta comparison
    ax = axes[0, 1]
    x = np.arange(len(df))
    ax.bar(x, df['delta_fusion'], alpha=0.7, label='Δ Fusion (vs Semantic)', color='green')
    ax.bar(x, df['delta_memory'], alpha=0.7, label='Δ Memory (vs Semantic)', color='orange')
    ax.axhline(0, color='red', linestyle='--', linewidth=1)
    ax.set_xlabel('Class Index', fontsize=11, fontweight='bold')
    ax.set_ylabel('Performance Delta (%)', fontsize=11, fontweight='bold')
    ax.set_title('Memory/Fusion Boost over Semantic', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Performance distribution
    ax = axes[1, 0]
    data_to_plot = [df['semantic_auroc'], df['memory_auroc'], df['fusion_auroc']]
    bp = ax.boxplot(data_to_plot, labels=['Semantic', 'Memory', 'Fusion'],
                     patch_artist=True, showmeans=True)
    colors = ['lightblue', 'lightgreen', 'lightyellow']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_ylabel('AUROC (%)', fontsize=11, fontweight='bold')
    ax.set_title('Overall Performance Distribution', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add mean values as text
    for i, (data, label) in enumerate(zip(data_to_plot, ['Semantic', 'Memory', 'Fusion'])):
        mean_val = data.mean()
        ax.text(i+1, mean_val + 2, f'{mean_val:.1f}%', 
                ha='center', fontsize=9, fontweight='bold')
    
    # 4. Winner analysis
    ax = axes[1, 1]
    semantic_wins = (df['semantic_auroc'] > df['memory_auroc']).sum()
    memory_wins = (df['memory_auroc'] > df['semantic_auroc']).sum()
    ties = (df['semantic_auroc'] == df['memory_auroc']).sum()
    
    categories = ['Semantic Wins', 'Memory Wins', 'Ties']
    values = [semantic_wins, memory_wins, ties]
    colors_pie = ['lightblue', 'lightgreen', 'lightgray']
    
    wedges, texts, autotexts = ax.pie(values, labels=categories, autopct='%1.1f%%',
                                        colors=colors_pie, startangle=90)
    ax.set_title('Branch Performance Comparison\n(Which branch is better?)', 
                 fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {output_path}")


def analyze_branch_complementarity(df):
    """Analyze how Memory and Semantic complement each other"""
    
    results = []
    
    for idx, row in df.iterrows():
        class_name = row['class']
        semantic = row['semantic_auroc']
        memory = row['memory_auroc']
        fusion = row['fusion_auroc']
        
        # Classify complementarity
        if memory > semantic + 5:
            comp_type = 'Memory-Dominant'
            reason = f'Memory救场 (+{memory - semantic:.1f}%)'
        elif semantic > memory + 5:
            comp_type = 'Semantic-Dominant'
            reason = f'Semantic领先 (+{semantic - memory:.1f}%)'
        elif fusion > max(semantic, memory) + 3:
            comp_type = 'Synergistic'
            reason = f'互补融合 (+{fusion - max(semantic, memory):.1f}%)'
        else:
            comp_type = 'Similar'
            reason = '双分支表现接近'
        
        results.append({
            'class': class_name,
            'semantic': semantic,
            'memory': memory,
            'fusion': fusion,
            'delta_memory': memory - semantic,
            'delta_fusion': fusion - semantic,
            'complementarity': comp_type,
            'reason': reason
        })
    
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description='Memory-Semantic branch analysis')
    parser.add_argument('--input-csv', type=str, required=True,
                        help='Path to full_metrics_k2.csv')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for plots and reports')
    
    args = parser.parse_args()
    
    # Load data
    df = pd.read_csv(args.input_csv)
    print(f"Loaded {len(df)} classes")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80 + "\n")
    
    plot_margin_semantic_correlation(df, output_dir / 'margin_semantic_correlation.png')
    plot_branch_comparison(df, output_dir / 'branch_comparison.png')
    
    # Analyze complementarity
    print("\n" + "="*80)
    print("BRANCH COMPLEMENTARITY ANALYSIS")
    print("="*80 + "\n")
    
    comp_df = analyze_branch_complementarity(df)
    comp_df.to_csv(output_dir / 'branch_complementarity.csv', index=False)
    print(f"✅ Saved: {output_dir / 'branch_complementarity.csv'}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80 + "\n")
    
    print("Overall Performance:")
    print(f"  Semantic: {df['semantic_auroc'].mean():.2f}% ± {df['semantic_auroc'].std():.2f}%")
    print(f"  Memory:   {df['memory_auroc'].mean():.2f}% ± {df['memory_auroc'].std():.2f}%")
    print(f"  Fusion:   {df['fusion_auroc'].mean():.2f}% ± {df['fusion_auroc'].std():.2f}%")
    
    print("\nPerformance Boost:")
    print(f"  Δ Memory:  {df['delta_memory'].mean():.2f}% ± {df['delta_memory'].std():.2f}%")
    print(f"  Δ Fusion:  {df['delta_fusion'].mean():.2f}% ± {df['delta_fusion'].std():.2f}%")
    
    print("\nMargin-Semantic Correlation:")
    r_pearson, p_pearson = pearsonr(df['margin_separation'], df['semantic_auroc'])
    r_spearman, p_spearman = spearmanr(df['margin_separation'], df['semantic_auroc'])
    print(f"  Pearson:  r = {r_pearson:.3f}, p = {p_pearson:.6f}")
    print(f"  Spearman: ρ = {r_spearman:.3f}, p = {p_spearman:.6f}")
    
    print("\nComplementarity Distribution:")
    print(comp_df['complementarity'].value_counts())
    
    print("\n" + "="*80)
    print(f"All results saved to: {output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
