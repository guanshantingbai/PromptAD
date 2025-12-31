#!/usr/bin/env python3
"""
Visualize MAP/LAP Reliability Metrics

Generate interpretable plots to understand:
1. When MAP is unreliable vs when LAP is unreliable
2. Relationship between reliability metrics and failure modes
3. Consistency patterns across classes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)


def load_data(metrics_path, failure_mode_path=None):
    """Load reliability metrics and failure mode classification"""
    
    metrics_df = pd.read_csv(metrics_path)
    
    # Load failure mode table if provided
    if failure_mode_path and Path(failure_mode_path).exists():
        failure_df = pd.read_csv(failure_mode_path)
        failure_df.columns = failure_df.columns.str.strip()
        failure_df['class'] = failure_df['class'].str.strip()
        
        # Merge
        merged = metrics_df.merge(
            failure_df[['class', 'failure_type', 'semantic', 'fusion']],
            on='class',
            how='left'
        )
        return merged
    else:
        # No failure mode data - add placeholder columns
        metrics_df['failure_type'] = 'Unknown'
        metrics_df['semantic'] = np.nan
        metrics_df['fusion'] = np.nan
        return metrics_df


def plot_normal_side_risk(df, output_dir):
    """Plot Section I: Normal-side Risk Indicators"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Risk comparison: MAP vs LAP
    ax = axes[0, 0]
    x = np.arange(len(df))
    width = 0.35
    
    ax.bar(x - width/2, df['R_MAP_eps'], width, label='MAP', alpha=0.7, color='steelblue')
    ax.bar(x + width/2, df['R_LAP_eps'], width, label='LAP', alpha=0.7, color='coral')
    
    ax.set_xlabel('Class Index', fontsize=12)
    ax.set_ylabel(f'Risk: P(margin < ε)', fontsize=12)
    ax.set_title('Normal-side Risk: MAP vs LAP (ε=0.05)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 2. Margin distribution comparison
    ax = axes[0, 1]
    ax.scatter(df['margin_MAP_median'], df['margin_LAP_median'],
               c=df['R_MAP_eps'] + df['R_LAP_eps'], cmap='Reds',
               s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    ax.plot([df['margin_MAP_median'].min(), df['margin_MAP_median'].max()],
            [df['margin_LAP_median'].min(), df['margin_LAP_median'].max()],
            'k--', alpha=0.3, label='MAP=LAP')
    
    ax.set_xlabel('Median Margin (MAP)', fontsize=12)
    ax.set_ylabel('Median Margin (LAP)', fontsize=12)
    ax.set_title('Margin Distribution: MAP vs LAP', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label('Combined Risk', fontsize=10)
    
    # 3. Risk vs Semantic Performance
    ax = axes[1, 0]
    
    # Separate by failure mode
    for ftype in df['failure_type'].unique():
        if pd.isna(ftype):
            continue
        mask = df['failure_type'] == ftype
        ax.scatter(df.loc[mask, 'R_MAP_eps'], df.loc[mask, 'semantic'],
                  label=ftype, alpha=0.7, s=80)
    
    ax.set_xlabel('MAP Risk: P(margin < ε)', fontsize=12)
    ax.set_ylabel('Semantic AUROC (%)', fontsize=12)
    ax.set_title('Does MAP Risk Predict Semantic Failure?', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 4. Which anchor is more reliable?
    ax = axes[1, 1]
    
    df['MAP_more_reliable'] = (df['R_MAP_eps'] < df['R_LAP_eps']).astype(int)
    df['reliability_gap'] = df['R_LAP_eps'] - df['R_MAP_eps']
    
    colors = ['steelblue' if x > 0 else 'coral' for x in df['reliability_gap']]
    ax.barh(range(len(df)), df['reliability_gap'], color=colors, alpha=0.7)
    
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['class'].str.split('-').str[1], fontsize=8)
    ax.set_xlabel('R_LAP - R_MAP (>0: MAP better, <0: LAP better)', fontsize=12)
    ax.set_title('Which Anchor is More Reliable on Normal Samples?', fontsize=14, fontweight='bold')
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/normal_side_risk.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/normal_side_risk.png")
    plt.close()


def plot_consistency_metrics(df, output_dir):
    """Plot Section II: Consistency / Stability Indicators"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Margin discrepancy distribution
    ax = axes[0, 0]
    
    # Group by failure mode
    for ftype in df['failure_type'].unique():
        if pd.isna(ftype):
            continue
        mask = df['failure_type'] == ftype
        ax.scatter(df.loc[mask, 'margin_discrepancy_mean'],
                  df.loc[mask, 'semantic'],
                  label=ftype, alpha=0.7, s=80)
    
    ax.set_xlabel('Mean Margin Discrepancy |m_MAP - m_LAP|', fontsize=12)
    ax.set_ylabel('Semantic AUROC (%)', fontsize=12)
    ax.set_title('Does MAP-LAP Inconsistency Hurt Performance?', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 2. Disagreement rate
    ax = axes[0, 1]
    
    df_sorted = df.sort_values('disagreement_rate', ascending=False)
    colors = ['red' if x > 0.1 else 'steelblue' for x in df_sorted['disagreement_rate']]
    
    ax.barh(range(len(df_sorted)), df_sorted['disagreement_rate'], color=colors, alpha=0.7)
    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(df_sorted['class'].str.split('-').str[1], fontsize=8)
    ax.set_xlabel('Disagreement Rate (opposite judgments)', fontsize=12)
    ax.set_title('How Often Do MAP and LAP Disagree?', fontsize=14, fontweight='bold')
    ax.axvline(0.1, color='red', linestyle='--', linewidth=1, label='High Risk')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    
    # 3. Margin correlation
    ax = axes[1, 0]
    
    # Scatter: correlation vs semantic performance
    scatter = ax.scatter(df['margin_correlation'], df['semantic'],
                        c=df['margin_discrepancy_mean'], cmap='Reds',
                        s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('Margin Correlation (MAP vs LAP)', fontsize=12)
    ax.set_ylabel('Semantic AUROC (%)', fontsize=12)
    ax.set_title('Does MAP-LAP Correlation Predict Performance?', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Discrepancy', fontsize=10)
    
    # 4. Consistency by failure mode
    ax = axes[1, 1]
    
    # Box plot of discrepancy grouped by failure mode
    valid_ftypes = df.dropna(subset=['failure_type'])
    if len(valid_ftypes) > 0:
        failure_order = valid_ftypes.groupby('failure_type')['margin_discrepancy_mean'].median().sort_values().index
        sns.boxplot(data=valid_ftypes, x='failure_type', y='margin_discrepancy_mean',
                   order=failure_order, ax=ax, palette='Set2')
        ax.set_xlabel('Failure Mode', fontsize=12)
        ax.set_ylabel('Mean Margin Discrepancy', fontsize=12)
        ax.set_title('Consistency Patterns by Failure Mode', fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/consistency_metrics.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/consistency_metrics.png")
    plt.close()


def plot_geometric_metrics(df, output_dir):
    """Plot Section III: Anchor Geometry Indicators"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Anchor similarity heatmap
    ax = axes[0, 0]
    
    # Create similarity matrix for each class (show average)
    cos_matrix = np.array([
        [1.0, df['cos_normal_MAP'].mean(), df['cos_normal_LAP'].mean()],
        [df['cos_normal_MAP'].mean(), 1.0, df['cos_MAP_LAP'].mean()],
        [df['cos_normal_LAP'].mean(), df['cos_MAP_LAP'].mean(), 1.0]
    ])
    
    im = ax.imshow(cos_matrix, cmap='RdYlGn_r', vmin=0, vmax=1)
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(['μ_n', 'μ_MAP', 'μ_LAP'], fontsize=12)
    ax.set_yticklabels(['μ_n', 'μ_MAP', 'μ_LAP'], fontsize=12)
    ax.set_title('Average Anchor Cosine Similarity', fontsize=14, fontweight='bold')
    
    # Add text annotations
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f'{cos_matrix[i, j]:.3f}',
                   ha='center', va='center', fontsize=11, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='Cosine Similarity')
    
    # 2. cos(μ_n, μ_MAP) vs cos(μ_n, μ_LAP)
    ax = axes[0, 1]
    
    scatter = ax.scatter(df['cos_normal_MAP'], df['cos_normal_LAP'],
                        c=df['semantic'], cmap='viridis',
                        s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='MAP=LAP')
    ax.set_xlabel('cos(μ_n, μ_MAP)', fontsize=12)
    ax.set_ylabel('cos(μ_n, μ_LAP)', fontsize=12)
    ax.set_title('Normal Anchor Overlap with MAP vs LAP', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Semantic AUROC (%)', fontsize=10)
    
    # 3. cos(μ_MAP, μ_LAP) vs Semantic Performance
    ax = axes[1, 0]
    
    for ftype in df['failure_type'].unique():
        if pd.isna(ftype):
            continue
        mask = df['failure_type'] == ftype
        ax.scatter(df.loc[mask, 'cos_MAP_LAP'], df.loc[mask, 'semantic'],
                  label=ftype, alpha=0.7, s=80)
    
    ax.set_xlabel('cos(μ_MAP, μ_LAP)', fontsize=12)
    ax.set_ylabel('Semantic AUROC (%)', fontsize=12)
    ax.set_title('Does MAP-LAP Similarity Affect Performance?', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 4. Anchor distances
    ax = axes[1, 1]
    
    # Bar plot comparison
    x = np.arange(len(df))
    width = 0.25
    
    ax.bar(x - width, df['dist_normal_MAP'], width, label='d(μ_n, μ_MAP)', alpha=0.7)
    ax.bar(x, df['dist_normal_LAP'], width, label='d(μ_n, μ_LAP)', alpha=0.7)
    ax.bar(x + width, df['dist_MAP_LAP'], width, label='d(μ_MAP, μ_LAP)', alpha=0.7)
    
    ax.set_xlabel('Class Index', fontsize=12)
    ax.set_ylabel('L2 Distance', fontsize=12)
    ax.set_title('Anchor Distances', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/geometric_metrics.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/geometric_metrics.png")
    plt.close()


def generate_summary_report(df, output_path):
    """Generate text summary report"""
    
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MAP/LAP Reliability Metrics Summary Report\n")
        f.write("="*80 + "\n\n")
        
        # Section I: Normal-side Risk
        f.write("SECTION I: Normal-side Risk Indicators\n")
        f.write("-"*80 + "\n\n")
        
        f.write("Average Risk across all classes:\n")
        f.write(f"  MAP: R_0={df['R_MAP_0'].mean():.4f}, R_ε={df['R_MAP_eps'].mean():.4f}\n")
        f.write(f"  LAP: R_0={df['R_LAP_0'].mean():.4f}, R_ε={df['R_LAP_eps'].mean():.4f}\n\n")
        
        f.write("Classes with highest MAP risk (R_MAP_eps > 0.5):\n")
        high_risk_map = df[df['R_MAP_eps'] > 0.5][['class', 'R_MAP_eps', 'margin_MAP_median', 'semantic']].sort_values('R_MAP_eps', ascending=False)
        if len(high_risk_map) > 0:
            f.write(high_risk_map.to_string(index=False) + "\n\n")
        else:
            f.write("  None\n\n")
        
        f.write("Classes with highest LAP risk (R_LAP_eps > 0.5):\n")
        high_risk_lap = df[df['R_LAP_eps'] > 0.5][['class', 'R_LAP_eps', 'margin_LAP_median', 'semantic']].sort_values('R_LAP_eps', ascending=False)
        if len(high_risk_lap) > 0:
            f.write(high_risk_lap.to_string(index=False) + "\n\n")
        else:
            f.write("  None\n\n")
        
        # Section II: Consistency
        f.write("\n" + "="*80 + "\n")
        f.write("SECTION II: Consistency / Stability Indicators\n")
        f.write("-"*80 + "\n\n")
        
        f.write("Average Consistency metrics:\n")
        f.write(f"  Margin discrepancy: {df['margin_discrepancy_mean'].mean():.4f} ± {df['margin_discrepancy_mean'].std():.4f}\n")
        f.write(f"  Disagreement rate: {df['disagreement_rate'].mean():.4f} ± {df['disagreement_rate'].std():.4f}\n")
        f.write(f"  Margin correlation: {df['margin_correlation'].mean():.4f} ± {df['margin_correlation'].std():.4f}\n\n")
        
        f.write("Classes with high MAP-LAP disagreement (rate > 0.1):\n")
        high_disagreement = df[df['disagreement_rate'] > 0.1][['class', 'disagreement_rate', 'margin_discrepancy_mean', 'failure_type', 'semantic']].sort_values('disagreement_rate', ascending=False)
        if len(high_disagreement) > 0:
            f.write(high_disagreement.to_string(index=False) + "\n\n")
        else:
            f.write("  None\n\n")
        
        # Section III: Geometry
        f.write("\n" + "="*80 + "\n")
        f.write("SECTION III: Anchor Geometry Indicators\n")
        f.write("-"*80 + "\n\n")
        
        f.write("Average Anchor Similarities:\n")
        f.write(f"  cos(μ_n, μ_MAP): {df['cos_normal_MAP'].mean():.4f} ± {df['cos_normal_MAP'].std():.4f}\n")
        f.write(f"  cos(μ_n, μ_LAP): {df['cos_normal_LAP'].mean():.4f} ± {df['cos_normal_LAP'].std():.4f}\n")
        f.write(f"  cos(μ_MAP, μ_LAP): {df['cos_MAP_LAP'].mean():.4f} ± {df['cos_MAP_LAP'].std():.4f}\n\n")
        
        f.write("Classes with high anchor overlap (cos(μ_n, μ_MAP) > 0.9):\n")
        high_overlap = df[df['cos_normal_MAP'] > 0.9][['class', 'cos_normal_MAP', 'cos_normal_LAP', 'failure_type', 'semantic']].sort_values('cos_normal_MAP', ascending=False)
        if len(high_overlap) > 0:
            f.write(high_overlap.to_string(index=False) + "\n\n")
        else:
            f.write("  None\n\n")
        
        # Key Insights
        f.write("\n" + "="*80 + "\n")
        f.write("KEY INSIGHTS FOR GATING MECHANISM DESIGN\n")
        f.write("="*80 + "\n\n")
        
        f.write("1. When to DISTRUST MAP:\n")
        f.write(f"   - High normal-side risk: R_MAP_eps > {df['R_MAP_eps'].quantile(0.75):.3f}\n")
        f.write(f"   - High anchor overlap: cos(μ_n, μ_MAP) > {df['cos_normal_MAP'].quantile(0.75):.3f}\n\n")
        
        f.write("2. When to DISTRUST LAP:\n")
        f.write(f"   - High normal-side risk: R_LAP_eps > {df['R_LAP_eps'].quantile(0.75):.3f}\n")
        f.write(f"   - High anchor overlap: cos(μ_n, μ_LAP) > {df['cos_normal_LAP'].quantile(0.75):.3f}\n\n")
        
        f.write("3. When to USE MIXTURE (both unreliable):\n")
        f.write(f"   - High disagreement: disagreement_rate > {df['disagreement_rate'].quantile(0.75):.3f}\n")
        f.write(f"   - High discrepancy: margin_discrepancy_mean > {df['margin_discrepancy_mean'].quantile(0.75):.3f}\n\n")
        
        f.write("4. Correlation with Failure Modes:\n")
        valid_ftypes = df.dropna(subset=['failure_type'])
        if len(valid_ftypes) > 0:
            for ftype in valid_ftypes['failure_type'].unique():
                mask = valid_ftypes['failure_type'] == ftype
                subset = valid_ftypes[mask]
                f.write(f"\n   {ftype} ({len(subset)} classes):\n")
                f.write(f"     - Avg R_MAP_eps: {subset['R_MAP_eps'].mean():.3f}\n")
                f.write(f"     - Avg R_LAP_eps: {subset['R_LAP_eps'].mean():.3f}\n")
                f.write(f"     - Avg disagreement: {subset['disagreement_rate'].mean():.3f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")
    
    print(f"Saved: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize MAP/LAP reliability metrics')
    parser.add_argument('--metrics-path', type=str,
                        default='./result/baseline/baseline_analysis/MAP_LAP/reliability_metrics_k2.csv',
                        help='Path to reliability metrics CSV')
    parser.add_argument('--failure-mode-path', type=str,
                        default=None,
                        help='Path to failure mode table (optional)')
    parser.add_argument('--output-dir', type=str,
                        default='./result/baseline/baseline_analysis/MAP_LAP',
                        help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Load data
    print("\n" + "="*80)
    print("MAP/LAP Reliability Metrics Visualization")
    print("="*80)
    print(f"Loading metrics from: {args.metrics_path}")
    if args.failure_mode_path:
        print(f"Loading failure modes from: {args.failure_mode_path}")
    else:
        print("No failure mode data provided (optional)")
    
    df = load_data(args.metrics_path, args.failure_mode_path)
    print(f"Total classes: {len(df)}")
    
    # Generate plots
    print("\nGenerating visualizations...")
    plot_normal_side_risk(df, args.output_dir)
    plot_consistency_metrics(df, args.output_dir)
    plot_geometric_metrics(df, args.output_dir)
    
    # Generate summary report
    report_path = f"{args.output_dir}/reliability_summary.txt"
    generate_summary_report(df, report_path)
    
    print("\n" + "="*80)
    print("Visualization complete!")
    print(f"Output directory: {args.output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
