"""
Failure Mode Analysis for PromptAD Baseline

Classifies each class into failure modes and analyzes correlations.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr


def classify_failure_mode(row: pd.Series) -> Tuple[str, str]:
    """
    Classify failure mode for a single class.
    
    Args:
        row: DataFrame row with metrics
        
    Returns:
        (failure_type, detailed_reason)
    """
    semantic = row['semantic_auroc']
    fusion = row['fusion_auroc']
    delta = row['delta_fusion']
    sep = row['margin_separation']
    cos_sim = row['anchor_cosine_sim']
    map_lap_gap = row['MAP_auroc'] - row['LAP_auroc']
    
    # Define thresholds
    MARGIN_COLLAPSE_THRESHOLD = 0.05
    ANCHOR_CLOSE_THRESHOLD = 0.92
    SEMANTIC_WEAK_THRESHOLD = 85.0
    DELTA_LARGE_THRESHOLD = 8.0
    SEMANTIC_STRONG_THRESHOLD = 95.0
    
    failure_types = []
    reasons = []
    
    # Mode A: Margin Collapse (严重失效)
    if sep < MARGIN_COLLAPSE_THRESHOLD:
        failure_types.append('A')
        reasons.append(f'margin_collapse(sep={sep:.3f})')
    
    # Mode B: Anchor Direction Collapse (锚点方向崩溃)
    if cos_sim > ANCHOR_CLOSE_THRESHOLD:
        failure_types.append('B')
        reasons.append(f'anchor_collapse(cos={cos_sim:.3f})')
    
    # Mode C: Semantic Weak, Fusion Dominant (语义弱但fusion救场)
    if semantic < SEMANTIC_WEAK_THRESHOLD and delta > DELTA_LARGE_THRESHOLD:
        failure_types.append('C')
        reasons.append(f'semantic_weak_fusion_saves(Δ={delta:.1f})')
    
    # Mode D: LAP Harmful (LAP拖后腿)
    if map_lap_gap > 0.2:
        failure_types.append('D')
        reasons.append(f'LAP_harmful(gap={map_lap_gap:.2f})')
    
    # Success cases
    if not failure_types:
        if semantic >= SEMANTIC_STRONG_THRESHOLD and abs(delta) < 3.0:
            return 'SUCCESS', 'semantic_strong_no_collapse'
        elif semantic >= SEMANTIC_WEAK_THRESHOLD and delta >= 0:
            return 'PARTIAL_SUCCESS', 'semantic_ok_fusion_helps'
        else:
            return 'UNKNOWN', 'no_clear_pattern'
    
    # Multiple failure modes
    failure_str = '+'.join(sorted(failure_types))
    reason_str = '; '.join(reasons)
    
    return failure_str, reason_str


def generate_failure_mode_table(
    mvtec_df: pd.DataFrame,
    visa_df: pd.DataFrame,
    output_path: Path
) -> pd.DataFrame:
    """
    Generate comprehensive failure mode table.
    """
    # Filter out summary rows
    mvtec_data = mvtec_df[~mvtec_df['class_name'].isin(['MEAN', 'STD', 'MEDIAN', 'MIN', 'MAX'])].copy()
    visa_data = visa_df[~visa_df['class_name'].isin(['MEAN', 'STD', 'MEDIAN', 'MIN', 'MAX'])].copy()
    
    # Add dataset prefix
    mvtec_data['class'] = 'mvtec-' + mvtec_data['class_name']
    visa_data['class'] = 'visa-' + visa_data['class_name']
    
    # Combine
    all_data = pd.concat([mvtec_data, visa_data], ignore_index=True)
    
    # Calculate MAP-LAP gap
    all_data['MAP_LAP_gap'] = all_data['MAP_auroc'] - all_data['LAP_auroc']
    
    # Classify failure modes
    all_data[['failure_type', 'failure_reason']] = all_data.apply(
        lambda row: pd.Series(classify_failure_mode(row)), axis=1
    )
    
    # Create summary table
    summary_cols = [
        'class',
        'semantic_auroc',
        'fusion_auroc',
        'delta_fusion',
        'margin_separation',
        'anchor_cosine_sim',
        'MAP_auroc',
        'LAP_auroc',
        'MAP_LAP_gap',
        'failure_type',
        'failure_reason'
    ]
    
    result_df = all_data[summary_cols].copy()
    result_df = result_df.rename(columns={
        'semantic_auroc': 'semantic',
        'fusion_auroc': 'fusion',
        'delta_fusion': 'delta',
        'margin_separation': 'sep',
        'anchor_cosine_sim': 'anchor_cos'
    })
    
    # Sort by failure type, then by semantic AUROC
    result_df = result_df.sort_values(['failure_type', 'semantic'], ascending=[True, False])
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False, float_format='%.4f')
    
    return result_df


def analyze_margin_semantic_correlation(
    df: pd.DataFrame,
    output_dir: Path
) -> Dict:
    """
    Analyze correlation between margin separation and semantic performance.
    """
    # Filter valid data
    valid_data = df[df['failure_type'].notna()].copy()
    
    sep = valid_data['sep'].values
    semantic = valid_data['semantic'].values
    
    # Calculate correlations
    pearson_r, pearson_p = pearsonr(sep, semantic)
    spearman_r, spearman_p = spearmanr(sep, semantic)
    
    # Create scatter plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Scatter with regression
    ax1 = axes[0]
    colors = []
    for ft in valid_data['failure_type']:
        if 'A' in str(ft):  # Margin collapse
            colors.append('red')
        elif 'C' in str(ft):  # Semantic weak
            colors.append('orange')
        elif 'SUCCESS' in str(ft):
            colors.append('green')
        else:
            colors.append('gray')
    
    ax1.scatter(sep, semantic, c=colors, alpha=0.6, s=80)
    
    # Add regression line
    z = np.polyfit(sep, semantic, 1)
    p = np.poly1d(z)
    x_line = np.linspace(sep.min(), sep.max(), 100)
    ax1.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)
    
    ax1.set_xlabel('Margin Separation', fontsize=12)
    ax1.set_ylabel('Semantic AUROC (%)', fontsize=12)
    ax1.set_title(f'Correlation: r={pearson_r:.3f}, p={pearson_p:.4f}', fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=85, color='blue', linestyle='--', alpha=0.5, label='Threshold=85')
    ax1.axvline(x=0.05, color='red', linestyle='--', alpha=0.5, label='Collapse Threshold')
    ax1.legend()
    
    # Plot 2: Boxplot by failure type
    ax2 = axes[1]
    failure_types = valid_data['failure_type'].unique()
    
    # Group by main failure type
    main_types = []
    for ft in valid_data['failure_type']:
        if 'A' in str(ft):
            main_types.append('A: Margin Collapse')
        elif 'B' in str(ft):
            main_types.append('B: Anchor Collapse')
        elif 'C' in str(ft):
            main_types.append('C: Semantic Weak')
        elif 'SUCCESS' in str(ft):
            main_types.append('Success')
        else:
            main_types.append('Other')
    
    valid_data['main_failure'] = main_types
    
    groups = valid_data.groupby('main_failure')['semantic'].apply(list)
    ax2.boxplot(groups.values, labels=groups.index, patch_artist=True)
    ax2.set_ylabel('Semantic AUROC (%)', fontsize=12)
    ax2.set_title('Semantic Performance by Failure Type', fontsize=13)
    ax2.grid(True, alpha=0.3, axis='y')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=15, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'margin_semantic_correlation.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Analyze by groups
    group_stats = valid_data.groupby('main_failure').agg({
        'semantic': ['mean', 'std', 'count'],
        'sep': ['mean', 'std'],
        'delta': ['mean', 'std']
    }).round(4)
    
    return {
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'group_stats': group_stats,
        'total_classes': len(valid_data)
    }


def print_failure_mode_summary(df: pd.DataFrame):
    """
    Print comprehensive failure mode summary.
    """
    print("=" * 100)
    print("FAILURE MODE CLASSIFICATION SUMMARY")
    print("=" * 100)
    print()
    
    # Count by failure type
    failure_counts = df['failure_type'].value_counts().sort_index()
    
    print("【Failure Mode Distribution】")
    print("-" * 100)
    for ft, count in failure_counts.items():
        percentage = count / len(df) * 100
        print(f"  {ft:<20} {count:>3} classes ({percentage:>5.1f}%)")
    
    print()
    print("【Failure Mode Definitions】")
    print("-" * 100)
    print("  A: Margin Collapse       - margin separation < 0.05")
    print("  B: Anchor Collapse       - cos(μ_n, μ_a) > 0.92")
    print("  C: Semantic Weak         - semantic < 85 and Δ > 8")
    print("  D: LAP Harmful           - MAP AUROC - LAP AUROC > 0.2")
    print("  A+B+C: Multiple failures combined")
    print()
    
    # Print each failure mode details
    for ft in sorted(failure_counts.index):
        subset = df[df['failure_type'] == ft]
        print()
        print(f"【{ft}】 ({len(subset)} classes)")
        print("-" * 100)
        
        for _, row in subset.iterrows():
            print(f"  {row['class']:<25} "
                  f"Sem:{row['semantic']:>6.2f}  "
                  f"Fus:{row['fusion']:>6.2f}  "
                  f"Δ:{row['delta']:>+6.2f}  "
                  f"Sep:{row['sep']:>6.3f}  "
                  f"Cos:{row['anchor_cos']:>6.3f}  "
                  f"MAP-LAP:{row['MAP_LAP_gap']:>+6.2f}")
            if row['failure_reason']:
                print(f"       └─ {row['failure_reason']}")


def main():
    # Paths
    base_dir = Path('result/baseline/baseline_analysis')
    mvtec_summary = base_dir / 'mvtec/k_2/seed_111/results/summary_report.csv'
    visa_summary = base_dir / 'visa/k_2/seed_111/results/summary_report.csv'
    output_dir = base_dir / 'combined_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading data...")
    mvtec_df = pd.read_csv(mvtec_summary)
    visa_df = pd.read_csv(visa_summary)
    
    # Generate failure mode table
    print("Classifying failure modes...")
    result_df = generate_failure_mode_table(
        mvtec_df, visa_df,
        output_dir / 'failure_mode_table.csv'
    )
    
    print(f"\n✅ Failure mode table saved to: {output_dir / 'failure_mode_table.csv'}")
    print()
    
    # Print summary
    print_failure_mode_summary(result_df)
    
    # Analyze correlation
    print()
    print("=" * 100)
    print("MARGIN-SEMANTIC CORRELATION ANALYSIS")
    print("=" * 100)
    print()
    
    corr_results = analyze_margin_semantic_correlation(result_df, output_dir)
    
    print(f"Pearson correlation:  r = {corr_results['pearson_r']:.4f}, p = {corr_results['pearson_p']:.6f}")
    print(f"Spearman correlation: ρ = {corr_results['spearman_r']:.4f}, p = {corr_results['spearman_p']:.6f}")
    print()
    
    if corr_results['pearson_p'] < 0.05:
        print("✅ Statistically significant positive correlation!")
        print("   → Larger margin separation strongly predicts better semantic performance")
    else:
        print("⚠️  Correlation not statistically significant (p > 0.05)")
    
    print()
    print("【Performance by Failure Type】")
    print("-" * 100)
    print(corr_results['group_stats'])
    print()
    
    print(f"\n✅ Correlation plot saved to: {output_dir / 'margin_semantic_correlation.png'}")
    print()
    print("=" * 100)


if __name__ == '__main__':
    main()
