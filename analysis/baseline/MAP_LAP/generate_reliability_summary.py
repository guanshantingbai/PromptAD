#!/usr/bin/env python3
"""
Generate comprehensive MAP/LAP reliability summary report.

This script generates a detailed report similar to reliability_summary.txt
analyzing MAP and LAP anchor reliability and their relationship to performance.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse


def format_section_header(title):
    """Format a section header"""
    line = "=" * 80
    return f"\n{line}\n{title}\n{line}\n"


def generate_normal_side_risk_section(df):
    """Generate Section I: Normal-side Risk Indicators"""
    
    output = format_section_header("SECTION I: Normal-side Risk Indicators")
    output += "\n"
    
    # Average risk
    output += "Average Risk across all classes:\n"
    output += f"  MAP: R_0={df['R_MAP_0'].mean():.4f}, R_ε={df['R_MAP_eps'].mean():.4f}\n"
    output += f"  LAP: R_0={df['R_LAP_0'].mean():.4f}, R_ε={df['R_LAP_eps'].mean():.4f}\n\n"
    
    # High MAP risk
    high_map = df[df['R_MAP_eps'] > 0.5].sort_values('R_MAP_eps', ascending=False)
    output += f"Classes with HIGH MAP risk (R_MAP_eps > 0.5): {len(high_map)} classes\n"
    if len(high_map) > 0:
        output += high_map[['class', 'R_MAP_eps', 'margin_MAP_median', 'semantic_auroc']].to_string(index=False)
        output += "\n\n"
    
    # High LAP risk
    high_lap = df[df['R_LAP_eps'] > 0.5].sort_values('R_LAP_eps', ascending=False)
    output += f"Classes with HIGH LAP risk (R_LAP_eps > 0.5): {len(high_lap)} classes\n"
    if len(high_lap) > 0:
        output += high_lap[['class', 'R_LAP_eps', 'margin_LAP_median', 'semantic_auroc']].to_string(index=False)
        output += "\n\n"
    
    # Risk comparison
    map_higher = (df['R_MAP_0'] > df['R_LAP_0']).sum()
    lap_higher = (df['R_LAP_0'] > df['R_MAP_0']).sum()
    output += f"Risk Comparison (R_0):\n"
    output += f"  MAP higher risk: {map_higher}/{len(df)} classes ({map_higher/len(df)*100:.1f}%)\n"
    output += f"  LAP higher risk: {lap_higher}/{len(df)} classes ({lap_higher/len(df)*100:.1f}%)\n\n"
    
    return output


def generate_consistency_section(df):
    """Generate Section II: Consistency / Stability Indicators"""
    
    output = format_section_header("SECTION II: Consistency / Stability Indicators")
    output += "\n"
    
    # Average consistency
    output += "Average Consistency metrics:\n"
    output += f"  Margin discrepancy: {df['margin_discrepancy_mean'].mean():.4f} ± {df['margin_discrepancy_mean'].std():.4f}\n"
    output += f"  Disagreement rate: {df['disagreement_rate'].mean():.4f} ± {df['disagreement_rate'].std():.4f}\n"
    output += f"  Margin correlation: {df['margin_correlation'].mean():.4f} ± {df['margin_correlation'].std():.4f}\n\n"
    
    # High disagreement
    high_disagree = df[df['disagreement_rate'] > 0.5].sort_values('disagreement_rate', ascending=False)
    output += f"Classes with HIGH MAP-LAP disagreement (rate > 0.5): {len(high_disagree)} classes\n"
    if len(high_disagree) > 0:
        cols = ['class', 'disagreement_rate', 'margin_discrepancy_mean', 'semantic_auroc', 'fusion_auroc']
        output += high_disagree[cols].to_string(index=False)
        output += "\n\n"
    
    # Negative correlation
    neg_corr = df[df['margin_correlation'] < 0].sort_values('margin_correlation')
    output += f"Classes with NEGATIVE margin correlation: {len(neg_corr)} classes\n"
    if len(neg_corr) > 0:
        cols = ['class', 'margin_correlation', 'semantic_auroc', 'fusion_auroc']
        output += neg_corr[cols].head(10).to_string(index=False)
        output += "\n\n"
    
    return output


def generate_anchor_geometry_section(df):
    """Generate Section III: Anchor Geometry Indicators"""
    
    output = format_section_header("SECTION III: Anchor Geometry Indicators")
    output += "\n"
    
    # Average similarities
    output += "Average Anchor Similarities:\n"
    output += f"  cos(μ_n, μ_MAP): {df['cos_normal_MAP'].mean():.4f} ± {df['cos_normal_MAP'].std():.4f}\n"
    output += f"  cos(μ_n, μ_LAP): {df['cos_normal_LAP'].mean():.4f} ± {df['cos_normal_LAP'].std():.4f}\n"
    output += f"  cos(μ_MAP, μ_LAP): {df['cos_MAP_LAP'].mean():.4f} ± {df['cos_MAP_LAP'].std():.4f}\n\n"
    
    # High anchor overlap (MAP-LAP alignment)
    high_overlap = df[df['cos_MAP_LAP'] > 0.92].sort_values('cos_MAP_LAP', ascending=False)
    output += f"Classes with HIGH MAP-LAP alignment (cos > 0.92): {len(high_overlap)} classes\n"
    if len(high_overlap) > 0:
        cols = ['class', 'cos_MAP_LAP', 'semantic_auroc', 'fusion_auroc']
        output += high_overlap[cols].to_string(index=False)
        output += "\n\n"
    
    # Normal-anchor collapse
    normal_map_collapse = df[df['cos_normal_MAP'] > 0.95].sort_values('cos_normal_MAP', ascending=False)
    output += f"Classes with Normal-MAP anchor collapse (cos > 0.95): {len(normal_map_collapse)} classes\n"
    if len(normal_map_collapse) > 0:
        cols = ['class', 'cos_normal_MAP', 'margin_MAP_mean', 'semantic_auroc']
        output += normal_map_collapse[cols].head(10).to_string(index=False)
        output += "\n\n"
    
    return output


def generate_failure_mode_analysis(df):
    """Generate Section IV: Failure Mode Analysis"""
    
    output = format_section_header("SECTION IV: Failure Mode vs MAP/LAP Reliability")
    output += "\n"
    
    # Load failure mode data
    failure_df = pd.read_csv('result/baseline/baseline_analysis/MAP_LAP/failure_mode_table.csv')
    
    # Merge
    df_merged = df.merge(failure_df[['class', 'failure_type']], on='class', how='left')
    
    # Group by failure type
    output += "Reliability metrics by Failure Mode:\n\n"
    output += f"{'Mode':<15} {'Count':>6} {'R_MAP_0':>10} {'R_LAP_0':>10} {'Disagree':>10} {'cos(M,L)':>10} {'Semantic':>10}\n"
    output += "-" * 80 + "\n"
    
    for mode in sorted(df_merged['failure_type'].dropna().unique()):
        mode_df = df_merged[df_merged['failure_type'] == mode]
        output += f"{mode:<15} {len(mode_df):>6} "
        output += f"{mode_df['R_MAP_0'].mean():>10.3f} "
        output += f"{mode_df['R_LAP_0'].mean():>10.3f} "
        output += f"{mode_df['disagreement_rate'].mean():>10.3f} "
        output += f"{mode_df['cos_MAP_LAP'].mean():>10.3f} "
        output += f"{mode_df['semantic_auroc'].mean():>10.2f}\n"
    
    output += "\n"
    
    return output


def generate_key_insights(df):
    """Generate Section V: Key Insights"""
    
    output = format_section_header("KEY INSIGHTS FOR GATING MECHANISM DESIGN")
    output += "\n"
    
    # Thresholds
    map_risk_thresh = df['R_MAP_0'].quantile(0.75)
    lap_risk_thresh = df['R_LAP_0'].quantile(0.75)
    disagree_thresh = df['disagreement_rate'].quantile(0.75)
    cos_thresh = df['cos_MAP_LAP'].quantile(0.75)
    
    output += "1. When to DISTRUST MAP:\n"
    output += f"   - High normal-side risk: R_MAP_0 > {map_risk_thresh:.3f}\n"
    output += f"   - High anchor overlap with normal: cos(μ_n, μ_MAP) > {df['cos_normal_MAP'].quantile(0.75):.3f}\n"
    output += f"   - Affected classes: {(df['R_MAP_0'] > map_risk_thresh).sum()}/{len(df)}\n\n"
    
    output += "2. When to TRUST LAP:\n"
    output += f"   - Low normal-side risk: R_LAP_0 < {lap_risk_thresh:.3f}\n"
    output += f"   - Good separation from normal: cos(μ_n, μ_LAP) < {df['cos_normal_LAP'].quantile(0.25):.3f}\n"
    output += f"   - Candidate classes: {(df['R_LAP_0'] < lap_risk_thresh).sum()}/{len(df)}\n\n"
    
    output += "3. When to USE ENSEMBLE (both unreliable):\n"
    output += f"   - High disagreement: disagreement_rate > {disagree_thresh:.3f}\n"
    output += f"   - High discrepancy: margin_discrepancy > {df['margin_discrepancy_mean'].quantile(0.75):.3f}\n"
    output += f"   - Affected classes: {(df['disagreement_rate'] > disagree_thresh).sum()}/{len(df)}\n\n"
    
    output += "4. LAP vs MAP Performance:\n"
    map_better = (df['R_MAP_0'] < df['R_LAP_0']).sum()
    lap_better = (df['R_LAP_0'] < df['R_MAP_0']).sum()
    output += f"   - MAP more reliable: {map_better}/{len(df)} classes ({map_better/len(df)*100:.1f}%)\n"
    output += f"   - LAP more reliable: {lap_better}/{len(df)} classes ({lap_better/len(df)*100:.1f}%)\n\n"
    
    # Correlation with semantic performance
    from scipy.stats import pearsonr
    r_map, p_map = pearsonr(df['R_MAP_0'], df['semantic_auroc'])
    r_lap, p_lap = pearsonr(df['R_LAP_0'], df['semantic_auroc'])
    
    output += "5. Risk vs Semantic Performance:\n"
    output += f"   - R_MAP_0 correlation: r={r_map:.3f}, p={p_map:.4f}\n"
    output += f"   - R_LAP_0 correlation: r={r_lap:.3f}, p={p_lap:.4f}\n\n"
    
    return output


def main():
    parser = argparse.ArgumentParser(description='Generate MAP/LAP reliability summary')
    parser.add_argument('--input-csv', type=str, required=True,
                        help='Path to full_metrics_k2.csv')
    parser.add_argument('--output-txt', type=str, required=True,
                        help='Output path for reliability summary')
    
    args = parser.parse_args()
    
    # Load data
    df = pd.read_csv(args.input_csv)
    print(f"Loaded {len(df)} classes")
    
    # Generate report sections
    report = format_section_header("MAP/LAP Reliability Metrics Summary Report")
    report += "\nGenerated from: Harmonic Fusion Baseline\n"
    report += f"Total classes analyzed: {len(df)}\n"
    
    report += generate_normal_side_risk_section(df)
    report += generate_consistency_section(df)
    report += generate_anchor_geometry_section(df)
    report += generate_failure_mode_analysis(df)
    report += generate_key_insights(df)
    
    report += format_section_header("END OF REPORT")
    
    # Save report
    output_path = Path(args.output_txt)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    
    print(f"\n✅ Report saved to: {output_path}")


if __name__ == '__main__':
    main()
