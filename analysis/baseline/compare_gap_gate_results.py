#!/usr/bin/env python3
"""
Compare baseline vs gap-gated fusion results for VisA dataset.

This script:
1. Loads baseline fusion results from result/baseline/visa/k_2/csv/
2. Loads gap-gated results from result/test_gap_gate/visa/k_2/csv/
3. Computes delta AUROC for each class
4. Generates summary statistics and visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Configuration
DATASET = "visa"
K_SHOT = 2

BASELINE_CSV = Path(f"../../result/baseline/{DATASET}/k_{K_SHOT}/csv/{DATASET}.csv")
GATED_CSV = Path(f"../../result/semantic_gate1/{DATASET}/k_{K_SHOT}/csv/{DATASET}.csv")
OUTPUT_DIR = Path(f"../../result/semantic_gate1/{DATASET}/k_{K_SHOT}/analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# VisA classes
VISA_CLASSES = [
    "candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1",
    "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum"
]


def load_results(csv_path):
    """Load CSV results and extract relevant metrics."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Results CSV not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Extract relevant columns
    results = {}
    for _, row in df.iterrows():
        class_name = row['class_name']
        results[class_name] = {
            'semantic_i_roc': row.get('semantic_i_roc', np.nan),
            'memory_i_roc': row.get('memory_i_roc', np.nan),
            'fusion_i_roc': row.get('fusion_i_roc', np.nan),
        }
    
    return results


def load_gap_meta(class_name):
    """Load gap meta information for a class."""
    meta_path = Path(f"../../result/semantic_gate1/{DATASET}/gap_analysis/{class_name}/{class_name}_gap_meta.json")
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            return json.load(f)
    return {}


def compare_results():
    """Compare baseline vs gated results."""
    print("=" * 80)
    print("Gap-Based Gating Comparison - VisA Dataset")
    print("=" * 80)
    print()
    
    # Load results
    print("Loading baseline results...")
    baseline = load_results(BASELINE_CSV)
    
    print("Loading gap-gated results...")
    gated = load_results(GATED_CSV)
    
    # Compare class by class
    comparison = []
    
    for class_name in VISA_CLASSES:
        if class_name not in baseline:
            print(f"[WARNING] {class_name} not found in baseline results, skipping...")
            continue
        if class_name not in gated:
            print(f"[WARNING] {class_name} not found in gated results, skipping...")
            continue
        
        b = baseline[class_name]
        g = gated[class_name]
        
        # Load gap meta
        meta = load_gap_meta(class_name)
        
        row = {
            'class_name': class_name,
            'semantic_AUROC': b['semantic_i_roc'],
            'memory_AUROC': b['memory_i_roc'],
            'baseline_fusion_AUROC': b['fusion_i_roc'],
            'gated_fusion_AUROC': g['fusion_i_roc'],
            'delta_AUROC': g['fusion_i_roc'] - b['fusion_i_roc'],
            't_gap': meta.get('t_gap', np.nan),
            'suppression_rate': meta.get('suppression_rate', np.nan) * 100,  # Convert to %
        }
        
        comparison.append(row)
    
    df_comp = pd.DataFrame(comparison)
    
    # Sort by delta AUROC (descending)
    df_comp = df_comp.sort_values('delta_AUROC', ascending=False)
    
    # Print results
    print()
    print("=" * 80)
    print("Individual Class Results")
    print("=" * 80)
    print()
    
    print(f"{'Class':<15} | {'Baseline':<8} | {'Gated':<8} | {'Δ AUROC':<10} | {'Suppression%':<12}")
    print("-" * 80)
    
    for _, row in df_comp.iterrows():
        delta_str = f"{row['delta_AUROC']:+.4f}"
        print(f"{row['class_name']:<15} | {row['baseline_fusion_AUROC']:>7.4f} | "
              f"{row['gated_fusion_AUROC']:>7.4f} | {delta_str:<10} | {row['suppression_rate']:>10.1f}%")
    
    # Summary statistics
    print()
    print("=" * 80)
    print("Summary Statistics")
    print("=" * 80)
    print()
    
    n_total = len(df_comp)
    n_improved = (df_comp['delta_AUROC'] > 0).sum()
    n_neutral = (df_comp['delta_AUROC'].abs() < 0.001).sum()
    n_degraded = (df_comp['delta_AUROC'] < -0.001).sum()
    
    mean_baseline = df_comp['baseline_fusion_AUROC'].mean()
    mean_gated = df_comp['gated_fusion_AUROC'].mean()
    mean_delta = df_comp['delta_AUROC'].mean()
    
    print(f"Total classes:       {n_total}")
    print(f"  Improved:          {n_improved} ({n_improved/n_total*100:.1f}%)")
    print(f"  Neutral (|Δ|<0.1%): {n_neutral} ({n_neutral/n_total*100:.1f}%)")
    print(f"  Degraded:          {n_degraded} ({n_degraded/n_total*100:.1f}%)")
    print()
    print(f"Mean Baseline AUROC: {mean_baseline:.4f}")
    print(f"Mean Gated AUROC:    {mean_gated:.4f}")
    print(f"Mean Δ AUROC:        {mean_delta:+.4f}")
    print()
    print(f"Max improvement:     {df_comp['delta_AUROC'].max():+.4f} ({df_comp.loc[df_comp['delta_AUROC'].idxmax(), 'class_name']})")
    print(f"Max degradation:     {df_comp['delta_AUROC'].min():+.4f} ({df_comp.loc[df_comp['delta_AUROC'].idxmin(), 'class_name']})")
    print()
    
    # Save comparison CSV
    output_csv = OUTPUT_DIR / "gap_gate_comparison.csv"
    df_comp.to_csv(output_csv, index=False)
    print(f"[INFO] Comparison CSV saved to: {output_csv}")
    
    # Visualization
    visualize_comparison(df_comp)
    
    return df_comp


def visualize_comparison(df):
    """Generate comparison visualizations."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Gap-Based Gating Results - {DATASET.upper()} Dataset', fontsize=14, fontweight='bold')
    
    # 1. Delta AUROC bar plot
    ax1 = axes[0, 0]
    colors = ['green' if x > 0 else 'red' if x < -0.001 else 'gray' for x in df['delta_AUROC']]
    ax1.barh(df['class_name'], df['delta_AUROC'], color=colors, alpha=0.7)
    ax1.axvline(0, color='black', linestyle='--', linewidth=1)
    ax1.set_xlabel('Δ AUROC (Gated - Baseline)')
    ax1.set_title('Per-Class Performance Change')
    ax1.grid(axis='x', alpha=0.3)
    
    # 2. Baseline vs Gated scatter
    ax2 = axes[0, 1]
    ax2.scatter(df['baseline_fusion_AUROC'], df['gated_fusion_AUROC'], 
                c=df['delta_AUROC'], cmap='RdYlGn', s=100, alpha=0.7, edgecolors='black')
    
    # Diagonal line
    min_val = min(df['baseline_fusion_AUROC'].min(), df['gated_fusion_AUROC'].min())
    max_val = max(df['baseline_fusion_AUROC'].max(), df['gated_fusion_AUROC'].max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='No change')
    
    ax2.set_xlabel('Baseline Fusion AUROC')
    ax2.set_ylabel('Gated Fusion AUROC')
    ax2.set_title('Baseline vs Gated Performance')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(ax2.collections[0], ax=ax2)
    cbar.set_label('Δ AUROC', rotation=270, labelpad=15)
    
    # 3. Suppression rate vs Delta AUROC
    ax3 = axes[1, 0]
    ax3.scatter(df['suppression_rate'], df['delta_AUROC'], s=100, alpha=0.7, edgecolors='black')
    
    # Add class labels for outliers
    for _, row in df.iterrows():
        if abs(row['delta_AUROC']) > 0.005:  # Label significant changes
            ax3.annotate(row['class_name'], 
                        (row['suppression_rate'], row['delta_AUROC']),
                        fontsize=8, alpha=0.7)
    
    ax3.axhline(0, color='black', linestyle='--', linewidth=1)
    ax3.set_xlabel('Suppression Rate (%)')
    ax3.set_ylabel('Δ AUROC')
    ax3.set_title('Suppression Rate vs Performance Change')
    ax3.grid(alpha=0.3)
    
    # 4. AUROC distribution comparison
    ax4 = axes[1, 1]
    
    baseline_aurocs = df['baseline_fusion_AUROC'].values
    gated_aurocs = df['gated_fusion_AUROC'].values
    
    x = np.arange(len(df))
    width = 0.35
    
    ax4.bar(x - width/2, baseline_aurocs, width, label='Baseline', alpha=0.7)
    ax4.bar(x + width/2, gated_aurocs, width, label='Gated', alpha=0.7)
    
    ax4.set_xlabel('Class Index')
    ax4.set_ylabel('AUROC')
    ax4.set_title('AUROC Comparison by Class')
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_png = OUTPUT_DIR / "gap_gate_comparison.png"
    plt.savefig(output_png, dpi=150, bbox_inches='tight')
    print(f"[INFO] Visualization saved to: {output_png}")
    
    plt.close()


if __name__ == "__main__":
    try:
        df_comparison = compare_results()
        
        print()
        print("=" * 80)
        print("Analysis Complete!")
        print("=" * 80)
        print()
        
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
