"""
Phase 2.1: Oracle-Indicator Correlation Analysis (ANALYSIS ONLY)

Goal: Measure correlation between oracle selection and reliability indicators.
NOT to build gating, fusion, or optimize any parameters.

Research question: Are oracle decisions correlated with branch reliability indicators?

Success criteria: Even AUC ~0.6-0.7 is meaningful, showing oracle is partially predictable.
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


def load_oracle_and_indicators(result_dir, dataset, k_shot, task, class_name):
    """
    Load oracle selection and reliability indicators for a class.
    
    Returns:
        dict with:
            - oracle_choices: (N,) array of 0/1 (0=semantic, 1=memory)
            - indicators: dict of (N,) arrays
            - metadata: dict with statistics
    """
    # For now, use oracle selection ratios from gate results
    # In future, will load per-sample indicators from metadata
    result_path = Path(result_dir) / dataset / f"k_{k_shot}" / "gate_results" / f"{class_name}_seed111_{task}.json"
    
    if not result_path.exists():
        return None
    
    with open(result_path, 'r') as f:
        results = json.load(f)
    
    oracle_data = results.get('oracle', {})
    if not oracle_data:
        return None
    
    # Extract statistics
    semantic_ratio = oracle_data.get('oracle_semantic_ratio', 50.0) / 100.0
    memory_ratio = oracle_data.get('oracle_memory_ratio', 50.0) / 100.0
    
    # NOTE: Currently we only have aggregated statistics, not per-sample data
    # This is a placeholder for when per-sample indicators are available
    return {
        'semantic_ratio': semantic_ratio,
        'memory_ratio': memory_ratio,
        'semantic_auroc': oracle_data.get('i_roc_semantic', 0.0),
        'memory_auroc': oracle_data.get('i_roc_memory', 0.0),
        'oracle_auroc': oracle_data.get('i_roc', 0.0),
        'oracle_gain': oracle_data.get('i_roc', 0.0) - max(
            oracle_data.get('i_roc_semantic', 0.0),
            oracle_data.get('i_roc_memory', 0.0)
        )
    }


def analyze_oracle_selection_summary(results_dict, output_dir):
    """
    Task 1: Dataset-level oracle selection summary.
    
    Compute and save:
    - Overall oracle selection ratios
    - Per-class oracle selection ratios
    """
    print("\n" + "="*80)
    print("Task 1: Oracle Selection Summary")
    print("="*80)
    
    summary_data = []
    
    for key, data in results_dict.items():
        dataset, k_shot, task, class_name = key
        if data is None:
            continue
        
        summary_data.append({
            'dataset': dataset,
            'k_shot': k_shot,
            'task': task,
            'class': class_name,
            'semantic_ratio': data['semantic_ratio'],
            'memory_ratio': data['memory_ratio'],
            'semantic_auroc': data['semantic_auroc'],
            'memory_auroc': data['memory_auroc'],
            'oracle_auroc': data['oracle_auroc'],
            'oracle_gain': data['oracle_gain']
        })
    
    df = pd.DataFrame(summary_data)
    
    # Save to CSV
    output_file = Path(output_dir) / "oracle_selection_summary.csv"
    df.to_csv(output_file, index=False)
    print(f"✅ Saved oracle selection summary: {output_file}")
    
    # Print statistics
    print(f"\nOverall Statistics:")
    print(f"  Datasets: {df['dataset'].unique()}")
    print(f"  Total classes: {len(df)}")
    print(f"  Semantic ratio: {df['semantic_ratio'].mean():.3f} ± {df['semantic_ratio'].std():.3f}")
    print(f"  Oracle gain: {df['oracle_gain'].mean():.2f}% ± {df['oracle_gain'].std():.2f}%")
    
    # Plot histogram
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Subplot 1: Oracle selection distribution
    axes[0].hist(df['semantic_ratio'], bins=20, alpha=0.7, color='steelblue', edgecolor='black')
    axes[0].axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Random (50%)')
    axes[0].set_xlabel('Semantic Selection Ratio', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Distribution of Oracle Semantic Selection', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    # Subplot 2: Oracle gain distribution
    axes[1].hist(df['oracle_gain'], bins=20, alpha=0.7, color='coral', edgecolor='black')
    axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2, label='No gain')
    axes[1].set_xlabel('Oracle Gain (%)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Distribution of Oracle Gain over Best Branch', fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plot_file = Path(output_dir) / "oracle_selection_histogram.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"📈 Saved histogram: {plot_file}")
    plt.close()
    
    return df


def analyze_indicator_distribution(indicators_data, oracle_choices, indicator_name, output_dir, suffix=""):
    """
    Task 2: Indicator distribution analysis.
    
    Plot distribution of indicator values for:
    - Group A: oracle selects semantic
    - Group B: oracle selects memory
    """
    # Split by oracle choice
    semantic_mask = oracle_choices == 0
    memory_mask = oracle_choices == 1
    
    group_a = indicators_data[semantic_mask]
    group_b = indicators_data[memory_mask]
    
    # Compute statistics
    mean_a, mean_b = group_a.mean(), group_b.mean()
    median_a, median_b = np.median(group_a), np.median(group_b)
    effect_size = mean_a - mean_b
    
    # Statistical test
    t_stat, p_value = stats.ttest_ind(group_a, group_b)
    
    print(f"\n{indicator_name}{suffix}:")
    print(f"  Group A (semantic): mean={mean_a:.3f}, median={median_a:.3f}, n={len(group_a)}")
    print(f"  Group B (memory):   mean={mean_b:.3f}, median={median_b:.3f}, n={len(group_b)}")
    print(f"  Effect size: {effect_size:.3f}")
    print(f"  t-test: t={t_stat:.3f}, p={p_value:.4f}")
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    positions = [1, 2]
    data_to_plot = [group_a, group_b]
    
    bp = ax.boxplot(data_to_plot, positions=positions, widths=0.6,
                    patch_artist=True, showmeans=True,
                    boxprops=dict(facecolor='lightblue', alpha=0.7),
                    medianprops=dict(color='red', linewidth=2),
                    meanprops=dict(marker='D', markerfacecolor='green', markersize=8))
    
    # Add violin plot overlay
    parts = ax.violinplot(data_to_plot, positions=positions, widths=0.8,
                          showmeans=False, showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor('steelblue')
        pc.set_alpha(0.3)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(['Oracle → Semantic', 'Oracle → Memory'], fontsize=11)
    ax.set_ylabel(f'{indicator_name} Value', fontsize=12)
    ax.set_title(f'Distribution of {indicator_name} by Oracle Choice{suffix}', 
                fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add effect size annotation
    ax.text(0.5, 0.95, f'Effect size: {effect_size:.3f}\np-value: {p_value:.4f}',
           transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plot_file = Path(output_dir) / f"indicator_{indicator_name}_distribution{suffix}.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    return {
        'indicator': indicator_name,
        'mean_semantic': mean_a,
        'mean_memory': mean_b,
        'effect_size': effect_size,
        'p_value': p_value
    }


def compute_indicator_auc(indicators_data, oracle_choices, indicator_name):
    """
    Task 3: Oracle predictability (binary classification test).
    
    Compute ROC-AUC using indicator as prediction score.
    
    Returns:
        AUC score (higher = better predictability)
    """
    # Oracle choice as binary label: 0=semantic, 1=memory
    # Indicator as prediction score
    
    try:
        auc = roc_auc_score(oracle_choices, indicators_data)
        return auc
    except Exception as e:
        print(f"⚠️  Failed to compute AUC for {indicator_name}: {e}")
        return np.nan


def analyze_hard_cases(results_dict, hard_classes, output_dir):
    """
    Task 4: Hard-case subset analysis.
    
    Compare indicator performance on full dataset vs hard cases.
    """
    print("\n" + "="*80)
    print("Task 4: Hard-Case Subset Analysis")
    print("="*80)
    
    hard_data = []
    
    for key, data in results_dict.items():
        dataset, k_shot, task, class_name = key
        if data is None:
            continue
        
        if class_name in hard_classes.get(dataset, []):
            hard_data.append({
                'dataset': dataset,
                'class': class_name,
                'semantic_ratio': data['semantic_ratio'],
                'oracle_gain': data['oracle_gain']
            })
    
    if hard_data:
        df_hard = pd.DataFrame(hard_data)
        print(f"\nHard cases identified: {len(df_hard)}")
        print(df_hard[['dataset', 'class', 'semantic_ratio', 'oracle_gain']].to_string(index=False))
        
        output_file = Path(output_dir) / "hard_cases_summary.csv"
        df_hard.to_csv(output_file, index=False)
        print(f"✅ Saved hard cases summary: {output_file}")
    else:
        print("⚠️  No hard cases found in current data")


def main():
    parser = argparse.ArgumentParser(description="Phase 2.1: Oracle-Indicator Correlation Analysis")
    parser.add_argument('--result_dir', type=str, default='result/gate',
                       help='Gate experiment result directory')
    parser.add_argument('--output_dir', type=str, default='result/gate2/phase2_1_analysis',
                       help='Output directory for analysis')
    parser.add_argument('--datasets', nargs='+', default=['mvtec', 'visa'],
                       help='Datasets to analyze')
    parser.add_argument('--k_shots', nargs='+', type=int, default=[4],
                       help='K-shot values to analyze')
    parser.add_argument('--task', type=str, default='cls',
                       choices=['cls', 'seg'],
                       help='Task type')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define class lists
    class_lists = {
        'mvtec': ['bottle', 'cable', 'capsule', 'carpet', 'grid', 
                 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
                 'tile', 'toothbrush', 'transistor', 'wood', 'zipper'],
        'visa': ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
                'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3',
                'pcb4', 'pipe_fryum']
    }
    
    # Define hard cases (from gate1 analysis)
    hard_classes = {
        'mvtec': ['screw', 'capsule'],
        'visa': ['capsules', 'macaroni2', 'pcb2']  # Classes with large oracle gain
    }
    
    print("="*80)
    print("Phase 2.1: Oracle-Indicator Correlation Analysis")
    print("="*80)
    print(f"Result directory: {args.result_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Datasets: {args.datasets}")
    print(f"K-shots: {args.k_shots}")
    print(f"Task: {args.task}")
    
    # Load all results
    results_dict = {}
    
    for dataset in args.datasets:
        for k_shot in args.k_shots:
            for class_name in class_lists.get(dataset, []):
                data = load_oracle_and_indicators(
                    args.result_dir, dataset, k_shot, args.task, class_name
                )
                if data is not None:
                    key = (dataset, k_shot, args.task, class_name)
                    results_dict[key] = data
    
    print(f"\n✅ Loaded data for {len(results_dict)} class configurations")
    
    # Task 1: Oracle Selection Summary
    df_summary = analyze_oracle_selection_summary(results_dict, args.output_dir)
    
    # Task 4: Hard-case analysis
    analyze_hard_cases(results_dict, hard_classes, args.output_dir)
    
    # NOTE: Tasks 2 & 3 require per-sample indicator data
    # Currently we only have aggregated statistics
    print("\n" + "="*80)
    print("⚠️  IMPORTANT NOTE")
    print("="*80)
    print("Tasks 2 & 3 (indicator distribution and AUC analysis) require per-sample data.")
    print("Current implementation only loads aggregated statistics from gate results.")
    print("\nNext steps:")
    print("1. Modify run_gate_experiment.py to save per-sample indicators in metadata")
    print("2. Re-run gate experiments with indicator computation")
    print("3. Update this script to load per-sample data")
    print("4. Then Tasks 2 & 3 can be completed")
    
    print("\n" + "="*80)
    print("Phase 2.1 Analysis Complete (Partial)")
    print("="*80)
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
