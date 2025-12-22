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


def load_oracle_and_indicators(result_dir, dataset, k_shot, task, class_name, use_real_data=False):
    """
    Load oracle selection and reliability indicators for a class.
    
    Args:
        use_real_data: If True, load per-sample data from per_sample/ directory
                       If False, use aggregated statistics from gate_results/
    
    Returns:
        dict with:
            - oracle_choices: (N,) array of 0/1 (0=semantic, 1=memory)
            - indicators: dict of (N,) arrays
            - metadata: dict with statistics
    """
    # Try loading per-sample data first (if use_real_data=True)
    if use_real_data:
        per_sample_path = Path(result_dir) / dataset / f"k_{k_shot}" / "per_sample" / task / f"{class_name}_seed111_per_sample.json"
        
        if per_sample_path.exists():
            with open(per_sample_path, 'r') as f:
                per_sample_json = json.load(f)
            
            per_sample_data = per_sample_json['per_sample_data']
            
            # Extract arrays
            oracle_choices = np.array([s['oracle_choice'] for s in per_sample_data])
            
            return {
                'oracle_choices': oracle_choices,
                'indicators': {
                    'r_mem_margin': np.array([s['r_mem_margin'] for s in per_sample_data]),
                    'r_mem_entropy': np.array([s['r_mem_entropy'] for s in per_sample_data]),
                    'r_mem_centroid': np.array([s['r_mem_centroid'] for s in per_sample_data]),
                    'r_sem_prompt_var': np.array([s['r_sem_prompt_var'] for s in per_sample_data]),
                    'r_sem_prompt_margin': np.array([s['r_sem_prompt_margin'] for s in per_sample_data]),
                    'r_sem_extremity': np.array([s['r_sem_extremity'] for s in per_sample_data])
                },
                'semantic_ratio': 1 - oracle_choices.mean(),
                'has_per_sample_data': True
            }
    
    # Fallback to aggregated statistics
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
        ),
        'has_per_sample_data': False
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
    parser.add_argument('--use_real_data', action='store_true',
                       help='Use real per-sample data from per_sample/ directory')
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
    print(f"Use real data: {args.use_real_data}")
    
    # Load all results
    results_dict = {}
    has_per_sample_count = 0
    
    for dataset in args.datasets:
        for k_shot in args.k_shots:
            for class_name in class_lists.get(dataset, []):
                data = load_oracle_and_indicators(
                    args.result_dir, dataset, k_shot, args.task, class_name,
                    use_real_data=args.use_real_data
                )
                if data is not None:
                    key = (dataset, k_shot, args.task, class_name)
                    results_dict[key] = data
                    if data.get('has_per_sample_data', False):
                        has_per_sample_count += 1
    
    print(f"\n✅ Loaded data for {len(results_dict)} class configurations")
    print(f"   Classes with per-sample data: {has_per_sample_count}")
    
    # Task 1: Oracle Selection Summary
    df_summary = analyze_oracle_selection_summary(results_dict, args.output_dir)
    
    # Task 4: Hard-case analysis
    analyze_hard_cases(results_dict, hard_classes, args.output_dir)
    
    # Tasks 2 & 3: Indicator analysis (only if we have per-sample data)
    if has_per_sample_count > 0:
        print("\n" + "="*80)
        print("Tasks 2 & 3: Per-Sample Indicator Analysis")
        print("="*80)
        print(f"Analyzing {has_per_sample_count} classes with per-sample data...\n")
        
        all_auc_results = []
        
        for key, data in results_dict.items():
            if not data.get('has_per_sample_data', False):
                continue
            
            dataset, k_shot, task, class_name = key
            print(f"\n{'='*60}")
            print(f"{dataset}/{class_name} (k={k_shot})")
            print(f"{'='*60}")
            
            oracle_choices = data['oracle_choices']
            indicators = data['indicators']
            
            print(f"  Samples: {len(oracle_choices)}")
            print(f"  Semantic ratio: {(oracle_choices == 0).mean()*100:.1f}%")
            
            # Task 2: Distribution analysis
            print("\n  📊 Task 2: Indicator Distribution")
            for name, values in indicators.items():
                sem_mask = oracle_choices == 0
                mem_mask = oracle_choices == 1
                sem_vals = values[sem_mask]
                mem_vals = values[mem_mask]
                
                if len(sem_vals) > 0 and len(mem_vals) > 0:
                    pooled_std = np.sqrt((sem_vals.std()**2 + mem_vals.std()**2) / 2)
                    cohens_d = (mem_vals.mean() - sem_vals.mean()) / (pooled_std + 1e-8)
                    t_stat, p_value = stats.ttest_ind(sem_vals, mem_vals)
                    
                    sig = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
                    print(f"    {name:20s}: d={cohens_d:+.3f}, p={p_value:.4f} {sig}")
            
            # Task 3: AUC analysis
            print("\n  📊 Task 3: Oracle Predictability (AUC)")
            auc_scores = {}
            
            for name, values in indicators.items():
                # Memory indicators: high -> memory (1)
                # Semantic indicators: high -> semantic (0), invert
                if name.startswith('r_sem'):
                    y_score = -values
                else:
                    y_score = values
                
                try:
                    if len(np.unique(oracle_choices)) > 1 and len(np.unique(y_score)) > 1:
                        auc = roc_auc_score(oracle_choices, y_score)
                        auc_scores[name] = auc
                        print(f"    {name:20s}: AUC = {auc:.3f}")
                    else:
                        auc_scores[name] = 0.5
                        print(f"    {name:20s}: AUC = 0.500 (uniform)")
                except Exception as e:
                    auc_scores[name] = 0.5
                    print(f"    {name:20s}: AUC = 0.500 (error)")
            
            # Average AUCs
            mem_auc = np.mean([auc_scores.get(f'r_mem_{x}', 0.5) for x in ['margin', 'entropy', 'centroid']])
            sem_auc = np.mean([auc_scores.get(f'r_sem_{x}', 0.5) for x in ['prompt_var', 'prompt_margin', 'extremity']])
            
            print(f"\n    Memory avg AUC:   {mem_auc:.3f}")
            print(f"    Semantic avg AUC: {sem_auc:.3f}")
            print(f"    Overall avg AUC:  {(mem_auc + sem_auc)/2:.3f}")
            
            auc_results = {
                'dataset': dataset,
                'class_name': class_name,
                'k_shot': k_shot,
                **auc_scores,
                'memory_avg_auc': mem_auc,
                'semantic_avg_auc': sem_auc,
                'overall_avg_auc': (mem_auc + sem_auc) / 2
            }
            all_auc_results.append(auc_results)
        
        # Save AUC results
        if all_auc_results:
            auc_df = pd.DataFrame(all_auc_results)
            auc_file = Path(args.output_dir) / 'indicator_auc_results.csv'
            auc_df.to_csv(auc_file, index=False)
            print(f"\n✅ Saved AUC results: {auc_file}")
            
            # Summary
            print("\n" + "="*80)
            print("SUMMARY: Indicator AUC Performance")
            print("="*80)
            print(f"Average Memory indicator AUC:   {auc_df['memory_avg_auc'].mean():.3f}")
            print(f"Average Semantic indicator AUC: {auc_df['semantic_avg_auc'].mean():.3f}")
            print(f"Overall average AUC:            {auc_df['overall_avg_auc'].mean():.3f}")
            
            avg_auc = auc_df['overall_avg_auc'].mean()
            print("\n💡 Interpretation:")
            if avg_auc > 0.60:
                print("✅ Hypothesis validated! Indicators correlate with oracle choices.")
                print("   Proceed to Phase 2.2: Design adaptive gating mechanism.")
            elif avg_auc > 0.55:
                print("⚠️  Weak correlation detected. Consider:")
                print("   - Adding more indicators")
                print("   - Improving normalization")
                print("   - Class-specific calibration")
            else:
                print("❌ No significant correlation. Consider:")
                print("   - Re-examining indicator design")
                print("   - Meta-learning approach")
                print("   - Alternative gating strategies")
    else:
        print("\n" + "="*80)
        print("⚠️  TASKS 2 & 3 SKIPPED")
        print("="*80)
        print("No per-sample data available.")
        print("\nTo generate per-sample data:")
        print("1. Run: ./run_phase2_1_real.sh")
        print("2. Then re-run this analysis with --use_real_data flag")
    
    print("\n" + "="*80)
    print("Phase 2.1 Analysis Complete")
    print("="*80)
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
