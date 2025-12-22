"""
Phase 1: Oracle Selection Pattern Analysis (ANALYSIS MODE ONLY)

Goal: Understand WHEN oracle chooses semantic vs memory branch.
NOT to build a gating mechanism yet.

Output:
- Oracle selection statistics per class
- Score difference distributions
- Preparation for Phase 2: correlation with reliability indicators

Explicitly NOT doing:
- Computing reliability indicators (needs model integration first)
- Fusing scores with adaptive weights
- Evaluating any "method" performance
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict


def load_gate_results(result_dir, dataset, k_shot, task, class_name):
    """Load gate experiment results including oracle metadata."""
    result_path = Path(result_dir) / dataset / f"k_{k_shot}" / "gate_results" / f"{class_name}_seed111_{task}.json"
    
    if not result_path.exists():
        return None
    
    with open(result_path, 'r') as f:
        return json.load(f)


def analyze_oracle_pattern(results_dict):
    """
    Analyze oracle selection pattern.
    
    Returns:
        analysis_dict with statistics about oracle choices
    """
    if results_dict is None:
        return None
    
    # Extract scores
    semantic_scores = np.array(results_dict['semantic_scores'])
    memory_scores = np.array(results_dict['memory_scores'])
    oracle_scores = np.array(results_dict['oracle_scores'])
    gt_labels = np.array(results_dict['gt_labels'])
    
    N = len(gt_labels)
    
    # Infer oracle selection (which branch was chosen)
    oracle_selections = np.zeros(N, dtype=int)  # 0=semantic, 1=memory
    
    for i in range(N):
        if gt_labels[i] == 1:  # anomaly
            # Oracle picks higher score
            oracle_selections[i] = 0 if semantic_scores[i] > memory_scores[i] else 1
        else:  # normal
            # Oracle picks lower score
            oracle_selections[i] = 0 if semantic_scores[i] < memory_scores[i] else 1
    
    # Compute statistics
    semantic_chosen_ratio = (oracle_selections == 0).mean()
    
    # Analyze score differences
    score_diff = semantic_scores - memory_scores  # positive = semantic higher
    
    # Separate by GT label
    normal_mask = gt_labels == 0
    anomaly_mask = gt_labels == 1
    
    analysis = {
        'n_samples': N,
        'n_normal': normal_mask.sum(),
        'n_anomaly': anomaly_mask.sum(),
        'semantic_chosen_ratio': semantic_chosen_ratio,
        'memory_chosen_ratio': 1 - semantic_chosen_ratio,
        'oracle_selections': oracle_selections,
        'score_diff': score_diff,
        'semantic_scores': semantic_scores,
        'memory_scores': memory_scores,
        'gt_labels': gt_labels,
        
        # Statistics by GT label
        'normal': {
            'semantic_chosen_ratio': (oracle_selections[normal_mask] == 0).mean() if normal_mask.sum() > 0 else 0,
            'score_diff_mean': score_diff[normal_mask].mean() if normal_mask.sum() > 0 else 0,
            'score_diff_std': score_diff[normal_mask].std() if normal_mask.sum() > 0 else 0,
        },
        'anomaly': {
            'semantic_chosen_ratio': (oracle_selections[anomaly_mask] == 0).mean() if anomaly_mask.sum() > 0 else 0,
            'score_diff_mean': score_diff[anomaly_mask].mean() if anomaly_mask.sum() > 0 else 0,
            'score_diff_std': score_diff[anomaly_mask].std() if anomaly_mask.sum() > 0 else 0,
        }
    }
    
    return analysis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', type=str, default='result/gate',
                       help='Gate experiment result directory')
    parser.add_argument('--dataset', type=str, default='mvtec',
                       choices=['mvtec', 'visa'])
    parser.add_argument('--k_shot', type=int, default=4,
                       choices=[1, 2, 4])
    parser.add_argument('--task', type=str, default='cls',
                       choices=['cls', 'seg'])
    parser.add_argument('--output_dir', type=str, default='result/gate/analysis',
                       help='Output directory for analysis')
    args = parser.parse_args()
    
    # Get class list
    if args.dataset == 'mvtec':
        classes = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 
                  'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
                  'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
    else:
        classes = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
                  'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3',
                  'pcb4', 'pipe_fryum']
    
    # Collect analyses
    all_analyses = {}
    summary_stats = []
    
    print(f"\n{'='*80}")
    print(f"Analyzing Oracle Selection Patterns")
    print(f"Dataset: {args.dataset} | K-shot: {args.k_shot} | Task: {args.task}")
    print(f"{'='*80}\n")
    
    for cls_name in classes:
        results = load_gate_results(
            args.result_dir, args.dataset, args.k_shot, args.task, cls_name
        )
        
        if results is None:
            print(f"⚠️  {cls_name}: No results found")
            continue
        
        analysis = analyze_oracle_pattern(results)
        if analysis is None:
            continue
        
        all_analyses[cls_name] = analysis
        
        # Print summary
        print(f"📊 {cls_name:15s} | "
              f"N={analysis['n_samples']:3d} | "
              f"Semantic chosen: {analysis['semantic_chosen_ratio']:.1%} | "
              f"Score diff: {analysis['score_diff'].mean():+.3f}±{analysis['score_diff'].std():.3f}")
        
        summary_stats.append({
            'class': cls_name,
            'semantic_ratio': analysis['semantic_chosen_ratio'],
            'score_diff_mean': analysis['score_diff'].mean(),
            'score_diff_std': analysis['score_diff'].std(),
            'n_samples': analysis['n_samples']
        })
    
    # Overall statistics
    print(f"\n{'='*80}")
    print("Overall Statistics:")
    print(f"{'='*80}")
    
    semantic_ratios = [s['semantic_ratio'] for s in summary_stats]
    print(f"Semantic chosen (avg across classes): {np.mean(semantic_ratios):.1%} ± {np.std(semantic_ratios):.1%}")
    print(f"Range: [{np.min(semantic_ratios):.1%}, {np.max(semantic_ratios):.1%}]")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save detailed analysis
    output_file = Path(args.output_dir) / f"oracle_analysis_{args.dataset}_k{args.k_shot}_{args.task}.json"
    with open(output_file, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_analyses = {}
        for cls_name, analysis in all_analyses.items():
            serializable_analyses[cls_name] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in analysis.items()
            }
        json.dump(serializable_analyses, f, indent=2)
    
    print(f"\n✅ Detailed analysis saved to: {output_file}")
    
    # Generate visualization
    plot_oracle_analysis(all_analyses, summary_stats, args)
    
    print(f"\n{'='*80}")
    print("Key Insights for Phase 1:")
    print(f"{'='*80}")
    print(f"1. Oracle selection distribution: {'BALANCED' if 0.4 < np.mean(semantic_ratios) < 0.6 else 'BIASED'}")
    print(f"   - Average semantic ratio: {np.mean(semantic_ratios):.1%}")
    print(f"   - Variance across classes: {np.std(semantic_ratios):.3f}")
    print(f"2. Score difference statistics provide baseline for Phase 2")
    print(f"3. Next: Integrate reliability indicators and check correlation")
    print(f"\n⚠️  Phase 1 = ANALYSIS ONLY. No gating, no adaptive fusion yet.")
    print(f"   Purpose: Validate that reliability indicators CAN predict oracle choices")
    print(f"   before building any gating mechanism.")


def plot_oracle_analysis(all_analyses, summary_stats, args):
    """Generate visualization plots."""
    output_dir = Path(args.output_dir)
    
    # Plot 1: Oracle selection ratio per class
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Subplot 1: Bar plot of semantic vs memory selection
    classes = [s['class'] for s in summary_stats]
    semantic_ratios = [s['semantic_ratio'] for s in summary_stats]
    memory_ratios = [1 - s['semantic_ratio'] for s in summary_stats]
    
    x = np.arange(len(classes))
    width = 0.35
    
    axes[0].bar(x - width/2, semantic_ratios, width, label='Semantic', color='steelblue')
    axes[0].bar(x + width/2, memory_ratios, width, label='Memory', color='coral')
    axes[0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_ylabel('Selection Ratio')
    axes[0].set_title(f'Oracle Branch Selection - {args.dataset.upper()} K={args.k_shot} {args.task.upper()}')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(classes, rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    # Subplot 2: Score difference distribution
    score_diffs = [s['score_diff_mean'] for s in summary_stats]
    colors = ['steelblue' if d > 0 else 'coral' for d in score_diffs]
    
    axes[1].bar(x, score_diffs, color=colors, alpha=0.7)
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    axes[1].set_ylabel('Score Diff (Semantic - Memory)')
    axes[1].set_xlabel('Class')
    axes[1].set_title('Average Score Difference (positive = Semantic higher)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(classes, rotation=45, ha='right')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plot_file = output_dir / f"oracle_selection_{args.dataset}_k{args.k_shot}_{args.task}.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"📈 Visualization saved to: {plot_file}")
    plt.close()


if __name__ == "__main__":
    main()
