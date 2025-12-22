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
    Analyze oracle selection pattern from aggregated statistics.
    
    Note: Gate results only contain aggregated stats, not per-sample data.
    We can only analyze oracle_semantic_ratio here.
    
    Returns:
        analysis_dict with oracle selection statistics
    """
    if results_dict is None:
        return None
    
    # Extract oracle statistics
    oracle_data = results_dict.get('oracle', {})
    
    if not oracle_data:
        return None
    
    semantic_ratio = oracle_data.get('oracle_semantic_ratio', 50.0)
    memory_ratio = oracle_data.get('oracle_memory_ratio', 50.0)
    
    # Also get individual branch AUROCs for context
    semantic_auroc = oracle_data.get('i_roc_semantic', 0.0)
    memory_auroc = oracle_data.get('i_roc_memory', 0.0)
    oracle_auroc = oracle_data.get('i_roc', 0.0)
    
    analysis = {
        'semantic_chosen_ratio': semantic_ratio / 100.0,
        'memory_chosen_ratio': memory_ratio / 100.0,
        'semantic_auroc': semantic_auroc,
        'memory_auroc': memory_auroc,
        'oracle_auroc': oracle_auroc,
        'oracle_gain': oracle_auroc - max(semantic_auroc, memory_auroc),
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
              f"Semantic: {analysis['semantic_chosen_ratio']:.1%} | "
              f"Memory: {analysis['memory_chosen_ratio']:.1%} | "
              f"AUROC - Sem: {analysis['semantic_auroc']:.1f} | Mem: {analysis['memory_auroc']:.1f} | Oracle: {analysis['oracle_auroc']:.1f} (+{analysis['oracle_gain']:.2f})")
        
        summary_stats.append({
            'class': cls_name,
            'semantic_ratio': analysis['semantic_chosen_ratio'],
            'memory_ratio': analysis['memory_chosen_ratio'],
            'semantic_auroc': analysis['semantic_auroc'],
            'memory_auroc': analysis['memory_auroc'],
            'oracle_auroc': analysis['oracle_auroc'],
            'oracle_gain': analysis['oracle_gain']
        })
    
    # Overall statistics
    print(f"\n{'='*80}")
    print("Overall Statistics:")
    print(f"{'='*80}")
    
    semantic_ratios = [s['semantic_ratio'] for s in summary_stats]
    oracle_gains = [s['oracle_gain'] for s in summary_stats]
    print(f"Semantic chosen (avg across classes): {np.mean(semantic_ratios):.1%} ± {np.std(semantic_ratios):.1%}")
    print(f"Range: [{np.min(semantic_ratios):.1%}, {np.max(semantic_ratios):.1%}]")
    print(f"\nOracle gain over best single branch:")
    print(f"  Average: {np.mean(oracle_gains):.2f}%")
    print(f"  Range: [{np.min(oracle_gains):.2f}, {np.max(oracle_gains):.2f}]%")
    
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
    
    # Plot 1: Oracle selection ratio + AUROC comparison
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Subplot 1: Bar plot of semantic vs memory selection
    classes = [s['class'] for s in summary_stats]
    semantic_ratios = [s['semantic_ratio'] for s in summary_stats]
    memory_ratios = [s['memory_ratio'] for s in summary_stats]
    
    x = np.arange(len(classes))
    width = 0.35
    
    axes[0].bar(x - width/2, semantic_ratios, width, label='Semantic', color='steelblue', alpha=0.8)
    axes[0].bar(x + width/2, memory_ratios, width, label='Memory', color='coral', alpha=0.8)
    axes[0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    axes[0].set_ylabel('Selection Ratio', fontsize=11)
    axes[0].set_title(f'Oracle Branch Selection - {args.dataset.upper()} K={args.k_shot} {args.task.upper()}', 
                     fontsize=13, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(classes, rotation=45, ha='right', fontsize=9)
    axes[0].legend(fontsize=10)
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].set_ylim([0, 1])
    
    # Subplot 2: AUROC comparison (Semantic vs Memory vs Oracle)
    semantic_aurocs = [s['semantic_auroc'] for s in summary_stats]
    memory_aurocs = [s['memory_auroc'] for s in summary_stats]
    oracle_aurocs = [s['oracle_auroc'] for s in summary_stats]
    
    width = 0.25
    axes[1].bar(x - width, semantic_aurocs, width, label='Semantic', color='steelblue', alpha=0.8)
    axes[1].bar(x, memory_aurocs, width, label='Memory', color='coral', alpha=0.8)
    axes[1].bar(x + width, oracle_aurocs, width, label='Oracle', color='gold', alpha=0.8)
    
    axes[1].set_ylabel('AUROC (%)', fontsize=11)
    axes[1].set_xlabel('Class', fontsize=11)
    axes[1].set_title('Branch Performance Comparison', fontsize=13, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(classes, rotation=45, ha='right', fontsize=9)
    axes[1].legend(fontsize=10)
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].set_ylim([50, 105])
    
    plt.tight_layout()
    plot_file = output_dir / f"oracle_selection_{args.dataset}_k{args.k_shot}_{args.task}.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"📈 Visualization saved to: {plot_file}")
    plt.close()


if __name__ == "__main__":
    main()
