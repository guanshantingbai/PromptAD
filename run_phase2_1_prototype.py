"""
Phase 2.1 Prototype: Quick validation using SIMULATED data

Goal: Validate hypothesis framework before implementing full per-sample computation

Strategy:
1. Use aggregated oracle statistics from gate results
2. SIMULATE per-sample indicators with realistic distributions
3. Test if analysis framework works correctly
4. Decide whether to proceed with full implementation

Note: This is a MOCK test to validate the analysis pipeline.
Real implementation will compute actual per-sample indicators.
"""

import argparse
import numpy as np
import json
from pathlib import Path
from sklearn.metrics import roc_auc_score
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


def load_gate_statistics(result_dir, dataset, k_shot, task, class_name):
    """Load oracle statistics from gate results."""
    result_path = Path(result_dir) / dataset / f"k_{k_shot}" / "gate_results" / f"{class_name}_seed111_{task}.json"
    
    if not result_path.exists():
        return None
    
    with open(result_path, 'r') as f:
        results = json.load(f)
    
    oracle_data = results.get('oracle', {})
    if not oracle_data:
        return None
    
    return {
        'semantic_ratio': oracle_data.get('oracle_semantic_ratio', 50.0) / 100.0,
        'semantic_auroc': oracle_data.get('i_roc_semantic', 0.0),
        'memory_auroc': oracle_data.get('i_roc_memory', 0.0),
        'oracle_auroc': oracle_data.get('i_roc', 0.0),
    }


def simulate_per_sample_data(stats, n_samples=100):
    """
    Simulate per-sample indicators based on oracle statistics.
    
    This is a MOCK to test the analysis framework.
    Real implementation will compute actual indicators.
    
    Simulation strategy:
    - Oracle choice based on semantic_ratio
    - Indicators have WEAK correlation with oracle (AUC ~0.6-0.65)
    - This tests if analysis can detect even weak signals
    """
    semantic_ratio = stats['semantic_ratio']
    
    # Generate oracle choices
    oracle_choices = np.random.binomial(1, 1 - semantic_ratio, n_samples)  # 0=semantic, 1=memory
    
    # Simulate indicators with weak correlation
    # When oracle chooses memory (1), memory indicators should be slightly higher
    
    # Memory indicators (higher = more reliable)
    r_mem_margin = np.random.randn(n_samples) * 1.0
    r_mem_margin += oracle_choices * 0.5  # Small effect: +0.5 SD when memory chosen
    
    r_mem_entropy = np.random.randn(n_samples) * 1.0
    r_mem_entropy += oracle_choices * 0.4
    
    r_mem_centroid = np.random.randn(n_samples) * 1.0
    r_mem_centroid += oracle_choices * 0.3
    
    # Semantic indicators (higher when semantic chosen)
    r_sem_prompt_var = np.random.randn(n_samples) * 1.0
    r_sem_prompt_var += (1 - oracle_choices) * 0.6  # Higher when semantic chosen
    
    r_sem_prompt_margin = np.random.randn(n_samples) * 1.0
    r_sem_prompt_margin += (1 - oracle_choices) * 0.5
    
    r_sem_extremity = np.random.randn(n_samples) * 1.0
    r_sem_extremity += (1 - oracle_choices) * 0.4
    
    per_sample_data = []
    for i in range(n_samples):
        per_sample_data.append({
            'sample_id': i,
            'oracle_choice': int(oracle_choices[i]),
            'r_mem_margin': float(r_mem_margin[i]),
            'r_mem_entropy': float(r_mem_entropy[i]),
            'r_mem_centroid': float(r_mem_centroid[i]),
            'r_sem_prompt_var': float(r_sem_prompt_var[i]),
            'r_sem_prompt_margin': float(r_sem_prompt_margin[i]),
            'r_sem_extremity': float(r_sem_extremity[i])
        })
    
    return per_sample_data


def compute_indicator_distribution_analysis(per_sample_data):
    """Task 2: Analyze indicator distribution by oracle choice."""
    oracle_choices = np.array([s['oracle_choice'] for s in per_sample_data])
    
    indicators = {
        'Memory Margin': np.array([s['r_mem_margin'] for s in per_sample_data]),
        'Memory Entropy': np.array([s['r_mem_entropy'] for s in per_sample_data]),
        'Memory Centroid': np.array([s['r_mem_centroid'] for s in per_sample_data]),
        'Semantic Prompt Var': np.array([s['r_sem_prompt_var'] for s in per_sample_data]),
        'Semantic Prompt Margin': np.array([s['r_sem_prompt_margin'] for s in per_sample_data]),
        'Semantic Extremity': np.array([s['r_sem_extremity'] for s in per_sample_data])
    }
    
    results = {}
    for name, values in indicators.items():
        sem_vals = values[oracle_choices == 0]
        mem_vals = values[oracle_choices == 1]
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((sem_vals.std()**2 + mem_vals.std()**2) / 2)
        cohens_d = (mem_vals.mean() - sem_vals.mean()) / (pooled_std + 1e-8)
        
        # T-test
        t_stat, p_value = stats.ttest_ind(sem_vals, mem_vals)
        
        results[name] = {
            'sem_mean': float(sem_vals.mean()),
            'mem_mean': float(mem_vals.mean()),
            'cohens_d': float(cohens_d),
            'p_value': float(p_value)
        }
    
    return results


def compute_indicator_auc(per_sample_data):
    """Task 3: Compute AUC for each indicator."""
    from sklearn.metrics import roc_auc_score
    
    oracle_choices = np.array([s['oracle_choice'] for s in per_sample_data])
    
    indicators = {
        'r_mem_margin': np.array([s['r_mem_margin'] for s in per_sample_data]),
        'r_mem_entropy': np.array([s['r_mem_entropy'] for s in per_sample_data]),
        'r_mem_centroid': np.array([s['r_mem_centroid'] for s in per_sample_data]),
        'r_sem_prompt_var': np.array([s['r_sem_prompt_var'] for s in per_sample_data]),
        'r_sem_prompt_margin': np.array([s['r_sem_prompt_margin'] for s in per_sample_data]),
        'r_sem_extremity': np.array([s['r_sem_extremity'] for s in per_sample_data])
    }
    
    auc_scores = {}
    for name, values in indicators.items():
        # Memory indicators: high value -> memory (1)
        # Semantic indicators: high value -> semantic (0), so we invert
        if name.startswith('r_sem'):
            y_score = -values  # Invert for semantic
        else:
            y_score = values
        
        try:
            auc = roc_auc_score(oracle_choices, y_score)
            auc_scores[f'auc_{name}'] = float(auc)
        except Exception as e:
            auc_scores[f'auc_{name}'] = 0.5
    
    # Average AUC
    auc_scores['auc_memory_avg'] = float(np.mean([
        auc_scores['auc_r_mem_margin'],
        auc_scores['auc_r_mem_entropy'],
        auc_scores['auc_r_mem_centroid']
    ]))
    
    auc_scores['auc_semantic_avg'] = float(np.mean([
        auc_scores['auc_r_sem_prompt_var'],
        auc_scores['auc_r_sem_prompt_margin'],
        auc_scores['auc_r_sem_extremity']
    ]))
    
    return auc_scores


def run_prototype_analysis(selected_classes, k_shot, task, result_dir, output_dir):
    """
    Run prototype analysis on selected classes using SIMULATED data.
    
    This validates the analysis framework before implementing full computation.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    print("\n" + "=" * 80)
    print("PROTOTYPE ANALYSIS (SIMULATED DATA)")
    print("=" * 80)
    print("\n⚠️  WARNING: Using SIMULATED per-sample indicators!")
    print("This is a MOCK test to validate analysis framework.")
    print("Real implementation will compute actual indicators.\n")
    
    for dataset, class_name in selected_classes:
        print(f"\nProcessing {dataset}/{class_name} (k={k_shot})...")
        
        # Load gate statistics
        stats = load_gate_statistics(result_dir, dataset, k_shot, task, class_name)
        
        if stats is None:
            print(f"❌ Skipping {dataset}/{class_name} - gate results not found")
            continue
        
        print(f"  Oracle semantic ratio: {stats['semantic_ratio']*100:.1f}%")
        print(f"  Oracle AUROC: {stats['oracle_auroc']:.4f}")
        
        # Simulate per-sample data
        per_sample_data = simulate_per_sample_data(stats, n_samples=100)
        
        # Task 2: Distribution analysis
        print("\n  📊 Task 2: Indicator distribution")
        dist_results = compute_indicator_distribution_analysis(per_sample_data)
        
        for name, res in dist_results.items():
            print(f"    {name}:")
            print(f"      Semantic mean: {res['sem_mean']:.3f}, Memory mean: {res['mem_mean']:.3f}")
            print(f"      Cohen's d: {res['cohens_d']:.3f}, p-value: {res['p_value']:.4f}")
        
        # Task 3: AUC computation
        print("\n  📊 Task 3: Indicator AUC")
        auc_results = compute_indicator_auc(per_sample_data)
        
        for key, value in auc_results.items():
            if key.startswith('auc_'):
                print(f"    {key}: {value:.3f}")
        
        # Save results
        class_results = {
            'dataset': dataset,
            'class_name': class_name,
            'k_shot': k_shot,
            'oracle_stats': stats,
            'distribution': dist_results,
            'auc': auc_results
        }
        all_results.append(class_results)
        
        # Save per-sample JSON
        per_sample_file = output_dir / f'{dataset}_{class_name}_k{k_shot}_per_sample_SIMULATED.json'
        with open(per_sample_file, 'w') as f:
            json.dump({
                'metadata': {
                    'dataset': dataset,
                    'class_name': class_name,
                    'k_shot': k_shot,
                    'n_samples': len(per_sample_data),
                    'note': 'SIMULATED data - not real indicators'
                },
                'per_sample_data': per_sample_data
            }, f, indent=2)
    
    # Save summary
    summary_file = output_dir / 'prototype_analysis_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✅ Summary saved: {summary_file}")
    
    # Decision criterion
    print("\n" + "=" * 80)
    print("DECISION CRITERION (based on SIMULATED data)")
    print("=" * 80)
    
    avg_auc_mem = np.mean([r['auc']['auc_memory_avg'] for r in all_results])
    avg_auc_sem = np.mean([r['auc']['auc_semantic_avg'] for r in all_results])
    
    print(f"\nAverage Memory indicator AUC: {avg_auc_mem:.3f}")
    print(f"Average Semantic indicator AUC: {avg_auc_sem:.3f}")
    print(f"Overall average AUC: {(avg_auc_mem + avg_auc_sem) / 2:.3f}")
    
    print("\n💡 Interpretation:")
    print("  - Simulated AUC ~0.6-0.65: Analysis framework works correctly")
    print("  - Next step: Implement REAL per-sample indicator computation")
    print("  - If real AUC > 0.60 → proceed to full 162-task implementation")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Phase 2.1 Prototype Analysis (SIMULATED)")
    parser.add_argument('--result_dir', type=str, default='result/gate')
    parser.add_argument('--output_dir', type=str, default='result/gate2/prototype')
    args = parser.parse_args()
    
    # Selected classes for prototype
    selected_classes = [
        ('mvtec', 'screw'),      # Max oracle gain +33.82%
        ('mvtec', 'capsule'),    # Moderate gain +7.94%
        ('visa', 'capsules'),    # High gain +29.72%
        ('visa', 'macaroni2'),   # High gain +26.72%
        ('visa', 'candle'),      # Low gain +4.89%
    ]
    
    k_shot = 4
    task = 'cls'
    
    # Run prototype analysis
    run_prototype_analysis(selected_classes, k_shot, task, args.result_dir, args.output_dir)


if __name__ == '__main__':
    main()
if __name__ == "__main__":
    main()
