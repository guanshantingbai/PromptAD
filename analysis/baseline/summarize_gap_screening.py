"""
Summarize gap screening results for all classes

Generate a CSV with:
- class_name
- semantic_AUROC
- gap_AUC_error (AUROC of gap as error predictor)
- error_rate
- FP_rate (based on normal samples)
- FN_rate (based on abnormal samples)
- gap_direction (positive/negative)
- gap_category (usable/unsure/disable)
"""

import argparse
import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.metrics import roc_auc_score


def load_gap_stats(dataset, class_name, root_dir='../../result/test_gate'):
    """Load gap statistics and metadata for a class"""
    base_path = Path(root_dir) / dataset / 'semantic_gap'
    
    # Load sample-level gap stats
    csv_path = base_path / f'{class_name}_sample_gap_stats.csv'
    if not csv_path.exists():
        return None
    
    df = pd.read_csv(csv_path)
    
    # Load metadata (contains semantic_threshold and error_rate)
    meta_path = base_path / f'{class_name}_gap_meta.json'
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            meta = json.load(f)
    else:
        meta = {}
    
    return df, meta


def compute_gap_auc_error(df):
    """Compute AUROC of gap as error predictor"""
    y_true = df['is_error_sem'].values
    y_score = -df['gap'].values  # Negative: low gap → high error score
    
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        # All correct or all error
        return np.nan
    
    auc = roc_auc_score(y_true, y_score)
    return auc


def compute_error_rates(df):
    """Compute overall, FP, and FN rates"""
    n_total = len(df)
    n_errors = df['is_error_sem'].sum()
    error_rate = n_errors / n_total if n_total > 0 else 0.0
    
    # FP rate: proportion of normal samples incorrectly classified as abnormal
    normal_df = df[df['label'] == 'normal']
    n_normal = len(normal_df)
    n_fp = (normal_df['error_type'] == 'FP').sum()
    fp_rate = n_fp / n_normal if n_normal > 0 else 0.0
    
    # FN rate: proportion of abnormal samples incorrectly classified as normal
    abnormal_df = df[df['label'] == 'abnormal']
    n_abnormal = len(abnormal_df)
    n_fn = (abnormal_df['error_type'] == 'FN').sum()
    fn_rate = n_fn / n_abnormal if n_abnormal > 0 else 0.0
    
    return error_rate, fp_rate, fn_rate


def categorize_gap(auc_error):
    """Categorize gap based on AUC_error"""
    if np.isnan(auc_error):
        return 'undefined'
    elif auc_error >= 0.70:
        return 'usable'
    elif auc_error >= 0.55:
        return 'unsure'
    else:
        return 'disable'


def get_semantic_auroc(dataset, class_name, root_dir='../../result/baseline'):
    """Get semantic AUROC from baseline results"""
    csv_path = Path(root_dir) / dataset / 'k_2' / 'csv' / f'{dataset}_CLS.csv'
    
    if not csv_path.exists():
        return np.nan
    
    df = pd.read_csv(csv_path)
    
    # Find the row for this class
    row = df[df['class_name'] == class_name]
    if len(row) == 0:
        return np.nan
    
    # Get semantic_i_roc
    if 'semantic_i_roc' in row.columns:
        return row['semantic_i_roc'].values[0]
    else:
        return np.nan


def main(args):
    # All classes
    mvtec_classes = [
        'bottle', 'cable', 'capsule', 'carpet', 'grid',
        'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
        'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
    ]
    
    visa_classes = [
        'candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1',
        'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum'
    ]
    
    results = []
    
    print("="*60)
    print("Gap Screening Summary")
    print("="*60)
    print()
    
    # Process MVTec classes
    print("[MVTec-AD Classes]")
    for class_name in mvtec_classes:
        result = load_gap_stats('mvtec', class_name, args.root_dir)
        if result is None:
            print(f"  {class_name:15s} - MISSING")
            continue
        
        df, meta = result
        
        # Get semantic AUROC
        semantic_auroc = get_semantic_auroc('mvtec', class_name, args.baseline_dir)
        
        # Compute gap AUC error
        gap_auc_error = compute_gap_auc_error(df)
        
        # Compute error rates
        error_rate, fp_rate, fn_rate = compute_error_rates(df)
        
        # Determine gap direction
        if np.isnan(gap_auc_error):
            gap_direction = 'undefined'
        elif gap_auc_error > 0.5:
            gap_direction = 'positive'  # Low gap → high error (expected)
        else:
            gap_direction = 'negative'  # High gap → high error (unexpected)
        
        # Categorize
        category = categorize_gap(gap_auc_error)
        
        results.append({
            'dataset': 'mvtec',
            'class_name': class_name,
            'semantic_AUROC': semantic_auroc,
            'gap_AUC_error': gap_auc_error,
            'error_rate': error_rate,
            'FP_rate': fp_rate,
            'FN_rate': fn_rate,
            'gap_direction': gap_direction,
            'gap_category': category,
        })
        
        print(f"  {class_name:15s} - Semantic: {semantic_auroc:.2f}%, Gap AUC: {gap_auc_error:.3f}, Category: {category}")
    
    print()
    print("[VisA Classes]")
    for class_name in visa_classes:
        result = load_gap_stats('visa', class_name, args.root_dir)
        if result is None:
            print(f"  {class_name:15s} - MISSING")
            continue
        
        df, meta = result
        
        # Get semantic AUROC
        semantic_auroc = get_semantic_auroc('visa', class_name, args.baseline_dir)
        
        # Compute gap AUC error
        gap_auc_error = compute_gap_auc_error(df)
        
        # Compute error rates
        error_rate, fp_rate, fn_rate = compute_error_rates(df)
        
        # Determine gap direction
        if np.isnan(gap_auc_error):
            gap_direction = 'undefined'
        elif gap_auc_error > 0.5:
            gap_direction = 'positive'
        else:
            gap_direction = 'negative'
        
        # Categorize
        category = categorize_gap(gap_auc_error)
        
        results.append({
            'dataset': 'visa',
            'class_name': class_name,
            'semantic_AUROC': semantic_auroc,
            'gap_AUC_error': gap_auc_error,
            'error_rate': error_rate,
            'FP_rate': fp_rate,
            'FN_rate': fn_rate,
            'gap_direction': gap_direction,
            'gap_category': category,
        })
        
        print(f"  {class_name:15s} - Semantic: {semantic_auroc:.2f}%, Gap AUC: {gap_auc_error:.3f}, Category: {category}")
    
    # Create DataFrame
    df_results = pd.DataFrame(results)
    
    # Save to CSV
    output_dir = Path(args.root_dir).parent / 'test_gate' / 'summary'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'gap_screening_all.csv'
    
    df_results.to_csv(output_path, index=False)
    print()
    print(f"[INFO] Summary saved to: {output_path}")
    
    # Print statistics
    print()
    print("="*60)
    print("Gap Category Statistics")
    print("="*60)
    
    category_counts = df_results['gap_category'].value_counts()
    for category in ['usable', 'unsure', 'disable', 'undefined']:
        count = category_counts.get(category, 0)
        pct = count / len(df_results) * 100 if len(df_results) > 0 else 0
        print(f"  {category:10s}: {count:2d} ({pct:.1f}%)")
    
    # Show usable classes
    print()
    print("="*60)
    print("Usable Classes (Gap AUC >= 0.70)")
    print("="*60)
    usable = df_results[df_results['gap_category'] == 'usable'].sort_values('gap_AUC_error', ascending=False)
    if len(usable) > 0:
        for _, row in usable.iterrows():
            print(f"  {row['dataset']:5s}-{row['class_name']:15s} | Gap AUC: {row['gap_AUC_error']:.3f} | Semantic: {row['semantic_AUROC']:.2f}%")
    else:
        print("  None")
    
    # Show disable classes
    print()
    print("="*60)
    print("Disable Classes (Gap AUC <= 0.50)")
    print("="*60)
    disable = df_results[df_results['gap_category'] == 'disable'].sort_values('gap_AUC_error')
    if len(disable) > 0:
        for _, row in disable.iterrows():
            print(f"  {row['dataset']:5s}-{row['class_name']:15s} | Gap AUC: {row['gap_AUC_error']:.3f} | Semantic: {row['semantic_AUROC']:.2f}%")
    else:
        print("  None")
    
    print()
    print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Summarize gap screening results')
    parser.add_argument('--root-dir', type=str, default='../../result/test_gate',
                        help='Root directory for gap analysis results')
    parser.add_argument('--baseline-dir', type=str, default='../../result/baseline',
                        help='Baseline results directory for semantic AUROC')
    
    args = parser.parse_args()
    main(args)
