"""
Gap Informativeness Analysis

验证 gap(x) = |s_normal - s_abnormal| 是否能够预测 semantic 分支的错误。

分析内容：
1. gap 与 semantic 错误率的关系（按 gap 分 bin）
2. gap 作为 error predictor 的 AUROC
3. gap 在正常/异常样本上的分布差异
4. gap 与错误类型（FP/FN）的关系
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve

def load_gap_stats(dataset, class_name, root_dir='./result/test_gate'):
    """Load gap statistics CSV for a given class"""
    csv_path = Path(root_dir) / dataset / 'semantic_gap' / f'{class_name}_sample_gap_stats.csv'
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Gap stats not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    return df


def analyze_gap_vs_error(df, n_bins=10):
    """
    分析 gap 与 semantic 错误率的关系
    
    将样本按 gap 分成 n_bins 个区间，统计每个区间的：
    - 错误率
    - FP 率
    - FN 率
    """
    # 创建 gap bins
    df['gap_bin'] = pd.qcut(df['gap'], q=n_bins, duplicates='drop', labels=False)
    
    bin_stats = []
    for bin_id in sorted(df['gap_bin'].unique()):
        bin_data = df[df['gap_bin'] == bin_id]
        
        total = len(bin_data)
        n_errors = bin_data['is_error_sem'].sum()
        n_fp = (bin_data['error_type'] == 'FP').sum()
        n_fn = (bin_data['error_type'] == 'FN').sum()
        
        n_normal = (bin_data['label'] == 'normal').sum()
        n_abnormal = (bin_data['label'] == 'abnormal').sum()
        
        bin_stats.append({
            'gap_bin': bin_id,
            'gap_min': bin_data['gap'].min(),
            'gap_max': bin_data['gap'].max(),
            'gap_mean': bin_data['gap'].mean(),
            'n_samples': total,
            'error_rate': n_errors / total if total > 0 else 0,
            'FP_rate': n_fp / n_normal if n_normal > 0 else 0,
            'FN_rate': n_fn / n_abnormal if n_abnormal > 0 else 0,
            'n_normal': n_normal,
            'n_abnormal': n_abnormal,
        })
    
    return pd.DataFrame(bin_stats)


def compute_gap_as_error_predictor(df):
    """
    将 gap 作为 error predictor，计算 AUROC
    
    - Label: is_error_sem (1 = error, 0 = correct)
    - Score: -gap (gap 小 → 更可能出错 → score 高)
    
    返回 AUROC_error（越接近 1，gap 越有预测能力）
    """
    y_true = df['is_error_sem'].values
    y_score = -df['gap'].values  # 负号：gap 小 → score 高
    
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        # All correct or all error, AUROC undefined
        return np.nan, None, None
    
    auc_error = roc_auc_score(y_true, y_score)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    
    return auc_error, fpr, tpr


def analyze_gap_distribution(df):
    """
    分析 gap 在不同类别样本上的分布
    """
    normal_gap = df[df['label'] == 'normal']['gap']
    abnormal_gap = df[df['label'] == 'abnormal']['gap']
    
    error_gap = df[df['is_error_sem'] == 1]['gap']
    correct_gap = df[df['is_error_sem'] == 0]['gap']
    
    fp_gap = df[df['error_type'] == 'FP']['gap']
    fn_gap = df[df['error_type'] == 'FN']['gap']
    
    stats = {
        'normal_gap_mean': normal_gap.mean(),
        'normal_gap_std': normal_gap.std(),
        'normal_gap_10pct': normal_gap.quantile(0.1),
        'normal_gap_median': normal_gap.median(),
        
        'abnormal_gap_mean': abnormal_gap.mean(),
        'abnormal_gap_std': abnormal_gap.std(),
        
        'error_gap_mean': error_gap.mean() if len(error_gap) > 0 else np.nan,
        'error_gap_std': error_gap.std() if len(error_gap) > 0 else np.nan,
        
        'correct_gap_mean': correct_gap.mean(),
        'correct_gap_std': correct_gap.std(),
        
        'fp_gap_mean': fp_gap.mean() if len(fp_gap) > 0 else np.nan,
        'fn_gap_mean': fn_gap.mean() if len(fn_gap) > 0 else np.nan,
    }
    
    return stats


def visualize_gap_analysis(df, bin_stats, class_name, output_dir):
    """
    生成可视化图表：
    1. Gap vs Error Rate
    2. Gap 分布（normal vs abnormal）
    3. Gap 分布（error vs correct）
    4. ROC curve (gap as error predictor)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Gap Informativeness Analysis: {class_name}', fontsize=16, fontweight='bold')
    
    # ===== Plot 1: Gap vs Error Rate =====
    ax = axes[0, 0]
    ax.plot(bin_stats['gap_mean'], bin_stats['error_rate'], 'o-', linewidth=2, markersize=8, label='Error Rate')
    ax.plot(bin_stats['gap_mean'], bin_stats['FP_rate'], 's--', linewidth=1.5, markersize=6, label='FP Rate')
    ax.plot(bin_stats['gap_mean'], bin_stats['FN_rate'], '^--', linewidth=1.5, markersize=6, label='FN Rate')
    ax.set_xlabel('Gap (mean in bin)', fontsize=12)
    ax.set_ylabel('Error Rate', fontsize=12)
    ax.set_title('Gap vs Semantic Error Rate', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    
    # ===== Plot 2: Gap Distribution (normal vs abnormal) =====
    ax = axes[0, 1]
    normal_gap = df[df['label'] == 'normal']['gap']
    abnormal_gap = df[df['label'] == 'abnormal']['gap']
    
    ax.hist(normal_gap, bins=30, alpha=0.6, label='Normal', color='blue', edgecolor='black')
    ax.hist(abnormal_gap, bins=30, alpha=0.6, label='Abnormal', color='red', edgecolor='black')
    ax.axvline(normal_gap.quantile(0.1), color='blue', linestyle='--', linewidth=2, label='Normal 10%')
    ax.set_xlabel('Gap', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Gap Distribution: Normal vs Abnormal', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    # ===== Plot 3: Gap Distribution (error vs correct) =====
    ax = axes[1, 0]
    error_gap = df[df['is_error_sem'] == 1]['gap']
    correct_gap = df[df['is_error_sem'] == 0]['gap']
    
    ax.hist(correct_gap, bins=30, alpha=0.6, label='Correct', color='green', edgecolor='black')
    ax.hist(error_gap, bins=30, alpha=0.6, label='Error', color='orange', edgecolor='black')
    ax.set_xlabel('Gap', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Gap Distribution: Error vs Correct', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    # ===== Plot 4: ROC Curve (gap as error predictor) =====
    ax = axes[1, 1]
    auc_error, fpr, tpr = compute_gap_as_error_predictor(df)
    
    if auc_error is not np.nan:
        ax.plot(fpr, tpr, linewidth=2, label=f'Gap as Error Predictor (AUC={auc_error:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('Gap as Error Predictor (ROC)', fontsize=14)
        ax.legend()
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'AUROC Undefined\n(All correct or all error)', 
                ha='center', va='center', fontsize=12)
        ax.set_title('Gap as Error Predictor (ROC)', fontsize=14)
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_dir) / f'{class_name}_gap_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[INFO] Visualization saved to: {output_path}")
    plt.close()


def main(args):
    # Load gap statistics
    df = load_gap_stats(args.dataset, args.class_name, args.root_dir)
    
    print(f"\n{'='*60}")
    print(f"Gap Informativeness Analysis: {args.class_name}")
    print(f"{'='*60}\n")
    
    # Basic statistics
    n_total = len(df)
    n_normal = (df['label'] == 'normal').sum()
    n_abnormal = (df['label'] == 'abnormal').sum()
    n_errors = df['is_error_sem'].sum()
    overall_error_rate = n_errors / n_total
    
    print(f"[Dataset Statistics]")
    print(f"  Total samples:   {n_total}")
    print(f"  Normal:          {n_normal} ({n_normal/n_total*100:.1f}%)")
    print(f"  Abnormal:        {n_abnormal} ({n_abnormal/n_total*100:.1f}%)")
    print(f"  Semantic errors: {n_errors} ({overall_error_rate*100:.1f}%)")
    print()
    
    # ===== Analysis 1: Gap vs Error Rate =====
    print(f"[Analysis 1: Gap vs Error Rate]")
    bin_stats = analyze_gap_vs_error(df, n_bins=10)
    print(bin_stats.to_string(index=False))
    print()
    
    # Save bin stats
    output_dir = Path(args.root_dir) / args.dataset / 'semantic_gap'
    output_dir.mkdir(parents=True, exist_ok=True)
    bin_stats.to_csv(output_dir / f'{args.class_name}_gap_bin_stats.csv', index=False)
    
    # ===== Analysis 2: Gap as Error Predictor =====
    print(f"[Analysis 2: Gap as Error Predictor]")
    auc_error, fpr, tpr = compute_gap_as_error_predictor(df)
    
    if auc_error is not np.nan:
        print(f"  AUROC (gap predicts error): {auc_error:.4f}")
        
        if auc_error > 0.7:
            print(f"  → ✅ Gap has STRONG predictive power for semantic errors!")
        elif auc_error > 0.6:
            print(f"  → ⚠️  Gap has MODERATE predictive power.")
        else:
            print(f"  → ❌ Gap has WEAK predictive power.")
    else:
        print(f"  AUROC undefined (all correct or all error)")
    print()
    
    # ===== Analysis 3: Gap Distribution =====
    print(f"[Analysis 3: Gap Distribution]")
    dist_stats = analyze_gap_distribution(df)
    
    print(f"  Normal samples:")
    print(f"    Mean gap:   {dist_stats['normal_gap_mean']:.3f} ± {dist_stats['normal_gap_std']:.3f}")
    print(f"    10% tile:   {dist_stats['normal_gap_10pct']:.3f}")
    print(f"    Median:     {dist_stats['normal_gap_median']:.3f}")
    print()
    print(f"  Abnormal samples:")
    print(f"    Mean gap:   {dist_stats['abnormal_gap_mean']:.3f} ± {dist_stats['abnormal_gap_std']:.3f}")
    print()
    print(f"  Error vs Correct:")
    print(f"    Error gap:   {dist_stats['error_gap_mean']:.3f} ± {dist_stats['error_gap_std']:.3f}")
    print(f"    Correct gap: {dist_stats['correct_gap_mean']:.3f} ± {dist_stats['correct_gap_std']:.3f}")
    print(f"    Δ (correct - error): {dist_stats['correct_gap_mean'] - dist_stats['error_gap_mean']:.3f}")
    print()
    
    if not np.isnan(dist_stats['fp_gap_mean']):
        print(f"  FP gap:  {dist_stats['fp_gap_mean']:.3f}")
    if not np.isnan(dist_stats['fn_gap_mean']):
        print(f"  FN gap:  {dist_stats['fn_gap_mean']:.3f}")
    print()
    
    # ===== Key Insights =====
    print(f"[Key Insights]")
    
    # Check if low gap → high error
    low_gap_bin = bin_stats.iloc[0]  # First bin (lowest gap)
    high_gap_bin = bin_stats.iloc[-1]  # Last bin (highest gap)
    
    print(f"  Lowest gap bin (gap < {low_gap_bin['gap_max']:.2f}):")
    print(f"    Error rate: {low_gap_bin['error_rate']*100:.1f}%")
    print()
    print(f"  Highest gap bin (gap > {high_gap_bin['gap_min']:.2f}):")
    print(f"    Error rate: {high_gap_bin['error_rate']*100:.1f}%")
    print()
    
    if low_gap_bin['error_rate'] > 1.5 * high_gap_bin['error_rate']:
        print(f"  → ✅ Low gap samples have SIGNIFICANTLY higher error rate!")
        print(f"  → Gap can be used for semantic suppression.")
    elif low_gap_bin['error_rate'] > high_gap_bin['error_rate']:
        print(f"  → ⚠️  Low gap samples have SLIGHTLY higher error rate.")
        print(f"  → Gap has weak signal, proceed with caution.")
    else:
        print(f"  → ❌ No clear relationship between gap and error rate.")
        print(f"  → Gap is NOT informative for semantic suppression.")
    print()
    
    # ===== Generate Visualization =====
    print(f"[Generating Visualization]")
    visualize_gap_analysis(df, bin_stats, args.class_name, output_dir)
    
    print(f"\n{'='*60}")
    print(f"Analysis complete for {args.class_name}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze gap informativeness')
    parser.add_argument('--dataset', type=str, default='mvtec', choices=['mvtec', 'visa'])
    parser.add_argument('--class_name', type=str, required=True)
    parser.add_argument('--root-dir', type=str, default='./result/test_gate')
    
    args = parser.parse_args()
    main(args)
