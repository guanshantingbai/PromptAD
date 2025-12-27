"""
Margin Distribution Analysis for PromptAD Baseline

This module analyzes the margin distribution (s_normal - s_abnormal) 
to understand the discriminative structure of the baseline model.
"""

import numpy as np
from typing import Dict, Tuple, List


def calculate_margin_statistics(
    margins: np.ndarray,
    labels: np.ndarray,
    percentiles: List[int] = [10, 25, 50, 75, 90]
) -> Dict[str, Dict[str, float]]:
    """
    Calculate comprehensive margin statistics for normal and abnormal samples.
    
    Args:
        margins: np.array of shape [N], margin = s_normal - s_abnormal
        labels: np.array of shape [N], binary labels (0=normal, 1=abnormal)
        percentiles: List of percentiles to compute
    
    Returns:
        Dictionary with statistics for 'normal' and 'abnormal' samples:
        {
            'normal': {
                'mean': float,
                'median': float,
                'std': float,
                'min': float,
                'max': float,
                'p10': float, 'p25': float, ...
            },
            'abnormal': {...}
        }
    """
    margins_normal = margins[labels == 0]
    margins_abnormal = margins[labels == 1]
    
    def compute_stats(data):
        stats = {
            'mean': float(np.mean(data)),
            'median': float(np.median(data)),
            'std': float(np.std(data)),
            'min': float(np.min(data)),
            'max': float(np.max(data)),
        }
        for p in percentiles:
            stats[f'p{p}'] = float(np.percentile(data, p))
        return stats
    
    return {
        'normal': compute_stats(margins_normal),
        'abnormal': compute_stats(margins_abnormal),
        'overall': compute_stats(margins)
    }


def analyze_split_side_risks(
    margins: np.ndarray,
    labels: np.ndarray,
    thresholds: List[float] = [0.0, 0.05, 0.1]
) -> Dict[str, Dict[str, float]]:
    """
    Analyze split-side risks: where classification errors occur.
    
    Normal-side risk: P(margin < threshold | y=0)
    Abnormal-side risk: P(margin > -threshold | y=1)
    
    Args:
        margins: np.array of shape [N]
        labels: np.array of shape [N]
        thresholds: List of margin thresholds to analyze
    
    Returns:
        {
            'normal_side_risk': {
                'threshold_0.0': float,  # P(m < 0 | normal)
                'threshold_0.05': float, # P(m < 0.05 | normal)
                ...
            },
            'abnormal_side_risk': {
                'threshold_0.0': float,  # P(m > 0 | abnormal)
                'threshold_0.05': float, # P(m > -0.05 | abnormal)
                ...
            }
        }
    """
    margins_normal = margins[labels == 0]
    margins_abnormal = margins[labels == 1]
    
    normal_side_risks = {}
    abnormal_side_risks = {}
    
    for thresh in thresholds:
        # Normal-side risk: fraction of normals with margin < threshold
        normal_side_risks[f'threshold_{thresh}'] = float(
            np.mean(margins_normal < thresh)
        )
        
        # Abnormal-side risk: fraction of abnormals with margin > -threshold
        abnormal_side_risks[f'threshold_{thresh}'] = float(
            np.mean(margins_abnormal > -thresh)
        )
    
    return {
        'normal_side_risk': normal_side_risks,
        'abnormal_side_risk': abnormal_side_risks
    }


def compute_overlap_ratio(
    margins: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.0
) -> Dict[str, float]:
    """
    Compute margin overlap ratio.
    
    Overlap occurs when:
    - Normal samples have margin < threshold (misclassified as abnormal)
    - Abnormal samples have margin > threshold (misclassified as normal)
    
    Args:
        margins: np.array of shape [N]
        labels: np.array of shape [N]
        threshold: Decision threshold (default 0.0)
    
    Returns:
        {
            'overlap_ratio': float,  # Overall overlap fraction
            'normal_overlap': float, # Fraction of normals in overlap
            'abnormal_overlap': float, # Fraction of abnormals in overlap
            'total_overlap_count': int
        }
    """
    margins_normal = margins[labels == 0]
    margins_abnormal = margins[labels == 1]
    
    # Count overlaps
    normal_overlap = np.sum(margins_normal < threshold)
    abnormal_overlap = np.sum(margins_abnormal > threshold)
    total_overlap = normal_overlap + abnormal_overlap
    
    return {
        'overlap_ratio': float(total_overlap / len(margins)),
        'normal_overlap': float(normal_overlap / len(margins_normal)),
        'abnormal_overlap': float(abnormal_overlap / len(margins_abnormal)),
        'total_overlap_count': int(total_overlap),
        'threshold': threshold
    }


def compute_fpr_at_tpr(
    scores: np.ndarray,
    labels: np.ndarray,
    target_tpr: float = 0.95
) -> float:
    """
    Compute FPR at a given TPR (e.g., FPR@95TPR).
    
    Args:
        scores: Anomaly scores (higher = more abnormal)
        labels: Binary labels (0=normal, 1=abnormal)
        target_tpr: Target true positive rate
    
    Returns:
        FPR at the given TPR
    """
    from sklearn.metrics import roc_curve
    
    fpr, tpr, thresholds = roc_curve(labels, scores)
    
    # Find the FPR at target TPR
    idx = np.argmin(np.abs(tpr - target_tpr))
    return float(fpr[idx])


def generate_margin_report(
    margins: np.ndarray,
    labels: np.ndarray,
    semantic_scores: np.ndarray,
    fusion_scores: np.ndarray,
    class_name: str
) -> Dict:
    """
    Generate comprehensive margin analysis report.
    
    Args:
        margins: Margin values
        labels: Ground truth labels
        semantic_scores: Semantic-only anomaly scores
        fusion_scores: Fusion anomaly scores
        class_name: Name of the class being analyzed
    
    Returns:
        Complete report dictionary
    """
    from sklearn.metrics import roc_auc_score
    
    report = {
        'class_name': class_name,
        'margin_statistics': calculate_margin_statistics(margins, labels),
        'split_side_risks': analyze_split_side_risks(margins, labels),
        'overlap_analysis': compute_overlap_ratio(margins, labels),
        'semantic_auroc': float(roc_auc_score(labels, semantic_scores)),
        'fusion_auroc': float(roc_auc_score(labels, fusion_scores)),
        'semantic_fpr95': compute_fpr_at_tpr(semantic_scores, labels, 0.95),
        'fusion_fpr95': compute_fpr_at_tpr(fusion_scores, labels, 0.95),
    }
    
    # Calculate delta
    report['delta_fusion'] = report['fusion_auroc'] - report['semantic_auroc']
    
    return report
