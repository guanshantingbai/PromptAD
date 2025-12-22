"""
Extended metrics for modular scoring framework.

Supports oracle evaluation and component-wise analysis.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from typing import Dict, Optional


def metric_cal_img_modular(
    img_scores,
    gt_list,
    map_scores=None,
    metadata: Optional[Dict] = None,
    score_mode: str = 'max'
) -> Dict:
    """
    Extended metric calculation supporting modular scoring.
    
    Args:
        img_scores: semantic branch scores (N,)
        gt_list: ground truth labels (N,)
        map_scores: memory branch spatial maps (N, H, W)
        metadata: optional metadata from scorer
        score_mode: 'semantic', 'memory', 'max', 'harmonic', 'oracle'
        
    Returns:
        result_dict: metrics including component-wise AUROCs
    """
    gt_list = np.asarray(gt_list, dtype=int)
    
    # Compute final fused score based on mode
    if score_mode == 'semantic':
        final_scores = img_scores
    elif score_mode == 'memory':
        # For memory-only, use max of spatial map
        if map_scores is not None:
            final_scores = map_scores.reshape(map_scores.shape[0], -1).max(axis=1)
        else:
            final_scores = img_scores
    elif score_mode in ['max', 'harmonic']:
        # Already fused in scorer
        final_scores = img_scores
    elif score_mode == 'oracle':
        # Already selected in scorer
        final_scores = img_scores
    else:
        # Fallback: default max fusion
        if map_scores is not None:
            max_map_scores = map_scores.reshape(map_scores.shape[0], -1).max(axis=1)
            final_scores = np.maximum(max_map_scores, img_scores)
        else:
            final_scores = img_scores
    
    # Main metric
    img_roc_auc = roc_auc_score(gt_list, final_scores)
    
    result_dict = {
        'i_roc': img_roc_auc * 100,
        'score_mode': score_mode
    }
    
    # Add component-wise metrics if both branches available in metadata
    if metadata is not None and 'semantic_scores' in metadata and 'memory_scores' in metadata:
        semantic_scores = metadata['semantic_scores']
        memory_scores = metadata['memory_scores']
        
        # Compute per-branch AUROCs
        semantic_auroc = roc_auc_score(gt_list, semantic_scores)
        
        if map_scores is not None:
            memory_map_max = map_scores.reshape(map_scores.shape[0], -1).max(axis=1)
            memory_auroc = roc_auc_score(gt_list, memory_map_max)
        else:
            memory_auroc = roc_auc_score(gt_list, memory_scores)
        
        result_dict.update({
            'i_roc_semantic': semantic_auroc * 100,
            'i_roc_memory': memory_auroc * 100,
            'gap': (img_roc_auc - max(semantic_auroc, memory_auroc)) * 100
        })
    
    # Oracle-specific metrics
    if score_mode == 'oracle' and metadata is not None:
        if 'selections' in metadata:
            selections = metadata['selections']
            result_dict['oracle_semantic_ratio'] = (selections == 0).mean() * 100
            result_dict['oracle_memory_ratio'] = (selections == 1).mean() * 100
    
    return result_dict


def analyze_reliability_signals(metadata: Dict, gt_list: np.ndarray) -> Dict:
    """
    Analyze correlation between reliability signals and correctness.
    
    This helps understand which signals are useful for gating.
    
    Args:
        metadata: metadata dict from scorer
        gt_list: ground truth labels
        
    Returns:
        analysis_dict: correlation statistics
    """
    analysis = {}
    
    if 'semantic_meta' in metadata:
        semantic_meta = metadata['semantic_meta']
        
        # Analyze prompt variance as reliability signal
        if 'prompt_variance' in semantic_meta:
            variance = semantic_meta['prompt_variance']
            # Compute correlation with error
            # (This is placeholder - need actual predictions vs GT comparison)
            analysis['semantic_prompt_variance_mean'] = variance.mean()
            analysis['semantic_prompt_variance_std'] = variance.std()
    
    if 'memory_meta' in metadata:
        memory_meta = metadata['memory_meta']
        
        # Analyze NN margin as reliability signal
        if 'nn_margin' in memory_meta:
            margin = memory_meta['nn_margin']
            analysis['memory_nn_margin_mean'] = margin.mean()
            analysis['memory_nn_margin_std'] = margin.std()
    
    return analysis
