"""
Anchor Geometry Analysis for PromptAD Baseline

This module analyzes the geometric relationships between normal and abnormal anchors
in the text embedding space.
"""

import numpy as np
import torch
from typing import Dict, Tuple


def analyze_anchor_geometry(
    mu_normal: np.ndarray,
    mu_abnormal: np.ndarray
) -> Dict[str, float]:
    """
    Analyze geometric relationship between normal and abnormal anchors.
    
    Args:
        mu_normal: Normal anchor, shape [D]
        mu_abnormal: Abnormal anchor, shape [D]
    
    Returns:
        {
            'cosine_similarity': float,  # cos(μ_n, μ_a)
            'l2_distance': float,        # ||μ_n - μ_a||
            'angular_distance': float,   # arccos(cos(μ_n, μ_a)) in degrees
        }
    """
    # Cosine similarity
    cos_sim = float(np.dot(mu_normal, mu_abnormal) / 
                    (np.linalg.norm(mu_normal) * np.linalg.norm(mu_abnormal)))
    
    # L2 distance
    l2_dist = float(np.linalg.norm(mu_normal - mu_abnormal))
    
    # Angular distance
    angular_dist = float(np.degrees(np.arccos(np.clip(cos_sim, -1.0, 1.0))))
    
    return {
        'cosine_similarity': cos_sim,
        'l2_distance': l2_dist,
        'angular_distance': angular_dist
    }


def analyze_anchor_decomposition(
    model,
    device: str = 'cuda:0'
) -> Dict[str, any]:
    """
    Analyze the composition of abnormal anchor (MAP vs LAP).
    
    This function temporarily constructs three variants of text features:
    1. μ_a^MAP: Only hand-crafted (MAP) prompts
    2. μ_a^LAP: Only learnable (LAP) prompts
    3. μ_a^ALL: Both MAP + LAP (original)
    
    Args:
        model: PromptAD model instance
        device: Device to run on
    
    Returns:
        {
            'mu_normal': np.array,
            'mu_abnormal_MAP': np.array,
            'mu_abnormal_LAP': np.array,
            'mu_abnormal_ALL': np.array,
            'geometry_MAP': dict,
            'geometry_LAP': dict,
            'geometry_ALL': dict,
            'MAP_weight': float,  # Relative contribution of MAP
            'LAP_weight': float   # Relative contribution of LAP
        }
    """
    model.eval()
    
    with torch.no_grad():
        # Get prompt embeddings
        normal_emb, abnormal_emb_handle, abnormal_emb_learned = model.prompt_learner()
        
        # Encode to features
        normal_features = model.encode_text_embedding(
            normal_emb, 
            model.tokenized_normal_prompts
        )
        
        abnormal_features_MAP = model.encode_text_embedding(
            abnormal_emb_handle,
            model.tokenized_abnormal_prompts_handle
        )
        
        abnormal_features_LAP = model.encode_text_embedding(
            abnormal_emb_learned,
            model.tokenized_abnormal_prompts_learned
        )
        
        # Compute anchors
        mu_normal = torch.mean(normal_features, dim=0, keepdim=True)
        mu_normal = mu_normal / mu_normal.norm(dim=-1, keepdim=True)
        mu_normal = mu_normal.squeeze(0).cpu().numpy()
        
        mu_abnormal_MAP = torch.mean(abnormal_features_MAP, dim=0, keepdim=True)
        mu_abnormal_MAP = mu_abnormal_MAP / mu_abnormal_MAP.norm(dim=-1, keepdim=True)
        mu_abnormal_MAP = mu_abnormal_MAP.squeeze(0).cpu().numpy()
        
        mu_abnormal_LAP = torch.mean(abnormal_features_LAP, dim=0, keepdim=True)
        mu_abnormal_LAP = mu_abnormal_LAP / mu_abnormal_LAP.norm(dim=-1, keepdim=True)
        mu_abnormal_LAP = mu_abnormal_LAP.squeeze(0).cpu().numpy()
        
        # Combined anchor (original)
        abnormal_features_ALL = torch.cat([abnormal_features_MAP, abnormal_features_LAP], dim=0)
        mu_abnormal_ALL = torch.mean(abnormal_features_ALL, dim=0, keepdim=True)
        mu_abnormal_ALL = mu_abnormal_ALL / mu_abnormal_ALL.norm(dim=-1, keepdim=True)
        mu_abnormal_ALL = mu_abnormal_ALL.squeeze(0).cpu().numpy()
    
    # Analyze geometry for each variant
    geometry_MAP = analyze_anchor_geometry(mu_normal, mu_abnormal_MAP)
    geometry_LAP = analyze_anchor_geometry(mu_normal, mu_abnormal_LAP)
    geometry_ALL = analyze_anchor_geometry(mu_normal, mu_abnormal_ALL)
    
    # Estimate relative contributions
    # (This is a simple approximation based on similarity to combined anchor)
    sim_MAP = np.dot(mu_abnormal_MAP, mu_abnormal_ALL)
    sim_LAP = np.dot(mu_abnormal_LAP, mu_abnormal_ALL)
    total_sim = sim_MAP + sim_LAP
    
    return {
        'mu_normal': mu_normal,
        'mu_abnormal_MAP': mu_abnormal_MAP,
        'mu_abnormal_LAP': mu_abnormal_LAP,
        'mu_abnormal_ALL': mu_abnormal_ALL,
        'geometry_MAP': geometry_MAP,
        'geometry_LAP': geometry_LAP,
        'geometry_ALL': geometry_ALL,
        'MAP_count': int(abnormal_features_MAP.shape[0]),
        'LAP_count': int(abnormal_features_LAP.shape[0]),
        'MAP_weight': float(sim_MAP / total_sim) if total_sim > 0 else 0.5,
        'LAP_weight': float(sim_LAP / total_sim) if total_sim > 0 else 0.5
    }


def evaluate_with_decomposed_anchors(
    model,
    dataloader,
    device: str = 'cuda:0'
) -> Dict[str, Dict]:
    """
    Evaluate AUROC and margin distributions using decomposed anchors.
    
    Compares:
    1. MAP-only anchor (μ_a^MAP)
    2. LAP-only anchor (μ_a^LAP)  
    3. Combined anchor (μ_a^ALL, original)
    
    Args:
        model: PromptAD model
        dataloader: Test data loader
        device: Device to run on
    
    Returns:
        {
            'MAP_only': {'auroc': float, 'margins': np.array, ...},
            'LAP_only': {'auroc': float, 'margins': np.array, ...},
            'ALL_combined': {'auroc': float, 'margins': np.array, ...}
        }
    """
    from sklearn.metrics import roc_auc_score
    
    model.eval()
    
    # Save original text_features
    original_text_features = model.text_features.clone()
    
    # Get decomposed anchors
    decomp = analyze_anchor_decomposition(model, device)
    
    # Collect all visual features and labels first
    all_visual_features = []
    all_labels = []
    
    with torch.no_grad():
        for (data, mask, label, name, img_type) in dataloader:
            data = data.to(device)
            visual_features = model.encode_image(data)
            all_visual_features.append(visual_features[0].cpu())  # CLS token
            all_labels.extend(label.numpy())
    
    all_visual_features = torch.cat(all_visual_features, dim=0).to(device)
    all_labels = np.array(all_labels)
    
    # Function to evaluate with specific anchor
    def evaluate_with_anchor(mu_normal, mu_abnormal):
        # Temporarily set text_features
        text_features = torch.from_numpy(
            np.stack([mu_normal, mu_abnormal], axis=0)
        ).to(device)
        if model.precision == 'fp16':
            text_features = text_features.half()
        model.text_features.copy_(text_features)
        
        # Calculate scores
        t = model.logit_scale
        logits = (t * all_visual_features @ model.text_features.T).cpu().numpy()
        margins = logits[:, 0] - logits[:, 1]
        scores = logits[:, 1]  # Abnormality score
        
        auroc = float(roc_auc_score(all_labels, scores))
        
        return {
            'auroc': auroc,
            'margins': margins,
            'logits': logits,
            'scores': scores
        }
    
    # Evaluate with each anchor variant
    results = {
        'MAP_only': evaluate_with_anchor(decomp['mu_normal'], decomp['mu_abnormal_MAP']),
        'LAP_only': evaluate_with_anchor(decomp['mu_normal'], decomp['mu_abnormal_LAP']),
        'ALL_combined': evaluate_with_anchor(decomp['mu_normal'], decomp['mu_abnormal_ALL'])
    }
    
    # Restore original text_features
    model.text_features.copy_(original_text_features)
    
    # Add labels for margin analysis
    for key in results:
        results[key]['labels'] = all_labels
    
    return results
