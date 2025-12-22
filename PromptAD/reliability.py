"""
Reliability estimation for two-branch anomaly detection.

=== DESIGN PHILOSOPHY ===

Phase 1 (Current): ANALYSIS MODE ONLY
- Compute reliability indicators from prior information
- Output separate indicators per branch (NO aggregation)
- Format: {r_mem_margin, r_mem_entropy, ..., r_sem_prompt_var, ...}
- Purpose: Validate correlation with oracle choices

Phase 2 (Future): Gating mechanism (only if Phase 1 shows promise)
- Add weighted combination of indicators
- Implement adaptive score fusion
- Evaluate as a "method"

=== CORE PRINCIPLES ===

Estimate per-sample reliability using ONLY prior information:
- Few-shot support set statistics
- Memory bank / gallery statistics  
- Prompt ensemble statistics

NO learnable parameters, NO dataset-specific tuning, NO ground-truth labels.
All normalization is calibrated on support set via leave-one-out or gallery stats.

Author: gate2-spatial-prompts branch
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional


class ReliabilityEstimator:
    """
    Computes reliability indicators for memory and semantic branches.
    
    All indicators are z-scores normalized using support-set statistics,
    making them comparable across samples and classes.
    """
    
    def __init__(self, temperature: float = 0.07):
        """
        Args:
            temperature: softmax temperature for entropy computation
        """
        self.temperature = temperature
        self.support_stats = {}  # Cache for support set statistics
        
    def calibrate_on_support(self, 
                            support_features: torch.Tensor,
                            gallery: torch.Tensor,
                            prompt_scores_support: Optional[torch.Tensor] = None):
        """
        Pre-compute statistics on support set for normalization.
        
        Args:
            support_features: (k_shot, D) features of support samples
            gallery: (k_shot, D) memory bank (same as support for few-shot)
            prompt_scores_support: (k_shot, num_prompts) if available
        """
        k_shot = support_features.shape[0]
        
        # === Memory branch statistics (leave-one-out) ===
        nn_margins = []
        entropies = []
        
        for i in range(k_shot):
            # Leave one out
            query_feat = support_features[i:i+1]  # (1, D)
            gallery_loo = torch.cat([support_features[:i], support_features[i+1:]], dim=0)  # (k-1, D)
            
            # Compute similarities (higher = more similar)
            similarities = query_feat @ gallery_loo.t()  # (1, k-1)
            
            # Top-2 similarities
            top2_sim, _ = similarities.topk(2, dim=-1, largest=True)  # (1, 2)
            margin = (top2_sim[:, 0] - top2_sim[:, 1]).item()
            nn_margins.append(margin)
            
            # Neighborhood entropy
            probs = F.softmax(similarities / self.temperature, dim=-1)  # (1, k-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).item()
            entropies.append(entropy)
        
        # Store statistics (median + MAD for robustness)
        self.support_stats['nn_margin_median'] = np.median(nn_margins)
        self.support_stats['nn_margin_mad'] = np.median(np.abs(nn_margins - np.median(nn_margins)))
        self.support_stats['entropy_median'] = np.median(entropies)
        self.support_stats['entropy_mad'] = np.median(np.abs(entropies - np.median(entropies)))
        
        # === Semantic branch statistics ===
        if prompt_scores_support is not None:
            # Prompt variance across support samples
            prompt_vars = prompt_scores_support.var(dim=1).cpu().numpy()  # (k_shot,)
            self.support_stats['prompt_var_median'] = np.median(prompt_vars)
            self.support_stats['prompt_var_mad'] = np.median(np.abs(prompt_vars - np.median(prompt_vars)))
            
            # Prompt margin (top-1 vs top-2)
            sorted_scores, _ = prompt_scores_support.sort(dim=1, descending=True)
            prompt_margins = (sorted_scores[:, 0] - sorted_scores[:, 1]).cpu().numpy()
            self.support_stats['prompt_margin_median'] = np.median(prompt_margins)
            self.support_stats['prompt_margin_mad'] = np.median(np.abs(prompt_margins - np.median(prompt_margins)))
    
    def compute_memory_reliability(self,
                                   query_features: torch.Tensor,
                                   gallery: torch.Tensor,
                                   topk: int = 5) -> Dict[str, np.ndarray]:
        """
        Compute memory branch reliability indicators.
        
        Args:
            query_features: (N, D) query sample features
            gallery: (M, D) memory bank features
            topk: number of neighbors for entropy computation
            
        Returns:
            reliability_dict with normalized z-scores (higher = more reliable)
        """
        N = query_features.shape[0]
        
        # Compute similarities to gallery
        similarities = query_features @ gallery.t()  # (N, M)
        
        # === 1. NN Margin ===
        top2_sim, _ = similarities.topk(2, dim=-1, largest=True)  # (N, 2)
        nn_margins = (top2_sim[:, 0] - top2_sim[:, 1]).cpu().numpy()  # (N,)
        
        # Normalize using support stats
        margin_median = self.support_stats.get('nn_margin_median', 0.1)
        margin_mad = self.support_stats.get('nn_margin_mad', 0.01) + 1e-6
        nn_margin_z = (nn_margins - margin_median) / margin_mad
        
        # === 2. Neighborhood Entropy ===
        topk_sim, _ = similarities.topk(topk, dim=-1, largest=True)  # (N, k)
        probs = F.softmax(topk_sim / self.temperature, dim=-1)  # (N, k)
        entropies = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).cpu().numpy()  # (N,)
        
        # Normalize (note: lower entropy = more reliable, so negate)
        entropy_median = self.support_stats.get('entropy_median', 1.0)
        entropy_mad = self.support_stats.get('entropy_mad', 0.1) + 1e-6
        entropy_z = -(entropies - entropy_median) / entropy_mad  # Negate so higher = better
        
        # === 3. Distance to Support Center (additional indicator) ===
        # Centroid of gallery
        gallery_centroid = gallery.mean(dim=0, keepdim=True)  # (1, D)
        centroid_sim = (query_features @ gallery_centroid.t()).squeeze(-1).cpu().numpy()  # (N,)
        
        # Normalize (higher similarity = closer to support = more reliable for normal)
        centroid_sim_z = (centroid_sim - centroid_sim.mean()) / (centroid_sim.std() + 1e-6)
        
        return {
            'nn_margin_z': nn_margin_z,
            'neighbor_entropy_z': entropy_z,
            'centroid_similarity_z': centroid_sim_z,
            'raw_nn_margin': nn_margins,
            'raw_entropy': entropies
        }
    
    def compute_semantic_reliability(self,
                                    prompt_scores: torch.Tensor) -> Dict[str, np.ndarray]:
        """
        Compute semantic branch reliability indicators.
        
        Args:
            prompt_scores: (N, num_prompts) anomaly scores per prompt
            
        Returns:
            reliability_dict with normalized z-scores (higher = more reliable)
        """
        N = prompt_scores.shape[0]
        
        # === 1. Prompt Variance ===
        prompt_vars = prompt_scores.var(dim=1).cpu().numpy()  # (N,)
        
        # Normalize (lower variance = more agreement = more reliable, so negate)
        var_median = self.support_stats.get('prompt_var_median', 0.01)
        var_mad = self.support_stats.get('prompt_var_mad', 0.001) + 1e-6
        prompt_var_z = -(prompt_vars - var_median) / var_mad  # Negate
        
        # === 2. Prompt Margin ===
        sorted_scores, _ = prompt_scores.sort(dim=1, descending=True)
        prompt_margins = (sorted_scores[:, 0] - sorted_scores[:, 1]).cpu().numpy()  # (N,)
        
        # Normalize (higher margin = stronger consensus = more reliable)
        margin_median = self.support_stats.get('prompt_margin_median', 0.1)
        margin_mad = self.support_stats.get('prompt_margin_mad', 0.01) + 1e-6
        prompt_margin_z = (prompt_margins - margin_median) / margin_mad
        
        # === 3. Score Extremity (additional indicator) ===
        # Distance from 0.5 (more extreme = more confident)
        mean_scores = prompt_scores.mean(dim=1).cpu().numpy()  # (N,)
        extremity = np.abs(mean_scores - 0.5)
        extremity_z = (extremity - extremity.mean()) / (extremity.std() + 1e-6)
        
        return {
            'prompt_variance_z': prompt_var_z,
            'prompt_margin_z': prompt_margin_z,
            'score_extremity_z': extremity_z,
            'raw_prompt_variance': prompt_vars,
            'raw_prompt_margin': prompt_margins
        }
    
    def compute_augmentation_consistency(self,
                                        query_features: torch.Tensor,
                                        score_fn,
                                        num_augs: int = 3) -> Dict[str, np.ndarray]:
        """
        Compute reliability via augmentation consistency (optional).
        
        Args:
            query_features: (N, D) query features
            score_fn: callable that takes features and returns scores
            num_augs: number of augmentations to apply
            
        Returns:
            aug_consistency_dict
        """
        # Placeholder - requires augmentation pipeline
        # Would apply lightweight transforms and measure score variance
        raise NotImplementedError("Augmentation consistency requires augmentation pipeline")
    
    def compare_branch_reliability(self,
                                   memory_reliability: Dict[str, np.ndarray],
                                   semantic_reliability: Dict[str, np.ndarray]) -> Dict:
        """
        Compare reliability between branches WITHOUT aggregation.
        
        Phase 1 analysis mode: Return raw indicators grouped by branch
        for correlation analysis with oracle choices.
        
        Args:
            memory_reliability: dict from compute_memory_reliability
            semantic_reliability: dict from compute_semantic_reliability
            
        Returns:
            comparison_dict with separate indicators (NO weighted combination)
        """
        comparison_meta = {
            # Memory branch indicators (separate, not combined)
            'r_mem_margin': memory_reliability['nn_margin_z'],
            'r_mem_entropy': memory_reliability['neighbor_entropy_z'],
            'r_mem_centroid': memory_reliability['centroid_similarity_z'],
            
            # Semantic branch indicators (separate, not combined)
            'r_sem_prompt_var': semantic_reliability['prompt_variance_z'],
            'r_sem_prompt_margin': semantic_reliability['prompt_margin_z'],
            'r_sem_extremity': semantic_reliability['score_extremity_z'],
            
            # Raw values for debugging
            'raw_memory': memory_reliability,
            'raw_semantic': semantic_reliability
        }
        
        return comparison_meta
    
    # === PHASE 1: ANALYSIS MODE ONLY ===
    # No adaptive gating, no score fusion, no final anomaly scores.
    # Only compute and output reliability indicators for oracle correlation analysis.
    # 
    # Future Phase 2 may add:
    # - compute_adaptive_weights() if indicators show strong correlation
    # - learnable combination weights
    # - dynamic gating mechanisms
