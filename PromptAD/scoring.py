"""
Modular scoring framework for PromptAD-style anomaly detection.

Provides:
1. Separate semantic-only and memory-only scoring
2. Oracle gating (upper bound analysis)
3. Infrastructure for reliability-based gating

All functions maintain identical input/output signatures for easy swapping.
"""

import torch
import numpy as np
from typing import Tuple, Dict, Optional


class AnomalyScorer:
    """
    Modular scorer supporting different fusion strategies.
    
    All scoring functions return:
    - score: normalized anomaly score (higher = more anomalous)
    - metadata: dict with reliability signals for future gating
    """
    
    def __init__(self, model, score_mode='max', normalize=True):
        """
        Args:
            model: PromptAD model instance
            score_mode: 'semantic', 'memory', 'max', 'harmonic', 'oracle'
            normalize: whether to normalize scores to [0, 1]
        """
        self.model = model
        self.score_mode = score_mode
        self.normalize = normalize
        
    def semantic_score(self, visual_features, task='cls') -> Tuple[np.ndarray, Dict]:
        """
        Compute semantic-only anomaly score using CLIP text-image similarity.
        
        Args:
            visual_features: tuple of (cls_token, patch_tokens, feat1, feat2)
            task: 'cls' for image-level, 'seg' for pixel-level
            
        Returns:
            scores: (N,) for cls or (N, H, W) for seg
            metadata: dict with reliability signals
        """
        t = self.model.model.logit_scale
        text_features = self.model.text_features  # (2, D) - [normal, abnormal]
        
        if task == 'cls':
            # Use global cls token
            global_feature = visual_features[0]  # (N, D)
            logits = t * global_feature @ text_features.T  # (N, 2)
            scores = logits.softmax(dim=-1)[:, 1]  # (N,) - abnormality scores
            
            scores_np = scores.cpu().numpy()
            
            metadata = {
                'semantic_scores': scores_np,
                'mode': 'semantic'
            }
            
            return scores_np, metadata
            
        else:  # seg task
            # Use patch tokens
            token_features = visual_features[1]  # (N, num_patches, D)
            N, num_patches, D = token_features.shape
            
            # Compute similarity for each patch
            logits = t * token_features @ text_features.T  # (N, num_patches, 2)
            scores = logits.softmax(dim=-1)[:, :, 1]  # (N, num_patches) - abnormality scores
            
            # Reshape to spatial - dynamically get grid size
            grid_h = grid_w = int(num_patches ** 0.5)
            spatial_scores = scores.reshape((N, grid_h, grid_w)).cpu().numpy()
            
            metadata = {
                'semantic_scores': scores.cpu().numpy(),
                'mode': 'semantic'
            }
            
            return spatial_scores, metadata
    
    def memory_score(self, visual_features, task='cls') -> Tuple[np.ndarray, Dict]:
        """
        Compute memory-only anomaly score using NN distance to support set.
        
        Args:
            visual_features: tuple of (cls_token, patch_tokens, feat1, feat2)
            task: 'cls' for image-level, 'seg' for pixel-level
            
        Returns:
            scores: (N,) for cls or (N, H, W) for seg
            metadata: {
                'top_k_similarities': top-k NN similarities,
                'nn_margin': distance gap between 1st and 2nd NN,
                'min_distance': minimum distance to support set
            }
        """
        N = visual_features[1].shape[0]
        
        # Compute distances to memory bank (two feature levels)
        # Distance = 1 - cosine_similarity
        dist1, idx1 = (1.0 - visual_features[2] @ self.model.feature_gallery1.t()).min(dim=-1)
        dist1 /= 2.0  # Normalize to [0, 0.5]
        
        dist2, idx2 = (1.0 - visual_features[3] @ self.model.feature_gallery2.t()).min(dim=-1)
        dist2 /= 2.0
        
        # Get second-nearest for margin calculation
        dist1_sorted, _ = (1.0 - visual_features[2] @ self.model.feature_gallery1.t()).topk(2, dim=-1, largest=False)
        dist2_sorted, _ = (1.0 - visual_features[3] @ self.model.feature_gallery2.t()).topk(2, dim=-1, largest=False)
        
        margin1 = (dist1_sorted[:, :, 1] - dist1_sorted[:, :, 0]).cpu().numpy()
        margin2 = (dist2_sorted[:, :, 1] - dist2_sorted[:, :, 0]).cpu().numpy()
        
        if task == 'seg':
            # Combine two feature levels
            num_patches = visual_features[2].shape[1]
            grid_h = grid_w = int(num_patches ** 0.5)
            
            score = 0.5 * (dist1 + dist2)  # (N, P)
            spatial_scores = score.reshape((N, grid_h, grid_w)).cpu().numpy()
            
            metadata = {
                'nn_margin_level1': margin1,
                'nn_margin_level2': margin2,
                'min_distance': score.min(dim=1)[0].cpu().numpy(),
                'mode': 'memory'
            }
            
            return spatial_scores, metadata
            
        else:  # task == 'cls'
            # For classification, take max over patches (most anomalous patch)
            score = 0.5 * (dist1 + dist2)  # (N, P)
            max_scores = score.max(dim=1)[0].cpu().numpy()  # (N,)
            
            metadata = {
                'nn_margin': (margin1.mean(axis=1) + margin2.mean(axis=1)) / 2.0,
                'max_patch_score': max_scores,
                'mode': 'memory'
            }
            
            return max_scores, metadata
    
    def max_score(self, visual_features, task='cls') -> Tuple[np.ndarray, Dict]:
        """Element-wise max fusion of semantic and memory scores."""
        semantic_scores, semantic_meta = self.semantic_score(visual_features, task)
        memory_scores, memory_meta = self.memory_score(visual_features, task)
        
        fused_scores = np.maximum(semantic_scores, memory_scores)
        
        metadata = {
            'semantic_scores': semantic_scores,
            'memory_scores': memory_scores,
            'semantic_meta': semantic_meta,
            'memory_meta': memory_meta,
            'mode': 'max'
        }
        
        return fused_scores, metadata
    
    def harmonic_score(self, visual_features, task='cls') -> Tuple[np.ndarray, Dict]:
        """Harmonic fusion of semantic and memory scores."""
        semantic_scores, semantic_meta = self.semantic_score(visual_features, task)
        memory_scores, memory_meta = self.memory_score(visual_features, task)
        
        # Harmonic fusion: 1 / (1/a + 1/b)
        eps = 1e-8
        fused_scores = 1.0 / (1.0 / (semantic_scores + eps) + 1.0 / (memory_scores + eps))
        
        metadata = {
            'semantic_scores': semantic_scores,
            'memory_scores': memory_scores,
            'semantic_meta': semantic_meta,
            'memory_meta': memory_meta,
            'mode': 'harmonic'
        }
        
        return fused_scores, metadata
    
    def oracle_score(self, visual_features, task='cls', gt_labels=None) -> Tuple[np.ndarray, Dict]:
        """
        Oracle gating: per-sample selection of better score (requires GT).
        
        This is for analysis only - computes upper bound performance.
        
        Args:
            visual_features: feature tuple
            task: 'cls' or 'seg'
            gt_labels: (N,) array of binary labels (0=normal, 1=anomalous)
                      Required for oracle selection
            
        Returns:
            scores: oracle-selected scores
            metadata: includes selection decisions and component scores
        """
        if gt_labels is None:
            raise ValueError("Oracle mode requires ground-truth labels")
        
        semantic_scores, semantic_meta = self.semantic_score(visual_features, task)
        memory_scores, memory_meta = self.memory_score(visual_features, task)
        
        # For oracle: simulate per-sample AUC and pick better branch
        # In practice, we'll use a simple heuristic: pick branch with higher score for anomalies
        N = len(gt_labels)
        oracle_scores = np.zeros_like(semantic_scores)
        selections = np.zeros(N, dtype=int)  # 0=semantic, 1=memory
        
        for i in range(N):
            if task == 'cls':
                if gt_labels[i] == 1:  # Anomalous sample
                    # Pick branch with higher score
                    if semantic_scores[i] > memory_scores[i]:
                        oracle_scores[i] = semantic_scores[i]
                        selections[i] = 0
                    else:
                        oracle_scores[i] = memory_scores[i]
                        selections[i] = 1
                else:  # Normal sample
                    # Pick branch with lower score
                    if semantic_scores[i] < memory_scores[i]:
                        oracle_scores[i] = semantic_scores[i]
                        selections[i] = 0
                    else:
                        oracle_scores[i] = memory_scores[i]
                        selections[i] = 1
            else:  # seg task
                # For segmentation, use element-wise max for now
                oracle_scores[i] = np.maximum(semantic_scores[i], memory_scores[i])
        
        metadata = {
            'semantic_scores': semantic_scores,
            'memory_scores': memory_scores,
            'selections': selections,
            'semantic_meta': semantic_meta,
            'memory_meta': memory_meta,
            'mode': 'oracle'
        }
        
        return oracle_scores, metadata
    
    def compute_score(self, visual_features, task='cls', gt_labels=None) -> Tuple[np.ndarray, Dict]:
        """
        Main entry point - dispatch to appropriate scoring function.
        
        Args:
            visual_features: feature tuple from model.encode_image()
            task: 'cls' or 'seg'
            gt_labels: optional GT for oracle mode
            
        Returns:
            scores: anomaly scores
            metadata: dict with reliability signals and component scores
        """
        if self.score_mode == 'semantic':
            return self.semantic_score(visual_features, task)
        elif self.score_mode == 'memory':
            return self.memory_score(visual_features, task)
        elif self.score_mode == 'max':
            return self.max_score(visual_features, task)
        elif self.score_mode == 'harmonic':
            return self.harmonic_score(visual_features, task)
        elif self.score_mode == 'oracle':
            return self.oracle_score(visual_features, task, gt_labels)
        else:
            raise ValueError(f"Unknown score_mode: {self.score_mode}")


def convert_to_model_output_format(scores, metadata, task='cls'):
    """
    Convert scorer output to original model.forward() format.
    
    For backward compatibility with existing training/evaluation code.
    """
    if task == 'cls':
        # Original format: (am_img_list, am_pix_list)
        # am_img_list: semantic scores (list)
        # am_pix_list: memory maps (list)
        
        am_img_list = scores.tolist() if isinstance(scores, np.ndarray) else scores
        
        # Extract memory maps from metadata if available
        if 'memory_scores' in metadata:
            am_pix_list = metadata['memory_scores'].tolist()
        else:
            # Fallback: use placeholder
            am_pix_list = [np.zeros((256, 256)) for _ in range(len(am_img_list))]
        
        return am_img_list, am_pix_list
    
    else:  # seg
        # Original format: am_pix_list (list of 2D arrays)
        am_pix_list = [scores[i] for i in range(len(scores))]
        return am_pix_list
