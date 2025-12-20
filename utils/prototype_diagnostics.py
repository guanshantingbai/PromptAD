"""
Lightweight inference-time diagnostics for multi-prototype PromptAD.
No backprop, no training, analysis-only.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional


class PrototypeDiagnostics:
    """Collect statistics about prototype usage and separation during inference."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all accumulated statistics."""
        self.normal_proto_counts = None  # Will be initialized based on num prototypes
        self.abnormal_proto_counts = None
        self.total_patches = 0
        self.total_images = 0
        self.patch_dominance_scores = []  # Track per-image dominance
        
    @torch.no_grad()
    def update_from_inference(
        self, 
        visual_features: Tuple[torch.Tensor, torch.Tensor],
        normal_prototypes: torch.Tensor,
        abnormal_prototypes: torch.Tensor,
        logit_scale: torch.Tensor
    ):
        """
        Update statistics during inference.
        
        Args:
            visual_features: (cls_token, patch_tokens) from model.encode_image()
            normal_prototypes: [K_normal, dim]
            abnormal_prototypes: [M_abnormal, dim]
            logit_scale: Temperature scaling factor
        """
        cls_token = visual_features[0]  # [N, dim]
        patch_tokens = visual_features[1]  # [N, num_patches, dim]
        
        N = patch_tokens.shape[0]
        num_patches = patch_tokens.shape[1]
        
        # Initialize counters on first call
        if self.normal_proto_counts is None:
            K_normal = normal_prototypes.shape[0]
            M_abnormal = abnormal_prototypes.shape[0]
            self.normal_proto_counts = torch.zeros(K_normal, dtype=torch.long)
            self.abnormal_proto_counts = torch.zeros(M_abnormal, dtype=torch.long)
        
        # 1. Track normal prototype usage (argmax per patch)
        t = logit_scale.float() if logit_scale.dtype == torch.float16 else logit_scale
        
        # Compute similarity for all patches
        normal_sim = t * patch_tokens @ normal_prototypes.T  # [N, num_patches, K_normal]
        normal_argmax = normal_sim.argmax(dim=-1)  # [N, num_patches]
        
        # Update normal prototype usage counts
        for proto_idx in range(normal_prototypes.shape[0]):
            count = (normal_argmax == proto_idx).sum().item()
            self.normal_proto_counts[proto_idx] += count
        
        # Track abnormal prototype usage (for reference)
        abnormal_sim = t * patch_tokens @ abnormal_prototypes.T  # [N, num_patches, M_abnormal]
        abnormal_argmax = abnormal_sim.argmax(dim=-1)  # [N, num_patches]
        
        for proto_idx in range(abnormal_prototypes.shape[0]):
            count = (abnormal_argmax == proto_idx).sum().item()
            self.abnormal_proto_counts[proto_idx] += count
        
        # 2. Track patch-level dominance per image
        for i in range(N):
            # For each image, compute which normal prototype is most frequent
            image_assignments = normal_argmax[i]  # [num_patches]
            unique, counts = torch.unique(image_assignments, return_counts=True)
            max_count = counts.max().item()
            dominance = max_count / num_patches
            self.patch_dominance_scores.append(dominance)
        
        self.total_patches += N * num_patches
        self.total_images += N
    
    def get_normal_prototype_frequencies(self) -> np.ndarray:
        """Return normalized frequencies of normal prototype usage."""
        if self.normal_proto_counts is None:
            return np.array([])
        frequencies = self.normal_proto_counts.numpy().astype(float)
        if self.total_patches > 0:
            frequencies /= self.total_patches
        return frequencies
    
    def get_abnormal_prototype_frequencies(self) -> np.ndarray:
        """Return normalized frequencies of abnormal prototype usage."""
        if self.abnormal_proto_counts is None:
            return np.array([])
        frequencies = self.abnormal_proto_counts.numpy().astype(float)
        if self.total_patches > 0:
            frequencies /= self.total_patches
        return frequencies
    
    @staticmethod
    @torch.no_grad()
    def compute_abnormal_collapse(abnormal_prototypes: torch.Tensor) -> Dict[str, float]:
        """
        Compute pairwise cosine similarity between abnormal prototypes.
        High similarity indicates collapse.
        
        Args:
            abnormal_prototypes: [M_abnormal, dim]
            
        Returns:
            dict with mean, min, max cosine similarity
        """
        # Normalize
        abnormal_norm = abnormal_prototypes / abnormal_prototypes.norm(dim=-1, keepdim=True)
        
        # Compute pairwise cosine similarity
        sim_matrix = abnormal_norm @ abnormal_norm.T  # [M, M]
        
        # Extract upper triangular (exclude diagonal)
        M = abnormal_prototypes.shape[0]
        if M <= 1:
            return {"mean": 0.0, "min": 0.0, "max": 0.0, "num_pairs": 0}
        
        upper_tri_indices = torch.triu_indices(M, M, offset=1)
        pairwise_sims = sim_matrix[upper_tri_indices[0], upper_tri_indices[1]]
        
        return {
            "mean": pairwise_sims.mean().item(),
            "min": pairwise_sims.min().item(),
            "max": pairwise_sims.max().item(),
            "num_pairs": pairwise_sims.numel()
        }
    
    @staticmethod
    @torch.no_grad()
    def compute_normal_abnormal_separation(
        normal_prototypes: torch.Tensor,
        abnormal_prototypes: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute angular separation between normal and abnormal prototypes.
        
        Args:
            normal_prototypes: [K_normal, dim]
            abnormal_prototypes: [M_abnormal, dim]
            
        Returns:
            dict with statistics about normal-abnormal cosine similarity
        """
        # Normalize
        normal_norm = normal_prototypes / normal_prototypes.norm(dim=-1, keepdim=True)
        abnormal_norm = abnormal_prototypes / abnormal_prototypes.norm(dim=-1, keepdim=True)
        
        # Compute mean normal prototype
        mean_normal = normal_norm.mean(dim=0, keepdim=True)
        mean_normal = mean_normal / mean_normal.norm(dim=-1, keepdim=True)
        
        # Compute similarity between each abnormal and mean normal
        sims = abnormal_norm @ mean_normal.T  # [M_abnormal, 1]
        sims = sims.squeeze(-1)
        
        # Also compute all pairwise normal-abnormal similarities
        all_sims = abnormal_norm @ normal_norm.T  # [M_abnormal, K_normal]
        
        return {
            "mean_to_mean_normal": sims.mean().item(),
            "min_to_mean_normal": sims.min().item(),
            "max_to_mean_normal": sims.max().item(),
            "std_to_mean_normal": sims.std().item(),
            "all_pairs_mean": all_sims.mean().item(),
            "all_pairs_min": all_sims.min().item(),
            "all_pairs_max": all_sims.max().item(),
        }
    
    def get_dominance_statistics(self) -> Dict[str, float]:
        """
        Return statistics about patch-level dominance.
        High dominance means one normal prototype dominates most patches in an image.
        """
        if len(self.patch_dominance_scores) == 0:
            return {
                "mean": 0.0,
                "median": 0.0,
                "images_above_80pct": 0,
                "images_above_90pct": 0,
            }
        
        scores = np.array(self.patch_dominance_scores)
        return {
            "mean": float(scores.mean()),
            "median": float(np.median(scores)),
            "min": float(scores.min()),
            "max": float(scores.max()),
            "std": float(scores.std()),
            "images_above_80pct": int((scores > 0.8).sum()),
            "images_above_90pct": int((scores > 0.9).sum()),
            "total_images": len(scores),
        }
    
    def print_summary(
        self, 
        normal_prototypes: torch.Tensor,
        abnormal_prototypes: torch.Tensor
    ):
        """Print comprehensive diagnostic summary."""
        print("\n" + "="*80)
        print("MULTI-PROTOTYPE DIAGNOSTICS SUMMARY")
        print("="*80)
        
        # 1. Normal prototype usage
        print("\n[1] Normal Prototype Usage Frequency:")
        normal_freqs = self.get_normal_prototype_frequencies()
        for i, freq in enumerate(normal_freqs):
            print(f"   Prototype {i:2d}: {freq:6.2%} ({int(freq * self.total_patches):,} patches)")
        
        if len(normal_freqs) > 0:
            max_freq = normal_freqs.max()
            min_freq = normal_freqs.min()
            print(f"   Range: {min_freq:.2%} - {max_freq:.2%}")
            print(f"   Balance score (1 - max_freq): {1 - max_freq:.2%}")
        
        # 2. Abnormal prototype collapse
        print("\n[2] Abnormal Prototype Collapse Check:")
        collapse_stats = self.compute_abnormal_collapse(abnormal_prototypes)
        print(f"   Pairwise cosine similarity (higher = more collapse):")
        print(f"     Mean: {collapse_stats['mean']:.4f}")
        print(f"     Min:  {collapse_stats['min']:.4f}")
        print(f"     Max:  {collapse_stats['max']:.4f}")
        print(f"     Pairs analyzed: {collapse_stats['num_pairs']}")
        
        # 3. Normal-abnormal separation
        print("\n[3] Normal vs Abnormal Angular Separation:")
        sep_stats = self.compute_normal_abnormal_separation(normal_prototypes, abnormal_prototypes)
        print(f"   Cosine similarity (lower = better separation):")
        print(f"     Each abnormal to mean normal:")
        print(f"       Mean: {sep_stats['mean_to_mean_normal']:.4f}")
        print(f"       Min:  {sep_stats['min_to_mean_normal']:.4f}")
        print(f"       Max:  {sep_stats['max_to_mean_normal']:.4f}")
        print(f"       Std:  {sep_stats['std_to_mean_normal']:.4f}")
        print(f"     All normal-abnormal pairs:")
        print(f"       Mean: {sep_stats['all_pairs_mean']:.4f}")
        print(f"       Min:  {sep_stats['all_pairs_min']:.4f}")
        print(f"       Max:  {sep_stats['all_pairs_max']:.4f}")
        
        # 4. Patch dominance
        print("\n[4] Patch-Level Dominance (single prototype dominates patches):")
        dom_stats = self.get_dominance_statistics()
        print(f"   Mean dominance: {dom_stats['mean']:.2%}")
        print(f"   Median dominance: {dom_stats['median']:.2%}")
        print(f"   Std: {dom_stats['std']:.4f}")
        print(f"   Images with >80% dominance: {dom_stats['images_above_80pct']}/{dom_stats['total_images']} "
              f"({dom_stats['images_above_80pct']/max(dom_stats['total_images'], 1):.1%})")
        print(f"   Images with >90% dominance: {dom_stats['images_above_90pct']}/{dom_stats['total_images']} "
              f"({dom_stats['images_above_90pct']/max(dom_stats['total_images'], 1):.1%})")
        
        print("\n" + "="*80 + "\n")
    
    def export_statistics(
        self,
        normal_prototypes: torch.Tensor,
        abnormal_prototypes: torch.Tensor
    ) -> Dict:
        """Export all statistics as a dictionary."""
        return {
            "normal_prototype_frequencies": self.get_normal_prototype_frequencies().tolist(),
            "abnormal_prototype_frequencies": self.get_abnormal_prototype_frequencies().tolist(),
            "abnormal_collapse": self.compute_abnormal_collapse(abnormal_prototypes),
            "normal_abnormal_separation": self.compute_normal_abnormal_separation(
                normal_prototypes, abnormal_prototypes
            ),
            "dominance_statistics": self.get_dominance_statistics(),
            "total_patches": self.total_patches,
            "total_images": self.total_images,
        }
