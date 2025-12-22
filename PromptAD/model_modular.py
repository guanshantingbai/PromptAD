"""
Minimal wrapper to integrate modular scoring into existing PromptAD model.

Usage:
    # Original model
    model = PromptAD(...)
    
    # Wrapped model with modular scoring
    from PromptAD.model_modular import PromptADModular
    model = PromptADModular(base_model=model, score_mode='max')
    
    # Use exactly as before
    scores = model(images, task='cls')
"""

import torch
import numpy as np
from .scoring import AnomalyScorer, convert_to_model_output_format
from scipy.ndimage import gaussian_filter
import torch.nn.functional as F


class PromptADModular:
    """
    Wrapper around PromptAD model to support modular scoring.
    
    Maintains backward compatibility with existing training/evaluation code.
    """
    
    def __init__(self, base_model, score_mode='max', return_metadata=False):
        """
        Args:
            base_model: original PromptAD model instance
            score_mode: 'semantic', 'memory', 'max', 'harmonic', 'oracle'
            return_metadata: if True, return (output, metadata) instead of just output
        """
        self.base_model = base_model
        self.scorer = AnomalyScorer(base_model, score_mode=score_mode)
        self.return_metadata = return_metadata
        self.score_mode = score_mode
        
        # Expose base model attributes for compatibility
        self.feature_gallery1 = base_model.feature_gallery1
        self.feature_gallery2 = base_model.feature_gallery2
        self.text_features = base_model.text_features
        self.transform = base_model.transform
        
    def __call__(self, images, task='cls', gt_labels=None):
        """
        Forward pass with modular scoring.
        
        Args:
            images: input images
            task: 'cls' or 'seg'
            gt_labels: optional GT for oracle mode
            
        Returns:
            Same format as original model.forward()
            OR (output, metadata) if return_metadata=True
        """
        # Extract visual features (same as original)
        visual_features = self.base_model.encode_image(images)
        
        # Compute scores using modular scorer
        scores, metadata = self.scorer.compute_score(
            visual_features, 
            task=task,
            gt_labels=gt_labels
        )
        
        # Apply post-processing for segmentation
        if task == 'seg':
            # Interpolate to output size
            N = scores.shape[0]
            scores_tensor = torch.from_numpy(scores).unsqueeze(1).float()
            scores_tensor = F.interpolate(
                scores_tensor, 
                size=(self.base_model.out_size_h, self.base_model.out_size_w),
                mode='bilinear', 
                align_corners=False
            )
            scores = scores_tensor.squeeze(1).numpy()
            
            # Apply gaussian smoothing
            am_pix_list = []
            for i in range(N):
                smoothed = gaussian_filter(scores[i], sigma=4)
                am_pix_list.append(smoothed)
            
            output = am_pix_list
            
        else:  # cls
            # Need to return memory maps too for metric calculation
            visual_anomaly_map = self.base_model.calculate_visual_anomaly_score(visual_features)
            anomaly_map = F.interpolate(
                visual_anomaly_map, 
                size=(self.base_model.out_size_h, self.base_model.out_size_w),
                mode='bilinear',
                align_corners=False
            )
            am_pix = anomaly_map.squeeze(1).numpy()
            am_pix_list = [am_pix[i] for i in range(am_pix.shape[0])]
            
            am_img_list = scores.tolist()
            output = (am_img_list, am_pix_list)
        
        if self.return_metadata:
            return output, metadata
        else:
            return output
    
    def forward(self, images, task='cls', gt_labels=None):
        """Alias for __call__ to match original interface."""
        return self(images, task, gt_labels)
    
    def encode_image(self, images):
        """Delegate to base model."""
        return self.base_model.encode_image(images)
    
    def setup_text_features(self, *args, **kwargs):
        """Delegate to base model."""
        return self.base_model.setup_text_features(*args, **kwargs)
    
    def setup_memory_bank(self, *args, **kwargs):
        """Delegate to base model."""
        return self.base_model.setup_memory_bank(*args, **kwargs)
    
    def train_mode(self):
        """Delegate to base model."""
        self.base_model.train_mode()
    
    def eval_mode(self):
        """Delegate to base model."""
        self.base_model.eval_mode()
    
    def state_dict(self):
        """Delegate to base model."""
        return self.base_model.state_dict()
    
    def load_state_dict(self, *args, **kwargs):
        """Delegate to base model."""
        return self.base_model.load_state_dict(*args, **kwargs)
    
    def to(self, device):
        """Delegate to base model and return self for chaining."""
        self.base_model = self.base_model.to(device)
        # Update references
        self.feature_gallery1 = self.base_model.feature_gallery1
        self.feature_gallery2 = self.base_model.feature_gallery2
        self.text_features = self.base_model.text_features
        return self
