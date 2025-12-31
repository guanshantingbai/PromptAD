#!/usr/bin/env python3
"""
Test MAP/LAP reliability metrics computation for a single class
"""

import os
import sys
from pathlib import Path

# Setup paths
project_root = Path(__file__).parent.parent.parent.parent.absolute()
os.chdir(project_root)
sys.path.insert(0, str(project_root))

import torch
import numpy as np

from PromptAD.model import PromptAD
from datasets.dataset import CLIPDataset
from datasets.mvtec import load_mvtec
from torch.utils.data import DataLoader

print("="*80)
print("Testing MAP/LAP Reliability Metrics - Single Class")
print("="*80)

# Test configuration
dataset = 'mvtec'
class_name = 'carpet'  # Use carpet as it's a simple texture class
k_shot = 2
seed = 111
device = 'cuda:0'

print(f"\nTest Config:")
print(f"  Dataset: {dataset}")
print(f"  Class: {class_name}")
print(f"  k-shot: {k_shot}")
print(f"  Device: {device}")

# Step 1: Load model
print("\n[Step 1/5] Loading model...")
try:
    kwargs = {
        'backbone': 'ViT-B-16-plus-240',
        'pretrained_dataset': 'laion400m_e32',
        'img_resize': 366,
        'img_cropsize': 336,
        'resolution': 336,
        'out_size_h': 336,
        'out_size_w': 336,
        'n_ctx': 12,
        'n_pro': 3,
        'n_ctx_ab': 3,
        'n_pro_ab': 4,
        'class_name': class_name,
        'k_shot': k_shot,
        'seed': seed,
        'use_cpu': 0,
        'device': device,
        'use_lap': True
    }
    
    model = PromptAD(**kwargs)
    model = model.to(device)
    print(f"  ✓ Model created")
    
    # Load checkpoint
    checkpoint_path = f"./result/baseline/{dataset}/k_{k_shot}/checkpoint/CLS-Seed_{seed}-{class_name}-check_point.pt"
    if not os.path.exists(checkpoint_path):
        print(f"  ✗ Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    model.load_state_dict(torch.load(checkpoint_path), strict=False)
    model.eval()
    print(f"  ✓ Checkpoint loaded: {checkpoint_path}")
    
except Exception as e:
    print(f"  ✗ Error loading model:")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 2: Load data
print("\n[Step 2/5] Loading normal support set...")
try:
    from PIL import Image
    from torch.utils.data import Dataset
    
    class SimpleImageDataset(Dataset):
        def __init__(self, img_paths, labels, transform=None):
            self.img_paths = img_paths
            self.labels = labels
            self.transform = transform
        
        def __len__(self):
            return len(self.img_paths)
        
        def __getitem__(self, idx):
            img = Image.open(self.img_paths[idx]).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, self.labels[idx]
    
    # load_mvtec returns (train_data, test_data), each is a tuple of 4 elements
    train_data, test_data = load_mvtec(class_name, k_shot)
    train_img, train_gt, train_label, train_type = train_data
    print(f"  ✓ Loaded {len(train_img)} normal samples")
    
    normal_dataset = SimpleImageDataset(
        train_img, train_label,
        transform=model.transform
    )
    
    normal_loader = DataLoader(
        normal_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )
    print(f"  ✓ DataLoader created")
    
except Exception as e:
    print(f"  ✗ Error loading data:")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 3: Extract anchors
print("\n[Step 3/5] Extracting MAP and LAP anchors...")
try:
    with torch.no_grad():
        # Get text embeddings from prompt learner
        normal_text_embeddings, abnormal_text_embeddings_map, abnormal_text_embeddings_lap = model.prompt_learner()
        
        print(f"  Normal embeddings shape: {normal_text_embeddings.shape}")
        print(f"  MAP embeddings shape: {abnormal_text_embeddings_map.shape}")
        print(f"  LAP embeddings shape: {abnormal_text_embeddings_lap.shape}")
        
        # Encode using the model's method with tokenized prompts
        normal_text_features = model.encode_text_embedding(
            normal_text_embeddings, 
            model.tokenized_normal_prompts
        )
        abnormal_text_features_map = model.encode_text_embedding(
            abnormal_text_embeddings_map, 
            model.tokenized_abnormal_prompts_handle
        )
        abnormal_text_features_lap = model.encode_text_embedding(
            abnormal_text_embeddings_lap, 
            model.tokenized_abnormal_prompts_learned
        )
        
        # Normalize
        normal_text_features = normal_text_features / normal_text_features.norm(dim=-1, keepdim=True)
        abnormal_text_features_map = abnormal_text_features_map / abnormal_text_features_map.norm(dim=-1, keepdim=True)
        abnormal_text_features_lap = abnormal_text_features_lap / abnormal_text_features_lap.norm(dim=-1, keepdim=True)
        
        # Average to get anchors
        mu_n = torch.mean(normal_text_features, dim=0, keepdim=True)
        mu_a_map = torch.mean(abnormal_text_features_map, dim=0, keepdim=True)
        mu_a_lap = torch.mean(abnormal_text_features_lap, dim=0, keepdim=True)
        mu_a_combined = torch.mean(torch.cat([abnormal_text_features_map, abnormal_text_features_lap], dim=0), dim=0, keepdim=True)
        
        # Re-normalize
        mu_n = mu_n / mu_n.norm(dim=-1, keepdim=True)
        mu_a_map = mu_a_map / mu_a_map.norm(dim=-1, keepdim=True)
        mu_a_lap = mu_a_lap / mu_a_lap.norm(dim=-1, keepdim=True)
        mu_a_combined = mu_a_combined / mu_a_combined.norm(dim=-1, keepdim=True)
        
        print(f"  ✓ Anchors extracted:")
        print(f"    μ_n shape: {mu_n.shape}")
        print(f"    μ_MAP shape: {mu_a_map.shape}")
        print(f"    μ_LAP shape: {mu_a_lap.shape}")
        
except Exception as e:
    print(f"  ✗ Error extracting anchors:")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 4: Compute margins on normal samples
print("\n[Step 4/5] Computing margins on normal samples...")
try:
    t = model.model.logit_scale
    margins_map = []
    margins_lap = []
    
    with torch.no_grad():
        for data, label in normal_loader:
            data = data.to(device)
            
            # Extract image features (returns [global_features, patch_features])
            global_features = model.encode_image(data)[0]  # [N, D]
            
            s_normal = t * (global_features @ mu_n.T).squeeze(-1)
            s_abnormal_map = t * (global_features @ mu_a_map.T).squeeze(-1)
            s_abnormal_lap = t * (global_features @ mu_a_lap.T).squeeze(-1)
            
            # Compute margins
            m_map = s_normal - s_abnormal_map
            m_lap = s_normal - s_abnormal_lap
            
            margins_map.extend(m_map.cpu().numpy())
            margins_lap.extend(m_lap.cpu().numpy())
    
    margins_map = np.array(margins_map)
    margins_lap = np.array(margins_lap)
    
    print(f"  ✓ Computed margins for {len(margins_map)} samples")
    print(f"    MAP margins: min={margins_map.min():.3f}, max={margins_map.max():.3f}, mean={margins_map.mean():.3f}")
    print(f"    LAP margins: min={margins_lap.min():.3f}, max={margins_lap.max():.3f}, mean={margins_lap.mean():.3f}")
    
except Exception as e:
    print(f"  ✗ Error computing margins:")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 5: Compute reliability metrics
print("\n[Step 5/5] Computing reliability metrics...")
try:
    epsilon = 0.05
    
    # Section I: Normal-side risk metrics
    metrics = {
        'R_MAP_0': np.mean(margins_map < 0),
        'R_LAP_0': np.mean(margins_lap < 0),
        'R_MAP_eps': np.mean(margins_map < epsilon),
        'R_LAP_eps': np.mean(margins_lap < epsilon),
        'margin_MAP_q10': np.percentile(margins_map, 10),
        'margin_MAP_median': np.median(margins_map),
        'margin_MAP_q90': np.percentile(margins_map, 90),
        'margin_MAP_mean': np.mean(margins_map),
        'margin_MAP_std': np.std(margins_map),
        'margin_LAP_q10': np.percentile(margins_lap, 10),
        'margin_LAP_median': np.median(margins_lap),
        'margin_LAP_q90': np.percentile(margins_lap, 90),
        'margin_LAP_mean': np.mean(margins_lap),
        'margin_LAP_std': np.std(margins_lap),
    }
    
    # Section II: Consistency metrics
    margin_discrepancy = np.abs(margins_map - margins_lap)
    opposite_signs = ((margins_map > 0) & (margins_lap < 0)) | ((margins_map < 0) & (margins_lap > 0))
    
    metrics.update({
        'margin_discrepancy_mean': np.mean(margin_discrepancy),
        'margin_discrepancy_median': np.median(margin_discrepancy),
        'margin_discrepancy_std': np.std(margin_discrepancy),
        'margin_discrepancy_max': np.max(margin_discrepancy),
        'disagreement_rate': np.mean(opposite_signs),
        'margin_correlation': np.corrcoef(margins_map, margins_lap)[0, 1],
    })
    
    # Section III: Geometric metrics
    with torch.no_grad():
        cos_n_map = (mu_n @ mu_a_map.T).item()
        cos_n_lap = (mu_n @ mu_a_lap.T).item()
        cos_map_lap = (mu_a_map @ mu_a_lap.T).item()
        cos_n_combined = (mu_n @ mu_a_combined.T).item()
        
        dist_n_map = torch.norm(mu_n - mu_a_map, p=2).item()
        dist_n_lap = torch.norm(mu_n - mu_a_lap, p=2).item()
        dist_map_lap = torch.norm(mu_a_map - mu_a_lap, p=2).item()
        dist_n_combined = torch.norm(mu_n - mu_a_combined, p=2).item()
    
    metrics.update({
        'cos_normal_MAP': cos_n_map,
        'cos_normal_LAP': cos_n_lap,
        'cos_MAP_LAP': cos_map_lap,
        'cos_normal_combined': cos_n_combined,
        'dist_normal_MAP': dist_n_map,
        'dist_normal_LAP': dist_n_lap,
        'dist_MAP_LAP': dist_map_lap,
        'dist_normal_combined': dist_n_combined,
    })
    
    print(f"  ✓ Computed {len(metrics)} metrics")
    
    # Display results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    print("\nSection I: Normal-side Risk Indicators:")
    print(f"  MAP Risk:")
    print(f"    R_0 (P(margin < 0)): {metrics['R_MAP_0']:.4f}")
    print(f"    R_ε (P(margin < {epsilon})): {metrics['R_MAP_eps']:.4f}")
    print(f"    Median margin: {metrics['margin_MAP_median']:.4f}")
    
    print(f"\n  LAP Risk:")
    print(f"    R_0 (P(margin < 0)): {metrics['R_LAP_0']:.4f}")
    print(f"    R_ε (P(margin < {epsilon})): {metrics['R_LAP_eps']:.4f}")
    print(f"    Median margin: {metrics['margin_LAP_median']:.4f}")
    
    print("\nSection II: Consistency Indicators:")
    print(f"  Margin discrepancy (mean): {metrics['margin_discrepancy_mean']:.4f}")
    print(f"  Disagreement rate: {metrics['disagreement_rate']:.4f}")
    print(f"  Margin correlation: {metrics['margin_correlation']:.4f}")
    
    print("\nSection III: Geometric Indicators:")
    print(f"  cos(μ_n, μ_MAP): {metrics['cos_normal_MAP']:.4f}")
    print(f"  cos(μ_n, μ_LAP): {metrics['cos_normal_LAP']:.4f}")
    print(f"  cos(μ_MAP, μ_LAP): {metrics['cos_MAP_LAP']:.4f}")
    print(f"  dist(μ_MAP, μ_LAP): {metrics['dist_MAP_LAP']:.4f}")
    
    print("\n" + "="*80)
    print("✓ TEST PASSED - All metrics computed successfully!")
    print("="*80)
    
except Exception as e:
    print(f"  ✗ Error computing metrics:")
    import traceback
    traceback.print_exc()
    sys.exit(1)
