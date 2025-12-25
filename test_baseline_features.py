#!/usr/bin/env python3
"""
Test script to check encode_image output in baseline
"""
import torch
from PromptAD import PromptAD
from PIL import Image

# Create model with proper parameters
kwargs = {
    'k_shot': 2,
    'img_resize': 240,
    'img_cropsize': 240,
}

model = PromptAD(
    out_size_h=400,
    out_size_w=400,
    device='cuda:0',
    backbone='ViT-B-16-plus-240',
    pretrained_dataset='laion400m_e32',
    n_ctx=4,
    n_pro=1,  # baseline uses 1
    n_ctx_ab=1,
    n_pro_ab=4,  # baseline uses 4
    class_name='bottle',
    precision='fp16',
    **kwargs
)

model = model.cuda()
model.eval()

# Create dummy image
dummy_batch = torch.randn(2, 3, 240, 240).cuda()

with torch.no_grad():
    features = model.encode_image(dummy_batch)
    
print("="*80)
print(f"encode_image() returns {len(features)} features:")
print("="*80)
for i, f in enumerate(features):
    print(f"features[{i}].shape = {f.shape}")
print("="*80)
