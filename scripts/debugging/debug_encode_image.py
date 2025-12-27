#!/usr/bin/env python3
"""
Debug: Check what encode_image returns
"""
import torch
import sys
sys.path.insert(0, '/home/zju/mywork/PromptAD')

from PromptAD import PromptAD
from PIL import Image
import numpy as np

# Create model
kwargs = {
    'k_shot': 2,
    'img_resize': 240,
    'img_cropsize': 240,
    'n_ctx': 4,
    'n_ctx_ab': 1,
    'n_pro': 3,
    'n_pro_ab': 6,
    'class_name': 'bottle',
    'device': 'cuda:0',
    'resolution': 400
}

model = PromptAD(
    out_size_h=400,
    out_size_w=400,
    device='cuda:0',
    backbone='ViT-B-16-plus-240',
    pretrained_dataset='laion400m_e32',
    n_ctx=4,
    n_pro=3,
    n_ctx_ab=1,
    n_pro_ab=6,
    class_name='bottle',
    precision='fp16',
    **kwargs
)

# Create a dummy image
dummy_img = Image.new('RGB', (240, 240), color='red')
transformed = model.transform(dummy_img)
batch = transformed.unsqueeze(0).to('cuda:0')

# Encode
with torch.no_grad():
    features = model.encode_image(batch)
    
print(f"\nNumber of features returned: {len(features)}")
for i, feat in enumerate(features):
    print(f"  features[{i}]: shape={feat.shape}, device={feat.device}, dtype={feat.dtype}")

print("\n" + "="*80)
print("BASELINE expects 4 features:")
print("  features[0]: CLS token for text matching")
print("  features[1]: Patch tokens for text matching") 
print("  features[2]: CLS token for memory bank matching")
print("  features[3]: Patch tokens for memory bank matching")
print("="*80)
