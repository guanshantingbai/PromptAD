"""
Debug script to check the actual shapes of visual_features returned by encode_image.
"""

import torch
from PromptAD.model import PromptAD

# Initialize model
kwargs = {
    'dataset': 'mvtec',
    'class_name': 'bottle',
    'device': 'cuda:0',
    'k_shot': 2,
    'precision': 'fp16',
    'backbone': 'ViT-B-16-plus-240',
    'pretrained_dataset': 'laion400m_e32',
    'version': 'V2',
    'n_ctx': 4,
    'n_ctx_ab': 1,
    'n_pro': 3,
    'n_pro_ab': 1,
    'seed': 111
}

model = PromptAD(**kwargs)
model = model.cuda()
model.eval()

# Create dummy input
dummy_input = torch.randn(2, 3, 240, 240).cuda()

# Encode
with torch.no_grad():
    visual_features = model.encode_image(dummy_input)

print(f"Number of features returned: {len(visual_features)}")
print()

for i, feat in enumerate(visual_features):
    if isinstance(feat, torch.Tensor):
        print(f"visual_features[{i}] shape: {feat.shape}")
        print(f"visual_features[{i}] dtype: {feat.dtype}")
        print(f"visual_features[{i}] device: {feat.device}")
    else:
        print(f"visual_features[{i}]: Not a tensor - {type(feat)}")
    print()

# Check if mid_features exist
if hasattr(model.model.visual, 'mid_feature1'):
    print(f"mid_feature1 shape: {model.model.visual.mid_feature1.shape}")
if hasattr(model.model.visual, 'mid_feature2'):
    print(f"mid_feature2 shape: {model.model.visual.mid_feature2.shape}")
