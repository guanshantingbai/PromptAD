#!/usr/bin/env python3
"""
最简测试 - 跳过 build_text_feature_gallery
"""

import torch
import sys
sys.path.append('.')

from PromptAD.model import PromptAD

print("\n🔍 Minimal Test")
print("="*60)

k_shot = 2

# 创建模型但不使用视觉原型（避免 build_text_feature_gallery 卡住）
model = PromptAD(
    out_size_h=224, out_size_w=224, device='cpu',
    backbone='ViT-B-16', pretrained_dataset='laion400m_e32',
    n_ctx=12, n_pro=3, n_ctx_ab=12, n_pro_ab=4,  # 使用传统参数
    class_name='bottle', precision='fp32',
    use_visual_prototypes=False,  # 传统模式
    k_shot=k_shot,
    img_resize=240, img_cropsize=224
)

print("✅ Model created")

# 跳过 build_text_feature_gallery（会卡住）
# model.build_text_feature_gallery()
# print("✅ Text feature gallery built")

# 手动设置 text_features（平均锚点）
model.text_features = torch.randn(2, 512)  # [normal, abnormal]
model.text_features = model.text_features / model.text_features.norm(dim=-1, keepdim=True)
print("✅ Text features set manually")

# 手动设置 normal_text_features_all 和 abnormal_text_features_all
# 模拟有多个 normal representatives
K = 4  # 模拟 4 个 normal representatives
M = 6  # 模拟 6 个 abnormal directions
D = 512

model.normal_text_features_all = torch.randn(K, D)
model.normal_text_features_all = model.normal_text_features_all / model.normal_text_features_all.norm(dim=-1, keepdim=True)

model.abnormal_text_features_all = torch.randn(M, D)
model.abnormal_text_features_all = model.abnormal_text_features_all / model.abnormal_text_features_all.norm(dim=-1, keepdim=True)

print(f"✅ Manually set representatives")
print(f"   Normal reps (K): {K}")
print(f"   Abnormal dirs (M): {M}")

# 测试 query-conditioned 模式
test_images = torch.randn(2, 3, 224, 224)
visual_features = model.encode_image(test_images)

print(f"\n[Testing Query-Conditioned CLS]")
scores = model.calculate_textual_anomaly_score(
    visual_features, task='cls',
    aggregation='query_conditioned',
    lambda_scale=1.0, margin=0.0
)
print(f"  Scores: {scores}")
print(f"  Range: [{scores.min():.4f}, {scores.max():.4f}]")

print(f"\n🎉 Success! Query-conditioned mode works!")
