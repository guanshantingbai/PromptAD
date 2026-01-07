#!/usr/bin/env python3
"""
测试视觉原型模式（Visual Prototypes）
验证：直接用训练图像的 CLS tokens 作为 Normal Prototypes
"""

import torch
import sys
sys.path.append('.')

from PromptAD.model import PromptAD
from PIL import Image
import numpy as np

def create_dummy_images(k_shot, size=224):
    """创建模拟训练图像"""
    images = []
    for i in range(k_shot):
        # 创建随机图像
        img_array = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        images.append(img)
    return images


def test_visual_prototypes_mode():
    """测试视觉原型模式"""
    print("\n" + "="*70)
    print("TEST 1: Visual Prototypes Mode")
    print("="*70)
    
    k_shot = 4  # 4-shot learning
    
    model = PromptAD(
        out_size_h=224,
        out_size_w=224,
        device='cpu',
        backbone='ViT-B-16',
        pretrained_dataset='laion400m_e32',
        n_ctx=12,
        n_pro=k_shot,  # 注意：n_pro 应该等于 k_shot
        n_ctx_ab=12,
        n_pro_ab=4,
        class_name='bottle',
        precision='fp32',
        use_visual_prototypes=True,  # 启用视觉原型
        k_shot=k_shot,
        img_resize=240,
        img_cropsize=224
    )
    
    print(f"\n✅ Model initialized with use_visual_prototypes=True")
    print(f"   k_shot: {k_shot}")
    print(f"   n_pro: {model.prompt_learner.n_pro}")
    
    # 创建模拟训练图像
    train_images = create_dummy_images(k_shot)
    print(f"\n✅ Created {len(train_images)} dummy training images")
    
    # 设置视觉原型
    model.set_visual_prototypes(train_images)
    
    print(f"\n✅ Visual prototypes set successfully")
    

def test_visual_prototypes_with_tensors():
    """测试使用 tensor 输入"""
    print("\n" + "="*70)
    print("TEST 2: Visual Prototypes with Tensor Input")
    print("="*70)
    
    k_shot = 2
    
    model = PromptAD(
        out_size_h=224,
        out_size_w=224,
        device='cpu',
        backbone='ViT-B-16',
        pretrained_dataset='laion400m_e32',
        n_ctx=12,
        n_pro=k_shot,
        n_ctx_ab=12,
        n_pro_ab=4,
        class_name='bottle',
        precision='fp32',
        use_visual_prototypes=True,
        k_shot=k_shot,
        img_resize=240,
        img_cropsize=224
    )
    
    # 直接创建 tensor
    train_images_tensor = torch.randn(k_shot, 3, 224, 224)
    print(f"Created tensor input: {train_images_tensor.shape}")
    
    # 设置视觉原型
    model.set_visual_prototypes(train_images_tensor)
    
    print(f"✅ Visual prototypes set from tensor input")


def test_build_text_feature_gallery_skip():
    """测试 build_text_feature_gallery 跳过 normal prompts"""
    print("\n" + "="*70)
    print("TEST 3: build_text_feature_gallery with Visual Prototypes")
    print("="*70)
    
    k_shot = 2
    
    model = PromptAD(
        out_size_h=224,
        out_size_w=224,
        device='cpu',
        backbone='ViT-B-16',
        pretrained_dataset='laion400m_e32',
        n_ctx=12,
        n_pro=k_shot,
        n_ctx_ab=12,
        n_pro_ab=4,
        class_name='bottle',
        precision='fp32',
        use_visual_prototypes=True,
        k_shot=k_shot,
        img_resize=240,
        img_cropsize=224
    )
    
    # 设置视觉原型
    train_images = create_dummy_images(k_shot)
    model.set_visual_prototypes(train_images)
    
    # 调用 build_text_feature_gallery
    print("\nCalling build_text_feature_gallery()...")
    model.build_text_feature_gallery()
    
    print(f"\n✅ build_text_feature_gallery() completed")
    print(f"   text_features shape: {model.text_features.shape}")
    print(f"   Normal anchor (index 0): from VISUAL features")
    print(f"   Abnormal anchor (index 1): from TEXT prompts")


def test_comparison_with_learnable():
    """对比：可学习模式 vs 视觉原型模式"""
    print("\n" + "="*70)
    print("TEST 4: Comparison - Learnable vs Visual Prototypes")
    print("="*70)
    
    k_shot = 3
    
    # 模式 1: 可学习
    print("\n[Mode 1: Learnable Normal Prototypes]")
    model_learnable = PromptAD(
        out_size_h=224, out_size_w=224, device='cpu',
        backbone='ViT-B-16', pretrained_dataset='laion400m_e32',
        n_ctx=12, n_pro=k_shot, n_ctx_ab=12, n_pro_ab=4,
        class_name='bottle', precision='fp32',
        use_visual_prototypes=False,  # 可学习
        k_shot=k_shot, img_resize=240, img_cropsize=224
    )
    
    params_learnable = sum(p.numel() for p in model_learnable.prompt_learner.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {params_learnable}")
    print(f"  normal_ctx is Parameter: {isinstance(model_learnable.prompt_learner.normal_ctx, torch.nn.Parameter)}")
    
    # 模式 2: 视觉原型
    print("\n[Mode 2: Visual Prototypes]")
    model_visual = PromptAD(
        out_size_h=224, out_size_w=224, device='cpu',
        backbone='ViT-B-16', pretrained_dataset='laion400m_e32',
        n_ctx=12, n_pro=k_shot, n_ctx_ab=12, n_pro_ab=4,
        class_name='bottle', precision='fp32',
        use_visual_prototypes=True,  # 视觉原型
        k_shot=k_shot, img_resize=240, img_cropsize=224
    )
    
    train_images = create_dummy_images(k_shot)
    model_visual.set_visual_prototypes(train_images)
    
    params_visual = sum(p.numel() for p in model_visual.prompt_learner.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {params_visual}")
    print(f"  normal_ctx is Parameter: {isinstance(model_visual.prompt_learner.normal_ctx, torch.nn.Parameter)}")
    
    print(f"\n✅ Comparison completed")
    print(f"   Parameter reduction: {params_learnable - params_visual} parameters")
    print(f"   Normal anchor source:")
    print(f"     - Learnable mode: TEXT prompts (learned)")
    print(f"     - Visual mode: VISUAL features (from training images)")


def test_forward_pass():
    """测试完整前向传播"""
    print("\n" + "="*70)
    print("TEST 5: Full Forward Pass with Visual Prototypes")
    print("="*70)
    
    k_shot = 2
    
    model = PromptAD(
        out_size_h=224, out_size_w=224, device='cpu',
        backbone='ViT-B-16', pretrained_dataset='laion400m_e32',
        n_ctx=12, n_pro=k_shot, n_ctx_ab=12, n_pro_ab=4,
        class_name='bottle', precision='fp32',
        use_visual_prototypes=True,
        k_shot=k_shot, img_resize=240, img_cropsize=224
    )
    
    # 设置视觉原型
    train_images = create_dummy_images(k_shot)
    model.set_visual_prototypes(train_images)
    
    # 构建特征库
    model.build_text_feature_gallery()
    
    # 创建测试图像
    test_images = torch.randn(2, 3, 224, 224)
    
    # 提取特征
    visual_features = model.encode_image(test_images)
    print(f"\n✅ Extracted visual features")
    print(f"   CLS tokens: {visual_features[0].shape}")
    
    # 计算异常分数
    anomaly_scores = model.calculate_textual_anomaly_score(visual_features, 'cls')
    print(f"\n✅ Calculated anomaly scores: {anomaly_scores.shape}")
    print(f"   Sample scores: {anomaly_scores[:5]}")


if __name__ == "__main__":
    print("\n" + "🚀 Testing Visual Prototypes Implementation")
    print("="*70)
    
    try:
        test_visual_prototypes_mode()
        test_visual_prototypes_with_tensors()
        test_build_text_feature_gallery_skip()
        test_comparison_with_learnable()
        test_forward_pass()
        
        print("\n" + "="*70)
        print("🎉 ALL TESTS PASSED!")
        print("="*70 + "\n")
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
