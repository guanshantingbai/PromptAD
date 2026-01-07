#!/usr/bin/env python3
"""
测试流形模式（Manifold Normal Features）的正确性
验证：
1. use_manifold_normal=False 时，normal_ctx 是可学习参数
2. use_manifold_normal=True 时，normal_ctx 是固定 buffer
3. set_manifold_normal_features() 方法能正确设置特征
"""

import torch
import sys
sys.path.append('.')

from PromptAD.model import PromptAD

def test_learnable_mode():
    """测试传统可学习模式"""
    print("\n" + "="*70)
    print("TEST 1: Learnable Mode (use_manifold_normal=False)")
    print("="*70)
    
    model = PromptAD(
        out_size_h=224, 
        out_size_w=224,
        device='cpu',
        backbone='ViT-B-16',
        pretrained_dataset='laion400m_e32',
        n_ctx=12,
        n_pro=4,
        n_ctx_ab=12,
        n_pro_ab=4,
        class_name='bottle',
        precision='fp32',
        use_manifold_normal=False,  # 传统模式
        k_shot=2,
        img_resize=240,
        img_cropsize=224
    )
    
    # 检查 normal_ctx 类型
    assert isinstance(model.prompt_learner.normal_ctx, torch.nn.Parameter), \
        "❌ In learnable mode, normal_ctx should be a Parameter"
    print("✅ normal_ctx is a learnable Parameter")
    
    # 检查是否在优化器中
    params = list(model.prompt_learner.parameters())
    normal_ctx_in_params = any(p is model.prompt_learner.normal_ctx for p in params)
    assert normal_ctx_in_params, \
        "❌ normal_ctx should be in trainable parameters"
    print(f"✅ normal_ctx is in trainable parameters (total: {len(params)})")
    
    print(f"   Shape: {model.prompt_learner.normal_ctx.shape}")
    print(f"   Requires grad: {model.prompt_learner.normal_ctx.requires_grad}")
    

def test_manifold_mode():
    """测试流形模式"""
    print("\n" + "="*70)
    print("TEST 2: Manifold Mode (use_manifold_normal=True)")
    print("="*70)
    
    model = PromptAD(
        out_size_h=224,
        out_size_w=224,
        device='cpu',
        backbone='ViT-B-16',
        pretrained_dataset='laion400m_e32',
        n_ctx=12,
        n_pro=4,
        n_ctx_ab=12,
        n_pro_ab=4,
        class_name='bottle',
        precision='fp32',
        use_manifold_normal=True,  # 流形模式
        k_shot=2,
        img_resize=240,
        img_cropsize=224
    )
    
    # 检查 normal_ctx 类型（应该是 buffer，不是 Parameter）
    assert not isinstance(model.prompt_learner.normal_ctx, torch.nn.Parameter), \
        "❌ In manifold mode, normal_ctx should NOT be a Parameter"
    print("✅ normal_ctx is a buffer (not a Parameter)")
    
    # 检查是否不在优化器中
    params = [p for p in model.prompt_learner.parameters()]
    normal_ctx_in_params = any(p is model.prompt_learner.normal_ctx for p in params)
    assert not normal_ctx_in_params, \
        "❌ normal_ctx should NOT be in trainable parameters"
    print(f"✅ normal_ctx is NOT in trainable parameters (total: {len(params)})")
    
    # 检查初始值为零
    assert torch.allclose(model.prompt_learner.normal_ctx, torch.zeros_like(model.prompt_learner.normal_ctx)), \
        "❌ Initial normal_ctx should be zeros"
    print("✅ Initial normal_ctx is zeros (placeholder)")
    
    print(f"   Shape: {model.prompt_learner.normal_ctx.shape}")
    print(f"   Requires grad: {model.prompt_learner.normal_ctx.requires_grad}")
    

def test_set_manifold_features():
    """测试设置流形特征"""
    print("\n" + "="*70)
    print("TEST 3: Setting Manifold Features")
    print("="*70)
    
    n_pro, n_ctx, ctx_dim = 4, 12, 512
    
    model = PromptAD(
        out_size_h=224,
        out_size_w=224,
        device='cpu',
        backbone='ViT-B-16',
        pretrained_dataset='laion400m_e32',
        n_ctx=n_ctx,
        n_pro=n_pro,
        n_ctx_ab=12,
        n_pro_ab=4,
        class_name='bottle',
        precision='fp32',
        use_manifold_normal=True,
        k_shot=2,
        img_resize=240,
        img_cropsize=224
    )
    
    # 生成模拟的流形特征（注意：模型用的是 fp32，但内部会转换为 fp16）
    manifold_features = torch.randn(n_pro, n_ctx, ctx_dim)
    print(f"Generated manifold features: {manifold_features.shape}")
    
    # 设置流形特征
    model.set_manifold_normal_features(manifold_features)
    
    # 验证设置成功（考虑精度转换）
    expected = manifold_features.to(dtype=torch.float16)
    assert torch.allclose(model.prompt_learner.normal_ctx, expected, atol=1e-3), \
        "❌ Manifold features not set correctly"
    print("✅ Manifold features set successfully")
    print(f"   Norm (mean): {model.prompt_learner.normal_ctx.norm(dim=-1).mean().item():.4f}")
    

def test_forward_pass():
    """测试前向传播"""
    print("\n" + "="*70)
    print("TEST 4: Forward Pass with Manifold Features")
    print("="*70)
    
    model = PromptAD(
        out_size_h=224,
        out_size_w=224,
        device='cpu',
        backbone='ViT-B-16',
        pretrained_dataset='laion400m_e32',
        n_ctx=12,
        n_pro=4,
        n_ctx_ab=12,
        n_pro_ab=4,
        class_name='bottle',
        precision='fp32',
        use_manifold_normal=True,
        k_shot=2,
        img_resize=240,
        img_cropsize=224
    )
    
    # 设置流形特征
    manifold_features = torch.randn(4, 12, 512)
    model.set_manifold_normal_features(manifold_features)
    
    # 测试 PromptLearner 的 forward
    try:
        normal_prompts, abnormal_handle, abnormal_learned = model.prompt_learner()
        print("✅ PromptLearner forward pass successful")
        print(f"   Normal prompts: {normal_prompts.shape}")
        print(f"   Abnormal handle: {abnormal_handle.shape}")
        print(f"   Abnormal learned: {abnormal_learned.shape}")
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        raise


def test_backward_compatibility():
    """测试向后兼容性（不传 use_manifold_normal）"""
    print("\n" + "="*70)
    print("TEST 5: Backward Compatibility (no use_manifold_normal param)")
    print("="*70)
    
    try:
        model = PromptAD(
            out_size_h=224,
            out_size_w=224,
            device='cpu',
            backbone='ViT-B-16',
            pretrained_dataset='laion400m_e32',
            n_ctx=12,
            n_pro=4,
            n_ctx_ab=12,
            n_pro_ab=4,
            class_name='bottle',
            precision='fp32',
            # 不传 use_manifold_normal，应该默认为 False
            k_shot=2,
            img_resize=240,
            img_cropsize=224
        )
        
        assert isinstance(model.prompt_learner.normal_ctx, torch.nn.Parameter), \
            "❌ Default should be learnable mode"
        print("✅ Backward compatible: default is learnable mode")
        
    except Exception as e:
        print(f"❌ Backward compatibility test failed: {e}")
        raise


if __name__ == "__main__":
    print("\n" + "🚀 Testing Manifold Normal Features Implementation")
    print("="*70)
    
    try:
        test_learnable_mode()
        test_manifold_mode()
        test_set_manifold_features()
        test_forward_pass()
        test_backward_compatibility()
        
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
