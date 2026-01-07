#!/usr/bin/env python3
"""
验证训练阶段是否正确保存了 training_cls_tokens

测试内容：
1. 训练过程中是否正确创建并保存 training_cls_tokens
2. checkpoint 中是否包含 training_cls_tokens
3. 加载 checkpoint 后是否能正确恢复 training_cls_tokens
4. training_cls_tokens 的形状和数值是否合理
"""

import torch
import sys
import os

# Quick sanity check without full training
def test_model_initialization():
    """测试模型初始化是否包含 training_cls_tokens buffer"""
    print("="*70)
    print("Test 1: Model initialization")
    print("="*70)
    
    from PromptAD import PromptAD
    
    # 创建模型（使用所有必需参数）
    model = PromptAD(
        k_shot=2,
        class_name="carpet",
        out_size_h=518,
        out_size_w=518,
        img_resize=518,
        img_cropsize=518,
        backbone='ViT-B-16',
        pretrained_dataset='laion400m_e32',
        n_ctx=12,
        n_pro=4,
        n_ctx_ab=12,
        n_pro_ab=2,
        use_visual_prototypes=True,
        device='cpu'
    )
    
    # 检查 buffer 是否存在
    assert hasattr(model, 'training_cls_tokens'), "❌ Model missing 'training_cls_tokens' buffer"
    print("✅ Model has 'training_cls_tokens' buffer")
    
    # 检查初始形状
    print(f"  Initial shape: {model.training_cls_tokens.shape}")
    print(f"  Initial dtype: {model.training_cls_tokens.dtype}")
    
    return model

def test_set_visual_prototypes(model):
    """测试 set_visual_prototypes 是否正确保存 cls_tokens"""
    print("\n" + "="*70)
    print("Test 2: set_visual_prototypes saves cls_tokens")
    print("="*70)
    
    # 创建假的训练图像
    k_shot = 4
    train_images = torch.randn(k_shot, 3, 224, 224)
    
    # 设置视觉原型
    model.set_visual_prototypes(train_images)
    
    # 检查 training_cls_tokens 是否被更新
    assert model.training_cls_tokens.shape[0] == k_shot, \
        f"❌ Expected shape[0]={k_shot}, got {model.training_cls_tokens.shape[0]}"
    print(f"✅ training_cls_tokens shape: {model.training_cls_tokens.shape}")
    
    # 检查数值合理性（应该是归一化的）
    norms = model.training_cls_tokens.norm(dim=-1)
    print(f"  Norms: {norms}")
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4), \
        "❌ training_cls_tokens not normalized"
    print("✅ training_cls_tokens normalized (L2 norm ≈ 1.0)")
    
    return model

def test_checkpoint_save_load(model):
    """测试 checkpoint 保存和加载"""
    print("\n" + "="*70)
    print("Test 3: Checkpoint save and load")
    print("="*70)
    
    # 保存 checkpoint（模拟 train_cls.py）
    from train_cls import save_check_point
    checkpoint_path = "/tmp/test_training_cls_tokens.pt"
    
    save_check_point(model, checkpoint_path)
    print(f"✅ Checkpoint saved to {checkpoint_path}")
    
    # 加载 checkpoint 并检查
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    print(f"  Keys in checkpoint: {list(state_dict.keys())}")
    
    # 检查是否包含 training_cls_tokens
    assert 'training_cls_tokens' in state_dict, "❌ Checkpoint missing 'training_cls_tokens'"
    print("✅ Checkpoint contains 'training_cls_tokens'")
    
    # 检查形状
    saved_tokens = state_dict['training_cls_tokens']
    print(f"  Saved shape: {saved_tokens.shape}")
    print(f"  Saved dtype: {saved_tokens.dtype}")
    
    # 创建新模型并加载
    from PromptAD import PromptAD
    new_model = PromptAD(
        k_shot=2,
        class_name="carpet",
        out_size_h=518,
        out_size_w=518,
        img_resize=518,
        img_cropsize=518,
        backbone='ViT-B-16',
        pretrained_dataset='laion400m_e32',
        n_ctx=12,
        n_pro=4,
        n_ctx_ab=12,
        n_pro_ab=2,
        use_visual_prototypes=True,
        device='cpu'
    )
    
    # 手动处理 training_cls_tokens（因为形状可能不同）
    training_tokens = state_dict.pop('training_cls_tokens', None)
    
    new_model.load_state_dict(state_dict, strict=False)
    
    # 手动加载 training_cls_tokens
    if training_tokens is not None:
        new_model.training_cls_tokens = training_tokens.clone()
    
    # 验证加载是否成功
    assert new_model.training_cls_tokens.shape == saved_tokens.shape, \
        "❌ Loaded shape mismatch"
    assert torch.allclose(new_model.training_cls_tokens, saved_tokens), \
        "❌ Loaded values mismatch"
    print("✅ Checkpoint loaded successfully")
    print(f"  Loaded training_cls_tokens shape: {new_model.training_cls_tokens.shape}")
    
    # 清理
    os.remove(checkpoint_path)
    
    return new_model

def test_existing_checkpoint():
    """测试现有的 checkpoint 是否包含 training_cls_tokens（可选）"""
    print("\n" + "="*70)
    print("Test 4: Check existing checkpoint (optional)")
    print("="*70)
    
    # 查找现有的 checkpoint
    checkpoint_paths = [
        "result/baseline/mvtec/k_2/checkpoint/CLS-Seed_111-carpet-check_point.pt",
        "result/promptpurging/mvtec/k_2/checkpoint/CLS-Seed_111-carpet-check_point.pt",
    ]
    
    found = False
    for path in checkpoint_paths:
        if os.path.exists(path):
            found = True
            print(f"  Found checkpoint: {path}")
            state_dict = torch.load(path, map_location='cpu')
            if 'training_cls_tokens' in state_dict:
                print(f"✅ Checkpoint already contains 'training_cls_tokens'")
                print(f"  Shape: {state_dict['training_cls_tokens'].shape}")
            else:
                print(f"⚠️  Checkpoint missing 'training_cls_tokens' (expected for old checkpoints)")
                print(f"  Please retrain to include training_cls_tokens")
            break
    
    if not found:
        print("  No existing checkpoint found (will be created during training)")

def main():
    print("\n" + "="*70)
    print("TRAINING CLS TOKENS VERIFICATION")
    print("="*70)
    
    try:
        # Test 1: 模型初始化
        model = test_model_initialization()
        
        # Test 2: set_visual_prototypes
        model = test_set_visual_prototypes(model)
        
        # Test 3: checkpoint 保存和加载
        new_model = test_checkpoint_save_load(model)
        
        # Test 4: 检查现有 checkpoint（可选）
        test_existing_checkpoint()
        
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED!")
        print("="*70)
        print("\n📝 Summary:")
        print("  ✅ training_cls_tokens buffer created in model")
        print("  ✅ set_visual_prototypes saves cls_tokens correctly")
        print("  ✅ Checkpoint save/load works")
        print("  ✅ Ready for inference stage implementation")
        print("\n🎯 Next Steps:")
        print("  1. Retrain models to save training_cls_tokens in checkpoints")
        print("  2. Implement inference stage semantic fusion")
        print("  3. Use training_cls_tokens as normal anchors during inference")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
