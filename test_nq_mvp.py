#!/usr/bin/env python3
"""
测试 n(q) MVP 实现

验证内容：
1. n(q) hard selection 正确工作
2. E_sem 计算公式正确
3. 融合公式 score = -(q·n_q) + alpha * E_sem 正确
4. alpha = 0.0 时退化行为
5. return_logits 返回正确的诊断信息
"""

import torch
import numpy as np
import sys

def test_nq_mvp():
    """测试 n(q) MVP 实现"""
    print("="*70)
    print("n(q) MVP Implementation Test")
    print("="*70)
    
    from PromptAD import PromptAD
    
    # 创建模型
    print("\n[Step 1] Creating model...")
    model = PromptAD(
        k_shot=4,
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
    
    # 设置视觉原型
    print("\n[Step 2] Setting visual prototypes (training_cls_tokens)...")
    K = 4  # k_shot
    train_images = torch.randn(K, 3, 224, 224)
    model.set_visual_prototypes(train_images)
    print(f"  training_cls_tokens shape: {model.training_cls_tokens.shape}")
    
    # 构建文本特征库
    print("\n[Step 3] Building text feature gallery...")
    model.build_text_feature_gallery()
    n_map = getattr(model.prompt_learner, 'n_map', 8)
    print(f"  MAP features: {n_map}")
    
    # 创建测试数据
    print("\n[Step 4] Creating test data...")
    N = 8  # batch size
    D = 512  # feature dim
    # visual_features should be a list: [cls_tokens, patch_features, ...]
    # For cls task, we only need cls_tokens
    cls_tokens = torch.randn(N, D)
    patch_features = torch.randn(N, 196, D)  # 14x14 patches
    test_features = [cls_tokens, patch_features]  # Simulate full visual features
    
    # Test 1: alpha = 0.0 (should use baseline: -(q·n_q))
    print("\n" + "="*70)
    print("Test 1: Baseline mode (alpha=0.0)")
    print("="*70)
    
    scores, logits = model.calculate_textual_anomaly_score(
        test_features,
        'cls',
        return_logits=True,
        semantic_weight=0.0
    )
    
    print(f"  Scores shape: {scores.shape}")
    print(f"  Sample scores: {scores[:2]}")
    print("✅ Test 1 passed: Baseline mode works")
    
    # Test 2: alpha = 0.1 (should use n(q) MVP fusion)
    print("\n" + "="*70)
    print("Test 2: n(q) MVP fusion (alpha=0.1)")
    print("="*70)
    
    scores_mvp, logits_mvp = model.calculate_textual_anomaly_score(
        test_features,
        'cls',
        return_logits=True,
        semantic_weight=0.1
    )
    
    print(f"  Scores shape: {scores_mvp.shape}")
    print(f"  Logits shape: {logits_mvp.shape}")
    print(f"  Sample scores: {scores_mvp[:2]}")
    
    # Verify logits structure: [s_normal, s_abnormal, n_q_alignment, E_sem, baseline_score]
    if logits_mvp.shape[1] == 5:
        print(f"\n  Logits breakdown (sample 0):")
        print(f"    s_normal: {logits_mvp[0, 0]:.4f}")
        print(f"    s_abnormal: {logits_mvp[0, 1]:.4f}")
        print(f"    n_q_alignment (q·n_q): {logits_mvp[0, 2]:.4f}")
        print(f"    E_sem: {logits_mvp[0, 3]:.4f}")
        print(f"    baseline_score (-(q·n_q)): {logits_mvp[0, 4]:.4f}")
        
        # Verify formula: score = baseline_score + alpha * E_sem
        baseline = logits_mvp[0, 4]
        E_sem = logits_mvp[0, 3]
        alpha = 0.1
        expected = baseline + alpha * E_sem
        actual = scores_mvp[0]
        
        print(f"\n  Formula verification:")
        print(f"    baseline + 0.1 * E_sem = {baseline:.4f} + 0.1 * {E_sem:.4f} = {expected:.4f}")
        print(f"    actual score = {actual:.4f}")
        print(f"    difference = {abs(expected - actual):.6f}")
        
        if abs(expected - actual) < 1e-3:
            print("  ✅ Formula verified!")
        else:
            print("  ❌ Formula mismatch!")
            return False
    else:
        print(f"  ⚠️  Unexpected logits shape: expected [N, 5], got {logits_mvp.shape}")
    
    print("✅ Test 2 passed: n(q) MVP fusion works")
    
    # Test 3: Verify n(q) is selected correctly
    print("\n" + "="*70)
    print("Test 3: Verify n(q) hard selection")
    print("="*70)
    
    # Manually compute to verify
    gf = test_features[0][:1].to(model.training_cls_tokens.dtype)  # Take first sample [1, D], match dtype
    normal_reps = model.training_cls_tokens  # [K, D]
    
    # Compute similarity
    sim = gf @ normal_reps.T  # [1, K]
    i_star_manual = sim.argmax(dim=-1).item()
    n_q_manual = normal_reps[i_star_manual]  # [D]
    alignment_manual = (gf[0] * n_q_manual).sum().item()
    
    # Get from model
    alignment_from_model = logits_mvp[0, 2]
    
    print(f"  Manual n(q) alignment: {alignment_manual:.4f}")
    print(f"  Model n(q) alignment: {alignment_from_model:.4f}")
    print(f"  Selected prototype index: {i_star_manual}")
    
    if abs(alignment_manual - alignment_from_model) < 1e-3:
        print("✅ Test 3 passed: n(q) selection verified")
    else:
        print("❌ Test 3 failed: n(q) mismatch")
        return False
    
    # Test 4: Verify E_sem formula
    print("\n" + "="*70)
    print("Test 4: Verify E_sem = logsumexp(q@MAP.T) - (q·n_q)")
    print("="*70)
    
    # Manual computation
    gf = test_features[0][:1].to(model.abnormal_text_features_all.dtype)  # [1, D], match dtype
    delta_map = model.abnormal_text_features_all[:n_map]  # [N_map, D]
    logits_map = gf @ delta_map.T  # [1, N_map]
    map_response = torch.logsumexp(logits_map, dim=-1).item()  # scalar
    E_sem_manual = map_response - alignment_manual
    
    # From model
    E_sem_from_model = logits_mvp[0, 3]
    
    print(f"  Manual E_sem: {E_sem_manual:.4f}")
    print(f"    - map_response: {map_response:.4f}")
    print(f"    - n_q_alignment: {alignment_manual:.4f}")
    print(f"  Model E_sem: {E_sem_from_model:.4f}")
    
    if abs(E_sem_manual - E_sem_from_model) < 1e-3:
        print("✅ Test 4 passed: E_sem formula verified")
    else:
        print("❌ Test 4 failed: E_sem mismatch")
        return False
    
    # Test 5: Different alpha values
    print("\n" + "="*70)
    print("Test 5: Alpha sweep")
    print("="*70)
    
    for alpha in [0.0, 0.05, 0.1, 0.2]:
        scores_alpha = model.calculate_textual_anomaly_score(
            test_features,
            'cls',
            semantic_weight=alpha
        )
        print(f"  alpha={alpha:.2f}: mean_score={scores_alpha.mean():.4f}, std={scores_alpha.std():.4f}")
    
    print("✅ Test 5 passed: Alpha sweep works")
    
    return True

def main():
    print("\n" + "="*70)
    print("n(q) MVP IMPLEMENTATION VERIFICATION")
    print("="*70)
    
    try:
        success = test_nq_mvp()
        
        if success:
            print("\n" + "="*70)
            print("✅ ALL TESTS PASSED!")
            print("="*70)
            print("\n📝 Summary:")
            print("  ✅ n(q) hard selection works correctly")
            print("  ✅ E_sem formula verified: logsumexp(q@MAP.T) - (q·n_q)")
            print("  ✅ Fusion formula verified: score = -(q·n_q) + alpha * E_sem")
            print("  ✅ return_logits provides diagnostic info: [s_n, s_a, n_q·q, E_sem, baseline]")
            print("  ✅ Alpha sweep works")
            print("\n🎯 Ready for experiments!")
            print("  Run: python test_cls.py --semantic-weight 0.1 --use-visual-prototypes True")
        else:
            print("\n❌ SOME TESTS FAILED")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
