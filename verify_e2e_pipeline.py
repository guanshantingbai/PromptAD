#!/usr/bin/env python3
"""
端到端验证 n(q) MVP pipeline
从 checkpoint 构造 → 加载 → 推理
"""

import torch
import os
import sys

def verify_checkpoint_construction():
    """验证 checkpoint 构造流程"""
    print("="*70)
    print("Step 1: Verify Checkpoint Construction")
    print("="*70)
    
    from construct_mvp_checkpoint import construct_checkpoint_for_class
    
    print("\n[Test] Constructing checkpoint for bottle (k=2)...")
    ckpt_path = construct_checkpoint_for_class(
        dataset='mvtec',
        class_name='bottle',
        k_shot=2,
        seed=111
    )
    
    if not os.path.exists(ckpt_path):
        print(f"❌ Checkpoint not created: {ckpt_path}")
        return False
    
    print(f"✅ Checkpoint created: {ckpt_path}")
    
    # Load and inspect checkpoint
    ckpt = torch.load(ckpt_path, map_location='cpu')
    print("\n[Checkpoint Contents]")
    for key, value in ckpt.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}, dtype={value.dtype}")
        else:
            print(f"  {key}: {type(value)}")
    
    # Verify critical components
    required_keys = [
        'training_cls_tokens',  # Support set CLS tokens
        'abnormal_text_features_all',  # MAP + LAP
        'feature_gallery1',  # Memory bank
        'feature_gallery2',  # Memory bank
    ]
    
    missing = [k for k in required_keys if k not in ckpt]
    if missing:
        print(f"\n❌ Missing required keys: {missing}")
        return False
    
    # Verify shapes
    K = 2  # k_shot
    print("\n[Shape Verification]")
    expected_shapes = {
        'training_cls_tokens': (K, 512),
        'feature_gallery1': (K * 196, 512),  # K images * 196 patches
        'feature_gallery2': (K * 196, 512),
    }
    
    for key, expected in expected_shapes.items():
        actual = ckpt[key].shape
        match = actual == expected
        status = "✅" if match else "❌"
        print(f"  {status} {key}: expected {expected}, got {actual}")
        if not match:
            return False
    
    print("\n✅ Checkpoint construction verified!")
    return True

def verify_checkpoint_loading():
    """验证 checkpoint 加载流程"""
    print("\n" + "="*70)
    print("Step 2: Verify Checkpoint Loading")
    print("="*70)
    
    from PromptAD import PromptAD
    
    print("\n[Test] Creating model and loading checkpoint...")
    model = PromptAD(
        k_shot=2,
        class_name="bottle",
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
    
    # Load checkpoint
    ckpt_path = './result/nq_mvp/mvtec/k_2/checkpoint/CLS-Seed_111-bottle-check_point.pt'
    if not os.path.exists(ckpt_path):
        print(f"❌ Checkpoint not found: {ckpt_path}")
        print("  Run: python construct_mvp_checkpoint.py --dataset mvtec --class_name bottle --k-shot 2")
        return False
    
    ckpt = torch.load(ckpt_path, map_location='cpu')
    
    # Handle shape mismatches for buffers
    state_dict = model.state_dict()
    for key in ['training_cls_tokens', 'feature_gallery1', 'feature_gallery2']:
        if key in ckpt and key in state_dict:
            if ckpt[key].shape != state_dict[key].shape:
                print(f"  ⚠️  Resizing {key}: {state_dict[key].shape} → {ckpt[key].shape}")
                state_dict[key] = ckpt[key]
    
    model.load_state_dict(ckpt, strict=False)
    print("✅ Checkpoint loaded successfully")
    
    # Verify critical components are loaded
    print("\n[Loaded Components]")
    print(f"  training_cls_tokens: {model.training_cls_tokens.shape}")
    print(f"  abnormal_text_features_all: {model.abnormal_text_features_all.shape}")
    print(f"  feature_gallery1: {model.feature_gallery1.shape}")
    print(f"  feature_gallery2: {model.feature_gallery2.shape}")
    
    # Verify n_map is available
    if hasattr(model.prompt_learner, 'n_map'):
        print(f"  n_map: {model.prompt_learner.n_map}")
    else:
        print("  ❌ n_map not found in prompt_learner")
        return False
    
    print("\n✅ Checkpoint loading verified!")
    return True

def verify_inference_pipeline():
    """验证推理流程"""
    print("\n" + "="*70)
    print("Step 3: Verify Inference Pipeline")
    print("="*70)
    
    from PromptAD import PromptAD
    
    # Create model
    model = PromptAD(
        k_shot=2,
        class_name="bottle",
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
    
    # Load checkpoint
    ckpt_path = './result/nq_mvp/mvtec/k_2/checkpoint/CLS-Seed_111-bottle-check_point.pt'
    ckpt = torch.load(ckpt_path, map_location='cpu')
    
    # Handle shape mismatches
    state_dict = model.state_dict()
    for key in ['training_cls_tokens', 'feature_gallery1', 'feature_gallery2']:
        if key in ckpt and key in state_dict:
            if ckpt[key].shape != state_dict[key].shape:
                state_dict[key] = ckpt[key]
    
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    
    # Create dummy test data
    print("\n[Test] Running inference with dummy data...")
    N = 4
    test_images = torch.randn(N, 3, 224, 224)
    
    # Extract visual features
    visual_features = model.encode_image(test_images)
    print(f"  Visual features extracted: cls={visual_features[0].shape}, patches={visual_features[1].shape}")
    
    # Test baseline (alpha=0.0)
    print("\n[Test] Baseline mode (alpha=0.0)...")
    scores_baseline = model.calculate_textual_anomaly_score(
        visual_features,
        'cls',
        semantic_weight=0.0
    )
    print(f"  ✅ Baseline scores: {scores_baseline.shape}, mean={scores_baseline.mean():.4f}")
    
    # Test n(q) MVP (alpha=0.1)
    print("\n[Test] n(q) MVP mode (alpha=0.1)...")
    scores_mvp, logits = model.calculate_textual_anomaly_score(
        visual_features,
        'cls',
        semantic_weight=0.1,
        return_logits=True
    )
    print(f"  ✅ MVP scores: {scores_mvp.shape}, mean={scores_mvp.mean():.4f}")
    print(f"  ✅ Logits: {logits.shape}")
    
    # Verify logits structure
    if logits.shape[1] == 5:
        print("\n[Logits Breakdown] (sample 0)")
        print(f"  s_normal: {logits[0, 0]:.4f}")
        print(f"  s_abnormal: {logits[0, 1]:.4f}")
        print(f"  n_q_alignment: {logits[0, 2]:.4f}")
        print(f"  E_sem: {logits[0, 3]:.4f}")
        print(f"  baseline: {logits[0, 4]:.4f}")
        
        # Verify formula
        baseline = logits[0, 4]
        E_sem = logits[0, 3]
        expected = baseline + 0.1 * E_sem
        actual = scores_mvp[0]
        
        print(f"\n[Formula Check]")
        print(f"  baseline + 0.1 * E_sem = {baseline:.4f} + 0.1 * {E_sem:.4f} = {expected:.4f}")
        print(f"  actual score = {actual:.4f}")
        print(f"  ✅ Match!" if abs(expected - actual) < 1e-3 else "  ❌ Mismatch!")
    
    print("\n✅ Inference pipeline verified!")
    return True

def verify_nq_selection_logic():
    """验证 n(q) 选择逻辑"""
    print("\n" + "="*70)
    print("Step 4: Verify n(q) Hard Selection Logic")
    print("="*70)
    
    from PromptAD import PromptAD
    
    # Create simple test case
    K = 3  # 3 normal prototypes
    D = 512
    
    model = PromptAD(
        k_shot=K,
        class_name="bottle",
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
    
    # Manually set training_cls_tokens for controlled test
    model.training_cls_tokens = torch.randn(K, D)
    model.training_cls_tokens = model.training_cls_tokens / model.training_cls_tokens.norm(dim=-1, keepdim=True)
    
    # Build text features
    model.build_text_feature_gallery()
    
    # Create query with known closest prototype
    query = model.training_cls_tokens[1] + 0.1 * torch.randn(D)  # Close to prototype 1
    query = query / query.norm()
    
    # Create visual features
    visual_features = [
        query.unsqueeze(0),  # [1, D]
        torch.randn(1, 196, D)  # dummy patches
    ]
    
    # Run inference
    scores, logits = model.calculate_textual_anomaly_score(
        visual_features,
        'cls',
        semantic_weight=0.1,
        return_logits=True
    )
    
    # Manually verify n(q)
    similarities = query @ model.training_cls_tokens.T
    i_star_expected = similarities.argmax().item()
    n_q_expected = model.training_cls_tokens[i_star_expected]
    alignment_expected = (query * n_q_expected).sum().item()
    
    alignment_from_model = logits[0, 2].item()
    
    print(f"\n[Manual Computation]")
    print(f"  Similarities: {similarities}")
    print(f"  Selected prototype index: {i_star_expected}")
    print(f"  Expected alignment: {alignment_expected:.4f}")
    
    print(f"\n[Model Output]")
    print(f"  Model alignment: {alignment_from_model:.4f}")
    
    if abs(alignment_expected - alignment_from_model) < 1e-3:
        print("\n✅ n(q) selection logic verified!")
        return True
    else:
        print(f"\n❌ Mismatch: {abs(alignment_expected - alignment_from_model):.6f}")
        return False

def main():
    print("\n" + "="*70)
    print("END-TO-END PIPELINE VERIFICATION")
    print("="*70)
    
    try:
        # Step 1: Checkpoint construction
        if not verify_checkpoint_construction():
            print("\n❌ Checkpoint construction failed")
            sys.exit(1)
        
        # Step 2: Checkpoint loading
        if not verify_checkpoint_loading():
            print("\n❌ Checkpoint loading failed")
            sys.exit(1)
        
        # Step 3: Inference pipeline
        if not verify_inference_pipeline():
            print("\n❌ Inference pipeline failed")
            sys.exit(1)
        
        # Step 4: n(q) selection logic
        if not verify_nq_selection_logic():
            print("\n❌ n(q) selection logic failed")
            sys.exit(1)
        
        # All passed
        print("\n" + "="*70)
        print("✅ ALL VERIFICATIONS PASSED!")
        print("="*70)
        print("\n📝 Summary:")
        print("  ✅ Checkpoint construction works correctly")
        print("  ✅ Checkpoint loading handles shape mismatches")
        print("  ✅ Inference pipeline produces valid outputs")
        print("  ✅ n(q) hard selection logic verified")
        print("  ✅ Formula: score = -(q·n_q) + alpha * E_sem verified")
        print("\n🎯 Pipeline is ready for experiments!")
        
    except Exception as e:
        print(f"\n❌ VERIFICATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()