#!/usr/bin/env python3
"""
简化测试 Query-Conditioned 模式
"""

import torch
import sys
sys.path.append('.')

from PromptAD.model import PromptAD

def quick_test():
    print("\n🔍 Quick Test: Query-Conditioned Mode")
    print("="*60)
    
    k_shot = 2
    
    # 创建模型
    model = PromptAD(
        out_size_h=224, out_size_w=224, device='cpu',
        backbone='ViT-B-16', pretrained_dataset='laion400m_e32',
        n_ctx=12, n_pro=k_shot, n_ctx_ab=12, n_pro_ab=4,
        class_name='bottle', precision='fp32',
        use_visual_prototypes=True, k_shot=k_shot,
        img_resize=240, img_cropsize=224
    )
    
    # 设置视觉原型
    train_images = torch.randn(k_shot, 3, 224, 224)
    model.set_visual_prototypes(train_images)
    model.build_text_feature_gallery()
    
    print(f"✅ Model ready")
    print(f"   Normal reps: {model.normal_text_features_all.shape}")
    print(f"   Abnormal dirs: {model.abnormal_text_features_all.shape}")
    
    # 测试 CLS
    test_images = torch.randn(2, 3, 224, 224)
    visual_features = model.encode_image(test_images)
    
    print(f"\n[CLS Task]")
    
    # Average
    scores_avg = model.calculate_textual_anomaly_score(
        visual_features, task='cls', aggregation='average'
    )
    print(f"  Average:    {scores_avg}")
    
    # Query-conditioned
    scores_qc = model.calculate_textual_anomaly_score(
        visual_features, task='cls', 
        aggregation='query_conditioned',
        lambda_scale=1.0, margin=0.0
    )
    print(f"  Query-cond: {scores_qc}")
    
    print(f"\n✅ CLS task works!")
    
    # 测试 SEG（只测试小批次）
    print(f"\n[SEG Task - testing with 1 image only]")
    test_images_seg = torch.randn(1, 3, 224, 224)
    visual_features_seg = model.encode_image(test_images_seg)
    
    print(f"  Token features: {visual_features_seg[1].shape}")
    
    try:
        anomaly_map = model.calculate_textual_anomaly_score(
            visual_features_seg, task='seg',
            aggregation='query_conditioned',
            lambda_scale=1.0, margin=0.0
        )
        print(f"  Anomaly map: {anomaly_map.shape}")
        print(f"  Range: [{anomaly_map.min():.4f}, {anomaly_map.max():.4f}]")
        print(f"\n✅ SEG task works!")
    except Exception as e:
        print(f"\n❌ SEG failed: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"🎉 Quick test completed!")
    

if __name__ == "__main__":
    quick_test()
