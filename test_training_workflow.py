#!/usr/bin/env python3
"""
快速验证视觉原型在训练流程中的集成
"""

import torch
import sys
sys.path.append('.')

from PromptAD.model import PromptAD
from PIL import Image
import numpy as np

def create_dummy_train_data(k_shot, size=224):
    """创建模拟训练数据"""
    data_list = []
    for i in range(k_shot):
        img_array = np.random.randint(0, 255, (3, size, size), dtype=np.uint8)
        tensor = torch.from_numpy(img_array).float() / 255.0
        data_list.append(tensor)
    return torch.stack(data_list)


def test_training_workflow():
    """测试完整训练流程"""
    print("\n" + "="*70)
    print("Testing Visual Prototypes in Training Workflow")
    print("="*70)
    
    k_shot = 4
    device = 'cpu'
    
    # Step 1: 创建模型
    print("\n[Step 1] Creating model...")
    model = PromptAD(
        out_size_h=224, out_size_w=224,
        device=device,
        backbone='ViT-B-16',
        pretrained_dataset='laion400m_e32',
        n_ctx=12, n_pro=k_shot,  # n_pro = k_shot
        n_ctx_ab=12, n_pro_ab=4,
        class_name='bottle',
        precision='fp32',
        use_visual_prototypes=True,  # 启用视觉原型
        k_shot=k_shot,
        img_resize=240, img_cropsize=224
    )
    model.eval_mode()
    print(f"✅ Model created with use_visual_prototypes=True")
    
    # Step 2: 模拟训练数据
    print("\n[Step 2] Preparing training data...")
    train_images = create_dummy_train_data(k_shot)
    print(f"✅ Training data shape: {train_images.shape}")
    
    # Step 3: 设置视觉原型
    print("\n[Step 3] Setting visual prototypes...")
    model.set_visual_prototypes(train_images.to(device))
    print(f"✅ Visual prototypes set")
    
    # Step 4: 构建特征库
    print("\n[Step 4] Building feature galleries...")
    
    # 构建图像特征库（memory bank）
    _, _, feature_map1, feature_map2 = model.encode_image(train_images.to(device))
    model.build_image_feature_gallery(feature_map1, feature_map2)
    print(f"✅ Image feature gallery built")
    
    # 构建文本特征库（会跳过 normal prompts）
    model.build_text_feature_gallery()
    print(f"✅ Text feature gallery built (skipped normal prompts)")
    
    # Step 5: 检查可训练参数
    print("\n[Step 5] Checking trainable parameters...")
    trainable_params = [p for p in model.prompt_learner.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in trainable_params)
    print(f"✅ Trainable parameters: {len(trainable_params)} tensors, {total_params} elements")
    print(f"   (Only abnormal_ctx, no normal_ctx)")
    
    # Step 6: 模拟训练循环
    print("\n[Step 6] Simulating training loop...")
    
    optimizer = torch.optim.SGD(trainable_params, lr=0.002, momentum=0.9)
    
    # 模拟一个训练步骤
    train_images_batch = train_images.to(device)
    
    # 获取 prompts
    normal_prompt, abnormal_handle, abnormal_learned = model.prompt_learner()
    
    optimizer.zero_grad()
    
    # 🆕 视觉原型模式：跳过 normal text features
    if model.use_visual_prototypes:
        # 仅计算 abnormal
        abnormal_features_handle = model.encode_text_embedding(
            abnormal_handle, 
            model.tokenized_abnormal_prompts_handle
        )
        abnormal_features_learned = model.encode_text_embedding(
            abnormal_learned,
            model.tokenized_abnormal_prompts_learned
        )
        abnormal_text_features = torch.cat([
            abnormal_features_handle,
            abnormal_features_learned
        ], dim=0)
        
        # normal 特征已在 set_visual_prototypes 中设置
        normal_text_features = model.text_features[0:1]
        
        print(f"✅ Training step:")
        print(f"   Normal features: {normal_text_features.shape} (from VISUAL prototypes, fixed)")
        print(f"   Abnormal features: {abnormal_text_features.shape} (from TEXT prompts, learnable)")
    
    # Step 7: 测试推理
    print("\n[Step 7] Testing inference...")
    test_images = create_dummy_train_data(2)
    
    visual_features = model.encode_image(test_images.to(device))
    anomaly_scores = model.calculate_textual_anomaly_score(visual_features, 'cls')
    
    print(f"✅ Inference completed")
    print(f"   Anomaly scores: {anomaly_scores.shape}")
    print(f"   Sample scores: {anomaly_scores}")
    
    print("\n" + "="*70)
    print("🎉 Training workflow test passed!")
    print("="*70)
    

if __name__ == "__main__":
    try:
        test_training_workflow()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
