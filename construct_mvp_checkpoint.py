#!/usr/bin/env python3
"""
构造 n(q) MVP checkpoint

功能：
1. 从训练集图像提取 cls tokens (training_cls_tokens)
2. 构造完整的 MAP/LAP text features
3. 保存到指定目录

输出 checkpoint 包含：
- training_cls_tokens: [K, D] - 支持集图像的 CLS embeddings
- abnormal_text_features_all: [N_map + N_lap, D] - MAP + LAP features
- normal_text_features_all: [N_pro, D] - Normal text features (可选)
- text_features: [2, D] - [normal_anchor, abnormal_anchor]
- feature_gallery1: [K, 196, D] - Memory bank (patch features)
- feature_gallery2: [K, 196, D] - Memory bank (patch features)
"""

import argparse
import torch
import os
from pathlib import Path

from PromptAD import PromptAD
from datasets import *
from utils.training_utils import setup_seed


def construct_mvp_checkpoint(
    dataset='mvtec',
    class_name='carpet',
    k_shot=2,
    seed=111,
    output_dir='./result/baseline',
    backbone='ViT-B-16-plus-240',
    pretrained_dataset='laion400m_e32',
    img_resize=518,
    img_cropsize=518,
    n_ctx=4,
    n_ctx_ab=1,
    n_pro=1,
    n_pro_ab=4,
    device='cuda:0'
):
    """
    构造 MVP checkpoint
    
    Args:
        dataset: 数据集名称
        class_name: 类别名称
        k_shot: 支持集图像数量
        seed: 随机种子
        output_dir: 输出目录
        backbone: CLIP backbone
        pretrained_dataset: 预训练数据集
        img_resize: 图像resize大小
        img_cropsize: 图像crop大小
        device: 设备
    """
    
    print("="*70)
    print("Constructing n(q) MVP Checkpoint")
    print("="*70)
    print(f"Dataset: {dataset}")
    print(f"Class: {class_name}")
    print(f"K-shot: {k_shot}")
    print(f"Seed: {seed}")
    print(f"Output: {output_dir}")
    print("")
    
    # Setup seed
    setup_seed(seed)
    
    # Create output directory
    checkpoint_dir = f"{output_dir}/{dataset}/k_{k_shot}/checkpoint"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Step 1: Create model
    print("[Step 1] Creating model...")
    model = PromptAD(
        k_shot=k_shot,
        class_name=class_name,
        out_size_h=img_resize,
        out_size_w=img_resize,
        img_resize=img_resize,
        img_cropsize=img_cropsize,
        backbone=backbone,
        pretrained_dataset=pretrained_dataset,
        n_ctx=n_ctx,
        n_pro=n_pro,
        n_ctx_ab=n_ctx_ab,
        n_pro_ab=n_pro_ab,
        use_visual_prototypes=True,
        device=device
    )
    print(f"  Model created on {device}")
    
    # Step 2: Load training data
    print("\n[Step 2] Loading training data...")
    from datasets import get_dataloader_from_args
    
    # Prepare kwargs for dataloader
    dataloader_kwargs = {
        'dataset': dataset,
        'class_name': class_name,
        'k_shot': k_shot,
        'seed': seed,
        'img_resize': img_resize,
        'img_cropsize': img_cropsize,
        'batch_size': k_shot,
        'num_workers': 0
    }
    
    train_loader, _ = get_dataloader_from_args(
        phase='train',
        perturbed=False,
        transform=model.transform,
        **dataloader_kwargs
    )
    
    # Get training images
    train_images_list = []
    for (data, mask, label, name, img_type) in train_loader:
        train_images_list.append(data)
        break  # Only need one batch (all training images)
    
    train_images = torch.cat(train_images_list, dim=0).to(device)
    print(f"  Loaded {train_images.shape[0]} training images")
    print(f"  Image shape: {train_images.shape}")
    
    # Step 3: Extract training_cls_tokens
    print("\n[Step 3] Extracting training cls tokens...")
    model.set_visual_prototypes(train_images)
    print(f"  training_cls_tokens shape: {model.training_cls_tokens.shape}")
    print(f"  training_cls_tokens dtype: {model.training_cls_tokens.dtype}")
    
    # Step 4: Build text feature gallery (MAP + LAP)
    print("\n[Step 4] Building text feature gallery...")
    # Ensure model is on correct device
    model = model.to(device)
    model.build_text_feature_gallery()
    n_map = getattr(model.prompt_learner, 'n_map', 8)
    n_lap = getattr(model.prompt_learner, 'n_lap', 2)
    print(f"  MAP features: {n_map}")
    print(f"  LAP features: {n_lap}")
    print(f"  Total abnormal features: {model.abnormal_text_features_all.shape}")
    print(f"  Text features (normal/abnormal anchors): {model.text_features.shape}")
    
    # Step 5: Build image feature gallery (memory bank)
    print("\n[Step 5] Building image feature gallery (memory bank)...")
    features1 = []
    features2 = []
    
    for (data, mask, label, name, img_type) in train_loader:
        data = data.to(device)
        _, _, feature_map1, feature_map2 = model.encode_image(data)
        features1.append(feature_map1)
        features2.append(feature_map2)
        break
    
    features1 = torch.cat(features1, dim=0)
    features2 = torch.cat(features2, dim=0)
    model.build_image_feature_gallery(features1, features2)
    
    print(f"  feature_gallery1 shape: {model.feature_gallery1.shape}")
    print(f"  feature_gallery2 shape: {model.feature_gallery2.shape}")
    
    # Step 6: Save checkpoint
    print("\n[Step 6] Saving checkpoint...")
    checkpoint_path = f"{checkpoint_dir}/CLS-Seed_{seed}-{class_name}-check_point.pt"
    
    selected_keys = [
        'feature_gallery1',
        'feature_gallery2',
        'text_features',
        'normal_text_features_all',
        'abnormal_text_features_all',
        'training_cls_tokens',  # 🆕 Support set cls tokens
    ]
    
    state_dict = model.state_dict()
    selected_state_dict = {k: v for k, v in state_dict.items() if k in selected_keys}
    
    torch.save(selected_state_dict, checkpoint_path)
    print(f"  ✅ Checkpoint saved to: {checkpoint_path}")
    
    # Step 7: Verify checkpoint
    print("\n[Step 7] Verifying checkpoint...")
    loaded_state_dict = torch.load(checkpoint_path, map_location='cpu')
    print(f"  Keys in checkpoint: {list(loaded_state_dict.keys())}")
    print(f"  training_cls_tokens shape: {loaded_state_dict['training_cls_tokens'].shape}")
    print(f"  abnormal_text_features_all shape: {loaded_state_dict['abnormal_text_features_all'].shape}")
    print(f"  text_features shape: {loaded_state_dict['text_features'].shape}")
    
    print("\n" + "="*70)
    print("✅ Checkpoint construction completed!")
    print("="*70)
    print(f"\nCheckpoint location: {checkpoint_path}")
    print(f"\nTo use this checkpoint:")
    print(f"  python test_cls.py \\")
    print(f"      --dataset {dataset} \\")
    print(f"      --class_name {class_name} \\")
    print(f"      --k-shot {k_shot} \\")
    print(f"      --checkpoint-dir {output_dir}/{dataset}/k_{k_shot} \\")
    print(f"      --semantic-weight 0.1 \\")
    print(f"      --use-visual-prototypes True")
    
    return checkpoint_path


def main():
    parser = argparse.ArgumentParser(description='Construct n(q) MVP checkpoint')
    parser.add_argument('--dataset', type=str, default='mvtec', choices=['mvtec', 'visa'])
    parser.add_argument('--class_name', type=str, default='carpet')
    parser.add_argument('--k-shot', type=int, default=2)
    parser.add_argument('--seed', type=int, default=111)
    parser.add_argument('--output-dir', type=str, default='./result/nq_mvp',
                        help='Output directory for checkpoint (default: ./result/nq_mvp)')
    parser.add_argument('--backbone', type=str, default='ViT-B-16-plus-240',
                        choices=['ViT-B-16-plus-240', 'ViT-B-16'])
    parser.add_argument('--pretrained-dataset', type=str, default='laion400m_e32')
    parser.add_argument('--img-resize', type=int, default=518)
    parser.add_argument('--img-cropsize', type=int, default=518)
    parser.add_argument('--n-ctx', type=int, default=4)
    parser.add_argument('--n-ctx-ab', type=int, default=1)
    parser.add_argument('--n-pro', type=int, default=1)
    parser.add_argument('--n-pro-ab', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--gpu-id', type=int, default=0)
    
    args = parser.parse_args()
    
    # Set GPU
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    if args.device.startswith('cuda'):
        args.device = f'cuda:0'  # Always use cuda:0 after CUDA_VISIBLE_DEVICES
    
    # Construct checkpoint
    checkpoint_path = construct_mvp_checkpoint(
        dataset=args.dataset,
        class_name=args.class_name,
        k_shot=args.k_shot,
        seed=args.seed,
        output_dir=args.output_dir,
        backbone=args.backbone,
        pretrained_dataset=args.pretrained_dataset,
        img_resize=args.img_resize,
        img_cropsize=args.img_cropsize,
        n_ctx=args.n_ctx,
        n_ctx_ab=args.n_ctx_ab,
        n_pro=args.n_pro,
        n_pro_ab=args.n_pro_ab,
        device=args.device
    )


if __name__ == '__main__':
    main()
