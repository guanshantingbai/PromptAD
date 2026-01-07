#!/usr/bin/env python3
"""
MVP Test: Semantic Fusion with MAP Only

Test the minimal viable product for semantic fusion:
- E_final = E_geom + alpha * E_sem
- E_sem = logsumexp(q @ MAP.T)
- No LAP, no normal anchor, no training changes

Usage:
    # Baseline (alpha=0.0, no semantic fusion)
    python test_mvp_semantic_fusion.py --class_name carpet --semantic-weight 0.0
    
    # With semantic fusion (alpha > 0)
    python test_mvp_semantic_fusion.py --class_name carpet --semantic-weight 0.1
    python test_mvp_semantic_fusion.py --class_name carpet --semantic-weight 0.2
    
    # Sweep alpha values
    for alpha in 0.0 0.05 0.1 0.15 0.2; do
        python test_mvp_semantic_fusion.py --class_name carpet --semantic-weight $alpha
    done
"""

import argparse
import torch
import numpy as np
from datasets import *
from PromptAD import *
from utils.metrics import *
from utils.eval_utils import *
from tqdm import tqdm
from PIL import Image


def test_mvp(args):
    """Test MVP semantic fusion"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n{'='*60}")
    print(f"MVP Semantic Fusion Test")
    print(f"{'='*60}")
    print(f"Dataset: {args.dataset}")
    print(f"Class: {args.class_name}")
    print(f"Semantic weight (alpha): {args.semantic_weight}")
    print(f"Use visual prototypes: {args.use_visual_prototypes}")
    print(f"{'='*60}\n")
    
    # Load model
    model_kwargs = {
        'out_size_h': args.resolution,
        'out_size_w': args.resolution,
        'device': device,
        'backbone': args.backbone,
        'pretrained_dataset': args.pretrained_dataset,
        'n_ctx': args.n_ctx,
        'n_pro': args.n_pro,
        'n_ctx_ab': args.n_ctx_ab,
        'n_pro_ab': args.n_pro_ab,
        'class_name': args.class_name,
        'precision': 'fp16',
        'use_visual_prototypes': args.use_visual_prototypes,
        'k_shot': args.k_shot,
        'img_resize': args.img_resize,
        'img_cropsize': args.img_cropsize
    }
    
    model = PromptAD(**model_kwargs).to(device)
    
    # Load checkpoint
    checkpoint_path = f"result/baseline/{args.dataset}/k_{args.k_shot}/checkpoint/CLS-Seed_{args.seed}-{args.class_name}-check_point.pt"
    print(f"Loading checkpoint: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path), strict=False)
    model.eval_mode()
    
    # Load visual prototypes if enabled
    if args.use_visual_prototypes:
        support_dataset = get_MVTecAD_dataset(args.class_name, args.img_resize, args.k_shot, 'train')
        support_images = [Image.fromarray(img) for img, _, _, _, _ in support_dataset]
        support_images = [model.transform(img) for img in support_images]
        support_images = torch.stack(support_images).to(device)
        
        with torch.no_grad():
            visual_features = model.encode_image(support_images)
            support_cls_tokens = visual_features[0]  # [k_shot, D]
        
        model.set_visual_prototypes(support_cls_tokens)
        print(f"✅ Visual prototypes loaded: {support_cls_tokens.shape}")
    
    # Build text feature gallery (MAP)
    model.build_text_feature_gallery()
    
    # Load test data
    test_dataset = get_MVTecAD_dataset(args.class_name, args.img_resize, args.k_shot, 'test')
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=0
    )
    
    # Run inference
    print(f"\nRunning inference...")
    scores_semantic = []
    scores_memory = []
    gt_labels = []
    
    for data, _, label, _, _ in tqdm(test_dataloader, desc="Testing"):
        # Preprocess images
        data = [model.transform(Image.fromarray(f.numpy())) for f in data]
        data = torch.stack(data, dim=0).to(device)
        
        # Get visual features
        with torch.no_grad():
            visual_features = model.encode_image(data)
            
            # Calculate semantic scores (with MVP fusion)
            semantic_scores = model.calculate_textual_anomaly_score(
                visual_features, 
                'cls', 
                semantic_weight=args.semantic_weight
            )
            
            # Calculate memory scores
            memory_scores = model.calculate_memory_image_score(visual_features)
        
        scores_semantic.extend(semantic_scores.tolist() if isinstance(semantic_scores, np.ndarray) else semantic_scores)
        scores_memory.extend(memory_scores.tolist())
        gt_labels.extend(label.numpy())
    
    # Calculate metrics
    scores_semantic = np.array(scores_semantic)
    scores_memory = np.array(scores_memory)
    gt_labels = np.array(gt_labels)
    
    # Fusion (simple harmonic mean for compatibility)
    eps = 1e-10
    scores_fusion = 1.0 / (1.0 / (scores_memory + eps) + 1.0 / (scores_semantic + eps))
    
    # Calculate AUROC
    from sklearn.metrics import roc_auc_score
    
    auroc_semantic = roc_auc_score(gt_labels, scores_semantic)
    auroc_memory = roc_auc_score(gt_labels, scores_memory)
    auroc_fusion = roc_auc_score(gt_labels, scores_fusion)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Results (alpha={args.semantic_weight})")
    print(f"{'='*60}")
    print(f"Semantic AUROC: {auroc_semantic:.4f}")
    print(f"Memory AUROC:   {auroc_memory:.4f}")
    print(f"Fusion AUROC:   {auroc_fusion:.4f}")
    
    if args.semantic_weight == 0.0:
        print(f"\n✅ Baseline mode (alpha=0): Semantic score should be unchanged")
    else:
        print(f"\n🔥 MVP mode (alpha={args.semantic_weight}): Semantic fusion enabled")
        print(f"   E_final = E_geom + {args.semantic_weight} * E_sem")
    
    print(f"{'='*60}\n")
    
    return {
        'semantic_auroc': auroc_semantic,
        'memory_auroc': auroc_memory,
        'fusion_auroc': auroc_fusion,
        'alpha': args.semantic_weight
    }


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def get_args():
    parser = argparse.ArgumentParser(description='MVP Semantic Fusion Test')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='mvtec', choices=['mvtec', 'visa'])
    parser.add_argument('--class_name', type=str, default='carpet')
    parser.add_argument('--k-shot', type=int, default=2)
    
    # Image parameters
    parser.add_argument('--img-resize', type=int, default=240)
    parser.add_argument('--img-cropsize', type=int, default=240)
    parser.add_argument('--resolution', type=int, default=400)
    parser.add_argument('--batch-size', type=int, default=32)
    
    # Model parameters
    parser.add_argument("--backbone", type=str, default="ViT-B-16",
                        choices=['ViT-B-16-plus-240', 'ViT-B-16'])
    parser.add_argument("--pretrained_dataset", type=str, default="laion400m_e32")
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--gpu-id", type=int, default=0)
    
    # Prompt parameters
    parser.add_argument("--n_ctx", type=int, default=4)
    parser.add_argument("--n_ctx_ab", type=int, default=12)
    parser.add_argument("--n_pro", type=int, default=1)
    parser.add_argument("--n_pro_ab", type=int, default=2)
    
    # 🔥 MVP parameters
    parser.add_argument("--semantic-weight", type=float, default=0.0,
                        help="MVP semantic fusion weight (alpha). Default 0.0 = baseline")
    parser.add_argument("--use-visual-prototypes", type=str2bool, default=True,
                        help="Use visual prototypes mode (required for MVP)")
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    import os
    
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = f"{args.gpu_id}"
    
    results = test_mvp(args)
    
    print(f"\n✅ Test complete!")
    print(f"Final AUROC (fusion): {results['fusion_auroc']:.4f}")
