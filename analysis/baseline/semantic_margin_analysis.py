#!/usr/bin/env python3
"""
Semantic Branch Margin Analysis

Analyze margin distribution (s_normal - s_abnormal) for the Semantic branch.
This is the PRIMARY failure mode analysis at the inference level.
"""

import os
import sys
from pathlib import Path

# Add project root to path
script_path = Path(__file__).absolute()
project_root = script_path.parent.parent.parent
os.chdir(project_root)
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

from PromptAD.model import PromptAD
from datasets.dataset import CLIPDataset
from datasets.mvtec import load_mvtec
from datasets.visa import load_visa


def load_model_and_data(dataset, class_name, k_shot, seed, checkpoint_dir, device='cuda:0'):
    """Load model and test data"""
    
    # Model kwargs
    kwargs = {
        'backbone': 'ViT-B-16-plus-240',
        'pretrained_dataset': 'laion400m_e32',
        'img_resize': 366,
        'img_cropsize': 336,
        'resolution': 336,
        'out_size_h': 336,
        'out_size_w': 336,
        'n_ctx': 12,
        'n_pro': 3,
        'n_ctx_ab': 3,
        'n_pro_ab': 4,
        'class_name': class_name,
        'k_shot': k_shot,
        'seed': seed,
        'use_cpu': 0,
        'device': device,
        'use_lap': True
    }
    
    # Load model
    model = PromptAD(**kwargs)
    model = model.to(device)
    
    # Load checkpoint
    # Try different naming patterns
    checkpoint_patterns = [
        f'CLS-Seed_{seed}-{class_name}-check_point.pt',
        f'{class_name}.pth',
        f'{class_name}.pt'
    ]
    
    checkpoint_path = None
    for pattern in checkpoint_patterns:
        path = Path(checkpoint_dir) / pattern
        if path.exists():
            checkpoint_path = path
            break
    
    if checkpoint_path is None:
        raise FileNotFoundError(f"Checkpoint not found for {class_name} in {checkpoint_dir}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    
    # Load test data
    if dataset == 'mvtec':
        train_data, test_data = load_mvtec(class_name, k_shot, seed)
    elif dataset == 'visa':
        train_data, test_data = load_visa(class_name, k_shot, seed)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    test_dataset = CLIPDataset(test_data, kwargs)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    return model, test_loader


def compute_semantic_margins(model, test_loader, device='cuda:0'):
    """
    Compute Semantic branch margins: s_normal - s_abnormal
    
    Returns:
        margins: np.array of margins
        labels: np.array of ground truth labels (0=normal, 1=abnormal)
    """
    margins_list = []
    labels_list = []
    
    with torch.no_grad():
        for data in tqdm(test_loader, desc="Computing margins"):
            imgs, label = data
            imgs = imgs.to(device)
            
            # Get Semantic scores
            semantic_scores, _, _ = model(imgs, 'cls')
            
            # semantic_scores is a list with one element (batch_size=1)
            score = semantic_scores[0]
            
            # Margin = score (already s_normal - s_abnormal in the model)
            margins_list.append(score)
            labels_list.append(label.item())
    
    return np.array(margins_list), np.array(labels_list)


def analyze_margin_statistics(margins, labels):
    """Compute comprehensive margin statistics"""
    
    margins_normal = margins[labels == 0]
    margins_abnormal = margins[labels == 1]
    
    def compute_stats(data, group_name, class_name):
        return {
            'mean': float(np.mean(data)),
            'median': float(np.median(data)),
            'std': float(np.std(data)),
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'p10': float(np.percentile(data, 10)),
            'p25': float(np.percentile(data, 25)),
            'p50': float(np.percentile(data, 50)),
            'p75': float(np.percentile(data, 75)),
            'p90': float(np.percentile(data, 90)),
            'group': group_name,
            'class': class_name
        }
    
    return [
        compute_stats(margins_normal, 'normal', ''),
        compute_stats(margins_abnormal, 'abnormal', '')
    ]


def main():
    parser = argparse.ArgumentParser(description='Semantic margin analysis')
    parser.add_argument('--dataset', type=str, required=True, choices=['mvtec', 'visa'])
    parser.add_argument('--class-name', type=str, required=True)
    parser.add_argument('--k-shot', type=int, default=2)
    parser.add_argument('--seed', type=int, default=111)
    parser.add_argument('--checkpoint-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--gpu-id', type=int, default=0)
    
    args = parser.parse_args()
    device = f'cuda:{args.gpu_id}'
    
    print(f"\n{'='*80}")
    print(f"Analyzing: {args.dataset} - {args.class_name}")
    print(f"{'='*80}\n")
    
    # Load model and data
    model, test_loader = load_model_and_data(
        args.dataset, args.class_name, args.k_shot, args.seed,
        args.checkpoint_dir, device
    )
    
    # Compute margins
    margins, labels = compute_semantic_margins(model, test_loader, device)
    
    # Analyze statistics
    stats = analyze_margin_statistics(margins, labels)
    
    # Add class name to stats
    for s in stats:
        s['class'] = args.class_name
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save margin stats
    stats_df = pd.DataFrame(stats)
    stats_path = output_dir / f'{args.class_name}_margin_stats.csv'
    stats_df.to_csv(stats_path, index=False)
    
    # Save raw data for visualization
    raw_path = output_dir / f'{args.class_name}_raw_data.npz'
    np.savez(raw_path, margins=margins, labels=labels)
    
    print(f"\n✅ Results saved:")
    print(f"  - {stats_path}")
    print(f"  - {raw_path}")
    
    # Print summary
    print(f"\n{'='*80}")
    print("MARGIN STATISTICS SUMMARY")
    print(f"{'='*80}\n")
    print(stats_df.to_string(index=False))
    print()
    
    # Compute separation
    sep = abs(stats[0]['mean'] - stats[1]['mean'])
    print(f"Margin Separation: {sep:.4f}")
    print(f"Normal  Mean: {stats[0]['mean']:.4f} ± {stats[0]['std']:.4f}")
    print(f"Abnormal Mean: {stats[1]['mean']:.4f} ± {stats[1]['std']:.4f}")


if __name__ == '__main__':
    main()
