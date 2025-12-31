#!/usr/bin/env python3
"""
MAP/LAP Reliability Metrics Computation (Inference-only)

Purpose: Compute distinguishable reliability indicators for MAP and LAP anchors
         WITHOUT designing gating formulas. Focus on measurable signals that
         explain failure modes (A, A+B, etc.)

Key Constraints:
1. Inference-only: Use baseline checkpoint weights, no MAP-only retraining
2. Label-free: All metrics computed on normal support set only
3. Distinguishable: Quantify "when MAP is unreliable" vs "when LAP is unreliable"
4. Interpretable: Directly relate to observed failure modes

Output: Class-level CSV with reliability indicators for gating mechanism design
"""

import os
import sys
from pathlib import Path

# MUST change directory and add to path BEFORE any project imports
# __file__ = .../PromptAD/analysis/baseline/MAP_LAP/compute_reliability_metrics.py
# parent = .../PromptAD/analysis/baseline/MAP_LAP
# parent.parent = .../PromptAD/analysis/baseline
# parent.parent.parent = .../PromptAD/analysis
# parent.parent.parent.parent = .../PromptAD ← This is what we want!
script_path = Path(__file__).absolute()
project_root = script_path.parent.parent.parent.parent
os.chdir(project_root)

# Add to BEGINNING of sys.path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print(f"Working directory: {os.getcwd()}")
print(f"Project root: {project_root}")

# NOW import everything else
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

# Project imports
from PromptAD.model import PromptAD
from datasets.dataset import CLIPDataset
from datasets.mvtec import load_mvtec
from datasets.visa import load_visa


def load_model_and_data(dataset, class_name, k_shot, seed, device='cuda:0'):
    """Load baseline model and normal support set"""
    
    # Model configuration
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
        'use_lap': True  # Baseline uses both MAP+LAP
    }
    
    # Load model
    model = PromptAD(**kwargs)
    model = model.to(device)
    
    # Load checkpoint
    checkpoint_path = f"./result/baseline/{dataset}/k_{k_shot}/checkpoint/CLS-Seed_{seed}-{class_name}-check_point.pt"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    model.load_state_dict(torch.load(checkpoint_path), strict=False)
    model.eval()
    
    # Load normal support set (k-shot samples)
    if dataset == 'mvtec':
        train_data, test_data = load_mvtec(class_name, k_shot)
        train_img, train_gt, train_label, train_type = train_data
    elif dataset == 'visa':
        train_data, test_data = load_visa(class_name, k_shot)
        train_img, train_gt, train_label, train_type = train_data
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    # Create simple dataset for loading images
    from PIL import Image
    from torch.utils.data import Dataset
    
    class SimpleImageDataset(Dataset):
        def __init__(self, img_paths, labels, transform=None):
            self.img_paths = img_paths
            self.labels = labels
            self.transform = transform
        
        def __len__(self):
            return len(self.img_paths)
        
        def __getitem__(self, idx):
            img = Image.open(self.img_paths[idx]).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, self.labels[idx]
    
    normal_dataset = SimpleImageDataset(
        train_img, train_label,
        transform=model.transform
    )
    
    normal_loader = DataLoader(
        normal_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )
    
    return model, normal_loader, kwargs


def extract_map_lap_anchors(model, device='cuda:0'):
    """
    Extract MAP and LAP anchors from baseline model
    
    Returns:
        mu_n: Normal anchor [1, D]
        mu_a_map: MAP-only abnormal anchor [1, D]
        mu_a_lap: LAP-only abnormal anchor [1, D]
        mu_a_combined: Combined (MAP+LAP) abnormal anchor [1, D]
    """
    with torch.no_grad():
        # Get text embeddings from prompt learner
        normal_text_embeddings, abnormal_text_embeddings_map, abnormal_text_embeddings_lap = model.prompt_learner()
        
        # Encode using the model's method with tokenized prompts
        normal_text_features = model.encode_text_embedding(
            normal_text_embeddings, 
            model.tokenized_normal_prompts
        )
        abnormal_text_features_map = model.encode_text_embedding(
            abnormal_text_embeddings_map, 
            model.tokenized_abnormal_prompts_handle
        )
        abnormal_text_features_lap = model.encode_text_embedding(
            abnormal_text_embeddings_lap, 
            model.tokenized_abnormal_prompts_learned
        )
        
        # Normalize
        normal_text_features = normal_text_features / normal_text_features.norm(dim=-1, keepdim=True)
        abnormal_text_features_map = abnormal_text_features_map / abnormal_text_features_map.norm(dim=-1, keepdim=True)
        abnormal_text_features_lap = abnormal_text_features_lap / abnormal_text_features_lap.norm(dim=-1, keepdim=True)
        
        # Average to get anchors
        mu_n = torch.mean(normal_text_features, dim=0, keepdim=True)  # [1, D]
        mu_a_map = torch.mean(abnormal_text_features_map, dim=0, keepdim=True)  # [1, D]
        mu_a_lap = torch.mean(abnormal_text_features_lap, dim=0, keepdim=True)  # [1, D]
        
        # Combined anchor (baseline default)
        all_abnormal = torch.cat([abnormal_text_features_map, abnormal_text_features_lap], dim=0)
        mu_a_combined = torch.mean(all_abnormal, dim=0, keepdim=True)  # [1, D]
        
        # Re-normalize anchors
        mu_n = mu_n / mu_n.norm(dim=-1, keepdim=True)
        mu_a_map = mu_a_map / mu_a_map.norm(dim=-1, keepdim=True)
        mu_a_lap = mu_a_lap / mu_a_lap.norm(dim=-1, keepdim=True)
        mu_a_combined = mu_a_combined / mu_a_combined.norm(dim=-1, keepdim=True)
    
    return mu_n, mu_a_map, mu_a_lap, mu_a_combined


def compute_normal_side_risk_metrics(model, normal_loader, mu_n, mu_a_map, mu_a_lap, device='cuda:0', epsilon=0.05):
    """
    Section I: Normal-side Risk Indicators
    
    For each normal sample x, compute:
        m_MAP(x) = s_normal(x) - s_abnormal_MAP(x)
        m_LAP(x) = s_normal(x) - s_abnormal_LAP(x)
    
    Output statistics:
        - R_MAP_0: P(m_MAP < 0) - proportion misclassified as abnormal
        - R_LAP_0: P(m_LAP < 0)
        - R_MAP_eps: P(m_MAP < epsilon) - proportion in danger zone
        - R_LAP_eps: P(m_LAP < epsilon)
        - Quantiles: q10, median, q90 of margins
    """
    t = model.model.logit_scale  # Temperature parameter
    
    margins_map = []
    margins_lap = []
    
    with torch.no_grad():
        for data, label in tqdm(normal_loader, desc="Computing normal-side margins"):
            data = data.to(device)
            
            # Extract image features (returns [global_features, patch_features])
            global_features = model.encode_image(data)[0]  # [B, D]
            
            # Compute similarities
            s_normal = t * (global_features @ mu_n.T).squeeze(-1)  # [B]
            s_abnormal_map = t * (global_features @ mu_a_map.T).squeeze(-1)  # [B]
            s_abnormal_lap = t * (global_features @ mu_a_lap.T).squeeze(-1)  # [B]
            
            # Compute margins
            m_map = s_normal - s_abnormal_map  # [B]
            m_lap = s_normal - s_abnormal_lap  # [B]
            
            margins_map.extend(m_map.cpu().numpy())
            margins_lap.extend(m_lap.cpu().numpy())
    
    margins_map = np.array(margins_map)
    margins_lap = np.array(margins_lap)
    
    # Compute risk metrics
    metrics = {
        # Risk of misclassification (margin < 0)
        'R_MAP_0': np.mean(margins_map < 0),
        'R_LAP_0': np.mean(margins_lap < 0),
        
        # Risk of being in danger zone (margin < epsilon)
        'R_MAP_eps': np.mean(margins_map < epsilon),
        'R_LAP_eps': np.mean(margins_lap < epsilon),
        
        # Margin distribution statistics
        'margin_MAP_q10': np.percentile(margins_map, 10),
        'margin_MAP_median': np.median(margins_map),
        'margin_MAP_q90': np.percentile(margins_map, 90),
        'margin_MAP_mean': np.mean(margins_map),
        'margin_MAP_std': np.std(margins_map),
        
        'margin_LAP_q10': np.percentile(margins_lap, 10),
        'margin_LAP_median': np.median(margins_lap),
        'margin_LAP_q90': np.percentile(margins_lap, 90),
        'margin_LAP_mean': np.mean(margins_lap),
        'margin_LAP_std': np.std(margins_lap),
    }
    
    return metrics, margins_map, margins_lap


def compute_consistency_metrics(margins_map, margins_lap, mu_a_map, mu_a_lap, device='cuda:0'):
    """
    Section II: Stability / Consistency Indicators
    
    On normal support set, compute discrepancy between MAP and LAP:
        d(x) = |s_abnormal_MAP(x) - s_abnormal_LAP(x)|
    
    Key insight: High discrepancy means MAP and LAP disagree strongly on normal samples,
                 indicating neither can be trusted alone (explains A+B failure mode)
    """
    # Discrepancy in margin values
    margin_discrepancy = np.abs(margins_map - margins_lap)
    
    # Anchor disagreement ratio: fraction of samples where MAP and LAP have opposite signs
    opposite_signs = ((margins_map > 0) & (margins_lap < 0)) | ((margins_map < 0) & (margins_lap > 0))
    
    metrics = {
        'margin_discrepancy_mean': np.mean(margin_discrepancy),
        'margin_discrepancy_median': np.median(margin_discrepancy),
        'margin_discrepancy_std': np.std(margin_discrepancy),
        'margin_discrepancy_max': np.max(margin_discrepancy),
        
        # Disagreement rate: how often do MAP and LAP give opposite judgments?
        'disagreement_rate': np.mean(opposite_signs),
        
        # Correlation between MAP and LAP margins
        'margin_correlation': np.corrcoef(margins_map, margins_lap)[0, 1],
    }
    
    return metrics


def compute_geometric_metrics(mu_n, mu_a_map, mu_a_lap, mu_a_combined):
    """
    Section III: Anchor Geometry Auxiliary Indicators
    
    Compute anchor-level metrics (not sample-level):
        - cos(μ_n, μ_a_MAP)
        - cos(μ_n, μ_a_LAP)
        - cos(μ_a_MAP, μ_a_LAP)
        - ||μ_a_MAP - μ_a_LAP||
    
    Purpose: Explain whether anchors are highly overlapped or redundant
    """
    with torch.no_grad():
        # Cosine similarities
        cos_n_map = (mu_n @ mu_a_map.T).item()
        cos_n_lap = (mu_n @ mu_a_lap.T).item()
        cos_map_lap = (mu_a_map @ mu_a_lap.T).item()
        cos_n_combined = (mu_n @ mu_a_combined.T).item()
        
        # L2 distances
        dist_n_map = torch.norm(mu_n - mu_a_map, p=2).item()
        dist_n_lap = torch.norm(mu_n - mu_a_lap, p=2).item()
        dist_map_lap = torch.norm(mu_a_map - mu_a_lap, p=2).item()
        dist_n_combined = torch.norm(mu_n - mu_a_combined, p=2).item()
    
    metrics = {
        'cos_normal_MAP': cos_n_map,
        'cos_normal_LAP': cos_n_lap,
        'cos_MAP_LAP': cos_map_lap,
        'cos_normal_combined': cos_n_combined,
        
        'dist_normal_MAP': dist_n_map,
        'dist_normal_LAP': dist_n_lap,
        'dist_MAP_LAP': dist_map_lap,
        'dist_normal_combined': dist_n_combined,
    }
    
    return metrics


def analyze_single_class(dataset, class_name, k_shot, seed, device='cuda:0', epsilon=0.05):
    """Compute all reliability metrics for a single class"""
    
    print(f"\n{'='*80}")
    print(f"Analyzing {dataset}-{class_name} (k={k_shot}, seed={seed})")
    print(f"{'='*80}")
    
    # Load model and data
    print("\n[1/4] Loading model and normal support set...")
    model, normal_loader, kwargs = load_model_and_data(dataset, class_name, k_shot, seed, device)
    
    # Extract anchors
    print("[2/4] Extracting MAP and LAP anchors...")
    mu_n, mu_a_map, mu_a_lap, mu_a_combined = extract_map_lap_anchors(model, device)
    
    print(f"  - Normal anchor: {mu_n.shape}")
    print(f"  - MAP anchor: {mu_a_map.shape} (from {model.prompt_learner.n_ab_handle} prompts)")
    print(f"  - LAP anchor: {mu_a_lap.shape} (from {model.prompt_learner.n_pro_ab * model.prompt_learner.n_pro} prompts)")
    
    # Section I: Normal-side risk metrics
    print(f"[3/4] Computing normal-side risk indicators (epsilon={epsilon})...")
    risk_metrics, margins_map, margins_lap = compute_normal_side_risk_metrics(
        model, normal_loader, mu_n, mu_a_map, mu_a_lap, device, epsilon
    )
    
    # Section II: Consistency metrics
    print("[4/4] Computing consistency indicators...")
    consistency_metrics = compute_consistency_metrics(margins_map, margins_lap, mu_a_map, mu_a_lap, device)
    
    # Section III: Geometric metrics
    geometric_metrics = compute_geometric_metrics(mu_n, mu_a_map, mu_a_lap, mu_a_combined)
    
    # Combine all metrics
    all_metrics = {
        'dataset': dataset,
        'class': f"{dataset}-{class_name}",
        'k_shot': k_shot,
        'seed': seed,
        **risk_metrics,
        **consistency_metrics,
        **geometric_metrics
    }
    
    return all_metrics


def analyze_all_classes(datasets=['mvtec', 'visa'], k_shot=2, seed=111, device='cuda:0', epsilon=0.05):
    """Compute reliability metrics for all classes in specified datasets"""
    
    # Define classes
    mvtec_classes = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut',
                     'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush',
                     'transistor', 'wood', 'zipper']
    
    visa_classes = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
                    'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
    
    results = []
    
    for dataset in datasets:
        classes = mvtec_classes if dataset == 'mvtec' else visa_classes
        
        for class_name in classes:
            try:
                metrics = analyze_single_class(dataset, class_name, k_shot, seed, device, epsilon)
                results.append(metrics)
                
                # Print summary
                print(f"\n>>> Summary for {dataset}-{class_name}:")
                print(f"  Normal-side Risk:")
                print(f"    MAP: R_0={metrics['R_MAP_0']:.3f}, R_ε={metrics['R_MAP_eps']:.3f}, median_m={metrics['margin_MAP_median']:.3f}")
                print(f"    LAP: R_0={metrics['R_LAP_0']:.3f}, R_ε={metrics['R_LAP_eps']:.3f}, median_m={metrics['margin_LAP_median']:.3f}")
                print(f"  Consistency:")
                print(f"    Discrepancy: mean={metrics['margin_discrepancy_mean']:.3f}, disagreement={metrics['disagreement_rate']:.3f}")
                print(f"  Geometry:")
                print(f"    cos(μ_n, μ_MAP)={metrics['cos_normal_MAP']:.3f}, cos(μ_n, μ_LAP)={metrics['cos_normal_LAP']:.3f}")
                print(f"    cos(μ_MAP, μ_LAP)={metrics['cos_MAP_LAP']:.3f}")
                
            except Exception as e:
                print(f"\n!!! Error processing {dataset}-{class_name}:")
                import traceback
                traceback.print_exc()
                continue
    
    return pd.DataFrame(results)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Compute MAP/LAP reliability metrics')
    parser.add_argument('--datasets', nargs='+', default=['mvtec', 'visa'],
                        help='Datasets to analyze')
    parser.add_argument('--k-shot', type=int, default=2,
                        help='Number of normal support samples')
    parser.add_argument('--seed', type=int, default=111,
                        help='Random seed')
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='GPU ID')
    parser.add_argument('--epsilon', type=float, default=0.05,
                        help='Margin danger zone threshold')
    parser.add_argument('--output-dir', type=str,
                        default='./result/baseline/baseline_analysis/MAP_LAP',
                        help='Output directory')
    
    args = parser.parse_args()
    
    device = f'cuda:{args.gpu_id}'
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Analyze all classes
    print("\n" + "="*80)
    print("MAP/LAP Reliability Metrics Analysis")
    print("="*80)
    print(f"Datasets: {args.datasets}")
    print(f"k-shot: {args.k_shot}")
    print(f"Epsilon: {args.epsilon}")
    print(f"Device: {device}")
    print(f"Output: {args.output_dir}")
    
    df = analyze_all_classes(
        datasets=args.datasets,
        k_shot=args.k_shot,
        seed=args.seed,
        device=device,
        epsilon=args.epsilon
    )
    
    # Save results
    output_path = f"{args.output_dir}/reliability_metrics_k{args.k_shot}.csv"
    df.to_csv(output_path, index=False)
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_path}")
    print(f"Total classes analyzed: {len(df)}")
    print(f"{'='*80}")
    
    # Generate summary statistics
    print("\n" + "="*80)
    print("Summary Statistics")
    print("="*80)
    
    print("\n>>> Normal-side Risk (averaged across all classes):")
    print(f"  MAP: R_0={df['R_MAP_0'].mean():.3f}±{df['R_MAP_0'].std():.3f}, "
          f"R_ε={df['R_MAP_eps'].mean():.3f}±{df['R_MAP_eps'].std():.3f}")
    print(f"  LAP: R_0={df['R_LAP_0'].mean():.3f}±{df['R_LAP_0'].std():.3f}, "
          f"R_ε={df['R_LAP_eps'].mean():.3f}±{df['R_LAP_eps'].std():.3f}")
    
    print("\n>>> Consistency:")
    print(f"  Margin discrepancy: {df['margin_discrepancy_mean'].mean():.3f}±{df['margin_discrepancy_mean'].std():.3f}")
    print(f"  Disagreement rate: {df['disagreement_rate'].mean():.3f}±{df['disagreement_rate'].std():.3f}")
    print(f"  Margin correlation: {df['margin_correlation'].mean():.3f}±{df['margin_correlation'].std():.3f}")
    
    print("\n>>> Anchor Geometry:")
    print(f"  cos(μ_n, μ_MAP): {df['cos_normal_MAP'].mean():.3f}±{df['cos_normal_MAP'].std():.3f}")
    print(f"  cos(μ_n, μ_LAP): {df['cos_normal_LAP'].mean():.3f}±{df['cos_normal_LAP'].std():.3f}")
    print(f"  cos(μ_MAP, μ_LAP): {df['cos_MAP_LAP'].mean():.3f}±{df['cos_MAP_LAP'].std():.3f}")


if __name__ == '__main__':
    main()
