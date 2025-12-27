"""
Baseline Evaluation Script for PromptAD

This script runs comprehensive evaluation with detailed margin analysis.
Outputs both performance metrics and diagnostic statistics.
"""

import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path

from datasets import *
from datasets import dataset_classes
from utils.csv_utils import *
from utils.metrics import *
from utils.training_utils import *
from PromptAD import *
from utils.eval_utils import *
from analysis.baseline import (
    generate_margin_report,
    analyze_anchor_geometry,
    analyze_anchor_decomposition,
    evaluate_with_decomposed_anchors
)

TASK = 'CLS_BASELINE'


def test_with_baseline_analysis(
    model,
    args,
    dataloader,
    device: str,
    img_dir: str,
    check_path: str,
):
    """Test with comprehensive baseline analysis."""
    
    model.eval_mode()
    model.load_state_dict(torch.load(check_path), strict=False)

    # Collect all data with detailed information
    all_margins = []
    all_logits = []
    all_semantic_scores = []
    all_memory_scores = []
    all_fusion_scores = []
    all_labels = []
    all_names = []

    for (data, mask, label, name, img_type) in dataloader:
        data = [model.transform(Image.fromarray(f.numpy())) for f in data]
        data = torch.stack(data, dim=0).to(device)

        # Get detailed outputs
        outputs = model(data, 'cls_detailed')
        
        all_margins.extend(outputs['margins'])
        all_logits.append(outputs['logits'])
        all_semantic_scores.extend(outputs['semantic_scores'])
        all_memory_scores.extend(outputs['memory_scores'])
        all_fusion_scores.extend(outputs['fusion_scores'])
        all_labels.extend(label.numpy())
        all_names.extend(name)

    # Convert to numpy
    all_margins = np.array(all_margins)
    all_logits = np.vstack(all_logits)
    all_semantic_scores = np.array(all_semantic_scores)
    all_memory_scores = np.array(all_memory_scores)
    all_fusion_scores = np.array(all_fusion_scores)
    all_labels = np.array(all_labels)

    # Generate comprehensive report
    report = generate_margin_report(
        margins=all_margins,
        labels=all_labels,
        semantic_scores=all_semantic_scores,
        fusion_scores=all_fusion_scores,
        class_name=args.class_name
    )

    # Analyze anchor geometry
    mu_normal = model.text_features[0].cpu().numpy()
    mu_abnormal = model.text_features[1].cpu().numpy()
    geometry = analyze_anchor_geometry(mu_normal, mu_abnormal)
    report['anchor_geometry'] = geometry

    # Analyze anchor decomposition (MAP vs LAP)
    decomposition = analyze_anchor_decomposition(model, device)
    report['anchor_decomposition'] = decomposition

    # Evaluate with decomposed anchors
    decomposed_results = evaluate_with_decomposed_anchors(model, dataloader, device)
    report['decomposed_evaluation'] = {
        'MAP_only_auroc': decomposed_results['MAP_only']['auroc'],
        'LAP_only_auroc': decomposed_results['LAP_only']['auroc'],
        'ALL_combined_auroc': decomposed_results['ALL_combined']['auroc']
    }

    return report, {
        'margins': all_margins,
        'logits': all_logits,
        'scores': {
            'semantic': all_semantic_scores,
            'memory': all_memory_scores,
            'fusion': all_fusion_scores
        },
        'labels': all_labels,
        'names': all_names
    }


def save_baseline_results(report: Dict, raw_data: Dict, output_dir: Path, class_name: str):
    """Save comprehensive baseline analysis results."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Main metrics CSV
    metrics_df = pd.DataFrame([{
        'class': class_name,
        'semantic_auroc': report['semantic_auroc'],
        'fusion_auroc': report['fusion_auroc'],
        'delta_fusion': report['delta_fusion'],
        'semantic_fpr95': report['semantic_fpr95'],
        'fusion_fpr95': report['fusion_fpr95'],
        
        # Margin statistics
        'margin_normal_mean': report['margin_statistics']['normal']['mean'],
        'margin_normal_median': report['margin_statistics']['normal']['median'],
        'margin_abnormal_mean': report['margin_statistics']['abnormal']['mean'],
        'margin_abnormal_median': report['margin_statistics']['abnormal']['median'],
        
        # Overlap
        'overlap_ratio': report['overlap_analysis']['overlap_ratio'],
        'normal_overlap': report['overlap_analysis']['normal_overlap'],
        'abnormal_overlap': report['overlap_analysis']['abnormal_overlap'],
        
        # Split-side risks
        'normal_risk_0.0': report['split_side_risks']['normal_side_risk']['threshold_0.0'],
        'abnormal_risk_0.0': report['split_side_risks']['abnormal_side_risk']['threshold_0.0'],
        
        # Anchor geometry
        'anchor_cos_sim': report['anchor_geometry']['cosine_similarity'],
        'anchor_l2_dist': report['anchor_geometry']['l2_distance'],
        'anchor_angular_dist': report['anchor_geometry']['angular_distance'],
        
        # Decomposition
        'MAP_only_auroc': report['decomposed_evaluation']['MAP_only_auroc'],
        'LAP_only_auroc': report['decomposed_evaluation']['LAP_only_auroc'],
        'MAP_weight': report['anchor_decomposition']['MAP_weight'],
        'LAP_weight': report['anchor_decomposition']['LAP_weight'],
    }])
    
    metrics_df.to_csv(output_dir / f'{class_name}_baseline_metrics.csv', index=False)
    
    # 2. Detailed margin statistics
    margin_stats = []
    for group in ['normal', 'abnormal']:
        stats = report['margin_statistics'][group]
        stats['group'] = group
        stats['class'] = class_name
        margin_stats.append(stats)
    
    pd.DataFrame(margin_stats).to_csv(
        output_dir / f'{class_name}_margin_stats.csv', index=False
    )
    
    # 3. Raw data (for plotting)
    np.savez(
        output_dir / f'{class_name}_raw_data.npz',
        margins=raw_data['margins'],
        logits=raw_data['logits'],
        semantic_scores=raw_data['scores']['semantic'],
        memory_scores=raw_data['scores']['memory'],
        fusion_scores=raw_data['scores']['fusion'],
        labels=raw_data['labels'],
        names=np.array(raw_data['names'], dtype=object)
    )
    
    print(f"\n✅ Results saved to {output_dir}")
    print(f"   - Metrics: {class_name}_baseline_metrics.csv")
    print(f"   - Margin stats: {class_name}_margin_stats.csv")
    print(f"   - Raw data: {class_name}_raw_data.npz")


def main(args):
    kwargs = vars(args)

    if kwargs['seed'] is None:
        kwargs['seed'] = 111

    setup_seed(kwargs['seed'])

    if kwargs['use_cpu'] == 0:
        device = f"cuda:0"
    else:
        device = f"cpu"
    kwargs['device'] = device

    # Prepare output directory for analysis results
    analysis_output_dir = Path(args.root_dir) / 'baseline_analysis' / args.dataset / f"k_{args.k_shot}" / f"seed_{args.seed}"
    analysis_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Checkpoint path - try multiple locations
    # Priority: 1. Explicit path 2. Standard training output 3. Legacy location
    if args.checkpoint_path:
        check_path = Path(args.checkpoint_path)
    else:
        # Try standard training output location (from run_cls.py)
        check_path_candidates = [
            Path(args.root_dir) / args.dataset / f"k_{args.k_shot}" / 'checkpoints' / f"{args.class_name}.pth",
            Path(args.root_dir) / 'baseline' / args.dataset / f"k_{args.k_shot}" / 'checkpoints' / f"{args.class_name}.pth",
            Path('./result') / args.dataset / f"k_{args.k_shot}" / 'checkpoints' / f"{args.class_name}.pth",
        ]
        
        check_path = None
        for candidate in check_path_candidates:
            if candidate.exists():
                check_path = candidate
                print(f"✅ Found checkpoint: {check_path}")
                break
        
        if check_path is None:
            print(f"❌ Checkpoint not found. Tried locations:")
            for candidate in check_path_candidates:
                print(f"   - {candidate}")
            raise FileNotFoundError(f"No checkpoint found for {args.class_name}")
    
    check_path = str(check_path)

    # Get test dataloader
    test_dataloader, test_dataset_inst = get_dataloader_from_args(
        phase='test', perturbed=False, **kwargs
    )

    kwargs['out_size_h'] = kwargs['resolution']
    kwargs['out_size_w'] = kwargs['resolution']

    # Get model
    model = PromptAD(**kwargs)
    model = model.to(device)

    # Run comprehensive baseline analysis
    print(f"\n{'='*60}")
    print(f"Baseline Analysis: {args.class_name}")
    print(f"{'='*60}\n")
    
    report, raw_data = test_with_baseline_analysis(
        model, args, test_dataloader, device, 
        img_dir=str(analysis_output_dir / 'vis'),
        check_path=check_path
    )

    # Print summary
    print(f"\n{'='*60}")
    print(f"Results Summary")
    print(f"{'='*60}")
    print(f"Semantic AUROC:  {report['semantic_auroc']:.2f}")
    print(f"Fusion AUROC:    {report['fusion_auroc']:.2f}")
    print(f"Δ Fusion:        {report['delta_fusion']:+.2f}")
    print(f"\nSemantic FPR@95: {report['semantic_fpr95']:.4f}")
    print(f"Fusion FPR@95:   {report['fusion_fpr95']:.4f}")
    print(f"\nMargin (Normal):  μ={report['margin_statistics']['normal']['mean']:.3f}, "
          f"σ={report['margin_statistics']['normal']['std']:.3f}")
    print(f"Margin (Abnormal): μ={report['margin_statistics']['abnormal']['mean']:.3f}, "
          f"σ={report['margin_statistics']['abnormal']['std']:.3f}")
    print(f"\nOverlap Ratio:    {report['overlap_analysis']['overlap_ratio']:.2%}")
    print(f"\nAnchor Geometry:")
    print(f"  cos(μ_n, μ_a):  {report['anchor_geometry']['cosine_similarity']:.4f}")
    print(f"  ||μ_n - μ_a||:  {report['anchor_geometry']['l2_distance']:.4f}")
    print(f"\nDecomposed AUROC:")
    print(f"  MAP-only:       {report['decomposed_evaluation']['MAP_only_auroc']:.2f}")
    print(f"  LAP-only:       {report['decomposed_evaluation']['LAP_only_auroc']:.2f}")
    print(f"  Combined (ALL): {report['decomposed_evaluation']['ALL_combined_auroc']:.2f}")
    print(f"{'='*60}\n")

    # Save results
    save_baseline_results(report, raw_data, analysis_output_dir / 'results', args.class_name)


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def get_args():
    parser = argparse.ArgumentParser(description='PromptAD Baseline Analysis')
    parser.add_argument('--dataset', type=str, default='mvtec', choices=['mvtec', 'visa'])
    parser.add_argument('--class_name', type=str, default='carpet')
    parser.add_argument('--checkpoint-path', type=str, default=None,
                        help='Explicit path to checkpoint file (optional, will auto-detect if not provided)')

    parser.add_argument('--img-resize', type=int, default=240)
    parser.add_argument('--img-cropsize', type=int, default=240)
    parser.add_argument('--resolution', type=int, default=400)

    parser.add_argument('--batch-size', type=int, default=400)
    parser.add_argument("--root-dir", type=str, default="./result")
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--gpu-id", type=int, default=0)

    # Method parameters
    parser.add_argument('--k-shot', type=int, default=2)
    parser.add_argument("--backbone", type=str, default="ViT-B-16-plus-240",
                        choices=['ViT-B-16-plus-240', 'ViT-B-16'])
    parser.add_argument("--pretrained_dataset", type=str, default="laion400m_e32")
    parser.add_argument("--use-cpu", type=int, default=0)

    # Prompt parameters
    parser.add_argument("--n_ctx", type=int, default=4)
    parser.add_argument("--n_ctx_ab", type=int, default=1)
    parser.add_argument("--n_pro", type=int, default=3)
    parser.add_argument("--n_pro_ab", type=int, default=4)

    parser.add_argument("--num-workers", type=int, default=0)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    import os

    args = get_args()
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ['CUDA_VISIBLE_DEVICES'] = f"{args.gpu_id}"
    main(args)
