"""
Demo script showing how to use the modular scoring framework.

This demonstrates:
1. Separate semantic-only and memory-only evaluation
2. Oracle upper bound analysis
3. Component-wise performance breakdown
"""

import argparse
import torch
import numpy as np
from datasets import get_dataloader_from_args, dataset_classes
from PromptAD import PromptAD
from PromptAD.model_modular import PromptADModular
from utils.metrics_modular import metric_cal_img_modular, analyze_reliability_signals
from utils.training_utils import setup_seed
from tqdm import tqdm


def evaluate_with_mode(model, dataloader, device, score_mode='max', task='cls'):
    """
    Evaluate model with specified scoring mode.
    
    Args:
        model: PromptADModular instance
        dataloader: test dataloader
        device: cuda/cpu
        score_mode: scoring mode
        task: 'cls' or 'seg'
        
    Returns:
        metrics dict
    """
    model.eval_mode()
    model.score_mode = score_mode
    model.scorer.score_mode = score_mode
    
    scores_img = []
    score_maps = []
    gt_list = []
    all_metadata = []
    
    for data, mask, label, name, img_type in tqdm(dataloader, desc=f'{score_mode} mode'):
        data = data.to(device)
        
        with torch.no_grad():
            if task == 'cls':
                # For oracle, pass ground truth
                if score_mode == 'oracle':
                    output, metadata = model(data, task=task, gt_labels=label.numpy())
                    score_img, score_map = output
                else:
                    model.return_metadata = True
                    output, metadata = model(data, task=task)
                    score_img, score_map = output
                    model.return_metadata = False
                
                scores_img.extend(score_img)
                score_maps.extend(score_map)
            else:  # seg
                if score_mode == 'oracle':
                    output, metadata = model(data, task=task, gt_labels=label.numpy())
                    score_map = output
                else:
                    model.return_metadata = True
                    output, metadata = model(data, task=task)
                    score_map = output
                    model.return_metadata = False
                
                score_maps.extend(score_map)
                # For seg, use max of map as img score
                scores_img.extend([s.max() for s in score_map])
        
        gt_list.extend(label.numpy())
        all_metadata.append(metadata)
    
    # Compute metrics
    if task == 'cls':
        metrics = metric_cal_img_modular(
            np.array(scores_img),
            np.array(gt_list),
            np.array(score_maps),
            metadata=all_metadata[-1] if all_metadata else None,
            score_mode=score_mode
        )
    else:
        # For seg, use pixel-level metrics
        from utils.metrics import metric_cal_pix
        metrics = metric_cal_pix(np.array(score_maps), gt_list)
    
    # Add reliability analysis
    if all_metadata:
        reliability = analyze_reliability_signals(all_metadata[-1], np.array(gt_list))
        metrics.update(reliability)
    
    return metrics


def main(args):
    # Setup
    setup_seed(args.seed)
    device = f"cuda:{args.gpu_id}" if not args.use_cpu else "cpu"
    
    # Create base model
    kwargs = vars(args)
    kwargs['device'] = device
    kwargs['out_size_h'] = kwargs['resolution']
    kwargs['out_size_w'] = kwargs['resolution']
    
    base_model = PromptAD(**kwargs).to(device)
    
    # Get dataloaders
    train_dataloader, _ = get_dataloader_from_args(
        phase='train', 
        perturbed=False, 
        transform=base_model.transform,
        **kwargs
    )
    
    test_dataloader, _ = get_dataloader_from_args(
        phase='test',
        perturbed=False,
        transform=base_model.transform,
        **kwargs
    )
    
    # Setup model (memory bank + text features)
    print(f"\n{'='*60}")
    print(f"Setting up model for class: {args.class_name}")
    print(f"{'='*60}\n")
    
    base_model.eval_mode()
    
    # Setup text features
    base_model.setup_text_features(args.class_name)
    
    # Setup memory bank
    features1 = []
    features2 = []
    for data, mask, label, name, img_type in train_dataloader:
        data = data.to(device)
        with torch.no_grad():
            _, _, feature_map1, feature_map2 = base_model.encode_image(data)
            features1.append(feature_map1)
            features2.append(feature_map2)
    
    features1 = torch.cat(features1, dim=0)
    features2 = torch.cat(features2, dim=0)
    base_model.setup_memory_bank(features1, features2)
    
    # Evaluate with different scoring modes
    modes_to_test = ['semantic', 'memory', 'max', 'harmonic', 'oracle']
    
    results = {}
    
    print(f"\n{'='*60}")
    print(f"Evaluation Results - {args.class_name}")
    print(f"{'='*60}\n")
    
    for mode in modes_to_test:
        print(f"\nEvaluating mode: {mode}")
        print("-" * 40)
        
        # Wrap model with modular interface
        wrapped_model = PromptADModular(base_model, score_mode=mode, return_metadata=False)
        
        # Evaluate
        metrics = evaluate_with_mode(
            wrapped_model,
            test_dataloader,
            device,
            score_mode=mode,
            task=args.task
        )
        
        results[mode] = metrics
        
        # Print results
        print(f"Image AUROC: {metrics.get('i_roc', 0.0):.2f}")
        
        if 'i_roc_semantic' in metrics:
            print(f"  ├─ Semantic: {metrics['i_roc_semantic']:.2f}")
            print(f"  ├─ Memory:   {metrics['i_roc_memory']:.2f}")
            print(f"  └─ Gap:      {metrics['gap']:.2f}")
        
        if mode == 'oracle' and 'oracle_semantic_ratio' in metrics:
            print(f"  Oracle selections:")
            print(f"    ├─ Semantic: {metrics['oracle_semantic_ratio']:.1f}%")
            print(f"    └─ Memory:   {metrics['oracle_memory_ratio']:.1f}%")
    
    # Summary table
    print(f"\n{'='*60}")
    print("Summary Table")
    print(f"{'='*60}")
    print(f"{'Mode':<12} {'AUROC':<10} {'Gap to Oracle':<15}")
    print("-" * 60)
    
    oracle_score = results.get('oracle', {}).get('i_roc', 0.0)
    
    for mode in modes_to_test:
        auroc = results[mode].get('i_roc', 0.0)
        gap = oracle_score - auroc
        print(f"{mode:<12} {auroc:>8.2f}  {gap:>12.2f}")
    
    print(f"{'='*60}\n")
    
    return results


def get_args():
    parser = argparse.ArgumentParser(description='Modular scoring demo')
    
    # Dataset args
    parser.add_argument('--dataset', type=str, default='mvtec', choices=['mvtec', 'visa'])
    parser.add_argument('--class_name', type=str, default='carpet')
    parser.add_argument('--k-shot', type=int, default=4)
    parser.add_argument('--task', type=str, default='cls', choices=['cls', 'seg'])
    
    # Model args
    parser.add_argument('--img-resize', type=int, default=240)
    parser.add_argument('--img-cropsize', type=int, default=240)
    parser.add_argument('--resolution', type=int, default=256)
    parser.add_argument('--backbone', type=str, default='ViT-B-16-plus-240')
    parser.add_argument('--pretrained_dataset', type=str, default='laion400m_e32')
    
    # Prompt args
    parser.add_argument('--n_ctx', type=int, default=4)
    parser.add_argument('--n_ctx_ab', type=int, default=1)
    parser.add_argument('--n_pro', type=int, default=3)
    parser.add_argument('--n_pro_ab', type=int, default=4)
    
    # System args
    parser.add_argument('--seed', type=int, default=111)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--use-cpu', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=0)
    
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    results = main(args)
